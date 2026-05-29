import os
import json
from dataclasses import replace
from pathlib import Path

import gradio as gr
from omegaconf import OmegaConf
from safetensors.torch import load_file

from lam.models import ModelLAM
from multiview_refine.optimization import RefinementConfig
from multiview_refine.pipeline import MultiViewRefinePipeline


DEFAULT_OUTPUT_ROOT = "output/multiview_refine"
DEFAULT_COLMAP_PATH = "colmap"
DEFAULT_INFER_CONFIG = "./configs/inference/lam-20k-8gpu.yaml"
DEFAULT_MODEL_NAME = "./model_zoo/lam_models/releases/lam/lam-20k/step_045500/"


def build_lam():
    model = ModelLAM(**OmegaConf.load(DEFAULT_INFER_CONFIG).model)
    ckpt = load_file(os.path.join(DEFAULT_MODEL_NAME, "model.safetensors"), device="cpu")
    state_dict = model.state_dict()
    for key, value in ckpt.items():
        if key in state_dict and state_dict[key].shape == value.shape:
            state_dict[key].copy_(value)
    model.to("cuda").eval()
    return model


def _pipe(workspace):
    if not workspace:
        raise gr.Error("Create a workspace first.")
    return MultiViewRefinePipeline(workspace)


def _gallery(path):
    if not path:
        return []
    path = Path(path)
    if path.is_file():
        return [str(path)]
    if path.exists():
        return [str(p) for p in sorted(path.glob("*.png"))]
    return []


def _image(path):
    if path and Path(path).is_file():
        return str(path)
    return None


def _latest_refine_gallery(workspace, stage):
    root = Path(workspace) / "debug" / "05_refine" / stage
    if not root.exists():
        return []
    steps = sorted([p for p in root.glob("step_*") if p.is_dir()])
    if not steps:
        return []
    return _gallery(steps[-1])


def create_workspace(camera_images_zip, layer1_lam_zip):
    if not camera_images_zip:
        raise gr.Error("Upload Camera Images ZIP.")
    if not layer1_lam_zip:
        raise gr.Error("Upload LAM Canonical Package ZIP.")
    pipe = MultiViewRefinePipeline.create(DEFAULT_OUTPUT_ROOT, None)
    camera_zip_path = camera_images_zip.name if hasattr(camera_images_zip, "name") else camera_images_zip
    layer1_zip_path = layer1_lam_zip.name if hasattr(layer1_lam_zip, "name") else layer1_lam_zip
    result = pipe.unpack_uploads(camera_zip_path, layer1_zip_path)
    workspace = str(pipe.workspace.root)
    return workspace, workspace, f"{result.message}\nNext: run Masks / FLAME.", _gallery(result.path)


def run_flame(workspace):
    result = _pipe(workspace).generate_masks_and_flame()
    return result.path, f"{result.message}\nExisting uploaded masks are kept. Next: run COLMAP.", _gallery(result.path)


def import_masks(workspace, mask_dir):
    if not mask_dir:
        raise gr.Error("Provide a mask directory.")
    result = _pipe(workspace).import_and_process_masks(mask_dir)
    return result.path, f"{result.message}\nNext: preview masks, then run Calibrate/Camera.", _gallery(result.path)


def restore_uploaded_masks(workspace):
    result = _pipe(workspace).restore_uploaded_masks()
    return result.path, f"{result.message}\nNext: run FLAME Tracking if flame_param/landmarks are still missing.", _gallery(result.path)


def run_colmap(workspace):
    result = _pipe(workspace).run_colmap(DEFAULT_COLMAP_PATH)
    return result.path, f"{result.message}\nNext: initialize Sim3.", _image(result.path)


def initialize_sim3(workspace):
    lam = build_lam()
    result = _pipe(workspace).initialize_sim3_from_layer1(lam)
    score = _projection_score_text(Path(workspace))
    return result.path, f"{result.message}\n{score}\nNext: preview alignment, then run Calibrate if the overlay is not tight.", _image(result.path)


def preview(workspace):
    lam = build_lam()
    result = _pipe(workspace).preview_alignment(lam, None)
    gallery = sorted(Path(result.path).glob("*.png"))
    status = _alignment_status_text(Path(workspace))
    return result.path, f"{result.message}\n{status}\nNext: run Calibrate before refinement if face position/scale is still offset.", [str(p) for p in gallery]


def refine_calibrate(workspace):
    if not workspace:
        raise gr.Error("Create or select a workspace first.")
    lam = build_lam()
    result = _pipe(workspace).calibrate_global_sim3_blackbox(lam)
    return (
        result.path,
        f"{result.message}\nNext: preview alignment, then run pose/appearance refinement.",
        [],
        _image(Path(workspace) / "debug" / "05_refine" / "loss_history.png"),
    )


def refine_camera(workspace):
    return _refine_stages(workspace, ["camera"], resume_required=False, next_step="run pose refinement.", resume_from_latest=False)


def refine_alignment_pose_appearance(workspace):
    return _refine_stages(
        workspace,
        ["camera", "pose", "appearance"],
        resume_required=False,
        next_step="preview the result or export the package.",
        resume_from_latest=False,
    )


def refine_pose(workspace):
    return _refine_stages(workspace, ["pose"], resume_required=False, next_step="run appearance refinement.")


def refine_appearance(workspace):
    return _refine_stages(
        workspace,
        ["appearance"],
        resume_required=False,
        next_step="export the package, or run geometry only as a separate diagnostic.",
    )


def refine_geometry(workspace):
    return _refine_stages(workspace, ["geometry_light"], resume_required=False, next_step="export the package.")


def _refine_stages(workspace, stage_names, resume_required, next_step, resume_from_latest=True):
    lam = build_lam()
    default_stages = {stage.name: stage for stage in RefinementConfig().stages}
    stages = [replace(default_stages[name]) for name in stage_names]
    config = RefinementConfig(stages=stages, output_dir=str(Path(workspace) / "refine"))
    resume = None
    latest = Path(workspace) / "refine" / "checkpoints" / "latest.pt"
    if resume_from_latest and latest.exists():
        resume = str(latest)
    elif resume_required:
        if not latest.exists():
            raise gr.Error("Run the previous refinement stage first.")
    result = _pipe(workspace).refine(lam, None, config, resume)
    gallery_stage = stage_names[-1]
    resume_note = "Resumed from latest checkpoint." if resume else "Started from current alignment and initial Gaussian."
    if stage_names == ["appearance"]:
        resume_note += " Safe appearance only tunes bounded color by default; diagnostics should use a fresh run."
    return (
        result.path,
        f"{result.message}\n{resume_note}\nNext: {next_step}",
        _latest_refine_gallery(workspace, gallery_stage),
        _image(Path(workspace) / "debug" / "05_refine" / "loss_history.png"),
    )


def export(workspace):
    result = _pipe(workspace).export()
    return result.path, result.message


def workspace_status(workspace):
    if not workspace:
        return "No workspace selected."
    root = Path(workspace)
    if not root.exists():
        return f"Workspace does not exist: {root}"

    def count_files(name, suffixes):
        directory = root / name
        if not directory.exists():
            return 0
        return len([p for p in directory.glob("*") if p.suffix.lower() in suffixes])

    lines = [
        f"Workspace: {root}",
        f"images: {count_files('data/images', {'.png', '.jpg', '.jpeg'})}",
        f"fg_masks: {count_files('data/fg_masks', {'.png', '.jpg', '.jpeg'})}",
        f"flame_param: {count_files('data/flame_param', {'.npz'})}",
        f"init.ply: {(root / 'data' / 'init.ply').exists()}",
        f"canonical_flame_param.npz: {(root / 'data' / 'canonical_flame_param.npz').exists()}",
        f"transforms_colmap_raw.json: {(root / 'colmap' / 'transforms_colmap_raw.json').exists()}",
        f"transforms_aligned.json: {(root / 'alignment' / 'transforms_aligned.json').exists()}",
        f"refined_gaussian.ply: {(root / 'refine' / 'refined_gaussian.ply').exists()}",
        f"export package: {(root / 'exports' / 'package.zip').exists()}",
    ]
    return "\n".join(lines)


def _projection_score_text(workspace: Path) -> str:
    report = workspace / "alignment" / "initial_sim3_report.json"
    if not report.exists():
        return "Projection score: unavailable."
    data = json.loads(report.read_text(encoding="utf-8"))
    score = data.get("projection_score")
    if not score:
        return "Projection score: unavailable."
    return (
        "Projection score: "
        f"{score.get('positive_depth', 0)}/{score.get('num_views', 0)} positive depth, "
        f"{score.get('in_frame', 0)}/{score.get('num_views', 0)} center in frame, "
        f"median center error {float(score.get('median_center_error_px', 0.0)):.1f}px."
    )


def _alignment_status_text(workspace: Path) -> str:
    calibrated = workspace / "alignment" / "landmark_calibration_report.json"
    if calibrated.exists():
        return "Alignment status: landmark calibrated. Use this preview for refinement decisions."
    return "Alignment status: initial Sim3 only. This is coarse; Run Calibrate is expected before refinement when overlay is offset."


def launch():
    with gr.Blocks(analytics_enabled=False) as demo:
        gr.Markdown("# LAM Layer 2 Multi-view Gaussian Refinement")
        workspace = gr.State("")
        with gr.Row():
            workspace_display = gr.Textbox(label="Workspace", interactive=False)
            refresh_btn = gr.Button("Refresh Status")
        status = gr.Textbox(label="Status", lines=10, interactive=False)
        output_path = gr.Textbox(label="Latest Output Path", interactive=False)

        gr.Markdown("## 1. Upload Inputs")
        with gr.Row():
            camera_images_zip = gr.File(label="Camera Images ZIP", file_types=[".zip"])
            layer1_lam_zip = gr.File(label="LAM Canonical Package ZIP", file_types=[".zip"])
        create_btn = gr.Button("Create Workspace", variant="primary")
        workspace_gallery = gr.Gallery(label="Input Preview", columns=2, height=480)
        create_btn.click(
            create_workspace,
            [camera_images_zip, layer1_lam_zip],
            [workspace, workspace_display, status, workspace_gallery],
        )

        gr.Markdown("## 2. FLAME Tracking / Masks")
        flame_btn = gr.Button("Run FLAME Tracking (Keep Uploaded Masks)", variant="primary")
        mask_dir = gr.Textbox(label="External Mask Dir (SAM/rembg/etc.)", placeholder="Path containing fg_masks/, masks/, or same-stem PNG masks")
        import_masks_btn = gr.Button("Import / Postprocess Masks")
        restore_masks_btn = gr.Button("Restore Masks From Uploaded ZIP")
        flame_gallery = gr.Gallery(label="Mask / Landmark Preview", columns=2, height=480)
        flame_btn.click(run_flame, [workspace], [output_path, status, flame_gallery])
        import_masks_btn.click(import_masks, [workspace, mask_dir], [output_path, status, flame_gallery])
        restore_masks_btn.click(restore_uploaded_masks, [workspace], [output_path, status, flame_gallery])

        gr.Markdown("## 3. COLMAP")
        colmap_btn = gr.Button("Run COLMAP", variant="primary")
        colmap_plot = gr.Image(label="COLMAP Camera Centers", type="filepath", height=480)
        colmap_btn.click(run_colmap, [workspace], [output_path, status, colmap_plot])

        gr.Markdown("## 4. Sim3 Alignment")
        init_sim3_btn = gr.Button("Initialize Sim3", variant="primary")
        sim3_plot = gr.Image(label="Sim3 Camera Alignment", type="filepath", height=480)
        init_sim3_btn.click(initialize_sim3, [workspace], [output_path, status, sim3_plot])

        gr.Markdown("## 5. Preview")
        preview_btn = gr.Button("Preview Alignment", variant="primary")
        preview_gallery = gr.Gallery(label="Alignment Overlays", columns=2, height=480)
        preview_btn.click(preview, [workspace], [output_path, status, preview_gallery])

        gr.Markdown("## 6. Refinement")
        with gr.Row():
            calibrate_btn = gr.Button("Run Calibrate", variant="primary")
            camera_btn = gr.Button("Run Camera")
            reuse_btn = gr.Button("Reuse COLMAP+FLAME: Camera + Pose + Appearance")
            pose_btn = gr.Button("Run Pose")
            appearance_btn = gr.Button("Run Appearance")
            geometry_btn = gr.Button("Run Geometry")
        refine_gallery = gr.Gallery(label="Latest Refinement Overlay", columns=2, height=480)
        loss_plot = gr.Image(label="Loss History", type="filepath", height=360)
        calibrate_btn.click(refine_calibrate, [workspace], [output_path, status, refine_gallery, loss_plot])
        camera_btn.click(refine_camera, [workspace], [output_path, status, refine_gallery, loss_plot])
        reuse_btn.click(refine_alignment_pose_appearance, [workspace], [output_path, status, refine_gallery, loss_plot])
        pose_btn.click(refine_pose, [workspace], [output_path, status, refine_gallery, loss_plot])
        appearance_btn.click(refine_appearance, [workspace], [output_path, status, refine_gallery, loss_plot])
        geometry_btn.click(refine_geometry, [workspace], [output_path, status, refine_gallery, loss_plot])

        gr.Markdown("## 7. Export")
        export_btn = gr.Button("Export Final Package", variant="primary")
        export_btn.click(export, [workspace], [output_path, status])
        refresh_btn.click(workspace_status, [workspace], [status])

        demo.queue()
        port = int(os.environ.get("GRADIO_SERVER_PORT", "7861"))
        demo.launch(server_name="127.0.0.1", server_port=port)


if __name__ == "__main__":
    launch()
