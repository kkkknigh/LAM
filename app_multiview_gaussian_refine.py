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
    return result.path, f"{result.message}\nNext: run COLMAP.", _gallery(result.path)


def restore_uploaded_masks(workspace):
    result = _pipe(workspace).restore_uploaded_masks()
    return result.path, f"{result.message}\nNext: run FLAME Tracking if flame_param/landmarks are still missing.", _gallery(result.path)


def run_colmap(workspace):
    result = _pipe(workspace).run_colmap(DEFAULT_COLMAP_PATH)
    return result.path, f"{result.message}\nNext: initialize alignment.", _image(result.path)


def initialize_sim3(workspace):
    lam = build_lam()
    result = _pipe(workspace).initialize_sim3_from_layer1(lam)
    score = _projection_score_text(Path(workspace))
    return result.path, f"{result.message}\n{score}\nNext: preview alignment, then calibrate alignment if the overlay is not tight.", _image(result.path)


def preview(workspace):
    lam = build_lam()
    result = _pipe(workspace).preview_alignment(lam, None)
    gallery = sorted(Path(result.path).glob("*.png"))
    status = _alignment_status_text(Path(workspace))
    return result.path, f"{result.message}\n{status}\nNext: calibrate alignment before refinement if face position/scale is still offset.", [str(p) for p in gallery]


def calibrate_alignment(workspace):
    if not workspace:
        raise gr.Error("Create or select a workspace first.")
    lam = build_lam()
    result = _pipe(workspace).calibrate_global_sim3_blackbox(lam)
    return (
        result.path,
        f"{result.message}\nNext: preview alignment, then run refinement.",
        _image(result.path),
        [],
    )


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


def refine_small_xyz_geometry(workspace):
    defaults = {stage.name: stage for stage in RefinementConfig().stages}
    stage = replace(defaults["geometry_xyz"])
    loss_weights = replace(RefinementConfig().loss_weights, xyz_anchor=0.01, knn_anchor=0.004, scale_limit=0.002)
    return _refine_stages(
        workspace,
        ["geometry_xyz"],
        resume_required=False,
        next_step="render final review or export the package.",
        resume_from_latest=True,
        stages=[stage],
        loss_weights=loss_weights,
    )


def _refine_stages(workspace, stage_names, resume_required, next_step, resume_from_latest=True, stages=None, loss_weights=None):
    lam = build_lam()
    if stages is None:
        default_stages = {stage.name: stage for stage in RefinementConfig().stages}
        stages = [replace(default_stages[name]) for name in stage_names]
    stages = [_app_stage(stage) for stage in stages]
    config = RefinementConfig(stages=stages, output_dir=str(Path(workspace) / "refine"))
    if loss_weights is not None:
        config.loss_weights = loss_weights
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


def _app_stage(stage):
    return stage


def export(workspace):
    result = _pipe(workspace).export()
    return result.path, result.message


def render_final_review(workspace):
    if not workspace:
        raise gr.Error("Create or select a workspace first.")
    lam = build_lam()
    result = _pipe(workspace).export_final_review(lam)
    review_dir = Path(result.path)
    overlays = sorted((review_dir / "overlays").glob("*.png"))
    video = review_dir / "final_review.mp4"
    return result.path, result.message, [str(p) for p in overlays], str(video) if video.exists() else None


def process_all(camera_images_zip, layer1_lam_zip, workspace):
    workspace_gallery = []
    flame_gallery = []
    colmap_plot = None
    alignment_plot = None
    alignment_gallery = []
    refine_gallery = []
    loss_plot = None
    review_gallery = []
    review_video = None
    latest_output = ""
    status_lines = []

    def outputs():
        return (
            workspace,
            workspace,
            latest_output,
            "\n".join(status_lines),
            workspace_gallery,
            flame_gallery,
            colmap_plot,
            alignment_plot,
            alignment_gallery,
            refine_gallery,
            loss_plot,
            review_gallery,
            review_video,
        )

    def record(step, result):
        nonlocal latest_output
        latest_output = result.path
        status_lines.append(f"[{step}] {result.message}")

    try:
        if camera_images_zip and layer1_lam_zip:
            pipe = MultiViewRefinePipeline.create(DEFAULT_OUTPUT_ROOT, None)
            camera_zip_path = camera_images_zip.name if hasattr(camera_images_zip, "name") else camera_images_zip
            layer1_zip_path = layer1_lam_zip.name if hasattr(layer1_lam_zip, "name") else layer1_lam_zip
            result = pipe.unpack_uploads(camera_zip_path, layer1_zip_path)
            workspace = str(pipe.workspace.root)
            workspace_gallery = _gallery(result.path)
            record("1/11 Workspace", result)
            yield outputs()
        elif workspace:
            pipe = _pipe(workspace)
            workspace = str(pipe.workspace.root)
            existing_preview = Path(workspace) / "debug" / "00_inputs"
            workspace_gallery = _gallery(existing_preview)
            status_lines.append(f"[1/11 Workspace] Using existing workspace: {workspace}")
            latest_output = workspace
            yield outputs()
        else:
            raise gr.Error("Upload both ZIP files or create/select a workspace first.")

        result = pipe.generate_masks_and_flame()
        flame_gallery = _gallery(result.path)
        record("2/11 Masks / FLAME", result)
        yield outputs()

        result = pipe.run_colmap(DEFAULT_COLMAP_PATH)
        colmap_plot = _image(result.path)
        record("3/11 COLMAP", result)
        yield outputs()

        status_lines.append("Loading LAM model for alignment and refinement.")
        yield outputs()
        lam = build_lam()

        result = pipe.initialize_sim3_from_layer1(lam)
        alignment_plot = _image(result.path)
        record("4/11 Initialize Alignment", result)
        status_lines.append(_projection_score_text(Path(workspace)))
        yield outputs()

        result = pipe.calibrate_global_sim3_blackbox(lam)
        alignment_plot = _image(result.path)
        record("5/11 Calibrate Alignment", result)
        yield outputs()

        result = pipe.preview_alignment(lam, None)
        alignment_gallery = [str(p) for p in sorted(Path(result.path).glob("*.png"))]
        record("6/11 Preview Alignment", result)
        status_lines.append(_alignment_status_text(Path(workspace)))
        yield outputs()

        default_stages = {stage.name: stage for stage in RefinementConfig().stages}
        stages = [replace(default_stages[name]) for name in ["camera", "pose", "appearance"]]
        config = RefinementConfig(stages=stages, output_dir=str(Path(workspace) / "refine"))
        result = pipe.refine(lam, None, config, resume=None)
        refine_gallery = _latest_refine_gallery(workspace, "appearance")
        loss_plot = _image(Path(workspace) / "debug" / "05_refine" / "loss_history.png")
        record("7/11 Refinement", result)
        yield outputs()

        geometry_stage = replace(default_stages["geometry_light"])
        geometry_config = RefinementConfig(stages=[geometry_stage], output_dir=str(Path(workspace) / "refine"))
        latest_checkpoint = Path(workspace) / "refine" / "checkpoints" / "latest.pt"
        result = pipe.refine(lam, None, geometry_config, resume=str(latest_checkpoint) if latest_checkpoint.exists() else None)
        refine_gallery = _latest_refine_gallery(workspace, "geometry_light")
        loss_plot = _image(Path(workspace) / "debug" / "05_refine" / "loss_history.png")
        record("8/11 Geometry", result)
        yield outputs()

        xyz_stage = replace(default_stages["geometry_xyz"])
        xyz_loss_weights = replace(RefinementConfig().loss_weights, xyz_anchor=0.01, knn_anchor=0.004, scale_limit=0.002)
        xyz_config = RefinementConfig(stages=[xyz_stage], output_dir=str(Path(workspace) / "refine"))
        xyz_config.loss_weights = xyz_loss_weights
        latest_checkpoint = Path(workspace) / "refine" / "checkpoints" / "latest.pt"
        result = pipe.refine(lam, None, xyz_config, resume=str(latest_checkpoint) if latest_checkpoint.exists() else None)
        refine_gallery = _latest_refine_gallery(workspace, "geometry_xyz")
        loss_plot = _image(Path(workspace) / "debug" / "05_refine" / "loss_history.png")
        record("9/11 Small XYZ Geometry", result)
        yield outputs()

        result = pipe.export()
        record("10/11 Export", result)
        yield outputs()

        result = pipe.export_final_review(lam)
        review_dir = Path(result.path)
        review_gallery = [str(p) for p in sorted((review_dir / "overlays").glob("*.png"))]
        review_video_path = review_dir / "final_review.mp4"
        review_video = str(review_video_path) if review_video_path.exists() else None
        record("11/11 Final Review", result)
        status_lines.append("Full multi-view process complete. package.zip and final_review.zip are ready in the export directory.")
        yield outputs()
    except Exception as exc:
        status_lines.append(f"FAILED: {type(exc).__name__}: {exc}")
        yield outputs()
        raise


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
        f"final review: {(root / 'exports' / 'final_review.zip').exists()}",
        f"final review video: {(root / 'exports' / 'final_review' / 'final_review.mp4').exists()}",
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
    return "Alignment status: initial Sim3 only. This is coarse; Calibrate Alignment is expected before refinement when overlay is offset."


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
        with gr.Row():
            create_btn = gr.Button("Create Workspace")
            process_all_btn = gr.Button("Process All", variant="primary")
        workspace_gallery = gr.Gallery(label="Input Preview", columns=2, height=480)
        create_btn.click(
            create_workspace,
            [camera_images_zip, layer1_lam_zip],
            [workspace, workspace_display, status, workspace_gallery],
        )

        gr.Markdown("## 2. Masks / FLAME")
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

        gr.Markdown("## 4. Camera Alignment / Calibration")
        with gr.Row():
            init_sim3_btn = gr.Button("Initialize Alignment", variant="primary")
            preview_btn = gr.Button("Preview Alignment")
            calibrate_btn = gr.Button("Calibrate Alignment")
        alignment_plot = gr.Image(label="Camera Alignment", type="filepath", height=420)
        alignment_gallery = gr.Gallery(label="Alignment Overlays", columns=2, height=480)
        init_sim3_btn.click(initialize_sim3, [workspace], [output_path, status, alignment_plot])
        preview_btn.click(preview, [workspace], [output_path, status, alignment_gallery])
        calibrate_btn.click(calibrate_alignment, [workspace], [output_path, status, alignment_plot, alignment_gallery])

        gr.Markdown("## 5. Refinement")
        with gr.Row():
            refine_btn = gr.Button("Run Refine", variant="primary")
            geometry_btn = gr.Button("Run Geometry")
            small_xyz_btn = gr.Button("Run Small XYZ Geometry (Optional / Advanced)")
        refine_gallery = gr.Gallery(label="Latest Refinement Overlay", columns=2, height=480)
        loss_plot = gr.Image(label="Loss History", type="filepath", height=360)
        refine_btn.click(refine_alignment_pose_appearance, [workspace], [output_path, status, refine_gallery, loss_plot])
        geometry_btn.click(refine_geometry, [workspace], [output_path, status, refine_gallery, loss_plot])
        small_xyz_btn.click(refine_small_xyz_geometry, [workspace], [output_path, status, refine_gallery, loss_plot])

        gr.Markdown("## 6. Export / Final Review")
        with gr.Row():
            export_btn = gr.Button("Export Package", variant="primary")
            review_btn = gr.Button("Render Final Review")
        review_gallery = gr.Gallery(label="Final Review Overlays", columns=2, height=480)
        review_video = gr.Video(label="Final Review Video", format="mp4", height=360)
        export_btn.click(export, [workspace], [output_path, status])
        review_btn.click(render_final_review, [workspace], [output_path, status, review_gallery, review_video])
        process_all_btn.click(
            process_all,
            [camera_images_zip, layer1_lam_zip, workspace],
            [
                workspace,
                workspace_display,
                output_path,
                status,
                workspace_gallery,
                flame_gallery,
                colmap_plot,
                alignment_plot,
                alignment_gallery,
                refine_gallery,
                loss_plot,
                review_gallery,
                review_video,
            ],
        )
        refresh_btn.click(workspace_status, [workspace], [status])

        demo.queue(concurrency_count=1, max_size=2)
        port = int(os.environ.get("GRADIO_SERVER_PORT", "7861"))
        demo.launch(server_name="127.0.0.1", server_port=port, max_threads=1)


if __name__ == "__main__":
    launch()
