import os
from pathlib import Path

import gradio as gr
from omegaconf import OmegaConf
from safetensors.torch import load_file

from lam.models import ModelLAM
from multiview_refine.optimization import RefinementConfig
from multiview_refine.pipeline import MultiViewRefinePipeline


def build_lam(config_path: str, model_name: str):
    model = ModelLAM(**OmegaConf.load(config_path).model)
    ckpt = load_file(os.path.join(model_name, "model.safetensors"), device="cpu")
    state_dict = model.state_dict()
    for key, value in ckpt.items():
        if key in state_dict and state_dict[key].shape == value.shape:
            state_dict[key].copy_(value)
    model.to("cuda").eval()
    return model


def _pipe(workspace):
    if not workspace:
        raise gr.Error("Create or select a workspace first.")
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
    root = Path(workspace) / "debug" / stage
    if not root.exists():
        return []
    steps = sorted([p for p in root.glob("step_*") if p.is_dir()])
    if not steps:
        return []
    return _gallery(steps[-1])


def create_workspace(output_root, job_id, camera_images_zip, layer1_lam_zip):
    pipe = MultiViewRefinePipeline.create(output_root or "output/multiview_refine", job_id or None)
    if camera_images_zip or layer1_lam_zip:
        if not camera_images_zip:
            raise gr.Error("Upload Camera Multi-view Images ZIP.")
        if not layer1_lam_zip:
            raise gr.Error("Upload Layer 1 LAM Canonical Package ZIP.")
        camera_zip_path = camera_images_zip.name if hasattr(camera_images_zip, "name") else camera_images_zip
        layer1_zip_path = layer1_lam_zip.name if hasattr(layer1_lam_zip, "name") else layer1_lam_zip
        result = pipe.unpack_uploads(camera_zip_path, layer1_zip_path)
        return str(pipe.workspace.root), result.message, _gallery(result.path)
    return str(pipe.workspace.root), "Workspace created.", []


def import_inputs(workspace, image_dir, mask_dir, flame_dir, colmap_dir, init_ply_path):
    result = _pipe(workspace).import_inputs(image_dir, mask_dir or None, flame_dir or None, colmap_dir or None, init_ply_path or None)
    return result.path, result.message, _gallery(result.path)


def run_flame(workspace):
    result = _pipe(workspace).generate_masks_and_flame()
    return result.path, result.message, _gallery(result.path)


def run_colmap(workspace, colmap_path):
    result = _pipe(workspace).run_colmap(colmap_path or "colmap")
    return result.path, result.message, _image(result.path)


def import_colmap(workspace, sparse_dir, colmap_path):
    result = _pipe(workspace).import_colmap(sparse_dir, colmap_path or "colmap")
    return result.path, result.message, _image(result.path)


def align_sim3(workspace, flame_target_transforms):
    if not flame_target_transforms:
        raise gr.Error("Provide calibrated target transforms. FLAME single-image tracking does not produce camera targets.")
    result = _pipe(workspace).align_sim3(flame_target_transforms)
    return result.path, result.message, _image(result.path)


def manual_sim3(workspace, scale, yaw, tx, ty, tz):
    result = _pipe(workspace).write_manual_alignment(scale, yaw, tx, ty, tz)
    return result.path, result.message, _image(result.path)


def preview(workspace, infer_config, model_name, init_ply):
    lam = build_lam(infer_config, model_name)
    result = _pipe(workspace).preview_alignment(lam, init_ply or None)
    gallery = sorted(Path(result.path).glob("*.png"))
    return result.path, result.message, [str(p) for p in gallery]


def _stage_config(stage, steps, lr, views_per_step, output_dir):
    from dataclasses import replace

    if stage == "all":
        return RefinementConfig(output_dir=output_dir)
    default_stages = {s.name: s for s in RefinementConfig().stages}
    stage_cfg = replace(default_stages[stage], steps=steps, lr=lr, views_per_step=views_per_step)
    return RefinementConfig(stages=[stage_cfg], output_dir=output_dir)


def refine_stage(workspace, infer_config, model_name, init_ply, stage, steps, lr, views_per_step, resume):
    lam = build_lam(infer_config, model_name)
    config = _stage_config(stage, steps, lr, views_per_step, workspace)
    result = _pipe(workspace).refine(lam, init_ply or None, config, resume or None)
    gallery_stage = "geometry_xyz" if stage == "all" else stage
    return result.path, result.message, _latest_refine_gallery(workspace, gallery_stage), _image(Path(workspace) / "debug" / "loss_history.png")


def export(workspace):
    result = _pipe(workspace).export()
    return result.path, result.message


def launch():
    with gr.Blocks(analytics_enabled=False) as demo:
        gr.Markdown("# LAM Layer 2 Multi-view Gaussian Refinement")
        workspace = gr.Textbox(label="Workspace", interactive=True)
        status = gr.Textbox(label="Status", lines=4, interactive=False)
        output_path = gr.Textbox(label="Output Path", interactive=False)

        with gr.Tab("0. Workspace"):
            with gr.Row():
                output_root = gr.Textbox(label="Output Root", value="output/multiview_refine")
                job_id = gr.Textbox(label="Job ID (optional)")
            with gr.Row():
                with gr.Group():
                    gr.Markdown("### Camera multi-view images")
                    camera_images_zip = gr.File(label="Camera Images ZIP", file_types=[".zip"])
                with gr.Group():
                    gr.Markdown("### Layer 1 LAM canonical package")
                    layer1_lam_zip = gr.File(label="LAM Canonical Package ZIP", file_types=[".zip"])
            create_btn = gr.Button("Create Workspace / Unpack Uploads", variant="primary")
            workspace_gallery = gr.Gallery(label="Input Preview", columns=2, height=480)
            create_btn.click(create_workspace, [output_root, job_id, camera_images_zip, layer1_lam_zip], [workspace, status, workspace_gallery])

        with gr.Tab("1. Import"):
            image_dir = gr.Textbox(label="Images Dir")
            mask_dir = gr.Textbox(label="Masks Dir (optional)")
            flame_dir = gr.Textbox(label="FLAME Param Dir (optional)")
            colmap_dir = gr.Textbox(label="COLMAP Dir (optional)")
            init_ply_path = gr.Textbox(label="Initial PLY Path (optional)")
            import_btn = gr.Button("Import Local Inputs")
            import_gallery = gr.Gallery(label="Input Preview", columns=2, height=480)
            import_btn.click(import_inputs, [workspace, image_dir, mask_dir, flame_dir, colmap_dir, init_ply_path], [output_path, status, import_gallery])

        with gr.Tab("2. Masks / FLAME"):
            flame_btn = gr.Button("Run FLAME Tracking + Generate Masks", variant="primary")
            flame_gallery = gr.Gallery(label="Mask / Landmark Preview", columns=2, height=480)
            flame_btn.click(run_flame, [workspace], [output_path, status, flame_gallery])

        with gr.Tab("3. COLMAP"):
            colmap_path = gr.Textbox(label="COLMAP Executable", value="colmap")
            sparse_dir = gr.Textbox(label="Existing Sparse Dir")
            with gr.Row():
                colmap_btn = gr.Button("Run COLMAP")
                import_colmap_btn = gr.Button("Import COLMAP Sparse")
            colmap_plot = gr.Image(label="COLMAP Camera Centers", type="filepath", height=480)
            colmap_btn.click(run_colmap, [workspace, colmap_path], [output_path, status, colmap_plot])
            import_colmap_btn.click(import_colmap, [workspace, sparse_dir, colmap_path], [output_path, status, colmap_plot])

        with gr.Tab("4. Sim3 Alignment"):
            flame_target = gr.Textbox(label="Calibrated Target Transforms")
            align_btn = gr.Button("Estimate Sim3 Alignment", variant="primary")
            sim3_plot = gr.Image(label="Sim3 Camera Alignment", type="filepath", height=480)
            align_btn.click(align_sim3, [workspace, flame_target], [output_path, status, sim3_plot])
            gr.Markdown("Manual fallback requires explicit user choice; identity is not applied silently.")
            with gr.Row():
                scale = gr.Number(label="Scale", value=1.0)
                yaw = gr.Number(label="Yaw Degrees", value=0.0)
                tx = gr.Number(label="Tx", value=0.0)
                ty = gr.Number(label="Ty", value=0.0)
                tz = gr.Number(label="Tz", value=0.0)
            manual_btn = gr.Button("Apply Manual Sim3")
            manual_btn.click(manual_sim3, [workspace, scale, yaw, tx, ty, tz], [output_path, status, sim3_plot])

        with gr.Tab("5. Preview"):
            infer_config = gr.Textbox(label="Infer Config", value="./configs/inference/lam-20k-8gpu.yaml")
            model_name = gr.Textbox(label="Model Name", value="./model_zoo/lam_models/releases/lam/lam-20k/step_045500/")
            init_ply_preview = gr.Textbox(label="Initial PLY Path (optional)")
            gallery = gr.Gallery(label="Alignment Overlays", columns=2, height=480)
            preview_btn = gr.Button("Preview Alignment", variant="primary")
            preview_btn.click(preview, [workspace, infer_config, model_name, init_ply_preview], [output_path, status, gallery])

        with gr.Tab("6. Refinement"):
            stage = gr.Dropdown(label="Stage", choices=["calibrate", "pose", "appearance", "geometry_light", "geometry_xyz", "all"], value="calibrate")
            steps = gr.Number(label="Steps", value=100, precision=0)
            lr = gr.Number(label="Learning Rate", value=1e-3)
            views_per_step = gr.Number(label="Views Per Step", value=4, precision=0)
            init_ply_refine = gr.Textbox(label="Initial PLY Path (optional)")
            resume = gr.Textbox(label="Resume Checkpoint (optional)")
            refine_btn = gr.Button("Run Selected Refinement Stage", variant="primary")
            refine_gallery = gr.Gallery(label="Latest Refinement Overlay", columns=2, height=480)
            loss_plot = gr.Image(label="Loss History", type="filepath", height=360)
            refine_btn.click(refine_stage, [workspace, infer_config, model_name, init_ply_refine, stage, steps, lr, views_per_step, resume], [output_path, status, refine_gallery, loss_plot])

        with gr.Tab("7. Export"):
            export_btn = gr.Button("Export Refined Gaussian", variant="primary")
            export_btn.click(export, [workspace], [output_path, status])

        demo.queue()
        demo.launch(server_name="127.0.0.1", server_port=7861)


if __name__ == "__main__":
    launch()
