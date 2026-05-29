import argparse
import json
import math
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import imageio.v2 as imageio
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from safetensors.torch import load_file

from lam.models import ModelLAM
from multiview_refine.data import load_multiview_bundle
from multiview_refine.optimization import LossWeights, MultiViewGaussianRefiner, RefinementConfig, RefinementStageConfig
from multiview_refine.pipeline import _load_renderable_gaussian
from multiview_refine.render_adapter import render_animate_gs_with_intrinsics
from multiview_refine.visualization import save_loss_plot, save_overlay_grid


DEFAULT_INFER_CONFIG = "./configs/inference/lam-20k-8gpu.yaml"
DEFAULT_MODEL_NAME = "./model_zoo/lam_models/releases/lam/lam-20k/step_045500/"


def build_lam(config_path: str = DEFAULT_INFER_CONFIG, model_dir: str = DEFAULT_MODEL_NAME):
    model = ModelLAM(**OmegaConf.load(config_path).model)
    ckpt = load_file(str(Path(model_dir) / "model.safetensors"), device="cpu")
    state_dict = model.state_dict()
    for key, value in ckpt.items():
        if key in state_dict and state_dict[key].shape == value.shape:
            state_dict[key].copy_(value)
    model.to("cuda").eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Run safe appearance refinement diagnostics.")
    parser.add_argument("--workspace", required=True, help="Path to a multiview_refine workspace.")
    parser.add_argument("--output-dir", default=None, help="Defaults to <workspace>/refine_diagnostics.")
    parser.add_argument("--resolution-scale", type=float, default=0.5)
    parser.add_argument("--turntable-frames", type=int, default=72)
    parser.add_argument("--turntable-size", type=int, default=768)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--skip-turntable", action="store_true")
    args = parser.parse_args()

    workspace = Path(args.workspace)
    out_root = Path(args.output_dir) if args.output_dir else workspace / "refine_diagnostics"
    out_root.mkdir(parents=True, exist_ok=True)

    lam = build_lam()
    baseline_ply = workspace / "data" / "init.ply"
    experiments = _experiment_configs(args.resolution_scale)
    summary = []

    baseline_dir = out_root / "baseline_render"
    _render_preview(lam, workspace, baseline_ply, baseline_dir)
    if not args.skip_turntable:
        _export_turntable(lam, workspace, baseline_ply, baseline_dir, args.turntable_frames, args.turntable_size, args.fps)
    summary.append({"name": "baseline_render", "output_dir": str(baseline_dir), "refined_ply": str(baseline_ply)})

    color_checkpoint = None
    for name, config, resume_name in experiments:
        exp_dir = out_root / name
        resume = None
        if resume_name == "appearance_color_only" and color_checkpoint is not None:
            resume = color_checkpoint
        refined_ply = _run_refine(lam, workspace, baseline_ply, exp_dir, config, resume=resume)
        _render_preview(lam, workspace, refined_ply, exp_dir)
        loss_plot = save_loss_plot(exp_dir / "loss_history.jsonl", exp_dir / "loss_history.png")
        if not args.skip_turntable:
            _export_turntable(lam, workspace, refined_ply, exp_dir, args.turntable_frames, args.turntable_size, args.fps)
        if name == "appearance_color_only":
            color_checkpoint = exp_dir / "checkpoints" / "latest.pt"
        summary.append({
            "name": name,
            "output_dir": str(exp_dir),
            "refined_ply": str(refined_ply),
            "resume": str(resume) if resume else None,
            "loss_plot": str(loss_plot) if loss_plot else None,
        })

    report = {"workspace": str(workspace), "experiments": summary}
    (out_root / "diagnostics_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def _experiment_configs(resolution_scale: float):
    pose = RefinementStageConfig(
        "pose",
        steps=200,
        lr=8e-4,
        resolution_scale=resolution_scale,
        optimize_expression=True,
        use_ssim=False,
        use_landmark=True,
    )
    appearance = RefinementStageConfig(
        "appearance",
        steps=300,
        lr=8e-4,
        resolution_scale=resolution_scale,
        optimize_appearance=True,
        optimize_exposure=False,
        optimize_opacity=False,
        use_ssim=True,
        appearance_delta_limit=0.08,
        rgb_mask_mode="render_target_intersection",
    )
    exposure = replace(appearance, name="appearance_color_exposure", optimize_appearance=False, optimize_exposure=True)
    opacity = replace(appearance, name="appearance_color_opacity", optimize_appearance=False, optimize_opacity=True)
    geometry = RefinementStageConfig(
        "geometry_light",
        steps=100,
        lr=2e-4,
        resolution_scale=resolution_scale,
        optimize_appearance=False,
        optimize_opacity=False,
        optimize_geometry=True,
        use_ssim=True,
        use_knn_anchor=True,
        appearance_delta_limit=0.08,
        rgb_mask_mode="render_target_intersection",
    )
    exposure_weights = LossWeights(exposure_reg=0.2)
    return [
        ("pose_only", RefinementConfig(stages=[pose]), None),
        ("appearance_color_only", RefinementConfig(stages=[appearance]), None),
        ("appearance_color_exposure", RefinementConfig(stages=[exposure], loss_weights=exposure_weights), "appearance_color_only"),
        ("appearance_color_opacity", RefinementConfig(stages=[opacity]), "appearance_color_only"),
        ("geometry_light_after_safe_appearance", RefinementConfig(stages=[geometry]), "appearance_color_only"),
    ]


def _run_refine(lam, workspace: Path, init_ply: Path, out_dir: Path, config: RefinementConfig, resume: Path | None = None) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    config.output_dir = str(out_dir)
    batch = load_multiview_bundle(workspace, require_undistorted=config.require_undistorted)
    gs = _load_renderable_gaussian(init_ply)
    gs.to_cuda()
    refiner = MultiViewGaussianRefiner(lam, config)
    refiner.debug_dir = out_dir / "overlays"
    refiner.run(gs, batch, resume=resume)
    return out_dir / "refined_gaussian.ply"


def _render_preview(lam, workspace: Path, ply_path: Path, out_dir: Path, max_views: int = 8) -> None:
    batch = load_multiview_bundle(workspace, require_undistorted=False).to("cuda", torch.float32)
    gs = _load_renderable_gaussian(ply_path)
    gs.to_cuda()
    query_points, flame_params = lam.renderer.get_query_points(batch.flame_params, device=batch.images.device)
    h, w = batch.images.shape[-2:]
    with torch.no_grad():
        out = render_animate_gs_with_intrinsics(lam.renderer, [gs], query_points, flame_params, batch.c2ws, batch.intrs, h, w, batch.bg_colors)
    save_overlay_grid(
        out_dir / "preview",
        batch.frame_ids,
        batch.images.detach().cpu(),
        batch.masks.detach().cpu(),
        out["comp_rgb"].detach().cpu(),
        out["comp_mask"].detach().cpu(),
        batch.landmarks_2d.detach().cpu() if batch.landmarks_2d is not None else None,
        max_items=max_views,
    )


def _export_turntable(lam, workspace: Path, ply_path: Path, out_dir: Path, num_frames: int, size: int, fps: int) -> Path:
    frame_dir = out_dir / "turntable_frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    video_path = out_dir / "turntable.mp4"
    device = torch.device("cuda")
    batch = load_multiview_bundle(workspace, require_undistorted=False).to(device, torch.float32)
    gs = _load_renderable_gaussian(ply_path)
    gs.to_cuda()

    xyz = gs.xyz.detach().cpu().numpy()
    center = np.median(xyz, axis=0).astype(np.float32)
    c2ws_np = batch.c2ws[0].detach().cpu().numpy()
    centers = c2ws_np[:, :3, 3]
    distance = float(np.median(np.linalg.norm(centers - center[None], axis=1))) * 1.6
    up = -c2ws_np[:, :3, 1]
    up = up / np.linalg.norm(up, axis=1, keepdims=True)
    up = up.mean(axis=0)
    up = up / np.linalg.norm(up)
    start_dir = centers[0] - center
    start_dir = start_dir - up * float(np.dot(start_dir, up))
    start_dir = start_dir / np.linalg.norm(start_dir)

    intr = torch.eye(4, device=device, dtype=torch.float32)
    intr[0, 0] = 1000.0 * float(size) / 768.0
    intr[1, 1] = 1000.0 * float(size) / 768.0
    intr[0, 2] = size / 2.0
    intr[1, 2] = size / 2.0
    all_c2ws = []
    for idx in range(num_frames):
        direction = _rodrigues(up.astype(np.float32), 2.0 * math.pi * idx / num_frames) @ start_dir
        all_c2ws.append(_look_at(center + distance * direction.astype(np.float32), center, up.astype(np.float32)))

    query_points, flame_params = lam.renderer.get_query_points(batch.flame_params, device=device)
    paths = []
    for start in range(0, num_frames, 6):
        end = min(start + 6, num_frames)
        n = end - start
        c2ws = torch.tensor(np.stack(all_c2ws[start:end]), device=device, dtype=torch.float32).unsqueeze(0)
        intrs = intr[None, None].repeat(1, n, 1, 1)
        bg = torch.ones((1, n, 3), device=device, dtype=torch.float32)
        flame = {}
        for key, value in flame_params.items():
            flame[key] = value if key == "betas" else value[:, 0:1].repeat(*([1, n] + [1] * (value.ndim - 2)))
        with torch.no_grad():
            out = render_animate_gs_with_intrinsics(lam.renderer, [gs], query_points, flame, c2ws, intrs, size, size, bg)
        imgs = out["comp_rgb"][0].detach().clamp(0, 1).cpu()
        for local_idx in range(n):
            idx = start + local_idx
            arr = (imgs[local_idx].permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
            path = frame_dir / f"frame_{idx:04d}.png"
            Image.fromarray(arr).save(path)
            paths.append(path)

    writer = imageio.get_writer(video_path, fps=fps, codec="libx264", quality=8, macro_block_size=16)
    try:
        for path in paths:
            writer.append_data(imageio.imread(path))
    finally:
        writer.close()
    return video_path


def _rodrigues(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    k = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=np.float32)
    eye = np.eye(3, dtype=np.float32)
    return eye + math.sin(angle) * k + (1.0 - math.cos(angle)) * (k @ k)


def _look_at(cam_pos: np.ndarray, target: np.ndarray, up_world: np.ndarray) -> np.ndarray:
    forward = target - cam_pos
    forward = forward / np.linalg.norm(forward)
    down = -up_world
    down = down - forward * float(np.dot(down, forward))
    down = down / np.linalg.norm(down)
    right = np.cross(down, forward)
    right = right / np.linalg.norm(right)
    down = np.cross(forward, right)
    down = down / np.linalg.norm(down)
    mat = np.eye(4, dtype=np.float32)
    mat[:3, 0] = right
    mat[:3, 1] = down
    mat[:3, 2] = forward
    mat[:3, 3] = cam_pos
    return mat


if __name__ == "__main__":
    main()
