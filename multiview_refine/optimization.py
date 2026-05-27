import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from lam.models.rendering.gaussian_model import GaussianModel

from .render_adapter import render_animate_gs_with_intrinsics
from .types import MultiViewBatch, TensorDict
from .visualization import append_jsonl, save_overlay_grid


@dataclass
class LossWeights:
    rgb: float = 1.0
    ssim: float = 0.15
    mask: float = 0.2
    mask_iou: float = 0.1
    mask_boundary: float = 0.1
    landmark: float = 0.02
    sim3_reg: float = 0.01
    camera_delta_reg: float = 0.01
    intrinsics_reg: float = 0.1
    exposure_reg: float = 0.01
    xyz_anchor: float = 0.001
    offset_anchor: float = 0.01
    scale_anchor: float = 0.001
    opacity_reg: float = 0.0001
    expr_reg: float = 0.01
    jaw_reg: float = 0.05
    eyes_reg: float = 0.05
    knn_anchor: float = 0.001
    scale_limit: float = 0.001


@dataclass
class RefinementStageConfig:
    name: str
    steps: int
    lr: float
    views_per_step: int = 4
    optimize_global_sim3: bool = False
    optimize_per_view_camera: bool = False
    optimize_appearance: bool = False
    optimize_geometry: bool = False
    optimize_expression: bool = False
    optimize_intrinsics: bool = False
    optimize_exposure: bool = False
    resolution_scale: float = 1.0
    use_ssim: bool = True
    use_landmark: bool = False
    use_knn_anchor: bool = False
    lr_decay_xyz: float = 0.2
    lr_decay_appearance: float = 0.5
    lr_decay_camera: float = 1.0
    lr_decay_expression: float = 1.0


@dataclass
class RefinementConfig:
    stages: List[RefinementStageConfig] = field(default_factory=lambda: [
        RefinementStageConfig(
            "calibrate",
            steps=300,
            lr=1e-3,
            resolution_scale=0.5,
            optimize_global_sim3=True,
            optimize_per_view_camera=True,
            optimize_intrinsics=True,
            use_ssim=True,
            use_landmark=True,
        ),
        RefinementStageConfig(
            "pose",
            steps=200,
            lr=8e-4,
            resolution_scale=0.5,
            optimize_expression=True,
            use_ssim=False,
            use_landmark=True,
        ),
        RefinementStageConfig(
            "appearance",
            steps=800,
            lr=5e-3,
            optimize_appearance=True,
            optimize_exposure=True,
            use_ssim=True,
        ),
        RefinementStageConfig(
            "geometry_light",
            steps=300,
            lr=1e-3,
            optimize_appearance=True,
            optimize_geometry=True,
            use_ssim=True,
            use_knn_anchor=True,
        ),
        RefinementStageConfig(
            "geometry_xyz",
            steps=100,
            lr=5e-4,
            optimize_geometry=True,
            use_ssim=True,
            use_knn_anchor=True,
        ),
    ])
    loss_weights: LossWeights = field(default_factory=LossWeights)
    device: str = "cuda"
    dtype: torch.dtype = torch.float32
    log_every: int = 20
    debug_every: int = 100
    output_dir: str = "output/multiview_refine/run"
    knn_k: int = 6
    knn_max_points: int = 30000
    require_undistorted: bool = True
    use_local_projection_adapter: bool = True
    log_grad_diagnostics: bool = True


@dataclass
class RenderContext:
    c2ws: torch.Tensor
    intrs: torch.Tensor
    flame_params: TensorDict
    landmark_2d: Optional[torch.Tensor] = None


class MultiViewGaussianRefiner:
    def __init__(self, lam_model, config: Optional[RefinementConfig] = None) -> None:
        self.renderer = lam_model.renderer
        self.config = config or RefinementConfig()
        self.output_dir = Path(self.config.output_dir)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.debug_dir = self.output_dir / "debug"
        self.best_metric = float("inf")

    def run(self, initial_gs: GaussianModel, batch: MultiViewBatch, resume: Optional[str | Path] = None) -> GaussianModel:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        device = torch.device(self.config.device)
        batch = batch.to(device=device, dtype=self.config.dtype)
        query_points, flame_params = self.renderer.get_query_points(batch.flame_params, device=device)
        gs_state = OptimizableGaussianState(
            initial_gs,
            query_points=query_points,
            knn_k=self.config.knn_k,
            knn_max_points=self.config.knn_max_points,
        ).to(device=device)
        camera_state = OptimizableCameraState(batch.c2ws.shape[1]).to(device=device)
        intr_state = OptimizableIntrinsicsState(batch.c2ws.shape[1]).to(device=device)
        exposure_state = OptimizableExposureState(batch.c2ws.shape[1]).to(device=device)
        expr_state = OptimizableExpressionState(batch.flame_params).to(device=device)
        sampler = RoundRobinViewSampler(batch.c2ws.shape[1], device=device)
        start_stage, start_step = 0, 0
        if resume:
            start_stage, start_step = self._load_checkpoint(resume, gs_state, camera_state, intr_state, exposure_state, expr_state)

        for stage_idx, stage in enumerate(self.config.stages):
            if stage_idx < start_stage:
                continue
            optimizer, scheduler = self._build_optimizer(gs_state, camera_state, intr_state, exposure_state, expr_state, stage)
            first_step = start_step if stage_idx == start_stage else 0
            sampler.reset()
            for step in range(first_step, stage.steps):
                view_idx = sampler.next(stage.views_per_step)
                sub_batch = _scale_batch(batch.index(view_idx), stage.resolution_scale)
                sub_flame = _index_flame_params(flame_params, view_idx)
                optimizer.zero_grad(set_to_none=True)
                render, context = self._render(gs_state, camera_state, intr_state, exposure_state, expr_state, query_points, sub_flame, sub_batch, stage)
                losses = compute_losses(
                    render,
                    sub_batch,
                    gs_state,
                    camera_state,
                    intr_state,
                    exposure_state,
                    expr_state,
                    context,
                    self.config.loss_weights,
                    stage,
                )
                losses["total"].backward()
                if self.config.log_grad_diagnostics and step == first_step:
                    self._log_grad_diagnostics(stage.name, step + 1, gs_state, camera_state, intr_state, exposure_state, expr_state)
                optimizer.step()
                scheduler.step()

                metric = float(losses["total"].detach().cpu())
                if metric < self.best_metric:
                    self.best_metric = metric
                    self._save_checkpoint("best.pt", stage_idx, step + 1, gs_state, camera_state, intr_state, exposure_state, expr_state)

                if step == 0 or (step + 1) % self.config.log_every == 0 or step + 1 == stage.steps:
                    metrics = {k: float(v.detach().cpu()) for k, v in losses.items()}
                    append_jsonl(self.output_dir / "loss_history.jsonl", {"stage": stage.name, "step": step + 1, **metrics})
                    print(f"[multiview-refine:{stage.name}] {step + 1}/{stage.steps} {metrics}")
                if step == 0 or (step + 1) % self.config.debug_every == 0 or step + 1 == stage.steps:
                    self._save_debug(stage.name, step + 1, sub_batch, render)
                    self._save_checkpoint("latest.pt", stage_idx, step + 1, gs_state, camera_state, intr_state, exposure_state, expr_state)
            start_step = 0

        refined = gs_state.to_gaussian_model()
        refined.save_ply(self.output_dir / "refined_gaussian.ply", rgb2sh=False, offset2xyz=False)
        torch.save(camera_state.state_dict(), self.output_dir / "camera_delta.pt")
        torch.save(intr_state.state_dict(), self.output_dir / "intrinsics_delta.pt")
        torch.save(exposure_state.state_dict(), self.output_dir / "exposure_delta.pt")
        torch.save(expr_state.state_dict(), self.output_dir / "pose_delta.pt")
        torch.save(gs_state.geometry_delta_state_dict(), self.output_dir / "gaussian_geometry_delta.pt")
        with (self.output_dir / "refine_config.json").open("w", encoding="utf-8") as fp:
            raw = asdict(self.config)
            raw["dtype"] = str(self.config.dtype)
            json.dump(raw, fp, indent=2)
        return refined

    def _render(
        self,
        gs_state: "OptimizableGaussianState",
        camera_state: "OptimizableCameraState",
        intr_state: "OptimizableIntrinsicsState",
        exposure_state: "OptimizableExposureState",
        expr_state: "OptimizableExpressionState",
        query_points: torch.Tensor,
        flame_params: TensorDict,
        batch: MultiViewBatch,
        stage: RefinementStageConfig,
    ) -> tuple[Dict[str, torch.Tensor], RenderContext]:
        h, w = batch.images.shape[-2:]
        flame_params = expr_state.apply(flame_params, batch.view_indices) if stage.optimize_expression else flame_params
        c2ws = camera_state(batch.c2ws, batch.view_indices)
        intrs = intr_state(batch.intrs, batch.view_indices)
        gs_model = gs_state.to_gaussian_model()
        if self.config.use_local_projection_adapter:
            render = render_animate_gs_with_intrinsics(self.renderer, [gs_model], query_points, flame_params, c2ws, intrs, h, w, batch.bg_colors)
        else:
            render = self.renderer.forward_animate_gs([gs_model], query_points, flame_params, c2ws, intrs, h, w, batch.bg_colors)
        render = dict(render)
        render["comp_rgb"] = exposure_state.apply(render["comp_rgb"], batch.view_indices)
        landmarks = None
        if stage.use_landmark and batch.landmarks_2d is not None:
            landmarks = _project_flame_landmarks(self.renderer, query_points, flame_params, c2ws, intrs, h, w)
        return render, RenderContext(c2ws=c2ws, intrs=intrs, flame_params=flame_params, landmark_2d=landmarks)

    def _build_optimizer(self, gs_state, camera_state, intr_state, exposure_state, expr_state, stage):
        gs_state.set_trainable(
            appearance=stage.optimize_appearance,
            light_geometry=stage.optimize_geometry and stage.name != "geometry_xyz",
            xyz_geometry=stage.name == "geometry_xyz",
        )
        camera_state.set_trainable(stage.optimize_global_sim3, stage.optimize_per_view_camera)
        intr_state.set_trainable(stage.optimize_intrinsics)
        exposure_state.set_trainable(stage.optimize_exposure)
        expr_state.set_trainable(stage.optimize_expression)
        groups = []
        _add_group(groups, [camera_state.delta_log_scale, camera_state.global_axis_angle, camera_state.global_translation], stage.lr, stage.lr_decay_camera)
        _add_group(groups, [camera_state.per_view_axis_angle, camera_state.per_view_translation], stage.lr * 0.25, stage.lr_decay_camera)
        _add_group(groups, list(intr_state.parameters()), stage.lr * 0.1, stage.lr_decay_camera)
        _add_group(groups, list(exposure_state.parameters()), stage.lr, stage.lr_decay_appearance)
        _add_group(groups, [gs_state.shs], stage.lr, stage.lr_decay_appearance)
        _add_group(groups, [gs_state.opacity_logit, gs_state.log_scaling, gs_state.rotation], stage.lr * 0.25, stage.lr_decay_xyz)
        _add_group(groups, [gs_state.xyz_delta, gs_state.offset_delta], stage.lr * 0.05, stage.lr_decay_xyz)
        _add_group(groups, list(expr_state.parameters()), stage.lr * 0.5, stage.lr_decay_expression)
        if not groups:
            raise ValueError(f"Stage {stage.name} has no trainable parameters")
        optimizer = torch.optim.Adam(groups)
        gammas = []
        for group in optimizer.param_groups:
            decay_ratio = float(group.pop("_decay_ratio", 0.5))
            gammas.append(decay_ratio ** (1.0 / max(stage.steps, 1)))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, [lambda step, g=g: g ** step for g in gammas])
        return optimizer, scheduler

    def _log_grad_diagnostics(self, stage: str, step: int, gs_state, camera_state, intr_state, exposure_state, expr_state) -> None:
        diagnostics = {
            "stage": stage,
            "step": step,
            "shs": _grad_norm([gs_state.shs]),
            "offset_delta": _grad_norm([gs_state.offset_delta]),
            "xyz_delta": _grad_norm([gs_state.xyz_delta]),
            "log_scaling": _grad_norm([gs_state.log_scaling]),
            "rotation": _grad_norm([gs_state.rotation]),
            "opacity_logit": _grad_norm([gs_state.opacity_logit]),
            "camera_global": _grad_norm([camera_state.delta_log_scale, camera_state.global_axis_angle, camera_state.global_translation]),
            "camera_per_view": _grad_norm([camera_state.per_view_axis_angle, camera_state.per_view_translation]),
            "intrinsics": _grad_norm(list(intr_state.parameters())),
            "exposure": _grad_norm(list(exposure_state.parameters())),
            "expression": _grad_norm(list(expr_state.parameters())),
        }
        append_jsonl(self.output_dir / "grad_diagnostics.jsonl", diagnostics)
        print(f"[multiview-refine:{stage}] grad diagnostics {diagnostics}")
        if (diagnostics["camera_global"] + diagnostics["camera_per_view"] + diagnostics["intrinsics"]) <= 1e-12:
            print(
                f"[multiview-refine:{stage}] warning: camera/intrinsics gradients are zero or unavailable; "
                "use manual/explicit Sim3 alignment if calibration does not move."
            )

    def _save_debug(self, stage: str, step: int, batch: MultiViewBatch, render: Dict[str, torch.Tensor]) -> None:
        save_overlay_grid(
            self.debug_dir / stage / f"step_{step:06d}",
            batch.frame_ids,
            batch.images.detach().cpu(),
            batch.masks.detach().cpu(),
            render["comp_rgb"].detach().cpu(),
            render["comp_mask"].detach().cpu(),
            batch.landmarks_2d.detach().cpu() if batch.landmarks_2d is not None else None,
        )

    def _save_checkpoint(self, name: str, stage_idx: int, step: int, gs_state, camera_state, intr_state, exposure_state, expr_state) -> None:
        payload = {
            "stage_idx": stage_idx,
            "step": step,
            "best_metric": self.best_metric,
            "gaussian": gs_state.state_dict(),
            "camera": camera_state.state_dict(),
            "intrinsics": intr_state.state_dict(),
            "exposure": exposure_state.state_dict(),
            "expression": expr_state.state_dict(),
        }
        torch.save(payload, self.checkpoint_dir / name)

    def _load_checkpoint(self, path: str | Path, gs_state, camera_state, intr_state, exposure_state, expr_state) -> tuple[int, int]:
        payload = torch.load(path, map_location="cpu")
        gs_state.load_state_dict(payload["gaussian"], strict=False)
        camera_state.load_state_dict(payload["camera"], strict=False)
        if "intrinsics" in payload:
            intr_state.load_state_dict(payload["intrinsics"], strict=False)
        if "exposure" in payload:
            exposure_state.load_state_dict(payload["exposure"], strict=False)
        if "expression" in payload:
            expr_state.load_state_dict(payload["expression"], strict=False)
        self.best_metric = float(payload.get("best_metric", float("inf")))
        return int(payload.get("stage_idx", 0)), int(payload.get("step", 0))


class RoundRobinViewSampler:
    def __init__(self, num_views: int, device) -> None:
        self.num_views = num_views
        self.device = device
        self.order = torch.randperm(num_views, device=device)
        self.cursor = 0

    def reset(self) -> None:
        self.order = torch.randperm(self.num_views, device=self.device)
        self.cursor = 0

    def next(self, views_per_step: int) -> torch.Tensor:
        if views_per_step <= 0 or views_per_step >= self.num_views:
            return torch.arange(self.num_views, device=self.device)
        chunks = []
        remaining = views_per_step
        while remaining > 0:
            available = self.num_views - self.cursor
            take = min(remaining, available)
            chunks.append(self.order[self.cursor:self.cursor + take])
            self.cursor += take
            remaining -= take
            if self.cursor >= self.num_views:
                self.reset()
        return torch.cat(chunks).sort()[0]


class OptimizableGaussianState(nn.Module):
    def __init__(self, gs: GaussianModel, query_points: torch.Tensor, knn_k: int = 6, knn_max_points: int = 30000) -> None:
        super().__init__()
        initial_xyz = gs.xyz.detach().clone()
        cano_points = query_points[0].detach().clone()
        bound_mode = initial_xyz.shape == cano_points.shape
        if bound_mode:
            base_offset = initial_xyz - cano_points
            anchor_offset = base_offset
            xyz_delta = torch.zeros_like(initial_xyz)
        else:
            print(
                "[multiview-refine] warning: initial Gaussian xyz shape does not match FLAME query points; "
                "falling back to baked xyz geometry mode."
            )
            base_offset = gs.offset.detach().clone()
            anchor_offset = base_offset
            xyz_delta = torch.zeros_like(initial_xyz)
        self.register_buffer("base_cano", cano_points if bound_mode else torch.zeros_like(initial_xyz))
        self.register_buffer("base_offset", base_offset)
        self.register_buffer("anchor_xyz", initial_xyz)
        self.register_buffer("anchor_offset", anchor_offset)
        self.register_buffer("bound_mode", torch.tensor(bound_mode, dtype=torch.bool))
        self.xyz_delta = nn.Parameter(xyz_delta)
        self.offset_delta = nn.Parameter(torch.zeros_like(base_offset))
        self.shs = nn.Parameter(gs.shs.detach().clone())
        self.opacity_logit = nn.Parameter(_inverse_sigmoid(gs.opacity.detach().clone().clamp(1e-4, 1.0 - 1e-4)))
        self.log_scaling = nn.Parameter(gs.scaling.detach().clone().clamp_min(1e-8).log())
        self.rotation = nn.Parameter(gs.rotation.detach().clone())
        self.register_buffer("anchor_scaling", gs.scaling.detach().clone())
        src, dst, dist = _build_knn_edges(initial_xyz, knn_k, knn_max_points)
        self.register_buffer("knn_src", src)
        self.register_buffer("knn_dst", dst)
        self.register_buffer("knn_dist", dist)

    @property
    def opacity(self) -> torch.Tensor:
        return torch.sigmoid(self.opacity_logit)

    @property
    def scaling(self) -> torch.Tensor:
        return torch.exp(self.log_scaling)

    @property
    def xyz(self) -> torch.Tensor:
        if bool(self.bound_mode.item()):
            return self.base_cano + self.base_offset + self.offset_delta
        return self.anchor_xyz + self.xyz_delta

    @property
    def offset(self) -> torch.Tensor:
        return self.base_offset + self.offset_delta

    def set_trainable(self, appearance: bool, light_geometry: bool, xyz_geometry: bool) -> None:
        self.shs.requires_grad_(appearance)
        self.opacity_logit.requires_grad_(appearance)
        self.log_scaling.requires_grad_(light_geometry)
        self.rotation.requires_grad_(light_geometry)
        self.offset_delta.requires_grad_(light_geometry or xyz_geometry)
        self.xyz_delta.requires_grad_(xyz_geometry and not bool(self.bound_mode.item()))

    def knn_anchor_loss(self) -> torch.Tensor:
        if self.knn_src.numel() == 0:
            return self.xyz.sum() * 0.0
        current = torch.linalg.norm(self.xyz[self.knn_src] - self.xyz[self.knn_dst], dim=-1)
        return F.smooth_l1_loss(current, self.knn_dist)

    def scale_limit_loss(self, min_scale: float = 1e-4, max_scale: float = 0.08) -> torch.Tensor:
        scale = self.scaling
        return F.relu(min_scale - scale).square().mean() + F.relu(scale - max_scale).square().mean()

    def to_gaussian_model(self) -> GaussianModel:
        return GaussianModel(
            xyz=self.xyz,
            offset=self.offset,
            shs=self.shs,
            opacity=self.opacity,
            scaling=self.scaling,
            rotation=F.normalize(self.rotation, dim=-1),
        )

    def geometry_delta_state_dict(self) -> dict:
        return {
            "bound_mode": bool(self.bound_mode.item()),
            "base_offset": self.base_offset.detach().cpu(),
            "offset_delta": self.offset_delta.detach().cpu(),
            "xyz_delta": self.xyz_delta.detach().cpu(),
            "effective_xyz": self.xyz.detach().cpu(),
        }


class OptimizableCameraState(nn.Module):
    def __init__(self, num_views: int) -> None:
        super().__init__()
        self.delta_log_scale = nn.Parameter(torch.zeros(1))
        self.global_axis_angle = nn.Parameter(torch.zeros(3))
        self.global_translation = nn.Parameter(torch.zeros(3))
        self.per_view_axis_angle = nn.Parameter(torch.zeros(num_views, 3))
        self.per_view_translation = nn.Parameter(torch.zeros(num_views, 3))
        self.set_trainable(False, False)

    def set_trainable(self, global_sim3: bool, per_view: bool) -> None:
        self.delta_log_scale.requires_grad_(global_sim3)
        self.global_axis_angle.requires_grad_(global_sim3)
        self.global_translation.requires_grad_(global_sim3)
        self.per_view_axis_angle.requires_grad_(per_view)
        self.per_view_translation.requires_grad_(per_view)

    def forward(self, c2ws: torch.Tensor, view_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        _, v = c2ws.shape[:2]
        if view_indices is None:
            view_indices = torch.arange(v, device=c2ws.device)
        view_indices = view_indices.to(device=c2ws.device, dtype=torch.long)
        scale = torch.exp(self.delta_log_scale)
        global_rot = _axis_angle_to_matrix(self.global_axis_angle).reshape(1, 1, 3, 3)
        global_translation = self.global_translation.reshape(1, 1, 3)
        out = c2ws.clone()
        out[..., :3, :3] = torch.matmul(global_rot, c2ws[..., :3, :3])
        out[..., :3, 3] = scale * torch.matmul(global_rot, c2ws[..., :3, 3:4]).squeeze(-1) + global_translation
        per_view_rot = _axis_angle_to_matrix(self.per_view_axis_angle[view_indices]).unsqueeze(0)
        per_view_translation = self.per_view_translation[view_indices].unsqueeze(0)
        out = out.clone()
        out[..., :3, :3] = torch.matmul(per_view_rot, out[..., :3, :3])
        out[..., :3, 3] = torch.matmul(per_view_rot, out[..., :3, 3:4]).squeeze(-1) + per_view_translation
        return out


class OptimizableIntrinsicsState(nn.Module):
    def __init__(self, num_views: int, max_focal_change: float = 0.03, max_principal_delta: float = 32.0) -> None:
        super().__init__()
        self.log_focal_scale = nn.Parameter(torch.zeros(num_views, 2))
        self.principal_delta = nn.Parameter(torch.zeros(num_views, 2))
        self.max_log_focal = math.log(1.0 + max_focal_change)
        self.max_principal_delta = max_principal_delta
        self.set_trainable(False)

    def set_trainable(self, enabled: bool) -> None:
        self.log_focal_scale.requires_grad_(enabled)
        self.principal_delta.requires_grad_(enabled)

    def forward(self, intrs: torch.Tensor, view_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        _, v = intrs.shape[:2]
        if view_indices is None:
            view_indices = torch.arange(v, device=intrs.device)
        view_indices = view_indices.to(device=intrs.device, dtype=torch.long)
        out = intrs.clone()
        focal_delta = self.max_log_focal * torch.tanh(self.log_focal_scale[view_indices])
        principal_delta = self.max_principal_delta * torch.tanh(self.principal_delta[view_indices])
        out[:, :, 0, 0] = out[:, :, 0, 0] * torch.exp(focal_delta[:, 0]).unsqueeze(0)
        out[:, :, 1, 1] = out[:, :, 1, 1] * torch.exp(focal_delta[:, 1]).unsqueeze(0)
        out[:, :, 0, 2] = out[:, :, 0, 2] + principal_delta[:, 0].unsqueeze(0)
        out[:, :, 1, 2] = out[:, :, 1, 2] + principal_delta[:, 1].unsqueeze(0)
        return out

    def regularization(self) -> torch.Tensor:
        return torch.tanh(self.log_focal_scale).square().mean() + torch.tanh(self.principal_delta).square().mean()


class OptimizableExposureState(nn.Module):
    def __init__(self, num_views: int, max_gain: float = 0.25, max_bias: float = 0.15) -> None:
        super().__init__()
        self.log_gain = nn.Parameter(torch.zeros(num_views, 3))
        self.bias = nn.Parameter(torch.zeros(num_views, 3))
        self.max_log_gain = math.log(1.0 + max_gain)
        self.max_bias = max_bias
        self.set_trainable(False)

    def set_trainable(self, enabled: bool) -> None:
        self.log_gain.requires_grad_(enabled)
        self.bias.requires_grad_(enabled)

    def apply(self, rgb: torch.Tensor, view_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        _, v = rgb.shape[:2]
        if view_indices is None:
            view_indices = torch.arange(v, device=rgb.device)
        view_indices = view_indices.to(device=rgb.device, dtype=torch.long)
        gain = torch.exp(self.max_log_gain * torch.tanh(self.log_gain[view_indices])).reshape(1, v, 3, 1, 1)
        bias = (self.max_bias * torch.tanh(self.bias[view_indices])).reshape(1, v, 3, 1, 1)
        return (rgb * gain + bias).clamp(0.0, 1.0)

    def regularization(self) -> torch.Tensor:
        return torch.tanh(self.log_gain).square().mean() + torch.tanh(self.bias).square().mean()


class OptimizableExpressionState(nn.Module):
    def __init__(self, flame_params: TensorDict) -> None:
        super().__init__()
        num_views = next(v for k, v in flame_params.items() if k != "betas").shape[1]
        expr_dim = flame_params.get("expr", torch.zeros(1, num_views, 50)).shape[-1]
        self.expr_delta = nn.Parameter(torch.zeros(1, num_views, expr_dim))
        self.jaw_delta = nn.Parameter(torch.zeros(1, num_views, 3))
        self.eyes_delta = nn.Parameter(torch.zeros(1, num_views, 6))
        self.set_trainable(False)

    def set_trainable(self, enabled: bool) -> None:
        self.expr_delta.requires_grad_(enabled)
        self.jaw_delta.requires_grad_(enabled)
        self.eyes_delta.requires_grad_(enabled)

    def apply(self, flame_params: TensorDict, view_indices: Optional[torch.Tensor] = None) -> TensorDict:
        out = dict(flame_params)
        v = out["expr"].shape[1]
        if view_indices is None:
            view_indices = torch.arange(v, device=out["expr"].device)
        view_indices = view_indices.to(device=out["expr"].device, dtype=torch.long)
        out["expr"] = out["expr"] + self.expr_delta[:, view_indices].clamp(-0.1, 0.1)
        out["jaw_pose"] = out["jaw_pose"] + self.jaw_delta[:, view_indices].clamp(-0.08, 0.08)
        out["eyes_pose"] = out["eyes_pose"] + self.eyes_delta[:, view_indices].clamp(-0.08, 0.08)
        return out

    def regularization(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.expr_delta.square().mean(), self.jaw_delta.square().mean(), self.eyes_delta.square().mean()


def compute_losses(
    render: Dict[str, torch.Tensor],
    batch: MultiViewBatch,
    gs,
    camera_state,
    intr_state,
    exposure_state,
    expr_state,
    context: RenderContext,
    weights: LossWeights,
    stage: RefinementStageConfig,
) -> Dict[str, torch.Tensor]:
    mask = batch.masks.clamp(0, 1)
    rgb_l1 = (render["comp_rgb"] - batch.images).abs() * mask
    rgb_loss = rgb_l1.sum() / mask.expand_as(render["comp_rgb"]).sum().clamp_min(1.0)
    ssim_loss = _masked_ssim_loss(render["comp_rgb"], batch.images, mask) if stage.use_ssim else rgb_loss * 0.0
    mask_l1 = F.l1_loss(render["comp_mask"], batch.masks)
    mask_iou = _soft_iou_loss(render["comp_mask"], batch.masks)
    mask_boundary = _boundary_mask_loss(render["comp_mask"], batch.masks)
    landmark = _landmark_loss(context.landmark_2d, batch.landmarks_2d, batch.images.shape[-2:]) if stage.use_landmark else rgb_loss * 0.0
    sim3_reg = camera_state.delta_log_scale.square().mean() + camera_state.global_axis_angle.square().mean() + camera_state.global_translation.square().mean()
    cam_reg = camera_state.per_view_axis_angle.square().mean() + camera_state.per_view_translation.square().mean()
    intr_reg = intr_state.regularization()
    exposure_reg = exposure_state.regularization()
    xyz_anchor = F.mse_loss(gs.xyz, gs.anchor_xyz)
    offset_anchor = F.mse_loss(gs.offset, gs.anchor_offset)
    scale_anchor = F.mse_loss(gs.scaling, gs.anchor_scaling)
    opacity_reg = gs.opacity.mean()
    expr_reg, jaw_reg, eyes_reg = expr_state.regularization()
    knn_anchor = gs.knn_anchor_loss() if stage.use_knn_anchor else rgb_loss * 0.0
    scale_limit = gs.scale_limit_loss()
    total = (
        weights.rgb * rgb_loss
        + weights.ssim * ssim_loss
        + weights.mask * mask_l1
        + weights.mask_iou * mask_iou
        + weights.mask_boundary * mask_boundary
        + weights.landmark * landmark
        + weights.sim3_reg * sim3_reg
        + weights.camera_delta_reg * cam_reg
        + weights.intrinsics_reg * intr_reg
        + weights.exposure_reg * exposure_reg
        + weights.xyz_anchor * xyz_anchor
        + weights.offset_anchor * offset_anchor
        + weights.scale_anchor * scale_anchor
        + weights.opacity_reg * opacity_reg
        + weights.expr_reg * expr_reg
        + weights.jaw_reg * jaw_reg
        + weights.eyes_reg * eyes_reg
        + weights.knn_anchor * knn_anchor
        + weights.scale_limit * scale_limit
    )
    return {
        "total": total,
        "rgb": rgb_loss.detach(),
        "ssim": ssim_loss.detach(),
        "mask": mask_l1.detach(),
        "mask_iou": mask_iou.detach(),
        "mask_boundary": mask_boundary.detach(),
        "landmark": landmark.detach(),
        "sim3_reg": sim3_reg.detach(),
        "camera_delta_reg": cam_reg.detach(),
        "intrinsics_reg": intr_reg.detach(),
        "exposure_reg": exposure_reg.detach(),
        "xyz_anchor": xyz_anchor.detach(),
        "offset_anchor": offset_anchor.detach(),
        "scale_anchor": scale_anchor.detach(),
        "opacity_reg": opacity_reg.detach(),
        "expr_reg": expr_reg.detach(),
        "jaw_reg": jaw_reg.detach(),
        "eyes_reg": eyes_reg.detach(),
        "knn_anchor": knn_anchor.detach(),
        "scale_limit": scale_limit.detach(),
    }


def _grad_norm(params: list[torch.nn.Parameter]) -> float:
    total = 0.0
    for param in params:
        if param.grad is None:
            continue
        total += float(param.grad.detach().float().square().sum().cpu())
    return math.sqrt(total)


def _add_group(groups: list, params: list[torch.nn.Parameter], lr: float, decay_ratio: float = 0.5) -> None:
    active = [p for p in params if p.requires_grad]
    if active:
        groups.append({"params": active, "lr": lr, "_decay_ratio": decay_ratio})


def _scale_batch(batch: MultiViewBatch, scale: float) -> MultiViewBatch:
    if scale >= 0.999:
        return batch
    h, w = batch.images.shape[-2:]
    size = (max(1, int(round(h * scale))), max(1, int(round(w * scale))))
    images = F.interpolate(batch.images.flatten(0, 1), size=size, mode="bilinear", align_corners=False).reshape(batch.images.shape[:2] + batch.images.shape[2:3] + size)
    masks = F.interpolate(batch.masks.flatten(0, 1), size=size, mode="bilinear", align_corners=False).reshape(batch.masks.shape[:2] + batch.masks.shape[2:3] + size)
    intrs = batch.intrs.clone()
    intrs[..., 0, 0] *= scale
    intrs[..., 1, 1] *= scale
    intrs[..., 0, 2] *= scale
    intrs[..., 1, 2] *= scale
    landmarks = batch.landmarks_2d
    if landmarks is not None and _landmarks_are_pixels(landmarks):
        landmarks = landmarks.clone()
        landmarks[..., :2] *= scale
    return MultiViewBatch(images, masks, batch.c2ws, intrs, batch.bg_colors, batch.flame_params, batch.frame_ids, landmarks, batch.view_indices)


def _index_flame_params(flame_params: TensorDict, indices: torch.Tensor) -> TensorDict:
    return {key: value if key == "betas" else value[:, indices] for key, value in flame_params.items()}


def _project_flame_landmarks(renderer, query_points: torch.Tensor, flame_params: TensorDict, c2ws: torch.Tensor, intrs: torch.Tensor, height: int, width: int) -> torch.Tensor:
    flame = {k: v[0] if k != "betas" else v for k, v in flame_params.items()}
    num_views = flame["expr"].shape[0]
    v_cano = query_points[0].unsqueeze(0).repeat(num_views, 1, 1)
    expr = torch.cat([flame["expr"], flame["teeth_bs"]], dim=-1) if getattr(renderer, "teeth_bs_flag", False) and "teeth_bs" in flame else flame["expr"]
    ret = renderer.flame_model.animation_forward(
        v_cano=v_cano,
        shape=flame["betas"].repeat(num_views, 1),
        expr=expr,
        rotation=flame["rotation"],
        neck=flame["neck_pose"],
        jaw=flame["jaw_pose"],
        eyes=flame["eyes_pose"],
        translation=flame["translation"],
        zero_centered_at_root_node=False,
        return_landmarks=True,
        return_verts_cano=False,
        static_offset=None,
    )
    landmarks = ret["landmarks"].unsqueeze(0)
    return _project_points(landmarks, c2ws, intrs, height, width)


def _project_points(points: torch.Tensor, c2ws: torch.Tensor, intrs: torch.Tensor, height: int, width: int) -> torch.Tensor:
    w2cs = torch.linalg.inv(c2ws)
    ones = torch.ones_like(points[..., :1])
    homog = torch.cat([points, ones], dim=-1)
    cam = torch.matmul(w2cs[:, :, None], homog[..., None]).squeeze(-1)[..., :3]
    z = cam[..., 2].clamp_min(1e-6)
    u = intrs[:, :, None, 0, 0] * (cam[..., 0] / z) + intrs[:, :, None, 0, 2]
    v = intrs[:, :, None, 1, 1] * (cam[..., 1] / z) + intrs[:, :, None, 1, 2]
    return torch.stack([u.clamp(-width, 2 * width), v.clamp(-height, 2 * height)], dim=-1)


def _landmark_loss(pred: Optional[torch.Tensor], target: Optional[torch.Tensor], image_hw: tuple[int, int]) -> torch.Tensor:
    if pred is None or target is None:
        ref = pred if pred is not None else target
        return torch.tensor(0.0, device=ref.device if ref is not None else "cpu")
    target_xy = target[..., :2]
    if not _landmarks_are_pixels(target):
        h, w = image_hw
        target_xy = target_xy.clone()
        target_xy[..., 0] *= w
        target_xy[..., 1] *= h
    count = min(pred.shape[2], target_xy.shape[2])
    pred = pred[:, :, :count]
    target_xy = target_xy[:, :, :count]
    valid = torch.isfinite(pred).all(dim=-1) & torch.isfinite(target_xy).all(dim=-1)
    if target.shape[-1] > 2:
        valid = valid & (target[..., :count, 2] > 0)
    if not valid.any():
        return pred.sum() * 0.0
    return F.smooth_l1_loss(pred[valid], target_xy[valid])


def _landmarks_are_pixels(landmarks: torch.Tensor) -> bool:
    finite = landmarks[..., :2][torch.isfinite(landmarks[..., :2])]
    return bool(finite.numel() > 0 and finite.detach().max() > 2.0)


def _masked_ssim_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    p = pred.flatten(0, 1)
    t = target.flatten(0, 1)
    mu_x = F.avg_pool2d(p, 3, 1, 1)
    mu_y = F.avg_pool2d(t, 3, 1, 1)
    sigma_x = F.avg_pool2d(p ** 2, 3, 1, 1) - mu_x ** 2
    sigma_y = F.avg_pool2d(t ** 2, 3, 1, 1) - mu_y ** 2
    sigma_xy = F.avg_pool2d(p * t, 3, 1, 1) - mu_x * mu_y
    ssim = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / ((mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2)).clamp_min(1e-6)
    weight = F.avg_pool2d(mask.flatten(0, 1), 3, 1, 1)
    return ((1.0 - ssim.clamp(-1, 1)) * weight).sum() / weight.sum().clamp_min(1.0)


def _soft_iou_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = pred.clamp(0, 1)
    target = target.clamp(0, 1)
    inter = (pred * target).sum()
    union = (pred + target - pred * target).sum().clamp_min(1.0)
    return 1.0 - inter / union


def _boundary_mask_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = pred.clamp(0, 1)
    target = target.clamp(0, 1)
    h, w = target.shape[-2:]
    k = max(3, min(11, min(h, w) // 64))
    dilated = F.max_pool2d(target.flatten(0, 1), k, 1, k // 2)
    eroded = -F.max_pool2d(-target.flatten(0, 1), k, 1, k // 2)
    boundary = (dilated - eroded).reshape_as(target)
    return ((pred - target).abs() * boundary).sum() / boundary.sum().clamp_min(1.0)


def _build_knn_edges(xyz: torch.Tensor, k: int, max_points: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n = xyz.shape[0]
    if n <= k + 1 or n > max_points:
        device = xyz.device
        return torch.empty(0, dtype=torch.long, device=device), torch.empty(0, dtype=torch.long, device=device), torch.empty(0, dtype=xyz.dtype, device=device)
    with torch.no_grad():
        src_all, dst_all, dist_all = [], [], []
        chunk = min(256, n)
        for start in range(0, n, chunk):
            end = min(n, start + chunk)
            dist = torch.cdist(xyz[start:end], xyz)
            vals, idx = torch.topk(dist, k + 1, dim=1, largest=False)
            idx = idx[:, 1:]
            vals = vals[:, 1:]
            src = torch.arange(start, end, device=xyz.device).unsqueeze(1).expand_as(idx)
            src_all.append(src.reshape(-1))
            dst_all.append(idx.reshape(-1))
            dist_all.append(vals.reshape(-1))
        return torch.cat(src_all).long(), torch.cat(dst_all).long(), torch.cat(dist_all).detach()


def _axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    theta = torch.linalg.norm(axis_angle, dim=-1, keepdim=True).clamp_min(1e-8)
    axis = axis_angle / theta
    x, y, z = axis.unbind(dim=-1)
    zeros = torch.zeros_like(x)
    k = torch.stack([zeros, -z, y, z, zeros, -x, -y, x, zeros], dim=-1).reshape(axis_angle.shape[:-1] + (3, 3))
    eye = torch.eye(3, device=axis_angle.device, dtype=axis_angle.dtype)
    eye = eye.reshape((1,) * (axis_angle.ndim - 1) + (3, 3))
    sin_t = torch.sin(theta)[..., None]
    cos_t = torch.cos(theta)[..., None]
    return eye + sin_t * k + (1.0 - cos_t) * torch.matmul(k, k)


def _inverse_sigmoid(value: torch.Tensor) -> torch.Tensor:
    return torch.log(value / (1.0 - value))


