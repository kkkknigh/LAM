from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Optional

import torch
from einops import rearrange

try:
    from diff_gaussian_rasterization_wda import GaussianRasterizationSettings, GaussianRasterizer
except Exception:
    from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

from lam.models.rendering.gaussian_model import GaussianModel


@dataclass
class LocalCamera:
    world_view_transform: torch.Tensor
    full_proj_transform: torch.Tensor
    camera_center: torch.Tensor
    height: int
    width: int


def render_animate_gs_with_intrinsics(
    renderer,
    gs_model_list: list[GaussianModel],
    query_points: torch.Tensor,
    flame_data: Dict[str, torch.Tensor],
    c2w: torch.Tensor,
    intrinsic: torch.Tensor,
    height: int,
    width: int,
    background_color: Optional[torch.Tensor],
    debug: bool = False,
) -> Dict[str, torch.Tensor]:
    batch_size = len(gs_model_list)
    out_list = []
    for b in range(batch_size):
        gs_model = gs_model_list[b]
        query_pt = query_points[b]
        animatable = renderer.animate_gs_model(
            gs_model,
            query_pt,
            renderer.get_sing_batch_smpl_data(flame_data, b),
            debug=debug,
        )
        if len(animatable) != c2w.shape[1]:
            raise ValueError(f"Animated GS views {len(animatable)} do not match cameras {c2w.shape[1]}")
        out_list.append(_render_single_batch(
            renderer,
            animatable,
            c2w[b],
            intrinsic[b],
            height,
            width,
            background_color[b] if background_color is not None else None,
        ))

    out = defaultdict(list)
    for item in out_list:
        for key, value in item.items():
            out[key].append(value)
    for key, value in out.items():
        if isinstance(value[0], torch.Tensor):
            out[key] = torch.stack(value, dim=0)
        else:
            out[key] = value
    for key in ["comp_rgb", "comp_mask", "comp_depth"]:
        out[key] = rearrange(out[key], "b v h w c -> b v c h w")
    return out


def _render_single_batch(
    renderer,
    gs_list: list[GaussianModel],
    c2ws: torch.Tensor,
    intrinsics: torch.Tensor,
    height: int,
    width: int,
    background_color: Optional[torch.Tensor],
) -> Dict[str, torch.Tensor]:
    out_list = []
    renderer.device = gs_list[0].xyz.device
    for v_idx, (c2w, intrinsic) in enumerate(zip(c2ws, intrinsics)):
        bg = background_color[v_idx] if background_color is not None else None
        out_list.append(_render_single_view(renderer, gs_list[v_idx], c2w, intrinsic, height, width, bg))

    out = defaultdict(list)
    for item in out_list:
        for key, value in item.items():
            out[key].append(value)
    out = {key: torch.stack(value, dim=0) for key, value in out.items()}
    out["3dgs"] = gs_list
    return out


def _render_single_view(
    renderer,
    gs: GaussianModel,
    c2w: torch.Tensor,
    intrinsic: torch.Tensor,
    height: int,
    width: int,
    background_color: Optional[torch.Tensor],
) -> Dict[str, torch.Tensor]:
    camera = _camera_from_c2w(c2w, intrinsic, height, width)
    screenspace_points = torch.zeros_like(gs.xyz, dtype=gs.xyz.dtype, requires_grad=True, device=gs.xyz.device) + 0
    try:
        screenspace_points.retain_grad()
    except Exception:
        pass

    fx = intrinsic[0, 0].clamp_min(1e-6)
    fy = intrinsic[1, 1].clamp_min(1e-6)
    tanfovx = float((float(width) / (2.0 * fx)).detach().cpu())
    tanfovy = float((float(height) / (2.0 * fy)).detach().cpu())
    raster_settings = GaussianRasterizationSettings(
        image_height=int(height),
        image_width=int(width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=background_color,
        scale_modifier=renderer.scaling_modifier,
        viewmatrix=camera.world_view_transform,
        projmatrix=camera.full_proj_transform.float(),
        sh_degree=renderer.sh_degree,
        campos=camera.camera_center,
        prefiltered=False,
        debug=False,
    )
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    colors_precomp = None
    shs = None
    if renderer.gs_net.use_rgb:
        colors_precomp = gs.shs.squeeze(1)
    else:
        shs = gs.shs

    with torch.autocast(device_type=gs.xyz.device.type, dtype=torch.float32):
        rendered_image, _radii, rendered_depth, rendered_alpha = rasterizer(
            means3D=gs.xyz.float(),
            means2D=screenspace_points.float(),
            shs=shs.float() if shs is not None else None,
            colors_precomp=colors_precomp.float() if colors_precomp is not None else None,
            opacities=gs.opacity.float(),
            scales=gs.scaling.float(),
            rotations=gs.rotation.float(),
            cov3D_precomp=None,
        )
    return {
        "comp_rgb": rendered_image.permute(1, 2, 0),
        "comp_rgb_bg": background_color,
        "comp_mask": rendered_alpha.permute(1, 2, 0),
        "comp_depth": rendered_depth.permute(1, 2, 0),
    }


def _camera_from_c2w(c2w: torch.Tensor, intrinsic: torch.Tensor, height: int, width: int) -> LocalCamera:
    w2c = torch.inverse(c2w)
    world_view_transform = w2c.transpose(0, 1)
    projection = _projection_from_intrinsics(intrinsic, height, width).transpose(0, 1).to(w2c.device)
    full_proj_transform = world_view_transform.unsqueeze(0).bmm(projection.unsqueeze(0)).squeeze(0)
    camera_center = world_view_transform.inverse()[3, :3]
    return LocalCamera(world_view_transform, full_proj_transform, camera_center, int(height), int(width))


def _projection_from_intrinsics(
    intrinsic: torch.Tensor,
    height: int,
    width: int,
    znear: float = 0.01,
    zfar: float = 100.0,
) -> torch.Tensor:
    fx = intrinsic[0, 0].clamp_min(1e-6)
    fy = intrinsic[1, 1].clamp_min(1e-6)
    cx = intrinsic[0, 2]
    cy = intrinsic[1, 2]
    p = torch.zeros(4, 4, device=intrinsic.device, dtype=intrinsic.dtype)
    p[0, 0] = 2.0 * fx / float(width)
    p[1, 1] = 2.0 * fy / float(height)
    p[0, 2] = 2.0 * cx / float(width) - 1.0
    p[1, 2] = 2.0 * cy / float(height) - 1.0
    p[3, 2] = 1.0
    p[2, 2] = zfar / (zfar - znear)
    p[2, 3] = -(zfar * znear) / (zfar - znear)
    return p

