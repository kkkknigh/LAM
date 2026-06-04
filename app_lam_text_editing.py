# Copyright (c) 2024-2025, The Alibaba 3DAIGC Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import base64
import hashlib
import json
import os
import tempfile
import time
import zipfile
from pathlib import Path

import gradio as gr
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

from text_edit.flame_text_editor import apply_text_edit


_LAYER2_PACKAGE_CACHE = {}
_LAYER2_RENDER_CACHE = {}
LAYER2_RENDER_SIZE_DEFAULT = 1024
LAYER2_RENDER_SIZE_MAX = 1536


def get_image_base64(path):
    with open(path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode()
    return f"data:image/png;base64,{encoded}"


def save_images2video(img_lst, v_pth, fps):
    from moviepy.editor import ImageSequenceClip

    images = [image.astype(np.uint8) for image in img_lst]
    clip = ImageSequenceClip(images, fps=int(fps))
    clip.write_videofile(
        v_pth,
        codec="libx264",
        audio=False,
        preset="slow",
        ffmpeg_params=["-crf", "16", "-pix_fmt", "yuv420p"],
    )
    print(f"Video saved successfully at {v_pth}")


def _safe_stem(name):
    stem = Path(str(name)).stem
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in stem)
    return safe or "layer2_text_render"


def _uploaded_package_path(package_file):
    if package_file is None:
        raise gr.Error("Upload the Layer 1/2 package.zip first.")

    if isinstance(package_file, (str, Path)):
        raw_path = package_file
    elif isinstance(package_file, dict):
        raw_path = package_file.get("path") or package_file.get("name")
    elif hasattr(package_file, "path"):
        raw_path = package_file.path
    elif hasattr(package_file, "name"):
        raw_path = package_file.name
    else:
        raise gr.Error("Unsupported upload object. Please upload package.zip again.")

    package_path = Path(str(raw_path)).expanduser()
    if not package_path.exists():
        raise gr.Error(f"Uploaded package does not exist: {package_path}")
    if package_path.is_dir():
        raise gr.Error("Upload Layer 1/2 package.zip, not a workspace directory.")
    if package_path.suffix.lower() != ".zip":
        raise gr.Error("Layer 1/2 input must be package.zip.")
    return package_path


def _safe_extract_zip(zip_path, target_root):
    target_root = Path(target_root)
    target_root.mkdir(parents=True, exist_ok=True)
    target_resolved = target_root.resolve()

    with zipfile.ZipFile(zip_path, "r") as zipf:
        for member in zipf.infolist():
            member_name = member.filename.replace("\\", "/")
            member_path = (target_root / member_name).resolve()
            if os.path.commonpath([str(target_resolved), str(member_path)]) != str(target_resolved):
                raise gr.Error(f"Unsafe path in ZIP package: {member.filename}")
        zipf.extractall(target_root)


def _extracted_layer2_package_root(package_file):
    package_path = _uploaded_package_path(package_file)
    stat = package_path.stat()
    package_key = f"{package_path.resolve()}::{stat.st_mtime_ns}::{stat.st_size}"
    cache_key = hashlib.sha1(package_key.encode("utf-8")).hexdigest()

    cached_root = _LAYER2_PACKAGE_CACHE.get(cache_key)
    if cached_root is not None and Path(cached_root).exists():
        return Path(cached_root), package_path

    cache_root = Path(tempfile.gettempdir()) / "lam_layer2_text_packages"
    extract_root = cache_root / f"{_safe_stem(package_path.name)}_{cache_key[:12]}"
    if not extract_root.exists() or not any(extract_root.iterdir()):
        _safe_extract_zip(package_path, extract_root)

    _LAYER2_PACKAGE_CACHE.clear()
    _LAYER2_PACKAGE_CACHE[cache_key] = extract_root
    return extract_root, package_path


def _resolve_layer2_package(package_file):
    root, package_path = _extracted_layer2_package_root(package_file)
    refined_ply_candidates = sorted(root.rglob("refined_gaussian.ply"))
    canonical_ply_candidates = sorted([
        path for path in root.rglob("*.ply")
        if "canonical" in path.stem.lower() and "offset" not in path.stem.lower()
    ])
    init_ply_candidates = sorted([path for path in root.rglob("*.ply") if path.name.lower() == "init.ply"])
    flame_candidates = sorted(root.rglob("canonical_flame_param.npz"))
    frame_flame_candidates = sorted(root.rglob("flame_param/*.npz"))
    transforms_candidates = sorted(root.rglob("transforms_aligned.json"))
    if not transforms_candidates:
        transforms_candidates = sorted(root.rglob("layer1_reference_transforms.json"))
    metadata_candidates = sorted(root.rglob("layer1_metadata.json"))

    package_kind = "layer2"
    ply_candidates = refined_ply_candidates
    if not ply_candidates:
        package_kind = "layer1"
        ply_candidates = canonical_ply_candidates or init_ply_candidates
    if not ply_candidates:
        raise gr.Error("Package is missing refined_gaussian.ply or a Layer 1 canonical .ply.")
    if not flame_candidates:
        raise gr.Error("Package is missing canonical_flame_param.npz.")

    def prefer_short(paths):
        return sorted(paths, key=lambda p: (len(p.parts), str(p)))[0]

    return {
        "root": root,
        "package_path": package_path,
        "ply_path": prefer_short(ply_candidates),
        "canonical_flame_path": prefer_short(flame_candidates),
        "frame_flame_path": prefer_short(frame_flame_candidates) if frame_flame_candidates else None,
        "transforms_path": prefer_short(transforms_candidates) if transforms_candidates else None,
        "metadata_path": prefer_short(metadata_candidates) if metadata_candidates else None,
        "package_kind": package_kind,
    }


def _load_renderable_gaussian_for_app(ply_path):
    from lam.models.rendering.gaussian_model import GaussianModel

    gs = GaussianModel(ply_path=str(ply_path), sh2rgb=False)
    gs.opacity = torch.sigmoid(gs.opacity)
    gs.scaling = torch.exp(gs.scaling)
    gs.to_cuda()
    return gs


def _load_layer2_shape(canonical_flame_path):
    with np.load(canonical_flame_path, allow_pickle=True) as raw:
        if "shape" in raw:
            shape = raw["shape"]
        elif "betas" in raw:
            shape = raw["betas"]
        else:
            raise gr.Error(f"{canonical_flame_path} must contain shape or betas.")
    return torch.as_tensor(shape, dtype=torch.float32).reshape(1, -1)


def _load_base_flame_params(canonical_flame_path, frame_flame_path=None):
    betas = _load_layer2_shape(canonical_flame_path)
    expr_dim = 100
    device = betas.device
    base = {
        "expr": torch.zeros(1, 1, expr_dim, dtype=torch.float32, device=device),
        "rotation": torch.zeros(1, 1, 3, dtype=torch.float32, device=device),
        "neck_pose": torch.zeros(1, 1, 3, dtype=torch.float32, device=device),
        "jaw_pose": torch.zeros(1, 1, 3, dtype=torch.float32, device=device),
        "eyes_pose": torch.zeros(1, 1, 6, dtype=torch.float32, device=device),
        "translation": torch.zeros(1, 1, 3, dtype=torch.float32, device=device),
        "betas": betas,
    }
    if frame_flame_path is None:
        return base

    with np.load(frame_flame_path, allow_pickle=True) as raw:
        for key in ["expr", "rotation", "neck_pose", "jaw_pose", "eyes_pose", "translation"]:
            if key not in raw:
                continue
            value = torch.as_tensor(raw[key], dtype=torch.float32)
            while value.ndim > 1 and value.shape[0] == 1:
                value = value[0]
            base[key] = value.reshape(1, 1, -1)
    return base


def _clone_flame_params(flame_params):
    return {key: value.clone() if torch.is_tensor(value) else value for key, value in flame_params.items()}


def _repeat_flame_params(flame_params, num_frames):
    repeated = {}
    for key, value in flame_params.items():
        if key == "betas":
            repeated[key] = value.clone()
        elif torch.is_tensor(value):
            repeated[key] = value[:, :1].repeat(1, num_frames, *([1] * (value.ndim - 2)))
        else:
            repeated[key] = value
    return repeated


def _build_text_motion_flame(base_flame_params, prompt, strength, num_frames):
    neutral = _repeat_flame_params(base_flame_params, num_frames)
    target = _clone_flame_params(neutral)
    prompt = str(prompt or "")
    if prompt.strip():
        base_betas = target["betas"].clone()
        target = apply_text_edit(target, prompt, strength=float(strength), verbose=False)
        target["betas"] = base_betas

    if num_frames <= 1:
        return target

    device = target["expr"].device
    dtype = target["expr"].dtype
    t = torch.linspace(0.0, 1.0, num_frames, device=device, dtype=dtype).reshape(1, num_frames, 1)
    ease = torch.sin(t * np.pi).clamp_min(0.0)
    speaking = bool(prompt.strip())
    syllable_count = max(1, min(12, len(prompt.replace(" ", "")) // 2)) if speaking else 1
    mouth_pulse = (0.5 - 0.5 * torch.cos(t * float(syllable_count) * 2.0 * np.pi)).pow(1.5)
    mouth_pulse = mouth_pulse * float(strength) * 0.18

    out = _clone_flame_params(neutral)
    for key, value in target.items():
        if key == "betas" or not torch.is_tensor(value):
            out[key] = value
            continue
        out[key] = neutral[key] + (value - neutral[key]) * ease
    if speaking and out["jaw_pose"].shape[-1] > 0:
        out["jaw_pose"][..., 0] = (out["jaw_pose"][..., 0] + mouth_pulse[..., 0]).clamp(-0.8, 0.8)
    if speaking and out["expr"].shape[-1] > 3:
        out["expr"][..., 3] = (out["expr"][..., 3] + mouth_pulse[..., 0] * 0.8).clamp(-3.0, 3.0)
    return out


def _look_at_torch(cam_pos, target, up_world):
    forward = target - cam_pos
    forward = forward / torch.linalg.norm(forward).clamp_min(1e-6)
    down = -up_world
    down = down - forward * torch.dot(down, forward)
    down = down / torch.linalg.norm(down).clamp_min(1e-6)
    right = torch.cross(down, forward, dim=0)
    right = right / torch.linalg.norm(right).clamp_min(1e-6)
    down = torch.cross(forward, right, dim=0)
    down = down / torch.linalg.norm(down).clamp_min(1e-6)

    c2w = torch.eye(4, device=cam_pos.device, dtype=torch.float32)
    c2w[:3, 0] = right
    c2w[:3, 1] = down
    c2w[:3, 2] = forward
    c2w[:3, 3] = cam_pos
    return c2w


def _build_front_camera_from_gs(gs, size=512, yaw_degrees=0.0, distance_scale=2.8):
    xyz = gs.xyz.detach().float()
    center = xyz.mean(dim=0)
    radius = torch.quantile(torch.linalg.norm(xyz - center, dim=1), 0.95).clamp_min(0.25)
    yaw = float(yaw_degrees) * np.pi / 180.0
    direction = torch.tensor([np.sin(yaw), 0.0, np.cos(yaw)], device=xyz.device, dtype=torch.float32)
    direction = direction / torch.linalg.norm(direction)
    up = torch.tensor([0.0, 1.0, 0.0], device=xyz.device, dtype=torch.float32)
    cam_pos = center + direction * radius * float(distance_scale)
    c2w = _look_at_torch(cam_pos, center, up).reshape(1, 1, 4, 4)

    intr = torch.eye(4, device=xyz.device, dtype=torch.float32)
    focal = float(size) * 1.35
    intr[0, 0] = focal
    intr[1, 1] = focal
    intr[0, 2] = float(size) / 2.0
    intr[1, 2] = float(size) / 2.0
    return c2w, intr.reshape(1, 1, 4, 4)


def _json_c2w_to_render_c2w(c2w):
    out = np.asarray(c2w, dtype=np.float32).copy()
    out[:3, 1:3] *= -1.0
    return out


def _square_preview_intrinsics(frame, size):
    width = float(frame.get("w") or frame.get("width") or float(frame["cx"]) * 2.0)
    height = float(frame.get("h") or frame.get("height") or float(frame["cy"]) * 2.0)
    if width <= 0.0 or height <= 0.0:
        raise ValueError("Invalid camera image size in transforms_aligned.json.")

    scale = float(size) / min(width, height)
    scaled_width = width * scale
    scaled_height = height * scale
    crop_x = max(0.0, (scaled_width - float(size)) * 0.5)
    crop_y = max(0.0, (scaled_height - float(size)) * 0.5)

    intr = np.eye(4, dtype=np.float32)
    intr[0, 0] = float(frame["fl_x"]) * scale
    intr[1, 1] = float(frame["fl_y"]) * scale
    intr[0, 2] = float(frame["cx"]) * scale - crop_x
    intr[1, 2] = float(frame["cy"]) * scale - crop_y
    return intr


def _rotation_matrix_around_axis(axis, angle):
    axis = axis / torch.linalg.norm(axis).clamp_min(1e-6)
    x, y, z = axis
    zeros = torch.zeros((), device=axis.device, dtype=axis.dtype)
    k = torch.stack([
        torch.stack([zeros, -z, y]),
        torch.stack([z, zeros, -x]),
        torch.stack([-y, x, zeros]),
    ])
    eye = torch.eye(3, device=axis.device, dtype=axis.dtype)
    return eye + torch.sin(angle) * k + (1.0 - torch.cos(angle)) * (k @ k)


def _orbit_camera_yaw(c2w, center, yaw_degrees):
    yaw = float(yaw_degrees) * np.pi / 180.0
    if abs(yaw) < 1e-6:
        return c2w

    up = -c2w[:3, 1].detach().float()
    if torch.linalg.norm(up) < 1e-6:
        up = torch.tensor([0.0, 0.0, 1.0], device=c2w.device, dtype=torch.float32)
    up = up / torch.linalg.norm(up).clamp_min(1e-6)
    rot = _rotation_matrix_around_axis(up, torch.tensor(yaw, device=c2w.device, dtype=torch.float32))
    cam_pos = center + rot @ (c2w[:3, 3].detach().float() - center)
    return _look_at_torch(cam_pos, center, up)


def _build_package_camera(source, gs, size, yaw_degrees):
    transforms_path = source.get("transforms_path")
    frames = []
    if transforms_path is not None:
        transforms_path = Path(transforms_path)
        if transforms_path.exists():
            with open(transforms_path, "r", encoding="utf-8") as fp:
                db = json.load(fp)
            frames = sorted(
                db.get("frames", []),
                key=lambda frame: str(frame.get("flame_param_path") or frame.get("file_path") or frame.get("image_name", "")),
            )
    metadata_path = source.get("metadata_path")
    if not frames and metadata_path is not None and Path(metadata_path).exists():
        with open(metadata_path, "r", encoding="utf-8") as fp:
            metadata = json.load(fp)
        reference_frame = metadata.get("reference_camera")
        if reference_frame is not None:
            frames = [reference_frame]
    if not frames:
        return None

    frame = frames[0]
    c2w_np = _json_c2w_to_render_c2w(frame["transform_matrix"])
    if c2w_np.shape != (4, 4):
        return None

    device = gs.xyz.device
    c2w = torch.as_tensor(c2w_np, device=device, dtype=torch.float32)
    center = gs.xyz.detach().float().mean(dim=0)
    c2w = _orbit_camera_yaw(c2w, center, yaw_degrees)
    intr = torch.as_tensor(_square_preview_intrinsics(frame, size), device=device, dtype=torch.float32)
    frame_name = frame.get("image_name") or frame.get("file_path") or "first frame"
    return c2w.reshape(1, 1, 4, 4), intr.reshape(1, 1, 4, 4), f"package camera ({frame_name})"


def _build_layer2_preview_camera(context, size, yaw_degrees):
    package_camera = _build_package_camera(context["source"], context["gs"], size, yaw_degrees)
    if package_camera is not None:
        return package_camera
    c2w, intr = _build_front_camera_from_gs(context["gs"], size=size, yaw_degrees=float(yaw_degrees))
    return c2w, intr, "estimated point-cloud camera"


def _load_layer2_render_context(package_file, lam_model):
    resolved = _resolve_layer2_package(package_file)
    cache_key = "|".join([
        str(resolved["package_path"].resolve()),
        str(resolved["ply_path"].resolve()),
        str(resolved["canonical_flame_path"].resolve()),
        str(resolved["frame_flame_path"].resolve()) if resolved["frame_flame_path"] else "",
    ])
    cached = _LAYER2_RENDER_CACHE.get(cache_key)
    if cached is not None:
        return cached

    gs = _load_renderable_gaussian_for_app(resolved["ply_path"])
    flame_params = _load_base_flame_params(resolved["canonical_flame_path"], resolved["frame_flame_path"])
    flame_params = {key: value.cuda() if torch.is_tensor(value) else value for key, value in flame_params.items()}
    with torch.no_grad():
        query_points, _ = lam_model.renderer.get_query_points(flame_params, device=gs.xyz.device)
    context = {
        "gs": gs,
        "renderer": lam_model.renderer,
        "query_points": query_points,
        "base_flame_params": flame_params,
        "source": resolved,
    }
    _LAYER2_RENDER_CACHE.clear()
    _LAYER2_RENDER_CACHE[cache_key] = context
    return context


def _render_frames(context, flame_params, c2ws, intrs, size, chunk_size=12):
    from multiview_refine.render_adapter import render_animate_gs_with_intrinsics

    num_frames = c2ws.shape[1]
    bg = torch.ones((1, num_frames, 3), device=context["gs"].xyz.device, dtype=torch.float32)
    frames = []
    for start in range(0, num_frames, chunk_size):
        end = min(start + chunk_size, num_frames)
        flame_chunk = {}
        for key, value in flame_params.items():
            flame_chunk[key] = value if key == "betas" or not torch.is_tensor(value) else value[:, start:end]
        with torch.no_grad():
            out = render_animate_gs_with_intrinsics(
                context["renderer"],
                [context["gs"]],
                context["query_points"],
                flame_chunk,
                c2ws[:, start:end],
                intrs[:, start:end],
                size,
                size,
                bg[:, start:end],
            )
        rgb = out["comp_rgb"][0].detach().clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()
        mask = out["comp_mask"][0].detach().clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()
        images = rgb * mask + (1.0 - mask) * 1.0
        frames.extend((np.clip(images, 0.0, 1.0) * 255.0).astype(np.uint8))
    return frames


def _layer2_video_chunk_size(size):
    size = int(size)
    if size >= 1536:
        return 2
    if size >= 1024:
        return 4
    return 12


def _status_text(context, prompt, camera_source=None):
    source = context["source"]
    text = (
        f"Package: {source['package_path']}\n"
        f"Package Type: {source.get('package_kind', 'unknown')}\n"
        f"Gaussian: {source['ply_path']}\n"
        f"FLAME: {source['canonical_flame_path']}\n"
        f"Prompt: {str(prompt or '').strip() or '(neutral)'}"
    )
    if camera_source is not None:
        text += f"\nCamera: {camera_source}"
    return text


def render_layer2_text_front(package_file, dialogue_prompt, edit_strength, yaw_degrees, render_size):
    if "lam" not in globals():
        raise gr.Error("LAM model is not initialized.")

    size = int(render_size)
    context = _load_layer2_render_context(package_file, lam)
    flame_params = _build_text_motion_flame(
        context["base_flame_params"],
        dialogue_prompt,
        float(edit_strength),
        num_frames=1,
    )
    flame_params = {key: value.cuda() if torch.is_tensor(value) else value for key, value in flame_params.items()}
    c2w, intr, camera_source = _build_layer2_preview_camera(context, size, float(yaw_degrees))
    image = _render_frames(context, flame_params, c2w, intr, size, chunk_size=1)[0]

    out_dir = Path("output") / "layer2_text_front"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{_safe_stem(context['source']['package_path'].name)}_{int(time.time() * 1000)}.png"
    Image.fromarray(image).save(out_path)
    status = f"{_status_text(context, dialogue_prompt, camera_source)}\nOutput: {size}x{size}"
    return str(out_path), status


def render_layer2_text_front_live(package_file, dialogue_prompt, edit_strength, yaw_degrees, render_size):
    if package_file is None:
        return gr.update(), "Upload Layer 1/2 package.zip first."
    return render_layer2_text_front(package_file, dialogue_prompt, edit_strength, yaw_degrees, render_size)


def render_layer2_text_front_video(
    package_file,
    dialogue_prompt,
    edit_strength,
    yaw_degrees,
    render_size,
    duration_sec,
    fps,
):
    if "lam" not in globals():
        raise gr.Error("LAM model is not initialized.")

    size = int(render_size)
    fps = int(fps)
    num_frames = max(2, min(120, int(round(float(duration_sec) * float(fps)))))
    context = _load_layer2_render_context(package_file, lam)
    flame_params = _build_text_motion_flame(
        context["base_flame_params"],
        dialogue_prompt,
        float(edit_strength),
        num_frames,
    )
    flame_params = {key: value.cuda() if torch.is_tensor(value) else value for key, value in flame_params.items()}
    c2w_one, intr_one, camera_source = _build_layer2_preview_camera(context, size, float(yaw_degrees))
    frames = _render_frames(
        context,
        flame_params,
        c2w_one.repeat(1, num_frames, 1, 1),
        intr_one.repeat(1, num_frames, 1, 1),
        size,
        chunk_size=_layer2_video_chunk_size(size),
    )

    out_dir = Path("output") / "layer2_text_front"
    out_dir.mkdir(parents=True, exist_ok=True)
    video_path = out_dir / f"{_safe_stem(context['source']['package_path'].name)}_{int(time.time() * 1000)}.mp4"
    save_images2video(frames, str(video_path), fps)
    status = f"{_status_text(context, dialogue_prompt, camera_source)}\nOutput: {size}x{size}\nFrames: {num_frames} @ {fps} fps"
    return str(video_path), status


def parse_configs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--infer", type=str)
    parser.add_argument("--blender_path", type=str, default="blender", help="Path to Blender executable")
    args, unknown = parser.parse_known_args()

    cfg = OmegaConf.create({"blender_path": args.blender_path})
    cli_cfg = OmegaConf.from_cli(unknown)

    if os.environ.get("APP_INFER") is not None:
        args.infer = os.environ.get("APP_INFER")
    if os.environ.get("APP_MODEL_NAME") is not None:
        cli_cfg.model_name = os.environ.get("APP_MODEL_NAME")

    args.config = args.infer if args.config is None else args.config
    if args.config is not None:
        cfg_train = OmegaConf.load(args.config)
        cfg.source_size = cfg_train.dataset.source_image_res
        cfg.src_head_size = cfg_train.dataset.get("src_head_size", 112)
        cfg.render_size = cfg_train.dataset.render_image.high

        relative_path = os.path.join(
            cfg_train.experiment.parent,
            cfg_train.experiment.child,
            os.path.basename(cli_cfg.model_name).split("_")[-1],
        )
        cfg.save_tmp_dump = os.path.join("exps", "save_tmp", relative_path)
        cfg.image_dump = os.path.join("exps", "images", relative_path)
        cfg.video_dump = os.path.join("exps", "videos", relative_path)

    if args.infer is not None:
        cfg_infer = OmegaConf.load(args.infer)
        cfg.merge_with(cfg_infer)
        cfg.setdefault("save_tmp_dump", os.path.join("exps", cli_cfg.model_name, "save_tmp"))
        cfg.setdefault("image_dump", os.path.join("exps", cli_cfg.model_name, "images"))
        cfg.setdefault("video_dump", os.path.join("dumps", cli_cfg.model_name, "videos"))
        cfg.setdefault("mesh_dump", os.path.join("dumps", cli_cfg.model_name, "meshes"))

    cfg.motion_video_read_fps = 30
    cfg.merge_with(cli_cfg)
    cfg.setdefault("logger", "INFO")

    assert cfg.model_name is not None, "model_name is required"
    return cfg


def demo_lam(lam_model, cfg):
    with gr.Blocks(analytics_enabled=False) as demo:
        logo_url = "./assets/images/logo.jpeg"
        logo_base64 = get_image_base64(logo_url) if os.path.exists(logo_url) else ""
        if logo_base64:
            gr.HTML(f"""
                <div style="display: flex; justify-content: center; align-items: center; text-align: center;">
                    <h1><img src="{logo_base64}" style="height:35px; display:inline-block;"/> LAM Layer 3 Text Expression Control</h1>
                </div>
            """)
        else:
            gr.Markdown("# LAM Layer 3 Text Expression Control")

        with gr.Row():
            with gr.Column(variant="panel", scale=1):
                layer2_package = gr.File(
                    label="Upload Layer 1/2 package.zip",
                    file_types=[".zip"],
                    type="file",
                )
                layer2_prompt_process_btn = gr.Button("Process Dialogue / 处理台词")
                layer2_dialogue_prompt = gr.Textbox(
                    label="Dialogue / Text Edit Prompt",
                    placeholder="Examples: smile, mouth open, blink, look left, frown",
                    value="",
                    lines=3,
                )
                with gr.Row():
                    layer2_strength = gr.Slider(
                        label="Expression Strength",
                        minimum=0.0,
                        maximum=1.0,
                        value=0.5,
                        step=0.05,
                    )
                    layer2_yaw = gr.Slider(
                        label="Front Yaw",
                        minimum=-45.0,
                        maximum=45.0,
                        value=0.0,
                        step=1.0,
                    )
                    layer2_render_size = gr.Slider(
                        label="Render Size",
                        minimum=256,
                        maximum=LAYER2_RENDER_SIZE_MAX,
                        value=LAYER2_RENDER_SIZE_DEFAULT,
                        step=64,
                    )
                with gr.Row():
                    layer2_duration = gr.Slider(
                        label="Video Seconds",
                        minimum=0.5,
                        maximum=5.0,
                        value=2.0,
                        step=0.5,
                    )
                    layer2_fps = gr.Slider(
                        label="Video FPS",
                        minimum=6,
                        maximum=30,
                        value=12,
                        step=1,
                    )
                with gr.Row():
                    layer2_render_btn = gr.Button("Render Front Preview", variant="primary")
                    layer2_video_btn = gr.Button("Render Dynamic Video")
                layer2_status = gr.Textbox(label="Status", interactive=False, lines=5)

            with gr.Column(variant="panel", scale=1):
                layer2_front_image = gr.Image(
                    label="Front Preview",
                    type="filepath",
                    height=640,
                    interactive=False,
                )
                layer2_front_video = gr.Video(
                    label="Dynamic Front Video",
                    format="mp4",
                    height=640,
                    autoplay=True,
                )

        preview_inputs = [layer2_package, layer2_dialogue_prompt, layer2_strength, layer2_yaw, layer2_render_size]
        video_inputs = preview_inputs + [layer2_duration, layer2_fps]
        layer2_render_btn.click(
            fn=render_layer2_text_front,
            inputs=preview_inputs,
            outputs=[layer2_front_image, layer2_status],
        )
        layer2_video_btn.click(
            fn=render_layer2_text_front_video,
            inputs=video_inputs,
            outputs=[layer2_front_video, layer2_status],
        )
        live_preview_kwargs = dict(
            fn=render_layer2_text_front_live,
            inputs=preview_inputs,
            outputs=[layer2_front_image, layer2_status],
            api_name=False,
            show_progress="minimal",
            queue=True,
        )
        layer2_package.change(**live_preview_kwargs)
        layer2_prompt_process_btn.click(**live_preview_kwargs)
        layer2_dialogue_prompt.submit(**live_preview_kwargs)
        layer2_dialogue_prompt.change(**live_preview_kwargs)
        layer2_strength.input(**live_preview_kwargs)
        layer2_yaw.input(**live_preview_kwargs)
        layer2_render_size.input(**live_preview_kwargs)

        demo.queue()
        demo.launch()


def _build_model(cfg):
    from lam.models import ModelLAM
    from safetensors.torch import load_file

    model = ModelLAM(**cfg.model)
    resume = os.path.join(cfg.model_name, "model.safetensors")
    print("=" * 100)
    print("loading pretrained weight from:", resume)
    if resume.endswith("safetensors"):
        ckpt = load_file(resume, device="cpu")
    else:
        ckpt = torch.load(resume, map_location="cpu")
    state_dict = model.state_dict()
    for key, value in ckpt.items():
        if key in state_dict:
            if state_dict[key].shape == value.shape:
                state_dict[key].copy_(value)
            else:
                print(f"WARN] mismatching shape for param {key}: ckpt {value.shape} != model {state_dict[key].shape}, ignored.")
        else:
            print(f"WARN] unexpected param {key}: {value.shape}")
    print("finish loading pretrained weight from:", resume)
    print("=" * 100)
    return model


def launch_gradio_app():
    global lam

    os.environ.update({
        "APP_ENABLED": "1",
        "APP_MODEL_NAME": "./model_zoo/lam_models/releases/lam/lam-20k/step_045500/",
        "APP_INFER": "./configs/inference/lam-20k-8gpu.yaml",
        "APP_TYPE": "infer.lam",
        "NUMBA_THREADING_LAYER": "omp",
    })

    cfg = parse_configs()
    lam = _build_model(cfg)
    lam.to("cuda")
    lam.eval()

    demo_lam(lam, cfg)


if __name__ == "__main__":
    launch_gradio_app()
