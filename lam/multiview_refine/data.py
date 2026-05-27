import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import torch
from PIL import Image

from .types import FLAME_KEYS, MultiViewBatch, MultiViewFrame, TensorDict


REQUIRED_FLAME_KEYS = {
    "expr",
    "rotation",
    "neck_pose",
    "jaw_pose",
    "eyes_pose",
    "translation",
    "betas",
}


def load_multiview_bundle(root: str | Path, transforms_name: str = "transforms_aligned.json", bg_color: float = 1.0) -> MultiViewBatch:
    frames = load_frames(root, transforms_name)
    images, masks = [], []
    image_size = None
    for frame in frames:
        image, mask = _load_rgb_mask(frame.image_path, frame.mask_path, bg_color)
        current_size = tuple(image.shape[-2:])
        if image_size is None:
            image_size = current_size
        elif current_size != image_size:
            raise ValueError(f"All images must have the same H/W. Got {current_size} for {frame.image_path}, expected {image_size}.")
        images.append(image)
        masks.append(mask)
    flame_params = _stack_flame_params(frames)
    _validate_flame_params(flame_params, len(frames), root=Path(root))
    return MultiViewBatch(
        images=torch.stack(images, dim=0).unsqueeze(0),
        masks=torch.stack(masks, dim=0).unsqueeze(0),
        c2ws=torch.stack([f.c2w for f in frames], dim=0).unsqueeze(0),
        intrs=torch.stack([f.intr for f in frames], dim=0).unsqueeze(0),
        bg_colors=torch.full((1, len(frames), 3), bg_color, dtype=torch.float32),
        flame_params=flame_params,
        frame_ids=[f.frame_id for f in frames],
        landmarks_2d=_stack_landmarks(frames),
        view_indices=torch.arange(len(frames), dtype=torch.long),
    )


def load_frames(root: str | Path, transforms_name: str = "transforms_aligned.json") -> List[MultiViewFrame]:
    root = Path(root)
    transforms_path = root / transforms_name
    if not transforms_path.exists():
        transforms_path = root / "transforms.json"
    if not transforms_path.exists():
        raise FileNotFoundError(f"No transforms found under {root}")
    db = json.loads(transforms_path.read_text(encoding="utf-8"))
    frames = []
    for idx, item in enumerate(sorted(db["frames"], key=lambda f: f.get("flame_param_path") or f.get("file_path") or f.get("image_name", ""))):
        image_value = item.get("file_path") or item.get("image_path") or item.get("image_name")
        if not image_value:
            raise KeyError(f"Frame {idx} is missing file_path/image_path/image_name")
        image_path = _resolve(root, image_value, image_dirs=["images"])
        mask_path = _resolve_optional(root, item.get("fg_mask_path") or item.get("mask_path"), image_path.stem, ["fg_masks", "masks"])
        if mask_path is None:
            raise FileNotFoundError(f"Missing mask for {image_path.name}; upload or generate masks first.")
        flame_path = _resolve_optional(root, item.get("flame_param_path"), image_path.stem, ["flame_param"])
        landmark_path = _resolve_optional(root, item.get("landmark_path"), image_path.stem, ["landmark2d"])
        flame_params = _load_flame(flame_path) if flame_path else {}
        if "betas" not in flame_params:
            flame_params["betas"] = _load_betas(root)
        landmarks = _load_landmarks(landmark_path)
        intr = _load_intr(item)
        frames.append(MultiViewFrame(
            frame_id=str(item.get("view_id", idx)),
            image_path=image_path,
            mask_path=mask_path,
            flame_param_path=flame_path,
            landmark_path=landmark_path,
            c2w=_load_c2w(item),
            intr=intr,
            flame_params=flame_params,
            landmarks_2d=landmarks,
        ))
    return frames


def write_lam_transforms_from_colmap(root: str | Path, colmap_json: str | Path, out_name: str = "transforms_colmap_raw.json") -> Path:
    root = Path(root)
    db = json.loads(Path(colmap_json).read_text(encoding="utf-8"))
    image_by_name = {p.name: p for p in sorted((root / "images").glob("*")) if p.suffix.lower() in {".png", ".jpg", ".jpeg"}}
    frames = []
    for idx, frame in enumerate(db["frames"]):
        name = Path(frame["image_name"]).name
        if name not in image_by_name:
            continue
        out = dict(frame)
        out["file_path"] = f"images/{name}"
        stem = image_by_name[name].stem
        mask = _resolve_optional(root, None, stem, ["fg_masks", "masks"])
        flame = _resolve_optional(root, None, stem, ["flame_param"])
        landmark = _resolve_optional(root, None, stem, ["landmark2d"])
        if mask:
            out["fg_mask_path"] = str(mask.relative_to(root)).replace("\\", "/")
        if flame:
            out["flame_param_path"] = str(flame.relative_to(root)).replace("\\", "/")
        if landmark:
            out["landmark_path"] = str(landmark.relative_to(root)).replace("\\", "/")
        out["timestep_index"] = idx
        out["camera_index"] = idx
        frames.append(out)
    out_path = root / out_name
    out_path.write_text(json.dumps({"frames": frames}, indent=2), encoding="utf-8")
    return out_path


def _load_c2w(frame: dict) -> torch.Tensor:
    c2w = np.array(frame["transform_matrix"], dtype=np.float32)
    c2w[:3, 1:3] *= -1
    return torch.from_numpy(c2w)


def _load_intr(frame: dict) -> torch.Tensor:
    intr = torch.eye(4, dtype=torch.float32)
    intr[0, 0] = float(frame["fl_x"])
    intr[1, 1] = float(frame["fl_y"])
    intr[0, 2] = float(frame["cx"])
    intr[1, 2] = float(frame["cy"])
    return intr


def _load_rgb_mask(image_path: Path, mask_path: Optional[Path], bg_color: float) -> tuple[torch.Tensor, torch.Tensor]:
    image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.float32) / 255.0
    if mask_path is None:
        raise FileNotFoundError(f"Missing mask for {image_path}")
    mask = np.asarray(Image.open(mask_path).convert("L"), dtype=np.float32) / 255.0
    if mask.shape[:2] != image.shape[:2]:
        raise ValueError(f"Mask size {mask.shape[:2]} does not match image size {image.shape[:2]} for {image_path.name}")
    image = image * mask[..., None] + bg_color * (1.0 - mask[..., None])
    return torch.from_numpy(image).permute(2, 0, 1), torch.from_numpy(mask).unsqueeze(0)


def _load_flame(path: Path) -> TensorDict:
    raw = np.load(path, allow_pickle=True)
    out = {}
    for key in FLAME_KEYS:
        if key in raw:
            value = torch.as_tensor(raw[key], dtype=torch.float32)
            while value.ndim > 1 and value.shape[0] == 1:
                value = value[0]
            out[key] = value
    if "betas" not in out and "shape" in raw:
        out["betas"] = torch.as_tensor(raw["shape"], dtype=torch.float32).reshape(-1)
    return out


def _load_betas(root: Path) -> torch.Tensor:
    path = root / "canonical_flame_param.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing canonical_flame_param.npz under {root}")
    raw = np.load(path, allow_pickle=True)
    if "shape" in raw:
        return torch.as_tensor(raw["shape"], dtype=torch.float32)
    if "betas" in raw:
        return torch.as_tensor(raw["betas"], dtype=torch.float32)
    raise KeyError(f"{path} must contain 'shape' or 'betas'")


def _load_landmarks(path: Optional[Path]) -> Optional[torch.Tensor]:
    if path is None:
        return None
    raw = np.load(path, allow_pickle=True)
    if "face_landmark_2d" not in raw:
        return None
    lm = torch.as_tensor(raw["face_landmark_2d"], dtype=torch.float32)
    while lm.ndim > 2 and lm.shape[0] == 1:
        lm = lm[0]
    return lm


def _stack_flame_params(frames: Iterable[MultiViewFrame]) -> TensorDict:
    merged = defaultdict(list)
    betas = None
    for frame in frames:
        for key, value in frame.flame_params.items():
            if key == "betas":
                betas = value
            else:
                merged[key].append(value)
    out = {key: torch.stack(values, dim=0).unsqueeze(0) for key, values in merged.items()}
    if betas is not None:
        out["betas"] = betas.reshape(1, -1)
    return out


def _validate_flame_params(flame_params: TensorDict, num_frames: int, root: Path) -> None:
    missing = sorted(REQUIRED_FLAME_KEYS - set(flame_params))
    if missing:
        raise KeyError(f"Missing FLAME params {missing} under {root}. Run FLAME tracking or provide flame_param/canonical_flame_param files.")
    for key, value in flame_params.items():
        if key == "betas":
            if value.ndim != 2 or value.shape[0] != 1:
                raise ValueError(f"FLAME betas must have shape [1, D], got {tuple(value.shape)}")
            continue
        if value.ndim < 3 or value.shape[0] != 1 or value.shape[1] != num_frames:
            raise ValueError(f"FLAME param '{key}' must have shape [1, {num_frames}, ...], got {tuple(value.shape)}")


def _stack_landmarks(frames: List[MultiViewFrame]) -> Optional[torch.Tensor]:
    if any(frame.landmarks_2d is None for frame in frames):
        return None
    return torch.stack([frame.landmarks_2d for frame in frames], dim=0).unsqueeze(0)


def _resolve(root: Path, value: str, image_dirs: List[str]) -> Path:
    path = Path(value)
    if path.is_absolute() and path.exists():
        return path
    direct = root / path
    if direct.exists():
        return direct
    for dirname in image_dirs:
        candidate = root / dirname / path.name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not resolve {value} under {root}")


def _resolve_optional(root: Path, value: Optional[str], stem: str, dirs: List[str]) -> Optional[Path]:
    if value:
        path = Path(value)
        if not path.is_absolute():
            path = root / path
        if path.exists():
            return path
    for dirname in dirs:
        base = root / dirname
        for suffix in [".npz", ".png", ".jpg", ".jpeg"]:
            candidate = base / f"{stem}{suffix}"
            if candidate.exists():
                return candidate
    return None
