import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image, ImageDraw


def save_overlay_grid(
    out_dir: str | Path,
    frame_ids: List[str],
    target_rgb: torch.Tensor,
    target_mask: torch.Tensor,
    pred_rgb: torch.Tensor,
    pred_mask: torch.Tensor,
    landmarks_2d: Optional[torch.Tensor] = None,
    max_items: int = 8,
) -> List[str]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    num = min(len(frame_ids), max_items, target_rgb.shape[1])
    for i in range(num):
        target = _to_uint8(target_rgb[0, i])
        pred = _to_uint8(pred_rgb[0, i])
        tmask = _mask_to_rgb(target_mask[0, i])
        pmask = _mask_to_rgb(pred_mask[0, i])
        overlay = (target.astype(np.float32) * 0.55 + pred.astype(np.float32) * 0.45).clip(0, 255).astype(np.uint8)
        if landmarks_2d is not None:
            _draw_landmarks(overlay, landmarks_2d[0, i])
        canvas = np.concatenate([target, pred, tmask, pmask, overlay], axis=1)
        path = out_dir / f"{i:04d}_{frame_ids[i]}.png"
        Image.fromarray(canvas).save(path)
        saved.append(str(path))
    return saved


def append_jsonl(path: str | Path, row: Dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(row) + "\n")


def save_workspace_image_previews(root: str | Path, out_dir: str | Path, max_items: int = 16, bg_color: float = 1.0) -> List[str]:
    root = Path(root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    images = sorted([p for p in (root / "images").glob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"}])
    saved = []
    for idx, image_path in enumerate(images[:max_items]):
        stem = image_path.stem
        image = _load_image_uint8(image_path)
        mask_path = _find_by_stem(root, stem, ["fg_masks", "masks"], [".png", ".jpg", ".jpeg"])
        landmark_path = _find_by_stem(root, stem, ["landmark2d"], [".npz"])
        mask = _load_mask_uint8(mask_path, image.shape[:2]) if mask_path else np.full(image.shape[:2], 255, dtype=np.uint8)
        masked = (image.astype(np.float32) * (mask[..., None] / 255.0) + 255.0 * bg_color * (1.0 - mask[..., None] / 255.0)).clip(0, 255).astype(np.uint8)
        landmark = image.copy()
        landmarks = _load_landmarks_np(landmark_path)
        if landmarks is not None:
            _draw_landmarks_np(landmark, landmarks)
        mask_rgb = np.repeat(mask[..., None], 3, axis=-1)
        canvas = _labeled_row([
            ("image", image),
            ("mask", mask_rgb),
            ("masked", masked),
            ("landmarks", landmark),
        ])
        path = out_dir / f"{idx:04d}_{stem}.png"
        Image.fromarray(canvas).save(path)
        saved.append(str(path))
    return saved


def save_transform_camera_plot(
    out_path: str | Path,
    transform_paths: Dict[str, str | Path],
    title: str = "camera centers",
) -> str:
    tracks = {}
    for name, path in transform_paths.items():
        path = Path(path)
        if path.exists():
            tracks[name] = _load_centers_from_transforms(path)
    return save_camera_tracks_plot(out_path, tracks, title=title)


def save_camera_tracks_plot(out_path: str | Path, tracks: Dict[str, np.ndarray], title: str = "camera centers") -> str:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 1200, 560
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    image = Image.fromarray(canvas)
    draw = ImageDraw.Draw(image)
    draw.text((16, 12), title, fill=(20, 24, 32))

    valid_tracks = {k: np.asarray(v, dtype=np.float32) for k, v in tracks.items() if len(v) > 0}
    if not valid_tracks:
        draw.text((16, 44), "no camera centers found", fill=(160, 40, 40))
        image.save(out_path)
        return str(out_path)

    colors = [(40, 98, 220), (220, 80, 56), (38, 150, 92), (150, 90, 190), (230, 150, 40)]
    panels = [
        ("top x/z", (0, 2), (40, 64, 570, 520)),
        ("front x/y", (0, 1), (630, 64, 1160, 520)),
    ]
    all_points = np.concatenate(list(valid_tracks.values()), axis=0)
    for panel_name, dims, box in panels:
        draw.rectangle(box, outline=(215, 220, 226), width=1)
        draw.text((box[0] + 8, box[1] + 8), panel_name, fill=(60, 67, 80))
        pts2 = all_points[:, dims]
        mn = pts2.min(axis=0)
        mx = pts2.max(axis=0)
        span = np.maximum(mx - mn, 1e-5)
        pad = span * 0.1
        mn -= pad
        mx += pad
        span = np.maximum(mx - mn, 1e-5)
        for track_idx, (name, points) in enumerate(valid_tracks.items()):
            color = colors[track_idx % len(colors)]
            mapped = [_map_point(p[list(dims)], mn, span, box) for p in points]
            if len(mapped) > 1:
                draw.line(mapped, fill=color, width=2)
            for i, point in enumerate(mapped):
                r = 4 if i in {0, len(mapped) - 1} else 3
                draw.ellipse((point[0] - r, point[1] - r, point[0] + r, point[1] + r), fill=color)
        legend_y = box[1] + 28
        for track_idx, name in enumerate(valid_tracks):
            color = colors[track_idx % len(colors)]
            draw.rectangle((box[0] + 8, legend_y, box[0] + 22, legend_y + 10), fill=color)
            draw.text((box[0] + 28, legend_y - 2), name, fill=(60, 67, 80))
            legend_y += 18
    image.save(out_path)
    return str(out_path)


def save_loss_plot(jsonl_path: str | Path, out_path: str | Path) -> Optional[str]:
    jsonl_path = Path(jsonl_path)
    if not jsonl_path.exists():
        return None
    rows = []
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    if not rows:
        return None

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 1200, 520
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((16, 12), "loss history", fill=(20, 24, 32))
    box = (56, 64, 1160, 470)
    draw.rectangle(box, outline=(215, 220, 226), width=1)
    preferred_metrics = [
        "total",
        "rgb",
        "ssim",
        "mask",
        "mask_iou",
        "mask_boundary",
        "landmark",
        "sim3_reg",
        "camera_delta_reg",
        "intrinsics_reg",
        "exposure_reg",
        "expr_reg",
        "jaw_reg",
        "eyes_reg",
        "xyz_anchor",
        "offset_anchor",
        "scale_anchor",
        "opacity_reg",
        "knn_anchor",
        "scale_limit",
    ]
    row_keys = set().union(*(row.keys() for row in rows))
    metrics = [metric for metric in preferred_metrics if metric in row_keys]
    metrics += sorted(row_keys - set(metrics) - {"stage", "step", "global_step", "lr"})
    colors = [
        (20, 82, 204),
        (220, 80, 56),
        (38, 150, 92),
        (150, 90, 190),
        (230, 150, 40),
        (70, 150, 170),
        (105, 115, 135),
        (190, 65, 120),
    ]
    x_values = np.arange(len(rows), dtype=np.float32)
    x_span = max(float(len(rows) - 1), 1.0)
    for metric_idx, metric in enumerate(metrics):
        values = np.asarray([row.get(metric, np.nan) for row in rows], dtype=np.float32)
        valid = np.isfinite(values)
        if valid.sum() < 1:
            continue
        finite_values = values[valid]
        mn = float(finite_values.min())
        mx = float(finite_values.max())
        span = max(mx - mn, 1e-8)
        points = []
        for x, value, is_valid in zip(x_values, values, valid):
            if not is_valid:
                continue
            px = box[0] + 12 + int((x / x_span) * (box[2] - box[0] - 24))
            py = box[3] - 12 - int(((float(value) - mn) / span) * (box[3] - box[1] - 24))
            points.append((px, py))
        color = colors[metric_idx % len(colors)]
        if len(points) > 1:
            draw.line(points, fill=color, width=2)
        elif points:
            px, py = points[0]
            draw.ellipse((px - 3, py - 3, px + 3, py + 3), fill=color)
        row = metric_idx // 6
        col = metric_idx % 6
        lx = box[0] + 10 + col * 178
        ly = 482 + row * 18
        if ly + 14 < height:
            draw.rectangle((lx, ly + 4, lx + 14, ly + 14), fill=color)
            draw.text((lx + 20, ly), metric, fill=(60, 67, 80))
    image.save(out_path)
    return str(out_path)


def _to_uint8(tensor: torch.Tensor) -> np.ndarray:
    arr = tensor.detach().cpu()
    if arr.ndim == 3 and arr.shape[0] in {1, 3}:
        arr = arr.permute(1, 2, 0)
    arr = arr.numpy()
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    return (arr.clip(0, 1) * 255).astype(np.uint8)


def _load_image_uint8(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def _load_mask_uint8(path: Path, image_hw: tuple[int, int]) -> np.ndarray:
    mask = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
    if mask.shape[:2] != image_hw:
        mask = np.asarray(Image.fromarray(mask).resize((image_hw[1], image_hw[0]), Image.Resampling.NEAREST), dtype=np.uint8)
    return mask


def _find_by_stem(root: Path, stem: str, dirs: List[str], suffixes: List[str]) -> Optional[Path]:
    for dirname in dirs:
        base = root / dirname
        for suffix in suffixes:
            candidate = base / f"{stem}{suffix}"
            if candidate.exists():
                return candidate
    return None


def _load_landmarks_np(path: Optional[Path]) -> Optional[np.ndarray]:
    if path is None:
        return None
    raw = np.load(path, allow_pickle=True)
    if "face_landmark_2d" not in raw:
        return None
    lm = np.asarray(raw["face_landmark_2d"], dtype=np.float32)
    while lm.ndim > 2 and lm.shape[0] == 1:
        lm = lm[0]
    return lm


def _labeled_row(items: List[tuple[str, np.ndarray]], target_h: int = 192) -> np.ndarray:
    panels = []
    for label, arr in items:
        panel = _resize_to_height(arr, target_h)
        labeled = np.full((panel.shape[0] + 24, panel.shape[1], 3), 245, dtype=np.uint8)
        labeled[24:, :, :] = panel
        image = Image.fromarray(labeled)
        ImageDraw.Draw(image).text((6, 5), label, fill=(30, 36, 45))
        panels.append(np.asarray(image, dtype=np.uint8))
    return np.concatenate(panels, axis=1)


def _resize_to_height(arr: np.ndarray, target_h: int) -> np.ndarray:
    h, w = arr.shape[:2]
    if h == target_h:
        return arr
    target_w = max(1, int(round(w * target_h / max(h, 1))))
    return np.asarray(Image.fromarray(arr).resize((target_w, target_h), Image.Resampling.BILINEAR), dtype=np.uint8)


def _load_centers_from_transforms(path: Path) -> np.ndarray:
    db = json.loads(path.read_text(encoding="utf-8"))
    centers = []
    for frame in db.get("frames", []):
        if "transform_matrix" not in frame:
            continue
        mat = np.asarray(frame["transform_matrix"], dtype=np.float32)
        if mat.shape == (4, 4):
            centers.append(mat[:3, 3])
    if not centers:
        return np.zeros((0, 3), dtype=np.float32)
    return np.stack(centers).astype(np.float32)


def _map_point(point: np.ndarray, mn: np.ndarray, span: np.ndarray, box: tuple[int, int, int, int]) -> tuple[int, int]:
    x0, y0, x1, y1 = box
    px = x0 + 16 + int(((float(point[0]) - float(mn[0])) / float(span[0])) * (x1 - x0 - 32))
    py = y1 - 16 - int(((float(point[1]) - float(mn[1])) / float(span[1])) * (y1 - y0 - 32))
    return px, py


def _mask_to_rgb(tensor: torch.Tensor) -> np.ndarray:
    arr = _to_uint8(tensor)
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    return arr


def _draw_landmarks(image: np.ndarray, landmarks: torch.Tensor) -> None:
    lm = landmarks.detach().cpu().numpy()
    _draw_landmarks_np(image, lm)


def _draw_landmarks_np(image: np.ndarray, landmarks: np.ndarray) -> None:
    lm = np.asarray(landmarks, dtype=np.float32)
    h, w = image.shape[:2]
    finite = lm[np.isfinite(lm).all(axis=1)]
    if finite.size and finite[:, :2].max() <= 2.0:
        lm = lm.copy()
        lm[:, 0] *= w
        lm[:, 1] *= h
    for x, y, *rest in lm:
        if np.isfinite(x) and np.isfinite(y) and 0 <= x < w and 0 <= y < h:
            x, y = int(x), int(y)
            image[max(0, y - 1):min(h, y + 2), max(0, x - 1):min(w, x + 2)] = [255, 40, 40]
