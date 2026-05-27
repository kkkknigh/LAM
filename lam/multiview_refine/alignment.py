import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List

import numpy as np


@dataclass
class Sim3Alignment:
    scale: float
    rotation: List[List[float]]
    translation: List[float]
    rmse: float
    source: str = "estimated"

    @property
    def R(self) -> np.ndarray:
        return np.asarray(self.rotation, dtype=np.float32)

    @property
    def T(self) -> np.ndarray:
        return np.asarray(self.translation, dtype=np.float32)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "Sim3Alignment":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


def estimate_sim3_umeyama(source_points: np.ndarray, target_points: np.ndarray, estimate_scale: bool = True) -> Sim3Alignment:
    source = np.asarray(source_points, dtype=np.float64)
    target = np.asarray(target_points, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("source_points and target_points must both have shape [N, 3]")
    if source.shape[0] < 3:
        raise ValueError("At least 3 corresponding camera centers are required for Sim(3) alignment")

    src_mean = source.mean(axis=0)
    tgt_mean = target.mean(axis=0)
    src_c = source - src_mean
    tgt_c = target - tgt_mean
    cov = src_c.T @ tgt_c / source.shape[0]
    u, singular, vt = np.linalg.svd(cov)
    fix = np.eye(3)
    if np.linalg.det(u @ vt) < 0:
        fix[-1, -1] = -1
    R_row = u @ fix @ vt
    if estimate_scale:
        var = np.sum(src_c ** 2) / source.shape[0]
        scale = float(np.sum(singular * np.diag(fix)) / max(var, 1e-9))
    else:
        scale = 1.0
    T = tgt_mean - scale * (src_mean @ R_row)
    aligned = scale * (source @ R_row) + T
    rmse = float(np.sqrt(np.mean(np.sum((aligned - target) ** 2, axis=1))))
    return Sim3Alignment(scale=scale, rotation=R_row.astype(np.float32).tolist(), translation=T.astype(np.float32).tolist(), rmse=rmse)


def apply_sim3_to_points(points: np.ndarray, sim3: Sim3Alignment) -> np.ndarray:
    return sim3.scale * (points @ sim3.R) + sim3.T


def apply_sim3_to_c2w(c2w: np.ndarray, sim3: Sim3Alignment) -> np.ndarray:
    c2w = np.asarray(c2w, dtype=np.float32)
    out = c2w.copy()
    # Row-vector convention for point alignment is X' = s * X @ R + T.
    # Camera axes are stored as columns in c2w, so the equivalent basis update is R^T @ basis.
    out[:3, :3] = sim3.R.T @ c2w[:3, :3]
    center = c2w[:3, 3]
    out[:3, 3] = apply_sim3_to_points(center[None], sim3)[0]
    return out.astype(np.float32)


def align_colmap_to_flame(
    colmap_transforms: str | Path,
    flame_target_transforms: str | Path,
    output_transforms: str | Path,
    output_sim3: str | Path,
) -> Sim3Alignment:
    colmap_db = json.loads(Path(colmap_transforms).read_text(encoding="utf-8"))
    flame_db = json.loads(Path(flame_target_transforms).read_text(encoding="utf-8"))
    flame_by_name = {_frame_name(frame): frame for frame in flame_db["frames"]}

    src_centers, tgt_centers, pairs = [], [], []
    for frame in colmap_db["frames"]:
        name = _frame_name(frame)
        if name not in flame_by_name:
            continue
        src = np.asarray(frame["transform_matrix"], dtype=np.float32)[:3, 3]
        tgt = np.asarray(flame_by_name[name]["transform_matrix"], dtype=np.float32)[:3, 3]
        src_centers.append(src)
        tgt_centers.append(tgt)
        pairs.append(name)
    if len(pairs) < 3:
        raise ValueError(f"Need at least 3 matched COLMAP/FLAME cameras, got {len(pairs)}")

    sim3 = estimate_sim3_umeyama(np.stack(src_centers), np.stack(tgt_centers), estimate_scale=True)
    aligned_frames = []
    for frame in colmap_db["frames"]:
        aligned = dict(frame)
        aligned["transform_matrix_colmap"] = frame["transform_matrix"]
        aligned["transform_matrix"] = apply_sim3_to_c2w(np.asarray(frame["transform_matrix"], dtype=np.float32), sim3).tolist()
        aligned_frames.append(aligned)
    output_transforms = Path(output_transforms)
    output_transforms.parent.mkdir(parents=True, exist_ok=True)
    output_transforms.write_text(json.dumps({"frames": aligned_frames, "matched_frames": pairs}, indent=2), encoding="utf-8")
    sim3.save(output_sim3)
    return sim3


def write_manual_sim3(path: str | Path, scale: float, yaw_degrees: float, translation: Iterable[float]) -> Sim3Alignment:
    yaw = np.deg2rad(yaw_degrees)
    R = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]], dtype=np.float32)
    sim3 = Sim3Alignment(scale=float(scale), rotation=R.tolist(), translation=list(map(float, translation)), rmse=-1.0, source="manual")
    sim3.save(path)
    return sim3


def _frame_name(frame: dict) -> str:
    value = frame.get("image_name") or frame.get("file_path") or frame.get("image_path")
    return Path(value).name
