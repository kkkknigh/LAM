''' '''
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Sequence

import numpy as np


@dataclass
class Sim3Alignment:
    scale: float
    rotation: List[List[float]]
    translation: List[float]
    rmse: float
    source: str = "estimated"
    inlier_count: int = 0
    total_count: int = 0
    inlier_rmse: float | None = None
    inlier_names: List[str] | None = None

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
    _validate_sim3_point_sets(source, target)

    # 估计把 source 相机中心映射到 target 坐标系的 Sim(3)：
    # 先把两组点去中心化，用 SVD 求最优旋转，再解尺度和平移。
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
    return Sim3Alignment(
        scale=scale,
        rotation=R_row.astype(np.float32).tolist(),
        translation=T.astype(np.float32).tolist(),
        rmse=rmse,
        inlier_count=source.shape[0],
        total_count=source.shape[0],
        inlier_rmse=rmse,
    )


def estimate_sim3_umeyama_ransac(
    source_points: np.ndarray,
    target_points: np.ndarray,
    estimate_scale: bool = True,
    names: Sequence[str] | None = None,
    inlier_threshold: float | None = None,
    max_iterations: int = 256,
    random_seed: int = 0,
) -> Sim3Alignment:
    source = np.asarray(source_points, dtype=np.float64)
    target = np.asarray(target_points, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("source_points and target_points must both have shape [N, 3]")
    if source.shape[0] < 3:
        raise ValueError("At least 3 corresponding camera centers are required for Sim(3) alignment")
    _validate_sim3_point_sets(source, target)
    if names is not None and len(names) != source.shape[0]:
        raise ValueError("names must have the same length as source_points")

    total = source.shape[0]
    if total == 3:
        sim3 = estimate_sim3_umeyama(source, target, estimate_scale=estimate_scale)
        sim3.source = "umeyama"
        sim3.inlier_names = list(names) if names is not None else None
        return sim3

    threshold = _auto_ransac_threshold(source, target, estimate_scale) if inlier_threshold is None else float(inlier_threshold)
    min_inliers = max(3, int(np.ceil(total * 0.5)))
    rng = np.random.default_rng(random_seed)
    best_mask = None
    best_score = (-1, -np.inf)

    for _ in range(max_iterations):
        sample = rng.choice(total, size=3, replace=False)
        if _is_degenerate(source[sample]) or _is_degenerate(target[sample]):
            continue
        try:
            candidate = estimate_sim3_umeyama(source[sample], target[sample], estimate_scale=estimate_scale)
        except (ValueError, np.linalg.LinAlgError):
            continue
        residuals = _sim3_residuals(source, target, candidate)
        mask = residuals <= threshold
        count = int(mask.sum())
        if count < min_inliers:
            continue
        median_residual = float(np.median(residuals[mask]))
        score = (count, -median_residual)
        if score > best_score:
            best_score = score
            best_mask = mask

    if best_mask is None:
        sim3 = estimate_sim3_umeyama(source, target, estimate_scale=estimate_scale)
        sim3.source = "umeyama"
        sim3.inlier_names = list(names) if names is not None else None
        return sim3

    for _ in range(3):
        sim3 = estimate_sim3_umeyama(source[best_mask], target[best_mask], estimate_scale=estimate_scale)
        residuals = _sim3_residuals(source, target, sim3)
        refined_mask = residuals <= threshold
        if int(refined_mask.sum()) < 3 or np.array_equal(refined_mask, best_mask):
            break
        best_mask = refined_mask

    sim3 = estimate_sim3_umeyama(source[best_mask], target[best_mask], estimate_scale=estimate_scale)
    residuals = _sim3_residuals(source, target, sim3)
    inlier_residuals = residuals[best_mask]
    sim3.rmse = float(np.sqrt(np.mean(residuals ** 2)))
    sim3.source = "ransac"
    sim3.inlier_count = int(best_mask.sum())
    sim3.total_count = total
    sim3.inlier_rmse = float(np.sqrt(np.mean(inlier_residuals ** 2)))
    sim3.inlier_names = [name for name, keep in zip(names, best_mask) if keep] if names is not None else None
    return sim3


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
    robust: bool = True,
    ransac_threshold: float | None = None,
) -> Sim3Alignment:
    # 输入是两份带 frames/transform_matrix 的 transforms JSON：
    # COLMAP 提供待对齐的 c2w，相同文件名的显式标定 target 提供 LAM/FLAME 坐标系中的目标 c2w。
    # 这里只用两边匹配帧的相机中心估计一个全局 Sim(3)，再应用到全部 COLMAP c2w。
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

    src_stack = np.stack(src_centers)
    tgt_stack = np.stack(tgt_centers)
    if robust:
        sim3 = estimate_sim3_umeyama_ransac(
            src_stack,
            tgt_stack,
            estimate_scale=True,
            names=pairs,
            inlier_threshold=ransac_threshold,
        )
    else:
        sim3 = estimate_sim3_umeyama(src_stack, tgt_stack, estimate_scale=True)
        sim3.inlier_names = list(pairs)
    aligned_frames = []
    for frame in colmap_db["frames"]:
        aligned = dict(frame)
        aligned["transform_matrix_colmap"] = frame["transform_matrix"]
        aligned["transform_matrix"] = apply_sim3_to_c2w(np.asarray(frame["transform_matrix"], dtype=np.float32), sim3).tolist()
        aligned_frames.append(aligned)
    output_transforms = Path(output_transforms)
    output_transforms.parent.mkdir(parents=True, exist_ok=True)
    output_transforms.write_text(
        json.dumps({"frames": aligned_frames, "matched_frames": pairs, "sim3_inlier_frames": sim3.inlier_names}, indent=2),
        encoding="utf-8",
    )
    sim3.save(output_sim3)
    return sim3


def _frame_name(frame: dict) -> str:
    value = frame.get("image_name") or frame.get("file_path") or frame.get("image_path")
    return Path(value).name


def _sim3_residuals(source: np.ndarray, target: np.ndarray, sim3: Sim3Alignment) -> np.ndarray:
    aligned = apply_sim3_to_points(np.asarray(source, dtype=np.float64), sim3)
    return np.linalg.norm(aligned - np.asarray(target, dtype=np.float64), axis=1)


def _auto_ransac_threshold(source: np.ndarray, target: np.ndarray, estimate_scale: bool) -> float:
    span = float(np.sqrt(np.mean(np.sum((target - target.mean(axis=0)) ** 2, axis=1))))
    floor = max(span * 0.02, 1e-4)
    try:
        initial = estimate_sim3_umeyama(source, target, estimate_scale=estimate_scale)
        residuals = _sim3_residuals(source, target, initial)
        median = float(np.median(residuals))
        mad = float(np.median(np.abs(residuals - median)))
        sigma = 1.4826 * mad
        if np.isfinite(sigma) and sigma > 0:
            return max(2.5 * sigma, floor)
    except (ValueError, np.linalg.LinAlgError):
        pass
    return floor


def _is_degenerate(points: np.ndarray) -> bool:
    centered = points - points.mean(axis=0)
    return np.linalg.matrix_rank(centered, tol=1e-8) < 2


def _validate_sim3_point_sets(source: np.ndarray, target: np.ndarray) -> None:
    if _is_degenerate(source):
        raise ValueError("Source camera centers are degenerate; need at least 3 non-collinear matched cameras.")
    if _is_degenerate(target):
        raise ValueError(
            "Target camera centers are degenerate; provide calibrated multi-view target transforms with "
            "non-collinear camera centers."
        )
