import json
import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


@dataclass
class ColmapCamera:
    camera_id: int
    model: str
    width: int
    height: int
    params: List[float]

    def intrinsics(self) -> tuple[float, float, float, float]:
        if self.model == "SIMPLE_PINHOLE":
            f, cx, cy = self.params[:3]
            return f, f, cx, cy
        if self.model in {"SIMPLE_RADIAL", "RADIAL"}:
            f, cx, cy = self.params[:3]
            return f, f, cx, cy
        if self.model == "PINHOLE":
            fx, fy, cx, cy = self.params[:4]
            return fx, fy, cx, cy
        if self.model in {"OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV"}:
            fx, fy, cx, cy = self.params[:4]
            return fx, fy, cx, cy
        raise ValueError(f"Unsupported COLMAP camera model: {self.model}")

    @property
    def has_distortion(self) -> bool:
        return self.model not in {"SIMPLE_PINHOLE", "PINHOLE"}


@dataclass
class ColmapImage:
    image_id: int
    qvec: np.ndarray
    tvec: np.ndarray
    camera_id: int
    name: str

    @property
    def w2c(self) -> np.ndarray:
        mat = np.eye(4, dtype=np.float32)
        mat[:3, :3] = qvec_to_rotmat(self.qvec)
        mat[:3, 3] = self.tvec
        return mat

    @property
    def c2w(self) -> np.ndarray:
        return np.linalg.inv(self.w2c).astype(np.float32)


def run_colmap_pipeline(
    image_dir: str | Path,
    colmap_dir: str | Path,
    colmap_path: str = "colmap",
    use_masks: bool = False,
) -> Path:
    image_dir = Path(image_dir)
    colmap_dir = Path(colmap_dir)
    if shutil.which(colmap_path) is None and not Path(colmap_path).exists():
        raise FileNotFoundError(f"COLMAP executable not found: {colmap_path}")
    image_count = len([p for p in image_dir.glob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"}])
    if image_count < 2:
        raise RuntimeError(f"COLMAP requires at least 2 images, found {image_count} under {image_dir}")
    mask_dir = image_dir.parent / "fg_masks"

    database_path = colmap_dir / "database.db"
    sparse_dir = colmap_dir / "sparse"
    attempts_dir = colmap_dir / "_attempts"
    if database_path.exists():
        database_path.unlink()
    if sparse_dir.exists():
        shutil.rmtree(sparse_dir)
    if attempts_dir.exists():
        shutil.rmtree(attempts_dir)
    sparse_dir.mkdir(parents=True, exist_ok=True)
    attempts_dir.mkdir(parents=True, exist_ok=True)

    attempts = [
        {
            "name": "exhaustive",
            "matcher": [
                colmap_path,
                "exhaustive_matcher",
                "--database_path",
                "",
                "--FeatureMatching.guided_matching",
                "1",
            ],
        },
        {
            "name": "sequential",
            "matcher": [
                colmap_path,
                "sequential_matcher",
                "--database_path",
                "",
                "--FeatureMatching.guided_matching",
                "1",
                "--SequentialMatching.overlap",
                "10",
                "--SequentialMatching.loop_detection",
                "1",
            ],
        },
    ]

    best_model_dir: Optional[Path] = None
    best_quality: Optional[dict] = None
    attempt_summaries = []

    for attempt in attempts:
        attempt_name = attempt["name"]
        attempt_database = attempts_dir / f"{attempt_name}.db"
        attempt_sparse = attempts_dir / attempt_name
        if attempt_database.exists():
            attempt_database.unlink()
        if attempt_sparse.exists():
            shutil.rmtree(attempt_sparse)
        attempt_sparse.mkdir(parents=True, exist_ok=True)

        try:
            feature_cmd = [
                colmap_path,
                "feature_extractor",
                "--database_path",
                str(attempt_database),
                "--image_path",
                str(image_dir),
                "--ImageReader.single_camera",
                "1",
                "--ImageReader.camera_model",
                "PINHOLE",
                "--SiftExtraction.max_num_features",
                "16000",
                "--SiftExtraction.peak_threshold",
                "0.004",
                "--SiftExtraction.edge_threshold",
                "12",
            ]
            if use_masks and mask_dir.is_dir():
                feature_cmd.extend(["--ImageReader.mask_path", str(mask_dir)])
            _run(feature_cmd)

            matcher_cmd = list(attempt["matcher"])
            matcher_cmd[3] = str(attempt_database)
            _run(matcher_cmd)

            _run([
                colmap_path,
                "mapper",
                "--database_path",
                str(attempt_database),
                "--image_path",
                str(image_dir),
                "--output_path",
                str(attempt_sparse),
                "--Mapper.multiple_models",
                "1",
            ])
        except RuntimeError as exc:
            attempt_summaries.append({
                "name": attempt_name,
                "registered_images": 0,
                "points3d": 0,
                "accepted": False,
                "reason": str(exc).splitlines()[-1][:300],
            })
            continue

        try:
            attempt_model_dir = select_colmap_model(attempt_sparse, colmap_path=colmap_path)
        except RuntimeError:
            attempt_summaries.append({
                "name": attempt_name,
                "registered_images": 0,
                "points3d": 0,
                "accepted": False,
                "reason": "mapper produced no valid model",
            })
            continue

        quality = assess_colmap_model_quality(attempt_model_dir, image_count, colmap_path=colmap_path)
        quality["name"] = attempt_name
        attempt_summaries.append(quality)
        if best_quality is None or _colmap_quality_tuple(quality) > _colmap_quality_tuple(best_quality):
            best_model_dir = attempt_model_dir
            best_quality = quality
        if quality["accepted"]:
            break

    if best_model_dir is None or best_quality is None:
        raise RuntimeError("COLMAP failed to produce any valid sparse model.")

    (colmap_dir / "colmap_attempt_summary.json").write_text(
        json.dumps({"image_count": image_count, "attempts": attempt_summaries}, indent=2),
        encoding="utf-8",
    )

    if not best_quality["accepted"]:
        raise RuntimeError(
            "COLMAP reconstruction is too sparse to trust: "
            f"registered {best_quality['registered_images']}/{image_count} images, "
            f"points3D {best_quality['points3d']}. "
            f"Best attempt was '{best_quality['name']}'."
        )

    if sparse_dir.exists():
        shutil.rmtree(sparse_dir)
    sparse_dir.mkdir(parents=True, exist_ok=True)
    selected_dst = sparse_dir / "0"
    shutil.copytree(best_model_dir, selected_dst)
    if database_path.exists():
        database_path.unlink()
    shutil.copy2(attempts_dir / f"{best_quality['name']}.db", database_path)
    return selected_dst


def import_colmap_sparse(sparse_dir: str | Path, out_json: str | Path, colmap_path: str = "colmap") -> Path:
    sparse_dir = select_colmap_model(sparse_dir, colmap_path=colmap_path)
    out_json = Path(out_json)
    text_dir = _ensure_text_model(sparse_dir, colmap_path)
    cameras = _read_cameras(text_dir / "cameras.txt")
    images = _read_images(text_dir / "images.txt")
    frames = []
    for image in sorted(images.values(), key=lambda item: item.name):
        cam = cameras[image.camera_id]
        raw_fx, raw_fy, raw_cx, raw_cy = cam.intrinsics()
        fx, fy, cx, cy, intrinsics_policy = _sanitize_pinhole_intrinsics(
            cam.width,
            cam.height,
            raw_fx,
            raw_fy,
            raw_cx,
            raw_cy,
        )
        frame = {
            "image_name": image.name,
            "camera_id": image.camera_id,
            "w": cam.width,
            "h": cam.height,
            "fl_x": fx,
            "fl_y": fy,
            "cx": cx,
            "cy": cy,
            "camera_model": cam.model,
            "camera_params": cam.params,
            "requires_undistorted": cam.has_distortion,
            "camera_angle_x": math.atan(cam.width / (fx * 2)) * 2,
            "camera_angle_y": math.atan(cam.height / (fy * 2)) * 2,
            "transform_matrix": image.c2w.tolist(),
        }
        if intrinsics_policy != "colmap":
            frame["intrinsics_policy"] = intrinsics_policy
            frame["colmap_intrinsics_raw"] = {
                "fl_x": raw_fx,
                "fl_y": raw_fy,
                "cx": raw_cx,
                "cy": raw_cy,
                "camera_angle_x": math.atan(cam.width / (raw_fx * 2)) * 2 if raw_fx > 0 else None,
                "camera_angle_y": math.atan(cam.height / (raw_fy * 2)) * 2 if raw_fy > 0 else None,
            }
        frames.append(frame)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    points_count = _count_colmap_points(text_dir / "points3D.txt")
    out_json.write_text(json.dumps({
        "colmap_model_dir": str(sparse_dir.resolve()),
        "colmap_model_stats": {
            "registered_images": len(frames),
            "points3d": points_count,
        },
        "frames": frames,
    }, indent=2), encoding="utf-8")
    return out_json


def select_colmap_model(sparse_dir: str | Path, colmap_path: str = "colmap") -> Path:
    sparse_dir = Path(sparse_dir)
    direct = _try_score_colmap_model(sparse_dir, colmap_path=colmap_path)
    if direct is not None:
        return sparse_dir
    candidates = []
    if sparse_dir.exists():
        for child in sparse_dir.iterdir():
            if not child.is_dir() or not child.name.isdigit():
                continue
            score = _try_score_colmap_model(child, colmap_path=colmap_path)
            if score is not None:
                candidates.append((score, child))
    if not candidates:
        raise RuntimeError(f"No valid COLMAP sparse model found under {sparse_dir}")
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def assess_colmap_model_quality(sparse_dir: str | Path, expected_images: int, colmap_path: str = "colmap") -> dict:
    sparse_dir = Path(sparse_dir)
    score = _try_score_colmap_model(sparse_dir, colmap_path=colmap_path)
    if score is None:
        return {
            "registered_images": 0,
            "points3d": 0,
            "expected_images": int(expected_images),
            "accepted": False,
            "reason": "invalid COLMAP model",
        }
    registered_images, points3d, _ = score
    min_registered_images = _minimum_registered_images(expected_images)
    accepted = registered_images >= min_registered_images
    reason = "accepted"
    if not accepted:
        reason = (
            f"registered images too low ({registered_images}/{expected_images}; "
            f"need at least {min_registered_images})"
        )
    return {
        "registered_images": int(registered_images),
        "points3d": int(points3d),
        "expected_images": int(expected_images),
        "min_registered_images": int(min_registered_images),
        "accepted": bool(accepted),
        "reason": reason,
    }


def qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
    qvec = np.asarray(qvec, dtype=np.float64)
    qvec = qvec / max(np.linalg.norm(qvec), 1e-12)
    return np.array([
        [
            1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
            2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
            2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
        ],
        [
            2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
            1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
            2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
        ],
        [
            2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
            2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
            1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
        ],
    ], dtype=np.float32)


def _default_pinhole_intrinsics(width: int, height: int) -> tuple[float, float, float, float]:
    focal = 1.35 * float(max(width, height))
    return focal, focal, float(width) / 2.0, float(height) / 2.0


def _sanitize_pinhole_intrinsics(
    width: int,
    height: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> tuple[float, float, float, float, str]:
    max_dim = float(max(width, height))
    min_dim = float(min(width, height))
    focal_ok = (
        np.isfinite(fx)
        and np.isfinite(fy)
        and 0.35 * min_dim <= float(fx) <= 6.0 * max_dim
        and 0.35 * min_dim <= float(fy) <= 6.0 * max_dim
    )
    principal_ok = (
        np.isfinite(cx)
        and np.isfinite(cy)
        and -0.25 * float(width) <= float(cx) <= 1.25 * float(width)
        and -0.25 * float(height) <= float(cy) <= 1.25 * float(height)
    )
    if focal_ok and principal_ok:
        return float(fx), float(fy), float(cx), float(cy), "colmap"
    fallback = _default_pinhole_intrinsics(width, height)
    return (*fallback, "default_pinhole_replaced_abnormal_colmap_intrinsics")


def _ensure_text_model(sparse_dir: Path, colmap_path: str) -> Path:
    if (sparse_dir / "cameras.txt").exists() and (sparse_dir / "images.txt").exists():
        return sparse_dir
    if shutil.which(colmap_path) is None and not Path(colmap_path).exists():
        raise FileNotFoundError("COLMAP text model not found and colmap executable is unavailable for conversion.")
    text_dir = sparse_dir.parent / f"{sparse_dir.name}_txt"
    text_dir.mkdir(parents=True, exist_ok=True)
    _run([colmap_path, "model_converter", "--input_path", str(sparse_dir), "--output_path", str(text_dir), "--output_type", "TXT"])
    return text_dir


def _try_score_colmap_model(sparse_dir: Path, colmap_path: str) -> Optional[tuple[int, int, int]]:
    if not _looks_like_colmap_model(sparse_dir):
        return None
    try:
        text_dir = _ensure_text_model(sparse_dir, colmap_path)
        images_count = _count_colmap_images(text_dir / "images.txt")
        points_count = _count_colmap_points(text_dir / "points3D.txt")
    except FileNotFoundError:
        raise
    except Exception:
        return None
    if images_count <= 0:
        return None
    numeric_name = int(sparse_dir.name) if sparse_dir.name.isdigit() else -1
    return images_count, points_count, -numeric_name


def _looks_like_colmap_model(path: Path) -> bool:
    if not path.is_dir():
        return False
    has_text = (path / "cameras.txt").exists() and (path / "images.txt").exists()
    has_binary = (path / "cameras.bin").exists() and (path / "images.bin").exists()
    return has_text or has_binary


def _count_colmap_images(path: Path) -> int:
    if not path.exists():
        return 0
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if not line.startswith("#")]
    count = 0
    idx = 0
    while idx < len(lines):
        line = lines[idx].strip()
        if not line:
            idx += 1
            continue
        parts = line.split()
        if len(parts) >= 10:
            count += 1
            idx += 2
        else:
            idx += 1
    return count


def _count_colmap_points(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line and not line.startswith("#"))


def _minimum_registered_images(expected_images: int) -> int:
    expected_images = max(int(expected_images), 0)
    if expected_images <= 3:
        return expected_images
    return max(3, int(math.ceil(0.5 * expected_images)))


def _colmap_quality_tuple(quality: dict) -> tuple:
    return (
        int(bool(quality.get("accepted"))),
        int(quality.get("registered_images", 0)),
        int(quality.get("points3d", 0)),
    )


def _read_cameras(path: Path) -> Dict[int, ColmapCamera]:
    cameras = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        camera_id = int(parts[0])
        cameras[camera_id] = ColmapCamera(
            camera_id=camera_id,
            model=parts[1],
            width=int(parts[2]),
            height=int(parts[3]),
            params=[float(v) for v in parts[4:]],
        )
    return cameras


def _read_images(path: Path) -> Dict[int, ColmapImage]:
    images = {}
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if not line.startswith("#")]
    idx = 0
    while idx < len(lines):
        line = lines[idx].strip()
        if not line:
            idx += 1
            continue
        parts = line.split()
        if len(parts) < 10:
            idx += 1
            continue
        image_id = int(parts[0])
        images[image_id] = ColmapImage(
            image_id=image_id,
            qvec=np.array([float(v) for v in parts[1:5]], dtype=np.float32),
            tvec=np.array([float(v) for v in parts[5:8]], dtype=np.float32),
            camera_id=int(parts[8]),
            name=" ".join(parts[9:]),
        )
        idx += 2
    return images


def _run(cmd: List[str]) -> None:
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        raise RuntimeError("Command failed:\n" + " ".join(cmd) + "\n" + proc.stdout[-4000:])
