import json
import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

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


def run_colmap_pipeline(image_dir: str | Path, colmap_dir: str | Path, colmap_path: str = "colmap") -> Path:
    image_dir = Path(image_dir)
    colmap_dir = Path(colmap_dir)
    if shutil.which(colmap_path) is None and not Path(colmap_path).exists():
        raise FileNotFoundError(f"COLMAP executable not found: {colmap_path}")
    database_path = colmap_dir / "database.db"
    sparse_dir = colmap_dir / "sparse"
    sparse_dir.mkdir(parents=True, exist_ok=True)
    _run([
        colmap_path,
        "feature_extractor",
        "--database_path",
        str(database_path),
        "--image_path",
        str(image_dir),
        "--ImageReader.camera_model",
        "PINHOLE",
    ])
    _run([colmap_path, "exhaustive_matcher", "--database_path", str(database_path)])
    _run([colmap_path, "mapper", "--database_path", str(database_path), "--image_path", str(image_dir), "--output_path", str(sparse_dir)])
    model_dir = sparse_dir / "0"
    if not model_dir.exists():
        raise RuntimeError(f"COLMAP mapper did not produce {model_dir}")
    return model_dir


def import_colmap_sparse(sparse_dir: str | Path, out_json: str | Path, colmap_path: str = "colmap") -> Path:
    sparse_dir = Path(sparse_dir)
    out_json = Path(out_json)
    text_dir = _ensure_text_model(sparse_dir, colmap_path)
    cameras = _read_cameras(text_dir / "cameras.txt")
    images = _read_images(text_dir / "images.txt")
    frames = []
    for image in sorted(images.values(), key=lambda item: item.name):
        cam = cameras[image.camera_id]
        fx, fy, cx, cy = cam.intrinsics()
        frames.append({
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
        })
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps({"frames": frames}, indent=2), encoding="utf-8")
    return out_json


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


def _ensure_text_model(sparse_dir: Path, colmap_path: str) -> Path:
    if (sparse_dir / "cameras.txt").exists() and (sparse_dir / "images.txt").exists():
        return sparse_dir
    if shutil.which(colmap_path) is None and not Path(colmap_path).exists():
        raise FileNotFoundError("COLMAP text model not found and colmap executable is unavailable for conversion.")
    text_dir = sparse_dir.parent / f"{sparse_dir.name}_txt"
    text_dir.mkdir(parents=True, exist_ok=True)
    _run([colmap_path, "model_converter", "--input_path", str(sparse_dir), "--output_path", str(text_dir), "--output_type", "TXT"])
    return text_dir


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
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line and not line.startswith("#")]
    for i in range(0, len(lines), 2):
        parts = lines[i].split()
        image_id = int(parts[0])
        images[image_id] = ColmapImage(
            image_id=image_id,
            qvec=np.array([float(v) for v in parts[1:5]], dtype=np.float32),
            tvec=np.array([float(v) for v in parts[5:8]], dtype=np.float32),
            camera_id=int(parts[8]),
            name=" ".join(parts[9:]),
        )
    return images


def _run(cmd: List[str]) -> None:
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        raise RuntimeError("Command failed:\n" + " ".join(cmd) + "\n" + proc.stdout[-4000:])
