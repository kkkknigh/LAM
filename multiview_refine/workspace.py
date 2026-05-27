import json
import shutil
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class MultiViewWorkspace:
    root: Path

    @property
    def images_dir(self) -> Path:
        return self.root / "images"

    @property
    def masks_dir(self) -> Path:
        fg_masks = self.root / "fg_masks"
        return fg_masks if fg_masks.exists() else self.root / "masks"

    @property
    def flame_dir(self) -> Path:
        return self.root / "flame_param"

    @property
    def landmark_dir(self) -> Path:
        return self.root / "landmark2d"

    @property
    def colmap_dir(self) -> Path:
        return self.root / "colmap"

    @property
    def debug_dir(self) -> Path:
        return self.root / "debug"

    @property
    def checkpoint_dir(self) -> Path:
        return self.root / "checkpoints"

    @property
    def state_path(self) -> Path:
        return self.root / "workspace_state.json"

    def write_state(self, **updates) -> dict:
        state = {}
        if self.state_path.exists():
            state = json.loads(self.state_path.read_text(encoding="utf-8"))
        state.update({k: _jsonable(v) for k, v in updates.items()})
        self.state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
        return state


def create_workspace(output_root: str | Path = "output/multiview_refine", job_id: Optional[str] = None) -> MultiViewWorkspace:
    output_root = Path(output_root)
    job_id = job_id or time.strftime("%Y%m%d_%H%M%S")
    ws = MultiViewWorkspace(output_root / job_id)
    for path in [ws.root, ws.debug_dir, ws.checkpoint_dir]:
        path.mkdir(parents=True, exist_ok=True)
    ws.write_state(job_id=job_id, root=str(ws.root), created_at=time.time())
    return ws


def unpack_camera_images_zip(zip_path: str | Path, workspace: MultiViewWorkspace) -> MultiViewWorkspace:
    source_root = _extract_zip_to_temp(zip_path, workspace.root / "_upload_camera_images")
    content_root = _single_top_folder(source_root)
    image_source = content_root / "images" if (content_root / "images").is_dir() else content_root
    image_files = _image_files(image_source)
    if not image_files:
        raise FileNotFoundError(f"No images found in camera image ZIP: {zip_path}")
    if workspace.images_dir.exists():
        shutil.rmtree(workspace.images_dir)
    workspace.images_dir.mkdir(parents=True, exist_ok=True)
    for image_path in image_files:
        shutil.copy2(image_path, workspace.images_dir / image_path.name)
    shutil.rmtree(source_root)
    workspace.write_state(camera_images_zip=str(zip_path), camera_images=len(image_files))
    return workspace


def unpack_layer1_lam_zip(zip_path: str | Path, workspace: MultiViewWorkspace) -> MultiViewWorkspace:
    source_root = _extract_zip_to_temp(zip_path, workspace.root / "_upload_layer1_lam")
    content_root = _single_top_folder(source_root)

    init_ply = _find_layer1_canonical_ply(content_root)
    canonical_flame = _find_canonical_flame_param(content_root)
    if canonical_flame is None:
        raise FileNotFoundError(
            "Layer 1 LAM package must contain canonical_flame_param.npz or *_canonical_flame_param.npz."
        )

    shutil.copy2(init_ply, workspace.root / "init.ply")
    shutil.copy2(canonical_flame, workspace.root / "canonical_flame_param.npz")
    _copy_layer1_frame_params(content_root, workspace.root / "layer1_frame_param")
    shutil.rmtree(source_root)
    workspace.write_state(layer1_lam_zip=str(zip_path), init_ply="init.ply", canonical_flame_param="canonical_flame_param.npz")
    return workspace


def import_local_inputs(
    workspace: MultiViewWorkspace,
    image_dir: str | Path,
    mask_dir: Optional[str | Path] = None,
    flame_dir: Optional[str | Path] = None,
    colmap_dir: Optional[str | Path] = None,
    init_ply: Optional[str | Path] = None,
) -> MultiViewWorkspace:
    _copy_dir(Path(image_dir), workspace.images_dir)
    if mask_dir:
        _copy_dir(Path(mask_dir), workspace.root / "fg_masks")
    if flame_dir:
        flame_dir = Path(flame_dir)
        source_flame_param = flame_dir / "flame_param" if (flame_dir / "flame_param").is_dir() else flame_dir
        _copy_dir(source_flame_param, workspace.flame_dir)
        canonical = flame_dir / "canonical_flame_param.npz"
        if canonical.exists():
            shutil.copy2(canonical, workspace.root / "canonical_flame_param.npz")
    if colmap_dir:
        _copy_dir(Path(colmap_dir), workspace.colmap_dir)
    if init_ply:
        shutil.copy2(init_ply, workspace.root / "init.ply")
    workspace.write_state(imported=True)
    return workspace


def validate_workspace_inputs(workspace: MultiViewWorkspace, require_masks: bool = True) -> dict:
    images = sorted([p for p in workspace.images_dir.glob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"}])
    if not images:
        raise FileNotFoundError(f"No images found under {workspace.images_dir}")
    masks = []
    if workspace.masks_dir.exists():
        masks = sorted([p for p in workspace.masks_dir.glob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"}])
    mask_stems = {p.stem for p in masks}
    missing_masks = [p.name for p in images if p.stem not in mask_stems]
    if require_masks and missing_masks:
        raise FileNotFoundError(f"Missing masks for images: {missing_masks[:10]}")
    report = {
        "num_images": len(images),
        "num_masks": len(masks),
        "missing_masks": missing_masks,
        "has_flame": workspace.flame_dir.exists(),
        "has_colmap": workspace.colmap_dir.exists(),
        "has_init_ply": (workspace.root / "init.ply").exists(),
    }
    workspace.write_state(validation=report)
    return report


def _copy_dir(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Directory not found: {src}")
    src_resolved = src.resolve()
    dst_resolved = dst.resolve() if dst.exists() else (dst.parent.resolve() / dst.name)
    if src_resolved == dst_resolved:
        return
    if src_resolved in dst_resolved.parents or dst_resolved in src_resolved.parents:
        raise ValueError(f"Refusing to copy overlapping directories: {src_resolved} -> {dst_resolved}")
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def _safe_extract(zipf: zipfile.ZipFile, target_root: Path) -> None:
    target_root = target_root.resolve()
    for member in zipf.infolist():
        target = (target_root / member.filename).resolve()
        if target != target_root and target_root not in target.parents:
            raise ValueError(f"Unsafe ZIP member path: {member.filename}")
    zipf.extractall(target_root)


def _extract_zip_to_temp(zip_path: str | Path, target_root: Path) -> Path:
    zip_path = Path(zip_path)
    if not zip_path.exists():
        raise FileNotFoundError(f"ZIP not found: {zip_path}")
    if target_root.exists():
        shutil.rmtree(target_root)
    target_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zipf:
        _safe_extract(zipf, target_root)
    return target_root


def _single_top_folder(root: Path) -> Path:
    children = [p for p in root.iterdir() if not p.name.startswith("__MACOSX")]
    dirs = [p for p in children if p.is_dir()]
    files = [p for p in children if p.is_file()]
    if len(dirs) == 1 and not files:
        return dirs[0]
    return root


def _image_files(root: Path) -> list[Path]:
    return sorted([p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}])


def _find_layer1_canonical_ply(root: Path) -> Path:
    ply_files = sorted([p for p in root.rglob("*.ply") if p.is_file()])
    if not ply_files:
        raise FileNotFoundError("Layer 1 LAM package must contain a canonical Gaussian .ply.")
    preferred = [p for p in ply_files if "canonical" in p.stem.lower() and "offset" not in p.stem.lower()]
    if preferred:
        return preferred[0]
    init_named = [p for p in ply_files if p.name.lower() == "init.ply"]
    if init_named:
        return init_named[0]
    non_offset = [p for p in ply_files if "offset" not in p.stem.lower()]
    if len(non_offset) == 1:
        return non_offset[0]
    offset_only = ", ".join(p.name for p in ply_files[:5])
    raise ValueError(
        "Layer 1 LAM package must provide absolute canonical Gaussian xyz, e.g. *_canonical.ply or init.ply. "
        f"Refusing ambiguous/offset-only PLY files: {offset_only}"
    )


def _find_canonical_flame_param(root: Path) -> Optional[Path]:
    candidates = sorted([p for p in root.rglob("*.npz") if p.is_file()])
    exact = [p for p in candidates if p.name == "canonical_flame_param.npz"]
    if exact:
        return exact[0]
    suffix = [p for p in candidates if p.name.endswith("_canonical_flame_param.npz")]
    if suffix:
        return suffix[0]
    return None


def _copy_layer1_frame_params(root: Path, dst: Path) -> None:
    candidates = [
        p for p in root.rglob("*.npz")
        if p.is_file() and p.name != "canonical_flame_param.npz" and not p.name.endswith("_canonical_flame_param.npz")
    ]
    if (root / "flame_param").is_dir():
        candidates.extend([p for p in (root / "flame_param").glob("*.npz") if p.is_file()])
    if not candidates:
        return
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)
    seen = set()
    for src in sorted(candidates):
        if src.resolve() in seen:
            continue
        seen.add(src.resolve())
        shutil.copy2(src, dst / src.name)


def _jsonable(value):
    if isinstance(value, Path):
        return str(value)
    return value
