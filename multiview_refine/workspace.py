import shutil
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


@dataclass
class MultiViewWorkspace:
    root: Path

    @property
    def inputs_dir(self) -> Path:
        return self.root / "inputs"

    @property
    def data_dir(self) -> Path:
        return self.root / "data"

    @property
    def images_dir(self) -> Path:
        return self.data_dir / "images"

    @property
    def masks_dir(self) -> Path:
        return self.data_dir / "fg_masks"

    @property
    def flame_dir(self) -> Path:
        return self.data_dir / "flame_param"

    @property
    def landmark_dir(self) -> Path:
        return self.data_dir / "landmark2d"

    @property
    def init_ply_path(self) -> Path:
        return self.data_dir / "init.ply"

    @property
    def canonical_flame_path(self) -> Path:
        return self.data_dir / "canonical_flame_param.npz"

    @property
    def layer1_metadata_path(self) -> Path:
        return self.data_dir / "layer1_metadata.json"

    @property
    def layer1_reference_transforms_path(self) -> Path:
        return self.data_dir / "layer1_reference_transforms.json"

    @property
    def colmap_dir(self) -> Path:
        return self.root / "colmap"

    @property
    def colmap_transforms_path(self) -> Path:
        return self.colmap_dir / "transforms_colmap_raw.json"

    @property
    def alignment_dir(self) -> Path:
        return self.root / "alignment"

    @property
    def aligned_transforms_path(self) -> Path:
        return self.alignment_dir / "transforms_aligned.json"

    @property
    def sim3_path(self) -> Path:
        return self.alignment_dir / "sim3_colmap_to_lam.json"

    @property
    def tracking_dir(self) -> Path:
        return self.root / "tracking_work"

    @property
    def refine_dir(self) -> Path:
        return self.root / "refine"

    @property
    def refined_gaussian_path(self) -> Path:
        return self.refine_dir / "refined_gaussian.ply"

    @property
    def loss_history_path(self) -> Path:
        return self.refine_dir / "loss_history.jsonl"

    @property
    def exports_dir(self) -> Path:
        return self.root / "exports"

    @property
    def debug_dir(self) -> Path:
        return self.root / "debug"

    @property
    def checkpoint_dir(self) -> Path:
        return self.refine_dir / "checkpoints"

def create_workspace(output_root: str | Path = "output/multiview_refine", job_id: Optional[str] = None) -> MultiViewWorkspace:
    output_root = Path(output_root)
    job_id = job_id or time.strftime("%Y%m%d_%H%M%S")
    ws = MultiViewWorkspace(output_root / job_id)
    suffix = 1
    while ws.root.exists() and any(ws.root.iterdir()):
        ws = MultiViewWorkspace(output_root / f"{job_id}_{suffix:02d}")
        suffix += 1
    for path in [
        ws.root,
        ws.inputs_dir,
        ws.data_dir,
        ws.colmap_dir,
        ws.alignment_dir,
        ws.tracking_dir,
        ws.refine_dir,
        ws.debug_dir,
        ws.checkpoint_dir,
        ws.exports_dir,
    ]:
        path.mkdir(parents=True, exist_ok=True)
    return ws


def unpack_camera_images_zip(zip_path: str | Path, workspace: MultiViewWorkspace) -> MultiViewWorkspace:
    zip_path = Path(zip_path)
    shutil.copy2(zip_path, workspace.inputs_dir / zip_path.name)
    source_root = _extract_zip_to_temp(zip_path, workspace.root / "_tmp_camera_images")
    content_root = _single_top_folder(source_root)
    if (content_root / "images").is_dir():
        image_files = _image_files(content_root / "images")
    else:
        image_files = sorted([p for p in content_root.iterdir() if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}])
    if not image_files:
        raise FileNotFoundError(f"No images found in camera image ZIP: {zip_path}")
    if workspace.images_dir.exists():
        shutil.rmtree(workspace.images_dir)
    workspace.images_dir.mkdir(parents=True, exist_ok=True)
    for image_path in image_files:
        shutil.copy2(image_path, workspace.images_dir / image_path.name)
    mask_source = None
    if (content_root / "fg_masks").is_dir():
        mask_source = content_root / "fg_masks"
    elif (content_root / "masks").is_dir():
        mask_source = content_root / "masks"
    if mask_source is not None:
        _copy_dir(mask_source, workspace.masks_dir)
    shutil.rmtree(source_root)
    return workspace


def unpack_layer1_lam_zip(zip_path: str | Path, workspace: MultiViewWorkspace) -> MultiViewWorkspace:
    zip_path = Path(zip_path)
    shutil.copy2(zip_path, workspace.inputs_dir / zip_path.name)
    source_root = _extract_zip_to_temp(zip_path, workspace.root / "_tmp_layer1_lam")
    content_root = _single_top_folder(source_root)

    init_ply = _find_layer1_canonical_ply(content_root)
    canonical_flame = _find_canonical_flame_param(content_root)
    if canonical_flame is None:
        raise FileNotFoundError(
            "Layer 1 LAM package must contain canonical_flame_param.npz or *_canonical_flame_param.npz."
        )

    shutil.copy2(init_ply, workspace.init_ply_path)
    shutil.copy2(canonical_flame, workspace.canonical_flame_path)
    metadata = content_root / "layer1_metadata.json"
    if metadata.exists():
        shutil.copy2(metadata, workspace.layer1_metadata_path)
    reference_transforms = content_root / "layer1_reference_transforms.json"
    if reference_transforms.exists():
        shutil.copy2(reference_transforms, workspace.layer1_reference_transforms_path)
    shutil.rmtree(source_root)
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
        _copy_dir(Path(mask_dir), workspace.masks_dir)
    if flame_dir:
        flame_dir = Path(flame_dir)
        source_flame_param = flame_dir / "flame_param" if (flame_dir / "flame_param").is_dir() else flame_dir
        _copy_dir(source_flame_param, workspace.flame_dir)
        canonical = flame_dir / "canonical_flame_param.npz"
        if canonical.exists():
            shutil.copy2(canonical, workspace.canonical_flame_path)
    if colmap_dir:
        _copy_dir(Path(colmap_dir), workspace.colmap_dir)
    if init_ply:
        shutil.copy2(init_ply, workspace.init_ply_path)
    return workspace


def validate_workspace_inputs(workspace: MultiViewWorkspace, require_masks: bool = True) -> dict:
    images = sorted([p for p in workspace.images_dir.glob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"}])
    if not images:
        raise FileNotFoundError(f"No images found under {workspace.images_dir}")
    masks = []
    if workspace.masks_dir.exists():
        masks = sorted([p for p in workspace.masks_dir.glob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"}])
    missing_masks = [
        p.name
        for p in images
        if find_stem_file(workspace.masks_dir, p.stem, [".png", ".jpg", ".jpeg"]) is None
    ]
    if require_masks and missing_masks:
        raise FileNotFoundError(f"Missing masks for images: {missing_masks[:10]}")
    report = {
        "num_images": len(images),
        "num_masks": len(masks),
        "missing_masks": missing_masks,
        "has_flame": workspace.flame_dir.exists(),
        "has_colmap": workspace.colmap_dir.exists(),
        "has_init_ply": workspace.init_ply_path.exists(),
    }
    return report


def find_stem_file(base: Path, stem: str, suffixes: Iterable[str]) -> Optional[Path]:
    base = Path(base)
    if not base.is_dir():
        return None
    suffixes = [suffix.lower() for suffix in suffixes]
    for suffix in suffixes:
        candidate = base / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    key = numeric_stem_key(stem)
    if key is None:
        return None
    matches = [
        path
        for path in base.iterdir()
        if path.is_file()
        and path.suffix.lower() in suffixes
        and numeric_stem_key(path.stem) == key
    ]
    return matches[0] if len(matches) == 1 else None


def numeric_stem_key(stem: str) -> Optional[int]:
    text = str(stem)
    return int(text) if text.isdigit() else None


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

