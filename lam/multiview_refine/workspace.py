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


def unpack_multiview_zip(zip_path: str | Path, workspace: MultiViewWorkspace) -> MultiViewWorkspace:
    zip_path = Path(zip_path)
    if not zip_path.exists():
        raise FileNotFoundError(f"ZIP not found: {zip_path}")
    with zipfile.ZipFile(zip_path, "r") as zipf:
        _safe_extract(zipf, workspace.root)
    _normalize_single_top_folder(workspace.root)
    workspace.write_state(input_zip=str(zip_path), unpacked=True)
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
        _copy_dir(Path(flame_dir), workspace.flame_dir)
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


def _normalize_single_top_folder(root: Path) -> None:
    children = [p for p in root.iterdir() if p.name not in {"debug", "checkpoints", "workspace_state.json"}]
    dirs = [p for p in children if p.is_dir()]
    files = [p for p in children if p.is_file()]
    if len(dirs) != 1 or files:
        return
    top = dirs[0]
    expected = {"images", "masks", "fg_masks", "flame_param", "colmap"}
    if not any((top / name).exists() for name in expected):
        return
    for child in top.iterdir():
        target = root / child.name
        if target.exists():
            continue
        shutil.move(str(child), str(target))
    try:
        top.rmdir()
    except OSError:
        pass


def _jsonable(value):
    if isinstance(value, Path):
        return str(value)
    return value
