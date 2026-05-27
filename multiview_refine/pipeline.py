import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

from lam.models.rendering.gaussian_model import GaussianModel

from .alignment import align_colmap_to_flame, write_manual_sim3
from .colmap import import_colmap_sparse, run_colmap_pipeline
from .data import load_multiview_bundle, write_lam_transforms_from_colmap
from .optimization import MultiViewGaussianRefiner, RefinementConfig
from .render_adapter import render_animate_gs_with_intrinsics
from .visualization import (
    save_loss_plot,
    save_overlay_grid,
    save_transform_camera_plot,
    save_workspace_image_previews,
)
from .workspace import (
    MultiViewWorkspace,
    create_workspace,
    import_local_inputs,
    unpack_camera_images_zip,
    unpack_layer1_lam_zip,
    validate_workspace_inputs,
)


@dataclass
class StepResult:
    message: str
    path: str = ""


class MultiViewRefinePipeline:
    def __init__(self, workspace: str | Path | MultiViewWorkspace) -> None:
        if isinstance(workspace, MultiViewWorkspace):
            self.workspace = workspace
        else:
            self.workspace = MultiViewWorkspace(Path(workspace))

    @classmethod
    def create(cls, output_root: str | Path = "output/multiview_refine", job_id: Optional[str] = None) -> "MultiViewRefinePipeline":
        return cls(create_workspace(output_root, job_id))

    def unpack_uploads(self, camera_images_zip: Optional[str | Path] = None, layer1_lam_zip: Optional[str | Path] = None) -> StepResult:
        if camera_images_zip:
            unpack_camera_images_zip(camera_images_zip, self.workspace)
        if layer1_lam_zip:
            unpack_layer1_lam_zip(layer1_lam_zip, self.workspace)
        report = validate_workspace_inputs(self.workspace, require_masks=False)
        debug_dir = self._save_workspace_preview("inputs")
        return StepResult(
            f"Workspace ready. Images: {report['num_images']}, masks: {report['num_masks']}, init_ply: {report['has_init_ply']}",
            str(debug_dir),
        )

    def import_inputs(
        self,
        image_dir: str | Path,
        mask_dir: Optional[str | Path] = None,
        flame_dir: Optional[str | Path] = None,
        colmap_dir: Optional[str | Path] = None,
        init_ply: Optional[str | Path] = None,
    ) -> StepResult:
        import_local_inputs(self.workspace, image_dir, mask_dir, flame_dir, colmap_dir, init_ply)
        report = validate_workspace_inputs(self.workspace, require_masks=False)
        debug_dir = self._save_workspace_preview("inputs")
        return StepResult(f"Imported. Images: {report['num_images']}, masks: {report['num_masks']}", str(debug_dir))

    def generate_masks_and_flame(
        self,
        alignment_model_path: str = "./model_zoo/flame_tracking_models/68_keypoints_model.pkl",
        vgghead_model_path: str = "./model_zoo/flame_tracking_models/vgghead/vgg_heads_l.trcd",
        human_matting_path: str = "./model_zoo/flame_tracking_models/matting/stylematte_synth.pt",
        facebox_model_path: str = "./model_zoo/flame_tracking_models/FaceBoxesV2.pth",
    ) -> StepResult:
        from tools.flame_tracking_single_image import FlameTrackingSingleImage

        images = sorted([p for p in self.workspace.images_dir.glob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"}])
        if not images:
            raise FileNotFoundError(f"No images found under {self.workspace.images_dir}")
        tracker = FlameTrackingSingleImage(
            output_dir=str(self.workspace.root / "tracking"),
            alignment_model_path=alignment_model_path,
            vgghead_model_path=vgghead_model_path,
            human_matting_path=human_matting_path,
            facebox_model_path=facebox_model_path,
            detect_iris_landmarks=False,
        )
        exports = []
        for image_path in images:
            code = tracker.preprocess(str(image_path))
            if code != 0:
                raise RuntimeError(f"FLAME preprocess failed for {image_path}")
            code = tracker.optimize()
            if code != 0:
                raise RuntimeError(f"FLAME optimize failed for {image_path}")
            code, export_dir = tracker.export()
            if code != 0:
                raise RuntimeError(f"FLAME export failed for {image_path}")
            exports.append(Path(export_dir))
        self._merge_tracking_exports(exports)
        validate_workspace_inputs(self.workspace, require_masks=True)
        debug_dir = self._save_workspace_preview("flame")
        return StepResult(f"Generated masks/FLAME for {len(exports)} views; tracking outputs restored to original image coordinates", str(debug_dir))

    def run_colmap(self, colmap_path: str = "colmap") -> StepResult:
        model_dir = run_colmap_pipeline(self.workspace.images_dir, self.workspace.colmap_dir, colmap_path=colmap_path)
        raw_json = import_colmap_sparse(model_dir, self.workspace.root / "transforms_colmap_sparse.json", colmap_path=colmap_path)
        transforms = write_lam_transforms_from_colmap(self.workspace.root, raw_json, out_name="transforms_colmap_raw.json")
        plot = self._save_colmap_visualization(transforms)
        return StepResult("COLMAP complete", str(plot))

    def import_colmap(self, sparse_dir: str | Path, colmap_path: str = "colmap") -> StepResult:
        raw_json = import_colmap_sparse(sparse_dir, self.workspace.root / "transforms_colmap_sparse.json", colmap_path=colmap_path)
        transforms = write_lam_transforms_from_colmap(self.workspace.root, raw_json, out_name="transforms_colmap_raw.json")
        plot = self._save_colmap_visualization(transforms)
        return StepResult("COLMAP imported", str(plot))

    def align_sim3(self, target_transforms: str | Path) -> StepResult:
        colmap_transforms = self.workspace.root / "transforms_colmap_raw.json"
        if not colmap_transforms.exists():
            raise FileNotFoundError("Missing transforms_colmap_raw.json. Run or import COLMAP first.")
        if not target_transforms:
            raise FileNotFoundError(
                "align_sim3 now requires explicit calibrated target transforms. "
                "Single-image FLAME tracking only provides masks/landmarks/FLAME params, not camera targets. "
                "Provide a calibrated target transforms JSON or use manual Sim3."
            )
        target_transforms = Path(target_transforms)
        sim3 = align_colmap_to_flame(
            colmap_transforms,
            target_transforms,
            self.workspace.root / "transforms_aligned.json",
            self.workspace.root / "sim3_colmap_to_lam.json",
        )
        plot = self._save_sim3_visualization(target_transforms)
        return StepResult(
            f"Sim3 estimated. scale={sim3.scale:.5f}, rmse={sim3.rmse:.5f}, inliers={sim3.inlier_count}/{sim3.total_count}",
            str(plot),
        )

    def write_manual_alignment(self, scale: float, yaw_degrees: float, tx: float, ty: float, tz: float) -> StepResult:
        sim3 = write_manual_sim3(self.workspace.root / "sim3_colmap_to_lam.json", scale, yaw_degrees, [tx, ty, tz])
        colmap_db = json.loads((self.workspace.root / "transforms_colmap_raw.json").read_text(encoding="utf-8"))
        from .alignment import apply_sim3_to_c2w

        for frame in colmap_db["frames"]:
            frame["transform_matrix_colmap"] = frame["transform_matrix"]
            frame["transform_matrix"] = apply_sim3_to_c2w(np.asarray(frame["transform_matrix"], dtype=np.float32), sim3).tolist()
        out = self.workspace.root / "transforms_aligned.json"
        out.write_text(json.dumps(colmap_db, indent=2), encoding="utf-8")
        plot = self._save_sim3_visualization(None)
        return StepResult("Manual Sim3 applied", str(plot))

    def preview_alignment(self, lam_model, init_ply: Optional[str | Path] = None, max_views: int = 8) -> StepResult:
        init_ply = Path(init_ply) if init_ply else self.workspace.root / "init.ply"
        if not init_ply.exists():
            raise FileNotFoundError(f"Missing init PLY: {init_ply}")
        batch = load_multiview_bundle(self.workspace.root).to("cuda", torch.float32)
        gs = _load_renderable_gaussian(init_ply)
        gs.to_cuda()
        query_points, flame_params = lam_model.renderer.get_query_points(batch.flame_params, device=batch.images.device)
        h, w = batch.images.shape[-2:]
        with torch.no_grad():
            out = render_animate_gs_with_intrinsics(lam_model.renderer, [gs], query_points, flame_params, batch.c2ws, batch.intrs, h, w, batch.bg_colors)
        saved = save_overlay_grid(
            self.workspace.debug_dir / "alignment",
            batch.frame_ids,
            batch.images.detach().cpu(),
            batch.masks.detach().cpu(),
            out["comp_rgb"].detach().cpu(),
            out["comp_mask"].detach().cpu(),
            batch.landmarks_2d.detach().cpu() if batch.landmarks_2d is not None else None,
            max_items=max_views,
        )
        return StepResult(f"Saved {len(saved)} preview overlays", str(self.workspace.debug_dir / "alignment"))

    def refine(self, lam_model, init_ply: Optional[str | Path] = None, config: Optional[RefinementConfig] = None, resume: Optional[str | Path] = None) -> StepResult:
        init_ply = Path(init_ply) if init_ply else self.workspace.root / "init.ply"
        if not init_ply.exists():
            raise FileNotFoundError(f"Missing init PLY: {init_ply}")
        config = config or RefinementConfig(output_dir=str(self.workspace.root))
        config.output_dir = str(self.workspace.root)
        batch = load_multiview_bundle(self.workspace.root, require_undistorted=config.require_undistorted)
        gs = _load_renderable_gaussian(init_ply)
        gs.to_cuda()
        refiner = MultiViewGaussianRefiner(lam_model, config)
        refiner.run(gs, batch, resume=resume)
        loss_plot = save_loss_plot(self.workspace.root / "loss_history.jsonl", self.workspace.debug_dir / "loss_history.png")
        return StepResult("Refinement complete", str(loss_plot or (self.workspace.root / "refined_gaussian.ply")))

    def export(self) -> StepResult:
        required = ["refined_gaussian.ply", "sim3_colmap_to_lam.json"]
        missing = [name for name in required if not (self.workspace.root / name).exists()]
        if missing:
            raise FileNotFoundError(f"Missing export files: {missing}")
        self.workspace.write_state(exported=True)
        return StepResult("Export ready", str(self.workspace.root))

    def _merge_tracking_exports(self, exports: list[Path]) -> None:
        for directory in [self.workspace.root / "fg_masks", self.workspace.flame_dir, self.workspace.landmark_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        images = sorted([p for p in self.workspace.images_dir.glob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"}])
        if len(exports) != len(images):
            raise ValueError(f"FLAME tracking exports/images mismatch: {len(exports)} exports for {len(images)} images")

        canonical_path = self.workspace.root / "canonical_flame_param.npz"
        if not canonical_path.exists():
            raise FileNotFoundError(
                f"Missing {canonical_path}. Import the Layer 1/LAM identity canonical_flame_param.npz before running FLAME tracking."
            )
        canonical_shape = _load_shape(canonical_path)
        if canonical_shape is None:
            raise KeyError(f"{canonical_path} must contain 'shape' or 'betas'")

        records = []
        for idx, export_dir in enumerate(exports):
            image_path = images[idx]
            stem = image_path.stem if idx < len(images) else f"{idx:05d}"
            meta = _load_preprocess_meta(self.workspace.root / "tracking" / "preprocess" / stem / "preprocess_meta.json", image_path)
            mask_src = next((export_dir / "fg_masks").glob("*"))
            flame_src = next((export_dir / "flame_param").glob("*.npz"))
            mask_dst = self.workspace.root / "fg_masks" / f"{stem}.png"
            flame_dst = self.workspace.flame_dir / f"{stem}.npz"
            landmark_dst = self.workspace.landmark_dir / f"{stem}.npz"
            _restore_tracking_mask(mask_src, image_path, meta, mask_dst)
            _write_tracking_flame_param(flame_src, flame_dst, canonical_shape)
            if (export_dir / "landmark2d" / "landmarks.npz").exists():
                _restore_tracking_landmarks(export_dir / "landmark2d" / "landmarks.npz", image_path, meta, landmark_dst)
            records.append({
                "image_name": image_path.name,
                "file_path": f"images/{image_path.name}",
                "fg_mask_path": f"fg_masks/{stem}.png",
                "flame_param_path": f"flame_param/{stem}.npz",
                "landmark_path": f"landmark2d/{stem}.npz" if landmark_dst.exists() else None,
                "tracking_export": str(export_dir),
            })

        (self.workspace.root / "flame_tracking_manifest.json").write_text(
            json.dumps(
                {
                    "coordinate_roles": {
                        "images": "original multi-view image pixels used by COLMAP and refinement",
                        "masks_landmarks": "restored from FLAME tracking crop coordinates into original image pixels",
                        "flame_params": "per-view expression/jaw/eyes and local pose cues in LAM FLAME parameter format",
                        "flame_global_pose": "canonicalized to zero rotation/translation",
                        "cameras": "not provided by single-image FLAME tracking; use COLMAP plus explicit/manual alignment",
                        "target_model": "LAM canonical FLAME-bound Gaussian space",
                    },
                    "shape_source": "layer1_or_imported",
                    "frames": records,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def _save_workspace_preview(self, subdir: str) -> Path:
        out_dir = self.workspace.debug_dir / subdir
        save_workspace_image_previews(self.workspace.root, out_dir)
        return out_dir

    def _save_colmap_visualization(self, transforms: str | Path) -> Path:
        out = self.workspace.debug_dir / "colmap" / "camera_centers.png"
        save_transform_camera_plot(out, {"colmap": transforms}, title="COLMAP camera centers")
        return out

    def _save_sim3_visualization(self, flame_target: Optional[str | Path]) -> Path:
        tracks = {
            "colmap raw": self.workspace.root / "transforms_colmap_raw.json",
            "aligned": self.workspace.root / "transforms_aligned.json",
        }
        if flame_target:
            tracks["flame target"] = flame_target
        out = self.workspace.debug_dir / "sim3" / "alignment_camera_centers.png"
        save_transform_camera_plot(out, tracks, title="Sim3 camera alignment")
        return out


def _load_renderable_gaussian(path: str | Path) -> GaussianModel:
    gs = GaussianModel(ply_path=str(path), sh2rgb=False)
    # GaussianModel.save_ply stores opacity as logit and scale as log. The renderer
    # consumes activated opacity/scale tensors directly.
    gs.opacity = torch.sigmoid(gs.opacity)
    gs.scaling = torch.exp(gs.scaling)
    return gs


def _load_shape(path: Path) -> Optional[np.ndarray]:
    if not path.exists():
        return None
    raw = np.load(path, allow_pickle=True)
    if "shape" in raw:
        return np.asarray(raw["shape"])
    if "betas" in raw:
        return np.asarray(raw["betas"])
    return None


def _load_preprocess_meta(path: Path, image_path: Path) -> dict:
    with Image.open(image_path) as image:
        width, height = image.size
    if not path.exists():
        raise FileNotFoundError(f"Missing FLAME tracking preprocess metadata: {path}")
    meta = json.loads(path.read_text(encoding="utf-8"))
    if "crop_bbox_xyxy" not in meta:
        raise KeyError(f"{path} must contain crop_bbox_xyxy")
    original_size = meta.get("original_size")
    if original_size and [int(original_size[0]), int(original_size[1])] != [width, height]:
        raise ValueError(f"Tracking metadata size {original_size} does not match source image size {[width, height]} for {image_path.name}")
    return meta


def _restore_tracking_mask(mask_src: Path, image_path: Path, meta: dict, output_path: Path) -> None:
    with Image.open(image_path) as image:
        width, height = image.size
    x0, y0, x1, y1 = _crop_bbox(meta, width, height)
    crop_w = max(1, x1 - x0)
    crop_h = max(1, y1 - y0)
    crop_mask = Image.open(mask_src).convert("L").resize((crop_w, crop_h), Image.Resampling.NEAREST)
    canvas = Image.new("L", (width, height), 0)
    paste_x0, paste_y0 = max(0, x0), max(0, y0)
    paste_x1, paste_y1 = min(width, x1), min(height, y1)
    if paste_x1 > paste_x0 and paste_y1 > paste_y0:
        src_x0, src_y0 = paste_x0 - x0, paste_y0 - y0
        region = crop_mask.crop((src_x0, src_y0, src_x0 + paste_x1 - paste_x0, src_y0 + paste_y1 - paste_y0))
        canvas.paste(region, (paste_x0, paste_y0))
    canvas.save(output_path)


def _restore_tracking_landmarks(landmark_src: Path, image_path: Path, meta: dict, output_path: Path) -> None:
    raw = dict(np.load(landmark_src, allow_pickle=True))
    if "face_landmark_2d" not in raw:
        return
    with Image.open(image_path) as image:
        width, height = image.size
    x0, y0, x1, y1 = _crop_bbox(meta, width, height)
    crop_w = max(1, x1 - x0)
    crop_h = max(1, y1 - y0)
    landmarks = np.asarray(raw["face_landmark_2d"], dtype=np.float32).copy()
    landmarks[..., 0] = landmarks[..., 0] * crop_w + x0
    landmarks[..., 1] = landmarks[..., 1] * crop_h + y0
    if landmarks.shape[-1] > 2:
        visible = np.isfinite(landmarks[..., :2]).all(axis=-1)
        landmarks[..., 2] = np.where(visible, 1.0, 0.0)
    raw["face_landmark_2d"] = landmarks
    np.savez(output_path, **raw)


def _write_tracking_flame_param(
    flame_src: Path,
    output_path: Path,
    canonical_shape: np.ndarray,
) -> None:
    raw = dict(np.load(flame_src, allow_pickle=True))
    if canonical_shape is not None:
        raw["shape"] = canonical_shape
        raw["betas"] = canonical_shape
    for key in ["rotation", "translation"]:
        if key in raw:
            raw[f"{key}_tracking"] = np.asarray(raw[key]).copy()
            raw[key] = np.zeros_like(raw[key])
    np.savez(output_path, **raw)


def _crop_bbox(meta: dict, width: int, height: int) -> tuple[int, int, int, int]:
    bbox = meta.get("crop_bbox_xyxy") or [0, 0, width, height]
    x0, y0, x1, y1 = [int(round(float(v))) for v in bbox]
    if x1 <= x0 or y1 <= y0:
        return 0, 0, width, height
    return x0, y0, x1, y1
