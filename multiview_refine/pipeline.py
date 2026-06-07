import json
import shutil
import zipfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import cv2
from PIL import Image, ImageDraw, ImageFont

from lam.models.rendering.gaussian_model import GaussianModel

from .alignment import Sim3Alignment, apply_sim3_to_c2w, estimate_sim3_umeyama_ransac
from .colmap import assess_colmap_model_quality, import_colmap_sparse, run_colmap_pipeline
from .data import load_frames, load_multiview_bundle, write_lam_transforms_from_colmap
from .optimization import (
    MultiViewGaussianRefiner,
    OptimizableCameraState,
    OptimizableExposureState,
    OptimizableExpressionState,
    OptimizableGaussianState,
    OptimizableIntrinsicsState,
    RefinementConfig,
    RefinementStageConfig,
    RoundRobinViewSampler,
    compute_losses,
    _index_flame_params,
    _landmark_loss,
    _project_flame_landmarks,
    _scale_batch,
)
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
    find_stem_file,
    import_local_inputs,
    numeric_stem_key,
    unpack_camera_images_zip,
    unpack_layer1_lam_zip,
    validate_workspace_inputs,
)
from .types import TensorDict


@dataclass
class StepResult:
    message: str
    path: str = ""


INITIAL_SIM3_SCALE_MULTIPLIERS = (0.35, 0.5, 0.7, 0.85, 1.0, 1.3, 1.7, 2.2, 2.8, 3.6, 4.6, 6.0, 8.0)
LANDMARK_CALIBRATE_SCALE_MIN = 0.25
LANDMARK_CALIBRATE_SCALE_MAX = 4.0
PNP_LANDMARK_INDICES = tuple(range(17, 68))
INITIAL_PNP_ACCEPT_MAX_LANDMARK_PX = 480.0
INITIAL_PNP_ACCEPT_MIN_COVERAGE = 0.75
INITIAL_PNP_ACCEPT_MAX_CENTER_ERROR_PX = 0.30
ALIGNMENT_INLIER_MIN_VIEWS = 6
ALIGNMENT_INLIER_MIN_FRACTION = 0.60


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
        mask_note = ""
        if report["num_masks"] and not report["missing_masks"]:
            copied = self._canonicalize_workspace_mask_names()
            mask_note = " Uploaded masks were kept without pixel postprocessing."
            if copied:
                mask_note += f" Copied {copied} numeric-name mask aliases."
            report = validate_workspace_inputs(self.workspace, require_masks=False)
        debug_dir = self._save_workspace_preview("00_inputs")
        return StepResult(
            f"Workspace ready. Images: {report['num_images']}, masks: {report['num_masks']}, init_ply: {report['has_init_ply']}.{mask_note}",
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
        debug_dir = self._save_workspace_preview("00_inputs")
        return StepResult(f"Imported. Images: {report['num_images']}, masks: {report['num_masks']}", str(debug_dir))

    def import_and_process_masks(
        self,
        mask_dir: str | Path,
        keep_largest_component: bool = True,
        close_radius: int = 9,
        feather_radius: int = 3,
    ) -> StepResult:
        mask_dir = Path(mask_dir)
        if not mask_dir.exists():
            raise FileNotFoundError(f"Missing mask dir: {mask_dir}")
        images = sorted([p for p in self.workspace.images_dir.glob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"}])
        if not images:
            raise FileNotFoundError(f"No images found under {self.workspace.images_dir}")
        self.workspace.masks_dir.mkdir(parents=True, exist_ok=True)
        processed = 0
        missing = []
        for image_path in images:
            mask_path = _find_mask_by_stem(mask_dir, image_path.stem)
            if mask_path is None:
                missing.append(image_path.name)
                continue
            out_path = self.workspace.masks_dir / f"{image_path.stem}.png"
            _process_external_mask(
                mask_path,
                image_path,
                out_path,
                keep_largest_component=keep_largest_component,
                close_radius=close_radius,
                feather_radius=feather_radius,
            )
            processed += 1
        if missing:
            raise FileNotFoundError(f"Missing masks for images: {missing[:10]}")
        validate_workspace_inputs(self.workspace, require_masks=True)
        debug_dir = self._save_workspace_preview("01_masks_imported")
        return StepResult(f"Imported and processed {processed} masks", str(debug_dir))

    def restore_uploaded_masks(self) -> StepResult:
        images = sorted([p for p in self.workspace.images_dir.glob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"}])
        if not images:
            raise FileNotFoundError(f"No images found under {self.workspace.images_dir}")
        image_by_stem = {path.stem: path for path in images}
        image_by_numeric_stem = {}
        for path in images:
            key = numeric_stem_key(path.stem)
            if key is not None:
                image_by_numeric_stem.setdefault(key, []).append(path)
        zip_paths = sorted(self.workspace.inputs_dir.glob("*.zip"), key=lambda path: path.stat().st_mtime, reverse=True)
        if not zip_paths:
            raise FileNotFoundError(f"No uploaded ZIP files found under {self.workspace.inputs_dir}")

        restored = 0
        seen = set()
        self.workspace.masks_dir.mkdir(parents=True, exist_ok=True)
        for zip_path in zip_paths:
            with zipfile.ZipFile(zip_path, "r") as zipf:
                for member in zipf.infolist():
                    path = Path(member.filename)
                    if member.is_dir() or path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
                        continue
                    if not any(part in {"fg_masks", "masks"} for part in path.parts):
                        continue
                    image_path = image_by_stem.get(path.stem)
                    if image_path is None:
                        numeric_matches = image_by_numeric_stem.get(numeric_stem_key(path.stem), [])
                        image_path = numeric_matches[0] if len(numeric_matches) == 1 else None
                    if image_path is None or image_path.stem in seen:
                        continue
                    with zipf.open(member) as fp:
                        with Image.open(fp) as mask:
                            mask = mask.convert("L")
                            with Image.open(image_path) as image:
                                width, height = image.size
                            if mask.size != (width, height):
                                mask = mask.resize((width, height), Image.Resampling.NEAREST)
                            mask.save(self.workspace.masks_dir / f"{image_path.stem}.png")
                    seen.add(image_path.stem)
                    restored += 1
            if restored == len(images):
                break
        missing = sorted(set(image_by_stem) - seen)
        if missing:
            raise FileNotFoundError(f"Could not restore uploaded masks for image stems: {missing[:10]}")
        debug_dir = self._save_workspace_preview("01_uploaded_masks_restored")
        return StepResult(f"Restored {restored} uploaded masks from input ZIP", str(debug_dir))

    def _canonicalize_workspace_mask_names(self) -> int:
        images = sorted([p for p in self.workspace.images_dir.glob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"}])
        copied = 0
        for image_path in images:
            exact = find_stem_file(self.workspace.masks_dir, image_path.stem, [".png", ".jpg", ".jpeg"])
            if exact is not None and exact.stem == image_path.stem:
                continue
            mask_path = _find_mask_by_stem(self.workspace.masks_dir, image_path.stem)
            if mask_path is None:
                continue
            out_path = self.workspace.masks_dir / f"{image_path.stem}{mask_path.suffix.lower()}"
            if not out_path.exists():
                shutil.copy2(mask_path, out_path)
                copied += 1
        return copied

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
            output_dir=str(self.workspace.tracking_dir),
            alignment_model_path=alignment_model_path,
            vgghead_model_path=vgghead_model_path,
            human_matting_path=human_matting_path,
            facebox_model_path=facebox_model_path,
            detect_iris_landmarks=False,
        )
        exports = []
        for idx, image_path in enumerate(images, start=1):
            code = tracker.preprocess(str(image_path))
            if code != 0:
                raise RuntimeError(f"FLAME preprocess failed for view {idx}/{len(images)}: {image_path.name}")
            code = tracker.optimize()
            if code != 0:
                raise RuntimeError(f"FLAME optimize failed for view {idx}/{len(images)}: {image_path.name}")
            code, export_dir = tracker.export()
            if code != 0:
                raise RuntimeError(f"FLAME export failed for view {idx}/{len(images)}: {image_path.name}")
            exports.append(Path(export_dir))
        preserved_masks = self._merge_tracking_exports(exports, preserve_existing_masks=True)
        validate_workspace_inputs(self.workspace, require_masks=True)
        debug_dir = self._save_workspace_preview("01_flame_tracking")
        return StepResult(
            f"Generated FLAME tracking for {len(exports)} views; preserved {preserved_masks} existing masks",
            str(debug_dir),
        )

    def run_colmap(self, colmap_path: str = "colmap") -> StepResult:
        model_dir = run_colmap_pipeline(
            self.workspace.images_dir,
            self.workspace.colmap_dir,
            colmap_path=colmap_path,
            use_masks=False,
        )
        raw_json = import_colmap_sparse(model_dir, self.workspace.colmap_dir / "transforms_colmap_sparse.json", colmap_path=colmap_path)
        transforms = write_lam_transforms_from_colmap(self.workspace.root, raw_json, out_name="colmap/transforms_colmap_raw.json")
        _assert_colmap_reconstruction_usable(self.workspace)
        plot = self._save_colmap_visualization(transforms)
        return StepResult(_colmap_step_message("COLMAP complete", raw_json), str(plot))

    def import_colmap(self, sparse_dir: str | Path, colmap_path: str = "colmap") -> StepResult:
        raw_json = import_colmap_sparse(sparse_dir, self.workspace.colmap_dir / "transforms_colmap_sparse.json", colmap_path=colmap_path)
        transforms = write_lam_transforms_from_colmap(self.workspace.root, raw_json, out_name="colmap/transforms_colmap_raw.json")
        _assert_colmap_reconstruction_usable(self.workspace)
        plot = self._save_colmap_visualization(transforms)
        return StepResult(_colmap_step_message("COLMAP imported", raw_json), str(plot))

    def initialize_sim3_from_layer1(self, lam_model=None) -> StepResult:
        if not self.workspace.colmap_transforms_path.exists():
            raise FileNotFoundError("Missing transforms_colmap_raw.json. Run or import COLMAP first.")
        if not self.workspace.init_ply_path.exists():
            raise FileNotFoundError("Missing Layer 1 canonical Gaussian data/init.ply.")
        colmap_db = json.loads(self.workspace.colmap_transforms_path.read_text(encoding="utf-8"))
        frames = sorted(colmap_db.get("frames", []), key=lambda frame: frame.get("image_name") or frame.get("file_path") or "")
        colmap_centers = []
        colmap_mats = []
        for frame in frames:
            mat = _json_c2w_to_render_c2w(np.asarray(frame["transform_matrix"], dtype=np.float32))
            if mat.shape == (4, 4):
                colmap_centers.append(mat[:3, 3])
                colmap_mats.append(mat)
        if not colmap_centers:
            raise FileNotFoundError("No COLMAP camera centers found. Run or import COLMAP first.")
        colmap_centers = np.stack(colmap_centers)
        source_forwards = np.stack([_camera_forward(mat) for mat in colmap_mats], axis=0)
        ray_scene_center, look_at_rmse = _estimate_camera_ray_intersection(colmap_centers, source_forwards)
        source_scene_center, source_anchor_report = _choose_colmap_scene_anchor(
            self.workspace,
            colmap_mats,
            ray_scene_center,
            look_at_rmse,
        )
        colmap_center = np.median(colmap_centers, axis=0).astype(np.float64)
        colmap_radius = float(np.median(np.linalg.norm(colmap_centers - source_scene_center, axis=1)))
        if colmap_radius < 1e-6:
            raise ValueError("COLMAP camera centers are degenerate; cannot estimate scale.")

        layer1_stats = _load_layer1_alignment_stats(self.workspace)
        target_center = layer1_stats["center"]
        reference_c2w = layer1_stats.get("reference_c2w")
        pnp_candidate = _estimate_initial_sim3_from_landmark_pnp(self.workspace, lam_model)
        if pnp_candidate is not None and _should_accept_initial_pnp_candidate(self.workspace, pnp_candidate):
            return self._write_initial_sim3_result(
                colmap_db=colmap_db,
                sim3=pnp_candidate["sim3"],
                colmap_center=colmap_center,
                ray_scene_center=ray_scene_center,
                look_at_rmse=look_at_rmse,
                source_anchor_report=source_anchor_report,
                source_anchor_center=source_scene_center,
                colmap_radius=colmap_radius,
                layer1_stats=layer1_stats,
                target_center=target_center,
                reference_center=None,
                reference_distance=None,
                fallback_distance=max(float(layer1_stats["radius_p95"]) * 5.0, 0.8),
                target_anchor_center=target_center,
                target_distance=pnp_candidate["target_distance"],
                rotation_policy=pnp_candidate["policy"],
                projection_score=pnp_candidate["projection_score"],
                landmark_score=pnp_candidate["landmark_score"],
                scale_candidates=None,
            )
        if reference_c2w is not None:
            reference_c2w = _json_c2w_to_render_c2w(reference_c2w)
            reference_center = reference_c2w[:3, 3]
            reference_distance = float(np.linalg.norm(reference_center - target_center))
        else:
            reference_center = None
            reference_distance = 0.0
        fallback_distance = max(float(layer1_stats["radius_p95"]) * 5.0, 0.8)
        target_distance = reference_distance if reference_distance > 1e-6 else fallback_distance
        if target_distance < 1e-6:
            target_distance = fallback_distance
        base_scale = float(target_distance / colmap_radius)

        if reference_c2w is not None:
            source = "layer1_reference_initialization"
        else:
            reference_center = target_center + np.array([0.0, 0.0, target_distance], dtype=np.float64)
            source = "layer1_first_view_front_initialization"
        source_anchor_center = source_scene_center
        target_anchor_center = target_center

        intrinsics = [_intrinsic_from_frame(frame) for frame in frames]
        scale_search = _build_initial_scale_candidates(
            base_scale=base_scale,
            colmap_radius=colmap_radius,
            fallback_distance=fallback_distance,
            reference_distance=reference_distance,
        )
        landmark_context = _prepare_initial_landmark_context(self.workspace, lam_model)
        alignment_candidate = _choose_initial_sim3_candidate(
            self.workspace,
            lam_model,
            landmark_context,
            colmap_mats,
            intrinsics,
            source_scene_center,
            source_anchor_center,
            target_anchor_center,
            target_center,
            scale_search,
            reference_c2w,
            target_distance,
        )
        scale = float(alignment_candidate["scale"])
        target_distance = float(scale * colmap_radius)
        R_row = alignment_candidate["rotation_row"]
        translation = alignment_candidate["translation"]
        sim3 = Sim3Alignment(
            scale=scale,
            rotation=R_row.astype(np.float32).tolist(),
            translation=translation.astype(np.float32).tolist(),
            rmse=-1.0,
            source=source,
            inlier_count=len(colmap_centers),
            total_count=len(colmap_centers),
            inlier_rmse=None,
            inlier_names=[Path(frame.get("image_name") or frame.get("file_path") or "").name for frame in frames],
        )
        return self._write_initial_sim3_result(
            colmap_db=colmap_db,
            sim3=sim3,
            colmap_center=colmap_center,
            ray_scene_center=ray_scene_center,
            look_at_rmse=look_at_rmse,
            source_anchor_report=source_anchor_report,
            source_anchor_center=source_anchor_center,
            colmap_radius=colmap_radius,
            layer1_stats=layer1_stats,
            target_center=target_center,
            reference_center=reference_center,
            reference_distance=reference_distance if reference_distance > 1e-6 else None,
            fallback_distance=fallback_distance,
            target_anchor_center=target_anchor_center,
            target_distance=target_distance,
            rotation_policy=alignment_candidate["policy"],
            projection_score=alignment_candidate["score"],
            landmark_score=alignment_candidate.get("landmark_score"),
            scale_candidates=scale_search,
        )

    def align_cameras(self, lam_model, init_ply: Optional[str | Path] = None) -> StepResult:
        init_result = self.initialize_sim3_from_layer1(lam_model)
        calibrate_result = self.calibrate_global_sim3_blackbox(lam_model, init_ply=init_ply)
        per_view_result = self.align_per_view_cameras(lam_model, init_ply=init_ply)
        message = f"{init_result.message}\n{calibrate_result.message}\n{per_view_result.message}"
        path = per_view_result.path or calibrate_result.path or init_result.path
        return StepResult(message=message, path=path)

    def calibrate_global_sim3_blackbox(self, lam_model, init_ply: Optional[str | Path] = None) -> StepResult:
        init_ply = Path(init_ply) if init_ply else self.workspace.init_ply_path
        if not init_ply.exists():
            raise FileNotFoundError(f"Missing init PLY: {init_ply}")
        if not self.workspace.aligned_transforms_path.exists() or not self.workspace.sim3_path.exists():
            raise FileNotFoundError("Missing initial Sim3 alignment. Run Initialize Sim3 first.")

        restored = _ensure_workspace_landmarks(self.workspace)
        before_score = _score_workspace_geometry(self.workspace)
        batch = load_multiview_bundle(self.workspace.root, require_undistorted=False).to("cuda", torch.float32)
        query_points, flame_params = lam_model.renderer.get_query_points(batch.flame_params, device=batch.images.device)
        coarse_seed = _landmark_coarse_seed(lam_model.renderer, query_points, flame_params, batch, self.workspace)
        calib_batch = batch
        calib_flame_params = flame_params
        inlier_view_indices = coarse_seed.get("inlier_view_indices") or []
        if len(inlier_view_indices) >= ALIGNMENT_INLIER_MIN_VIEWS and len(inlier_view_indices) < batch.c2ws.shape[1]:
            calib_batch = batch.index(inlier_view_indices)
            calib_flame_params = _index_flame_params(flame_params, calib_batch.view_indices)
        candidate = _landmark_calibrate_sim3(
            lam_model.renderer,
            query_points,
            calib_flame_params,
            calib_batch,
            self.workspace,
            init_scale=coarse_seed["scale"],
            init_rotation=coarse_seed["rotation"],
            init_translation=coarse_seed["translation"],
        )
        accepted, reason, after_score = _validate_landmark_calibration_delta(
            self.workspace,
            before_score,
            candidate,
        )
        if not accepted:
            _append_landmark_calibration_acceptance(self.workspace, accepted=False, reason=reason, before=before_score, after=after_score)
            plot = self._save_sim3_visualization(None)
            suffix = f"; restored {restored} landmark files" if restored else ""
            return StepResult(
                f"Landmark Sim3 calibration skipped: {reason}{suffix}. Keeping initial alignment.",
                str(plot),
            )
        _bake_render_delta_into_alignment(self.workspace, candidate["scale"], candidate["rotation"], candidate["translation"], "landmark_calibrated")
        _write_projection_diagnostics(self.workspace, lam_model=lam_model)
        _append_landmark_calibration_acceptance(self.workspace, accepted=True, reason=reason, before=before_score, after=after_score)
        plot = self._save_sim3_visualization(None)
        report = self.workspace.alignment_dir / "landmark_calibration_report.json"
        suffix = f"; restored {restored} landmark files" if restored else ""
        if calib_batch.c2ws.shape[1] != batch.c2ws.shape[1]:
            suffix += f"; calibrated on {calib_batch.c2ws.shape[1]}/{batch.c2ws.shape[1]} inlier views"
        note = "" if reason == "accepted" else f" ({reason})"
        return StepResult(f"Landmark Sim3 calibrated{note}. loss={candidate['loss']:.5f}{suffix}", str(plot if plot else report))

    def preview_alignment(self, lam_model, init_ply: Optional[str | Path] = None, max_views: int = 8) -> StepResult:
        init_ply = Path(init_ply) if init_ply else self.workspace.init_ply_path
        if not init_ply.exists():
            raise FileNotFoundError(f"Missing init PLY: {init_ply}")
        batch = load_multiview_bundle(self.workspace.root, require_undistorted=False).to("cuda", torch.float32)
        gs = _load_renderable_gaussian(init_ply)
        gs.to_cuda()
        query_points, flame_params = lam_model.renderer.get_query_points(batch.flame_params, device=batch.images.device)
        h, w = batch.images.shape[-2:]
        with torch.no_grad():
            out = render_animate_gs_with_intrinsics(lam_model.renderer, [gs], query_points, flame_params, batch.c2ws, batch.intrs, h, w, batch.bg_colors)
        saved = save_overlay_grid(
            self.workspace.debug_dir / "04_preview",
            batch.frame_ids,
            batch.images.detach().cpu(),
            batch.masks.detach().cpu(),
            out["comp_rgb"].detach().cpu(),
            out["comp_mask"].detach().cpu(),
            batch.landmarks_2d.detach().cpu() if batch.landmarks_2d is not None else None,
            max_items=max_views,
        )
        return StepResult(f"Saved {len(saved)} preview overlays", str(self.workspace.debug_dir / "04_preview"))

    def align_per_view_cameras(self, lam_model, init_ply: Optional[str | Path] = None) -> StepResult:
        init_ply = Path(init_ply) if init_ply else self.workspace.init_ply_path
        if not init_ply.exists():
            raise FileNotFoundError(f"Missing init PLY: {init_ply}")
        if not self.workspace.aligned_transforms_path.exists():
            raise FileNotFoundError("Missing aligned transforms. Run Align Cameras first.")

        result = _optimize_per_view_camera_alignment(self.workspace, lam_model, init_ply)
        _bake_per_view_camera_delta_into_alignment(self.workspace, result["camera_state"], result["intr_state"])
        _write_projection_diagnostics(self.workspace, lam_model=lam_model)
        plot = self._save_sim3_visualization(None)
        report_path = self.workspace.alignment_dir / "per_view_camera_alignment_report.json"
        return StepResult(
            "Per-view camera/intrinsics alignment finished. "
            f"inlier landmark={result['best_landmark_px_inliers']:.1f}px, "
            f"inlier face-box center={result['best_face_box_inliers'].get('center_px_median', float('nan')):.1f}px, "
            f"inlier size_rel={result['best_face_box_inliers'].get('size_rel_median', float('nan')):.3f}, "
            f"train views={result['used_view_count']}/{result['total_view_count']}, "
            f"max focal delta={result['intrinsics_summary'].get('max_focal_percent', float('nan')):.1f}%",
            str(plot if plot else report_path),
        )

    def refine(self, lam_model, init_ply: Optional[str | Path] = None, config: Optional[RefinementConfig] = None, resume: Optional[str | Path] = None) -> StepResult:
        init_ply = Path(init_ply) if init_ply else self.workspace.init_ply_path
        if not init_ply.exists():
            raise FileNotFoundError(f"Missing init PLY: {init_ply}")
        _assert_projection_not_empty(self.workspace)
        _assert_alignment_ready_for_refine(self.workspace)
        config = config or RefinementConfig(output_dir=str(self.workspace.refine_dir))
        config.output_dir = str(self.workspace.refine_dir)
        refine_view_selection = _load_refine_train_view_selection(self.workspace)
        if (
            config.train_view_indices is None
            and refine_view_selection["selected_view_indices"]
            and len(refine_view_selection["selected_view_indices"]) < refine_view_selection["num_views"]
        ):
            config.train_view_indices = list(refine_view_selection["selected_view_indices"])
        batch = load_multiview_bundle(self.workspace.root, require_undistorted=config.require_undistorted)
        gs = _load_renderable_gaussian(init_ply)
        gs.to_cuda()
        refiner = MultiViewGaussianRefiner(lam_model, config)
        refiner.run(gs, batch, resume=resume)
        loss_plot = save_loss_plot(self.workspace.loss_history_path, self.workspace.debug_dir / "05_refine" / "loss_history.png")
        subset_note = ""
        if config.train_view_indices is not None and refine_view_selection["num_views"] > 0:
            subset_note = (
                f" using {len(config.train_view_indices)}/{refine_view_selection['num_views']} "
                f"alignment inlier views ({refine_view_selection['source']})"
            )
        return StepResult(f"Refinement complete{subset_note}", str(loss_plot or self.workspace.refined_gaussian_path))

    def export(self) -> StepResult:
        required = [self.workspace.refined_gaussian_path, self.workspace.sim3_path, self.workspace.aligned_transforms_path, self.workspace.canonical_flame_path]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Missing export files: {missing}")
        self.workspace.exports_dir.mkdir(parents=True, exist_ok=True)
        package_path = self.workspace.exports_dir / "package.zip"
        if package_path.exists():
            package_path.unlink()
        _write_baked_export_alignment(self.workspace)
        optional = [
            self.workspace.alignment_dir / "initial_sim3_report.json",
            self.workspace.layer1_metadata_path,
            self.workspace.layer1_reference_transforms_path,
            self.workspace.refine_dir / "camera_delta.pt",
            self.workspace.refine_dir / "intrinsics_delta.pt",
            self.workspace.refine_dir / "exposure_delta.pt",
            self.workspace.refine_dir / "pose_delta.pt",
            self.workspace.refine_dir / "gaussian_geometry_delta.pt",
            self.workspace.refine_dir / "refine_config.json",
            self.workspace.loss_history_path,
        ]
        export_required = [self.workspace.refined_gaussian_path, self.workspace.canonical_flame_path]
        for src in export_required + [path for path in optional if path.exists()]:
            shutil.copy2(src, self.workspace.exports_dir / src.name)
        flame_export_dir = self.workspace.exports_dir / "data" / "flame_param"
        if flame_export_dir.exists():
            shutil.rmtree(flame_export_dir)
        flame_files = sorted(self.workspace.flame_dir.glob("*.npz")) if self.workspace.flame_dir.exists() else []
        if flame_files:
            flame_export_dir.mkdir(parents=True, exist_ok=True)
            for src in flame_files:
                shutil.copy2(src, flame_export_dir / src.name)
        for src in [self.workspace.sim3_path, self.workspace.aligned_transforms_path]:
            shutil.copy2(src, self.workspace.exports_dir / f"{src.stem}.initial{src.suffix}")
        with zipfile.ZipFile(package_path, "w", compression=zipfile.ZIP_DEFLATED) as zipf:
            for path in sorted(self.workspace.exports_dir.rglob("*")):
                if not path.is_file() or path == package_path:
                    continue
                rel_path = path.relative_to(self.workspace.exports_dir)
                if rel_path.parts and rel_path.parts[0] == "final_review":
                    continue
                if path.name == "final_review.zip":
                    continue
                zipf.write(path, arcname=str(rel_path).replace("\\", "/"))
        return StepResult("Export ready", str(self.workspace.exports_dir))

    def export_final_review(self, lam_model, fps: int = 6, max_views: int = 24) -> StepResult:
        if not self.workspace.refined_gaussian_path.exists():
            raise FileNotFoundError(f"Missing refined Gaussian: {self.workspace.refined_gaussian_path}")

        out_dir = self.workspace.exports_dir / "final_review"
        overlays_dir = out_dir / "overlays"
        if out_dir.exists():
            shutil.rmtree(out_dir)
        overlays_dir.mkdir(parents=True, exist_ok=True)

        batch = load_multiview_bundle(self.workspace.root, require_undistorted=False).to("cuda", torch.float32)
        gs = _load_renderable_gaussian(self.workspace.refined_gaussian_path)
        gs.to_cuda()
        camera_delta = _load_torch_state(self.workspace.refine_dir / "camera_delta.pt")
        intrinsics_delta = _load_torch_state(self.workspace.refine_dir / "intrinsics_delta.pt")
        pose_delta = _load_torch_state(self.workspace.refine_dir / "pose_delta.pt")
        batch = _apply_refine_deltas_to_batch(
            batch,
            camera_delta=camera_delta,
            intrinsics_delta=intrinsics_delta,
            pose_delta=pose_delta,
        )
        query_points, flame_params = lam_model.renderer.get_query_points(batch.flame_params, device=batch.images.device)
        h, w = batch.images.shape[-2:]
        with torch.no_grad():
            out = render_animate_gs_with_intrinsics(lam_model.renderer, [gs], query_points, flame_params, batch.c2ws, batch.intrs, h, w, batch.bg_colors)

        saved = save_overlay_grid(
            overlays_dir,
            batch.frame_ids,
            batch.images.detach().cpu(),
            batch.masks.detach().cpu(),
            out["comp_rgb"].detach().cpu(),
            out["comp_mask"].detach().cpu(),
            batch.landmarks_2d.detach().cpu() if batch.landmarks_2d is not None else None,
            max_items=max_views,
        )

        rgb = out["comp_rgb"][0].detach().clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()
        mask = out["comp_mask"][0].detach().clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()
        frames = (rgb * mask + (1.0 - mask) * 1.0).clip(0, 1)
        frames = (frames * 255.0).astype(np.uint8)
        video_path = out_dir / "final_review.mp4"
        from lam.utils.video import images_to_video
        images_to_video(frames, str(video_path), int(fps), gradio_codec=True)

        zip_path = self.workspace.exports_dir / "final_review.zip"
        if zip_path.exists():
            zip_path.unlink()
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zipf:
            for path in sorted(out_dir.rglob("*")):
                if path.is_file():
                    zipf.write(path, arcname=str(path.relative_to(out_dir)).replace("\\", "/"))

        applied = {
            "camera_delta": camera_delta is not None,
            "intrinsics_delta": intrinsics_delta is not None,
            "pose_delta": pose_delta is not None,
        }
        (out_dir / "final_review_report.json").write_text(
            json.dumps(
                {
                    "applied_refine_deltas": applied,
                    "num_views": int(batch.c2ws.shape[1]),
                    "max_views": int(max_views),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        delta_note = ", ".join([key for key, enabled in applied.items() if enabled]) or "none"
        return StepResult(f"Final review ready. Overlays: {len(saved)}. Applied refine deltas: {delta_note}", str(out_dir))

    def export_pose_sweep_review(self, lam_model, fps: int = 6, max_views: int = 9) -> StepResult:
        if not self.workspace.refined_gaussian_path.exists():
            raise FileNotFoundError(f"Missing refined Gaussian: {self.workspace.refined_gaussian_path}")

        out_dir = self.workspace.exports_dir / "pose_sweep_review"
        frames_dir = out_dir / "frames"
        if out_dir.exists():
            shutil.rmtree(out_dir)
        frames_dir.mkdir(parents=True, exist_ok=True)

        batch = load_multiview_bundle(self.workspace.root, require_undistorted=False).to("cuda", torch.float32)
        gs = _load_renderable_gaussian(self.workspace.refined_gaussian_path)
        gs.to_cuda()
        camera_delta = _load_torch_state(self.workspace.refine_dir / "camera_delta.pt")
        intrinsics_delta = _load_torch_state(self.workspace.refine_dir / "intrinsics_delta.pt")
        pose_delta = _load_torch_state(self.workspace.refine_dir / "pose_delta.pt")
        batch = _apply_refine_deltas_to_batch(
            batch,
            camera_delta=camera_delta,
            intrinsics_delta=intrinsics_delta,
            pose_delta=pose_delta,
        )
        query_points, flame_params = lam_model.renderer.get_query_points(batch.flame_params, device=batch.images.device)
        sweep_specs = _build_pose_sweep_specs(flame_params)
        h, w = batch.images.shape[-2:]

        montage_dir = out_dir / "montages"
        montage_dir.mkdir(parents=True, exist_ok=True)
        montage_frames = []
        report_rows = []
        rep_indices = _representative_view_indices(int(batch.c2ws.shape[1]), count=min(3, int(batch.c2ws.shape[1])))
        hold_frames = max(1, int(round(float(fps) * 1.5)))
        for idx, spec in enumerate(sweep_specs):
            sweep_flame = _apply_pose_sweep(flame_params, spec)
            with torch.no_grad():
                out = render_animate_gs_with_intrinsics(
                    lam_model.renderer,
                    [gs],
                    query_points,
                    sweep_flame,
                    batch.c2ws,
                    batch.intrs,
                    h,
                    w,
                    batch.bg_colors,
                )
            label = f"{idx:02d}_{spec['name']}"
            save_overlay_grid(
                frames_dir / label,
                batch.frame_ids,
                batch.images.detach().cpu(),
                batch.masks.detach().cpu(),
                out["comp_rgb"].detach().cpu(),
                out["comp_mask"].detach().cpu(),
                batch.landmarks_2d.detach().cpu() if batch.landmarks_2d is not None else None,
                max_items=max_views,
            )
            caption = _pose_sweep_caption(spec)
            montage = _make_pose_sweep_montage(
                out["comp_rgb"][0],
                out["comp_mask"][0],
                batch.frame_ids,
                rep_indices,
                title=caption["title"],
                subtitle="",
                footer="",
            )
            Image.fromarray(montage).save(montage_dir / f"{label}.png")
            for _ in range(hold_frames):
                montage_frames.append(montage.copy())
            report_rows.append(
                {
                    "name": spec["name"],
                    "description": spec["description"],
                    "edits": spec["edits"],
                    "caption": caption,
                }
            )

        video_path = out_dir / "pose_sweep.mp4"
        from lam.utils.video import images_to_video
        frames = np.stack(montage_frames, axis=0)
        images_to_video(frames, str(video_path), int(fps), gradio_codec=True)

        _save_pose_sweep_contact(montage_dir, out_dir / "pose_sweep_contact.jpg")
        _write_pose_sweep_notes(
            out_dir / "pose_sweep_notes.md",
            report_rows,
            rep_indices,
            int(batch.c2ws.shape[1]),
            hold_frames,
            int(fps),
        )

        (out_dir / "pose_sweep_report.json").write_text(
            json.dumps(
                {
                    "applied_refine_deltas": {
                        "camera_delta": camera_delta is not None,
                        "intrinsics_delta": intrinsics_delta is not None,
                        "pose_delta": pose_delta is not None,
                    },
                    "num_views": int(batch.c2ws.shape[1]),
                    "representative_view_indices": rep_indices,
                    "video_fps": int(fps),
                    "hold_frames_per_segment": hold_frames,
                    "num_sweeps": len(report_rows),
                    "sweeps": report_rows,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return StepResult(f"Pose sweep review ready. Sweeps: {len(report_rows)}", str(out_dir))

    def export_pose_transition_demo(
        self,
        lam_model,
        fps: int = 10,
        transition_frames: int = 10,
        hold_frames: int = 5,
        max_views: int = 3,
    ) -> StepResult:
        if not self.workspace.refined_gaussian_path.exists():
            raise FileNotFoundError(f"Missing refined Gaussian: {self.workspace.refined_gaussian_path}")

        out_dir = self.workspace.exports_dir / "pose_transition_demo"
        peaks_dir = out_dir / "peaks"
        if out_dir.exists():
            shutil.rmtree(out_dir)
        peaks_dir.mkdir(parents=True, exist_ok=True)

        batch = load_multiview_bundle(self.workspace.root, require_undistorted=False).to("cuda", torch.float32)
        gs = _load_renderable_gaussian(self.workspace.refined_gaussian_path)
        gs.to_cuda()
        camera_delta = _load_torch_state(self.workspace.refine_dir / "camera_delta.pt")
        intrinsics_delta = _load_torch_state(self.workspace.refine_dir / "intrinsics_delta.pt")
        pose_delta = _load_torch_state(self.workspace.refine_dir / "pose_delta.pt")
        batch = _apply_refine_deltas_to_batch(
            batch,
            camera_delta=camera_delta,
            intrinsics_delta=intrinsics_delta,
            pose_delta=pose_delta,
        )
        total_views = int(batch.c2ws.shape[1])
        rep_indices = _representative_view_indices(total_views, count=min(max_views, total_views))
        batch = batch.index(rep_indices)
        local_view_indices = list(range(int(batch.c2ws.shape[1])))
        query_points, flame_params = lam_model.renderer.get_query_points(batch.flame_params, device=batch.images.device)
        sweep_specs = _build_pose_sweep_specs(flame_params)
        neutral_spec = sweep_specs[0]
        probe_specs = sweep_specs[1:] if len(sweep_specs) > 1 else sweep_specs
        h, w = batch.images.shape[-2:]

        transition_frames = max(int(transition_frames), 2)
        hold_frames = max(int(hold_frames), 1)
        intro_hold = max(int(round(float(fps) * 0.8)), 1)
        outro_hold = max(int(round(float(fps) * 0.5)), 1)
        montage_frames = []
        segment_rows = []

        def render_montage(spec: dict, strength: float) -> np.ndarray:
            sweep_flame = _apply_pose_sweep(flame_params, spec, strength=float(strength))
            with torch.no_grad():
                out = render_animate_gs_with_intrinsics(
                    lam_model.renderer,
                    [gs],
                    query_points,
                    sweep_flame,
                    batch.c2ws,
                    batch.intrs,
                    h,
                    w,
                    batch.bg_colors,
                )
            caption = _pose_sweep_caption(spec)
            footer = ""
            if spec.get("name") != "neutral":
                footer = f"强度 {float(strength):.0%}"
            return _make_pose_sweep_montage(
                out["comp_rgb"][0],
                out["comp_mask"][0],
                batch.frame_ids,
                local_view_indices,
                title=caption["title"],
                subtitle="",
                footer=footer,
            )

        neutral_caption = _pose_sweep_caption(neutral_spec)
        neutral_montage = render_montage(neutral_spec, strength=0.0)
        Image.fromarray(neutral_montage).save(peaks_dir / "00_neutral.png")
        for _ in range(intro_hold):
            montage_frames.append(neutral_montage.copy())
        segment_rows.append(
            {
                "name": neutral_spec["name"],
                "description": neutral_spec["description"],
                "edits": neutral_spec["edits"],
                "caption": neutral_caption,
                "duration_frames": intro_hold,
            }
        )

        for idx, spec in enumerate(probe_specs, start=1):
            caption = _pose_sweep_caption(spec)
            peak_montage = None
            segment_frame_count = 0
            for alpha in _ease_values(transition_frames, 0.0, 1.0):
                peak_montage = render_montage(spec, strength=float(alpha))
                montage_frames.append(peak_montage)
                segment_frame_count += 1
            if peak_montage is None:
                continue
            Image.fromarray(peak_montage).save(peaks_dir / f"{idx:02d}_{spec['name']}.png")
            for _ in range(hold_frames):
                montage_frames.append(peak_montage.copy())
                segment_frame_count += 1
            for alpha in _ease_values(transition_frames, 1.0, 0.0)[1:]:
                montage_frames.append(render_montage(spec, strength=float(alpha)))
                segment_frame_count += 1
            segment_rows.append(
                {
                    "name": spec["name"],
                    "description": spec["description"],
                    "edits": spec["edits"],
                    "caption": caption,
                    "duration_frames": segment_frame_count,
                }
            )

        for _ in range(outro_hold):
            montage_frames.append(neutral_montage.copy())

        video_path = out_dir / "pose_transition_demo.mp4"
        from lam.utils.video import images_to_video
        images_to_video(np.stack(montage_frames, axis=0), str(video_path), int(fps), gradio_codec=True)

        _save_pose_sweep_contact(peaks_dir, out_dir / "pose_transition_contact.jpg")
        _write_pose_transition_notes(
            out_dir / "pose_transition_notes.md",
            segment_rows,
            rep_indices,
            total_views,
            transition_frames,
            hold_frames,
            int(fps),
        )
        (out_dir / "pose_transition_report.json").write_text(
            json.dumps(
                {
                    "applied_refine_deltas": {
                        "camera_delta": camera_delta is not None,
                        "intrinsics_delta": intrinsics_delta is not None,
                        "pose_delta": pose_delta is not None,
                    },
                    "representative_view_indices": rep_indices,
                    "num_views_total": total_views,
                    "num_views_selected": int(batch.c2ws.shape[1]),
                    "video_fps": int(fps),
                    "transition_frames": transition_frames,
                    "hold_frames": hold_frames,
                    "intro_hold_frames": intro_hold,
                    "outro_hold_frames": outro_hold,
                    "num_frames": len(montage_frames),
                    "segments": segment_rows,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return StepResult(
            f"Pose transition demo ready. Segments: {len(segment_rows)}. Frames: {len(montage_frames)}",
            str(out_dir),
        )

    def _merge_tracking_exports(self, exports: list[Path], preserve_existing_masks: bool = True) -> int:
        for directory in [self.workspace.masks_dir, self.workspace.flame_dir, self.workspace.landmark_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        images = sorted([p for p in self.workspace.images_dir.glob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"}])
        if len(exports) != len(images):
            raise ValueError(f"FLAME tracking exports/images mismatch: {len(exports)} exports for {len(images)} images")

        canonical_path = self.workspace.canonical_flame_path
        if not canonical_path.exists():
            raise FileNotFoundError(
                f"Missing {canonical_path}. Import the Layer 1/LAM identity canonical_flame_param.npz before running FLAME tracking."
            )
        canonical_shape = _load_shape(canonical_path)
        if canonical_shape is None:
            raise KeyError(f"{canonical_path} must contain 'shape' or 'betas'")

        records = []
        preserved_masks = 0
        for idx, export_dir in enumerate(exports):
            image_path = images[idx]
            stem = image_path.stem if idx < len(images) else f"{idx:05d}"
            meta = _load_preprocess_meta(self.workspace.tracking_dir / "preprocess" / stem / "preprocess_meta.json", image_path)
            mask_src = next((export_dir / "fg_masks").glob("*"))
            flame_src = next((export_dir / "flame_param").glob("*.npz"))
            mask_dst = self.workspace.masks_dir / f"{stem}.png"
            flame_dst = self.workspace.flame_dir / f"{stem}.npz"
            landmark_dst = self.workspace.landmark_dir / f"{stem}.npz"
            mask_source = "flame_tracking"
            existing_mask = _find_mask_by_stem(self.workspace.masks_dir, stem)
            mask_record_path = mask_dst
            if preserve_existing_masks and existing_mask is not None:
                if existing_mask.stem != stem:
                    mask_record_path = self.workspace.masks_dir / f"{stem}{existing_mask.suffix.lower()}"
                    if not mask_record_path.exists():
                        shutil.copy2(existing_mask, mask_record_path)
                else:
                    mask_record_path = existing_mask
                preserved_masks += 1
                mask_source = "existing_upload"
            else:
                _restore_tracking_mask(mask_src, image_path, meta, mask_dst)
                mask_record_path = mask_dst
            _write_tracking_flame_param(flame_src, flame_dst, canonical_shape)
            if (export_dir / "landmark2d" / "landmarks.npz").exists():
                _restore_tracking_landmarks(export_dir / "landmark2d" / "landmarks.npz", image_path, meta, landmark_dst)
            records.append({
                "image_name": image_path.name,
                "file_path": f"data/images/{image_path.name}",
                "fg_mask_path": str(mask_record_path.relative_to(self.workspace.root)).replace("\\", "/"),
                "fg_mask_source": mask_source,
                "flame_param_path": f"data/flame_param/{stem}.npz",
                "landmark_path": f"data/landmark2d/{stem}.npz" if landmark_dst.exists() else None,
                "tracking_export": str(export_dir),
            })

        (self.workspace.tracking_dir / "flame_tracking_manifest.json").write_text(
            json.dumps(
                {
                    "coordinate_roles": {
                        "images": "original multi-view image pixels used by COLMAP and refinement",
                        "masks_landmarks": "restored from FLAME tracking crop coordinates into original image pixels",
                        "flame_params": "per-view expression/jaw/eyes and local pose cues in LAM FLAME parameter format",
                        "flame_global_pose": "canonicalized to zero rotation/translation",
                        "cameras": "not provided by single-image FLAME tracking; use COLMAP plus Layer1 Sim3 initialization or explicit calibrated alignment",
                        "target_model": "LAM canonical FLAME-bound Gaussian space",
                    },
                    "shape_source": "layer1_or_imported",
                    "frames": records,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return preserved_masks

    def _save_workspace_preview(self, subdir: str) -> Path:
        out_dir = self.workspace.debug_dir / subdir
        save_workspace_image_previews(self.workspace.root, out_dir)
        return out_dir

    def _save_colmap_visualization(self, transforms: str | Path) -> Path:
        out = self.workspace.debug_dir / "02_colmap" / "camera_centers.png"
        save_transform_camera_plot(out, {"colmap": transforms}, title="COLMAP camera centers")
        return out

    def _save_sim3_visualization(self, target_transforms: Optional[str | Path]) -> Path:
        tracks = {
            "colmap raw": self.workspace.colmap_transforms_path,
            "aligned": self.workspace.aligned_transforms_path,
        }
        if target_transforms:
            tracks["target"] = target_transforms
        out = self.workspace.debug_dir / "03_alignment" / "alignment_camera_centers.png"
        save_transform_camera_plot(out, tracks, title="Sim3 camera alignment")
        return out

    def _write_initial_sim3_result(
        self,
        *,
        colmap_db: dict,
        sim3: Sim3Alignment,
        colmap_center: np.ndarray,
        ray_scene_center: np.ndarray,
        look_at_rmse: float,
        source_anchor_report: dict,
        source_anchor_center: np.ndarray,
        colmap_radius: float,
        layer1_stats: dict,
        target_center: np.ndarray,
        reference_center: Optional[np.ndarray],
        reference_distance: Optional[float],
        fallback_distance: float,
        target_anchor_center: np.ndarray,
        target_distance: float,
        rotation_policy: str,
        projection_score: dict,
        landmark_score: Optional[dict],
        scale_candidates: Optional[list[float]],
    ) -> StepResult:
        sim3.save(self.workspace.sim3_path)
        for frame in colmap_db["frames"]:
            frame["transform_matrix_colmap"] = frame["transform_matrix"]
            render_c2w = _json_c2w_to_render_c2w(np.asarray(frame["transform_matrix"], dtype=np.float32))
            aligned_render_c2w = apply_sim3_to_c2w(render_c2w, sim3)
            frame["transform_matrix"] = _render_c2w_to_json_c2w(aligned_render_c2w).tolist()
            frame["camera_convention"] = "nerf_json_yz_flip_from_render_opencv"
        self.workspace.aligned_transforms_path.parent.mkdir(parents=True, exist_ok=True)
        colmap_db["sim3_source"] = sim3.source
        self.workspace.aligned_transforms_path.write_text(json.dumps(colmap_db, indent=2), encoding="utf-8")
        report_path = self.workspace.alignment_dir / "initial_sim3_report.json"
        report_path.write_text(
            json.dumps(
                {
                    "source": sim3.source,
                    "colmap_center": np.asarray(colmap_center, dtype=np.float64).tolist(),
                    "colmap_scene_anchor": source_anchor_report,
                    "colmap_look_at_center": np.asarray(ray_scene_center, dtype=np.float64).tolist(),
                    "colmap_look_at_rmse": look_at_rmse,
                    "source_anchor_center": np.asarray(source_anchor_center, dtype=np.float64).tolist(),
                    "colmap_radius_median": float(colmap_radius),
                    "layer1_center": np.asarray(target_center, dtype=np.float64).tolist(),
                    "layer1_radius_p95": float(layer1_stats["radius_p95"]),
                    "target_camera_distance": float(target_distance),
                    "reference_camera_distance": float(reference_distance) if reference_distance is not None else None,
                    "fallback_camera_distance": float(fallback_distance),
                    "reference_camera_available": reference_center is not None,
                    "reference_center": np.asarray(reference_center, dtype=np.float64).tolist() if reference_center is not None else None,
                    "target_anchor_center": np.asarray(target_anchor_center, dtype=np.float64).tolist(),
                    "first_view_target_center": np.asarray(reference_center, dtype=np.float64).tolist() if reference_center is not None else None,
                    "scale": float(sim3.scale),
                    "scale_candidates": scale_candidates,
                    "translation": sim3.translation,
                    "rotation_policy": rotation_policy,
                    "projection_score": projection_score,
                    "landmark_score": landmark_score,
                    "camera_convention": "json stores NeRF-style c2w; renderer/data converts by flipping c2w Y/Z axes",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        _write_projection_diagnostics(self.workspace)
        plot = self._save_sim3_visualization(None)
        return StepResult(
            f"Layer1 Sim3 initialized. scale={float(sim3.scale):.5f}, COLMAP radius={float(colmap_radius):.5f}, target distance={float(target_distance):.5f}",
            str(plot),
        )


def _load_renderable_gaussian(path: str | Path) -> GaussianModel:
    gs = GaussianModel(ply_path=str(path), sh2rgb=False)
    # GaussianModel.save_ply stores opacity as logit and scale as log. The renderer
    # consumes activated opacity/scale tensors directly.
    gs.opacity = torch.sigmoid(gs.opacity)
    gs.scaling = torch.exp(gs.scaling)
    return gs


def _colmap_step_message(prefix: str, raw_json: str | Path) -> str:
    db = json.loads(Path(raw_json).read_text(encoding="utf-8"))
    stats = db.get("colmap_model_stats") or {}
    model_dir = db.get("colmap_model_dir")
    registered = stats.get("registered_images", len(db.get("frames", [])))
    points3d = stats.get("points3d")
    details = f"selected model: {model_dir}; registered images: {registered}"
    if points3d is not None:
        details += f"; points3D: {points3d}"
    return f"{prefix} ({details})"


def _json_c2w_to_render_c2w(c2w: np.ndarray) -> np.ndarray:
    out = np.asarray(c2w, dtype=np.float64).copy()
    out[:3, 1:3] *= -1.0
    return out


def _render_c2w_to_json_c2w(c2w: np.ndarray) -> np.ndarray:
    return _json_c2w_to_render_c2w(c2w).astype(np.float32)


def _camera_forward(c2w_render: np.ndarray) -> np.ndarray:
    # Renderer/OpenCV convention: camera looks along its local +Z axis.
    return np.asarray(c2w_render, dtype=np.float64)[:3, 2]


def _estimate_camera_ray_intersection(centers: np.ndarray, forwards: np.ndarray) -> tuple[np.ndarray, float]:
    centers = np.asarray(centers, dtype=np.float64)
    forwards = np.asarray(forwards, dtype=np.float64)
    if centers.shape != forwards.shape or centers.ndim != 2 or centers.shape[1] != 3:
        raise ValueError("centers and forwards must both have shape [N, 3]")
    a = np.zeros((3, 3), dtype=np.float64)
    b = np.zeros(3, dtype=np.float64)
    valid = 0
    for center, forward in zip(centers, forwards):
        f = _normalize(forward, np.zeros(3, dtype=np.float64))
        if np.linalg.norm(f) < 1e-8:
            continue
        projector = np.eye(3, dtype=np.float64) - np.outer(f, f)
        a += projector
        b += projector @ center
        valid += 1
    if valid < 2:
        return np.median(centers, axis=0).astype(np.float64), float("inf")
    try:
        point = np.linalg.solve(a + np.eye(3, dtype=np.float64) * 1e-8, b)
    except np.linalg.LinAlgError:
        point = np.linalg.lstsq(a, b, rcond=None)[0]
    residuals = []
    for center, forward in zip(centers, forwards):
        f = _normalize(forward, np.zeros(3, dtype=np.float64))
        if np.linalg.norm(f) < 1e-8:
            continue
        residuals.append(np.linalg.norm(np.cross(point - center, f)))
    rmse = float(np.sqrt(np.mean(np.square(residuals)))) if residuals else float("inf")
    if not np.isfinite(point).all():
        point = np.median(centers, axis=0).astype(np.float64)
    return point.astype(np.float64), rmse


def _choose_colmap_scene_anchor(
    workspace: MultiViewWorkspace,
    colmap_mats: list[np.ndarray],
    ray_scene_center: np.ndarray,
    look_at_rmse: float,
) -> tuple[np.ndarray, dict]:
    ray_score = _score_colmap_scene_anchor(colmap_mats, ray_scene_center)
    report = {
        "source": "camera_ray_intersection",
        "center": np.asarray(ray_scene_center, dtype=np.float64).tolist(),
        "score": ray_score,
        "look_at_rmse": float(look_at_rmse),
    }
    points_center, points_report = _load_colmap_points_center(workspace)
    if points_center is None:
        return np.asarray(ray_scene_center, dtype=np.float64), report

    points_score = _score_colmap_scene_anchor(colmap_mats, points_center)
    report["points3d"] = {**points_report, "score": points_score}
    if _colmap_anchor_score_tuple(points_score) >= _colmap_anchor_score_tuple(ray_score):
        report["source"] = "colmap_points3d_median"
        report["center"] = np.asarray(points_center, dtype=np.float64).tolist()
        return np.asarray(points_center, dtype=np.float64), report
    return np.asarray(ray_scene_center, dtype=np.float64), report


def _load_colmap_points_center(workspace: MultiViewWorkspace) -> tuple[Optional[np.ndarray], dict]:
    paths = _colmap_points3d_candidates(workspace)
    if not paths:
        return None, {"path": None, "num_points": 0}
    for path in paths:
        points = []
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            try:
                points.append([float(parts[1]), float(parts[2]), float(parts[3])])
            except ValueError:
                continue
        if len(points) >= 3:
            arr = np.asarray(points, dtype=np.float64)
            arr = arr[np.isfinite(arr).all(axis=1)]
            if len(arr) >= 3:
                return np.median(arr, axis=0), {
                    "path": str(path),
                    "num_points": int(len(arr)),
                    "center_policy": "median",
                }
    return None, {"path": str(paths[0]), "num_points": 0}


def _colmap_points3d_candidates(workspace: MultiViewWorkspace) -> list[Path]:
    selected_model = _selected_colmap_model_dir(workspace)
    paths: list[Path] = []
    if selected_model is not None:
        selected_text = selected_model.parent / f"{selected_model.name}_txt"
        for candidate in [selected_model / "points3D.txt", selected_text / "points3D.txt"]:
            if candidate.exists():
                paths.append(candidate)
    fallback = sorted((workspace.colmap_dir / "sparse").rglob("points3D.txt"))
    seen = {path.resolve() for path in paths}
    paths.extend(path for path in fallback if path.resolve() not in seen)
    return paths


def _selected_colmap_model_dir(workspace: MultiViewWorkspace) -> Optional[Path]:
    for path in [workspace.colmap_transforms_path, workspace.colmap_dir / "transforms_colmap_sparse.json"]:
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        model_dir = data.get("colmap_model_dir")
        if not model_dir:
            continue
        model_path = Path(model_dir)
        if not model_path.is_absolute():
            model_path = workspace.root / model_path
        if model_path.exists():
            return model_path
    return None


def _score_colmap_scene_anchor(colmap_mats: list[np.ndarray], point: np.ndarray) -> dict:
    depths = []
    point = np.asarray(point, dtype=np.float64)
    homog = np.concatenate([point, np.ones(1, dtype=np.float64)])
    for c2w in colmap_mats:
        cam = (np.linalg.inv(np.asarray(c2w, dtype=np.float64)) @ homog)[:3]
        depths.append(float(cam[2]))
    if not depths:
        return {"positive_depth": 0, "num_views": 0, "min_depth": None, "median_depth": None}
    depths_arr = np.asarray(depths, dtype=np.float64)
    return {
        "positive_depth": int(np.sum(depths_arr > 1e-6)),
        "num_views": int(len(depths_arr)),
        "min_depth": float(np.min(depths_arr)),
        "median_depth": float(np.median(depths_arr)),
    }


def _colmap_anchor_score_tuple(score: dict) -> tuple:
    return (
        int(score.get("positive_depth", 0)),
        float(score.get("median_depth") or -float("inf")),
        float(score.get("min_depth") or -float("inf")),
    )


def _choose_initial_sim3_candidate(
    workspace: MultiViewWorkspace,
    lam_model,
    landmark_context,
    colmap_mats: list[np.ndarray],
    intrinsics: list[np.ndarray],
    source_scene_center: np.ndarray,
    source_anchor_center: np.ndarray,
    target_anchor_center: np.ndarray,
    target_center: np.ndarray,
    scales: list[float],
    reference_c2w: Optional[np.ndarray],
    target_distance: float,
) -> dict:
    source_front = np.asarray(colmap_mats[0], dtype=np.float64)
    reference_forward = None
    reference_up = None
    if reference_c2w is not None:
        base_target_forward = np.asarray(target_center, dtype=np.float64) - np.asarray(reference_c2w[:3, 3], dtype=np.float64)
        base_target_up = np.asarray(reference_c2w[:3, 1], dtype=np.float64)
        reference_forward = _normalize(base_target_forward, np.array([0.0, 0.0, -1.0], dtype=np.float64))
        reference_up = _normalize(base_target_up, np.array([0.0, -1.0, 0.0], dtype=np.float64))
        target_options = [
            ("tgt", base_target_forward, base_target_up),
            ("tgt_forward_flip", -base_target_forward, base_target_up),
            ("tgt_up_flip", base_target_forward, -base_target_up),
            ("tgt_both_flip", -base_target_forward, -base_target_up),
        ]
    else:
        reference_center = np.asarray(target_center, dtype=np.float64) + np.array([0.0, 0.0, target_distance], dtype=np.float64)
        base_target_forward = np.asarray(target_center, dtype=np.float64) - reference_center
        # Match the default single-image LAM export camera. Its JSON transform is
        # identity at +Z; after the standard NeRF Y/Z column flip the renderer
        # camera has local +Y pointing to world -Y and local +Z toward the head.
        base_target_up = np.array([0.0, -1.0, 0.0], dtype=np.float64)
        target_options = [("tgt_lam_front", base_target_forward, base_target_up)]

    source_options = [
        ("src+z", _camera_forward(source_front), source_front[:3, 1]),
    ]
    best = None
    for source_name, source_forward, source_up in source_options:
        for target_name, target_forward, target_up in target_options:
            R_col = _rotation_from_forward_up(source_forward, source_up, target_forward, target_up)
            R_row = R_col.T
            for scale in scales:
                translation = target_anchor_center - float(scale) * (source_anchor_center @ R_row)
                sim3 = Sim3Alignment(
                    scale=float(scale),
                    rotation=R_row.astype(np.float32).tolist(),
                    translation=translation.astype(np.float32).tolist(),
                    rmse=-1.0,
                )
                score = _score_initial_sim3_projection(colmap_mats, intrinsics, sim3, target_center)
                if reference_forward is not None and reference_up is not None:
                    aligned_front = apply_sim3_to_c2w(source_front, sim3)
                    aligned_forward = _normalize(_camera_forward(aligned_front), reference_forward)
                    aligned_up = _normalize(aligned_front[:3, 1], reference_up)
                    score["reference_forward_dot"] = float(np.dot(aligned_forward, reference_forward))
                    score["reference_up_dot"] = float(np.dot(aligned_up, reference_up))
                landmark_score = _score_initial_sim3_landmarks(workspace, lam_model, landmark_context, sim3)
                candidate = {
                    "scale": float(scale),
                    "rotation_row": R_row,
                    "translation": translation,
                    "policy": f"{source_name}->{target_name}",
                    "score": score,
                    "landmark_score": landmark_score,
                }
                if best is None or _initial_candidate_score_tuple(candidate) > _initial_candidate_score_tuple(best):
                    best = candidate
    if best is None:
        raise RuntimeError("Could not build any initial Sim3 candidate.")
    return best


def _score_initial_sim3_projection(
    colmap_mats: list[np.ndarray],
    intrinsics: list[np.ndarray],
    sim3: Sim3Alignment,
    target_center: np.ndarray,
) -> dict:
    positive = 0
    in_frame = 0
    abs_center_error = []
    min_depth = float("inf")
    for c2w, intr in zip(colmap_mats, intrinsics):
        aligned = apply_sim3_to_c2w(np.asarray(c2w, dtype=np.float64), sim3)
        uv, z = _project_point_np(target_center, aligned, intr)
        if z > 1e-6:
            positive += 1
        min_depth = min(min_depth, float(z))
        width = float(intr[0, 2]) * 2.0
        height = float(intr[1, 2]) * 2.0
        if z > 1e-6 and 0.0 <= uv[0] < width and 0.0 <= uv[1] < height:
            in_frame += 1
        abs_center_error.append(float(np.linalg.norm(uv - np.array([intr[0, 2], intr[1, 2]], dtype=np.float64))))
    return {
        "positive_depth": positive,
        "in_frame": in_frame,
        "num_views": len(colmap_mats),
        "median_center_error_px": float(np.median(abs_center_error)) if abs_center_error else float("inf"),
        "min_depth": min_depth,
    }


def _projection_score_tuple(score: dict) -> tuple:
    forward_dot = float(score.get("reference_forward_dot", 0.0))
    up_dot = float(score.get("reference_up_dot", 0.0))
    return (
        int(score.get("positive_depth", 0)),
        int(score.get("in_frame", 0)),
        int(forward_dot > 0.5),
        int(up_dot > 0.5),
        min(forward_dot, up_dot),
        forward_dot + up_dot,
        -float(score.get("median_center_error_px", float("inf"))),
        float(score.get("min_depth", -float("inf"))),
    )


def _initial_candidate_score_tuple(candidate: dict) -> tuple:
    landmark = candidate.get("landmark_score") or {}
    score = candidate["score"]
    landmark_valid = int(bool(landmark.get("valid")))
    landmark_negative = -float(landmark.get("landmark_px", float("inf")))
    landmark_in_frame = int(landmark.get("in_frame", 0))
    landmark_coverage = float(landmark.get("coverage", 0.0))
    return (
        landmark_valid,
        landmark_negative,
        landmark_in_frame,
        landmark_coverage,
        *_projection_score_tuple(score),
    )


def _build_initial_scale_candidates(
    base_scale: float,
    colmap_radius: float,
    fallback_distance: float,
    reference_distance: float,
) -> list[float]:
    seeds = []
    if np.isfinite(base_scale) and base_scale > 1e-8:
        seeds.append(float(base_scale))
    fallback_scale = float(fallback_distance / max(colmap_radius, 1e-8))
    if np.isfinite(fallback_scale) and fallback_scale > 1e-8:
        seeds.append(fallback_scale)
    if reference_distance > 1e-8:
        reference_scale = float(reference_distance / max(colmap_radius, 1e-8))
        if np.isfinite(reference_scale) and reference_scale > 1e-8:
            seeds.append(reference_scale)

    candidates = []
    for seed in seeds:
        for multiplier in INITIAL_SIM3_SCALE_MULTIPLIERS:
            value = float(seed * multiplier)
            if np.isfinite(value) and value > 1e-6:
                candidates.append(value)
    if not candidates:
        candidates = [max(base_scale, 1e-3)]

    deduped = []
    for value in sorted(candidates):
        if not deduped or abs(value - deduped[-1]) / max(deduped[-1], 1e-6) > 0.03:
            deduped.append(value)
    return deduped


def _prepare_initial_landmark_context(workspace: MultiViewWorkspace, lam_model):
    if lam_model is None or not workspace.colmap_transforms_path.exists():
        return None
    try:
        batch = load_multiview_bundle(
            workspace.root,
            transforms_name="colmap/transforms_colmap_raw.json",
            require_undistorted=False,
        ).to("cuda", torch.float32)
    except Exception:
        return None
    if batch.landmarks_2d is None:
        return None
    h, w = batch.images.shape[-2:]
    try:
        query_points, flame_params = lam_model.renderer.get_query_points(batch.flame_params, device=batch.images.device)
    except Exception:
        return None
    return {
        "batch": batch,
        "query_points": query_points,
        "flame_params": flame_params,
        "image_hw": (h, w),
    }


def _score_initial_sim3_landmarks(
    workspace: MultiViewWorkspace,
    lam_model,
    landmark_context,
    sim3: Sim3Alignment,
) -> dict:
    if lam_model is None:
        return {"valid": False, "reason": "missing_lam_model"}
    if landmark_context is None:
        return {"valid": False, "reason": "missing_landmark_context"}

    batch = landmark_context["batch"]
    query_points = landmark_context["query_points"]
    flame_params = landmark_context["flame_params"]
    h, w = landmark_context["image_hw"]
    try:
        c2ws = batch.c2ws.clone()
        delta_rot = torch.as_tensor(np.asarray(sim3.rotation, dtype=np.float32), device=batch.images.device, dtype=batch.images.dtype).reshape(1, 1, 3, 3)
        delta_t = torch.as_tensor(np.asarray(sim3.translation, dtype=np.float32), device=batch.images.device, dtype=batch.images.dtype).reshape(1, 1, 3)
        scale = torch.as_tensor(float(sim3.scale), device=batch.images.device, dtype=batch.images.dtype)
        c2ws[..., :3, :3] = torch.matmul(delta_rot, c2ws[..., :3, :3])
        c2ws[..., :3, 3] = scale * torch.matmul(delta_rot, c2ws[..., :3, 3:4]).squeeze(-1) + delta_t
        with torch.no_grad():
            pred = _project_flame_landmarks(lam_model.renderer, query_points, flame_params, c2ws, batch.intrs, h, w)
            _, landmark_px = _landmark_loss(pred, batch.landmarks_2d, (h, w), beta=0.002)
        target = batch.landmarks_2d[..., :2]
        if _landmarks_are_normalized(target):
            target = target.clone()
            target[..., 0] *= float(w)
            target[..., 1] *= float(h)
        count = min(pred.shape[2], target.shape[2])
        pred_used = pred[:, :, :count]
        target_used = target[:, :, :count]
        valid = torch.isfinite(pred_used).all(dim=-1) & torch.isfinite(target_used).all(dim=-1)
        if batch.landmarks_2d.shape[-1] > 2:
            valid = valid & (batch.landmarks_2d[:, :, :count, 2] > 0)
        pred_uv = pred_used
        in_frame = (
            valid
            & (pred_uv[..., 0] >= 0.0)
            & (pred_uv[..., 0] < float(w))
            & (pred_uv[..., 1] >= 0.0)
            & (pred_uv[..., 1] < float(h))
        )
        valid_views = torch.any(valid, dim=-1)
        in_frame_views = torch.any(in_frame, dim=-1)
        num_views = int(valid_views.sum().item())
        covered = int(in_frame_views.sum().item())
        return {
            "valid": bool(num_views > 0),
            "landmark_px": float(landmark_px.detach().cpu()),
            "num_views": num_views,
            "in_frame": covered,
            "coverage": float(covered / max(num_views, 1)),
        }
    except Exception as exc:
        return {"valid": False, "reason": f"score_failed: {exc}"}


def _should_accept_initial_pnp_candidate(workspace: MultiViewWorkspace, candidate: dict) -> bool:
    landmark = candidate.get("landmark_score") or {}
    projection = candidate.get("projection_score") or {}
    if not landmark.get("valid"):
        return False
    max_dim = _workspace_image_max_dim(workspace)
    landmark_px = float(landmark.get("landmark_px", float("inf")))
    coverage = float(landmark.get("coverage", 0.0))
    center_error = float(projection.get("median_center_error_px", float("inf")))
    return (
        np.isfinite(landmark_px)
        and landmark_px <= INITIAL_PNP_ACCEPT_MAX_LANDMARK_PX
        and coverage >= INITIAL_PNP_ACCEPT_MIN_COVERAGE
        and np.isfinite(center_error)
        and center_error <= INITIAL_PNP_ACCEPT_MAX_CENTER_ERROR_PX * max_dim
    )


def _estimate_initial_sim3_from_landmark_pnp(workspace: MultiViewWorkspace, lam_model) -> Optional[dict]:
    if lam_model is None:
        return None
    try:
        frames = load_frames(workspace.root, transforms_name="colmap/transforms_colmap_raw.json", require_undistorted=False)
        batch = load_multiview_bundle(workspace.root, transforms_name="colmap/transforms_colmap_raw.json", require_undistorted=False).to("cuda", torch.float32)
    except Exception:
        return None
    if len(frames) < 3 or batch.landmarks_2d is None:
        return None

    try:
        query_points, flame_params = lam_model.renderer.get_query_points(batch.flame_params, device=batch.images.device)
        flame = {k: v[0] if k != "betas" else v for k, v in flame_params.items()}
        num_views = flame["expr"].shape[0]
        v_cano = query_points[0].unsqueeze(0).repeat(num_views, 1, 1)
        expr = torch.cat([flame["expr"], flame["teeth_bs"]], dim=-1) if getattr(lam_model.renderer, "teeth_bs_flag", False) and "teeth_bs" in flame else flame["expr"]
        ret = lam_model.renderer.flame_model.animation_forward(
            v_cano=v_cano,
            shape=flame["betas"].repeat(num_views, 1),
            expr=expr,
            rotation=flame["rotation"],
            neck=flame["neck_pose"],
            jaw=flame["jaw_pose"],
            eyes=flame["eyes_pose"],
            translation=flame["translation"],
            zero_centered_at_root_node=False,
            return_landmarks=True,
            return_verts_cano=False,
            static_offset=None,
        )
        canonical_landmarks = ret["landmarks"][0].detach().cpu().float().numpy().astype(np.float64)
    except Exception:
        return None

    raw_landmarks = batch.landmarks_2d.detach().cpu()
    target = raw_landmarks[..., :2]
    h, w = batch.images.shape[-2:]
    if _landmarks_are_normalized(target):
        target = target.clone()
        target[..., 0] *= float(w)
        target[..., 1] *= float(h)
    confidences = raw_landmarks[..., 2] if raw_landmarks.shape[-1] > 2 else torch.ones_like(target[..., 0])

    colmap_centers = []
    pnp_centers = []
    valid_names = []
    for view_idx, frame in enumerate(frames):
        pnp_center = _solve_pnp_camera_center(
            canonical_landmarks,
            target[0, view_idx].detach().cpu().numpy().astype(np.float64),
            confidences[0, view_idx].detach().cpu().numpy().astype(np.float64),
            frame.intr.detach().cpu().numpy().astype(np.float64),
        )
        if pnp_center is None:
            continue
        colmap_centers.append(frame.c2w[:3, 3].detach().cpu().numpy().astype(np.float64))
        pnp_centers.append(pnp_center)
        valid_names.append(frame.frame_id)

    if len(colmap_centers) < 3:
        return None

    source = np.asarray(colmap_centers, dtype=np.float64)
    target_centers = np.asarray(pnp_centers, dtype=np.float64)
    try:
        sim3 = estimate_sim3_umeyama_ransac(
            source,
            target_centers,
            estimate_scale=True,
            names=valid_names,
            max_iterations=256,
        )
    except Exception:
        return None

    colmap_db = json.loads(workspace.colmap_transforms_path.read_text(encoding="utf-8"))
    layer1_stats = _load_layer1_alignment_stats(workspace)
    projection_score = _score_initial_sim3_projection(
        [_json_c2w_to_render_c2w(np.asarray(frame["transform_matrix"], dtype=np.float64)) for frame in colmap_db.get("frames", [])],
        [_intrinsic_from_frame(frame) for frame in colmap_db.get("frames", [])],
        sim3,
        layer1_stats["center"],
    )
    landmark_context = _prepare_initial_landmark_context(workspace, lam_model)
    landmark_score = _score_initial_sim3_landmarks(workspace, lam_model, landmark_context, sim3)
    aligned_centers = np.asarray([apply_sim3_to_c2w(frame.c2w.detach().cpu().numpy().astype(np.float64), sim3)[:3, 3] for frame in frames], dtype=np.float64)
    target_distance = float(np.median(np.linalg.norm(aligned_centers - layer1_stats["center"], axis=1)))
    return {
        "sim3": sim3,
        "target_distance": target_distance,
        "policy": "landmark_pnp_camera_center_sim3",
        "projection_score": projection_score,
        "landmark_score": landmark_score,
    }


def _solve_pnp_camera_center(
    canonical_landmarks: np.ndarray,
    image_landmarks: np.ndarray,
    confidences: np.ndarray,
    intr: np.ndarray,
) -> Optional[np.ndarray]:
    object_points = []
    image_points = []
    for idx in PNP_LANDMARK_INDICES:
        if idx >= canonical_landmarks.shape[0] or idx >= image_landmarks.shape[0]:
            continue
        if not np.isfinite(canonical_landmarks[idx]).all() or not np.isfinite(image_landmarks[idx]).all():
            continue
        if idx < confidences.shape[0] and float(confidences[idx]) <= 0.0:
            continue
        object_points.append(canonical_landmarks[idx])
        image_points.append(image_landmarks[idx])
    if len(object_points) < 12:
        return None

    object_points = np.asarray(object_points, dtype=np.float64)
    image_points = np.asarray(image_points, dtype=np.float64)
    camera_matrix = np.asarray(
        [
            [intr[0, 0], 0.0, intr[0, 2]],
            [0.0, intr[1, 1], intr[1, 2]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)
    try:
        ok, rvec, tvec, _ = cv2.solvePnPRansac(
            object_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_EPNP,
            reprojectionError=12.0,
            iterationsCount=256,
            confidence=0.999,
        )
        if not ok:
            return None
        ok_refine, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            rvec=rvec,
            tvec=tvec,
            useExtrinsicGuess=True,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok_refine:
            return None
    except Exception:
        return None

    R, _ = cv2.Rodrigues(rvec)
    center = -R.T @ tvec.reshape(3)
    if not np.isfinite(center).all():
        return None
    return center.astype(np.float64)


def _score_workspace_geometry(workspace: MultiViewWorkspace, delta: Optional[dict] = None) -> dict:
    layer1_stats = _load_layer1_alignment_stats(workspace)
    center = np.asarray(layer1_stats["center"], dtype=np.float64)
    radius = float(layer1_stats["radius_p95"])
    db = json.loads(workspace.aligned_transforms_path.read_text(encoding="utf-8"))
    rows = []
    for frame in db.get("frames", []):
        c2w = _json_c2w_to_render_c2w(np.asarray(frame["transform_matrix"], dtype=np.float64))
        if delta is not None:
            c2w = _apply_global_render_delta_np(c2w, delta["scale"], delta["rotation"], delta["translation"])
        intr = _intrinsic_from_frame(frame)
        uv, z = _project_point_np(center, c2w, intr)
        rows.append({
            "image_name": frame.get("image_name") or frame.get("file_path"),
            "image_size": [int(frame["w"]), int(frame["h"])],
            "center_z": float(z),
            "center_uv": uv.tolist(),
            "center_in_frame": bool(z > 1e-6 and 0 <= uv[0] < float(frame["w"]) and 0 <= uv[1] < float(frame["h"])),
            "principal_point": [float(intr[0, 2]), float(intr[1, 2])],
            "approx_radius_px": float(max(intr[0, 0], intr[1, 1]) * radius / max(abs(z), 1e-6)),
        })
    in_frame = sum(1 for row in rows if row["center_in_frame"])
    positive = sum(1 for row in rows if row["center_z"] > 1e-6)
    center_errors = []
    radius_values = []
    for row in rows:
        center_errors.append(float(np.linalg.norm(np.asarray(row["center_uv"]) - np.asarray(row["principal_point"], dtype=np.float64))))
        radius_values.append(float(row["approx_radius_px"]))
    return {
        "positive_depth": int(positive),
        "in_frame": int(in_frame),
        "num_views": int(len(rows)),
        "median_center_error_px": float(np.median(center_errors)) if center_errors else float("inf"),
        "median_approx_radius_px": float(np.median(radius_values)) if radius_values else float("inf"),
        "rows": rows,
    }


def _workspace_image_max_dim(workspace: MultiViewWorkspace) -> float:
    for path in [workspace.aligned_transforms_path, workspace.colmap_transforms_path]:
        if not path.exists():
            continue
        try:
            db = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        dims = [
            max(float(frame.get("w", 0.0)), float(frame.get("h", 0.0)))
            for frame in db.get("frames", [])
            if frame.get("w") and frame.get("h")
        ]
        if dims:
            return max(dims)
    return 4096.0


def _should_soft_accept_landmark_calibration(
    before_score: dict,
    after_score: dict,
    landmark_px: float,
    landmark_limit: float,
    max_dim: float,
) -> bool:
    num_views = max(int(after_score.get("num_views", 0)), 1)
    before_in_frame = int(before_score.get("in_frame", 0))
    after_in_frame = int(after_score.get("in_frame", 0))
    before_error = float(before_score.get("median_center_error_px", float("inf")))
    after_error = float(after_score.get("median_center_error_px", float("inf")))
    after_radius = float(after_score.get("median_approx_radius_px", float("inf")))

    landmark_soft_limit = max(landmark_limit * 1.10, landmark_limit + 48.0)
    min_coverage_gain = max(3, int(np.ceil(0.15 * num_views)))
    target_in_frame = min(num_views, max(int(np.ceil(0.90 * num_views)), before_in_frame + min_coverage_gain))
    error_strongly_improved = (
        not np.isfinite(before_error)
        or after_error <= before_error * 0.45
        or after_error <= before_error - 0.20 * max_dim
    )
    return (
        np.isfinite(landmark_px)
        and landmark_px <= landmark_soft_limit
        and after_in_frame >= target_in_frame
        and np.isfinite(after_error)
        and after_error <= 0.12 * max_dim
        and error_strongly_improved
        and np.isfinite(after_radius)
        and after_radius <= 0.85 * max_dim
    )


def _face_box_metrics_for_workspace(workspace: MultiViewWorkspace, delta: Optional[dict] = None) -> dict:
    try:
        from app_multiview_gaussian_refine import build_lam
    except Exception:
        return {"valid": False, "reason": "lam_builder_unavailable"}
    try:
        lam_model = build_lam()
        batch = load_multiview_bundle(workspace.root, require_undistorted=False).to("cuda", torch.float32)
        query_points, flame_params = lam_model.renderer.get_query_points(batch.flame_params, device=batch.images.device)
        target, valid, h, w = _prepare_landmark_targets(batch)
        c2ws = batch.c2ws
        if delta is not None:
            c2ws = _apply_delta_to_c2ws(
                batch.c2ws,
                float(delta["scale"]),
                torch.as_tensor(np.asarray(delta["rotation"], dtype=np.float32), device=batch.images.device, dtype=batch.images.dtype),
                torch.as_tensor(np.asarray(delta["translation"], dtype=np.float32), device=batch.images.device, dtype=batch.images.dtype),
            )
        with torch.no_grad():
            pred = _project_flame_landmarks(lam_model.renderer, query_points, flame_params, c2ws, batch.intrs, h, w)
        return _face_box_metrics_from_points(pred, target, valid)
    except Exception as exc:
        return {"valid": False, "reason": f"face_box_eval_failed: {exc}"}


def _validate_landmark_calibration_delta(workspace: MultiViewWorkspace, before_score: dict, candidate: dict) -> tuple[bool, str, dict]:
    delta = {
        "scale": float(candidate["scale"]),
        "rotation": np.asarray(candidate["rotation"], dtype=np.float64),
        "translation": np.asarray(candidate["translation"], dtype=np.float64),
    }
    after_score = _score_workspace_geometry(workspace, delta=delta)
    after_face_box = _face_box_metrics_for_workspace(workspace, delta=delta)
    num_views = max(int(before_score.get("num_views", 0)), 1)
    max_dim = _workspace_image_max_dim(workspace)
    landmark_px = float(candidate.get("landmark_px", float("inf")))
    landmark_limit = max(250.0, 0.12 * max_dim)
    soft_accept = _should_soft_accept_landmark_calibration(before_score, after_score, landmark_px, landmark_limit, max_dim)
    face_box_accept = False
    if after_face_box.get("valid"):
        face_box_accept = (
            float(after_face_box.get("center_px_median", float("inf"))) <= 0.06 * max_dim
            and float(after_face_box.get("size_rel_median", float("inf"))) <= 0.18
        )
    if not np.isfinite(landmark_px) or (landmark_px > landmark_limit and not soft_accept and not face_box_accept):
        return False, f"landmark residual is too high ({landmark_px:.1f}px > {landmark_limit:.1f}px)", after_score
    scale = float(candidate["scale"])
    if scale <= LANDMARK_CALIBRATE_SCALE_MIN + 0.01 or scale >= LANDMARK_CALIBRATE_SCALE_MAX - 0.01:
        return False, f"scale hit optimizer bound ({scale:.3f})", after_score
    if after_score["positive_depth"] < before_score.get("positive_depth", 0):
        return False, "calibration reduces positive-depth views", after_score
    min_required_in_frame = max(1, int(np.ceil(0.5 * num_views)))
    if after_score["in_frame"] < min_required_in_frame:
        return False, "calibration moves the model center out of most frames", after_score
    if after_score["in_frame"] + 1 < before_score.get("in_frame", 0):
        return False, "calibration makes center-in-frame coverage worse", after_score
    before_error = float(before_score.get("median_center_error_px", float("inf")))
    after_error = float(after_score.get("median_center_error_px", float("inf")))
    if np.isfinite(before_error) and after_error > max(before_error * 1.25, before_error + 0.05 * max_dim):
        return False, "calibration increases median center error", after_score
    before_radius = float(before_score.get("median_approx_radius_px", float("inf")))
    after_radius = float(after_score.get("median_approx_radius_px", float("inf")))
    radius_limit = max(before_radius * 1.8, before_radius + 0.20 * max_dim)
    if np.isfinite(before_radius) and after_radius > radius_limit and not soft_accept:
        return False, "calibration makes projected radius much larger", after_score
    if face_box_accept:
        return True, "accepted via face-box alignment", after_score
    reason = "accepted"
    if soft_accept and (landmark_px > landmark_limit or (np.isfinite(before_radius) and after_radius > radius_limit)):
        reason = "accepted via strong geometry improvement"
    return True, reason, after_score


def _append_landmark_calibration_acceptance(
    workspace: MultiViewWorkspace,
    accepted: bool,
    reason: str,
    before: dict,
    after: dict,
) -> None:
    report_path = workspace.alignment_dir / "landmark_calibration_report.json"
    report = {}
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
    report["accepted"] = bool(accepted)
    report["acceptance_reason"] = reason
    report["geometry_before"] = _compact_geometry_score(before)
    report["geometry_after"] = _compact_geometry_score(after)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def _compact_geometry_score(score: dict) -> dict:
    return {
        "positive_depth": score.get("positive_depth"),
        "in_frame": score.get("in_frame"),
        "num_views": score.get("num_views"),
        "median_center_error_px": score.get("median_center_error_px"),
        "median_approx_radius_px": score.get("median_approx_radius_px"),
    }


def _apply_global_render_delta_np(
    c2w: np.ndarray,
    scale: float,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    out = np.asarray(c2w, dtype=np.float64).copy()
    rotation = np.asarray(rotation, dtype=np.float64)
    translation = np.asarray(translation, dtype=np.float64)
    out[:3, :3] = rotation @ out[:3, :3]
    out[:3, 3] = float(scale) * (rotation @ out[:3, 3]) + translation
    return out


def _write_projection_diagnostics(workspace: MultiViewWorkspace, lam_model=None, max_views: int = 9) -> Path:
    geometry_score = _score_workspace_geometry(workspace)
    rows = geometry_score["rows"]

    render_rows = []
    if lam_model is not None:
        batch = load_multiview_bundle(workspace.root, require_undistorted=False).to("cuda", torch.float32)
        gs = _load_renderable_gaussian(workspace.init_ply_path)
        gs.to_cuda()
        query_points, flame_params = lam_model.renderer.get_query_points(batch.flame_params, device=batch.images.device)
        sub_batch = _subsample_for_alignment(batch, max_views=max_views, scale=0.25)
        sub_flame = {k: v if k == "betas" else v[:, sub_batch.view_indices] for k, v in flame_params.items()}
        with torch.no_grad():
            out = render_animate_gs_with_intrinsics(
                lam_model.renderer,
                [gs],
                query_points,
                sub_flame,
                sub_batch.c2ws,
                sub_batch.intrs,
                sub_batch.images.shape[-2],
                sub_batch.images.shape[-1],
                sub_batch.bg_colors,
            )
        pred_mask = out["comp_mask"].detach()
        for idx, view_idx in enumerate(sub_batch.view_indices.detach().cpu().tolist()):
            render_rows.append({
                "view_index": int(view_idx),
                "frame_id": sub_batch.frame_ids[idx],
                "pred_mask_coverage": float((pred_mask[0, idx] > 0.02).float().mean().cpu()),
                "pred_mask_mean": float(pred_mask[0, idx].mean().cpu()),
            })

    report = {
        "geometry": rows,
        "render": render_rows,
        "any_center_in_frame": any(row["center_in_frame"] for row in rows),
        "any_render_mask": any(row["pred_mask_coverage"] > 1e-5 for row in render_rows) if render_rows else None,
    }
    path = workspace.alignment_dir / "projection_diagnostics.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if render_rows and not report["any_render_mask"]:
        raise RuntimeError(
            "Initial Sim3 is invalid: Gaussian renders an empty mask in all sampled views. "
            f"See {path} and rerun Initialize Sim3 after fixing camera alignment."
        )
    return path


def _ensure_workspace_landmarks(workspace: MultiViewWorkspace) -> int:
    images = sorted([p for p in workspace.images_dir.glob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"}])
    workspace.landmark_dir.mkdir(parents=True, exist_ok=True)
    restored = 0
    for image_path in images:
        dst = workspace.landmark_dir / f"{image_path.stem}.npz"
        if dst.exists():
            continue
        src = workspace.tracking_dir / "preprocess" / image_path.stem / "landmark2d" / "landmarks.npz"
        meta_path = workspace.tracking_dir / "preprocess" / image_path.stem / "preprocess_meta.json"
        if not src.exists() or not meta_path.exists():
            continue
        meta = _load_preprocess_meta(meta_path, image_path)
        _restore_tracking_landmarks(src, image_path, meta, dst)
        restored += 1
    if restored:
        db = json.loads(workspace.aligned_transforms_path.read_text(encoding="utf-8"))
        for frame in db.get("frames", []):
            name = Path(frame.get("image_name") or frame.get("file_path") or "").stem
            landmark = workspace.landmark_dir / f"{name}.npz"
            if landmark.exists():
                frame["landmark_path"] = str(landmark.relative_to(workspace.root)).replace("\\", "/")
        workspace.aligned_transforms_path.write_text(json.dumps(db, indent=2), encoding="utf-8")
    return restored


def _assert_projection_not_empty(workspace: MultiViewWorkspace) -> None:
    diagnostics = workspace.alignment_dir / "projection_diagnostics.json"
    if not diagnostics.exists():
        return
    report = json.loads(diagnostics.read_text(encoding="utf-8"))
    if report.get("any_render_mask") is False:
        raise RuntimeError("Sim3 invalid, rerun Initialize Sim3. Projection diagnostics show an empty Gaussian render.")


def _assert_colmap_reconstruction_usable(workspace: MultiViewWorkspace) -> None:
    transforms_path = workspace.colmap_dir / "transforms_colmap_sparse.json"
    if not transforms_path.exists():
        return
    try:
        db = json.loads(transforms_path.read_text(encoding="utf-8"))
    except Exception:
        return
    expected_images = len([p for p in workspace.images_dir.glob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"}])
    model_dir = db.get("colmap_model_dir")
    if not model_dir:
        return
    quality = assess_colmap_model_quality(model_dir, expected_images)
    if quality.get("accepted"):
        return
    raise RuntimeError(
        "COLMAP reconstruction is not usable for alignment: "
        f"registered {quality.get('registered_images', 0)}/{quality.get('expected_images', expected_images)} images, "
        f"points3D {quality.get('points3d', 0)}. "
        "Rerun COLMAP or import a denser sparse model before alignment."
    )


def _assert_alignment_ready_for_refine(workspace: MultiViewWorkspace) -> None:
    calibration_report = workspace.alignment_dir / "landmark_calibration_report.json"
    if not calibration_report.exists():
        return
    report = json.loads(calibration_report.read_text(encoding="utf-8"))
    if report.get("accepted") is not False:
        return

    geometry = report.get("geometry_before") or {}
    in_frame = int(geometry.get("in_frame") or 0)
    num_views = int(geometry.get("num_views") or 0)
    median_error = float(geometry.get("median_center_error_px") or float("inf"))
    max_dim = _workspace_image_max_dim(workspace)
    min_in_frame = max(1, int(np.ceil(0.75 * max(num_views, 1))))
    if in_frame >= min_in_frame and median_error <= 0.35 * max_dim:
        return

    reason = report.get("acceptance_reason", "landmark calibration failed sanity checks")
    raise RuntimeError(
        "Alignment is not reliable enough for refinement: "
        f"landmark calibration was rejected ({reason}); initial center coverage is {in_frame}/{num_views}. "
        "Rerun Initialize/Calibrate Alignment or inspect alignment/projection_diagnostics.json before refining."
    )


def _load_refine_train_view_selection(workspace: MultiViewWorkspace) -> dict:
    per_view_report = workspace.alignment_dir / "per_view_camera_alignment_report.json"
    if per_view_report.exists():
        try:
            data = json.loads(per_view_report.read_text(encoding="utf-8"))
            selection = data.get("optimization_inlier_views") or {}
            selected = [int(v) for v in selection.get("selected_view_indices", [])]
            frame_ids = [str(v) for v in selection.get("selected_frame_ids", [])]
            total = int(selection.get("num_views") or 0)
            if selected:
                return {
                    "source": "per_view_camera_alignment",
                    "selected_view_indices": selected,
                    "selected_frame_ids": frame_ids,
                    "outlier_frame_ids": [str(v) for v in selection.get("outlier_frame_ids", [])],
                    "num_views": total if total > 0 else max(selected) + 1,
                }
        except Exception:
            pass

    calibration_report = workspace.alignment_dir / "landmark_calibration_report.json"
    if calibration_report.exists():
        try:
            data = json.loads(calibration_report.read_text(encoding="utf-8"))
            selected = [int(v) for v in data.get("used_view_indices", [])]
            frame_ids = [str(v) for v in data.get("used_frame_ids", [])]
            if selected:
                return {
                    "source": "landmark_global_sim3",
                    "selected_view_indices": selected,
                    "selected_frame_ids": frame_ids,
                    "outlier_frame_ids": [],
                    "num_views": int(data.get("geometry_after", {}).get("num_views") or data.get("num_views") or max(selected) + 1),
                }
        except Exception:
            pass

    try:
        batch = load_multiview_bundle(workspace.root, require_undistorted=False)
        num_views = int(batch.c2ws.shape[1])
    except Exception:
        num_views = 0
    return {
        "source": "all_views",
        "selected_view_indices": list(range(num_views)),
        "selected_frame_ids": [str(v) for v in range(num_views)],
        "outlier_frame_ids": [],
        "num_views": num_views,
    }


def _intrinsic_from_frame(frame: dict) -> np.ndarray:
    intr = np.eye(4, dtype=np.float64)
    intr[0, 0] = float(frame["fl_x"])
    intr[1, 1] = float(frame["fl_y"])
    intr[0, 2] = float(frame["cx"])
    intr[1, 2] = float(frame["cy"])
    return intr


def _project_point_np(point: np.ndarray, c2w_render: np.ndarray, intr: np.ndarray) -> tuple[np.ndarray, float]:
    w2c = np.linalg.inv(c2w_render)
    homog = np.concatenate([np.asarray(point, dtype=np.float64), np.ones(1)])
    cam = (w2c @ homog)[:3]
    z = float(cam[2])
    denom = z if abs(z) > 1e-8 else np.sign(z) * 1e-8 if z != 0 else 1e-8
    uv = np.array([
        intr[0, 0] * cam[0] / denom + intr[0, 2],
        intr[1, 1] * cam[1] / denom + intr[1, 2],
    ], dtype=np.float64)
    return uv, z


def _subsample_for_alignment(batch, max_views: int = 9, scale: float = 0.25):
    num = batch.c2ws.shape[1]
    if num > max_views:
        idx = torch.linspace(0, num - 1, max_views, device=batch.images.device).round().long().unique()
        batch = batch.index(idx)
    if scale >= 0.999:
        return batch
    h, w = batch.images.shape[-2:]
    size = (max(1, int(round(h * scale))), max(1, int(round(w * scale))))
    images = F.interpolate(batch.images.flatten(0, 1), size=size, mode="bilinear", align_corners=False).reshape(batch.images.shape[:2] + batch.images.shape[2:3] + size)
    masks = F.interpolate(batch.masks.flatten(0, 1), size=size, mode="bilinear", align_corners=False).reshape(batch.masks.shape[:2] + batch.masks.shape[2:3] + size)
    intrs = batch.intrs.clone()
    intrs[..., 0, 0] *= scale
    intrs[..., 1, 1] *= scale
    intrs[..., 0, 2] *= scale
    intrs[..., 1, 2] *= scale
    return type(batch)(images, masks, batch.c2ws, intrs, batch.bg_colors, batch.flame_params, batch.frame_ids, batch.landmarks_2d, batch.view_indices)


def _blackbox_calibrate_sim3(renderer, gs, query_points, flame_params, batch, workspace: MultiViewWorkspace) -> dict:
    sub_batch = _subsample_for_alignment(batch, max_views=7, scale=0.25)
    sub_flame = {k: v if k == "betas" else v[:, sub_batch.view_indices] for k, v in flame_params.items()}
    rotations = _candidate_rotations()
    scale_factors = [0.7, 0.85, 1.0, 1.15, 1.35]
    shifts = [0.0, -0.15, 0.15]
    candidates = []
    with torch.no_grad():
        for rot in rotations:
            for scale in scale_factors:
                for tx in shifts:
                    for ty in shifts:
                        for tz in shifts:
                            translation = torch.tensor([tx, ty, tz], device=sub_batch.c2ws.device, dtype=sub_batch.c2ws.dtype)
                            c2ws = _apply_delta_to_c2ws(sub_batch.c2ws, scale, rot.to(sub_batch.c2ws.device, sub_batch.c2ws.dtype), translation)
                            out = render_animate_gs_with_intrinsics(
                                renderer,
                                [gs],
                                query_points,
                                sub_flame,
                                c2ws,
                                sub_batch.intrs,
                                sub_batch.images.shape[-2],
                                sub_batch.images.shape[-1],
                                sub_batch.bg_colors,
                            )
                            metrics = _score_alignment_render(out["comp_rgb"], out["comp_mask"], sub_batch.images, sub_batch.masks)
                            candidates.append({
                                "score": metrics["score"],
                                "scale": float(scale),
                                "rotation": rot.detach().cpu().numpy(),
                                "translation": translation.detach().cpu().numpy(),
                                "metrics": metrics,
                            })
    best = max(candidates, key=lambda item: item["score"])
    serializable = []
    for item in candidates:
        serializable.append({
            "score": float(item["score"]),
            "scale": float(item["scale"]),
            "rotation": item["rotation"].astype(float).tolist(),
            "translation": item["translation"].astype(float).tolist(),
            "metrics": item["metrics"],
        })
    report = {
        "source": "blackbox_global_sim3_grid",
        "best": {
            "score": float(best["score"]),
            "scale": float(best["scale"]),
            "rotation": best["rotation"].astype(float).tolist(),
            "translation": best["translation"].astype(float).tolist(),
            "metrics": best["metrics"],
        },
        "num_candidates": len(serializable),
        "candidates": sorted(serializable, key=lambda item: item["score"], reverse=True)[:50],
    }
    (workspace.alignment_dir / "blackbox_calibration_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if best["metrics"]["pred_mask_coverage"] <= 1e-5 or best["metrics"].get("pred_mask_min_coverage", 0.0) <= 1e-5:
        raise RuntimeError("Black-box calibration failed: the best candidate still has empty sampled views.")
    return best


def _prepare_landmark_targets(batch) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    if batch.landmarks_2d is None:
        raise FileNotFoundError("Missing 2D landmarks. Run or import FLAME tracking landmarks first.")
    h, w = batch.images.shape[-2:]
    target = batch.landmarks_2d[..., :2]
    if _landmarks_are_normalized(target):
        target = target.clone()
        target[..., 0] *= float(w)
        target[..., 1] *= float(h)
    valid = torch.isfinite(target).all(dim=-1)
    if batch.landmarks_2d.shape[-1] > 2:
        valid = valid & (batch.landmarks_2d[..., 2] > 0)
    return target, valid, h, w


def _face_box_stats_torch(points: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    fill_max = torch.full_like(points, 1e6)
    fill_min = torch.full_like(points, -1e6)
    mins = torch.where(valid.unsqueeze(-1), points, fill_max).amin(dim=2)
    maxs = torch.where(valid.unsqueeze(-1), points, fill_min).amax(dim=2)
    center = 0.5 * (mins + maxs)
    size = torch.linalg.norm((maxs - mins).clamp_min(1.0), dim=-1)
    good = valid.sum(dim=2) >= 5
    return center, size, good


def _face_box_metrics_from_points(pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> dict:
    count = min(pred.shape[2], target.shape[2], valid.shape[2])
    pred_used = pred[:, :, :count]
    target_used = target[:, :, :count]
    valid_used = valid[:, :, :count]
    pred_center, pred_size, pred_good = _face_box_stats_torch(pred_used, valid_used)
    target_center, target_size, target_good = _face_box_stats_torch(target_used, valid_used)
    use = pred_good & target_good
    if not bool(use.any()):
        return {"valid": False, "used_views": 0, "num_views": int(target.shape[1]), "coverage": 0.0}
    center_error = torch.linalg.norm(pred_center[use] - target_center[use], dim=-1)
    size_ratio = pred_size[use] / target_size[use].clamp_min(1.0)
    size_rel = (size_ratio - 1.0).abs()
    return {
        "valid": True,
        "used_views": int(use.sum().item()),
        "num_views": int(target.shape[1]),
        "coverage": float(use.sum().item() / max(target.shape[1], 1)),
        "center_px_median": float(torch.median(center_error).detach().cpu()),
        "center_px_mean": float(center_error.mean().detach().cpu()),
        "size_ratio_median": float(torch.median(size_ratio).detach().cpu()),
        "size_ratio_mean": float(size_ratio.mean().detach().cpu()),
        "size_rel_median": float(torch.median(size_rel).detach().cpu()),
        "size_rel_mean": float(size_rel.mean().detach().cpu()),
    }


def _alignment_view_diagnostics(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    image_hw: tuple[int, int],
    frame_ids: Optional[list[str]] = None,
    view_indices: Optional[torch.Tensor] = None,
) -> list[dict]:
    count = min(pred.shape[2], target.shape[2], valid.shape[2])
    pred_used = pred[:, :, :count]
    target_used = target[:, :, :count]
    valid_used = valid[:, :, :count]
    h, w = image_hw
    weights = _landmark_semantic_weights_torch(count, pred.device, pred.dtype).reshape(1, 1, count)
    px_error = torch.linalg.norm(pred_used - target_used, dim=-1)
    valid_f = valid_used.to(dtype=pred.dtype)
    weighted_denom = (weights * valid_f).sum(dim=-1).clamp_min(1.0)
    landmark_px = ((px_error * weights * valid_f).sum(dim=-1) / weighted_denom).squeeze(0)
    pred_center, pred_size, pred_good = _face_box_stats_torch(pred_used, valid_used)
    target_center, target_size, target_good = _face_box_stats_torch(target_used, valid_used)
    center_error = torch.linalg.norm(pred_center - target_center, dim=-1).squeeze(0)
    size_ratio = (pred_size / target_size.clamp_min(1.0)).squeeze(0)
    size_rel = (size_ratio - 1.0).abs()
    in_frame = (
        valid_used
        & (pred_used[..., 0] >= 0.0)
        & (pred_used[..., 0] < float(w))
        & (pred_used[..., 1] >= 0.0)
        & (pred_used[..., 1] < float(h))
    )
    valid_count = valid_used.sum(dim=-1).squeeze(0)
    in_frame_ratio = (in_frame.to(dtype=pred.dtype).sum(dim=-1) / valid_f.sum(dim=-1).clamp_min(1.0)).squeeze(0)
    if view_indices is None:
        source_indices = list(range(pred.shape[1]))
    else:
        source_indices = [int(v) for v in view_indices.detach().cpu().tolist()]
    rows = []
    for local_idx in range(pred.shape[1]):
        frame_id = frame_ids[local_idx] if frame_ids and local_idx < len(frame_ids) else str(local_idx)
        rows.append({
            "batch_index": int(local_idx),
            "view_index": int(source_indices[local_idx]),
            "frame_id": frame_id,
            "usable": bool(valid_count[local_idx].item() >= 5 and pred_good[0, local_idx].item() and target_good[0, local_idx].item()),
            "valid_landmarks": int(valid_count[local_idx].item()),
            "landmark_px": float(landmark_px[local_idx].detach().cpu()),
            "center_px": float(center_error[local_idx].detach().cpu()),
            "size_ratio": float(size_ratio[local_idx].detach().cpu()),
            "size_rel": float(size_rel[local_idx].detach().cpu()),
            "in_frame_ratio": float(in_frame_ratio[local_idx].detach().cpu()),
        })
    return rows


def _select_alignment_inlier_views(
    rows: list[dict],
    image_hw: tuple[int, int],
    min_views: Optional[int] = None,
) -> dict:
    h, w = image_hw
    max_dim = float(max(h, w))
    usable = [
        row for row in rows
        if row.get("usable")
        and np.isfinite(float(row.get("landmark_px", float("inf"))))
        and np.isfinite(float(row.get("center_px", float("inf"))))
        and np.isfinite(float(row.get("size_rel", float("inf"))))
    ]
    if not usable:
        selected = [int(row["view_index"]) for row in rows]
        return {
            "num_views": len(rows),
            "usable_views": 0,
            "num_inliers": len(selected),
            "selected_view_indices": selected,
            "selected_frame_ids": [str(row["frame_id"]) for row in rows],
            "outlier_view_indices": [],
            "outlier_frame_ids": [],
            "thresholds": {"reason": "no_usable_views"},
            "rows": [{**row, "is_inlier": True} for row in rows],
        }

    usable_count = len(usable)
    target_count = max(
        min_views or 0,
        ALIGNMENT_INLIER_MIN_VIEWS,
        int(np.ceil(ALIGNMENT_INLIER_MIN_FRACTION * usable_count)),
    )
    target_count = max(1, min(target_count, usable_count))

    center = np.asarray([float(row["center_px"]) for row in usable], dtype=np.float64)
    landmark = np.asarray([float(row["landmark_px"]) for row in usable], dtype=np.float64)
    size_rel = np.asarray([float(row["size_rel"]) for row in usable], dtype=np.float64)
    in_frame_ratio = np.asarray([float(row["in_frame_ratio"]) for row in usable], dtype=np.float64)

    center_med = float(np.median(center))
    center_mad = float(np.median(np.abs(center - center_med)))
    landmark_med = float(np.median(landmark))
    landmark_mad = float(np.median(np.abs(landmark - landmark_med)))
    size_med = float(np.median(size_rel))
    size_mad = float(np.median(np.abs(size_rel - size_med)))

    center_limit = min(
        max(float(np.quantile(center, 0.70)), center_med + 1.5 * max(center_mad, 8.0)),
        0.22 * max_dim,
    )
    landmark_limit = min(
        max(float(np.quantile(landmark, 0.70)), landmark_med + 1.5 * max(landmark_mad, 12.0)),
        0.22 * max_dim,
    )
    size_limit = min(
        max(float(np.quantile(size_rel, 0.70)), size_med + 1.5 * max(size_mad, 0.015), 0.08),
        0.35,
    )
    in_frame_limit = min(max(0.40, float(np.quantile(in_frame_ratio, 0.25))), 0.85)

    initial = [
        row for row in usable
        if float(row["center_px"]) <= center_limit
        and float(row["landmark_px"]) <= landmark_limit
        and float(row["size_rel"]) <= size_limit
        and float(row["in_frame_ratio"]) >= in_frame_limit
    ]
    if len(initial) >= target_count:
        selected = sorted(int(row["view_index"]) for row in initial)
    else:
        center_scale = max(center_med, 1.0)
        landmark_scale = max(landmark_med, 1.0)
        size_scale = max(size_med, 0.02)
        scored = []
        for row in usable:
            score = (
                float(row["center_px"]) / center_scale
                + 0.75 * float(row["landmark_px"]) / landmark_scale
                + 0.50 * float(row["size_rel"]) / size_scale
                + 0.30 * max(0.0, in_frame_limit - float(row["in_frame_ratio"]))
            )
            scored.append((score, int(row["view_index"])))
        scored.sort(key=lambda item: item[0])
        selected = sorted(idx for _, idx in scored[:target_count])

    selected_set = set(selected)
    enriched_rows = [{**row, "is_inlier": bool(int(row["view_index"]) in selected_set)} for row in rows]
    selected_frame_ids = [str(row["frame_id"]) for row in enriched_rows if row["is_inlier"]]
    outlier_rows = [row for row in enriched_rows if not row["is_inlier"]]
    return {
        "num_views": len(rows),
        "usable_views": usable_count,
        "num_inliers": len(selected),
        "selected_view_indices": selected,
        "selected_frame_ids": selected_frame_ids,
        "outlier_view_indices": [int(row["view_index"]) for row in outlier_rows],
        "outlier_frame_ids": [str(row["frame_id"]) for row in outlier_rows],
        "thresholds": {
            "center_px": float(center_limit),
            "landmark_px": float(landmark_limit),
            "size_rel": float(size_limit),
            "in_frame_ratio": float(in_frame_limit),
            "target_count": int(target_count),
        },
        "rows": enriched_rows,
    }


def _summarize_alignment_inlier_views(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    image_hw: tuple[int, int],
    frame_ids: Optional[list[str]] = None,
    view_indices: Optional[torch.Tensor] = None,
    min_views: Optional[int] = None,
) -> dict:
    rows = _alignment_view_diagnostics(pred, target, valid, image_hw, frame_ids=frame_ids, view_indices=view_indices)
    return _select_alignment_inlier_views(rows, image_hw, min_views=min_views)


def _coarse_alignment_score_tuple(face_box: dict, landmark_px: float, coverage_views: int) -> tuple:
    return (
        int(bool(face_box.get("valid"))),
        int(coverage_views),
        -float(face_box.get("center_px_median", float("inf"))),
        -float(face_box.get("size_rel_median", float("inf"))),
        -float(landmark_px),
        float(face_box.get("coverage", 0.0)),
    )


def _landmark_coarse_seed(renderer, query_points, flame_params, batch, workspace: MultiViewWorkspace) -> dict:
    target, valid, h, w = _prepare_landmark_targets(batch)
    weights = _landmark_semantic_weights_torch(min(batch.landmarks_2d.shape[2], 68), batch.images.device, batch.images.dtype)
    scale_values = [0.75, 1.0, 1.25, 1.5, 1.8, 2.2, 2.8]
    yaw_pitch_roll = [
        (0.0, 0.0, 0.0),
        (-15.0, 0.0, 0.0),
        (15.0, 0.0, 0.0),
        (0.0, -10.0, 0.0),
        (0.0, 10.0, 0.0),
        (0.0, 0.0, -10.0),
        (0.0, 0.0, 10.0),
    ]
    shift_values = [-0.75, 0.0, 0.75]
    depth_shift_values = [-0.3, 0.0, 0.3]
    image_shift = 0.06
    world_depth_shift = 0.08

    scored = []
    with torch.no_grad():
        for scale in scale_values:
            for yaw_deg, pitch_deg, roll_deg in yaw_pitch_roll:
                yaw = np.deg2rad(yaw_deg)
                pitch = np.deg2rad(pitch_deg)
                roll = np.deg2rad(roll_deg)
                rotation_np = (
                    _axis_angle_to_matrix_np(np.array([0.0, yaw, 0.0], dtype=np.float64))
                    @ _axis_angle_to_matrix_np(np.array([pitch, 0.0, 0.0], dtype=np.float64))
                    @ _axis_angle_to_matrix_np(np.array([0.0, 0.0, roll], dtype=np.float64))
                )
                rotation = torch.as_tensor(rotation_np, device=batch.images.device, dtype=batch.images.dtype)
                for tx_mul in shift_values:
                    for ty_mul in shift_values:
                        for tz_mul in depth_shift_values:
                            translation = torch.tensor(
                                [tx_mul * image_shift, ty_mul * image_shift, tz_mul * world_depth_shift],
                                device=batch.images.device,
                                dtype=batch.images.dtype,
                            )
                            c2ws = _apply_delta_to_c2ws(batch.c2ws, float(scale), rotation, translation)
                            pred = _project_flame_landmarks(renderer, query_points, flame_params, c2ws, batch.intrs, h, w)
                            count = min(pred.shape[2], target.shape[2], weights.shape[0])
                            pred_used = pred[:, :, :count]
                            target_used = target[:, :, :count]
                            valid_used = valid[:, :, :count]
                            if not bool(valid_used.any()):
                                continue
                            px_error = torch.linalg.norm(pred_used - target_used, dim=-1)
                            weights_used = weights[:count].reshape(1, 1, count)
                            valid_f = valid_used.to(dtype=pred.dtype)
                            landmark_px = float(((px_error * weights_used * valid_f).sum() / (weights_used * valid_f).sum().clamp_min(1.0)).detach().cpu())
                            in_frame = (
                                valid_used
                                & (pred_used[..., 0] >= 0.0)
                                & (pred_used[..., 0] < float(w))
                                & (pred_used[..., 1] >= 0.0)
                                & (pred_used[..., 1] < float(h))
                            )
                            valid_views = torch.any(valid_used, dim=-1)
                            in_frame_views = torch.any(in_frame, dim=-1)
                            num_views = int(valid_views.sum().item())
                            covered = int(in_frame_views.sum().item())
                            face_box = _face_box_metrics_from_points(pred_used, target_used, valid_used)
                            scored.append(
                                {
                                    "label": f"s{scale:.2f}_y{yaw_deg:.0f}_p{pitch_deg:.0f}_r{roll_deg:.0f}_tx{tx_mul:.2f}_ty{ty_mul:.2f}_tz{tz_mul:.2f}",
                                    "score": _coarse_alignment_score_tuple(face_box, landmark_px, covered),
                                    "landmark_px": landmark_px,
                                    "in_frame": covered,
                                    "num_views": num_views,
                                    "coverage": float(covered / max(num_views, 1)),
                                    "face_box": face_box,
                                    "scale": float(scale),
                                    "rotation": rotation.detach().cpu().numpy().astype(np.float64),
                                    "translation": translation.detach().cpu().numpy().astype(np.float64),
                                }
                            )
    if not scored:
        raise RuntimeError("Landmark coarse Sim3 search produced no valid candidates.")
    scored.sort(key=lambda item: item["score"], reverse=True)
    best = scored[0]
    report = {
        "source": "landmark_coarse_seed",
        "best": {
            "label": best["label"],
            "landmark_px": float(best["landmark_px"]),
            "in_frame": int(best["in_frame"]),
            "num_views": int(best["num_views"]),
            "coverage": float(best["coverage"]),
            "face_box": best["face_box"],
            "scale": float(best["scale"]),
            "rotation": best["rotation"].astype(float).tolist(),
            "translation": best["translation"].astype(float).tolist(),
        },
        "num_candidates": len(scored),
        "candidates": [
            {
                "label": item["label"],
                "landmark_px": float(item["landmark_px"]),
                "in_frame": int(item["in_frame"]),
                "num_views": int(item["num_views"]),
                "coverage": float(item["coverage"]),
                "face_box": item["face_box"],
                "scale": float(item["scale"]),
                "rotation": item["rotation"].astype(float).tolist(),
                "translation": item["translation"].astype(float).tolist(),
            }
            for item in scored[:24]
        ],
    }
    with torch.no_grad():
        best_rot = torch.as_tensor(best["rotation"], device=batch.images.device, dtype=batch.images.dtype)
        best_trans = torch.as_tensor(best["translation"], device=batch.images.device, dtype=batch.images.dtype)
        best_c2ws = _apply_delta_to_c2ws(batch.c2ws, float(best["scale"]), best_rot, best_trans)
        best_pred = _project_flame_landmarks(renderer, query_points, flame_params, best_c2ws, batch.intrs, h, w)
    inlier_views = _summarize_alignment_inlier_views(
        best_pred,
        target,
        valid,
        (h, w),
        frame_ids=batch.frame_ids,
        view_indices=batch.view_indices,
        min_views=max(ALIGNMENT_INLIER_MIN_VIEWS, int(np.ceil(0.60 * batch.c2ws.shape[1]))),
    )
    best["inlier_view_indices"] = list(inlier_views["selected_view_indices"])
    best["inlier_frame_ids"] = list(inlier_views["selected_frame_ids"])
    report["inlier_views"] = inlier_views
    (workspace.alignment_dir / "landmark_coarse_seed_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return best


def _landmark_calibrate_sim3(
    renderer,
    query_points,
    flame_params,
    batch,
    workspace: MultiViewWorkspace,
    init_scale: float = 1.0,
    init_rotation: Optional[np.ndarray] = None,
    init_translation: Optional[np.ndarray] = None,
) -> dict:
    if batch.landmarks_2d is None:
        raise FileNotFoundError("Missing 2D landmarks. Run or import FLAME tracking landmarks first.")
    h, w = batch.images.shape[-2:]
    target = batch.landmarks_2d[..., :2]
    if _landmarks_are_normalized(target):
        target = target.clone()
        target[..., 0] *= float(w)
        target[..., 1] *= float(h)
    valid = torch.isfinite(target).all(dim=-1)
    if batch.landmarks_2d.shape[-1] > 2:
        valid = valid & (batch.landmarks_2d[..., 2] > 0)
    valid_count = int(valid.sum().detach().cpu())
    if valid_count < 12:
        raise ValueError("Not enough valid 2D landmarks for landmark Sim3 calibration.")

    init_scale = float(np.clip(init_scale, LANDMARK_CALIBRATE_SCALE_MIN, LANDMARK_CALIBRATE_SCALE_MAX))
    init_rotation = np.eye(3, dtype=np.float64) if init_rotation is None else np.asarray(init_rotation, dtype=np.float64)
    init_translation = np.zeros(3, dtype=np.float64) if init_translation is None else np.asarray(init_translation, dtype=np.float64)

    log_scale = torch.nn.Parameter(torch.log(torch.tensor([init_scale], device=batch.images.device, dtype=batch.images.dtype)))
    axis_angle = torch.nn.Parameter(_matrix_to_axis_angle_torch(torch.as_tensor(init_rotation, device=batch.images.device, dtype=batch.images.dtype)))
    translation = torch.nn.Parameter(torch.as_tensor(init_translation, device=batch.images.device, dtype=batch.images.dtype).reshape(3))
    optimizer = torch.optim.Adam([log_scale, axis_angle, translation], lr=5e-3)
    history = []
    best = None
    for step in range(240):
        optimizer.zero_grad(set_to_none=True)
        scale = torch.exp(log_scale).clamp(LANDMARK_CALIBRATE_SCALE_MIN, LANDMARK_CALIBRATE_SCALE_MAX)
        rot = _axis_angle_to_matrix_torch(axis_angle).reshape(1, 1, 3, 3)
        new_rot = torch.matmul(rot, batch.c2ws[..., :3, :3])
        new_t = scale * torch.matmul(rot, batch.c2ws[..., :3, 3:4]).squeeze(-1) + translation.reshape(1, 1, 3)
        upper = torch.cat([new_rot, new_t.unsqueeze(-1)], dim=-1)
        c2ws = torch.cat([upper, batch.c2ws[..., 3:4, :]], dim=-2)
        pred = _project_flame_landmarks(renderer, query_points, flame_params, c2ws, batch.intrs, h, w)
        count = min(pred.shape[2], target.shape[2])
        pred_used = pred[:, :, :count]
        target_used = target[:, :, :count]
        valid_used = valid[:, :, :count]
        residual = (pred_used - target_used) / torch.tensor([float(w), float(h)], device=pred.device, dtype=pred.dtype)
        weights = _landmark_semantic_weights_torch(count, pred.device, pred.dtype).reshape(1, 1, count, 1)
        valid_f = valid_used.to(dtype=pred.dtype).unsqueeze(-1)
        data_map = F.smooth_l1_loss(residual, torch.zeros_like(residual), beta=0.002, reduction="none")
        denom = ((weights * valid_f).sum() * residual.shape[-1]).clamp_min(1.0)
        data_loss = (data_map * weights * valid_f).sum() / denom
        px_error = torch.linalg.norm(pred_used - target_used, dim=-1)
        px_loss = (px_error * valid_used.to(dtype=pred.dtype)).sum() / valid_used.to(dtype=pred.dtype).sum().clamp_min(1.0)
        reg = 0.005 * (log_scale.square().mean() + axis_angle.square().mean()) + 0.02 * translation.square().mean()
        loss = data_loss + reg
        loss.backward()
        optimizer.step()
        row = {
            "step": step + 1,
            "loss": float(loss.detach().cpu()),
            "data_loss": float(data_loss.detach().cpu()),
            "landmark_px": float(px_loss.detach().cpu()),
            "scale": float(scale.detach().cpu()),
            "translation": translation.detach().cpu().numpy().astype(float).tolist(),
        }
        history.append(row)
        if best is None or row["loss"] < best["loss"]:
            best = {
                **row,
                "rotation": rot.detach().cpu().numpy()[0, 0].astype(float).tolist(),
            }
    if best is None:
        raise RuntimeError("Landmark Sim3 calibration produced no candidates.")
    report = {
        "source": "landmark_global_sim3",
        "best": best,
        "history_tail": history[-20:],
        "valid_landmarks": valid_count,
        "num_views": int(batch.c2ws.shape[1]),
        "used_view_indices": [int(v) for v in batch.view_indices.detach().cpu().tolist()] if batch.view_indices is not None else list(range(int(batch.c2ws.shape[1]))),
        "used_frame_ids": list(batch.frame_ids),
    }
    report["face_box"] = _face_box_metrics_from_points(pred_used.detach(), target_used.detach(), valid_used.detach())
    (workspace.alignment_dir / "landmark_calibration_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return {
        "loss": best["loss"],
        "landmark_px": best["landmark_px"],
        "scale": best["scale"],
        "rotation": np.asarray(best["rotation"], dtype=np.float64),
        "translation": np.asarray(best["translation"], dtype=np.float64),
    }


def _axis_angle_to_matrix_torch(axis_angle: torch.Tensor) -> torch.Tensor:
    theta = torch.linalg.norm(axis_angle)
    x, y, z = axis_angle.unbind()
    zero = torch.zeros((), device=axis_angle.device, dtype=axis_angle.dtype)
    k = torch.stack([
        torch.stack([zero, -z, y]),
        torch.stack([z, zero, -x]),
        torch.stack([-y, x, zero]),
    ])
    eye = torch.eye(3, device=axis_angle.device, dtype=axis_angle.dtype)
    theta2 = theta * theta
    small = theta < 1e-4
    a = torch.where(small, 1.0 - theta2 / 6.0 + theta2 * theta2 / 120.0, torch.sin(theta) / theta.clamp_min(1e-8))
    b = torch.where(small, 0.5 - theta2 / 24.0 + theta2 * theta2 / 720.0, (1.0 - torch.cos(theta)) / theta2.clamp_min(1e-8))
    return eye + a * k + b * (k @ k)


def _matrix_to_axis_angle_torch(matrix: torch.Tensor) -> torch.Tensor:
    matrix = matrix.reshape(3, 3)
    trace = torch.trace(matrix)
    cos_theta = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0)
    theta = torch.acos(cos_theta)
    if bool(theta.detach().abs() < 1e-5):
        return torch.zeros(3, device=matrix.device, dtype=matrix.dtype)
    skew = (matrix - matrix.transpose(0, 1)) / (2.0 * torch.sin(theta).clamp_min(1e-8))
    axis = torch.stack([skew[2, 1], skew[0, 2], skew[1, 0]])
    return axis * theta


def _landmarks_are_normalized(landmarks: torch.Tensor) -> bool:
    finite = landmarks[torch.isfinite(landmarks)]
    return bool(finite.numel() > 0 and finite.detach().max() <= 2.0)


def _landmark_semantic_weights_torch(count: int, device, dtype) -> torch.Tensor:
    weights = torch.ones(count, device=device, dtype=dtype)
    if count >= 68:
        weights[:17] = 0.35
        weights[17:27] = 0.8
        weights[27:36] = 1.25
        weights[36:48] = 1.35
        weights[48:68] = 1.0
    return weights


def _candidate_rotations() -> list[torch.Tensor]:
    mats = [torch.eye(3)]
    for deg in [-30, -15, 15, 30]:
        mats.append(torch.tensor(_axis_angle_to_matrix_np(np.array([0.0, np.deg2rad(deg), 0.0])), dtype=torch.float32))
    return mats


def _apply_delta_to_c2ws(c2ws: torch.Tensor, scale: float, rot: torch.Tensor, translation: torch.Tensor) -> torch.Tensor:
    out = c2ws.clone()
    rot = rot.reshape(1, 1, 3, 3)
    out[..., :3, :3] = torch.matmul(rot, out[..., :3, :3])
    out[..., :3, 3] = float(scale) * torch.matmul(rot, out[..., :3, 3:4]).squeeze(-1) + translation.reshape(1, 1, 3)
    return out


def _score_alignment_render(pred_rgb: torch.Tensor, pred_mask: torch.Tensor, target_rgb: torch.Tensor, target_mask: torch.Tensor) -> dict:
    pred = pred_mask.clamp(0, 1)
    target = target_mask.clamp(0, 1)
    reduce_dims = tuple(range(2, pred.ndim))
    inter_per_view = (pred * target).sum(dim=reduce_dims)
    union_per_view = (pred + target - pred * target).sum(dim=reduce_dims).clamp_min(1.0)
    iou_per_view = inter_per_view / union_per_view
    coverage_per_view = (pred > 0.02).float().mean(dim=reduce_dims)
    target_coverage_per_view = (target > 0.2).float().mean(dim=reduce_dims).clamp_min(1e-6)
    coverage_ratio_per_view = torch.minimum(
        coverage_per_view / target_coverage_per_view,
        target_coverage_per_view / coverage_per_view.clamp_min(1e-6),
    )
    over_coverage = (coverage_per_view - target_coverage_per_view).clamp_min(0.0).mean()
    empty_penalty = (coverage_per_view < 0.05).float().mean()
    iou = iou_per_view.mean()
    coverage = coverage_per_view.mean()
    min_coverage = coverage_per_view.min()
    target_coverage = target_coverage_per_view.mean().clamp_min(1e-6)
    coverage_ratio = coverage_ratio_per_view.mean()
    rgb = ((pred_rgb - target_rgb).abs() * target).sum() / target.expand_as(pred_rgb).sum().clamp_min(1.0)
    score = 5.0 * iou + 0.25 * coverage_ratio - 1.5 * over_coverage - 2.0 * empty_penalty - 0.1 * rgb
    return {
        "score": float(score.detach().cpu()),
        "mask_iou": float(iou.detach().cpu()),
        "pred_mask_coverage": float(coverage.detach().cpu()),
        "pred_mask_min_coverage": float(min_coverage.detach().cpu()),
        "target_mask_coverage": float(target_coverage.detach().cpu()),
        "coverage_ratio": float(coverage_ratio.detach().cpu()),
        "over_coverage": float(over_coverage.detach().cpu()),
        "empty_view_fraction": float(empty_penalty.detach().cpu()),
        "rgb_l1": float(rgb.detach().cpu()),
    }


def _bake_render_delta_into_alignment(workspace: MultiViewWorkspace, scale: float, rotation: np.ndarray, translation: np.ndarray, source_suffix: str) -> None:
    aligned_db = json.loads(workspace.aligned_transforms_path.read_text(encoding="utf-8"))
    sim3 = Sim3Alignment.load(workspace.sim3_path)
    rotation = np.asarray(rotation, dtype=np.float64)
    translation = np.asarray(translation, dtype=np.float64)
    for frame in aligned_db.get("frames", []):
        render_c2w = _json_c2w_to_render_c2w(np.asarray(frame["transform_matrix"], dtype=np.float64))
        out = render_c2w.copy()
        out[:3, :3] = rotation @ out[:3, :3]
        out[:3, 3] = float(scale) * (rotation @ out[:3, 3]) + translation
        frame["transform_matrix_pre_alignment_delta"] = frame["transform_matrix"]
        frame["transform_matrix"] = _render_c2w_to_json_c2w(out).tolist()
    aligned_db["sim3_source"] = f"{sim3.source}+{source_suffix}"
    aligned_db["alignment_delta_source"] = source_suffix
    aligned_db.pop("blackbox_calibrated", None)
    workspace.aligned_transforms_path.write_text(json.dumps(aligned_db, indent=2), encoding="utf-8")
    composed = _compose_sim3_with_global_delta(sim3, scale, rotation, translation)
    composed.source = f"{sim3.source}+{source_suffix}"
    composed.save(workspace.sim3_path)


def _write_baked_export_alignment(workspace: MultiViewWorkspace) -> None:
    aligned_db = json.loads(workspace.aligned_transforms_path.read_text(encoding="utf-8"))
    sim3 = Sim3Alignment.load(workspace.sim3_path)
    camera_delta_path = workspace.refine_dir / "camera_delta.pt"
    intrinsics_delta_path = workspace.refine_dir / "intrinsics_delta.pt"
    camera_delta = _load_torch_state(camera_delta_path)
    intrinsics_delta = _load_torch_state(intrinsics_delta_path)

    global_scale, global_rot, global_translation = _global_camera_delta(camera_delta)
    composed_sim3 = _compose_sim3_with_global_delta(sim3, global_scale, global_rot, global_translation)
    composed_sim3.source = f"{sim3.source}+refine_global_delta" if camera_delta else sim3.source
    composed_sim3.save(workspace.exports_dir / workspace.sim3_path.name)

    frames = aligned_db.get("frames", [])
    sorted_indices = sorted(range(len(frames)), key=lambda idx: _frame_sort_key(frames[idx]))
    view_index_by_frame = {frame_idx: view_idx for view_idx, frame_idx in enumerate(sorted_indices)}
    baked_frames = []
    for frame_idx, frame in enumerate(frames):
        view_idx = view_index_by_frame[frame_idx]
        baked = dict(frame)
        c2w = _json_c2w_to_render_c2w(np.asarray(frame["transform_matrix"], dtype=np.float64))
        baked["transform_matrix_pre_refine"] = frame["transform_matrix"]
        baked["transform_matrix"] = _render_c2w_to_json_c2w(_apply_camera_delta_to_c2w(c2w, camera_delta, view_idx)).tolist()
        _apply_intrinsics_delta_to_frame(baked, intrinsics_delta, view_idx)
        baked_frames.append(baked)
    aligned_db["frames"] = baked_frames
    aligned_db["sim3_source"] = composed_sim3.source
    aligned_db["export_baked_refine_deltas"] = {
        "global_camera_delta": camera_delta is not None,
        "per_view_camera_delta": bool(camera_delta and "per_view_axis_angle" in camera_delta),
        "intrinsics_delta": intrinsics_delta is not None,
    }
    (workspace.exports_dir / workspace.aligned_transforms_path.name).write_text(json.dumps(aligned_db, indent=2), encoding="utf-8")
    (workspace.exports_dir / "export_alignment_report.json").write_text(
        json.dumps(
            {
                "sim3": "sim3_colmap_to_lam.json includes the optimized global camera delta when camera_delta.pt exists.",
                "transforms": "transforms_aligned.json has camera/intrinsics deltas baked into every frame.",
                "initial_backups": [
                    "sim3_colmap_to_lam.initial.json",
                    "transforms_aligned.initial.json",
                ],
                "camera_delta_found": camera_delta is not None,
                "intrinsics_delta_found": intrinsics_delta is not None,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _optimize_per_view_camera_alignment(
    workspace: MultiViewWorkspace,
    lam_model,
    init_ply: Path,
    steps: int = 180,
) -> dict:
    device = torch.device("cuda")
    dtype = torch.float32
    full_batch = load_multiview_bundle(workspace.root, require_undistorted=False).to(device, dtype)
    query_points, flame_params = lam_model.renderer.get_query_points(full_batch.flame_params, device=device)
    target_full, valid_full, full_h, full_w = _prepare_landmark_targets(full_batch)
    with torch.no_grad():
        pred_full = _project_flame_landmarks(lam_model.renderer, query_points, flame_params, full_batch.c2ws, full_batch.intrs, full_h, full_w)
    inlier_views = _summarize_alignment_inlier_views(
        pred_full,
        target_full,
        valid_full,
        (full_h, full_w),
        frame_ids=full_batch.frame_ids,
        view_indices=full_batch.view_indices,
        min_views=max(ALIGNMENT_INLIER_MIN_VIEWS, int(np.ceil(0.65 * full_batch.c2ws.shape[1]))),
    )
    selected = list(inlier_views["selected_view_indices"])
    batch = full_batch.index(selected) if 0 < len(selected) < full_batch.c2ws.shape[1] else full_batch
    gs = _load_renderable_gaussian(init_ply)
    gs.to_cuda()
    gs_state = OptimizableGaussianState(gs, query_points=query_points, knn_k=6, knn_max_points=30000).to(device=device)
    gs_state.set_trainable(False, False, False, False)
    camera_state = OptimizableCameraState(full_batch.c2ws.shape[1]).to(device=device)
    camera_state.set_trainable(False, True)
    intr_state = OptimizableIntrinsicsState(full_batch.c2ws.shape[1], max_focal_change=0.10, max_principal_delta=32.0).to(device=device)
    intr_state.set_trainable(True)
    intr_state.principal_delta.requires_grad_(False)
    exposure_state = OptimizableExposureState(full_batch.c2ws.shape[1]).to(device=device)
    exposure_state.set_trainable(False)
    expr_state = OptimizableExpressionState(full_batch.flame_params).to(device=device)
    expr_state.set_trainable(False)
    sampler = RoundRobinViewSampler(batch.c2ws.shape[1], device=device)
    full_alignment_eval = lambda: _evaluate_per_view_alignment_metrics_cached(
        lam_model,
        full_batch,
        query_points,
        flame_params,
        camera_state,
        intr_state,
        target_full,
        valid_full,
        full_h,
        full_w,
    )
    target_train, valid_train, train_h, train_w = _prepare_landmark_targets(batch)
    flame_params_train = _index_flame_params(flame_params, batch.view_indices)
    train_alignment_eval = lambda: _evaluate_per_view_alignment_metrics_cached(
        lam_model,
        batch,
        query_points,
        flame_params_train,
        camera_state,
        intr_state,
        target_train,
        valid_train,
        train_h,
        train_w,
    )

    stage = RefinementStageConfig(
        name="alignment_per_view_camera",
        steps=steps,
        lr=6e-4,
        views_per_step=min(6, batch.c2ws.shape[1]),
        optimize_per_view_camera=True,
        optimize_intrinsics=True,
        resolution_scale=0.5,
        use_ssim=True,
        use_rgb_loss=False,
        use_mask_loss=True,
        use_landmark=True,
        use_colmap_relative_reg=True,
    )
    weights = RefinementConfig().loss_weights
    optimizer = torch.optim.Adam(
        [
            {"params": [camera_state.per_view_axis_angle, camera_state.per_view_translation], "lr": stage.lr * 0.25},
            {"params": [intr_state.log_focal_scale], "lr": stage.lr * 0.08},
        ]
    )
    history = []
    best = None
    for step_idx in range(stage.steps):
        view_idx = sampler.next(stage.views_per_step)
        sub_batch = _scale_batch(batch.index(view_idx), stage.resolution_scale)
        sub_flame = _index_flame_params(flame_params, sub_batch.view_indices)
        optimizer.zero_grad(set_to_none=True)
        h, w = sub_batch.images.shape[-2:]
        c2ws = camera_state(sub_batch.c2ws, sub_batch.view_indices)
        intrs = intr_state(sub_batch.intrs, sub_batch.view_indices)
        render = render_animate_gs_with_intrinsics(
            lam_model.renderer,
            [gs_state.to_gaussian_model()],
            query_points,
            sub_flame,
            c2ws,
            intrs,
            h,
            w,
            sub_batch.bg_colors,
        )
        landmark_2d = _project_flame_landmarks(lam_model.renderer, query_points, sub_flame, c2ws, intrs, h, w)
        context = type("RenderContextLite", (), {
            "c2ws": c2ws,
            "base_c2ws": sub_batch.c2ws,
            "view_indices": sub_batch.view_indices,
            "intrs": intrs,
            "flame_params": sub_flame,
            "landmark_2d": landmark_2d,
        })()
        losses = compute_losses(
            dict(render),
            sub_batch,
            gs_state,
            camera_state,
            intr_state,
            exposure_state,
            expr_state,
            context,
            weights,
            stage,
        )
        focal_consistency = _shared_focal_consistency_penalty(intr_state)
        total = losses["total"] + 1.5 * focal_consistency
        total.backward()
        optimizer.step()

        full_eval = full_alignment_eval()
        train_eval = train_alignment_eval()
        row = {
            "step": step_idx + 1,
            "loss": float(total.detach().cpu()),
            "landmark_px": float(full_eval["landmark_px"]),
            "landmark_px_inliers": float(train_eval["landmark_px"]),
            "face_box": full_eval["face_box"],
            "face_box_inliers": train_eval["face_box"],
            "intrinsics_summary": full_eval["intrinsics_summary"],
        }
        history.append(row)
        key = (
            -int(bool(train_eval["face_box"].get("valid"))),
            float(train_eval["face_box"].get("center_px_median", float("inf"))),
            float(train_eval["face_box"].get("size_rel_median", float("inf"))),
            float(row["landmark_px_inliers"]),
        )
        if best is None or key < best["key"]:
            best = {
                "key": key,
                "step": step_idx + 1,
                "loss": row["loss"],
                "landmark_px": row["landmark_px"],
                "landmark_px_inliers": row["landmark_px_inliers"],
                "face_box": full_eval["face_box"],
                "face_box_inliers": train_eval["face_box"],
                "intrinsics_summary": full_eval["intrinsics_summary"],
                "state": {
                    "per_view_axis_angle": camera_state.per_view_axis_angle.detach().cpu().clone(),
                    "per_view_translation": camera_state.per_view_translation.detach().cpu().clone(),
                    "log_focal_scale": intr_state.log_focal_scale.detach().cpu().clone(),
                },
            }

    if best is None:
        raise RuntimeError("Per-view camera alignment produced no valid state.")

    camera_state.per_view_axis_angle.data.copy_(best["state"]["per_view_axis_angle"].to(device=device, dtype=dtype))
    camera_state.per_view_translation.data.copy_(best["state"]["per_view_translation"].to(device=device, dtype=dtype))
    intr_state.log_focal_scale.data.copy_(best["state"]["log_focal_scale"].to(device=device, dtype=dtype))
    report = {
        "source": "alignment_per_view_camera",
        "best_step": int(best["step"]),
        "best_loss": float(best["loss"]),
        "best_landmark_px": float(best["landmark_px"]),
        "best_landmark_px_inliers": float(best["landmark_px_inliers"]),
        "best_face_box": best["face_box"],
        "best_face_box_inliers": best["face_box_inliers"],
        "best_intrinsics": best["intrinsics_summary"],
        "optimization_inlier_views": inlier_views,
        "used_view_indices": [int(v) for v in batch.view_indices.detach().cpu().tolist()] if batch.view_indices is not None else list(range(int(batch.c2ws.shape[1]))),
        "used_frame_ids": list(batch.frame_ids),
        "history_tail": history[-20:],
    }
    (workspace.alignment_dir / "per_view_camera_alignment_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return {
        "camera_state": camera_state,
        "intr_state": intr_state,
        "best_landmark_px": float(best["landmark_px"]),
        "best_landmark_px_inliers": float(best["landmark_px_inliers"]),
        "best_face_box": best["face_box"],
        "best_face_box_inliers": best["face_box_inliers"],
        "intrinsics_summary": best["intrinsics_summary"],
        "used_view_count": int(batch.c2ws.shape[1]),
        "total_view_count": int(full_batch.c2ws.shape[1]),
    }


def _evaluate_per_view_alignment_metrics_cached(
    lam_model,
    batch,
    query_points,
    flame_params,
    camera_state: OptimizableCameraState,
    intr_state: OptimizableIntrinsicsState,
    target: torch.Tensor,
    valid: torch.Tensor,
    h: int,
    w: int,
) -> dict:
    c2ws = camera_state(batch.c2ws, batch.view_indices)
    intrs = intr_state(batch.intrs, batch.view_indices)
    with torch.no_grad():
        pred = _project_flame_landmarks(lam_model.renderer, query_points, flame_params, c2ws, intrs, h, w)
        _, landmark_px = _landmark_loss(pred, batch.landmarks_2d, (h, w), beta=0.002)
    return {
        "landmark_px": float(landmark_px.detach().cpu()),
        "face_box": _face_box_metrics_from_points(pred, target, valid),
        "intrinsics_summary": _summarize_intrinsics_state(intr_state),
    }


def _bake_per_view_camera_delta_into_alignment(
    workspace: MultiViewWorkspace,
    camera_state: OptimizableCameraState,
    intr_state: OptimizableIntrinsicsState,
) -> None:
    aligned_db = json.loads(workspace.aligned_transforms_path.read_text(encoding="utf-8"))
    per_view_axis = camera_state.per_view_axis_angle.detach().cpu().numpy().astype(np.float64)
    per_view_translation = camera_state.per_view_translation.detach().cpu().numpy().astype(np.float64)
    intrinsics_state = _intrinsics_state_dict_for_bake(intr_state, shared_focal=True)
    frames = sorted(aligned_db.get("frames", []), key=_frame_sort_key)
    for view_idx, frame in enumerate(frames):
        render_c2w = _json_c2w_to_render_c2w(np.asarray(frame["transform_matrix"], dtype=np.float64))
        if view_idx < per_view_axis.shape[0]:
            rot = _axis_angle_to_matrix_np(per_view_axis[view_idx])
            trans = per_view_translation[view_idx]
            render_c2w[:3, :3] = rot @ render_c2w[:3, :3]
            render_c2w[:3, 3] = rot @ render_c2w[:3, 3] + trans
        frame["transform_matrix_pre_per_view_alignment"] = frame["transform_matrix"]
        frame["transform_matrix"] = _render_c2w_to_json_c2w(render_c2w).tolist()
        frame["intrinsics_pre_per_view_alignment"] = {
            "fl_x": float(frame["fl_x"]),
            "fl_y": float(frame["fl_y"]),
            "cx": float(frame["cx"]),
            "cy": float(frame["cy"]),
        }
        _apply_intrinsics_delta_to_frame(frame, intrinsics_state, view_idx)
    aligned_db["per_view_camera_alignment"] = {
        "enabled": True,
        "num_views": int(per_view_axis.shape[0]),
        "intrinsics": _summarize_intrinsics_state(intr_state),
    }
    workspace.aligned_transforms_path.write_text(json.dumps(aligned_db, indent=2), encoding="utf-8")


def _shared_focal_consistency_penalty(intr_state: OptimizableIntrinsicsState) -> torch.Tensor:
    focal = intr_state.log_focal_scale
    if focal.numel() == 0 or focal.shape[0] <= 1:
        return focal.sum() * 0.0
    mean = focal.mean(dim=0, keepdim=True)
    return F.mse_loss(focal, mean.expand_as(focal))


def _intrinsics_state_dict_for_bake(intr_state: OptimizableIntrinsicsState, shared_focal: bool = True) -> dict:
    log_focal = intr_state.log_focal_scale.detach().cpu().clone()
    principal = intr_state.principal_delta.detach().cpu().clone()
    if shared_focal and log_focal.ndim == 2 and log_focal.shape[0] > 0:
        mean = log_focal.mean(dim=0, keepdim=True)
        log_focal = mean.expand_as(log_focal).clone()
        principal.zero_()
    return {
        "log_focal_scale": log_focal,
        "principal_delta": principal,
    }


def _summarize_intrinsics_state(intr_state: OptimizableIntrinsicsState) -> dict:
    focal = intr_state.log_focal_scale.detach().cpu()
    principal = intr_state.principal_delta.detach().cpu()
    if focal.numel() == 0:
        return {
            "max_focal_percent": 0.0,
            "shared_focal_percent": 0.0,
            "focal_scale_mean": [1.0, 1.0],
            "focal_scale_std": [0.0, 0.0],
            "principal_delta_max_px": 0.0,
        }
    max_log_focal = float(intr_state.max_log_focal)
    focal_scale = torch.exp(max_log_focal * torch.tanh(focal))
    focal_mean = focal_scale.mean(dim=0)
    focal_std = focal_scale.std(dim=0, unbiased=False) if focal_scale.shape[0] > 1 else torch.zeros_like(focal_mean)
    shared_log = focal.mean(dim=0, keepdim=True)
    shared_scale = torch.exp(max_log_focal * torch.tanh(shared_log)).reshape(-1)
    max_focal_percent = float(((focal_scale - 1.0).abs().max() * 100.0).item())
    shared_focal_percent = float(((shared_scale - 1.0).abs().max() * 100.0).item())
    principal_max = 0.0
    if principal.numel() > 0:
        principal_max = float((intr_state.max_principal_delta * torch.tanh(principal).abs().max()).item())
    return {
        "max_focal_percent": max_focal_percent,
        "shared_focal_percent": shared_focal_percent,
        "focal_scale_mean": focal_mean.detach().cpu().numpy().astype(float).tolist(),
        "focal_scale_std": focal_std.detach().cpu().numpy().astype(float).tolist(),
        "principal_delta_max_px": principal_max,
    }


def _load_torch_state(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    state = torch.load(path, map_location="cpu")
    return {key: value.detach().cpu() if torch.is_tensor(value) else value for key, value in state.items()}


def _apply_refine_deltas_to_batch(
    batch,
    camera_delta: Optional[dict] = None,
    intrinsics_delta: Optional[dict] = None,
    pose_delta: Optional[dict] = None,
):
    c2ws = batch.c2ws.clone()
    intrs = batch.intrs.clone()
    flame_params = {
        key: value.clone() if torch.is_tensor(value) else value
        for key, value in batch.flame_params.items()
    }
    device = c2ws.device
    dtype = c2ws.dtype
    num_views = int(c2ws.shape[1])

    for view_idx in range(num_views):
        c2w_np = c2ws[0, view_idx].detach().cpu().numpy().astype(np.float64)
        c2w_np = _apply_camera_delta_to_c2w(c2w_np, camera_delta, view_idx)
        c2ws[0, view_idx] = torch.from_numpy(c2w_np).to(device=device, dtype=dtype)

        frame = {
            "fl_x": float(intrs[0, view_idx, 0, 0].item()),
            "fl_y": float(intrs[0, view_idx, 1, 1].item()),
            "cx": float(intrs[0, view_idx, 0, 2].item()),
            "cy": float(intrs[0, view_idx, 1, 2].item()),
        }
        _apply_intrinsics_delta_to_frame(frame, intrinsics_delta, view_idx)
        intrs[0, view_idx, 0, 0] = float(frame["fl_x"])
        intrs[0, view_idx, 1, 1] = float(frame["fl_y"])
        intrs[0, view_idx, 0, 2] = float(frame["cx"])
        intrs[0, view_idx, 1, 2] = float(frame["cy"])

    if pose_delta:
        expr = _state_tensor(pose_delta, "expr_delta", device=device, dtype=dtype)
        jaw = _state_tensor(pose_delta, "jaw_delta", device=device, dtype=dtype)
        eyes = _state_tensor(pose_delta, "eyes_delta", device=device, dtype=dtype)
        if expr is not None and "expr" in flame_params:
            flame_params["expr"] = flame_params["expr"] + expr.clamp(-0.1, 0.1)
        if jaw is not None and "jaw_pose" in flame_params:
            flame_params["jaw_pose"] = flame_params["jaw_pose"] + jaw.clamp(-0.08, 0.08)
        if eyes is not None and "eyes_pose" in flame_params:
            flame_params["eyes_pose"] = flame_params["eyes_pose"] + eyes.clamp(-0.08, 0.08)

    return type(batch)(
        images=batch.images,
        masks=batch.masks,
        c2ws=c2ws,
        intrs=intrs,
        bg_colors=batch.bg_colors,
        flame_params=flame_params,
        frame_ids=list(batch.frame_ids),
        landmarks_2d=batch.landmarks_2d,
        view_indices=batch.view_indices,
    )


def _build_pose_sweep_specs(flame_params: TensorDict) -> list[dict]:
    expr_dim = int(flame_params["expr"].shape[-1]) if "expr" in flame_params else 0

    def expr_edit(index: int, value: float) -> dict:
        return {"type": "expr", "index": int(index), "value": float(value)}

    top_expr_dims = []
    if expr_dim > 0 and "expr" in flame_params:
        expr = flame_params["expr"][0].detach().cpu().numpy()
        score = expr.std(axis=0) + 0.25 * np.abs(expr).mean(axis=0)
        top_expr_dims = [int(idx) for idx in np.argsort(-score)[: min(2, expr_dim)].tolist()]

    specs = [
        {"name": "neutral", "description": "base refined pose/expression", "edits": []},
    ]
    for expr_idx in top_expr_dims:
        specs.append(
            {
                "name": f"expr_{expr_idx:02d}_plus",
                "description": f"expression channel {expr_idx} positive probe",
                "edits": [expr_edit(expr_idx, 0.12)],
            }
        )
    specs.extend([
        {"name": "mouth_open", "description": "jaw opening probe", "edits": [{"type": "jaw", "index": 0, "value": 0.05}]},
        {"name": "eyes_rotate", "description": "eyes pose probe", "edits": [{"type": "eyes", "index": 0, "value": 0.06}, {"type": "eyes", "index": 3, "value": 0.06}]},
        {"name": "turn_left", "description": "head yaw left probe", "edits": [{"type": "rotation", "index": 1, "value": -0.08}]},
        {"name": "turn_right", "description": "head yaw right probe", "edits": [{"type": "rotation", "index": 1, "value": 0.08}]},
    ])
    return specs


def _apply_pose_sweep(flame_params: TensorDict, spec: dict, strength: float = 1.0) -> TensorDict:
    out = {
        key: value.clone() if torch.is_tensor(value) else value
        for key, value in flame_params.items()
    }
    strength = float(strength)
    for edit in spec.get("edits", []):
        kind = str(edit.get("type"))
        index = int(edit.get("index", 0))
        value = float(edit.get("value", 0.0)) * strength
        if kind == "expr" and "expr" in out and 0 <= index < out["expr"].shape[-1]:
            out["expr"][..., index] = out["expr"][..., index] + value
        elif kind == "jaw" and "jaw_pose" in out and 0 <= index < out["jaw_pose"].shape[-1]:
            out["jaw_pose"][..., index] = out["jaw_pose"][..., index] + value
        elif kind == "eyes" and "eyes_pose" in out and 0 <= index < out["eyes_pose"].shape[-1]:
            out["eyes_pose"][..., index] = out["eyes_pose"][..., index] + value
        elif kind == "rotation" and "rotation" in out and 0 <= index < out["rotation"].shape[-1]:
            out["rotation"][..., index] = out["rotation"][..., index] + value
    return out


def _pose_sweep_caption(spec: dict) -> dict:
    name = str(spec.get("name", "probe"))
    if name == "neutral":
        return {
            "title": "基准状态",
            "subtitle": "当前 refined 结果，不额外施加表情或动作扰动",
            "footer": "用来和后续变化做对比",
        }
    if name.startswith("expr_") and name.endswith("_plus"):
        expr_token = name[len("expr_") : -len("_plus")]
        try:
            expr_idx = int(expr_token)
        except ValueError:
            expr_idx = expr_token
        return {
            "title": f"表情通道 {expr_idx} 增强",
            "subtitle": "自动选出的高变化 expression 维度，语义未命名",
            "footer": "重点看局部肌肉联动是否自然",
        }
    if name == "mouth_open":
        return {
            "title": "张嘴",
            "subtitle": "对应 jaw pose 开合扰动",
            "footer": "重点看嘴唇、下巴和口周形变",
        }
    if name == "eyes_rotate":
        return {
            "title": "眼部转动",
            "subtitle": "对应 eyes pose 扰动",
            "footer": "重点看眼睑和眼周联动",
        }
    if name == "turn_left":
        return {
            "title": "头部左转",
            "subtitle": "对应 head rotation yaw 负向扰动",
            "footer": "重点看轮廓、鼻梁和遮挡变化",
        }
    if name == "turn_right":
        return {
            "title": "头部右转",
            "subtitle": "对应 head rotation yaw 正向扰动",
            "footer": "重点看轮廓、鼻梁和遮挡变化",
        }
    return {
        "title": name,
        "subtitle": str(spec.get("description", "")),
        "footer": "未提供额外说明",
    }


def _representative_view_indices(num_views: int, count: int = 3) -> list[int]:
    if num_views <= 0:
        return []
    if num_views <= count:
        return list(range(num_views))
    values = np.linspace(0, num_views - 1, count)
    return sorted({int(round(v)) for v in values})


def _make_pose_sweep_montage(
    comp_rgb: torch.Tensor,
    comp_mask: torch.Tensor,
    frame_ids: list[str],
    view_indices: list[int],
    title: str,
    subtitle: str = "",
    footer: str = "",
) -> np.ndarray:
    panels = []
    for view_idx in view_indices:
        rgb = comp_rgb[view_idx].detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
        mask = comp_mask[view_idx].detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
        image = ((rgb * mask + (1.0 - mask) * 1.0).clip(0, 1) * 255.0).astype(np.uint8)
        panels.append((frame_ids[view_idx], Image.fromarray(image)))

    panel_w, panel_h = 340, 260
    has_subtitle = bool(subtitle)
    header_h = 82 if has_subtitle else 58
    footer_h = 24
    canvas = Image.new("RGB", (panel_w * len(panels), panel_h + header_h + footer_h), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = _load_annotation_font(26)
    body_font = _load_annotation_font(18)
    small_font = _load_annotation_font(16)
    draw.text((12, 8), title, fill=(20, 24, 32), font=title_font)
    if subtitle:
        draw.text((12, 38), subtitle, fill=(55, 63, 78), font=body_font)
    if footer:
        footer_y = 38 if not subtitle else 60
        draw.text((12, footer_y), footer, fill=(86, 95, 110), font=small_font)
    for idx, (label, image) in enumerate(panels):
        thumb = image.copy()
        thumb.thumbnail((panel_w - 8, panel_h - 26))
        x = idx * panel_w + (panel_w - thumb.width) // 2
        y = header_h + (panel_h - 24 - thumb.height) // 2
        canvas.paste(thumb, (x, y))
        draw.text((idx * panel_w + 8, header_h + panel_h + 2), label, fill=(60, 67, 80), font=small_font)
    canvas = _pad_image_to_multiple(canvas, multiple=16, fill="white")
    return np.asarray(canvas, dtype=np.uint8)


def _save_pose_sweep_contact(montage_dir: Path, out_path: Path, cols: int = 2) -> Optional[Path]:
    paths = sorted(montage_dir.glob("*.png"))
    if not paths:
        return None
    images = [Image.open(path).convert("RGB") for path in paths]
    panel_w = max(image.width for image in images)
    panel_h = max(image.height for image in images)
    rows = int(np.ceil(len(images) / max(cols, 1)))
    sheet = Image.new("RGB", (cols * panel_w, rows * panel_h), "white")
    for idx, image in enumerate(images):
        x = (idx % cols) * panel_w
        y = (idx // cols) * panel_h
        sheet.paste(image, (x, y))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, quality=90)
    return out_path


def _pad_image_to_multiple(image: Image.Image, multiple: int = 16, fill: str | tuple[int, int, int] = "white") -> Image.Image:
    if multiple <= 1:
        return image
    width, height = image.size
    padded_w = int(np.ceil(width / multiple) * multiple)
    padded_h = int(np.ceil(height / multiple) * multiple)
    if padded_w == width and padded_h == height:
        return image
    canvas = Image.new(image.mode, (padded_w, padded_h), fill)
    canvas.paste(image, (0, 0))
    return canvas


def _ease_values(num_frames: int, start: float, end: float) -> np.ndarray:
    num_frames = max(int(num_frames), 1)
    if num_frames == 1:
        return np.asarray([float(end)], dtype=np.float32)
    t = np.linspace(0.0, 1.0, num_frames, dtype=np.float32)
    eased = 0.5 - 0.5 * np.cos(np.pi * t)
    return float(start) + (float(end) - float(start)) * eased


@lru_cache(maxsize=8)
def _load_annotation_font(size: int):
    candidates = [
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/msyhbd.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("C:/Windows/Fonts/simsun.ttc"),
        Path("C:/Windows/Fonts/arial.ttf"),
    ]
    for path in candidates:
        if path.exists():
            try:
                return ImageFont.truetype(str(path), size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def _write_pose_sweep_notes(
    out_path: Path,
    report_rows: list[dict],
    representative_view_indices: list[int],
    num_views: int,
    hold_frames: int,
    fps: int,
) -> Path:
    seconds = float(hold_frames) / float(max(fps, 1))
    lines = [
        "# Pose Sweep Notes",
        "",
        f"- representative views: {representative_view_indices} / total {num_views}",
        f"- segment hold: {hold_frames} frames (~{seconds:.2f}s) at {fps} fps",
        "",
    ]
    for idx, row in enumerate(report_rows, start=1):
        caption = row.get("caption", {})
        lines.append(f"## {idx}. {caption.get('title', row.get('name', 'probe'))}")
        lines.append("")
        lines.append(f"- raw name: `{row.get('name', '')}`")
        lines.append(f"- note: {caption.get('subtitle', row.get('description', ''))}")
        lines.append(f"- observe: {caption.get('footer', '')}")
        lines.append(f"- edits: `{json.dumps(row.get('edits', []), ensure_ascii=False)}`")
        lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def _write_pose_transition_notes(
    out_path: Path,
    segment_rows: list[dict],
    representative_view_indices: list[int],
    num_views: int,
    transition_frames: int,
    hold_frames: int,
    fps: int,
) -> Path:
    transition_sec = float(transition_frames) / float(max(fps, 1))
    hold_sec = float(hold_frames) / float(max(fps, 1))
    lines = [
        "# Pose Transition Demo Notes",
        "",
        f"- representative views: {representative_view_indices} / total {num_views}",
        f"- transition: {transition_frames} frames (~{transition_sec:.2f}s)",
        f"- hold at peak: {hold_frames} frames (~{hold_sec:.2f}s)",
        f"- fps: {fps}",
        "",
    ]
    for idx, row in enumerate(segment_rows, start=1):
        caption = row.get("caption", {})
        lines.append(f"## {idx}. {caption.get('title', row.get('name', 'segment'))}")
        lines.append("")
        lines.append(f"- raw name: `{row.get('name', '')}`")
        lines.append(f"- note: {caption.get('subtitle', row.get('description', ''))}")
        lines.append(f"- observe: {caption.get('footer', '')}")
        lines.append(f"- duration_frames: {int(row.get('duration_frames', 0))}")
        lines.append(f"- edits: `{json.dumps(row.get('edits', []), ensure_ascii=False)}`")
        lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def _global_camera_delta(camera_delta: Optional[dict]) -> tuple[float, np.ndarray, np.ndarray]:
    if not camera_delta:
        return 1.0, np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64)
    log_scale = float(_state_array(camera_delta, "delta_log_scale", [0.0]).reshape(-1)[0])
    axis_angle = _state_array(camera_delta, "global_axis_angle", [0.0, 0.0, 0.0]).reshape(3)
    translation = _state_array(camera_delta, "global_translation", [0.0, 0.0, 0.0]).reshape(3)
    return float(np.exp(log_scale)), _axis_angle_to_matrix_np(axis_angle), translation


def _compose_sim3_with_global_delta(
    sim3: Sim3Alignment,
    delta_scale: float,
    delta_rot_col: np.ndarray,
    delta_translation: np.ndarray,
) -> Sim3Alignment:
    delta_rot_row = delta_rot_col.T
    initial_rot = np.asarray(sim3.rotation, dtype=np.float64)
    initial_translation = np.asarray(sim3.translation, dtype=np.float64)
    rotation = initial_rot @ delta_rot_row
    translation = delta_scale * (initial_translation @ delta_rot_row) + delta_translation
    return Sim3Alignment(
        scale=float(sim3.scale) * float(delta_scale),
        rotation=rotation.astype(np.float32).tolist(),
        translation=translation.astype(np.float32).tolist(),
        rmse=sim3.rmse,
        source=sim3.source,
        inlier_count=sim3.inlier_count,
        total_count=sim3.total_count,
        inlier_rmse=sim3.inlier_rmse,
        inlier_names=sim3.inlier_names,
    )


def _apply_camera_delta_to_c2w(c2w: np.ndarray, camera_delta: Optional[dict], view_idx: int) -> np.ndarray:
    if not camera_delta:
        return c2w.copy()
    global_scale, global_rot, global_translation = _global_camera_delta(camera_delta)
    out = c2w.copy()
    out[:3, :3] = global_rot @ out[:3, :3]
    out[:3, 3] = global_scale * (global_rot @ out[:3, 3]) + global_translation
    per_view_axis = _state_array(camera_delta, "per_view_axis_angle", None)
    per_view_translation = _state_array(camera_delta, "per_view_translation", None)
    if per_view_axis is not None and view_idx < per_view_axis.shape[0]:
        rot = _axis_angle_to_matrix_np(per_view_axis[view_idx])
        trans = per_view_translation[view_idx] if per_view_translation is not None and view_idx < per_view_translation.shape[0] else np.zeros(3)
        out[:3, :3] = rot @ out[:3, :3]
        out[:3, 3] = rot @ out[:3, 3] + trans
    return out


def _apply_intrinsics_delta_to_frame(frame: dict, intrinsics_delta: Optional[dict], view_idx: int) -> None:
    if not intrinsics_delta:
        return
    focal = _state_array(intrinsics_delta, "log_focal_scale", None)
    principal = _state_array(intrinsics_delta, "principal_delta", None)
    if focal is not None and view_idx < focal.shape[0]:
        max_log_focal = np.log(1.03)
        frame["fl_x"] = float(frame["fl_x"]) * float(np.exp(max_log_focal * np.tanh(focal[view_idx, 0])))
        frame["fl_y"] = float(frame["fl_y"]) * float(np.exp(max_log_focal * np.tanh(focal[view_idx, 1])))
    if principal is not None and view_idx < principal.shape[0]:
        max_principal_delta = 32.0
        frame["cx"] = float(frame["cx"]) + float(max_principal_delta * np.tanh(principal[view_idx, 0]))
        frame["cy"] = float(frame["cy"]) + float(max_principal_delta * np.tanh(principal[view_idx, 1]))


def _state_array(state: dict, key: str, default) -> Optional[np.ndarray]:
    value = state.get(key)
    if value is None:
        return None if default is None else np.asarray(default, dtype=np.float64)
    if torch.is_tensor(value):
        return value.detach().cpu().numpy().astype(np.float64)
    return np.asarray(value, dtype=np.float64)


def _state_tensor(state: Optional[dict], key: str, device, dtype) -> Optional[torch.Tensor]:
    if not state or key not in state:
        return None
    value = state[key]
    if torch.is_tensor(value):
        return value.to(device=device, dtype=dtype)
    return torch.as_tensor(value, device=device, dtype=dtype)


def _axis_angle_to_matrix_np(axis_angle: np.ndarray) -> np.ndarray:
    vector = np.asarray(axis_angle, dtype=np.float64).reshape(3)
    angle = float(np.linalg.norm(vector))
    if not np.isfinite(angle) or angle < 1e-12:
        return np.eye(3, dtype=np.float64)
    axis = vector / angle
    x, y, z = axis
    k = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)
    return np.eye(3, dtype=np.float64) + np.sin(angle) * k + (1.0 - np.cos(angle)) * (k @ k)


def _frame_sort_key(frame: dict) -> str:
    return str(frame.get("flame_param_path") or frame.get("file_path") or frame.get("image_name", ""))


def _load_layer1_alignment_stats(workspace: MultiViewWorkspace) -> dict:
    metadata = {}
    if workspace.layer1_metadata_path.exists():
        metadata = json.loads(workspace.layer1_metadata_path.read_text(encoding="utf-8"))

    center = np.asarray(metadata.get("canonical_center", []), dtype=np.float64)
    radius_p95 = float(metadata.get("canonical_radius_p95", 0.0) or 0.0)
    if center.shape != (3,) or radius_p95 <= 1e-8:
        gs = GaussianModel(ply_path=str(workspace.init_ply_path), sh2rgb=False)
        xyz = gs.xyz.detach().cpu().float().numpy().astype(np.float64)
        if xyz.size == 0:
            raise ValueError(f"Layer 1 canonical Gaussian has no points: {workspace.init_ply_path}")
        center = xyz.mean(axis=0)
        radius_p95 = float(np.percentile(np.linalg.norm(xyz - center, axis=1), 95))

    reference_frame = metadata.get("reference_camera")
    if reference_frame is None and workspace.layer1_reference_transforms_path.exists():
        ref_db = json.loads(workspace.layer1_reference_transforms_path.read_text(encoding="utf-8"))
        frames = ref_db.get("frames", [])
        reference_frame = frames[0] if frames else None
    reference_c2w = None
    if reference_frame and "transform_matrix" in reference_frame:
        mat = np.asarray(reference_frame["transform_matrix"], dtype=np.float64)
        if mat.shape == (4, 4):
            reference_c2w = mat

    return {"center": center, "radius_p95": radius_p95, "reference_c2w": reference_c2w}


def _rotation_from_forward_up(
    source_forward: np.ndarray,
    source_up: np.ndarray,
    target_forward: np.ndarray,
    target_up: np.ndarray,
) -> np.ndarray:
    source_basis = _basis_from_forward_up(source_forward, source_up)
    target_basis = _basis_from_forward_up(target_forward, target_up)
    R_col = target_basis @ source_basis.T
    if np.linalg.det(R_col) < 0:
        target_basis[:, 0] *= -1.0
        R_col = target_basis @ source_basis.T
    if not np.isfinite(R_col).all():
        return np.eye(3, dtype=np.float64)
    u, _, vt = np.linalg.svd(R_col)
    R_col = u @ vt
    if np.linalg.det(R_col) < 0:
        u[:, -1] *= -1.0
        R_col = u @ vt
    return R_col.astype(np.float64)


def _basis_from_forward_up(forward: np.ndarray, up: np.ndarray) -> np.ndarray:
    f = _normalize(forward, np.array([0.0, 0.0, -1.0], dtype=np.float64))
    u_hint = _normalize(up, np.array([0.0, 1.0, 0.0], dtype=np.float64))
    right = np.cross(u_hint, f)
    if np.linalg.norm(right) < 1e-8:
        u_hint = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        right = np.cross(u_hint, f)
    right = _normalize(right, np.array([1.0, 0.0, 0.0], dtype=np.float64))
    true_up = _normalize(np.cross(f, right), np.array([0.0, 1.0, 0.0], dtype=np.float64))
    return np.stack([right, true_up, f], axis=1)


def _normalize(value: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    norm = float(np.linalg.norm(value))
    if not np.isfinite(norm) or norm < 1e-8:
        return np.asarray(fallback, dtype=np.float64)
    return value / norm


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


def _find_mask_by_stem(mask_dir: Path, stem: str) -> Optional[Path]:
    for dirname in ["fg_masks", "masks", ""]:
        base = mask_dir / dirname if dirname else mask_dir
        candidate = find_stem_file(base, stem, [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"])
        if candidate is not None:
            return candidate
    return None


def _process_external_mask(
    mask_path: Path,
    image_path: Path,
    output_path: Path,
    keep_largest_component: bool,
    close_radius: int,
    feather_radius: int,
) -> None:
    with Image.open(image_path) as image:
        width, height = image.size
    with Image.open(mask_path) as source_mask:
        mask = source_mask.convert("L")
        if mask.size != (width, height):
            mask = mask.resize((width, height), Image.Resampling.NEAREST)
        arr = np.asarray(mask, dtype=np.uint8)
    original_binary = arr > 32
    binary = arr > 32
    if keep_largest_component:
        binary = _largest_component(binary)
    binary = _binary_close(binary, max(0, int(close_radius)))
    has_soft_alpha = bool(np.any((arr > 0) & (arr < 255)))
    if has_soft_alpha:
        out = arr.copy()
        out[~binary] = 0
        out[binary & ~original_binary] = 255
    else:
        out = (binary.astype(np.uint8) * 255)
    if feather_radius > 0:
        k = max(3, int(feather_radius) * 2 + 1)
        out = cv2.GaussianBlur(out, (k, k), sigmaX=float(feather_radius))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(out, mode="L").save(output_path)


def _normalize_external_mask(mask_path: Path, image_path: Path, output_path: Path) -> None:
    with Image.open(image_path) as image:
        width, height = image.size
    with Image.open(mask_path) as source_mask:
        mask = source_mask.convert("L")
        if mask.size != (width, height):
            mask = mask.resize((width, height), Image.Resampling.NEAREST)
        arr = np.asarray(mask, dtype=np.uint8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr, mode="L").save(output_path)


def _largest_component(mask: np.ndarray) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if count <= 2:
        return mask
    areas = stats[:, cv2.CC_STAT_AREA]
    areas[0] = 0
    return labels == int(np.argmax(areas))


def _binary_close(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask
    k = max(3, radius | 1)
    kernel = np.ones((k, k), dtype=np.uint8)
    closed = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel) > 0
    flood = (~closed).astype(np.uint8)
    h, w = flood.shape[:2]
    canvas = np.zeros((h + 2, w + 2), dtype=np.uint8)
    for x in range(w):
        if flood[0, x] == 1:
            cv2.floodFill(flood, canvas, (x, 0), 2)
        if flood[h - 1, x] == 1:
            cv2.floodFill(flood, canvas, (x, h - 1), 2)
    for y in range(h):
        if flood[y, 0] == 1:
            cv2.floodFill(flood, canvas, (0, y), 2)
        if flood[y, w - 1] == 1:
            cv2.floodFill(flood, canvas, (w - 1, y), 2)
    holes = flood == 1
    return closed | holes


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
