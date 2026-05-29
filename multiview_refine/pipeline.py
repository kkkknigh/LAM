import json
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import cv2
from PIL import Image

from lam.models.rendering.gaussian_model import GaussianModel

from .alignment import Sim3Alignment, apply_sim3_to_c2w
from .colmap import import_colmap_sparse, run_colmap_pipeline
from .data import load_multiview_bundle, write_lam_transforms_from_colmap
from .optimization import MultiViewGaussianRefiner, RefinementConfig
from .optimization import _project_flame_landmarks
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
        mask_note = ""
        if report["num_masks"] and not report["missing_masks"]:
            self._normalize_workspace_masks()
            mask_note = " Uploaded masks were normalized without shape postprocessing."
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
                    stem = path.stem
                    if stem not in image_by_stem or stem in seen:
                        continue
                    with zipf.open(member) as fp:
                        with Image.open(fp) as mask:
                            mask = mask.convert("L")
                            image_path = image_by_stem[stem]
                            with Image.open(image_path) as image:
                                width, height = image.size
                            if mask.size != (width, height):
                                mask = mask.resize((width, height), Image.Resampling.NEAREST)
                            mask.save(self.workspace.masks_dir / f"{stem}.png")
                    seen.add(stem)
                    restored += 1
            if restored == len(images):
                break
        missing = sorted(set(image_by_stem) - seen)
        if missing:
            raise FileNotFoundError(f"Could not restore uploaded masks for image stems: {missing[:10]}")
        debug_dir = self._save_workspace_preview("01_uploaded_masks_restored")
        return StepResult(f"Restored {restored} uploaded masks from input ZIP", str(debug_dir))

    def _normalize_workspace_masks(self) -> int:
        images = sorted([p for p in self.workspace.images_dir.glob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"}])
        processed = 0
        for image_path in images:
            mask_path = _find_mask_by_stem(self.workspace.masks_dir, image_path.stem)
            if mask_path is None:
                continue
            _normalize_external_mask(mask_path, image_path, self.workspace.masks_dir / f"{image_path.stem}.png")
            processed += 1
        return processed

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
        model_dir = run_colmap_pipeline(self.workspace.images_dir, self.workspace.colmap_dir, colmap_path=colmap_path)
        raw_json = import_colmap_sparse(model_dir, self.workspace.colmap_dir / "transforms_colmap_sparse.json", colmap_path=colmap_path)
        transforms = write_lam_transforms_from_colmap(self.workspace.root, raw_json, out_name="colmap/transforms_colmap_raw.json")
        plot = self._save_colmap_visualization(transforms)
        return StepResult("COLMAP complete", str(plot))

    def import_colmap(self, sparse_dir: str | Path, colmap_path: str = "colmap") -> StepResult:
        raw_json = import_colmap_sparse(sparse_dir, self.workspace.colmap_dir / "transforms_colmap_sparse.json", colmap_path=colmap_path)
        transforms = write_lam_transforms_from_colmap(self.workspace.root, raw_json, out_name="colmap/transforms_colmap_raw.json")
        plot = self._save_colmap_visualization(transforms)
        return StepResult("COLMAP imported", str(plot))

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
        source_scene_center, look_at_rmse = _estimate_camera_ray_intersection(colmap_centers, source_forwards)
        colmap_center = np.median(colmap_centers, axis=0).astype(np.float64)
        colmap_radius = float(np.median(np.linalg.norm(colmap_centers - source_scene_center, axis=1)))
        if colmap_radius < 1e-6:
            raise ValueError("COLMAP camera centers are degenerate; cannot estimate scale.")

        layer1_stats = _load_layer1_alignment_stats(self.workspace)
        target_center = layer1_stats["center"]
        reference_c2w = layer1_stats.get("reference_c2w")
        if reference_c2w is not None:
            reference_c2w = _json_c2w_to_render_c2w(reference_c2w)
            reference_center = reference_c2w[:3, 3]
            target_distance = float(np.linalg.norm(reference_center - target_center))
        else:
            reference_center = None
            target_distance = max(float(layer1_stats["radius_p95"]) * 5.0, 0.8)
        if target_distance < 1e-6:
            target_distance = max(float(layer1_stats["radius_p95"]) * 5.0, 0.8)
        scale = float(target_distance / colmap_radius)

        if reference_c2w is not None:
            source = "layer1_reference_initialization"
        else:
            reference_center = target_center + np.array([0.0, 0.0, target_distance], dtype=np.float64)
            source = "layer1_first_view_front_initialization"
        source_anchor_center = source_scene_center
        target_anchor_center = target_center

        intrinsics = [_intrinsic_from_frame(frame) for frame in frames]
        alignment_candidate = _choose_initial_sim3_candidate(
            colmap_mats,
            intrinsics,
            source_scene_center,
            source_anchor_center,
            target_anchor_center,
            target_center,
            target_distance,
            scale,
            reference_c2w,
        )
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
                    "colmap_center": colmap_center.tolist(),
                    "colmap_look_at_center": source_scene_center.tolist(),
                    "colmap_look_at_rmse": look_at_rmse,
                    "source_anchor_center": source_anchor_center.tolist(),
                    "colmap_radius_median": colmap_radius,
                    "layer1_center": target_center.tolist(),
                    "layer1_radius_p95": float(layer1_stats["radius_p95"]),
                    "target_camera_distance": target_distance,
                    "reference_camera_available": reference_c2w is not None,
                    "reference_center": reference_center.tolist() if reference_center is not None else None,
                    "target_anchor_center": target_anchor_center.tolist(),
                    "first_view_target_center": reference_center.tolist() if reference_center is not None else None,
                    "scale": scale,
                    "translation": sim3.translation,
                    "rotation_policy": alignment_candidate["policy"],
                    "projection_score": alignment_candidate["score"],
                    "camera_convention": "json stores NeRF-style c2w; renderer/data converts by flipping c2w Y/Z axes",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        _write_projection_diagnostics(self.workspace, lam_model=lam_model)
        plot = self._save_sim3_visualization(None)
        return StepResult(
            f"Layer1 Sim3 initialized. scale={scale:.5f}, COLMAP radius={colmap_radius:.5f}, target distance={target_distance:.5f}",
            str(plot),
        )

    def calibrate_global_sim3_blackbox(self, lam_model, init_ply: Optional[str | Path] = None) -> StepResult:
        init_ply = Path(init_ply) if init_ply else self.workspace.init_ply_path
        if not init_ply.exists():
            raise FileNotFoundError(f"Missing init PLY: {init_ply}")
        if not self.workspace.aligned_transforms_path.exists() or not self.workspace.sim3_path.exists():
            raise FileNotFoundError("Missing initial Sim3 alignment. Run Initialize Sim3 first.")

        restored = _ensure_workspace_landmarks(self.workspace)
        batch = load_multiview_bundle(self.workspace.root, require_undistorted=False).to("cuda", torch.float32)
        query_points, flame_params = lam_model.renderer.get_query_points(batch.flame_params, device=batch.images.device)
        candidate = _landmark_calibrate_sim3(lam_model.renderer, query_points, flame_params, batch, self.workspace)
        _bake_render_delta_into_alignment(self.workspace, candidate["scale"], candidate["rotation"], candidate["translation"], "landmark_calibrated")
        _write_projection_diagnostics(self.workspace, lam_model=lam_model)
        plot = self._save_sim3_visualization(None)
        report = self.workspace.alignment_dir / "landmark_calibration_report.json"
        suffix = f"; restored {restored} landmark files" if restored else ""
        return StepResult(f"Landmark Sim3 calibrated. loss={candidate['loss']:.5f}{suffix}", str(plot if plot else report))

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

    def refine(self, lam_model, init_ply: Optional[str | Path] = None, config: Optional[RefinementConfig] = None, resume: Optional[str | Path] = None) -> StepResult:
        init_ply = Path(init_ply) if init_ply else self.workspace.init_ply_path
        if not init_ply.exists():
            raise FileNotFoundError(f"Missing init PLY: {init_ply}")
        _assert_projection_not_empty(self.workspace)
        config = config or RefinementConfig(output_dir=str(self.workspace.refine_dir))
        config.output_dir = str(self.workspace.refine_dir)
        batch = load_multiview_bundle(self.workspace.root, require_undistorted=config.require_undistorted)
        gs = _load_renderable_gaussian(init_ply)
        gs.to_cuda()
        refiner = MultiViewGaussianRefiner(lam_model, config)
        refiner.run(gs, batch, resume=resume)
        loss_plot = save_loss_plot(self.workspace.loss_history_path, self.workspace.debug_dir / "05_refine" / "loss_history.png")
        return StepResult("Refinement complete", str(loss_plot or self.workspace.refined_gaussian_path))

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
        for src in [self.workspace.sim3_path, self.workspace.aligned_transforms_path]:
            shutil.copy2(src, self.workspace.exports_dir / f"{src.stem}.initial{src.suffix}")
        with zipfile.ZipFile(package_path, "w", compression=zipfile.ZIP_DEFLATED) as zipf:
            for path in sorted(self.workspace.exports_dir.rglob("*")):
                if path.is_file() and path != package_path:
                    rel = path.relative_to(self.workspace.exports_dir)
                    if rel.name == "final_review.zip" or (rel.parts and rel.parts[0] == "final_review"):
                        continue
                    zipf.write(path, arcname=str(rel).replace("\\", "/"))
        return StepResult("Export ready", str(self.workspace.exports_dir))

    def export_final_review(
        self,
        lam_model,
        fps: int = 8,
        max_side: int = 1024,
        chunk_size: int = 4,
    ) -> StepResult:
        required = [self.workspace.refined_gaussian_path, self.workspace.sim3_path, self.workspace.aligned_transforms_path]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Missing final review files: {missing}")

        self.workspace.exports_dir.mkdir(parents=True, exist_ok=True)
        _write_baked_export_alignment(self.workspace)
        review_dir = self.workspace.exports_dir / "final_review"
        overlays_dir = review_dir / "overlays"
        if review_dir.exists():
            shutil.rmtree(review_dir)
        overlays_dir.mkdir(parents=True, exist_ok=True)

        transforms_path = self.workspace.exports_dir / self.workspace.aligned_transforms_path.name
        batch = load_multiview_bundle(self.workspace.root, transforms_name=transforms_path, require_undistorted=False).to("cuda", torch.float32)
        batch = _scale_multiview_batch_to_max_side(batch, max_side=max_side)
        gs = _load_renderable_gaussian(self.workspace.refined_gaussian_path)
        gs.to_cuda()
        query_points, flame_params = lam_model.renderer.get_query_points(batch.flame_params, device=batch.images.device)
        flame_params = _apply_pose_delta_to_flame_params(flame_params, self.workspace.refine_dir / "pose_delta.pt")

        video_frames = []
        num_views = batch.c2ws.shape[1]
        with torch.no_grad():
            for start in range(0, num_views, max(1, int(chunk_size))):
                end = min(num_views, start + max(1, int(chunk_size)))
                indices = torch.arange(start, end, device=batch.images.device)
                sub_batch = batch.index(indices)
                sub_flame = {key: value if key == "betas" else value[:, sub_batch.view_indices] for key, value in flame_params.items()}
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
                for local_idx, frame_id in enumerate(sub_batch.frame_ids):
                    target = _tensor_chw_to_uint8(sub_batch.images[0, local_idx])
                    render = _tensor_chw_to_uint8(out["comp_rgb"][0, local_idx])
                    mask = _tensor_chw_to_uint8(out["comp_mask"][0, local_idx])
                    overlay = (target.astype(np.float32) * 0.55 + render.astype(np.float32) * 0.45).clip(0, 255).astype(np.uint8)
                    review = _make_review_frame(target, render, mask, overlay)
                    name = _safe_filename(f"{start + local_idx:04d}_{frame_id}.png")
                    Image.fromarray(review).save(overlays_dir / name)
                    video_frames.append(review)

        video_path = review_dir / "final_review.mp4"
        if video_frames:
            _write_review_video(video_frames, video_path, fps=fps)
        package_path = self.workspace.exports_dir / "final_review.zip"
        if package_path.exists():
            package_path.unlink()
        with zipfile.ZipFile(package_path, "w", compression=zipfile.ZIP_DEFLATED) as zipf:
            for path in sorted(review_dir.rglob("*")):
                if path.is_file():
                    zipf.write(path, arcname=str(path.relative_to(review_dir.parent)).replace("\\", "/"))
        return StepResult(f"Final review ready. Frames: {len(video_frames)}", str(review_dir))

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
            if preserve_existing_masks and mask_dst.exists():
                _normalize_external_mask(mask_dst, image_path, mask_dst)
                preserved_masks += 1
                mask_source = "existing_upload"
            else:
                _restore_tracking_mask(mask_src, image_path, meta, mask_dst)
            _write_tracking_flame_param(flame_src, flame_dst, canonical_shape)
            if (export_dir / "landmark2d" / "landmarks.npz").exists():
                _restore_tracking_landmarks(export_dir / "landmark2d" / "landmarks.npz", image_path, meta, landmark_dst)
            records.append({
                "image_name": image_path.name,
                "file_path": f"data/images/{image_path.name}",
                "fg_mask_path": f"data/fg_masks/{stem}.png",
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


def _load_renderable_gaussian(path: str | Path) -> GaussianModel:
    gs = GaussianModel(ply_path=str(path), sh2rgb=False)
    # GaussianModel.save_ply stores opacity as logit and scale as log. The renderer
    # consumes activated opacity/scale tensors directly.
    gs.opacity = torch.sigmoid(gs.opacity)
    gs.scaling = torch.exp(gs.scaling)
    return gs


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


def _choose_initial_sim3_candidate(
    colmap_mats: list[np.ndarray],
    intrinsics: list[np.ndarray],
    source_scene_center: np.ndarray,
    source_anchor_center: np.ndarray,
    target_anchor_center: np.ndarray,
    target_center: np.ndarray,
    target_distance: float,
    scale: float,
    reference_c2w: Optional[np.ndarray],
) -> dict:
    source_front = np.asarray(colmap_mats[0], dtype=np.float64)
    if reference_c2w is not None:
        base_target_forward = np.asarray(target_center, dtype=np.float64) - np.asarray(reference_c2w[:3, 3], dtype=np.float64)
        base_target_up = np.asarray(reference_c2w[:3, 1], dtype=np.float64)
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
            translation = target_anchor_center - scale * (source_anchor_center @ R_row)
            sim3 = Sim3Alignment(
                scale=float(scale),
                rotation=R_row.astype(np.float32).tolist(),
                translation=translation.astype(np.float32).tolist(),
                rmse=-1.0,
            )
            score = _score_initial_sim3_projection(colmap_mats, intrinsics, sim3, target_center)
            candidate = {
                "rotation_row": R_row,
                "translation": translation,
                "policy": f"{source_name}->{target_name}",
                "score": score,
            }
            if best is None or _projection_score_tuple(score) > _projection_score_tuple(best["score"]):
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
    return (
        int(score.get("positive_depth", 0)),
        int(score.get("in_frame", 0)),
        -float(score.get("median_center_error_px", float("inf"))),
        float(score.get("min_depth", -float("inf"))),
    )


def _write_projection_diagnostics(workspace: MultiViewWorkspace, lam_model=None, max_views: int = 9) -> Path:
    layer1_stats = _load_layer1_alignment_stats(workspace)
    center = np.asarray(layer1_stats["center"], dtype=np.float64)
    radius = float(layer1_stats["radius_p95"])
    db = json.loads(workspace.aligned_transforms_path.read_text(encoding="utf-8"))
    rows = []
    for frame in db.get("frames", []):
        c2w = _json_c2w_to_render_c2w(np.asarray(frame["transform_matrix"], dtype=np.float64))
        intr = _intrinsic_from_frame(frame)
        uv, z = _project_point_np(center, c2w, intr)
        rows.append({
            "image_name": frame.get("image_name") or frame.get("file_path"),
            "center_z": float(z),
            "center_uv": uv.tolist(),
            "center_in_frame": bool(z > 1e-6 and 0 <= uv[0] < float(frame["w"]) and 0 <= uv[1] < float(frame["h"])),
            "approx_radius_px": float(max(intr[0, 0], intr[1, 1]) * radius / max(abs(z), 1e-6)),
        })

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


def _landmark_calibrate_sim3(renderer, query_points, flame_params, batch, workspace: MultiViewWorkspace) -> dict:
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

    log_scale = torch.nn.Parameter(torch.zeros(1, device=batch.images.device, dtype=batch.images.dtype))
    axis_angle = torch.nn.Parameter(torch.zeros(3, device=batch.images.device, dtype=batch.images.dtype))
    translation = torch.nn.Parameter(torch.zeros(3, device=batch.images.device, dtype=batch.images.dtype))
    optimizer = torch.optim.Adam([log_scale, axis_angle, translation], lr=5e-3)
    history = []
    best = None
    for step in range(240):
        optimizer.zero_grad(set_to_none=True)
        scale = torch.exp(log_scale).clamp(0.5, 1.5)
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
    }
    (workspace.alignment_dir / "landmark_calibration_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return {
        "loss": best["loss"],
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


def _load_torch_state(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    state = torch.load(path, map_location="cpu")
    return {key: value.detach().cpu() if torch.is_tensor(value) else value for key, value in state.items()}


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


def _scale_multiview_batch_to_max_side(batch, max_side: int):
    max_side = int(max_side)
    if max_side <= 0:
        return batch
    h, w = batch.images.shape[-2:]
    side = max(int(h), int(w))
    if side <= max_side:
        return batch
    scale = float(max_side) / float(side)
    size = (max(1, int(round(h * scale))), max(1, int(round(w * scale))))
    images = F.interpolate(batch.images.flatten(0, 1), size=size, mode="bilinear", align_corners=False).reshape(batch.images.shape[:2] + batch.images.shape[2:3] + size)
    masks = F.interpolate(batch.masks.flatten(0, 1), size=size, mode="bilinear", align_corners=False).reshape(batch.masks.shape[:2] + batch.masks.shape[2:3] + size)
    intrs = batch.intrs.clone()
    intrs[..., 0, 0] *= scale
    intrs[..., 1, 1] *= scale
    intrs[..., 0, 2] *= scale
    intrs[..., 1, 2] *= scale
    landmarks = batch.landmarks_2d
    if landmarks is not None:
        landmarks = landmarks.clone()
        finite = torch.isfinite(landmarks[..., :2]).all(dim=-1)
        if finite.any() and float(landmarks[..., :2][finite].max().detach().cpu()) > 2.0:
            landmarks[..., :2] *= scale
    return type(batch)(images, masks, batch.c2ws, intrs, batch.bg_colors, batch.flame_params, batch.frame_ids, landmarks, batch.view_indices)


def _apply_pose_delta_to_flame_params(flame_params: dict, pose_delta_path: Path) -> dict:
    if not pose_delta_path.exists():
        return flame_params
    state = _load_torch_state(pose_delta_path)
    if not state:
        return flame_params
    out = dict(flame_params)
    for flame_key, state_key, limit in [
        ("expr", "expr_delta", 0.1),
        ("jaw_pose", "jaw_delta", 0.08),
        ("eyes_pose", "eyes_delta", 0.08),
    ]:
        if flame_key not in out or state_key not in state:
            continue
        delta = state[state_key].to(device=out[flame_key].device, dtype=out[flame_key].dtype)
        if delta.shape[1] != out[flame_key].shape[1]:
            continue
        out[flame_key] = out[flame_key] + delta.clamp(-limit, limit)
    return out


def _tensor_chw_to_uint8(tensor: torch.Tensor) -> np.ndarray:
    arr = tensor.detach().cpu()
    if arr.ndim == 3 and arr.shape[0] in {1, 3}:
        arr = arr.permute(1, 2, 0)
    arr = arr.numpy()
    if arr.ndim == 2:
        arr = arr[..., None]
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    return (arr.clip(0, 1) * 255).astype(np.uint8)


def _make_review_frame(target: np.ndarray, render: np.ndarray, mask: np.ndarray, overlay: np.ndarray) -> np.ndarray:
    target_h = 384
    panels = [_resize_np_to_height(item, target_h) for item in [target, render, mask, overlay]]
    return np.concatenate(panels, axis=1)


def _resize_np_to_height(image: np.ndarray, target_h: int) -> np.ndarray:
    h, w = image.shape[:2]
    if h == target_h:
        return image
    target_w = max(1, int(round(w * target_h / max(h, 1))))
    return np.asarray(Image.fromarray(image).resize((target_w, target_h), Image.Resampling.BILINEAR), dtype=np.uint8)


def _write_review_video(frames: list[np.ndarray], output_path: Path, fps: int = 8) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not frames:
        return
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(max(1, int(fps))),
        (int(width), int(height)),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {output_path}")
    try:
        for frame in frames:
            if frame.shape[:2] != (height, width):
                frame = np.asarray(Image.fromarray(frame).resize((width, height), Image.Resampling.BILINEAR), dtype=np.uint8)
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def _safe_filename(name: str) -> str:
    invalid = '<>:"/\\|?*'
    return "".join("_" if ch in invalid else ch for ch in str(name))


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
        if not base.is_dir():
            continue
        for suffix in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]:
            candidate = base / f"{stem}{suffix}"
            if candidate.exists():
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
