import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from lam.models.rendering.gaussian_model import GaussianModel

from .alignment import align_colmap_to_flame, write_manual_sim3
from .colmap import import_colmap_sparse, run_colmap_pipeline
from .data import load_multiview_bundle, write_lam_transforms_from_colmap
from .optimization import MultiViewGaussianRefiner, RefinementConfig
from .visualization import (
    save_loss_plot,
    save_overlay_grid,
    save_transform_camera_plot,
    save_workspace_image_previews,
)
from .workspace import MultiViewWorkspace, create_workspace, import_local_inputs, unpack_multiview_zip, validate_workspace_inputs


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

    def unpack(self, zip_path: str | Path, init_ply: Optional[str | Path] = None) -> StepResult:
        unpack_multiview_zip(zip_path, self.workspace)
        if init_ply:
            shutil.copy2(init_ply, self.workspace.root / "init.ply")
        report = validate_workspace_inputs(self.workspace, require_masks=False)
        debug_dir = self._save_input_visualization()
        return StepResult(f"Workspace ready. Images: {report['num_images']}, masks: {report['num_masks']}", str(debug_dir))

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
        debug_dir = self._save_input_visualization()
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
        debug_dir = self._save_flame_visualization()
        return StepResult(f"Generated masks/FLAME for {len(exports)} views", str(debug_dir))

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

    def align_sim3(self, flame_target_transforms: Optional[str | Path] = None) -> StepResult:
        colmap_transforms = self.workspace.root / "transforms_colmap_raw.json"
        if not colmap_transforms.exists():
            raise FileNotFoundError("Missing transforms_colmap_raw.json. Run or import COLMAP first.")
        flame_target = Path(flame_target_transforms) if flame_target_transforms else self._build_flame_target_transforms()
        sim3 = align_colmap_to_flame(
            colmap_transforms,
            flame_target,
            self.workspace.root / "transforms_aligned.json",
            self.workspace.root / "sim3_colmap_to_lam.json",
        )
        plot = self._save_sim3_visualization(flame_target)
        return StepResult(f"Sim3 estimated. scale={sim3.scale:.5f}, rmse={sim3.rmse:.5f}", str(plot))

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
            out = lam_model.renderer.forward_animate_gs(
                [gs],
                query_points,
                flame_params,
                batch.c2ws,
                batch.intrs,
                h,
                w,
                batch.bg_colors,
            )
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
        batch = load_multiview_bundle(self.workspace.root)
        gs = _load_renderable_gaussian(init_ply)
        gs.to_cuda()
        config = config or RefinementConfig(output_dir=str(self.workspace.root))
        config.output_dir = str(self.workspace.root)
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
        shape = None
        images = sorted([p for p in self.workspace.images_dir.glob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"}])
        target_frames = []
        for idx, export_dir in enumerate(exports):
            stem = images[idx].stem if idx < len(images) else f"{idx:05d}"
            mask_src = next((export_dir / "fg_masks").glob("*"))
            flame_src = next((export_dir / "flame_param").glob("*.npz"))
            shutil.copy2(mask_src, self.workspace.root / "fg_masks" / f"{stem}.png")
            shutil.copy2(flame_src, self.workspace.flame_dir / f"{stem}.npz")
            if (export_dir / "landmark2d" / "landmarks.npz").exists():
                shutil.copy2(export_dir / "landmark2d" / "landmarks.npz", self.workspace.landmark_dir / f"{stem}.npz")
            if shape is None and (export_dir / "canonical_flame_param.npz").exists():
                shape = export_dir / "canonical_flame_param.npz"
            transforms_path = export_dir / "transforms.json"
            if transforms_path.exists() and idx < len(images):
                db = json.loads(transforms_path.read_text(encoding="utf-8"))
                frame = dict(db["frames"][0])
                frame["file_path"] = f"images/{images[idx].name}"
                frame["image_name"] = images[idx].name
                frame["fg_mask_path"] = f"fg_masks/{stem}.png"
                frame["flame_param_path"] = f"flame_param/{stem}.npz"
                frame["landmark_path"] = f"landmark2d/{stem}.npz"
                frame["timestep_index"] = idx
                frame["camera_index"] = idx
                target_frames.append(frame)
        if shape is not None:
            shutil.copy2(shape, self.workspace.root / "canonical_flame_param.npz")
        if len(target_frames) >= 3:
            (self.workspace.root / "transforms_flame_target.json").write_text(
                json.dumps({"frames": target_frames}, indent=2),
                encoding="utf-8",
            )

    def _build_flame_target_transforms(self) -> Path:
        out = self.workspace.root / "transforms_flame_target.json"
        if not out.exists():
            raise FileNotFoundError(
                "Missing transforms_flame_target.json. Run FLAME tracking first or provide explicit target transforms; "
                "Sim(3) identity/circular fake alignment is intentionally not used."
            )
        return out

    def _save_input_visualization(self) -> Path:
        out_dir = self.workspace.debug_dir / "inputs"
        save_workspace_image_previews(self.workspace.root, out_dir)
        return out_dir

    def _save_flame_visualization(self) -> Path:
        out_dir = self.workspace.debug_dir / "flame"
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
