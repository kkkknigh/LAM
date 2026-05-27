import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from omegaconf import OmegaConf
from safetensors.torch import load_file

from lam.models import ModelLAM
from lam.multiview_refine.optimization import RefinementConfig, RefinementStageConfig
from lam.multiview_refine.pipeline import MultiViewRefinePipeline


def build_lam(config_path: str, model_name: str):
    cfg = OmegaConf.load(config_path)
    model = ModelLAM(**cfg.model)
    ckpt_path = os.path.join(model_name, "model.safetensors")
    ckpt = load_file(ckpt_path, device="cpu")
    state_dict = model.state_dict()
    for key, value in ckpt.items():
        if key in state_dict and state_dict[key].shape == value.shape:
            state_dict[key].copy_(value)
    model.to("cuda").eval()
    return model


def build_refine_config(args) -> RefinementConfig:
    if args.stage == "calibrate":
        stages = [RefinementStageConfig(
            "calibrate",
            steps=args.steps,
            lr=args.lr,
            views_per_step=args.views_per_step,
            resolution_scale=0.5,
            optimize_global_sim3=True,
            optimize_per_view_camera=True,
            optimize_intrinsics=True,
            use_ssim=True,
            use_landmark=True,
        )]
    elif args.stage == "pose":
        stages = [RefinementStageConfig(
            "pose",
            steps=args.steps,
            lr=args.lr,
            views_per_step=args.views_per_step,
            resolution_scale=0.5,
            optimize_expression=True,
            use_landmark=True,
        )]
    elif args.stage == "appearance":
        stages = [RefinementStageConfig(
            "appearance",
            steps=args.steps,
            lr=args.lr,
            views_per_step=args.views_per_step,
            optimize_appearance=True,
            optimize_exposure=True,
            use_ssim=True,
        )]
    elif args.stage == "geometry_light":
        stages = [RefinementStageConfig(
            "geometry_light",
            steps=args.steps,
            lr=args.lr,
            views_per_step=args.views_per_step,
            optimize_appearance=True,
            optimize_geometry=True,
            use_ssim=True,
            use_knn_anchor=True,
        )]
    elif args.stage == "geometry_xyz":
        stages = [RefinementStageConfig(
            "geometry_xyz",
            steps=args.steps,
            lr=args.lr,
            views_per_step=args.views_per_step,
            optimize_geometry=True,
            use_ssim=True,
            use_knn_anchor=True,
        )]
    else:
        return RefinementConfig(output_dir=args.workspace, device=args.device)
    return RefinementConfig(stages=stages, output_dir=args.workspace, device=args.device)


def main():
    parser = argparse.ArgumentParser(description="Layer 2 multi-view Gaussian refinement")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("create")
    p.add_argument("--output-root", default="output/multiview_refine")
    p.add_argument("--job-id")
    p.add_argument("--zip")
    p.add_argument("--init-ply")

    p = sub.add_parser("import-inputs")
    p.add_argument("--workspace", required=True)
    p.add_argument("--image-dir", required=True)
    p.add_argument("--mask-dir")
    p.add_argument("--flame-dir")
    p.add_argument("--colmap-dir")
    p.add_argument("--init-ply")

    p = sub.add_parser("flame")
    p.add_argument("--workspace", required=True)

    p = sub.add_parser("colmap")
    p.add_argument("--workspace", required=True)
    p.add_argument("--colmap-path", default="colmap")

    p = sub.add_parser("import-colmap")
    p.add_argument("--workspace", required=True)
    p.add_argument("--sparse-dir", required=True)
    p.add_argument("--colmap-path", default="colmap")

    p = sub.add_parser("align-sim3")
    p.add_argument("--workspace", required=True)
    p.add_argument("--flame-target-transforms")

    p = sub.add_parser("manual-sim3")
    p.add_argument("--workspace", required=True)
    p.add_argument("--scale", type=float, required=True)
    p.add_argument("--yaw", type=float, default=0.0)
    p.add_argument("--tx", type=float, default=0.0)
    p.add_argument("--ty", type=float, default=0.0)
    p.add_argument("--tz", type=float, default=0.0)

    refine_commands = [
        "preview",
        "calibrate",
        "pose",
        "appearance",
        "geometry-light",
        "geometry-xyz",
        "refine-appearance",
        "refine-geometry",
        "refine-expression",
        "refine-all",
    ]
    for name in refine_commands:
        p = sub.add_parser(name)
        p.add_argument("--workspace", required=True)
        p.add_argument("--infer-config", required=True)
        p.add_argument("--model-name", required=True)
        p.add_argument("--init-ply")
        p.add_argument("--steps", type=int, default=100)
        p.add_argument("--lr", type=float, default=1e-3)
        p.add_argument("--views-per-step", type=int, default=4)
        p.add_argument("--device", default="cuda")
        p.add_argument("--resume")

    p = sub.add_parser("export")
    p.add_argument("--workspace", required=True)

    args = parser.parse_args()

    if args.cmd == "create":
        pipe = MultiViewRefinePipeline.create(args.output_root, args.job_id)
        if args.zip:
            result = pipe.unpack(args.zip, args.init_ply)
        else:
            result = pipe.workspace.write_state(created=True)
            print(pipe.workspace.root)
            return
    else:
        pipe = MultiViewRefinePipeline(args.workspace)
        if args.cmd == "import-inputs":
            result = pipe.import_inputs(args.image_dir, args.mask_dir, args.flame_dir, args.colmap_dir, args.init_ply)
        elif args.cmd == "flame":
            result = pipe.generate_masks_and_flame()
        elif args.cmd == "colmap":
            result = pipe.run_colmap(args.colmap_path)
        elif args.cmd == "import-colmap":
            result = pipe.import_colmap(args.sparse_dir, args.colmap_path)
        elif args.cmd == "align-sim3":
            result = pipe.align_sim3(args.flame_target_transforms)
        elif args.cmd == "manual-sim3":
            result = pipe.write_manual_alignment(args.scale, args.yaw, args.tx, args.ty, args.tz)
        elif args.cmd == "preview":
            lam = build_lam(args.infer_config, args.model_name)
            result = pipe.preview_alignment(lam, args.init_ply)
        elif args.cmd in set(refine_commands) - {"preview"}:
            stage = {
                "calibrate": "calibrate",
                "pose": "pose",
                "appearance": "appearance",
                "geometry-light": "geometry_light",
                "geometry-xyz": "geometry_xyz",
                "refine-appearance": "appearance",
                "refine-geometry": "geometry_light",
                "refine-expression": "pose",
                "refine-all": "all",
            }[args.cmd]
            args.stage = stage
            lam = build_lam(args.infer_config, args.model_name)
            result = pipe.refine(lam, args.init_ply, build_refine_config(args), resume=args.resume)
        elif args.cmd == "export":
            result = pipe.export()
        else:
            raise NotImplementedError(args.cmd)

    print(f"{result.message}\n{result.path}")


if __name__ == "__main__":
    main()
