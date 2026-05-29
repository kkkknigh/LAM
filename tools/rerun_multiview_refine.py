import argparse
import os
import sys
from dataclasses import replace
from pathlib import Path

from omegaconf import OmegaConf
from safetensors.torch import load_file

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lam.models import ModelLAM
from multiview_refine.optimization import RefinementConfig
from multiview_refine.pipeline import MultiViewRefinePipeline


DEFAULT_INFER_CONFIG = "./configs/inference/lam-20k-8gpu.yaml"
DEFAULT_MODEL_NAME = "./model_zoo/lam_models/releases/lam/lam-20k/step_045500/"


def build_lam(config_path: str = DEFAULT_INFER_CONFIG, model_name: str = DEFAULT_MODEL_NAME):
    model = ModelLAM(**OmegaConf.load(config_path).model)
    ckpt = load_file(os.path.join(model_name, "model.safetensors"), device="cpu")
    state_dict = model.state_dict()
    for key, value in ckpt.items():
        if key in state_dict and state_dict[key].shape == value.shape:
            state_dict[key].copy_(value)
    model.to("cuda").eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--stages", nargs="+", default=["camera", "pose", "appearance"])
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--debug-every", type=int, default=100)
    parser.add_argument("--skip-alignment", action="store_true")
    args = parser.parse_args()

    workspace = Path(args.workspace)
    pipe = MultiViewRefinePipeline(workspace)
    lam = build_lam()

    if not args.skip_alignment:
        print("[rerun] initialize sim3", flush=True)
        print(pipe.initialize_sim3_from_layer1(lam), flush=True)
        print("[rerun] landmark calibrate", flush=True)
        print(pipe.calibrate_global_sim3_blackbox(lam), flush=True)

    defaults = {stage.name: stage for stage in RefinementConfig().stages}
    stages = [replace(defaults[name]) for name in args.stages]
    config = RefinementConfig(
        stages=stages,
        output_dir=str(workspace / "refine"),
        log_every=args.log_every,
        debug_every=args.debug_every,
    )
    print(f"[rerun] refine {'+'.join(args.stages)}", flush=True)
    print(pipe.refine(lam, None, config, resume=None), flush=True)
    print("[rerun] done", flush=True)


if __name__ == "__main__":
    main()
