"""Validate official checkpoint keys and text encoding without sampling motion."""

import json
import os
from argparse import Namespace
from pathlib import Path
import sys

import torch


def main():
    root = Path(__file__).resolve().parents[2]
    os.chdir(root / "text2interaction")
    sys.path.insert(0, str(Path.cwd()))
    from utils.model_util import create_model_and_diffusion
    from model.hoi_diff import HOIDiff
    from diffusion.gaussian_diffusion import LocalMotionDiffusion
    from sample.marker2smpl import smplh_model_male

    args = Namespace(**json.loads(Path("save/pretrained/args.json").read_text()))
    context = Namespace(dataset=Namespace())
    model, _ = create_model_and_diffusion(args, context, ModelClass=HOIDiff,
                                        DiffusionClass=LocalMotionDiffusion, diff_steps=1000)
    state = torch.load("save/pretrained/model.pt", map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=True)
    # MDM overrides _apply/train without returning self; do not chain cuda/eval.
    model.cuda()
    model.eval()
    with torch.no_grad():
        text = model.encode_text([
            "A person lifts a box with both hands.",
            "A person carries a box in front of the body.",
        ])
    assert text.shape == (2, 256) and torch.isfinite(text).all()
    report = {
        "checkpoint_keys": len(state), "missing_keys": missing,
        "unexpected_keys": unexpected, "architecture": args.arch,
        "dataset_config": args.dataset, "smplh_sampler_import": "passed",
        "text_encoder_shape": list(text.shape), "text_encoder_finite": True,
        "motion_sampling_performed": False,
    }
    (root / "custom_data/manifests/stage5_checkpoint_verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
