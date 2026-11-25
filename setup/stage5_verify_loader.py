"""Verify the actual Text2Interaction test loader without sampling new motion."""

import json
import os
from pathlib import Path
import random
import sys

import numpy as np
import torch


def main():
    root = Path(__file__).resolve().parents[2]
    os.chdir(root / "text2interaction")
    sys.path.insert(0, str(Path.cwd()))
    from data_loaders.get_data import DatasetConfig, get_dataset_loader

    random.seed(11)
    np.random.seed(11)
    torch.manual_seed(11)
    loader = get_dataset_loader(DatasetConfig(name="behave", batch_size=2, num_frames=300,
                                            split="test", hml_mode="text_only", training_stage=3))
    official = json.loads((root / "custom_data/manifests/stage5_official_segments.json").read_text())
    expected = set(official["selected"])
    actual = set(loader.dataset.t2m_dataset.name_list)
    assert actual == expected and len(actual) >= 2, (actual, expected)
    motion, kwargs = next(iter(loader))
    y = kwargs["y"]
    assert motion.shape == (2, 962, 1, 283), motion.shape
    assert sorted(y["lengths"].tolist()) == [129, 283]
    assert y["obj_points"].shape == (2, 340, 3)
    assert y["obj_bps"].shape == (2, 1, 1024, 3)
    for name, value in (("motion", motion), ("obj_points", y["obj_points"]), ("obj_bps", y["obj_bps"])):
        assert torch.isfinite(value).all(), name
    for index, length in enumerate(y["lengths"].tolist()):
        assert 40 <= length <= 300
        assert y["mask"][index].sum().item() == length
        assert torch.count_nonzero(motion[index, :, :, length:]) == 0
    for name in actual:
        source = root / "data/behave/sequences_seg" / name
        canonical = root / "data/behave/sequences_canonical" / name
        action = np.load(canonical / "action.npy")
        assert np.array_equal(action, np.load(source / "action.npy"))
        assert action.shape == (16,) and action.sum() == 1
        assert (canonical / "text.txt").read_bytes() == (source / "text.txt").read_bytes()

    report = {"dataset": "behave", "loader": "get_dataset_loader", "training_stage": 3,
              "mode": "text_only", "split": "test", "valid_sequence_count": len(loader.dataset),
              "sequence_ids": sorted(actual), "motion_batch_shape": list(motion.shape),
              "lengths": y["lengths"].tolist(), "object_points_shape": list(y["obj_points"].shape),
              "object_bps_shape": list(y["obj_bps"].shape), "captions_in_batch": y["text"],
              "padding_and_masks_valid": True, "source_action_metadata_preserved": True,
              "all_checked_values_finite": True, "motion_sampling_performed": False}
    (root / "custom_data/manifests/stage5_loader_verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
