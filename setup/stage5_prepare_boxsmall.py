"""Create the pilot object's official vertex samples and 1024-point BPS."""

import json
import os
from pathlib import Path

import numpy as np
import torch
import trimesh
from bps_torch.bps import bps_torch


def main():
    root = Path(__file__).resolve().parents[2]
    folder = root / "data/behave/objects/boxsmall"
    mesh = trimesh.load(folder / "boxsmall.obj", force="mesh")
    # process_behave.py already centers the mesh; do not apply another transform.
    assert np.linalg.norm(mesh.vertices.mean(axis=0)) < 1e-6, "Run BEHAVE conversion before sampling"
    indices = np.load(root / "assets/sample_objids/behave/boxsmall.npy")
    assert indices.min() >= 0 and indices.max() < len(mesh.vertices)
    points = mesh.vertices[indices]
    np.save(folder / "sample_points.npy", points)
    assert np.array_equal(np.load(folder / "sample_points.npy"), mesh.vertices[indices])

    basis = torch.from_numpy(np.load(root / "assets/bps_basis_set_1024_1.npy")).float().cuda()
    geometry = torch.from_numpy(np.asarray(mesh.vertices)[None]).float().cuda()
    deltas = bps_torch().encode(x=geometry, feature_type=["deltas"], custom_basis=basis[None])["deltas"]
    assert deltas.shape == (1, 1024, 3) and torch.isfinite(deltas).all()
    output_root = root / "custom_data/assets/behave/objects_bps"
    output = output_root / "boxsmall/boxsmall_1024.npy"
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, deltas.detach().cpu().numpy())
    link = root / "data/behave/objects_bps"
    if link.exists() or link.is_symlink():
        if link.resolve() != output_root.resolve():
            raise FileExistsError(link)
    else:
        link.symlink_to(os.path.relpath(output_root, link.parent))

    converted = []
    for name in ("Date03_Sub03_boxsmall", "Date04_Sub05_boxsmall", "Date07_Sub08_boxsmall"):
        seq = root / "data/behave/sequences" / name
        with np.load(seq / "human.npz", allow_pickle=True) as human, np.load(seq / "object.npz", allow_pickle=True) as obj:
            n = len(human["poses"])
            assert human["poses"].shape == (n, 156)
            assert human["trans"].shape == obj["trans"].shape == obj["angles"].shape == (n, 3)
            assert str(obj["name"]) == "boxsmall"
            for data, keys in ((human, ("poses", "betas", "trans")), (obj, ("trans", "angles"))):
                assert all(np.isfinite(data[key]).all() for key in keys)
            converted.append({"source_sequence": name, "frames": n, "finite": True,
                              "official_caption_available": False, "canonical_test_segment": False})
    report = {
        "dataset": "behave", "object": "boxsmall", "mesh_vertices": len(mesh.vertices),
        "mesh_center": mesh.vertices.mean(axis=0).tolist(), "sample_points_shape": list(points.shape),
        "official_vertex_indices_preserved": True, "bps_shape": list(deltas.shape),
        "bps_finite": True, "converted_source_sequences": converted,
        "canonicalization_with_annotated_segments_performed": False,
    }
    (root / "custom_data/manifests/stage5_boxsmall_verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
