"""Exercise stage 4 imports, CUDA kernels and EGL video rendering without model assets."""

import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "text2interaction"))
LOG_DIR = ROOT / "custom_data" / "logs"


def main():
    import numpy as np
    import torch

    imports = [
        "pytorch3d", "pytorch3d._C", "pointnet2_ops", "smplx", "trimesh",
        "pyrender", "clip", "spacy", "pytorch_lightning", "transformers",
        "torchmetrics", "open3d", "openmesh", "bps_torch", "chumpy",
        "human_body_prior.tools.tgm_conversion", "human_body_prior.body_model.body_model", "omegaconf", "scipy",
        "data_loaders.behave.data.dataset", "model.hoi_diff", "utils.guidance", "render.visualize_marker",
    ]
    for name in imports:
        importlib.import_module(name)
        print("IMPORT OK:", name, flush=True)

    assert torch.cuda.is_available(), "CUDA is not available"
    x = torch.arange(64, device="cuda", dtype=torch.float32).reshape(8, 8)
    assert torch.allclose((x @ x.T).cpu(), x.cpu() @ x.cpu().T)

    from pytorch3d.ops import knn_points
    xyz = torch.rand(2, 64, 3, device="cuda")
    nearest = knn_points(xyz, xyz, K=1)
    assert torch.allclose(nearest.dists, torch.zeros_like(nearest.dists), atol=1e-5)
    print("CUDA OK: PyTorch3D nearest neighbours", flush=True)

    from pointnet2_ops import pointnet2_utils
    indices = pointnet2_utils.furthest_point_sample(xyz, 16)
    gathered = pointnet2_utils.gather_operation(xyz.transpose(1, 2).contiguous(), indices)
    assert gathered.shape == (2, 3, 16)
    assert torch.isfinite(gathered).all()
    print("CUDA OK: PointNet2 sampling and gather", flush=True)

    from model.points_encoder import PointNet2Encoder
    encoder = PointNet2Encoder(c_in=1, c_out=512, num_keypoints=1).cuda().eval()
    with torch.no_grad():
        features = encoder(torch.rand(2, 400, 4, device="cuda"))
    assert features.shape == (2, 1, 512)
    assert torch.isfinite(features).all()
    print("CUDA OK: InterAct PointNet2 encoder forward", flush=True)

    from bps_torch.bps import bps_torch
    encoded = bps_torch(custom_basis=xyz[:1, :16]).encode(xyz, feature_type=["deltas"])
    assert encoded["deltas"].shape == (2, 16, 3)
    assert torch.isfinite(encoded["deltas"]).all()
    assert torch.allclose(encoded["deltas"][0], torch.zeros_like(encoded["deltas"][0]), atol=1e-5)
    print("CUDA OK: BPS encoding and Chamfer distance", flush=True)

    import spacy
    assert spacy.load("en_core_web_sm")("A person lifts a box.")[0].pos_
    print("NLP OK: spaCy English pipeline", flush=True)

    from human_body_prior.tools import tgm_conversion
    assert hasattr(tgm_conversion, "angle_axis_to_rotation_matrix")
    identity_rotation = tgm_conversion.angle_axis_to_rotation_matrix(torch.zeros(1, 3, device="cuda"))
    assert torch.allclose(identity_rotation, torch.eye(4, device="cuda").unsqueeze(0), atol=1e-5)

    import imageio.v2 as imageio
    import pyrender
    import trimesh
    scene = pyrender.Scene(bg_color=[0.05, 0.05, 0.05, 1.0])
    node = scene.add(pyrender.Mesh.from_trimesh(trimesh.creation.icosphere(radius=0.3)))
    camera_pose = np.eye(4)
    camera_pose[2, 3] = 2.0
    scene.add(pyrender.PerspectiveCamera(yfov=np.pi / 3.0), pose=camera_pose)
    scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=3.0), pose=camera_pose)
    renderer = pyrender.OffscreenRenderer(256, 256)
    frames = []
    try:
        for i in range(6):
            pose = np.eye(4)
            pose[0, 3] = (i - 2.5) * 0.05
            scene.set_pose(node, pose)
            color, depth = renderer.render(scene)
            assert np.any(depth > 0), "Empty render"
            frames.append(color)
    finally:
        renderer.delete()
    video = LOG_DIR / "stage4_render_smoke.mp4"
    imageio.mimwrite(str(video), frames, fps=6)
    subprocess.run(["ffmpeg", "-v", "error", "-i", str(video), "-f", "null", "-"], check=True)
    print("RENDER OK:", video, flush=True)

    distributions = ["torch", "torchvision", "torchaudio", "pytorch3d", "pointnet2-ops",
                     "numpy", "spacy", "pytorch-lightning", "torchmetrics", "transformers",
                     "human-body-prior", "openmesh", "open3d"]
    report = {
        "python": sys.version,
        "environment": sys.prefix,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "versions": {name: importlib.metadata.version(name) for name in distributions},
        "imports": imports,
        "tests_passed": ["imports", "torch_cuda_matmul", "pytorch3d_cuda_knn",
                         "pointnet2_cuda_sampling", "interact_pointnet2_encoder", "bps_chamfer_cuda", "spacy_english", "hbp_conversion",
                         "egl_render", "mp4_encode_decode"],
        "render_video": str(video.relative_to(ROOT)),
    }
    (LOG_DIR / "stage4_verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print("STAGE 4 VERIFICATION PASSED", flush=True)


if __name__ == "__main__":
    main()
