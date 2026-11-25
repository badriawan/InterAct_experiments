"""Stage the two official downloaded model files without unpacking Torch internals."""

import json
import os
from pathlib import Path
import shutil
import zipfile


def link(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        if target.resolve() != source.resolve():
            raise FileExistsError(target)
        return
    target.symlink_to(os.path.relpath(source, target.parent))


def main():
    root = Path(__file__).resolve().parents[2]
    custom = root / "custom_data"
    pretrained = custom / "assets/t2m_interact/pretrained"
    pretrained.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(custom / "downloads/pretrained.archive") as archive:
        for filename in ("model.pt", "args.json"):
            target = pretrained / filename
            if target.exists():
                raise FileExistsError(target)
            with archive.open("pretrained/" + filename) as source, target.open("wb") as dest:
                shutil.copyfileobj(source, dest)
    evaluator = custom / "assets/eval/markersbps.ckpt"
    evaluator.parent.mkdir(parents=True, exist_ok=True)
    if evaluator.exists():
        raise FileExistsError(evaluator)
    shutil.copy2(custom / "downloads/evaluator.archive", evaluator)
    link(pretrained, root / "text2interaction/save/pretrained")
    link(evaluator, root / "text2interaction/assets/eval/markersbps.ckpt")
    report = {
        "pretrained_url": "https://drive.google.com/file/d/1vfskohWxr7gBuve1MLD1RlGut_xSN8mL/view",
        "evaluator_url": "https://drive.google.com/file/d/1-bpafRyaVHdX4TsltDHiGIxcjw-k1Fnf/view",
        "model": str(pretrained.relative_to(root) / "model.pt"),
        "args": json.loads((pretrained / "args.json").read_text()),
        "evaluator": str(evaluator.relative_to(root)),
    }
    (custom / "manifests/stage5_checkpoints.json").write_text(json.dumps(report, indent=2) + "\n")
    print("Staged official InterAct model, args.json and markersbps.ckpt.")


if __name__ == "__main__":
    main()
