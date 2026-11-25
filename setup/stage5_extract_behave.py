"""Extract official BEHAVE input files; retain original downloads unchanged."""

import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import zipfile


def safe_target(root, name):
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("Unsafe archive path: " + name)
    target = root.joinpath(*path.parts)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(target)
    return target


def main():
    root = Path(__file__).resolve().parents[2]
    custom = root / "custom_data"
    assets = custom / "assets/behave"
    sequences = assets / "sequences"
    counts = {"raw_sequence_files": 0, "object_files": 0}
    with tarfile.open(custom / "downloads/behave-30fps-params-v1.tar") as archive:
        for member in archive:
            if not member.isfile():
                if member.issym() or member.islnk():
                    raise ValueError("Archive links are not expected")
                continue
            target = safe_target(sequences, member.name)
            with archive.extractfile(member) as source, target.open("wb") as dest:
                shutil.copyfileobj(source, dest)
            counts["raw_sequence_files"] += 1
    with zipfile.ZipFile(custom / "downloads/behave-objects.zip") as archive:
        for member in archive.infolist():
            if member.is_dir():
                continue
            target = safe_target(assets, member.filename)
            with archive.open(member) as source, target.open("wb") as dest:
                shutil.copyfileobj(source, dest)
            counts["object_files"] += 1
    data = root / "data/behave"
    data.mkdir(parents=True, exist_ok=True)
    for name in ("sequences", "objects"):
        source = assets / name
        target = data / name
        if target.exists() or target.is_symlink():
            if target.resolve() != source.resolve():
                raise FileExistsError(target)
        else:
            target.symlink_to(os.path.relpath(source, target.parent))
    counts["raw_sequences"] = len([p for p in sequences.iterdir() if p.is_dir()])
    counts["object_categories"] = len([p for p in (assets / "objects").iterdir() if p.is_dir()])
    (custom / "manifests/stage5_behave_inventory.json").write_text(json.dumps(counts, indent=2) + "\n")
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
