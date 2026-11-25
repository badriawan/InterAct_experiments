"""Prepare official BEHAVE pilot segments using scoped repository scripts."""

import ast
import hashlib
import json
import os
from pathlib import Path
import runpy
import shutil
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SOURCES = {}
BEHAVE_GLOBALS = {
    "device", "to_cpu", "MODEL_PATH", "SUBJECT_SMPLX_CACHE",
    "smplh_model_male", "smplh_model_female", "smplh10",
}


class PilotScope(ast.NodeTransformer):
    def __init__(self, segments):
        self.segments = segments

    def visit_Assign(self, node):
        names = {target.id for target in node.targets if isinstance(target, ast.Name)}
        if "datasets" in names:
            node.value = ast.List(elts=[ast.Constant(value="behave")], ctx=ast.Load())
        if "data_name" in names and self.segments is not None:
            node.value = ast.List(elts=[ast.Constant(value=name) for name in self.segments], ctx=ast.Load())
        if "smplh10" in names and isinstance(node.value, ast.Dict):
            pairs = [(k, v) for k, v in zip(node.value.keys, node.value.values)
                     if isinstance(k, ast.Constant) and k.value in ("male", "female")]
            node.value.keys = [p[0] for p in pairs]
            node.value.values = [p[1] for p in pairs]
        return self.generic_visit(node)


def run_source(relative, segments=None, only_behave_models=False):
    path = ROOT / relative
    source = path.read_text()
    tree = ast.parse(source, filename=str(path))
    if only_behave_models:
        kept = []
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.If)):
                kept.append(node)
            elif isinstance(node, ast.Assign):
                names = {target.id for target in node.targets if isinstance(target, ast.Name)}
                if names & BEHAVE_GLOBALS:
                    kept.append(node)
        tree.body = kept
    tree = PilotScope(segments).visit(tree)
    ast.fix_missing_locations(tree)
    print("RUN", relative, "segments=", segments, flush=True)
    exec(compile(tree, str(path), "exec"), {"__name__": "__main__", "__file__": str(path)})
    SOURCES[relative] = {"sha256": hashlib.sha256(source.encode()).hexdigest(),
                         "dataset_scope": ["behave"], "segments": segments,
                         "body_models": ["smplh_male_10", "smplh_female_10"] if only_behave_models else None}


def main():
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    custom = ROOT / "custom_data"
    report_path = custom / "manifests/stage5_annotated_preprocessing.json"
    if report_path.exists():
        raise FileExistsError("Preprocessing already completed; verify existing outputs instead of rerunning")
    official = json.loads((custom / "manifests/stage5_official_segments.json").read_text())
    selected = official["selected"]
    assert len(selected) >= 2
    for folder in ("sequences_seg", "sequences_canonical"):
        source = custom / "assets/behave" / folder
        source.mkdir(parents=True, exist_ok=True)
        link = ROOT / "data/behave" / folder
        if link.exists() or link.is_symlink():
            assert link.resolve() == source.resolve(), str(link)
        else:
            link.symlink_to(os.path.relpath(source, link.parent))
    for name in selected:
        assert not (ROOT / "data/behave/sequences_canonical" / name / "human.npz").exists(), name

    run_source("process/process_text.py")
    for name in selected:
        folder = ROOT / "data/behave/sequences_seg" / name
        assert (folder / "text.txt").is_file(), name
    run_source("process/canonicalize_obj.py", segments=selected)
    # The object mesh has reached its final centered state for this pilot.
    runpy.run_path(str(custom / "setup/stage5_prepare_boxsmall.py"), run_name="__main__")
    run_source("process/canonicalize_human.py", segments=selected, only_behave_models=True)
    # The original canonicalizer copies text.txt into the action destinations.
    # Restore the genuine source action files without changing the pose algorithm.
    for name in selected:
        source = ROOT / "data/behave/sequences_seg" / name
        target = ROOT / "data/behave/sequences_canonical" / name
        for filename in ("action.txt", "action.npy"):
            shutil.copy2(source / filename, target / filename)
        action = np.load(target / "action.npy")
        assert action.shape == (16,) and action.sum() == 1
    run_source("process/motion_representation.py", segments=selected, only_behave_models=True)
    outputs = []
    for name in selected:
        folder = ROOT / "data/behave/sequences_canonical" / name
        motion = np.load(folder / "motion.npy")
        expected = next(record for record in official["candidates"] if record["id"] == name)
        assert motion.shape == (expected["frames"], 962), (name, motion.shape)
        assert np.isfinite(motion).all()
        assert len((folder / "text.txt").read_text().splitlines()) == 3
        outputs.append({"id": name, "motion_shape": list(motion.shape), "finite": True,
                        "in_official_test_split": True, "caption_variants": 3})
    report = {"selected": selected, "outputs": outputs, "sources": SOURCES,
              "neutral_model_required_for_this_scoped_path": False,
              "source_action_files_restored_after_canonicalization": True,
              "motion_generation_performed": False}
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
