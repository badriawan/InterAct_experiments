"""Run the repository's BEHAVE conversion on the three pilot source sequences."""

import ast
import hashlib
import json
from pathlib import Path
import os


def main():
    root = Path(__file__).resolve().parents[2]
    os.chdir(root)
    selected = ["Date03_Sub03_boxsmall", "Date04_Sub05_boxsmall", "Date07_Sub08_boxsmall"]
    for name in selected:
        for filename in ("human.npz", "object.npz"):
            if (root / "data/behave/sequences" / name / filename).exists():
                raise FileExistsError("Converted output already exists: " + name + "/" + filename)
    source_path = root / "process/process_behave.py"
    source = source_path.read_text()
    tree = ast.parse(source, filename=str(source_path))
    replacements = 0
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "data_name" for t in node.targets):
            node.value = ast.List(elts=[ast.Constant(value=name) for name in selected], ctx=ast.Load())
            replacements += 1
    assert replacements == 1, "Upstream sequence-selection statement changed"
    ast.fix_missing_locations(tree)
    print("Running original BEHAVE conversion for:", selected, flush=True)
    # Only the source-sequence list is narrowed; the repository transformation is unchanged.
    exec(compile(tree, str(source_path), "exec"), {"__name__": "__main__", "__file__": str(source_path)})
    report = {"source": "process/process_behave.py", "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
              "sequences": selected, "only_runtime_change": "data_name restricted to pilot source sequences",
              "annotations_applied": False, "canonical_test_segments_created": False}
    (root / "custom_data/manifests/stage5_behave_conversion.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
