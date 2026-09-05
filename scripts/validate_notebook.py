#!/usr/bin/env python3
"""Validate the AASIST notebook without executing its GPU workload."""

from __future__ import annotations

import ast
import json
import re
from IPython.core.inputtransformer2 import TransformerManager
import nbformat
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = (
    REPOSITORY_ROOT
    / "notebooks"
    / "aasist_deepfake_finetuning.ipynb"
)


def main() -> None:
    with NOTEBOOK_PATH.open("r", encoding="utf-8") as handle:
        notebook = json.load(handle)

    if notebook.get("nbformat") != 4:
        raise ValueError("Expected Jupyter notebook format 4.")

    cells = notebook.get("cells")
    if not isinstance(cells, list) or not cells:
        raise ValueError("Notebook must contain cells.")

    nbformat.validate(nbformat.from_dict(notebook))
    failures: list[str] = []
    full_source: list[str] = []

    for index, cell in enumerate(cells):
        source = "".join(cell.get("source", []))
        full_source.append(source)

        if cell.get("cell_type") != "code":
            continue
        if cell.get("outputs"):
            failures.append(f"cell {index}: committed outputs are not allowed")
        if cell.get("execution_count") is not None:
            failures.append(f"cell {index}: execution_count must be null")

        if re.search(r"[\U0001F000-\U0001FAFF\u2600-\u27BF\uFE0F\u200D]", source):
            failures.append(f"cell {index}: emoji characters are not allowed")
        source = TransformerManager().transform_cell(source)

        try:
            ast.parse(source, filename=f"{NOTEBOOK_PATH.name}:cell-{index}")
        except SyntaxError as error:
            failures.append(f"cell {index}: {error}")

    joined_source = "\n".join(full_source)
    required_fragments = {
        "pinned upstream revision": "AASIST_REVISION",
        "official output unpacking": "_, logits = model(",
        "official label order": '"spoof": 0',
        "missing-file filtering": 'dropna(subset=["path"])',
        "balanced evaluation": "balanced_accuracy_score",
        "equal error rate": "equal_error_rate",
    }
    for description, fragment in required_fragments.items():
        if fragment not in joined_source:
            failures.append(f"missing safeguard: {description}")

    forbidden_fragments = {
        "runtime source patching": "write_text(",
        "adapter head": "ADAPT_HEAD",
        "manual accuracy": "Accuracy = 0.98",
        "illustrative matrix": "Illustrative Only",
    }
    for description, fragment in forbidden_fragments.items():
        if fragment in joined_source:
            failures.append(f"forbidden migration artifact: {description}")

    if failures:
        raise SystemExit("\n".join(failures))

    print(
        f"Validated {NOTEBOOK_PATH.relative_to(REPOSITORY_ROOT)} "
        f"({len(cells)} cells)"
    )


if __name__ == "__main__":
    main()

