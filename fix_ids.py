#!/usr/bin/env python3
"""
fix_ids.py
Give notebook cells new, unique IDs in every .ipynb file in a folder
and all of its subfolders. This clears MyST's "Duplicate identifier"
warnings caused by copied notebooks.

Usage:
    python fix_ids.py                      # current folder and everything below it
    python fix_ids.py chapters             # a specific folder
    python fix_ids.py --duplicates-only    # only change IDs that clash

Close the notebooks in Jupyter before running, or Jupyter may save the
old IDs back over the fixed files.
"""
import argparse
import sys
import uuid
from collections import Counter
from pathlib import Path

import nbformat

SKIP_DIRS = {".git", "_build", ".ipynb_checkpoints", "node_modules", "__pycache__"}


def find_notebooks(folder):
    for path in sorted(folder.rglob("*.ipynb")):
        if not SKIP_DIRS.intersection(path.relative_to(folder).parts):
            yield path


def new_id(used):
    while True:
        cid = uuid.uuid4().hex[:8]
        if cid not in used:
            used.add(cid)
            return cid


def main():
    ap = argparse.ArgumentParser(description="Regenerate notebook cell IDs recursively.")
    ap.add_argument("folder", nargs="?", default=".", help="folder to process (default: .)")
    ap.add_argument("--duplicates-only", action="store_true",
                    help="only change IDs that are used more than once")
    args = ap.parse_args()

    folder = Path(args.folder).resolve()
    if not folder.is_dir():
        sys.exit(f"Not a folder: {folder}")

    notebooks = list(find_notebooks(folder))
    if not notebooks:
        sys.exit(f"No .ipynb files found in {folder}")

    # Read every notebook first so duplicates can be found across all of them
    books = {}
    for path in notebooks:
        try:
            books[path] = nbformat.read(path, as_version=4)
        except Exception as err:
            print(f"  SKIP  {path.relative_to(folder)}  (could not read: {err})")

    counts = Counter(c.get("id") for nb in books.values() for c in nb.cells)
    used = set(counts)
    seen = set()
    total = 0

    for path, nb in books.items():
        changed = 0
        for cell in nb.cells:
            cid = cell.get("id")
            if args.duplicates_only:
                # keep the first copy of each ID; replace later copies and missing IDs
                if cid and counts[cid] == 1:
                    continue
                if cid and cid not in seen:
                    seen.add(cid)
                    continue
            cell["id"] = new_id(used)
            changed += 1

        if changed:
            nb.nbformat_minor = max(nb.nbformat_minor, 5)  # cell IDs need format 4.5+
            nbformat.write(nb, path)
            total += changed
            print(f"  fixed {path.relative_to(folder)}  ({changed} cell{'s' if changed != 1 else ''})")

    print(f"\n{len(books)} notebook(s) checked, {total} cell ID(s) replaced.")


if __name__ == "__main__":
    main()
