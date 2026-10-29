#!/usr/bin/env python3
"""
find_external_citations.py

Scan a MyST book's .md and .ipynb files for citations/links that MyST resolves
through an external service (doi.org, arXiv, etc.) instead of your BibTeX file.

For every DOI found, it also checks your .bib files and, if that DOI is already
there, tells you which BibTeX key to use instead.

Usage:
    python find_external_citations.py            # scan current folder
    python find_external_citations.py path/to/book
    python find_external_citations.py --include-code   # also report hits inside code blocks
"""
import argparse
import json
import re
import sys
from pathlib import Path

SKIP_DIRS = {"_build", ".git", "node_modules", ".ipynb_checkpoints", ".venv", "venv", "__pycache__"}

DOI_CORE = r"10\.\d{4,9}/[^\s\"'<>\]\)}`,;]+"

# (label, regex). Order matters: earlier, more specific patterns claim a span first.
PATTERNS = [
    ("doi.org URL",            re.compile(r"(?:https?://)?(?:dx\.)?doi\.org/(" + DOI_CORE + ")", re.I)),
    ("doi: link scheme",       re.compile(r"(?<![\w.])doi:\s*(" + DOI_CORE + ")", re.I)),
    ("DOI in {cite} role",     re.compile(r"\{cite(?::\w+)?\}`([^`]*" + r"10\.\d{4,9}/" + r"[^`]*)`")),
    ("@DOI citation",          re.compile(r"(?<![\w])@(" + DOI_CORE + ")")),
    ("DOI as link target",     re.compile(r"\]\(\s*<?(" + DOI_CORE + r")>?\s*\)")),
    ("DOI in angle brackets",  re.compile(r"<(" + DOI_CORE + r")>")),
    ("arXiv URL",              re.compile(r"(?:https?://)?(?:www\.)?arxiv\.org/(?:abs|pdf)/([\w.\-/]+?)(?:\.pdf|v\d+)?(?=[\s)\]>\"']|$)", re.I)),
    ("arxiv: link scheme",     re.compile(r"(?<![\w.])arxiv:\s*([\w.\-/]+)", re.I)),
    ("rrid: link scheme",      re.compile(r"(?<![\w.])rrid:\s*([\w:\-_]+)", re.I)),
    ("ror: link scheme",       re.compile(r"(?<![\w.])ror:\s*([\w]+)", re.I)),
    ("Zenodo record URL",      re.compile(r"https?://(?:www\.)?zenodo\.org/(?:records?|doi)/([\w./\-]+)", re.I)),
    ("PubMed URL",             re.compile(r"https?://(?:pubmed\.ncbi\.nlm\.nih\.gov|www\.ncbi\.nlm\.nih\.gov/pubmed)/(\d+)", re.I)),
    ("bare DOI in text",       re.compile(r"(?<![\w/])(" + DOI_CORE + ")")),
]

CODE_FENCE = re.compile(r"^(\s*)(`{3,}|~{3,})(.*)$")
CODE_DIRECTIVES = {"{code}", "{code-block}", "{code-cell}", "{sourcecode}"}


def norm_doi(d: str) -> str:
    return d.strip().rstrip(".").lower()


def load_bib_dois(root: Path) -> dict:
    """Map normalised DOI -> bib key from every .bib file under root."""
    dois = {}
    entry_re = re.compile(r"@\w+\s*\{\s*([^,\s]+)\s*,(.*?)(?=\n@|\Z)", re.S)
    doi_re = re.compile(r"\bdoi\s*=\s*[{\"]\s*(?:https?://(?:dx\.)?doi\.org/)?([^}\"]+?)\s*[}\"]", re.I)
    for bib in root.rglob("*.bib"):
        if any(p in SKIP_DIRS for p in bib.parts):
            continue
        text = bib.read_text(encoding="utf-8", errors="replace")
        for key, body in entry_re.findall(text):
            m = doi_re.search(body)
            if m:
                dois[norm_doi(m.group(1))] = f"{key} ({bib.name})"
    return dois


def iter_text_lines(path: Path):
    """Yield (location_label, line_text) for markdown content."""
    if path.suffix == ".ipynb":
        try:
            nb = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"!! Could not parse {path}: {e}", file=sys.stderr)
            return
        for ci, cell in enumerate(nb.get("cells", []), start=1):
            if cell.get("cell_type") not in ("markdown", "raw"):
                continue
            src = cell.get("source", "")
            src = "".join(src) if isinstance(src, list) else src
            for li, line in enumerate(src.splitlines(), start=1):
                yield f"cell {ci}, line {li}", line, ci
    else:
        for li, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
            yield f"line {li}", line, 0


def scan_file(path: Path, include_code: bool):
    hits = []
    fence = None          # current fence marker when inside a code block
    fence_is_code = False
    last_cell = None
    for loc, line, cell in iter_text_lines(path):
        if cell != last_cell:            # new notebook cell resets fence state
            fence, fence_is_code, last_cell = None, False, cell
        m = CODE_FENCE.match(line)
        if m:
            marker, info = m.group(2), m.group(3).strip()
            if fence is None:
                fence = marker
                word = info.split()[0] if info else ""
                # ```{note} etc. are MyST directives containing markdown; only plain/code fences are code
                fence_is_code = not word.startswith("{") or word in CODE_DIRECTIVES
                continue
            elif marker[0] == fence[0] and len(marker) >= len(fence) and not info:
                fence, fence_is_code = None, False
                continue
        in_code = fence is not None and fence_is_code
        if in_code and not include_code:
            continue

        claimed = []
        for label, rx in PATTERNS:
            for mm in rx.finditer(line):
                s, e = mm.span()
                if any(s < ce and e > cs for cs, ce in claimed):
                    continue
                claimed.append((s, e))
                hits.append((loc, label + (" [in code block]" if in_code else ""), mm.group(0), mm.group(1)))
    return hits


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", nargs="?", default=".")
    ap.add_argument("--include-code", action="store_true", help="also report matches inside code blocks")
    args = ap.parse_args()
    root = Path(args.root).resolve()

    bib_dois = load_bib_dois(root)
    print(f"Loaded {len(bib_dois)} DOIs from .bib files under {root}\n")

    files = sorted(p for ext in ("*.md", "*.ipynb") for p in root.rglob(ext)
                   if not any(part in SKIP_DIRS for part in p.parts))
    total, in_bib = 0, 0
    for f in files:
        hits = scan_file(f, args.include_code)
        if not hits:
            continue
        print(f"== {f.relative_to(root)}")
        for loc, label, text, ident in hits:
            total += 1
            note = ""
            doi_match = re.search(DOI_CORE, ident)
            if doi_match:
                key = bib_dois.get(norm_doi(doi_match.group(0)))
                if key:
                    in_bib += 1
                    note = f"   -> already in bib as {key}"
                else:
                    note = "   -> NOT in any .bib file"
            print(f"   {loc:<22} {label:<24} {text}{note}")
        print()

    print(f"Scanned {len(files)} files: {total} external citation link(s) found"
          + (f", {in_bib} of which already have a BibTeX entry." if total else "."))


if __name__ == "__main__":
    main()
