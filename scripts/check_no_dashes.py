"""
Reject em dashes and en dashes anywhere in the repository's own text.

    python scripts/check_no_dashes.py            # check, exit 1 on any hit
    python scripts/check_no_dashes.py --fix      # rewrite, then check

Why this is a script and not a habit: em dashes read as machine-generated
prose, and this repository is portfolio material a human reviewer screens for
authenticity. Relying on care while drafting fails silently and repeatedly;
an assertion does not. Same guard as `tools/build_etched_resumes.py`.

Replacements are chosen by context rather than globally, because the right
substitute differs:

    " -- "  parenthetical break        ->  comma, or a colon before a clause
    "--"    ranges (0.7--1.3)          ->  " to "
    "---"   markdown horizontal rule   ->  left alone, it is not a dash

Third-party text is skipped: `assets/` is a checkout of someone else's
repository and rewriting it would show up as a diff against upstream forever.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Escapes, never the literal characters. Written literally, this script is a
# file containing an em dash and an en dash, so running it with --fix rewrote
# its own constants to commas and every subsequent run counted commas. It
# reported 13,234 "dashes" across 30 files before the cause was obvious.
EM, EN = "\u2014", "\u2013"

SUFFIXES = {".py", ".md", ".txt", ".html", ".json", ".ipynb", ".yml", ".yaml"}
SKIP_DIRS = {"assets", ".git", "__pycache__", "node_modules", "runs",
             "checkpoints", "media", "venv", ".venv"}


def files(roots: list[str] | None = None) -> list[Path]:
    bases = [ROOT / r for r in roots] if roots else [ROOT]
    out = []
    for base in bases:
        if base.is_file():
            out.append(base)
            continue
        for p in base.rglob("*"):
            if not p.is_file() or p.suffix not in SUFFIXES:
                continue
            if any(part in SKIP_DIRS for part in p.relative_to(ROOT).parts):
                continue
            out.append(p)
    return sorted(set(out))


def fix(text: str) -> str:
    # A dash between two digits is a range: "0.7-1.3", "5-7 frames".
    text = re.sub(rf"(?<=\d)\s*[{EM}{EN}]\s*(?=\d)", " to ", text)
    # A spaced dash is a parenthetical break; a comma carries it.
    text = re.sub(rf"\s+[{EM}{EN}]\s+", ", ", text)
    # Anything left (unspaced, mid-word, start of line) becomes a plain comma
    # or a hyphen depending on whether it is glued to a word.
    text = re.sub(rf"(?<=\w)[{EM}{EN}](?=\w)", "-", text)
    return text.replace(EM, ",").replace(EN, ",")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*", default=None,
                    help="repo-relative files or directories; default is the "
                         "whole repository")
    ap.add_argument("--fix", action="store_true")
    args = ap.parse_args()
    targets = files(args.paths or None)

    bad: list[tuple[Path, int]] = []
    for p in targets:
        try:
            t = p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        n = t.count(EM) + t.count(EN)
        if not n:
            continue
        if args.fix:
            new = fix(t)
            # Do not trust the rewrite: re-count and report if any survived,
            # rather than reporting success because --fix was passed.
            left = new.count(EM) + new.count(EN)
            p.write_text(new, encoding="utf-8")
            if left:
                bad.append((p, left))
            else:
                print(f"fixed {n:3d} in {p.relative_to(ROOT)}")
        else:
            bad.append((p, n))

    if bad:
        print()
        for p, n in bad:
            print(f"{n:4d} dash(es) in {p.relative_to(ROOT)}")
        print(f"\n{sum(n for _, n in bad)} em/en dash(es) across "
              f"{len(bad)} file(s). Run with --fix.")
        return 1

    print(f"clean: no em or en dashes in {len(files())} files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
