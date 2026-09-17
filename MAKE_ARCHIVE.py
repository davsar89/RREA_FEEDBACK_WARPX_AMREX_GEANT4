#!/usr/bin/env python3
"""Archive this project, skipping generated simulation output.

Every tracked regular file, minus files above a size cut and generated-output
trees. External submodule source is not
copied; ``.gitmodules`` records its fork URLs, and a Git clone also
checks out the exact gitlinks. The result contains source, decks, docs,
configs, and the small data bundles that make a run reproducible -- not
gigabytes of frames.

    python MAKE_ARCHIVE.py --review              # complete review archive
    python MAKE_ARCHIVE.py                       # compact tracked-source archive
    python MAKE_ARCHIVE.py --max-bytes 20000000  # keep bigger files
    python MAKE_ARCHIVE.py --output /tmp/x.tar.gz
    python MAKE_ARCHIVE.py --include-untracked    # add untracked, non-ignored

Review mode raises the size limit so the tracked packed schema-6 tables and
holdouts are included.  It does not include captures, renders, builds,
credentials, or generated code bundles.  The skipped list is printed with
sizes so nothing disappears silently.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent
DEFAULT_MAX_BYTES = 5_000_000
REVIEW_MAX_BYTES = 100_000_000

# Derived trees: regenerate them, never archive them.  (Most are gitignored
# already; listing them keeps --include-untracked honest too.)
SKIP_DIRS = (
    ".git", "tmp", "runs_rrea", "diags", "external", "build",
    "e125_figures", "e125_radio", "e125_videos", "patch_candidate", ".claude",
)
SKIP_NAMES = ("CODE_BUNDLE.md",)

def git(*args: str) -> list[str]:
    result = subprocess.run(
        ["git", "-C", str(REPO), *args],
        capture_output=True, text=True, check=True)
    return [line for line in result.stdout.splitlines() if line]


def is_skipped(rel: Path) -> bool:
    if rel.name in SKIP_NAMES:
        return True
    return any(part in SKIP_DIRS for part in rel.parts)


def collect(include_untracked: bool) -> list[Path]:
    paths = git("ls-files")
    if include_untracked:
        paths += git("ls-files", "--others", "--exclude-standard")
    seen: dict[Path, None] = {}
    for name in paths:
        rel = Path(name)
        if not is_skipped(rel) and (REPO / rel).is_file():
            seen.setdefault(rel, None)
    return sorted(seen)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=None,
                        help="archive path; .zip (default) or .tar.gz")
    parser.add_argument("--max-bytes", type=int, default=None,
                        help=("skip files larger than this; defaults to "
                              f"{DEFAULT_MAX_BYTES} normally and "
                              f"{REVIEW_MAX_BYTES} with --review"))
    parser.add_argument(
        "--review", action="store_true",
        help="raise the size limit for tracked tables and review sources",
    )
    parser.add_argument("--include-untracked", action="store_true",
                        help="also archive untracked files git does not ignore")
    args = parser.parse_args()

    try:
        sha = git("rev-parse", "--short", "HEAD")[0]
    except (subprocess.CalledProcessError, IndexError):
        sha = "nogit"
    max_bytes = args.max_bytes if args.max_bytes is not None else (
        REVIEW_MAX_BYTES if args.review else DEFAULT_MAX_BYTES)
    default_stem = (
        "rrea_warpx_amrex_review" if args.review else "rrea_warpx_amrex")
    output = args.output or REPO / f"{default_stem}_{sha}.zip"

    kept: list[tuple[Path, int]] = []
    skipped: list[tuple[Path, int]] = []
    for rel in collect(args.include_untracked):
        size = (REPO / rel).stat().st_size
        (skipped if size > max_bytes else kept).append((rel, size))

    if output.suffixes[-2:] == [".tar", ".gz"] or output.suffix == ".tgz":
        with tarfile.open(output, "w:gz") as tar:
            for rel, _ in kept:
                tar.add(REPO / rel, arcname=str(Path(sha) / rel))
    else:
        with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as zf:
            for rel, _ in kept:
                zf.write(REPO / rel, arcname=str(Path(sha) / rel))

    total = sum(size for _, size in kept)
    print(f"wrote {output}")
    print(f"  {len(kept)} files, {total / 1e6:.1f} MB raw "
          f"-> {output.stat().st_size / 1e6:.1f} MB compressed (HEAD {sha})")
    if skipped:
        print(f"  skipped {len(skipped)} file(s) over "
              f"{max_bytes / 1e6:.1f} MB:")
        for rel, size in sorted(skipped, key=lambda item: -item[1]):
            print(f"    {size / 1e6:8.1f} MB  {rel}")
    print(f"  derived trees always skipped: {', '.join(SKIP_DIRS)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
