#!/usr/bin/env python3
"""Archive this project, skipping generated simulation output.

Includes tracked regular files within the size limit; submodule sources are
excluded. Clone with the gitlinks to obtain those dependencies. Generated
output and credentials remain excluded by Git's tracked/ignored file selection.

    python MAKE_ARCHIVE.py                       # tracked-source archive
    python MAKE_ARCHIVE.py --max-bytes 100000000  # keep bigger files
    python MAKE_ARCHIVE.py --output /tmp/x.tar.gz
    python MAKE_ARCHIVE.py --include-untracked    # add untracked, non-ignored

The default size limit keeps the packed schema-6 tables and skips the largest
literature PDFs. Skipped files are listed with their sizes.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent
DEFAULT_MAX_BYTES = 20_000_000

# Untracked trees .gitignore does not cover; --include-untracked skips them.
SKIP_DIRS = ("build", ".claude")

def git(*args: str) -> list[str]:
    result = subprocess.run(
        ["git", "-C", str(REPO), *args],
        capture_output=True, text=True, check=True)
    return [line for line in result.stdout.splitlines() if line]


def is_skipped(rel: Path) -> bool:
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
    parser.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_BYTES,
                        help="skip files larger than this")
    parser.add_argument("--include-untracked", action="store_true",
                        help="also archive untracked files git does not ignore")
    args = parser.parse_args()

    try:
        sha = git("rev-parse", "--short", "HEAD")[0]
    except (subprocess.CalledProcessError, IndexError):
        sha = "nogit"
    output = args.output or REPO / f"rrea_warpx_amrex_{sha}.zip"

    kept: list[tuple[Path, int]] = []
    skipped: list[tuple[Path, int]] = []
    for rel in collect(args.include_untracked):
        size = (REPO / rel).stat().st_size
        (skipped if size > args.max_bytes else kept).append((rel, size))

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
              f"{args.max_bytes / 1e6:.1f} MB:")
        for rel, size in sorted(skipped, key=lambda item: -item[1]):
            print(f"    {size / 1e6:8.1f} MB  {rel}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
