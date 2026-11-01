#!/usr/bin/env python3
"""Create a deterministic share-package checksum manifest."""
from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "CHECKSUMS.sha256"
EXCLUDED = {"CHECKSUMS.sha256"}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    rows = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(ROOT).as_posix()
        if relative in EXCLUDED or "__pycache__" in path.parts or relative.startswith("data/") and relative != "data/README.md":
            continue
        rows.append(f"{digest(path)}  {relative}")
    OUTPUT.write_text("\n".join(rows) + "\n")
    print(f"wrote {len(rows)} checksums to {OUTPUT}")


if __name__ == "__main__":
    main()
