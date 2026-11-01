#!/usr/bin/env python3
"""Record the edited English source bytes without overwriting original provenance."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT / "provenance" / "ORIGINAL_SNAPSHOT.json"
OUTPUT = ROOT / "snapshot.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    original = json.loads(ORIGINAL.read_text(encoding="utf-8"))
    files = {}
    for relative in sorted(original["code_sha256"]):
        path = ROOT / "code" / relative
        if not path.is_file():
            raise FileNotFoundError(f"missing source listed by archive: {relative}")
        files[relative] = sha256(path)
    snapshot = {
        "code_sha256": files,
        "parent_snapshot_sha256": original.get("parent_snapshot_sha256"),
        "protocol_sha256": original.get("protocol_sha256"),
        "derived_from_original_snapshot_sha256": sha256(ORIGINAL),
        "source_copy": "FOMradio English-language reorganization",
    }
    temporary = OUTPUT.with_name(f"{OUTPUT.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(snapshot, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, OUTPUT)
    print(f"recorded {len(files)} current source files in {OUTPUT}")


if __name__ == "__main__":
    main()
