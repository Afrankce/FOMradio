#!/usr/bin/env python3
"""Verify the FOMradio copy without conflating it with the original archive."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "checkpoints" / "t3_step5000.pt"
CHECKPOINT_SHA256 = "611910c197d5a7a79a549654912ae7d637ac8bd030fb099a0bca69c43fc1cd97"
MODEL_SHA256 = "b642b1aa1c5f2da9e54af109cfb52e044eae4d2e669906dadddd9846c4147f78"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_checksums() -> int:
    manifest = ROOT / "CHECKSUMS.sha256"
    if not manifest.is_file():
        raise FileNotFoundError(f"missing current-copy checksum manifest: {manifest}")
    count = 0
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, separator, relative = line.partition("  ")
        if not separator or not relative:
            raise ValueError(f"malformed checksum row: {line!r}")
        path = ROOT / relative
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"current-copy checksum mismatch: {relative}")
        count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checksums", action="store_true",
                        help="also verify all files in the current FOMradio checksum manifest")
    parser.add_argument("--checksums-only", action="store_true",
                        help="verify current-copy files without importing PyTorch")
    args = parser.parse_args()

    package = json.loads((ROOT / "PACKAGE_MANIFEST.json").read_text(encoding="utf-8"))
    if package.get("package") != "FOMradio":
        raise RuntimeError("incorrect project manifest")
    if args.checksums_only:
        print(json.dumps({"status": "ok", "project": "FOMradio",
                          "current_copy_checked_files": verify_checksums()}, indent=2))
        return
    checkpoint_sha256 = sha256(CHECKPOINT)
    if checkpoint_sha256 != CHECKPOINT_SHA256:
        raise RuntimeError(f"archived checkpoint SHA256 mismatch: {checkpoint_sha256}")

    code = ROOT / "code"
    sys.path.insert(0, str(code))
    import torch
    from beam_boundary.common import state_sha
    from cross_array.model import SourceObservationUNO

    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    if int(payload["step"]) != 5000:
        raise RuntimeError(f"unexpected checkpoint step: {payload['step']}")
    model = SourceObservationUNO("T3")
    model.load_state_dict(payload["model"], strict=True)
    model_sha256 = state_sha(model.state_dict())
    if model_sha256 != MODEL_SHA256:
        raise RuntimeError(f"model state SHA256 mismatch: {model_sha256}")

    result = {
        "status": "ok",
        "project": "FOMradio",
        "archived_arm": "T3",
        "checkpoint_step": 5000,
        "checkpoint_sha256": checkpoint_sha256,
        "model_sha256": model_sha256,
        "model_parameters": sum(p.numel() for p in model.parameters()),
        "current_copy_checked_files": verify_checksums() if args.checksums else None,
        "original_source_snapshot": "provenance/ORIGINAL_SNAPSHOT.json",
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
