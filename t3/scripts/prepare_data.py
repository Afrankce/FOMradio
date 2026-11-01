#!/usr/bin/env python3
"""Rebuild the exact T3 processed cache from the official raw dataset."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from cross_array.geometry import regions_from_height


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def load_npy(path: Path) -> np.ndarray:
    return np.load(path, allow_pickle=False)


def resize(value: np.ndarray) -> torch.Tensor:
    return F.interpolate(torch.as_tensor(value)[None, None].float(), (256, 256), mode="bilinear", align_corners=False)[0]


def normalize_map(value: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    value = np.asarray(value, np.float32)
    if value.shape != (128, 128) or not np.isfinite(value).all():
        raise ValueError(f"invalid radiomap shape/content: {value.shape}")
    valid = (value > -300) & (value < 0)
    if not np.all(valid | (value == 1000) | (value == -300)):
        raise ValueError("unexpected radiomap labels")
    field = torch.from_numpy(np.where(valid, (value + 300) / 300, 0))[None, None]
    field = F.interpolate(field, (256, 256), mode="bilinear", align_corners=False)[0]
    mask = F.interpolate(torch.from_numpy(valid.astype(np.float32))[None, None], (256, 256), mode="nearest")[0].bool()
    return torch.where(mask, field, 0), mask


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", choices=("train", "val", "test"), default=("train", "val", "test"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    raw = args.raw_root.resolve()
    data = ROOT / "data"
    data.mkdir(exist_ok=True)
    (data / "geometry").mkdir(exist_ok=True)
    tx = torch.zeros(1, 256, 256)
    tx[:, 127, 127] = 1
    beam_cache: dict[str, torch.Tensor] = {}
    height_cache: dict[str, torch.Tensor] = {}
    source_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    split_info = {}

    def checked(relative: str, expected: str | None = None) -> np.ndarray:
        path = raw / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        if expected and sha256(path) != expected:
            raise RuntimeError(f"SHA256 mismatch: {relative}")
        return load_npy(path)

    def beam(relative: str) -> torch.Tensor:
        if relative not in beam_cache:
            beam_cache[relative] = resize((np.clip(checked(relative), -300, 0) + 300) / 300)
        return beam_cache[relative]

    for split in args.splits:
        records = json.loads((ROOT / "manifests" / f"{split}_records.json").read_text())
        outputs = [data / f"{split}_{key}.npy" for key in ("condition", "target", "valid")]
        if any(path.exists() for path in outputs) and not args.overwrite:
            raise FileExistsError(f"{split} cache exists; pass --overwrite to replace it")
        arrays = {
            "condition": np.lib.format.open_memmap(outputs[0], mode="w+", dtype=np.float32, shape=(len(records), 6, 256, 256)),
            "target": np.lib.format.open_memmap(outputs[1], mode="w+", dtype=np.float32, shape=(len(records), 1, 256, 256)),
            "valid": np.lib.format.open_memmap(outputs[2], mode="w+", dtype=bool, shape=(len(records), 1, 256, 256)),
        }
        for index, record in enumerate(records):
            scene = record["scene_id"]
            if scene not in height_cache:
                height = checked(record["height_path"], record["height_sha256"])
                if height.shape != (256, 256) or not np.isfinite(height).all() or height.min() < 0 or height.max() > 445:
                    raise ValueError(f"invalid height map: {scene}")
                height_cache[scene] = resize(height / 445)
            if scene not in source_cache:
                source_cache[scene] = normalize_map(checked(record["source_radiomap_path"], record["source_sha256"]))
            target, valid = normalize_map(checked(record["radiomap_path"], record["sha256"]))
            source, source_valid = source_cache[scene]
            arrays["condition"][index] = torch.cat((tx, height_cache[scene], beam(record["beam_map_path"]),
                beam(record["source_beam_map_path"]), source, source_valid.float())).numpy()
            arrays["target"][index] = target.numpy()
            arrays["valid"][index] = valid.numpy()
            if index % 100 == 0:
                print(json.dumps({"split": split, "done": index + 1, "total": len(records)}), flush=True)
        for array in arrays.values():
            array.flush()
        (data / f"{split}_records.json").write_text(json.dumps(records, indent=2))
        scenes = sorted({record["scene_id"] for record in records})
        split_info[split] = {"records": len(records), "scenes": len(scenes)}
        if split != "train":
            for scene in scenes:
                height_path = raw / f"height_maps/{scene}/{scene}_height_matrix.npy"
                coord_path = height_path.with_name(f"{scene}_height_matrix_coords.npz")
                with np.load(coord_path, allow_pickle=False) as coordinates:
                    regions = regions_from_height(load_npy(height_path), coordinates)
                np.savez_compressed(data / "geometry" / f"{scene}.npz", **regions)
    audit = {"passed": True, "raw_root": str(raw), "splits": split_info,
             "source_is_real_observation": True, "source_angle_invariant": True,
             "prepared_by": "scripts/prepare_data.py"}
    (ROOT / "input_audit.json").write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()

