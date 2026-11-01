"""Prepare the existing seed-42 protocol against an actual extracted Dataset root."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np

from data_loaders.multiconfig import HeightStats, compute_train_height_max, _write_immutable_json
from experiments.cross_frequency import (
    select_zero_degree_configurations_same_frequency, validate_same_frequency_records,
)
from experiments.multiconfig_manifest import (
    ARRAY_SPECS, ManifestRecord, canonical_json_bytes, load_schema_lock,
    load_or_create_scene_split, write_manifest_jsonl,
)
from experiments.provenance import sha256_file


def prepare(dataset_root: Path, output: Path):
    root = dataset_root.resolve()
    schema_path = Path(__file__).parent / "experiments/multiconfig_schema.json"
    schema = load_schema_lock(schema_path)
    selected = select_zero_degree_configurations_same_frequency(schema, 6700000000, tuple(ARRAY_SPECS))
    scenes = tuple(p.parent.name for p in root.glob("height_maps/u*/u*_height_matrix.npy"))

    class Inventory:
        def scene_ids_by_array(self):
            return {array: scenes for array in ARRAY_SPECS}

    split_path = output / "scene_split_seed42.json"
    split = load_or_create_scene_split(split_path, Inventory())
    lookup = {scene: name for name in ("train", "val", "test") for scene in getattr(split, name)}
    manifests, train_records, audit = {}, None, {}
    for array, selection in selected.items():
        beams = list((root / "beam_maps" / selection.config_id / "u0").glob(
            f"beam_{selection.beam_id:02d}_angle_*_matrix.npy"))
        if len(beams) != 1:
            raise ValueError(f"Expected exactly one beam for {array}, got {beams}")
        records = []
        for scene in sorted(scenes, key=lambda x: int(x[1:])):
            paths = {
                "height_path": root / "height_maps" / scene / f"{scene}_height_matrix.npy",
                "beam_map_path": beams[0],
                "radiomap_path": root / "radiomaps" / f"{selection.config_id}_beam{selection.beam_id:02d}" / f"{scene}_labeled_radiomap.npy",
            }
            for key, path in paths.items():
                relative = str(path.relative_to(root))
                if relative not in audit:
                    data = np.load(path, allow_pickle=False)
                    label = {"height_path": "height", "beam_map_path": "beam_map", "radiomap_path": "radiomap"}[key]
                    expected = schema.raw["source_metadata"][label]
                    if list(data.shape) != expected["shape"] or str(data.dtype) != expected["dtype"]:
                        raise ValueError(f"Shape/dtype mismatch: {path}")
                    if not np.isfinite(data).all():
                        raise ValueError(f"Nonfinite source: {path}")
                    audit[relative] = {"sha256": sha256_file(path), "shape": list(data.shape), "dtype": str(data.dtype)}
            records.append(ManifestRecord(
                sample_key=f"{scene}|{array}|freq6700000000|angle0.0|beam{selection.beam_id:02d}",
                split=lookup[scene], scene_id=scene, array_name=array,
                array_rows=selection.rows, array_cols=selection.cols, frequency_hz=6700000000,
                config_id=selection.config_id, beam_id=selection.beam_id, steering_deg=0.0,
                **{k: str(p.relative_to(root)) for k, p in paths.items()},
            ))
        validate_same_frequency_records(records, split=split, selected=selected, array_size=array)
        path = output / f"manifest_samefreq_6.7ghz_{array}_0deg.jsonl"
        write_manifest_jsonl(path, records)
        manifests[path.name] = sha256_file(path)
        train_records = [r for r in records if r.split == "train"]
        print(f"{array}: 800 checked maps; beam={selection.beam_id}; config={selection.config_id}", flush=True)
    summary = compute_train_height_max(root, train_records)
    stats = HeightStats(
        schema_version=1, height_max=summary.height_max, derived_from="train",
        scene_count=summary.scene_count, height_files=summary.height_files,
        split_sha256=sha256_file(split_path), manifest_sha256=manifests,
        schema_sha256=hashlib.sha256(canonical_json_bytes(schema.raw)).hexdigest(),
        schema_identity=dict(schema.identities),
    )
    _write_immutable_json(output / "height_stats_train.json", stats.to_dict())
    _write_immutable_json(output / "source_audit.json", {"dataset_root": str(root), "files": audit})
    print(f"train-only height_max={summary.height_max}; checked {len(audit)} unique files", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.dataset_root, args.output)
