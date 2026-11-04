"""Validation-only inference; target maps are read after prediction."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from anchor_training import build_system
from dataset import QueryLoader
from train_a_full import atomic, sha


@torch.inference_mode()
def predict(system, queries, loader):
    device = next(system.velocity.parameters()).device
    condition, beam = loader.inputs(queries, device, arm="A")
    meta = [{"scene_id": f"{q['scene']}|array{q['array']}|f{q['frequency_hz']}|angle{q['angle']}",
             "steering_deg": q["angle"]} for q in queries]
    state, _, _, _ = system.source(beam, meta, "val-draw-0")
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        embedding = system.velocity.embed_model(condition)
    for i in range(8):
        t = torch.full((len(queries),), i / 8, device=device, dtype=torch.float32)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            velocity = system.velocity(image=condition, x=state, step=t, embedding=embedding)
        state = state + velocity.float() / 8.0
    return F.interpolate(state, size=(128, 128), mode="bilinear", align_corners=False).float().cpu().numpy()[:, 0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--checkpoint-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--limit", type=int, default=0)
    a = p.parse_args()
    if a.manifest.name != "val.json":
        raise ValueError("only frozen validation manifest permitted")
    records = json.loads(a.manifest.read_text(encoding="utf-8"))
    if len(records) != 240 or any(
        r["query"]["split"] != "val" or r["query"]["array"] != 8
        or r["query"]["angle"] not in (-14, 0, 14) for r in records
    ):
        raise ValueError("manifest does not match the frozen validation split")
    if a.limit:
        records = records[:a.limit]
    committed = json.loads((a.checkpoint_dir / "committed.json").read_text(encoding="utf-8"))
    ckpath = a.checkpoint_dir / committed["path"]
    if sha(ckpath) != committed["sha256"]:
        raise RuntimeError("checkpoint checksum mismatch")
    ck = torch.load(ckpath, map_location="cpu", weights_only=False)
    identity = ck["identity"]
    if identity["seed"] != 42 or identity["group"] not in (
        "uniform_fixed", "beam_fixed", "uniform_learned", "beam_learned"
    ):
        raise ValueError("invalid checkpoint identity")
    group = identity["group"]
    system = build_system(42, identity["gradient_scale"],
                          "beam" if group.startswith("beam_") else "uniform",
                          group.endswith("_learned"),
                          anchor_count=identity["anchor_count"],
                          mix_strength=identity["mix_strength"],
                          scale_floor=identity["scale_floor"])
    system.load_state_dict(ck["model"])
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for validation")
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    system = system.cuda().eval()
    loader = QueryLoader(a.dataset)
    rows, started = [], time.time()
    for i in range(0, len(records), a.batch):
        chunk = records[i:i + a.batch]
        predictions = predict(system, [r["query"] for r in chunk], loader)
        if not np.isfinite(predictions).all():
            raise RuntimeError("nonfinite prediction")
        # Scoring is deliberately separate from prediction and target-free inputs.
        for record, prediction in zip(chunk, predictions):
            gt = np.load(a.dataset / record["supervision"]["target"], allow_pickle=False)
            valid = (gt > -300) & (gt < 0)
            if not valid.any():
                raise ValueError("empty scoring support")
            mae = float(np.abs(prediction[valid] * 300 - 300 - gt[valid]).mean())
            rows.append({"scene": record["query"]["scene"],
                         "angle": record["query"]["angle"], "mae_db": mae})
    result = {"group": group, "seed": 42, "checkpoint_step": ck["step"],
              "checkpoint_sha256": committed["sha256"], "manifest_sha256": sha(a.manifest),
              "count": len(rows), "nfe": 8, "draws": 1, "batch": a.batch,
              "mae_db_macro": float(np.mean([r["mae_db"] for r in rows])),
              "elapsed_seconds": time.time() - started,
              "purpose": "validation smoke" if a.limit else "checkpoint selection", "rows": rows}
    atomic(a.out, result)
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}), flush=True)


if __name__ == "__main__":
    main()
