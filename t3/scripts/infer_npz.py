#!/usr/bin/env python3
"""Run T3 Rectified-Flow inference from a normalized six-channel condition."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from cross_array.model import SourceObservationUNO
from cross_array.paths import initial_state
from model.matern_source import SourceSampler


def load_condition(path: Path) -> torch.Tensor:
    with np.load(path, allow_pickle=False) as payload:
        if "condition" not in payload:
            raise ValueError("input NPZ must contain `condition`")
        value = np.asarray(payload["condition"], np.float32)
    if value.ndim == 3:
        value = value[None]
    if value.ndim != 4 or value.shape[1:] != (6, 256, 256):
        raise ValueError(f"expected [B,6,256,256], received {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError("condition contains NaN/Inf")
    valid = value[:, 5]
    if not np.all((valid == 0) | (valid == 1)):
        raise ValueError("Y8_valid channel must be binary")
    return torch.from_numpy(value)


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/t3_step5000.pt")
    parser.add_argument("--scene-id", default="external-scene")
    parser.add_argument("--draws", type=int, default=4)
    parser.add_argument("--nfe", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.draws < 1 or args.nfe < 1:
        raise ValueError("draws and nfe must be positive")

    device = torch.device(args.device)
    condition = load_condition(args.input).to(device)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    torch.manual_seed(42)
    model = SourceObservationUNO("T3").to(device)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    source = SourceSampler("MS", resolution=256, seed=42)
    config = {
        "experiment_kind": "cross_array_transport",
        "source_mean_rule": "fixed_real_y8_0deg_masked_zero",
        "source_noise_scale": 1.0,
    }
    metadata = [{"scene_id": args.scene_id, "steering_deg": 0.0} for _ in range(len(condition))]
    autocast = (lambda: torch.autocast("cuda", dtype=torch.float16)) if device.type == "cuda" else nullcontext
    with autocast():
        embedding = model.embed_model(condition)
    predictions, initials = [], []
    for draw in range(args.draws):
        noise = source.sample(metadata, device=device, namespace=f"inference-draw-{draw}")
        state = initial_state(condition, noise, config)
        initials.append(state.cpu().numpy())
        for step in range(args.nfe):
            with autocast():
                velocity = model(image=condition, x=state, step=step / args.nfe, embedding=embedding)
            state = state + velocity.float() / args.nfe
        predictions.append(state.cpu().numpy())
    result = np.stack(predictions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        prediction_normalized=result,
        prediction_db=result * 300.0 - 300.0,
        initial_state=np.stack(initials),
        nfe=args.nfe,
        checkpoint_step=payload["step"],
        channel_order=np.array(["Tx", "Height", "Beam32", "Beam8_0deg", "Y8_0deg", "Y8_valid"]),
    )
    print(args.output)


if __name__ == "__main__":
    main()

