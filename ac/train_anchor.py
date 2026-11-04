"""Matched one-seed A variants with finite, optionally Beam-guided anchors."""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import signal
import time

import numpy as np
import torch
import torch.distributed as dist

from anchor_training import build_system, make_optimizer, seed_for
from dataset import QueryLoader, training_targets
from model.matern_source import path_triplet
from training.complex_grad_scaler import ComplexGradScaler
from training.detail_losses import detail_components
from training.optimization import OptimizerStepScheduler
from train_a_full import atomic, cpu_all_reduce, indices_for, save, sha, state_sha, sync_gradients


ROOT = Path(__file__).resolve().parent
GROUPS = {
    "uniform_fixed": ("uniform", False),
    "beam_fixed": ("beam", False),
    "uniform_learned": ("uniform", True),
    "beam_learned": ("beam", True),
}
STOP = False


def request_stop(*_):
    global STOP
    STOP = True


def source_identity(group, args, gradient_scale):
    files = ("PLAN.md", "anchor_source.py", "anchor_training.py", "train_anchor.py",
             "beam_scale_source.py", "beam_matern_source.py", "dataset.py",
             "train_a_full.py")
    return {
        "group": group, "seed": args.seed, "manifest_sha256": sha(args.manifest),
        "physics_sha256": sha(args.physics), "dataset_root": str(args.dataset),
        "source_sha256": {name: sha(ROOT / name) for name in files},
        # Public release hashes the unpacked vendored source rather than the
        # original server archive. Its identity is intentionally new.
        "vendor_source_sha256": {p.relative_to(ROOT).as_posix(): sha(p)
                                 for p in sorted((ROOT / "vendor").rglob("*.py"))},
        "torch": torch.__version__, "numpy": np.__version__, "cuda": torch.version.cuda,
        "global_batch": 128, "world_size": int(os.environ["WORLD_SIZE"]),
        "microbatch": args.micro, "total_steps": 5000, "warmup_steps": 500,
        "anchor_count": 4096, "mix_strength": 0.5, "scale_floor": 0.05,
        "gradient_scale": gradient_scale, "backend": "gloo_cpu_reduce",
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--group", choices=GROUPS, required=True)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--physics", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--stop", type=int, required=True)
    p.add_argument("--micro", type=int, default=4)
    a = p.parse_args()
    if a.seed != 42 or a.stop < 1 or a.stop > 5000:
        raise ValueError("this exploratory run is limited to seed 42 and <=5000 updates")
    expected_dir = (ROOT / "runs" / (a.group + "_seed42")).resolve()
    if a.out.resolve() != expected_dir:
        raise ValueError("output must remain in the isolated group directory")
    rank, world, local = (int(os.environ[k]) for k in ("RANK", "WORLD_SIZE", "LOCAL_RANK"))
    if 128 % world or (128 // world) % a.micro:
        raise ValueError("incompatible global/micro batch")
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    dist.init_process_group("gloo", timeout=timedelta(minutes=20))
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    a.out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(a.manifest.read_text(encoding="utf-8"))
    if len(manifest) != 1680 or any(
        r["query"]["split"] != "train" or r["query"]["array"] != 8
        or r["query"]["angle"] not in (-14, 0, 14)
        or int(r["query"]["frequency_hz"]) != 6_700_000_000 for r in manifest
    ):
        raise ValueError("train manifest does not match the frozen U6G protocol")
    physics = json.loads(a.physics.read_text(encoding="utf-8"))
    gradient_scale = float(physics["training_beam_gradient_p95"])
    loader = QueryLoader(a.dataset)
    q_mode, learn_scales = GROUPS[a.group]
    system = build_system(a.seed, gradient_scale, q_mode, learn_scales).to(device)
    optimizer = make_optimizer(system)
    scaler = ComplexGradScaler("cuda")
    scheduler = OptimizerStepScheduler(optimizer, total_steps=5000, warmup_steps=500)
    identity = source_identity(a.group, a, gradient_scale)
    committed = a.out / "committed.json"
    start = 0
    if committed.exists():
        record = json.loads(committed.read_text(encoding="utf-8"))
        ckpath = a.out / record["path"]
        if sha(ckpath) != record["sha256"]:
            raise RuntimeError("checkpoint checksum mismatch")
        ck = torch.load(ckpath, map_location="cpu", weights_only=False)
        if ck["identity"] != identity:
            raise RuntimeError("resume identity mismatch")
        system.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optim"])
        scaler.load_state_dict(ck["scaler"])
        scheduler.load_state_dict(ck["scheduler"])
        start = ck["step"]
        if state_sha(system) != record["model_sha256"]:
            raise RuntimeError("model resume mismatch")
        del ck
    if a.stop < start:
        raise ValueError("stop precedes committed checkpoint")
    if a.stop > 200:
        if a.stop != (start // 250 + 1) * 250:
            raise RuntimeError("full runs must stop at the next validation boundary")
        if start >= 250 and not (a.out / "validation" / f"step_{start:05d}.json").exists():
            raise RuntimeError("preceding validation is missing")
    if rank == 0:
        atomic(a.out / "identity.json", identity)
        print(json.dumps({"phase": "ready", "group": a.group, "step": start,
                          "q_mode": q_mode, "learn_scales": learn_scales,
                          "world_size": world, "microbatch": a.micro,
                          "source_extent": system.source.spatial.extent}), flush=True)
    local_batch = 128 // world
    step, overflow = start, 0
    while step < a.stop:
        tick = time.time()
        indices = indices_for(manifest, a.seed, step)
        selected = indices[rank * local_batch:(rank + 1) * local_batch]
        records = [manifest[i] for i in selected]
        queries = [r["query"] for r in records]
        condition, beam = loader.inputs(queries, device, arm="A")
        target, mask = training_targets(a.dataset, records, device)
        meta = [{"scene_id": q["scene"] + "|slot" + str(rank * local_batch + j),
                 "steering_deg": q["angle"]} for j, q in enumerate(queries)]
        generator = torch.Generator(device=device).manual_seed(seed_for(a.seed, "time", step, rank))
        times = torch.rand(local_batch, generator=generator, device=device)
        counts = torch.stack([detail_components(torch.zeros_like(target), torch.zeros_like(target), mask)[k][1]
                              for k in ("fm", "haar1", "haar2")]).double()
        counts = cpu_all_reduce(counts)
        if (counts <= 0).any():
            raise RuntimeError("empty mask support")
        optimizer.zero_grad(set_to_none=True)
        totals = torch.zeros(3, device=device, dtype=torch.float64)
        last_weights = None
        for i in range(0, local_batch, a.micro):
            sl = slice(i, i + a.micro)
            z, _, weights, _ = system.source(beam[sl], meta[sl], "train-update-" + str(step))
            xt, velocity = path_triplet(z, target[sl], times[sl])
            with torch.autocast("cuda", dtype=torch.float16):
                prediction = system.velocity(image=condition[sl], x=xt, step=times[sl])
            terms = detail_components(prediction, velocity, mask[sl])
            numerator = torch.stack([terms[k][0] for k in ("fm", "haar1", "haar2")])
            loss = (numerator / counts.to(numerator.dtype) * numerator.new_tensor([1., .05, .05])).sum()
            scaler.scale(loss).backward()
            totals += numerator.detach().double()
            last_weights = weights.detach().mean(dim=(0, 2, 3))
        sync_gradients(system)
        totals = cpu_all_reduce(totals)
        scale_grad = 0.0
        if learn_scales:
            norms = [p.grad.detach().float().square().sum() for p in system.source.scale_net.parameters()
                     if p.grad is not None]
            scale_grad = float(torch.stack(norms).sum().sqrt() / scaler.get_scale()) if norms else 0.0
        old_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() < old_scale:
            overflow += 1
            if rank == 0:
                print(json.dumps({"phase": "scale_backoff", "step": step,
                                  "scale": scaler.get_scale()}), flush=True)
            if overflow > 30:
                raise RuntimeError("too many FP16 overflows")
            continue
        overflow = 0
        scheduler.step()
        step += 1
        if rank == 0:
            means = (totals / counts).tolist()
            progress = {"group": a.group, "phase": "training", "step": step,
                        "fm": means[0], "haar1": means[1], "haar2": means[2],
                        "loss": means[0] + .05 * (means[1] + means[2]),
                        "seconds": time.time() - tick, "scale_grad_norm": scale_grad,
                        "scale_weights_last_microbatch": last_weights.cpu().tolist(),
                        "lr_next": optimizer.param_groups[0]["lr"], "scale": scaler.get_scale(),
                        "peak_gpu_gib": torch.cuda.max_memory_allocated() / 1024**3,
                        "timestamp": time.time()}
            atomic(a.out / "progress.json", progress)
            print(json.dumps(progress), flush=True)
        stop = torch.tensor(int(STOP), device="cpu")
        dist.all_reduce(stop, op=dist.ReduceOp.MAX)
        if step % 250 == 0 or step == a.stop or stop.item():
            save(a.out, system, optimizer, scaler, scheduler, step, identity)
        if stop.item():
            break
    if rank == 0:
        atomic(a.out / "state.json", {"phase": "segment_stopped", "group": a.group,
                                     "successful_updates": step, "timestamp": time.time()})
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
