from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch
import numpy as np
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from data_loaders.multiconfig import multiconfig_collate
from evaluation.radioflow_sampling import euler_cfg_sample
from evaluation.radiomap_metrics import MetricAccumulator
from experiments.provenance import sha256_file
from model.attention_multiscale_uno import AttentionMultiscaleUNO2d
from model.fno import count_tensor_parameters, count_real_scalar_parameters
from training.complex_grad_scaler import ComplexGradScaler
from training.multiconfig_trainer import seed_everything
from training.same_frequency_config import SameFrequencyTrainConfig
from training.same_frequency_trainer import (
    infer_manifest_selection, preflight_same_frequency,
    build_same_frequency_checkpoint_identity, _atomic_json,
)
from training.width32_distributed import Width32Config, Width32Trainer, source_digest


def main():
    parser = argparse.ArgumentParser()
    for name in ("dataset-root", "manifest-path", "height-stats-path", "run-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--array-size", choices=("8x8", "16x16", "32x32"), required=True)
    parser.add_argument("--resume", choices=("none", "auto"), default="none")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--train-batch-size", type=int, choices=(128, 64), default=128)
    parser.add_argument("--eval-batch-size", type=int, choices=(256, 128), default=256)
    parser.add_argument("--capacity-check", action="store_true",
                        help="with smoke: stress full eval batch using repeated validation input, no test data")
    controls = parser.add_mutually_exclusive_group()
    controls.add_argument("--smoke-optimizer-steps", type=int)
    controls.add_argument("--stop-after-epoch", type=int)
    controls.add_argument("--test-only", action="store_true",
                          help="evaluate best EMA once after completed training; reuse verified metrics on rerun")
    args = parser.parse_args()
    if args.capacity_check and not args.smoke_optimizer_steps:
        parser.error("--capacity-check requires --smoke-optimizer-steps")
    rank, world = int(os.getenv("RANK", "0")), int(os.getenv("WORLD_SIZE", "1"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    torch.set_num_threads(2)
    seed_everything(42)
    model = AttentionMultiscaleUNO2d(operator_width=32)
    beam_id, _ = infer_manifest_selection(args.manifest_path, args.array_size)
    root = args.run_root / "_smoke" if args.smoke_optimizer_steps else args.run_root
    rank_root = root if rank == 0 else root / f"rank_{rank}"
    base = SameFrequencyTrainConfig(
        dataset_root=args.dataset_root, manifest_path=args.manifest_path,
        height_stats_path=args.height_stats_path, run_root=rank_root,
        array_size=args.array_size, beam_id=beam_id,
    )
    cfg = Width32Config(base, world, count_tensor_parameters(model),
                        count_real_scalar_parameters(model), source_digest(Path(__file__).parent),
                        train_batch_size=args.train_batch_size, eval_batch_size=args.eval_batch_size)
    context = preflight_same_frequency(cfg)
    if args.preflight_only:
        print(json.dumps({"status": "preflight_complete", **cfg.to_record()}, allow_nan=False), flush=True)
        return
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", timeout=timedelta(minutes=10))
    try:
        path = rank_root / "config.json"
        if path.exists() and json.loads(path.read_text())["config_sha256"] != cfg.config_sha256:
            raise RuntimeError("Existing width32 config identity differs")
        if not path.exists():
            _atomic_json(path, cfg.to_record())
        last = rank_root / "last.pt"
        if args.resume == "none" and last.exists() and not args.test_only:
            raise RuntimeError("Refusing to overwrite an existing run; use --resume auto")
        model = model.to(device)
        # Same CPU initialization; broadcast real views to explicitly establish identity.
        for value in model.state_dict().values():
            dist.broadcast(torch.view_as_real(value) if value.is_complex() else value, src=0)
        # The benchmark helper intentionally only accepts the base seed 42.
        # Rank-specific sampling streams are explicit and do not alter initialization.
        random.seed(cfg.seed + rank)
        np.random.seed(cfg.seed + rank)
        torch.manual_seed(cfg.seed + rank)
        torch.cuda.manual_seed_all(cfg.seed + rank)
        generator = torch.Generator().manual_seed(cfg.seed + rank)
        sampler = DistributedSampler(context.train_dataset, num_replicas=world, rank=rank,
                                     shuffle=True, seed=cfg.seed, drop_last=False)
        if len(context.train_dataset) % world:
            raise RuntimeError("Distributed sampler must not pad/repeat training scenes")
        train = DataLoader(context.train_dataset, batch_size=cfg.micro_batch_size,
                           sampler=sampler, generator=generator, num_workers=cfg.num_workers,
                           collate_fn=multiconfig_collate, pin_memory=True)
        val = DataLoader(context.val_dataset, batch_size=cfg.eval_batch_size, shuffle=False,
                         num_workers=cfg.num_workers if rank == 0 else 0,
                         collate_fn=multiconfig_collate)
        identity = build_same_frequency_checkpoint_identity(cfg, context, model)
        trainer = Width32Trainer(cfg, model, train, val, device, generator, identity,
                                 scaler=ComplexGradScaler("cuda", enabled=True))
        if args.resume == "auto" and not args.test_only:
            existence = [None] * world
            dist.all_gather_object(existence, last.exists())
            if any(existence) and not all(existence):
                raise RuntimeError("Incomplete rank checkpoint set; cannot safely resume")
            if all(existence):
                trainer.resume(last)
                states = [None] * world
                dist.all_gather_object(states, (trainer.completed_epochs, trainer.optimizer_step))
                committed = json.loads((root / "committed_epoch.json").read_text())
                if len(set(states)) != 1 or committed["completed_epochs"] != trainer.completed_epochs:
                    raise RuntimeError("Rank checkpoint epochs disagree; incomplete checkpoint transaction")
        if rank == 0:
            print(json.dumps({"event": "evaluation_start" if args.test_only else "training_start",
                              **cfg.to_record()}, allow_nan=False), flush=True)
        if args.test_only:
            runtime = json.loads((root / "training_runtime.json").read_text())
            if runtime["status"] != "complete":
                raise RuntimeError("Test evaluation requires completed training")
            best_hash = sha256_file(root / "best.pt")
            output = root / "test_metrics.json"
            if output.exists():
                result = json.loads(output.read_text())
                if result["config_sha256"] != cfg.config_sha256 or result["checkpoint_sha256"] != best_hash:
                    raise RuntimeError("Existing immutable test results belong to a different run")
            else:
                trainer.resume(rank_root / "best.pt")
                trainer.val_loader = DataLoader(context.test_dataset,
                    batch_size=cfg.eval_batch_size, shuffle=False,
                    num_workers=cfg.num_workers if rank == 0 else 0,
                    collate_fn=multiconfig_collate)
                metrics = trainer.validate()
                if metrics["n_samples"] != 160:
                    raise RuntimeError("Test must cover exactly 160 distinct scenes")
                result = {"status": "test_complete", "array_size": cfg.array_size,
                          "config_sha256": cfg.config_sha256, "checkpoint_sha256": best_hash,
                          "eval_batch_size": cfg.eval_batch_size, "cfg_scale": 1.0,
                          "euler_steps": 2, "metrics": metrics}
                if rank == 0:
                    _atomic_json(output, result)
                dist.barrier()
        elif args.smoke_optimizer_steps:
            result = trainer.run_smoke(args.smoke_optimizer_steps)
            checksums = []
            for network in (trainer.model, trainer.ema.ema_model):
                digest = hashlib.sha256()
                for value in network.state_dict().values():
                    digest.update(value.detach().cpu().numpy().tobytes())
                checksums.append(digest.hexdigest())
            gathered = [None] * world
            dist.all_gather_object(gathered, checksums)
            if any(item != checksums for item in gathered):
                raise RuntimeError("Model/EMA diverged between ranks")
            result["rank_model_ema_checksums_equal"] = True
            # Strictly reload every rank's newly written full-state checkpoint.
            trainer.resume(Path(result["checkpoint"]))
            if args.capacity_check and rank == 0:
                sample = context.val_dataset[0]
                n = cfg.eval_batch_size
                condition = sample["condition"].unsqueeze(0).repeat(n, 1, 1, 1).to(device)
                target = sample["target"].unsqueeze(0).repeat(n, 1, 1, 1).to(device)
                mask = sample["valid_mask"].unsqueeze(0).repeat(n, 1, 1, 1).to(device)
                trainer.ema.ema_model.eval()
                prediction = euler_cfg_sample(trainer.ema.ema_model, condition,
                                              torch.randn_like(target), cfg_scale=1, steps=2)
                accumulator = MetricAccumulator()
                accumulator.update(prediction, target, mask)
                assert accumulator.compute()["n_samples"] == n
                result["eval_capacity_batch"] = n
                result["capacity_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
                del condition, target, mask, prediction
                torch.cuda.empty_cache()
            dist.barrier()
        else:
            result = trainer.fit(stop_after_epoch=args.stop_after_epoch)
        if rank == 0:
            print(json.dumps(result, allow_nan=False), flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
