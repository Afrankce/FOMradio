"""Isolated, resumable Matérn runner; does not modify the width32 baseline."""
from __future__ import annotations

import argparse
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import random

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from data_loaders.multiconfig import multiconfig_collate
from evaluation.matern_ablation_metrics import scene_diagnostics
from evaluation.radiomap_metrics import MetricAccumulator
from experiments.provenance import sha256_file
from model.fno import count_tensor_parameters, count_real_scalar_parameters
from model.matern_multiscale_uno import MaternMultiscaleUNO2d, VARIANTS
from training.complex_grad_scaler import ComplexGradScaler
from training.matern_ablation import MaternConfig, MaternTrainer, predict_frozen
from training.multiconfig_trainer import seed_everything
from training.same_frequency_config import SameFrequencyTrainConfig
from training.same_frequency_trainer import infer_manifest_selection, preflight_same_frequency, build_same_frequency_checkpoint_identity, _atomic_json
from training.width32_distributed import source_digest


def evaluate(trainer, context, split, root):
    output = root / f"{split}_diagnostics.json"
    checkpoint = root / "best.pt"
    digest = sha256_file(checkpoint)
    if output.exists():
        previous = json.loads(output.read_text())
        if previous["checkpoint_sha256"] != digest or previous["config_sha256"] != trainer.cfg.config_sha256:
            raise RuntimeError("Existing evaluation has a different checkpoint/config identity")
        return
    payload = torch.load(checkpoint, map_location=trainer.device, weights_only=False)
    if payload["run_identity"]["config_sha256"] != trainer.cfg.config_sha256:
        raise RuntimeError("Evaluation checkpoint config mismatch")
    trainer.ema.ema_model.load_state_dict(payload["ema"])
    del payload
    network = trainer.ema.ema_model
    network.eval()
    dataset = context.val_dataset if split == "val" else context.test_dataset
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=2, collate_fn=multiconfig_collate)
    records, aggregated = [], []
    for steps in (2, 4, 8, 12):
        for realization in range(4):
            accumulator = MetricAccumulator()
            for batch in loader:
                with torch.inference_mode():
                    prediction = predict_frozen(network, trainer.bank_sampler, batch, trainer.device,
                                                steps=steps, realization=realization, use_amp=trainer.cfg.use_amp)
                    target, mask = batch["target"].to(trainer.device), batch["valid_mask"].to(trainer.device)
                    accumulator.update(prediction, target, mask)
                    for index, metadata in enumerate(batch["metadata"]):
                        row = scene_diagnostics(prediction[index:index+1], target[index:index+1], mask[index:index+1])
                        records.append(dict(scene_id=metadata["scene_id"], steps=steps, realization=realization, **row))
                # Retain all N=2 predictions in numerical form; no cherry-picked plots.
                if steps == 2:
                    directory = root / f"{split}_predictions" / f"seed_{realization}"
                    directory.mkdir(parents=True, exist_ok=True)
                    for index, metadata in enumerate(batch["metadata"]):
                        np.savez_compressed(directory / f"{metadata['scene_id']}.npz",
                            prediction=prediction[index].cpu().numpy(), target=batch["target"][index].numpy(),
                            valid_mask=batch["valid_mask"][index].numpy())
            aggregated.append(dict(steps=steps, realization=realization, **accumulator.compute()))
            print(json.dumps(dict(event="evaluation_progress", split=split, **aggregated[-1])), flush=True)
    _atomic_json(output, dict(split=split, array_size=trainer.cfg.array_size, variant=trainer.cfg.variant,
        config_sha256=trainer.cfg.config_sha256, checkpoint_sha256=digest, records=records, aggregated=aggregated,
        fine_metric="Haar finest two details on identically 2x2-averaged benchmark; complete-support only",
        training_seeds=[42], source_sha256=trainer.cfg.source_sha256))


def main():
    parser = argparse.ArgumentParser()
    for name in ("dataset-root", "manifest-path", "height-stats-path", "run-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--array-size", choices=("8x8", "16x16"), required=True)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--resume", choices=("none", "auto"), default="none")
    parser.add_argument("--smoke-optimizer-steps", type=int)
    parser.add_argument("--stop-after-epoch", type=int)
    parser.add_argument("--evaluate", choices=("val", "test"))
    args = parser.parse_args()
    rank, world = int(os.getenv("RANK", "0")), int(os.getenv("WORLD_SIZE", "1"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    torch.set_num_threads(2)
    seed_everything(42)
    model = MaternMultiscaleUNO2d(args.variant, operator_width=32)
    beam_id, _ = infer_manifest_selection(args.manifest_path, args.array_size)
    root = args.run_root / "_smoke" if args.smoke_optimizer_steps else args.run_root
    rank_root = root if rank == 0 else root / f"rank_{rank}"
    base = SameFrequencyTrainConfig(dataset_root=args.dataset_root, manifest_path=args.manifest_path,
            height_stats_path=args.height_stats_path, run_root=rank_root, array_size=args.array_size, beam_id=beam_id)
    cfg = MaternConfig(base, world, count_tensor_parameters(model), count_real_scalar_parameters(model),
                       source_digest(Path(__file__).parent), variant=args.variant)
    context = preflight_same_frequency(cfg)
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", timeout=timedelta(minutes=30))
    try:
        path = rank_root / "config.json"
        record = {**cfg.to_record(), "trainable_real_parameters": sum(p.numel() * (2 if p.is_complex() else 1)
                                                                      for p in model.parameters() if p.requires_grad)}
        if path.exists() and json.loads(path.read_text())["config_sha256"] != cfg.config_sha256:
            raise RuntimeError("Refusing incompatible run configuration")
        if not path.exists():
            _atomic_json(path, record)
        last = rank_root / "last.pt"
        if last.exists() and args.resume == "none" and not args.evaluate:
            raise RuntimeError("Refusing to overwrite existing training")
        model.to(device)
        for value in model.state_dict().values():
            dist.broadcast(torch.view_as_real(value) if value.is_complex() else value, src=0)
        random.seed(42 + rank)
        np.random.seed(42 + rank)
        torch.manual_seed(42 + rank)
        torch.cuda.manual_seed_all(42 + rank)
        generator = torch.Generator().manual_seed(42 + rank)
        sampler = DistributedSampler(context.train_dataset, num_replicas=world, rank=rank, shuffle=True, seed=42)
        if len(context.train_dataset) % world:
            raise RuntimeError("Training sampler would repeat scenes")
        train = DataLoader(context.train_dataset, batch_size=cfg.micro_batch_size, sampler=sampler, generator=generator,
                           num_workers=2, collate_fn=multiconfig_collate, pin_memory=True)
        val = DataLoader(context.val_dataset, batch_size=cfg.eval_batch_size, shuffle=False,
                         num_workers=2 if rank == 0 else 0, collate_fn=multiconfig_collate)
        identity = build_same_frequency_checkpoint_identity(cfg, context, model)
        trainer = MaternTrainer(cfg, model, train, val, device, generator, identity,
                                scaler=ComplexGradScaler("cuda", enabled=True))
        if rank == 0:
            _atomic_json(root / "sampler_specification.json", {"levels": trainer.bank_sampler.specification()})
        if args.evaluate:
            if json.loads((root / "training_runtime.json").read_text())["status"] != "complete":
                raise RuntimeError("Evaluation requires completed training")
            if rank == 0:
                evaluate(trainer, context, args.evaluate, root)
            dist.barrier()
            return
        if args.resume == "auto":
            existence = [None] * world
            dist.all_gather_object(existence, last.exists())
            if any(existence) and not all(existence):
                raise RuntimeError("Incomplete distributed checkpoint transaction")
            if all(existence):
                trainer.resume(last)
                states = [None] * world
                dist.all_gather_object(states, (trainer.completed_epochs, trainer.optimizer_step))
                committed = json.loads((root / "committed_epoch.json").read_text())
                if len(set(states)) != 1 or committed["completed_epochs"] != trainer.completed_epochs:
                    raise RuntimeError("Distributed checkpoint epochs disagree")
        if rank == 0:
            print(json.dumps(dict(event="training_start", **record)), flush=True)
        if args.smoke_optimizer_steps:
            result = trainer.run_smoke(args.smoke_optimizer_steps)
            hashes = []
            for network in (model, trainer.ema.ema_model):
                digest = hashlib.sha256()
                for value in network.state_dict().values():
                    digest.update(value.detach().cpu().numpy().tobytes())
                hashes.append(digest.hexdigest())
            gathered = [None] * world
            dist.all_gather_object(gathered, hashes)
            if any(item != hashes for item in gathered):
                raise RuntimeError("Distributed model or EMA divergence")
            trainer.resume(Path(result["checkpoint"]))
            result["validation"] = trainer.validate()
            result["rank_model_ema_checksums_equal"] = True
            if rank == 0:
                _atomic_json(root / "smoke_verified.json", result)
        else:
            result = trainer.fit(stop_after_epoch=args.stop_after_epoch)
        if rank == 0:
            print(json.dumps(result), flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
