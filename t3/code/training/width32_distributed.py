"""Width-only ablation with synchronous, complex-safe data parallelism.

Gradients are SUM-reduced after normalizing by the GLOBAL valid-pixel count.
Complex gradients communicate through their real views. Reduction precedes
GradScaler's overflow check, so all ranks skip/update together. This is not
PyTorch's DDP wrapper; it performs one fused all-reduce per accumulation window.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist

from experiments.multiconfig_manifest import canonical_json_bytes
from training.same_frequency_config import SameFrequencyTrainConfig
from training.same_frequency_multiscale_uno_config import MultiscaleUNOTrainConfig
from training.multiconfig_trainer import MultiConfigSRMTrainer, _atomic_json
from training.masked_flow_loss import masked_velocity_mse
from training.optimization import AccumulationWindowResult, _scaled_optimizer_step


@dataclass(frozen=True)
class Width32Config:
    base: SameFrequencyTrainConfig
    world_size: int
    tensor_parameter_count: int
    real_scalar_parameter_count: int
    source_sha256: str
    train_batch_size: int = 128
    eval_batch_size: int = 256

    def __post_init__(self):
        if self.world_size not in (1, 2, 4, 8):
            raise ValueError("world size must be 1, 2, 4, or 8")
        if (self.train_batch_size, self.eval_batch_size) not in ((128, 256), (64, 128)):
            raise ValueError("Registered batch pairs are train/eval=128/256 or 64/128")

    def __getattr__(self, name):
        return getattr(MultiscaleUNOTrainConfig(self.base), name)

    @property
    def model_size(self):
        return "attention_multiscale_uno_width32"

    @property
    def operator_width(self):
        return 32

    @property
    def micro_batch_size(self):
        return self.train_batch_size // self.world_size

    @property
    def accumulation_steps(self):
        return 1

    @property
    def effective_batch_size(self):
        return self.train_batch_size

    @property
    def optimizer_steps_per_epoch(self):
        return math.ceil(self.train_samples / self.train_batch_size)

    @property
    def planned_optimizer_steps(self):
        return self.optimizer_steps_per_epoch * self.max_epochs

    @property
    def warmup_steps(self):
        return int(self.planned_optimizer_steps * self.warmup_ratio)

    def scientific_payload(self):
        payload = MultiscaleUNOTrainConfig(self.base).scientific_payload()
        payload.update(
            experiment="same_frequency_6.7_single_beam_attention_multiscale_uno_width32",
            model_size=self.model_size, operator_width=32,
            tensor_parameter_count=self.tensor_parameter_count,
            real_scalar_parameter_count=self.real_scalar_parameter_count,
            world_size=self.world_size, micro_batch_size=self.micro_batch_size,
            accumulation_steps=self.accumulation_steps, effective_batch_size=self.train_batch_size,
            train_batch_size=self.train_batch_size, eval_batch_size=self.eval_batch_size,
            optimizer_steps_per_epoch=self.optimizer_steps_per_epoch,
            planned_optimizer_steps=self.planned_optimizer_steps, warmup_steps=self.warmup_steps,
            gradient_reduction="sum_global_valid_pixel_normalized_real_view",
            data_parallel_backend="synchronous_nccl_manual",
            sampler="distributed_seed42_epoch_permutation_no_padding",
            source_sha256=self.source_sha256,
        )
        return payload

    @property
    def config_sha256(self):
        return hashlib.sha256(canonical_json_bytes(self.scientific_payload())).hexdigest()

    def to_record(self, invocation=None):
        return {**self.scientific_payload(), "config_sha256": self.config_sha256,
                **{key: str(getattr(self, key).resolve()) for key in
                   ("dataset_root", "manifest_path", "height_stats_path", "run_root")}}


def sum_real_view_gradients(model):
    parameters = list(model.parameters())
    used = torch.tensor([p.grad is not None for p in parameters],
                        dtype=torch.int32, device=parameters[0].device)
    dist.all_reduce(used, op=dist.ReduceOp.MAX)
    used_flags = used.tolist()
    selected, views = [], []
    for p, flag in zip(parameters, used_flags):
        if not flag:
            continue
        if p.grad is None:
            p.grad = torch.zeros_like(p)
        view = torch.view_as_real(p.grad) if p.grad.is_complex() else p.grad
        selected.append(view)
        views.append(view.reshape(-1))
    flat = torch.cat(views)
    dist.all_reduce(flat, op=dist.ReduceOp.SUM)
    offset = 0
    for view in selected:
        size = view.numel()
        view.copy_(flat[offset:offset + size].view_as(view))
        offset += size


class Width32Trainer(MultiConfigSRMTrainer):
    def _run_window(self, window):
        self.optimizer.zero_grad(set_to_none=True)
        counts = torch.tensor([
            sum(int(b["valid_mask"].sum()) for b in window),
            sum(int(b["target"].shape[0]) for b in window),
        ], dtype=torch.int64, device=self.device)
        dist.all_reduce(counts)
        total_valid, samples = map(int, counts.tolist())
        if total_valid <= 0 or not 0 < samples <= self.cfg.train_batch_size:
            raise RuntimeError(f"Invalid global accumulation window: {total_valid}, {samples}")
        numerator = 0.0
        for batch in window:
            predicted, target, mask = self._prediction_triplet(self.model, batch)
            n = int(mask.sum())
            loss = masked_velocity_mse(predicted, target, mask)
            self.scaler.scale(loss * (n / total_valid)).backward()
            numerator += float(loss.detach()) * n
        sum_real_view_gradients(self.model)
        ran = _scaled_optimizer_step(model=self.model, optimizer=self.optimizer,
                                     scaler=self.scaler, scheduler=self.scheduler, ema=self.ema)
        error = torch.tensor(numerator, dtype=torch.float64, device=self.device)
        dist.all_reduce(error)
        self.optimizer.zero_grad(set_to_none=True)
        batches = len(window) * dist.get_world_size()
        self.micro_batches_seen += batches
        self.samples_seen += samples
        self.optimizer_step += int(ran)
        return AccumulationWindowResult(ran, batches, samples, total_valid, float(error))

    def train_one_epoch(self):
        self.train_loader.sampler.set_epoch(self.next_epoch_index)
        result = super().train_one_epoch()
        if result["samples"] != 560:
            raise RuntimeError("Global epoch must contain 560 unique training scenes")
        return result

    def validate(self):
        result = [super().validate() if dist.get_rank() == 0 else None]
        dist.broadcast_object_list(result, src=0)
        return result[0]

    def _save(self, name, *, smoke=False):
        # Each rank keeps its own RNG/loader state; all model/optimizer states agree.
        path = super()._save(name, smoke=smoke)
        dist.barrier()
        if dist.get_rank() == 0 and name == "last.pt":
            _atomic_json(self.cfg.run_dir / "committed_epoch.json",
                         {"completed_epochs": self.completed_epochs,
                          "world_size": dist.get_world_size()}, replace_existing=True)
            print(json.dumps({"event": "epoch_complete", **self.history[-1]}, allow_nan=False), flush=True)
        return path

    def _runtime_payload(self, status):
        return {**super()._runtime_payload(status), "rank": dist.get_rank(),
                "world_size": dist.get_world_size(),
                "effective_batch_size": self.cfg.train_batch_size,
                "eval_batch_size": self.cfg.eval_batch_size}


def source_digest(root: Path):
    digest = hashlib.sha256()
    paths = [p for p in root.rglob("*.py") if not any(
        part in {"work", "figures", "runs", "tests", "__pycache__", ".venv"}
        for part in p.relative_to(root).parts)]
    for path in sorted(paths):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()
