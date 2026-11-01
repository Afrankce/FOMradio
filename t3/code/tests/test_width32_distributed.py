import copy
import math
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DistributedSampler

from model.attention_multiscale_uno import AttentionMultiscaleUNO2d
from model.fno import count_tensor_parameters, count_real_scalar_parameters
from training.same_frequency_config import SameFrequencyTrainConfig
from training.same_frequency_multiscale_uno_config import MultiscaleUNOTrainConfig
from training.width32_distributed import Width32Config, sum_real_view_gradients


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([1+2j, 3-1j], dtype=torch.cdouble))
        self.bias = torch.nn.Parameter(torch.tensor(0.5, dtype=torch.double))

    def forward(self, x):
        return (x * self.weight).real.sum(-1) + self.bias


def worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2)
    try:
        model = Toy()
        reference = copy.deepcopy(model)
        x = torch.tensor([[1+1j, 2-1j], [2+3j, 1-4j], [3+2j, 4+1j]], dtype=torch.cdouble)
        target = torch.tensor([1., 3., 2.], dtype=torch.double)
        local = slice(0, 1) if rank == 0 else slice(1, 3)
        # Deliberately unequal valid counts: one versus two.
        ((model(x[local]) - target[local]).square().sum() / 3).backward()
        sum_real_view_gradients(model)
        (reference(x) - target).square().mean().backward()
        for p, q in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(p.grad, q.grad, rtol=1e-12, atol=1e-12)
        a = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
        b = torch.optim.AdamW(reference.parameters(), lr=1e-3, weight_decay=1e-5)
        a.step()
        b.step()
        for p, q in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(p, q, rtol=1e-12, atol=1e-12)
    finally:
        dist.destroy_process_group()


def test_complex_global_valid_weighted_gradient(tmp_path):
    mp.spawn(worker, args=(str(tmp_path / "rendezvous"),), nprocs=2, join=True)


@pytest.mark.parametrize("world", [1, 2, 4, 8])
@pytest.mark.parametrize("batch,eval_batch", [(128,256), (64,128)])
def test_batch_and_scene_coverage(world, batch, eval_batch, tmp_path):
    base = SameFrequencyTrainConfig(tmp_path, tmp_path, tmp_path, tmp_path)
    cfg = Width32Config(base, world, 1, 1, "a" * 64, batch, eval_batch)
    assert cfg.micro_batch_size * cfg.accumulation_steps * world == batch
    partitions = [list(DistributedSampler(range(560), num_replicas=world, rank=rank, seed=42))
                  for rank in range(world)]
    assert sorted(x for group in partitions for x in group) == list(range(560))
    assert all(math.ceil(len(group) / cfg.micro_batch_size) == cfg.optimizer_steps_per_epoch for group in partitions)
    assert cfg.planned_optimizer_steps == math.ceil(560 / batch) * 1000
    assert cfg.warmup_steps == math.ceil(560 / batch) * 100
    assert MultiscaleUNOTrainConfig(base).operator_width == 24
    assert cfg.config_sha256 != MultiscaleUNOTrainConfig(base).config_sha256


def test_width32_architecture_only_change():
    torch.set_num_threads(1)
    baseline = AttentionMultiscaleUNO2d()
    widened = AttentionMultiscaleUNO2d(operator_width=32)
    assert baseline.operator_width == 24
    assert widened.operator_width == 32
    assert baseline.operator_modes == widened.operator_modes
    assert baseline.operator_padding == widened.operator_padding
    assert count_tensor_parameters(baseline) == 3059355
    assert count_real_scalar_parameters(baseline) == 3925659
    assert count_real_scalar_parameters(widened) > count_real_scalar_parameters(baseline)
