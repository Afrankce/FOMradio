"""Same-budget matched-seed ablations; auxiliary RNG is checkpoint-independent."""
from dataclasses import dataclass

import torch
import torch.distributed as dist

from evaluation.radioflow_sampling import euler_cfg_sample, make_sample_noise
from evaluation.radiomap_metrics import MetricAccumulator
from model.matern_multiscale_uno import MaternBankSampler
from training.width32_distributed import Width32Config, Width32Trainer


@dataclass(frozen=True)
class MaternConfig(Width32Config):
    variant: str = "B2"

    @property
    def model_size(self):
        return f"attention_multiscale_uno_width32_matern_{self.variant}"

    def scientific_payload(self):
        return {**super().scientific_payload(), "experiment": "matern_encoder_pilot_v1", "variant": self.variant,
                "external_source": "iid_normal_unchanged", "auxiliary_innovations": "trajectory_frozen",
                "matern_nu": 1., "matern_length_convention": "r/lambda_no_sqrt_2nu",
                "stage_base_pixel_lengths": [2, 4, 8], "initial_injection_gain": .05,
                "fixed_mix": self.variant != "M3", "feedback": "previous_encoder_output_condition_time" if self.variant == "M3" else "none",
                "stage_length_schedule": "not_applicable" if self.variant == "B2" else (
                    "constant_domain_fraction" if self.variant == "M1" else "doubles_per_downsample"),
                "innovation_sampler": "white_crop" if self.variant == "B2" else "PSD_checked_circulant_embedding_crop",
                "pilot_training_seeds": [42]}


def predict_frozen(network, sampler, batch, device, *, steps=2, realization=0, use_amp=True):
    condition = batch["condition"].to(device, non_blocking=True)
    noise = torch.stack([make_sample_noise(str(m["scene_id"]), float(m["steering_deg"]),
                        shape=tuple(batch["target"].shape[1:]), base_seed=42 + realization)
                        for m in batch["metadata"]]).to(device)
    bank = sampler.sample(batch["metadata"], device=device, namespace=f"eval:{realization}")
    with network.use_innovations(bank):
        return euler_cfg_sample(network, condition, noise, cfg_scale=1., steps=steps, use_amp=use_amp)


class MaternTrainer(Width32Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bank_sampler = MaternBankSampler(self.cfg.variant)

    def _prediction_triplet(self, model, batch):
        bank = self.bank_sampler.sample(batch["metadata"], device=self.device, namespace=f"train:{self.next_epoch_index}")
        with model.use_innovations(bank):
            return super()._prediction_triplet(model, batch)

    def validate(self):
        result = [None]
        if dist.get_rank() == 0:
            network = self.ema.ema_model
            network.eval()
            accumulator = MetricAccumulator()
            # Match the registered eval batch cap; actual val set contains 80.
            for batch in self.val_loader:
                prediction = predict_frozen(network, self.bank_sampler, batch, self.device, use_amp=self.cfg.use_amp)
                accumulator.update(prediction, batch["target"].to(self.device), batch["valid_mask"].to(self.device))
            result[0] = accumulator.compute()
        dist.broadcast_object_list(result, src=0)
        return result[0]
