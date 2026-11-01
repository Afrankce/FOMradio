"""Source-only baseline plus one auxiliary objective and non-invasive diagnostics."""
from dataclasses import dataclass
import torch
import torch.distributed as dist
from training.matern_source import SourceConfig, SourceTrainer
from training.detail_losses import detail_components, auxiliary, parameter_group as base_parameter_group, gradient_statistics
from training.width32_distributed import sum_real_view_gradients
from training.optimization import AccumulationWindowResult, _scaled_optimizer_step
from training.multiconfig_trainer import _atomic_json


@dataclass(frozen=True)
class ConditionalConfig(SourceConfig):
    condition_variant: str = 'CASA'
    variant: str = 'L2'
    auxiliary_weight: float = .1
    parameter_gradient_interval: int = 25

    @property
    def model_size(self):
        return 'conditional_operator_uno_width32_' + self.condition_variant.lower()

    def scientific_payload(self):
        return {**super().scientific_payload(), 'experiment': 'conditional_operator_pilot_v1',
                'condition_variant': self.condition_variant,
                'operator_conditioning': 'rank2 shared spectral kernel; bounded input/output modulation',
                'rank': 2, 'modulation_bound': .5,
                'parameter_budget': 'FILM and NSOP exactly equal; common layers identically initialized',
                'loss_variant': self.variant, 'auxiliary_weight': self.auxiliary_weight,
                'loss_domain': 'normalized velocity; shared 2x2 mean before auxiliary filters',
                'auxiliary_definition': 'valid-neighbor squared differences' if self.variant == 'L1'
                    else 'equal mean of first two orthonormal Haar detail-band MSEs',
                'loss_normalization': 'each term separately normalized by global support count',
                'parameter_gradient_interval': self.parameter_gradient_interval,
                'gradient_diagnostics': 'global unscaled real-coordinate raw gradients, before optimizer',
                'weight_selection': 'fixed 0.1 untuned pilot; no test selection'}


def parameter_group(name):
    return 'conditional_operator' if '.condition_operator.' in name else base_parameter_group(name)


class ConditionalTrainer(SourceTrainer):
    def _run_window(self, window):
        if len(window) != 1:
            raise RuntimeError('This registered pilot requires accumulation_steps=1')
        self.optimizer.zero_grad(set_to_none=True)
        predicted, target, mask = self._prediction_triplet(self.model, window[0])
        terms = detail_components(predicted, target, mask)
        keys = list(terms)
        counts = torch.stack([terms[k][1] for k in keys] + [torch.tensor(predicted.shape[0], device=self.device)])
        dist.all_reduce(counts)
        if (counts <= 0).any():
            raise RuntimeError('No global valid support for a registered loss component')
        means = {k: terms[k][0] / counts[i] for i, k in enumerate(keys)}
        fm, aux = means['fm'], auxiliary(means, self.cfg.variant)
        weight = self.cfg.auxiliary_weight
        scale = self.scaler.get_scale()
        # Output-space gradients every step. Aggregate sufficient statistics,
        # not per-rank cosines, so uneven last batches remain correctly weighted.
        og = [torch.autograd.grad(self.scaler.scale(term), predicted, retain_graph=True)[0].float()/scale
              for term in (fm, aux)]
        stats = torch.stack([og[0].double().square().sum(), og[1].double().square().sum(),
                             (og[0].double()*og[1].double()).sum()])
        dist.all_reduce(stats)
        aa, bb, ab = map(float, stats.tolist())
        output_stats = dict(fm_norm=aa**.5, aux_norm=bb**.5, weighted_aux_norm=weight*bb**.5,
            weighted_ratio=weight*(bb/aa)**.5 if aa else None,
            cosine=ab/(aa*bb)**.5 if aa*bb else None)
        del og
        parameter_stats = None
        if self.optimizer_step % self.cfg.parameter_gradient_interval == 0:
            named = [(n,p) for n,p in self.model.named_parameters() if p.requires_grad]
            flat = []
            for term in (fm, aux):
                gradients = torch.autograd.grad(self.scaler.scale(term), [p for _,p in named],
                                               retain_graph=True, allow_unused=True)
                pieces = []
                for (_,p), g in zip(named, gradients):
                    g = torch.zeros_like(p) if g is None else g.detach()
                    pieces.append((torch.view_as_real(g) if g.is_complex() else g).reshape(-1).float()/scale)
                vector = torch.cat(pieces)
                dist.all_reduce(vector)
                if not torch.isfinite(vector).all():
                    raise RuntimeError('Nonfinite parameter diagnostic gradient')
                flat.append(vector)
            parameter_stats = {'all': gradient_statistics(*flat, weight)}
            offsets, offset = {}, 0
            for name,p in named:
                size = p.numel()*(2 if p.is_complex() else 1)
                offsets.setdefault(parameter_group(name), []).append((offset, offset+size))
                offset += size
            for group, ranges in offsets.items():
                parts = [torch.cat([v[a:b] for a,b in ranges]) for v in flat]
                parameter_stats[group] = gradient_statistics(*parts, weight)
            del flat
        total = fm + weight*aux
        self.scaler.scale(total).backward()
        sum_real_view_gradients(self.model)
        ran = _scaled_optimizer_step(model=self.model, optimizer=self.optimizer, scaler=self.scaler,
                                     scheduler=self.scheduler, ema=self.ema)
        sums = torch.stack([terms[k][0].detach().double() for k in keys])
        dist.all_reduce(sums)
        global_means = {k: float(sums[i]/counts[i]) for i,k in enumerate(keys)}
        record = dict(epoch_index=self.next_epoch_index, optimizer_step_before=self.optimizer_step,
            variant=self.cfg.condition_variant, loss_variant=self.cfg.variant, weight=weight, losses=global_means,
            auxiliary=float(auxiliary(global_means,self.cfg.variant)),
            total=global_means['fm']+weight*auxiliary(global_means,self.cfg.variant),
            support_counts={k:int(counts[i]) for i,k in enumerate(keys)},
            output_gradients=output_stats, parameter_gradients=parameter_stats,
            optimizer_ran=ran, amp_scale_before=scale, amp_scale_after=self.scaler.get_scale())
        if dist.get_rank() == 0:
            directory = self.cfg.run_dir/'loss_diagnostics'
            directory.mkdir(exist_ok=True)
            _atomic_json(directory/f'window_{self.micro_batches_seen:07d}.json',record,replace_existing=True)
        self.optimizer.zero_grad(set_to_none=True)
        batches = dist.get_world_size()
        self.micro_batches_seen += batches
        self.samples_seen += int(counts[-1])
        self.optimizer_step += int(ran)
        # Preserve legacy train_loss as FM MSE, not a misleading mixed-unit total.
        return AccumulationWindowResult(ran,batches,int(counts[-1]),int(counts[0]),float(sums[0]))

    def _save(self, name, *, smoke=False):
        path = super()._save(name, smoke=smoke)
        if name == 'last.pt' and not smoke and self.completed_epochs in (1,10,50):
            error = [None]
            if dist.get_rank() == 0:
                try:
                    from torch.utils.data import Subset
                    from training.matern_source import evaluate_dataset
                    # DataLoader creates worker seeds even for deterministic
                    # inference. Extra previews must not advance training RNG.
                    with torch.random.fork_rng(devices=[self.device.index]):
                        evaluate_dataset(self.ema.ema_model,self.source_sampler,
                            Subset(self.val_loader.dataset,list(range(4))),self.device,
                            self.cfg.run_dir/'previews'/f'epoch_{self.completed_epochs:04d}.json',
                            steps=(2,8),realizations=1,epoch=self.completed_epochs)
                except Exception as caught:
                    error[0] = repr(caught)
            dist.broadcast_object_list(error,src=0)
            if error[0]: raise RuntimeError(error[0])
        return path
