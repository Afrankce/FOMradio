"""Fixed MS source, CASA, FM+0.1 Haar; only spectral gate varies."""
from dataclasses import dataclass
import torch
import torch.distributed as dist
from training.no_ema_experiment import NoEMAConfig, NoEMATrainer
from training.detail_losses import detail_components, auxiliary, parameter_group as base_group, gradient_statistics
from training.width32_distributed import sum_real_view_gradients
from training.optimization import AccumulationWindowResult, _scaled_optimizer_step
from training.multiconfig_trainer import _atomic_json

@dataclass(frozen=True)
class SpectralConfig(NoEMAConfig):
    condition_variant: str = 'B'

    @property
    def model_size(self):
        return 'spectral_gate_uno_width32_'+self.condition_variant.lower()

    def scientific_payload(self):
        p=super().scientific_payload()
        for key in ('rank','modulation_bound','weight_selection'):
            p.pop(key,None)
        return {**p,'experiment':'conditional_spectral_gate_v1','spectral_variant':self.condition_variant,
                'condition_fusion':'unchanged CASA; new gate only in spectral branch',
                'operator_conditioning':'bounded radial / amplitude / complex conditional gate',
                'parameter_budget':'shared backbone weights; added parameters reported, not equal capacity',
                'gate_hidden_width':32,'gate_eta':.5,'gate_beta':0.7853981633974483,
                'radial_alpha':.7,'radial_gamma':'.5*tanh(raw), raw initially 0',
                'gate_time_input':'existing learned 512-d time embedding projected to 4 channels',
                'gate_boundaries':'kx=0 and x-Nyquist left unchanged; legacy baseline projection preserved',
                'gate_epsilon':1e-6,'gate_init_seed':271828,'gate_shared_between_stages':False,
                'gate_stages':'four encoder, bottleneck, four decoder','gate_initialization':'identity',
                'amplitude_descriptors':'log1p magnitudes normalized by per-channel retained-band RMS',
                'phase_descriptors':'AP only: real/imag normalized relative cross-spectrum; not RF phase',
                'primary_endpoint':'epoch1000 online, Euler N8; no test checkpoint selection'}

def parameter_group(name):
    if '.spectral_gate.' in name:
        if '.amplitude_head.' in name:return 'gate_amplitude_head'
        if '.phase_head.' in name:return 'gate_phase_head'
        if '.align.' in name:return 'gate_condition_alignment'
        if '.radial_raw' in name:return 'gate_radial'
        return 'gate_descriptor_network'
    return base_group(name)

class SpectralTrainer(NoEMATrainer):
    def _run_window(self, window):
        if len(window) != 1:
            raise RuntimeError('This registered pilot requires accumulation_steps=1')
        for _, gate in self.model.gates():
            gate.capture = self.optimizer_step % self.cfg.parameter_gradient_interval == 0
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
            gate_diagnostics={name: gate.diagnostic for name,gate in self.model.gates() if gate.capture},
            cuda_peak_allocated_mib=torch.cuda.max_memory_allocated(self.device)/1024**2,
            support_counts={k:int(counts[i]) for i,k in enumerate(keys)},
            output_gradients=output_stats, parameter_gradients=parameter_stats,
            optimizer_ran=ran, amp_scale_before=scale, amp_scale_after=self.scaler.get_scale())
        if dist.get_rank() == 0:
            directory = self.cfg.run_dir/'loss_diagnostics'
            directory.mkdir(exist_ok=True)
            _atomic_json(directory/f'window_{self.micro_batches_seen:07d}.json',record,replace_existing=True)
        for _,gate in self.model.gates():gate.capture=False
        self.optimizer.zero_grad(set_to_none=True)
        batches = dist.get_world_size()
        self.micro_batches_seen += batches
        self.samples_seen += int(counts[-1])
        self.optimizer_step += int(ran)
        # Preserve legacy train_loss as FM MSE, not a misleading mixed-unit total.
        return AccumulationWindowResult(ran,batches,int(counts[-1]),int(counts[0]),float(sums[0]))


    def milestone(self):
        # Extra evaluation must not change subsequent training RNG.
        with torch.random.fork_rng(devices=[self.device.index]):
            for _,g in self.model.gates():g.capture=False
            super().milestone()
            self.gate_probe(self.cfg.run_dir/'milestones'/f'epoch_{self.completed_epochs:04d}')

    @torch.inference_mode()
    def gate_probe(self,directory):
        import numpy as np
        from model.matern_source import path_triplet
        directory.mkdir(parents=True,exist_ok=True)
        # Actual sampled trajectory at fixed times; no target used for gate input.
        with torch.random.fork_rng(devices=[self.device.index]):
            batch=next(iter(self.val_loader))
            condition=batch['condition'][:1].to(self.device)
            metadata=batch['metadata'][:1]
            x=self.source_sampler.sample(metadata,device=self.device,namespace='eval:0')
            network=self.model
            training=network.training
            network.eval()
            embedding=network.embed_model(condition)
            rows=[]
            for n in range(8):
                t=torch.full((1,),n/8,device=self.device)
                capture=n in (0,2,4,6,7)
                for _,gate in network.gates():gate.capture=capture
                with torch.autocast(device_type='cuda',dtype=torch.float16,enabled=self.cfg.use_amp):
                    v=network.forward_with_cfg(image=condition,x=x,step=t,embedding=embedding,cfg_scale=1.)
                if capture:
                    for name,gate in network.gates():
                        rows.append(dict(stage=name,time=n/8,scene=metadata[0]['scene_id'],**gate.diagnostic))
                        file=directory/f'gate_{name}_t{n}.npz'
                        np.savez_compressed(file,**gate.arrays)
                x=x+v/8
            for _,gate in network.gates():gate.capture=False
            network.train(training)
            _atomic_json(directory/'gate_probe.json',dict(rows=rows,trajectory='Euler N8 actual inference; fixed source eval:0',
                scene=metadata[0]['scene_id'],epoch=self.completed_epochs),replace_existing=True)
