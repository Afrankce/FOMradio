"""Successful-update FM training; transactional, deterministic restart."""
import argparse
from datetime import timedelta
import os
from pathlib import Path
import signal
import time
import numpy as np
import torch
import torch.distributed as dist
from model.matern_source import SourceSampler, path_triplet
from training.complex_grad_scaler import ComplexGradScaler
from training.detail_losses import detail_components
from training.optimization import OptimizerStepScheduler
from .common import atomic, read, sha, state_sha, tensor_sha, seed_for, load_data, batch, checkpoint, identity, verify
from .model import BoundaryUNO
from cross_array.paths import KINDS, initial_state

STOP = False


def signal_stop(*_):
    global STOP
    STOP = True


def indices_for(records, step, size=128):
    rng = np.random.default_rng(seed_for(42, 'batch', step)); indices = []
    for i, angle in enumerate((-14., 0., 14.)):
        pool = [j for j, r in enumerate(records) if r['steering_deg'] == angle]
        n = size // 3 + int((i - step) % 3 < size % 3)
        indices.extend(rng.choice(pool, n, replace=False).tolist())
    rng.shuffle(indices)
    return indices


def save(out, model, optimizer, scaler, scheduler, step, history, ident):
    rank = dist.get_rank(); hashes = [None] * dist.get_world_size()
    dist.all_gather_object(hashes, state_sha(model.state_dict()))
    if len(set(hashes)) != 1: raise RuntimeError('Model diverged across ranks')
    if rank == 0:
        old = read(out / 'committed.json') if (out / 'committed.json').exists() else {'slot': 1}
        slot = 1 - old['slot']; dest = out / f'checkpoint_{slot}.pt'; temp = dest.with_suffix('.pt.tmp')
        torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), scaler=scaler.state_dict(),
            scheduler=scheduler.state_dict(), step=step, history=history, identity=ident), temp)
        with temp.open('rb') as stream: os.fsync(stream.fileno())
        os.replace(temp, dest)
        atomic(out / 'committed.json', dict(step=step, slot=slot, checkpoint=dest.name, sha256=sha(dest),
                model_sha256=hashes[0], rank_checksums_equal=True, updated=time.time()))
        atomic(out / 'history.json', history)
    dist.barrier()


def sync_gradients(model):
    values = []
    for p in model.parameters():
        if p.grad is None: p.grad = torch.zeros_like(p)
        values.append(torch.view_as_real(p.grad) if p.grad.is_complex() else p.grad)
    flat = torch.cat([v.reshape(-1).double() for v in values])
    dist.all_reduce(flat, op=dist.ReduceOp.SUM)
    offset = 0
    for value in values:
        value.copy_(flat[offset:offset + value.numel()].view_as(value)); offset += value.numel()


def main():
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True); p.add_argument('--arm', required=True)
    p.add_argument('--base', default='E2'); p.add_argument('--stop', type=int, required=True); p.add_argument('--out', type=Path)
    args = p.parse_args(); root = args.root.resolve(); cfg = read(root / 'config.json'); verify(root)
    rank, world, local = map(lambda k: int(os.environ[k]), ('RANK', 'WORLD_SIZE', 'LOCAL_RANK'))
    if world != cfg['world_size']: raise RuntimeError('World size is frozen')
    torch.cuda.set_device(local); device = torch.device('cuda', local); torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True); dist.init_process_group('nccl', timeout=timedelta(minutes=20))
    signal.signal(signal.SIGTERM, signal_stop); signal.signal(signal.SIGINT, signal_stop)
    torch.manual_seed(cfg['seed'])
    if cfg.get('experiment_kind') in KINDS:
        from cross_array.model import SourceObservationUNO
        model = SourceObservationUNO(args.arm).to(device)
    else:
        model = BoundaryUNO(args.arm, args.base, cfg.get('phase_input_only', False)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'], foreach=False)
    scaler = ComplexGradScaler('cuda'); scheduler = OptimizerStepScheduler(optimizer, total_steps=cfg['total_steps'], warmup_steps=cfg['warmup_steps'])
    out = args.out.resolve() if args.out else root / 'jobs' / args.arm; out.mkdir(parents=True, exist_ok=True)
    ident = identity(root, args.arm, args.base); history = []; step = 0
    if (out / 'committed.json').exists():
        saved, commit = checkpoint(out)
        if saved['identity'] != ident: raise RuntimeError('Resume identity mismatch')
        model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
        scaler.load_state_dict(saved['scaler']); scheduler.load_state_dict(saved['scheduler'])
        step, history = saved['step'], saved['history']
        verified = state_sha(model.state_dict()) == commit['model_sha256']
        verified &= scheduler.state_dict() == saved['scheduler'] and scaler.state_dict() == saved['scaler']
        for key, state in optimizer.state_dict()['state'].items():
            for name, value in state.items():
                if isinstance(value, torch.Tensor): verified &= torch.equal(value.cpu(), saved['optimizer']['state'][key][name].cpu())
        if not verified: raise RuntimeError('Checkpoint restoration failed')
        if rank == 0: atomic(out / f'resume_verification_step_{step:06d}.json', dict(passed=True, updated=time.time()))
        del saved
    arrays, records = load_data(root, 'train'); source = SourceSampler('MS', resolution=256, seed=42)
    if rank == 0:
        specification = source.specification()
        if cfg.get('experiment_kind') == 'cross_array_transport':
            specification = dict(base_noise=specification, mean_rule=cfg['source_mean_rule'],
                noise_scale=cfg['source_noise_scale'], coordinate='normalized dB', conditional=True)
        atomic(out / 'source.json', specification)
    local_batch = cfg['global_batch'] // world; micro = cfg['microbatch']; began = time.time(); overflow = 0
    while step < args.stop:
        tick = time.time(); all_indices = indices_for(records, step, cfg['global_batch'])
        indices = all_indices[rank * local_batch:(rank + 1) * local_batch]
        condition, target, mask = batch(arrays, indices, device)
        metadata = [dict(records[i], scene_id=f"{records[i]['scene_id']}|slot{rank * local_batch + j}") for j, i in enumerate(indices)]
        noise = source.sample(metadata, device=device, namespace=f'train-update-{step}')
        z = initial_state(condition, noise, cfg)
        times = torch.rand(local_batch, generator=torch.Generator(device=device).manual_seed(seed_for(42, 'time', step, rank)), device=device)
        counts = torch.stack([detail_components(torch.zeros_like(target), torch.zeros_like(target), mask)[k][1]
                              for k in ('fm', 'haar1', 'haar2')]).double()
        dist.all_reduce(counts); optimizer.zero_grad(set_to_none=True); totals = torch.zeros(3, device=device, dtype=torch.float64)
        if (counts <= 0).any(): raise RuntimeError('Empty global loss support')
        for start in range(0, local_batch, micro):
            sl = slice(start, start + micro); xt, velocity = path_triplet(z[sl], target[sl], times[sl])
            with torch.autocast('cuda', dtype=torch.float16):
                predicted = model(image=condition[sl], x=xt, step=times[sl])
            terms = detail_components(predicted, velocity, mask[sl])
            numerator = torch.stack([terms[k][0] for k in ('fm', 'haar1', 'haar2')])
            weights = numerator.new_tensor([1., .05, .05])
            loss = (numerator / counts.to(numerator.dtype) * weights).sum()
            scaler.scale(loss).backward(); totals += numerator.detach().double()
        sync_gradients(model); dist.all_reduce(totals)
        old_scale = scaler.get_scale(); scaler.step(optimizer); scaler.update()
        if scaler.get_scale() < old_scale:
            overflow += 1
            if rank == 0: atomic(out / 'progress.json', dict(arm=args.arm, phase='scale_backoff', step=step, scale=scaler.get_scale(), updated=time.time()))
            if overflow > 30: raise RuntimeError('Repeated FP16 overflow')
            continue
        overflow = 0; scheduler.step(); step += 1
        if rank == 0:
            means = (totals / counts).tolist()
            row = dict(step=step, fm=means[0], haar1=means[1], haar2=means[2], loss=means[0] + .05 * (means[1] + means[2]),
                seconds=time.time() - tick, elapsed_segment_s=time.time() - began, learning_rate=optimizer.param_groups[0]['lr'],
                scale=scaler.get_scale(), peak_gpu_gib=torch.cuda.max_memory_allocated() / 1024**3, updated=time.time())
            if step <= 8:
                trace = dict(indices=all_indices, z_sha256=tensor_sha(z), t_sha256=tensor_sha(times))
                if cfg.get('experiment_kind') == 'cross_array_transport': trace['noise_sha256'] = tensor_sha(noise)
                atomic(out / 'trace' / f'step_{step:06d}.json', trace)
            if step == 1 or step % cfg['log_every'] == 0 or step == args.stop:
                history.append(row); atomic(out / 'history.json', history)
                print(__import__('json').dumps(dict(arm=args.arm, **row)), flush=True)
            atomic(out / 'progress.json', dict(arm=args.arm, phase='training', **row))
        stopping = torch.tensor(int(STOP), device=device); dist.all_reduce(stopping, op=dist.ReduceOp.MAX)
        if step % cfg['checkpoint_every'] == 0 or step == args.stop or stopping.item():
            save(out, model, optimizer, scaler, scheduler, step, history, ident)
        if stopping.item(): break
    dist.destroy_process_group()


if __name__ == '__main__': main()
