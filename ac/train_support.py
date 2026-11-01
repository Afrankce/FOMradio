"""A pilot: no-radio-observation fixed Matérn + beam-conditioned AP-UNO.

The pilot intentionally refuses to run past update 200 until the planned
validation/checkpoint-selection protocol has been implemented and tested.
"""
import argparse
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import signal
import time
import numpy as np
import torch
import torch.distributed as dist
from dataset import QueryLoader,training_targets
from model.matern_source import SourceSampler,path_triplet
from cross_array.model import SourceObservationUNO
from training.complex_grad_scaler import ComplexGradScaler
from training.detail_losses import detail_components
from training.optimization import OptimizerStepScheduler

STOP=False
def request_stop(*_):
    global STOP
    STOP=True

def seed_for(*values):
    return int.from_bytes(hashlib.sha256('|'.join(map(str,values)).encode()).digest()[:8],'big')%(2**63-1)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024**2),b''): h.update(block)
    return h.hexdigest()

def atomic(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    with tmp.open('w') as f:
        json.dump(obj,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)

def state_sha(model):
    h=hashlib.sha256()
    for k,v in sorted(model.state_dict().items()):
        h.update(k.encode()); h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def indices_for(records,seed,step,size=128):
    rng=np.random.default_rng(seed_for(seed,'batch',step));out=[]
    for i,angle in enumerate((-14,0,14)):
        pool=[j for j,r in enumerate(records) if r['query']['angle']==angle]
        n=size//3+int((i-step)%3<size%3)
        out.extend(rng.choice(pool,n,replace=False).tolist())
    rng.shuffle(out);return out

def sync_gradients(model):
    values=[]
    for p in model.parameters():
        if p.grad is None: p.grad=torch.zeros_like(p)
        values.append(torch.view_as_real(p.grad) if p.grad.is_complex() else p.grad)
    # Gloo communicates CPU tensors; model evaluation/backprop remain on GPU.
    # This avoids an NCCL/driver mismatch without modifying the server setup.
    flat=torch.cat([v.reshape(-1).double() for v in values]).cpu()
    dist.all_reduce(flat,op=dist.ReduceOp.SUM)
    offset=0
    for value in values:
        value.copy_(flat[offset:offset+value.numel()].view_as(value).to(value.device));offset+=value.numel()

def cpu_all_reduce(value):
    host=value.detach().cpu()
    dist.all_reduce(host,op=dist.ReduceOp.SUM)
    return host.to(value.device)

def sample_paired(samplers,meta,device,namespace):
    z=samplers['MS'].sample(meta,device=device,namespace=namespace+'-base')
    uniform=0
    for label in ('S2','S8','S32'):
        uniform=uniform+samplers[label].sample(meta,device=device,namespace=namespace+'-'+label)/np.sqrt(3.)
    return np.sqrt(.75)*z+.5*uniform

def save(out,model,optim,scaler,scheduler,step,identity):
    rank=dist.get_rank(); hashes=[None]*dist.get_world_size()
    dist.all_gather_object(hashes,state_sha(model))
    if len(set(hashes))!=1: raise RuntimeError('model ranks diverged')
    if rank==0:
        committed=out/'committed.json'
        prev=json.loads(committed.read_text()) if committed.exists() else {'slot':1}
        slot=1-prev['slot'];path=out/f'checkpoint_{slot}.pt';temp=path.with_suffix('.pt.tmp')
        torch.save(dict(model=model.state_dict(),optim=optim.state_dict(),scaler=scaler.state_dict(),
                        scheduler=scheduler.state_dict(),step=step,identity=identity),temp)
        with temp.open('rb') as f: os.fsync(f.fileno())
        os.replace(temp,path)
        atomic(committed,dict(step=step,slot=slot,path=path.name,sha256=sha(path),
                              model_sha256=hashes[0],timestamp=time.time()))
    dist.barrier()

def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--stop',type=int,required=True)
    p.add_argument('--micro',type=int,default=2);a=p.parse_args()
    if a.stop>200: raise ValueError('Pilot only: full-run validation is not yet implemented')
    rank,world,local=(int(os.environ[k]) for k in ('RANK','WORLD_SIZE','LOCAL_RANK'))
    if 128%world or (128//world)%a.micro: raise ValueError('incompatible global/micro batch')
    torch.cuda.set_device(local);device=torch.device('cuda',local);torch.set_num_threads(2)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)
    dist.init_process_group('gloo',timeout=timedelta(minutes=20))
    signal.signal(signal.SIGTERM,request_stop);signal.signal(signal.SIGINT,request_stop)
    a.out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads(a.manifest.read_text()); loader=QueryLoader(a.dataset)
    torch.manual_seed(a.seed);model=SourceObservationUNO('T0').to(device)
    optim=torch.optim.AdamW(model.parameters(),lr=1e-3,betas=(.9,.999),weight_decay=1e-5,foreach=False)
    scaler=ComplexGradScaler('cuda');scheduler=OptimizerStepScheduler(optim,total_steps=5000,warmup_steps=500)
    identity={'seed':a.seed,'manifest_sha256':sha(a.manifest),'arm':'A','global_batch':128,
              'world_size':world,'microbatch':a.micro,'backend':'gloo_cpu_reduce',
              'total_steps':5000,'plan':
              '340a447165bc3dc421efcfc78b693864c977b2b01fb301a939dce7700d25ab04',
              'train_source_sha256':sha(__file__),'dataset_source_sha256':sha(Path(__file__).with_name('dataset.py'))}
    start=0;committed=a.out/'committed.json'
    if committed.exists():
        record=json.loads(committed.read_text());path=a.out/record['path']
        if sha(path)!=record['sha256']:raise RuntimeError('checkpoint checksum mismatch')
        ck=torch.load(path,map_location='cpu',weights_only=False)
        if ck['identity']!=identity:raise RuntimeError('resume identity mismatch')
        model.load_state_dict(ck['model']);optim.load_state_dict(ck['optim'])
        scaler.load_state_dict(ck['scaler']);scheduler.load_state_dict(ck['scheduler']);start=ck['step']
        if state_sha(model)!=record['model_sha256']:raise RuntimeError('model resume mismatch')
        del ck
    samplers={s:SourceSampler(s,resolution=256,seed=a.seed) for s in ('MS','S2','S8','S32')}
    if rank==0:
        atomic(a.out/'identity.json',identity)
        atomic(a.out/'source.json',{'pairing':'independent MS and independent scales',
                                   'MS':samplers['MS'].specification()})
        print(json.dumps({'phase':'ready','step':start,'parameters':sum(x.numel() for x in model.parameters()),
                          'world_size':world,'microbatch':a.micro,'source_extent':samplers['MS'].extent}),flush=True)
    local_batch=128//world;step=start;overflow=0;starttime=time.time()
    while step<a.stop:
        tick=time.time();all_indices=indices_for(manifest,a.seed,step)
        idx=all_indices[rank*local_batch:(rank+1)*local_batch]
        records=[manifest[i] for i in idx];queries=[r['query'] for r in records]
        condition,_=loader.inputs(queries,device,arm='A');target,mask=training_targets(a.dataset,records,device)
        meta=[{'scene_id':q['scene']+'|slot'+str(rank*local_batch+j),'steering_deg':q['angle']} for j,q in enumerate(queries)]
        t=torch.rand(local_batch,generator=torch.Generator(device=device).manual_seed(seed_for(a.seed,'time',step,rank)),device=device)
        counts=torch.stack([detail_components(torch.zeros_like(target),torch.zeros_like(target),mask)[k][1]
                            for k in ('fm','haar1','haar2')]).double()
        counts=cpu_all_reduce(counts)
        if (counts<=0).any():raise RuntimeError('empty mask support')
        optim.zero_grad(set_to_none=True);totals=torch.zeros(3,device=device,dtype=torch.float64)
        for i in range(0,local_batch,a.micro):
            sl=slice(i,i+a.micro)
            z=sample_paired(samplers,meta[sl],device,'train-update-'+str(step))
            xt,velocity=path_triplet(z,target[sl],t[sl])
            with torch.autocast('cuda',dtype=torch.float16):
                pred=model(image=condition[sl],x=xt,step=t[sl])
            terms=detail_components(pred,velocity,mask[sl])
            numerator=torch.stack([terms[k][0] for k in ('fm','haar1','haar2')])
            loss=(numerator/counts.to(numerator.dtype)*numerator.new_tensor([1.,.05,.05])).sum()
            scaler.scale(loss).backward();totals+=numerator.detach().double()
        sync_gradients(model);totals=cpu_all_reduce(totals)
        old_scale=scaler.get_scale();scaler.step(optim);scaler.update()
        if scaler.get_scale()<old_scale:
            overflow+=1
            if rank==0:print(json.dumps({'phase':'scale_backoff','step':step,'scale':scaler.get_scale()}),flush=True)
            if overflow>30:raise RuntimeError('too many FP16 overflows')
            continue
        overflow=0;scheduler.step();step+=1
        if rank==0:
            means=(totals/counts).tolist()
            progress={'arm':'A','phase':'training','step':step,'fm':means[0],'haar1':means[1],
                      'haar2':means[2],'loss':means[0]+.05*(means[1]+means[2]),
                      'seconds':time.time()-tick,'elapsed_seconds':time.time()-starttime,
                      'lr_next':optim.param_groups[0]['lr'],'scale':scaler.get_scale(),
                      'peak_gpu_gib':torch.cuda.max_memory_allocated()/1024**3,'timestamp':time.time()}
            atomic(a.out/'progress.json',progress);print(json.dumps(progress),flush=True)
        stop=torch.tensor(int(STOP),device='cpu');dist.all_reduce(stop,op=dist.ReduceOp.MAX)
        if step%250==0 or step==a.stop or stop.item():save(a.out,model,optim,scaler,scheduler,step,identity)
        if stop.item():break
    if rank==0:atomic(a.out/'state.json',{'phase':'pilot_stopped','successful_updates':step,'timestamp':time.time()})
    dist.destroy_process_group()


if __name__=='__main__':main()
