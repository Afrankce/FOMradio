"""C full training, with separate environmental operator and beam-conditioned source.

Training after the pilot is segmented at verified 250-step validation boundaries.
"""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import signal
import time
import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F
from dataset import QueryLoader,training_targets
from cross_array.model import SourceObservationUNO
from model.matern_source import path_triplet
from training.detail_losses import detail_components
from training.complex_grad_scaler import ComplexGradScaler
from training.optimization import OptimizerStepScheduler
from beam_matern_source import BeamMaternSource
from env_operator import EnvironmentalPrediction
from train_support import seed_for,sha,atomic,state_sha,indices_for,sync_gradients,cpu_all_reduce,save

STOP=False
def request_stop(*_):
    global STOP
    STOP=True


class CSystem(nn.Module):
    def __init__(self,seed,gradient_scale):
        super().__init__()
        torch.manual_seed(seed)
        self.velocity=SourceObservationUNO('T0')
        # Preserve the common A/C velocity initialization exactly.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed_for(seed,'environment-initialization'))
            self.environment=EnvironmentalPrediction()
            self.noise=BeamMaternSource(seed,gradient_scale)

    def source(self,condition,beam,queries,meta,namespace,uniform=False):
        mu,basis,field,coeff=self.environment(condition[:,:3],queries)
        noise,q=self.noise(beam,meta,namespace,uniform=uniform)
        return mu+noise,mu,q,basis,field,coeff


def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--physics',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--stop',type=int,required=True);p.add_argument('--micro',type=int,default=2);a=p.parse_args()
    if a.stop>5000:raise ValueError('stop exceeds frozen 5000 updates')
    rank,world,local=(int(os.environ[k]) for k in ('RANK','WORLD_SIZE','LOCAL_RANK'))
    if 128%world or (128//world)%a.micro:raise ValueError('incompatible global/micro batch')
    torch.cuda.set_device(local);device=torch.device('cuda',local);torch.set_num_threads(2)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)
    dist.init_process_group('gloo',timeout=timedelta(minutes=20))
    signal.signal(signal.SIGTERM,request_stop);signal.signal(signal.SIGINT,request_stop)
    a.out.mkdir(parents=True,exist_ok=True)
    records=json.loads(a.manifest.read_text());physics=json.loads(a.physics.read_text());loader=QueryLoader(a.dataset)
    system=CSystem(a.seed,physics['training_beam_gradient_p95']).to(device)
    groups=[{'params':list(system.velocity.parameters())+list(system.environment.parameters()),'weight_decay':1e-5},
            {'params':[system.noise.a,system.noise.d],'weight_decay':0.}]
    optimizer=torch.optim.AdamW(groups,lr=1e-3,betas=(.9,.999),foreach=False)
    scaler=ComplexGradScaler('cuda');scheduler=OptimizerStepScheduler(optimizer,total_steps=5000,warmup_steps=500)
    local_files=['train_c_full.py','train_support.py','dataset.py','beam_matern_source.py','env_operator.py','array_physics.py']
    identity={'seed':a.seed,'manifest_sha256':sha(a.manifest),'physics_sha256':sha(a.physics),
              'source_sha256':{p:sha(Path(__file__).with_name(p)) for p in local_files},
              'arm':'C','global_batch':128,'world_size':world,'microbatch':a.micro,'backend':'gloo_cpu_reduce',
              'total_steps':5000,'plan':'340a447165bc3dc421efcfc78b693864c977b2b01fb301a939dce7700d25ab04'}
    start=0;committed=a.out/'committed.json'
    if committed.exists():
        rec=json.loads(committed.read_text());path=a.out/rec['path']
        if sha(path)!=rec['sha256']:raise RuntimeError('checkpoint checksum mismatch')
        saved=torch.load(path,map_location='cpu',weights_only=False)
        if saved['identity']!=identity:
            old=dict(identity);old['source_sha256']=dict(identity['source_sha256'])
            old['source_sha256'].pop('train_c_full.py')
            old['source_sha256']['train_c.py']=sha(Path(__file__).with_name('train_c.py'))
            if saved['identity']!=old or saved['step']!=200:raise RuntimeError('unrecognized pilot identity')
            if rank==0:atomic(a.out/'migration.json',{'from':'train_c.py','step':200,
                'checkpoint_sha256':rec['sha256'],'old_source_sha256':old['source_sha256']['train_c.py'],
                'new_source_sha256':identity['source_sha256']['train_c_full.py'],
                'reason':'enable scheduled validation'})
        system.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optim'])
        scaler.load_state_dict(saved['scaler']);scheduler.load_state_dict(saved['scheduler']);start=saved['step']
        if state_sha(system)!=rec['model_sha256']:raise RuntimeError('model resume mismatch')
        del saved
    if a.stop>200:
        if a.stop != (start//250+1)*250:raise RuntimeError('stop must be the next validation boundary')
        if start>=250:
            val_path=a.out/'validation'/f'step_{start:05d}.json'
            if not val_path.exists():raise RuntimeError('preceding validation missing')
            val=json.loads(val_path.read_text())
            if val['count']!=240 or val['checkpoint_sha256']!=rec['sha256']:raise RuntimeError('preceding validation mismatch')
    if rank==0:
        atomic(a.out/'identity.json',identity)
        print(json.dumps({'phase':'ready','step':start,
                          'velocity_parameters':sum(p.numel() for p in system.velocity.parameters()),
                          'environment_parameters':sum(p.numel() for p in system.environment.parameters()),
                          'source_parameters':2,'world_size':world,'microbatch':a.micro,
                          'source_extent':system.noise.extent}),flush=True)
    local_batch=128//world;step=start;overflows=0;started=time.time()
    while step<a.stop:
        tick=time.time();all_indices=indices_for(records,a.seed,step)
        subset=all_indices[rank*local_batch:(rank+1)*local_batch]
        batch=[records[i] for i in subset];queries=[r['query'] for r in batch]
        condition,beam=loader.inputs(queries,device,arm='C')
        target,mask=training_targets(a.dataset,batch,device)
        metadata=[{'scene_id':q['scene']+'|slot'+str(rank*local_batch+j),'steering_deg':q['angle']} for j,q in enumerate(queries)]
        times=torch.rand(local_batch,generator=torch.Generator(device=device).manual_seed(seed_for(a.seed,'time',step,rank)),device=device)
        counts=torch.stack([detail_components(torch.zeros_like(target),torch.zeros_like(target),mask)[k][1]
                            for k in ('fm','haar1','haar2')]).double()
        counts=cpu_all_reduce(counts)
        if (counts<=0).any():raise RuntimeError('empty support')
        optimizer.zero_grad(set_to_none=True);totals=torch.zeros(4,device=device,dtype=torch.float64)
        for i in range(0,local_batch,a.micro):
            sl=slice(i,i+a.micro)
            x0,mu,q,_,_,_=system.source(condition[sl],beam[sl],queries[sl],metadata[sl],
                                        'train-update-'+str(step))
            if not torch.isfinite(x0).all() or not torch.isfinite(mu).all():raise ValueError('non-finite C source')
            xt,velocity=path_triplet(x0,target[sl],times[sl])
            with torch.autocast('cuda',dtype=torch.float16):
                pred=system.velocity(image=condition[sl],x=xt,step=times[sl])
            terms=detail_components(pred,velocity,mask[sl])
            sums=torch.stack([terms[k][0] for k in ('fm','haar1','haar2')])
            mean_loss=torch.where(mask[sl],(mu-target[sl]).abs(),0.).sum()
            loss=(sums/counts.to(sums.dtype)*sums.new_tensor([1.,.05,.05])).sum()+.1*mean_loss/counts[0]
            scaler.scale(loss).backward()
            totals+=torch.cat((sums.detach().double(),mean_loss.detach().double().view(1)))
        sync_gradients(system);totals=cpu_all_reduce(totals)
        old=scaler.get_scale();scaler.step(optimizer);scaler.update()
        if scaler.get_scale()<old:
            overflows+=1
            if rank==0:print(json.dumps({'phase':'scale_backoff','step':step,'scale':scaler.get_scale()}),flush=True)
            if overflows>30:raise RuntimeError('too many FP16 overflows')
            continue
        overflows=0;scheduler.step();step+=1
        if rank==0:
            means=(totals/torch.cat((counts,counts[:1]))).tolist()
            progress={'arm':'C','phase':'training','step':step,'fm':means[0],'haar1':means[1],
                      'haar2':means[2],'mu_l1':means[3],
                      'loss':means[0]+.05*(means[1]+means[2])+.1*means[3],
                      'alpha':float(F.softplus(system.noise.a).detach().cpu()),
                      'beta':float(F.softplus(system.noise.d).detach().cpu()),
                      'seconds':time.time()-tick,'elapsed_seconds':time.time()-started,
                      'lr_next':optimizer.param_groups[0]['lr'],'scale':scaler.get_scale(),
                      'peak_gpu_gib':torch.cuda.max_memory_allocated()/1024**3,'timestamp':time.time()}
            atomic(a.out/'progress.json',progress);print(json.dumps(progress),flush=True)
        stop=torch.tensor(int(STOP));dist.all_reduce(stop,op=dist.ReduceOp.MAX)
        if step%250==0 or step==a.stop or stop.item():save(a.out,system,optimizer,scaler,scheduler,step,identity)
        if stop.item():break
    if rank==0:atomic(a.out/'state.json',{'phase':'segment_stopped','successful_updates':step,'timestamp':time.time()})
    dist.destroy_process_group()


if __name__=='__main__':main()
