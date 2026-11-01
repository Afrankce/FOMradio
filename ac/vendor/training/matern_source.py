"""Fixed U-NO / external-source-only experiment; no innovation bank."""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import time
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from data_loaders.multiconfig import multiconfig_collate
from evaluation.radioflow_sampling import euler_cfg_sample
from evaluation.radiomap_metrics import MetricAccumulator
from evaluation.matern_ablation_metrics import scene_diagnostics
from model.matern_source import SourceSampler, private_seed, path_triplet
from training.width32_distributed import Width32Config, Width32Trainer
from training.same_frequency_trainer import _atomic_json


@dataclass(frozen=True)
class SourceConfig(Width32Config):
    source: str = 'W'

    @property
    def model_size(self):
        return 'attention_multiscale_uno_width32_source_only'

    @property
    def early_stopping_patience(self):
        return 1001  # Disabled for the fixed 1000-epoch budget.

    def scientific_payload(self):
        return {**super().scientific_payload(), 'experiment':'matern_external_source_pilot_v1',
                'external_source':self.source, 'internal_noise':False, 'feedback':False,
                'source_specification':SourceSampler(self.source).specification(),
                'checkpoint_selection':'epoch1000_last_EMA', 'early_stopping_patience':1001,
                'milestone_interval':100, 'milestone_steps':[2,8], 'milestone_realizations':4,
                'training_seed':42, 'path':'independent_linear_sigma0'}


def digest_state(state):
    h=hashlib.sha256()
    for key,value in state.items():
        h.update(key.encode()); h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def predict_source(network, sampler, batch, device, steps=2, realization=0, use_amp=True):
    condition=batch['condition'].to(device,non_blocking=True)
    x0=sampler.sample(batch['metadata'],device=device,namespace=f'eval:{realization}')
    return euler_cfg_sample(network,condition,x0,cfg_scale=1.,steps=steps,use_amp=use_amp)


def extended_diagnostics(prediction,target,mask):
    row=scene_diagnostics(prediction,target,mask)
    p=prediction.clamp(0,1).double()*300; y=target.double()*300
    for axis,label in [(-2,'vertical'),(-1,'horizontal')]:
        for lag in (1,2,4,8,16):
            a,b=[slice(None)]*4,[slice(None)]*4
            a[axis],b[axis]=slice(lag,None),slice(None,-lag);a,b=tuple(a),tuple(b)
            valid=mask[a]&mask[b]; count=int(valid.sum())
            key=f'{label}_lag{lag}'
            row[key+'_count']=count
            row[key+'_gradient_mse']=float(((p[a]-p[b])-(y[a]-y[b]))[valid].square().mean()) if count else None
            for f,name in [(p,'pred'),(y,'target')]:
                row[key+'_'+name+'_variogram']=float((f[a]-f[b])[valid].square().mean()/2) if count else None
    return row


@torch.inference_mode()
def evaluate_dataset(network,sampler,dataset,device,output,*,steps=(2,8),realizations=4,
                     epoch=1000,save_predictions=True,time_diagnostics=False,use_amp=True):
    output=Path(output); output.parent.mkdir(parents=True,exist_ok=True)
    identity=dict(source=sampler.specification(),epoch=epoch,steps=list(steps),realizations=realizations,
                  ema_sha256=digest_state(network.state_dict()))
    if output.exists():
        old=json.loads(output.read_text())
        if old['identity']!=identity:
            raise RuntimeError(f'Existing evaluation identity mismatch: {output}')
        return old
    network.eval(); records=[];aggregated=[];time_rows=[]
    loader=DataLoader(dataset,batch_size=16,shuffle=False,num_workers=2,collate_fn=multiconfig_collate)
    start=time.perf_counter()
    for n in steps:
        for realization in range(realizations):
            acc=MetricAccumulator()
            for batch in loader:
                prediction=predict_source(network,sampler,batch,device,n,realization,use_amp)
                target=batch['target'].to(device);mask=batch['valid_mask'].to(device)
                acc.update(prediction,target,mask)
                for index,m in enumerate(batch['metadata']):
                    row=extended_diagnostics(prediction[index:index+1],target[index:index+1],mask[index:index+1])
                    records.append(dict(scene_id=m['scene_id'],steps=n,realization=realization,**row))
                    if save_predictions and n in (2,8):
                        directory=output.parent/(output.stem+'_predictions')/f'N{n}'/f'seed_{realization}'
                        directory.mkdir(parents=True,exist_ok=True)
                        np.savez_compressed(directory/f"{m['scene_id']}.npz",prediction=prediction[index].cpu().numpy(),
                            target=batch['target'][index].numpy(),valid_mask=batch['valid_mask'][index].numpy())
            aggregated.append(dict(steps=n,realization=realization,**acc.compute()))
            print(json.dumps(dict(event='source_evaluation',epoch=epoch,path=str(output),**aggregated[-1])),flush=True)
    if time_diagnostics:
        for batch in loader:
            condition=batch['condition'].to(device);target=batch['target'].to(device);mask=batch['valid_mask'].to(device)
            x0=sampler.sample(batch['metadata'],device=device,namespace='time_diagnostic')
            embedding=network.embed_model(condition)
            for bin_index in range(5):
                times=torch.tensor([(bin_index+float(torch.rand((),generator=torch.Generator().manual_seed(
                    private_seed(m,f'time_diag:{bin_index}')))))/5 for m in batch['metadata']],device=device)
                xt,ut=path_triplet(x0,target,times)
                with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=use_amp and device.type=='cuda'):
                    vt=network.forward_with_cfg(image=condition,x=xt,step=times,embedding=embedding,cfg_scale=1.)
                for i,m in enumerate(batch['metadata']):
                    v=mask[i]; truth=ut[i][v].double()
                    time_rows.append(dict(scene_id=m['scene_id'],bin=bin_index,time=float(times[i]),
                        mse=float((vt[i][v].double()-truth).square().mean()),target_velocity_variance=float(truth.var(unbiased=False)),
                        target_velocity_mean_square=float(truth.square().mean())))
    payload=dict(identity=identity,records=records,aggregated=aggregated,time_diagnostics=time_rows,
                 n_scenes=len(dataset),elapsed_seconds=time.perf_counter()-start,
                 fine_metric='complete-support Haar finest two levels after matched 2x2 benchmark average')
    _atomic_json(output,payload)
    return payload


class SourceTrainer(Width32Trainer):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.source_sampler=SourceSampler(self.cfg.source)

    def _prediction_triplet(self,model,batch):
        condition=batch['condition'].to(self.device); target=batch['target'].to(self.device)
        mask=batch['valid_mask'].to(self.device)
        x0=self.source_sampler.sample(batch['metadata'],device=self.device,namespace=f'train:{self.next_epoch_index}')
        # Counter-based time stream, distinct from source / loader / CFG streams.
        times=torch.tensor([float(torch.rand((),generator=torch.Generator().manual_seed(
            private_seed(m,f'time:{self.next_epoch_index}')))) for m in batch['metadata']],device=self.device)
        xt,ut=path_triplet(x0,target,times)
        with torch.autocast(device_type=self.device.type,dtype=torch.float16,enabled=self.cfg.use_amp and self.device.type=='cuda'):
            vt=model(image=condition,x=xt,step=times)
        return vt,ut,mask

    def validate(self):
        result=[None]
        if dist.get_rank()==0:
            network=self.ema.ema_model;network.eval();acc=MetricAccumulator()
            for batch in self.val_loader:
                prediction=predict_source(network,self.source_sampler,batch,self.device,use_amp=self.cfg.use_amp)
                acc.update(prediction,batch['target'].to(self.device),batch['valid_mask'].to(self.device))
            result[0]=acc.compute()
        dist.broadcast_object_list(result,src=0)
        return result[0]

    def _save(self,name,*,smoke=False):
        path=super()._save(name,smoke=smoke)
        if name=='last.pt' and not smoke:
            if self.completed_epochs>=100 and (1-self.optimizer_step/(self.completed_epochs*5))>.01:
                raise RuntimeError('AMP skipped more than 1% of planned updates; protocol stability failure')
            if self.completed_epochs%100==0:
                result=[None]
                if dist.get_rank()==0:
                    try:
                        self.milestone()
                    except Exception as error:
                        result[0]=repr(error)
                dist.broadcast_object_list(result,src=0)
                if result[0]: raise RuntimeError(result[0])
        return path

    def milestone(self):
        directory=self.cfg.run_dir/'milestones'/f'epoch_{self.completed_epochs:04d}'
        directory.mkdir(parents=True,exist_ok=True)
        if not (directory/'ema.pt').exists():
            temporary=directory/'ema.pt.tmp';torch.save(self.ema.ema_model.state_dict(),temporary);temporary.replace(directory/'ema.pt')
        evaluate_dataset(self.ema.ema_model,self.source_sampler,self.val_loader.dataset,self.device,
                         directory/'val.json',epoch=self.completed_epochs,time_diagnostics=True,use_amp=self.cfg.use_amp)
