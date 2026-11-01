"""Read-only, matched validation gradient probes; never changes training state."""
import argparse
import hashlib
import json
from pathlib import Path
import sys


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--epoch',type=int,required=True);a=p.parse_args();root=a.root.resolve()
    sys.path.insert(0,str(root/'code'))
    import torch
    from model.spectral_gate_uno import SpectralGateUNO2d
    from model.matern_source import SourceSampler,path_triplet
    from data_loaders.multiconfig import multiconfig_collate
    from data_loaders.same_frequency import SameFrequencyRadiomapDataset,load_same_frequency_height_max
    from training.same_frequency_trainer import SCHEMA_PATH
    from experiments.multiconfig_manifest import load_schema_lock
    from training.detail_losses import detail_components,auxiliary
    from training.matern_source import digest_state
    from scripts.monitor_matern_suite import atomic
    torch.set_num_threads(2);torch.manual_seed(42)
    scope=json.loads((root/'execution_scope.json').read_text())
    reference=Path(scope['reference_run']);cfg=json.loads((reference/'config.json').read_text())
    hm=load_same_frequency_height_max(root/'manifests/height_stats_train.json',split_path=root/'manifests/scene_split_seed42.json')
    ds=SameFrequencyRadiomapDataset(dataset_root=Path(cfg['dataset_root']),
        manifest_path=root/'manifests/manifest_samefreq_6.7ghz_8x8_0deg.jsonl',split='val',
        array_size='8x8',height_max=hm,expected_beam_id=4,
        source_metadata=load_schema_lock(SCHEMA_PATH).raw['source_metadata'])
    # First four dataset-order validation scenes, fixed independently of errors.
    batch=multiconfig_collate([ds[i] for i in range(4)])
    c,y,mask=[batch[k].to('cuda') for k in ('condition','target','valid_mask')]
    sampler=SourceSampler('MS');x0=sampler.sample(batch['metadata'],device=torch.device('cuda'),namespace='no_drop_gradient_probe')
    rows=[];identities={}
    for label,directory in [('Drop025',reference),('NoDrop',root/'AP')]:
        checkpoint=directory/'milestones'/f'epoch_{a.epoch:04d}'/'ema.pt'
        state=torch.load(checkpoint,map_location='cpu',weights_only=True)
        model=SpectralGateUNO2d('AP',operator_width=32,cfg_drop_prob=.25 if label=='Drop025' else 0.)
        model.load_state_dict(state,strict=True);del state
        digest=digest_state(model.state_dict());model.cuda().eval()
        identities[label]=dict(weights_sha256=digest,checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            checkpoint=str(checkpoint),weights='online, legacy filename ema.pt')
        named=list(model.named_parameters())
        for time in (0.,.25,.5,.75):
            t=torch.full((len(y),),time,device='cuda');xt,ut=path_triplet(x0,y,t)
            gradients=[];losses=[];encoder_norms=[]
            for conditional in (True,False):
                emb=model.embed_model(c)
                if not conditional:emb=tuple(torch.zeros_like(v) for v in emb)
                pred=model._velocity(c if conditional else torch.zeros_like(c),xt,t,emb)
                terms=detail_components(pred,ut,mask)
                means={k:v[0]/v[1] for k,v in terms.items()}
                loss=means['fm']+.1*auxiliary(means,'L2')
                grad=torch.autograd.grad(loss,[p for _,p in named],allow_unused=True)
                shared=[];enc_sq=0.
                for (name,param),g in zip(named,grad):
                    if g is None:g=torch.zeros_like(param)
                    if not torch.isfinite(g).all():raise RuntimeError('Nonfinite probe gradient')
                    v=torch.view_as_real(g) if g.is_complex() else g
                    if name.startswith('embed_model.'):
                        enc_sq+=float(v.double().square().sum())
                    else:shared.append(v.reshape(-1).detach())
                gradients.append(torch.cat(shared));encoder_norms.append(enc_sq**.5)
                losses.append({k:float(v) for k,v in means.items()}|{'total':float(loss)})
                del grad,pred,emb,loss,terms,means,shared
            ga,gb=gradients;aa=ga.double().square().sum();bb=gb.double().square().sum()
            cos=float((ga.double()*gb.double()).sum()/torch.sqrt(aa*bb)) if aa>0 and bb>0 else None
            rows.append(dict(model=label,epoch=a.epoch,time=time,conditional=losses[0],null=losses[1],
                shared_conditional_norm=float(aa.sqrt()),shared_null_norm=float(bb.sqrt()),
                shared_conditional_null_cosine=cos,condition_encoder_norm=encoder_norms[0],
                null_encoder_norm=encoder_norms[1]))
            del gradients,ga,gb
        if digest_state(model.state_dict())!=digest:raise RuntimeError('Probe mutated weights')
        del model;torch.cuda.empty_cache()
    atomic(root/'monitoring/probes'/f'epoch_{a.epoch:04d}.json',dict(rows=rows,identities=identities,
        scenes=[m['scene_id'] for m in batch['metadata']],epoch=a.epoch,fp32=True,
        optimizer_created=False,weights_unchanged=True,source='matched fixed MS draw, analytical training path',
        limitation='Four validation scenes, diagnostic not population inference. NoDrop null input is OOD; its null loss is NOT a training objective.'))


if __name__=='__main__':main()
