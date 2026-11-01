"""Validation-only Euler inference, with labels loaded only after prediction."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
import torch.nn.functional as F
from dataset import QueryLoader
from cross_array.model import SourceObservationUNO
from model.matern_source import SourceSampler
from train_support import sample_paired,sha,atomic
from train_c import CSystem


@torch.inference_mode()
def predict(arm,network,queries,loader,seed):
    device=next(network.parameters()).device
    condition,beam=loader.inputs(queries,device,arm)
    meta=[{'scene_id':f"{q['scene']}|array{q['array']}|f{q['frequency_hz']}|angle{q['angle']}",
           'steering_deg':q['angle']} for q in queries]
    if arm=='A':
        samplers={s:SourceSampler(s,resolution=256,seed=seed) for s in ('MS','S2','S8','S32')}
        state=sample_paired(samplers,meta,device,'val-draw-0');velocity=network
    else:
        state,_,_,_,_,_=network.source(condition,beam,queries,meta,'val-draw-0');velocity=network.velocity
    with torch.autocast('cuda',dtype=torch.float16):emb=velocity.embed_model(condition)
    for i in range(8):
        t=torch.full((len(queries),),i/8,device=device,dtype=torch.float32)
        with torch.autocast('cuda',dtype=torch.float16):
            v=velocity(image=condition,x=state,step=t,embedding=emb)
        state=state+v.float()/8.
    return F.interpolate(state,size=(128,128),mode='bilinear',align_corners=False).float().cpu().numpy()[:,0]


def main():
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=['A','C'],required=True)
    p.add_argument('--dataset',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--checkpoint-dir',type=Path,required=True);p.add_argument('--physics',type=Path)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--batch',type=int,default=4)
    p.add_argument('--limit',type=int,default=0);a=p.parse_args()
    if a.manifest.name!='val.json':raise ValueError('only frozen validation manifest permitted')
    records=json.loads(a.manifest.read_text());assert len(records)==240
    if any(r['query']['split']!='val' or r['query']['array']!=8 or r['query']['angle'] not in (-14,0,14) for r in records):
        raise ValueError('manifest is not the frozen validation split')
    if a.limit:records=records[:a.limit]
    commit=json.loads((a.checkpoint_dir/'committed.json').read_text());path=a.checkpoint_dir/commit['path']
    if sha(path)!=commit['sha256']:raise RuntimeError('checkpoint hash mismatch')
    saved=torch.load(path,map_location='cpu',weights_only=False);ident=saved['identity'];seed=ident['seed']
    if ident['arm']!=a.arm:raise RuntimeError('checkpoint arm mismatch')
    if a.arm=='A':network=SourceObservationUNO('T0')
    else:
        if not a.physics:raise ValueError('C requires physics diagnostics')
        network=CSystem(seed,json.loads(a.physics.read_text())['training_beam_gradient_p95'])
    network.load_state_dict(saved['model']);network=network.cuda().eval();loader=QueryLoader(a.dataset)
    rows=[];start=time.time();outliers=0;total=0
    for start_idx in range(0,len(records),a.batch):
        chunk=records[start_idx:start_idx+a.batch]
        pred=predict(a.arm,network,[r['query'] for r in chunk],loader,seed)
        if not np.isfinite(pred).all():raise RuntimeError('nonfinite prediction')
        outliers+=int(((pred<0)|(pred>1)).sum());total+=pred.size
        # Offline scoring begins only after target-free inference returns.
        for r,prediction in zip(chunk,pred):
            gt=np.load(a.dataset/r['supervision']['target'],allow_pickle=False)
            valid=(gt>-300)&(gt<0)
            if not valid.any():raise ValueError('empty scoring support')
            mae=float(np.abs(prediction[valid]*300-300-gt[valid]).mean())
            rows.append({'scene':r['query']['scene'],'angle':r['query']['angle'],'mae_db':mae})
    result={'arm':a.arm,'seed':seed,'checkpoint_step':saved['step'],'checkpoint_sha256':commit['sha256'],
            'manifest_sha256':sha(a.manifest),'count':len(rows),'nfe':8,'draws':1,'batch':a.batch,
            'mae_db_macro':float(np.mean([r['mae_db'] for r in rows])),
            'outside_range_fraction':outliers/total,'elapsed_seconds':time.time()-start,
            'purpose':'validation pipeline smoke' if a.limit else 'checkpoint selection', 'rows':rows}
    atomic(a.out,result); print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)


if __name__=='__main__':main()
