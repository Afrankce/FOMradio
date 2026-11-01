"""No-observation A/C inference and offline scoring on frozen U6G splits.

One process evaluates one model/seed and checkpoints every group/draw. It can
resume completed units only if the full code/manifest/model identity matches.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
from functools import lru_cache
import numpy as np
import torch
import torch.nn.functional as F
from cross_array.model import SourceObservationUNO
from dataset import QueryLoader
from model.matern_source import SourceSampler
from train_c import CSystem
from train_support import sample_paired,sha,atomic
from metrics_ac import boundary_from_height,score_map

ROOT=Path(__file__).resolve().parent
DATA=Path(os.environ.get('U6G_DATASET_ROOT','/path/to/U6G/Dataset'))
GROUPS=('seen_config','array','angle_interpolation','angle_extrapolation','joint')


def stable_hash(obj):return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def disk_guard():
    if shutil.disk_usage(ROOT).free<25*1024**3:raise RuntimeError('less than 25 GiB free')


@lru_cache(maxsize=3000)
def target(rel):
    value=np.asarray(np.load(DATA/rel,allow_pickle=False),dtype=np.float32)
    if value.shape!=(128,128) or not np.isfinite(value).all():raise ValueError('invalid target')
    valid=(value>-300)&(value<0)
    if not (valid|(value==-300)|(value==1000)).all():raise ValueError('unexpected target sentinel')
    return value,valid


def metadata_for(queries,group,draw,validation=False):
    if validation:
        return [{'scene_id':f"{q['scene']}|array{q['array']}|f{q['frequency_hz']}|angle{q['angle']}",
                 'steering_deg':q['angle']} for q in queries], 'val-draw-0'
    return [{'scene_id':f"test|{group}|{q['scene']}|array{q['array']}|f{q['frequency_hz']}|angle{q['angle']}|draw{draw}",
             'steering_deg':q['angle']} for q in queries], f'test-{group}-draw-{draw}'


@torch.inference_mode()
def predict(model,arm,samplers,queries,loader,group,draw,validation=False):
    """Prediction interface only sees query inputs, never target/mask paths."""
    device=next(model.parameters()).device
    condition,beam=loader.inputs(queries,device,arm)
    meta,namespace=metadata_for(queries,group,draw,validation)
    mu=None
    if arm=='A':
        state=sample_paired(samplers,meta,device,namespace);velocity=model
    else:
        state,mu,_,_,_,_=model.source(condition,beam,queries,meta,namespace)
        velocity=model.velocity
    with torch.autocast('cuda',dtype=torch.float16):embedding=velocity.embed_model(condition)
    for i in range(8):
        t=torch.full((len(queries),),i/8,device=device,dtype=torch.float32)
        with torch.autocast('cuda',dtype=torch.float16):
            v=velocity(image=condition,x=state,step=t,embedding=embedding)
        state=state+v.float()/8.
    pred=F.interpolate(state,size=(128,128),mode='bilinear',align_corners=False).float().cpu().numpy()[:,0]
    mu_native=None if mu is None else F.interpolate(mu,size=(128,128),mode='bilinear',align_corners=False).float().cpu().numpy()[:,0]
    if not np.isfinite(pred).all() or (mu_native is not None and not np.isfinite(mu_native).all()):
        raise RuntimeError('non-finite prediction or mean')
    return pred*300.-300., None if mu_native is None else mu_native*300.-300.


def identity(arm,seed,spec,checkpoint):
    files=['evaluate_test_ac.py','metrics_ac.py','dataset.py','train_support.py','train_c.py',
           'env_operator.py','beam_matern_source.py','array_physics.py','vendor_code.tar.gz','test_run_spec.json']
    files += [f'manifests/{x}.json' for x in GROUPS]
    return {'arm':arm,'seed':seed,'checkpoint_sha256':checkpoint,'plan_sha256':spec['frozen_plan_sha256'],
            'files_sha256':{name:sha(ROOT/name) for name in files},'torch':torch.__version__,
            'numpy':np.__version__,'cuda':torch.version.cuda,'nfe':8,'draws':4}


def read_best(arm,seed):
    folder=ROOT/'runs'/f'{arm}_seed{seed}'
    best=json.loads((folder/'best.json').read_text());path=folder/'best.pt'
    if sha(path)!=best['checkpoint_sha256']:raise RuntimeError('best checkpoint digest mismatch')
    ck=torch.load(path,map_location='cpu',weights_only=False)
    if ck['step']!=best['step'] or ck['identity']['arm']!=arm or ck['identity']['seed']!=seed:
        raise RuntimeError('best checkpoint identity mismatch')
    return best,ck


def load_model(arm,seed,ck):
    if arm=='A':model=SourceObservationUNO('T0')
    else:
        p=json.loads((ROOT/'diagnostics'/'c_physics.json').read_text())
        model=CSystem(seed,p['training_beam_gradient_p95'])
    model.load_state_dict(ck['model']);return model.cuda().eval()


def score_batch(pred,mu,records,loader):
    # Label files are read here, only after the prediction call returns.
    rows=[]
    for i,record in enumerate(records):
        q=record['query'];y,valid=target(record['supervision']['target'])
        boundary=boundary_from_height(loader._height(q['height']))
        metric=score_map(pred[i],y,valid,boundary)
        if mu is not None:metric['mu_mae_db']=float(np.abs(mu[i][valid]-y[valid]).mean())
        rows.append({'scene':q['scene'],'angle':q['angle'],'array':q['array'],**metric})
    return rows


def smoke(arm,seed,model,samplers,loader,best,limit,batch):
    records=json.loads((ROOT/'manifests'/'val.json').read_text())[:limit]
    old=json.loads((ROOT/best['validation']).read_text())
    oldrows={(r['scene'],r['angle']):r['mae_db'] for r in old['rows']}
    rows=[]
    for i in range(0,len(records),batch):
        sub=records[i:i+batch]
        pred,mu=predict(model,arm,samplers,[r['query'] for r in sub],loader,'val',0,validation=True)
        rows+=score_batch(pred,mu,sub,loader)
    error=max(abs(r['mae_db']-oldrows[(r['scene'],r['angle'])]) for r in rows)
    if error>1e-4:raise RuntimeError(f'validation smoke mismatch {error}')
    return {'arm':arm,'seed':seed,'n':len(rows),'max_mae_difference_db':error,
            'input_contract':'prediction called before target is loaded'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=('A','C'),required=True)
    p.add_argument('--seed',type=int,choices=(42,43),required=True);p.add_argument('--batch',type=int,default=4)
    p.add_argument('--validation-smoke',type=int,default=0);a=p.parse_args()
    torch.set_num_threads(2);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    spec=json.loads((ROOT/'test_run_spec.json').read_text())
    if sha(ROOT/'plan_frozen.md')!=spec['frozen_plan_sha256']:raise RuntimeError('frozen plan changed')
    best,ck=read_best(a.arm,a.seed)
    model=load_model(a.arm,a.seed,ck);del ck
    loader=QueryLoader(DATA)
    samplers={s:SourceSampler(s,resolution=256,seed=a.seed) for s in ('MS','S2','S8','S32')} if a.arm=='A' else None
    if a.validation_smoke:
        print(json.dumps(smoke(a.arm,a.seed,model,samplers,loader,best,a.validation_smoke,a.batch)),flush=True)
        return
    ident=identity(a.arm,a.seed,spec,best['checkpoint_sha256']);identity_hash=stable_hash(ident)
    out=ROOT/spec['test_output_root']/f'{a.arm}_seed{a.seed}';out.mkdir(parents=True,exist_ok=True)
    if (out/'identity.json').exists():
        if json.loads((out/'identity.json').read_text())!=ident:raise RuntimeError('identity changed on resume')
    else:atomic(out/'identity.json',ident)
    scenes=sorted({r['query']['scene'] for r in json.loads((ROOT/'manifests'/'array.json').read_text())},
                  key=lambda x:int(x[1:]))
    if len(scenes)!=160:raise RuntimeError('wrong test scene count')
    selected={scenes[0],scenes[79],scenes[159]}
    atomic(out/'selection.json',{'scene_ids':sorted(selected,key=lambda x:int(x[1:])),
                                 'array':32,'angle':0,'selection':'numeric scene order 1,80,160'})
    start=time.time();total_done=0
    try:
        for group in GROUPS:
            records=json.loads((ROOT/'manifests'/f'{group}.json').read_text())
            if len(records)!=spec['groups_expected_queries'][group] or len({r['query']['scene'] for r in records})!=160:
                raise RuntimeError('manifest count mismatch')
            if any(r['query']['split']!='test' or r['query']['group']!=group for r in records):
                raise RuntimeError('wrong test group')
            for draw in range(4):
                disk_guard();dest=out/f'{group}_draw{draw}.json'
                if dest.exists():
                    saved=json.loads(dest.read_text())
                    if saved['identity_sha256']!=identity_hash or len(saved['rows'])!=len(records):
                        raise RuntimeError('incomplete or stale existing result')
                    total_done+=len(records);continue
                rows=[]
                for i in range(0,len(records),a.batch):
                    sub=records[i:i+a.batch];queries=[r['query'] for r in sub]
                    predictions,mus=predict(model,a.arm,samplers,queries,loader,group,draw)
                    rows+=score_batch(predictions,mus,sub,loader)
                    if group=='array':
                        for j,r in enumerate(sub):
                            q=r['query']
                            if q['scene'] in selected and q['angle']==0:
                                y,valid=target(r['supervision']['target'])
                                path=out/'selected_maps'/f"{q['scene']}_draw{draw}.npz"
                                path.parent.mkdir(exist_ok=True)
                                temp=path.with_suffix('.npz.tmp')
                                with temp.open('wb') as f:
                                    np.savez_compressed(f,prediction_db=predictions[j],ground_truth_db=y,valid=valid,
                                                        mean_db=mus[j] if mus is not None else np.array([]))
                                    f.flush();os.fsync(f.fileno())
                                os.replace(temp,path)
                    if i==0 or (i//a.batch)%10==0:
                        atomic(out/'progress.json',{'phase':'testing','arm':a.arm,'seed':a.seed,'group':group,
                               'draw':draw,'queries_done':min(i+a.batch,len(records)),'queries_total':len(records),
                               'completed_predictions':total_done+i+len(sub),'elapsed_seconds':time.time()-start,
                               'timestamp':time.time()})
                atomic(dest,{'arm':a.arm,'seed':a.seed,'group':group,'draw':draw,'identity_sha256':identity_hash,
                             'checkpoint_step':best['step'],'rows':rows})
                total_done+=len(records)
                print(json.dumps({'arm':a.arm,'seed':a.seed,'group':group,'draw':draw,'count':len(rows),
                                  'completed_predictions':total_done,'elapsed_seconds':time.time()-start}),flush=True)
        atomic(out/'progress.json',{'phase':'complete','arm':a.arm,'seed':a.seed,
                                   'completed_predictions':total_done,'timestamp':time.time()})
    except Exception as exc:
        atomic(out/'progress.json',{'phase':'error','arm':a.arm,'seed':a.seed,'error':str(exc),'timestamp':time.time()})
        raise


if __name__=='__main__':main()
