"""Deterministic no-model source-copy reference, using target-only score masks."""
import argparse
from pathlib import Path
import time
import numpy as np
import torch
from torch.nn import functional as F
from beam_boundary.common import atomic, read, load_data, verify
from beam_boundary.evaluate import metrics, response, summarize
from .geometry import regional_metrics, summarize_regions


def evaluate(root, split):
    dest=root/'source_copy'/split
    if (dest/'complete.json').exists():return
    arrays,records=load_data(root,split);raw=Path(read(root/'input_audit.json')['raw_root'])
    scenes=sorted({r['scene_id'] for r in records});rows=[];deltas=[]
    for count,scene in enumerate(scenes):
        indices=sorted([i for i,r in enumerate(records) if r['scene_id']==scene],key=lambda i:records[i]['steering_deg'])
        c=np.asarray(arrays['condition'][indices]);mu=np.where(c[0,5]>.5,c[0,4],0.)
        pred=mu*300-300;gt=np.asarray(arrays['target'][indices,0])*300-300
        valid=np.asarray(arrays['valid'][indices,0]);angles=[records[i]['steering_deg'] for i in indices];zero=angles.index(0.)
        native_pred=F.interpolate(torch.from_numpy(mu)[None,None],(128,128),mode='bilinear',align_corners=False)[0,0].numpy()*300-300
        with np.load(root/'data/geometry'/f'{scene}.npz') as regions:
            for j,i in enumerate(indices):
                native=np.load(raw/records[i]['radiomap_path']);nv=(native>-300)&(native<0)
                row=dict(scene=scene,angle=angles[j],draw=0,**metrics(pred,gt[j],valid[j],c[0,1]),
                    native_rmse_db=float(np.sqrt(np.mean((native_pred[nv]-native[nv])**2))),
                    source_missing_target_valid_pixels=int((valid[j] & (c[0,5]<.5)).sum()),
                    **regional_metrics(native_pred,native,nv,regions))
                rows.append(row)
                if j!=zero:deltas.append(dict(scene=scene,angle=angles[j],draw=0,
                    **response(pred,pred,gt[j],gt[zero],valid[j]&valid[zero])))
        atomic(root/'source_copy/progress.json',dict(split=split,scenes_done=count+1,scenes_total=len(scenes),updated=time.time()))
    summary=summarize(rows,deltas);summary.update(summarize_regions(rows))
    summary.update(kind='deterministic_source_copy',model=False,draws=1,
        missing_source_policy='normalized zero, identical to T3 source mean; target masks unchanged',
        source_missing_target_valid_pixels=sum(r['source_missing_target_valid_pixels'] for r in rows))
    atomic(dest/'rows.json',dict(maps=rows,responses=deltas));atomic(dest/'summary.json',summary)
    atomic(dest/'complete.json',dict(split=split,scenes=len(scenes),angles=8,draws=1,model=False,updated=time.time()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root.resolve()
    verify(root);torch.set_num_threads(2)
    for split in ('val','test'):evaluate(root,split)
