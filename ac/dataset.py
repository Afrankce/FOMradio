"""No-observation U6G interfaces. Query inputs exclude all label paths."""
from functools import lru_cache
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F


def preprocess_target(native):
    if native.ndim != 4 or native.shape[1:] != (1,128,128):
        raise ValueError('native target must be Bx1x128x128')
    if not torch.isfinite(native).all(): raise ValueError('nonfinite label')
    valid=(native>-300)&(native<0)
    if not (valid|(native==1000)|(native==-300)).all():
        raise ValueError('unexpected label sentinel')
    weighted=torch.where(valid,(native+300)/300,0.)
    num=F.interpolate(weighted,size=(256,256),mode='bilinear',align_corners=False)
    den=F.interpolate(valid.float(),size=(256,256),mode='bilinear',align_corners=False)
    support=den>1-1e-6
    value=num/den.clamp_min(1e-8)
    return torch.where(support,value,0.),support


class QueryLoader:
    def __init__(self,root): self.root=Path(root)

    @lru_cache(maxsize=800)
    def _height(self,path):
        a=np.asarray(np.load(self.root/path,allow_pickle=False),np.float32)
        if a.shape!=(256,256) or not np.isfinite(a).all() or a.min()<0 or a.max()>445:
            raise ValueError(f'invalid height {path}')
        return a/445.

    @lru_cache(maxsize=32)
    def _beam(self,path):
        a=np.asarray(np.load(self.root/path,allow_pickle=False),np.float32)
        if a.shape!=(128,128) or not np.isfinite(a).all(): raise ValueError(f'invalid beam {path}')
        return (np.clip(a,-300,0)+300)/300

    def inputs(self,queries,device,arm='A'):
        # Only these whitelisted metadata keys reach this prediction interface.
        if arm not in ('A','C'): raise ValueError(arm)
        h=torch.from_numpy(np.stack([self._height(q['height']) for q in queries]))[:,None].to(device)
        b=torch.from_numpy(np.stack([self._beam(q['beam']) for q in queries]))[:,None].to(device)
        b=F.interpolate(b,size=(256,256),mode='bilinear',align_corners=False)
        tx=torch.zeros_like(h);tx[:,:,127,127]=1
        f=torch.tensor([np.log(q['frequency_hz']/6.7e9) for q in queries],device=device,dtype=torch.float32)
        f=f[:,None,None,None].expand_as(h)
        z=torch.zeros_like(h)
        condition=torch.cat((tx,h,b if arm=='A' else f,z,z,z),dim=1)
        return condition,b


def training_targets(root,records,device):
    # Separate explicit supervision path. It is never called by inputs().
    a=np.stack([np.load(Path(root)/r['supervision']['target'],allow_pickle=False)
                for r in records]).astype(np.float32,copy=False)
    return preprocess_target(torch.from_numpy(a[:,None]).to(device))
