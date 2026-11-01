"""Separate environmental FNO and known array-excitation approximation."""
from functools import lru_cache
import math
import numpy as np
import torch
from torch import nn
from model.spectral_gate_uno import SpectralGateUNO2d
from array_physics import C0,codebook,offsets


@lru_cache(maxsize=256)
def excitation_coefficients(n,frequency,angle):
    # Coefficients follow the frozen exploratory basis, not measured channels.
    # The + sign matches the declared basis. With symmetric basis directions it
    # is an index reversal of Step6's propagation-phase sign; see audit notes.
    w=codebook(int(n),float(angle)); xyz=offsets(int(n),float(frequency))
    result=[]
    for theta in (-30.,-10.,10.,30.):
        rad=np.deg2rad(theta);direction=np.array([np.cos(rad),np.sin(rad),0.])
        gain_db=8.-min(12*(theta/65.)**2,30.)
        basis=np.sqrt(10**(gain_db/10))*np.exp(2j*np.pi*float(frequency)/C0*(xyz@direction))
        result.append(np.dot(np.conj(w),basis))
    return np.asarray(result,dtype=np.complex64)


class EnvironmentalPrediction(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=SpectralGateUNO2d('AP',operator_width=32,cfg_drop_prob=0.)
        old=self.net.projection_output
        self.net.projection_output=nn.Conv2d(old.in_channels,8,1)
        nn.init.xavier_uniform_(self.net.projection_output.weight)
        nn.init.zeros_(self.net.projection_output.bias)

    def forward(self,condition,queries,amp=True):
        # condition is strictly [Tx,height,frequency]. No beam, array, or labels.
        if condition.ndim!=4 or condition.shape[1]!=3:raise ValueError('environment requires [Tx,H,F]')
        zeros=torch.zeros_like(condition[:,:1]); times=torch.zeros(condition.shape[0],device=condition.device)
        with torch.autocast(device_type=condition.device.type,dtype=torch.float16,enabled=condition.is_cuda and amp):
            out=self.net(image=condition,x=zeros,step=times)
        with torch.autocast(device_type=condition.device.type,enabled=False):
            out=out.float()
            basis=torch.complex(out[:,0::2],out[:,1::2])/100.
            coeff=np.stack([excitation_coefficients(q['array'],q['frequency_hz'],q['angle']) for q in queries])
            coeff=torch.from_numpy(coeff).to(condition.device)
            field=(basis*coeff[:,:,None,None]).sum(1,keepdim=True)
            wavelength=torch.tensor([C0/q['frequency_hz'] for q in queries],device=condition.device,dtype=torch.float32)
            power=(wavelength[:,None,None,None]/(4*math.pi)).square()*field.abs().square()
            mu=(10*torch.log10(power+1e-30)+300.)/300.
            return mu,basis,field,coeff
