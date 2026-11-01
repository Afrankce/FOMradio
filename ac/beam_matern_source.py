"""Differentiable beam-conditioned Matérn source (not active measurement)."""
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from model.matern_source import SourceSampler


def spatial_gradient(b):
    gx=torch.empty_like(b); gy=torch.empty_like(b)
    gx[...,1:-1]=(b[...,2:]-b[...,:-2])/2
    gx[...,0]=b[...,1]-b[...,0];gx[...,-1]=b[...,-1]-b[...,-2]
    gy[...,1:-1,:]=(b[...,2:,:]-b[...,:-2,:])/2
    gy[...,0,:]=b[...,1,:]-b[...,0,:];gy[...,-1,:]=b[...,-1,:]-b[...,-2,:]
    return torch.sqrt(gx.square()+gy.square())


def probability_map(b,alpha,beta,scale,eta=.2,tau=1.):
    g=(spatial_gradient(b)/max(float(scale),1e-6)).clamp(0,1)
    score=(alpha*b+beta*g)/tau
    q=torch.softmax(score.flatten(1),dim=1).view_as(b)
    return (1-eta)*q+eta/(b.shape[-2]*b.shape[-1])


def filter_noise(white,amplitude):
    size=white.shape[-2:]
    return torch.fft.irfft2(torch.fft.rfft2(white.float(),norm='ortho')*amplitude,
                            s=size,norm='ortho')


def variance_spectrum(amplitudes):
    n=amplitudes.shape[-2]
    kernels=torch.fft.irfft2(amplitudes,s=(n,n),norm='backward')
    # The convolution kernel is irfft(amplitude), not the covariance itself.
    return torch.fft.rfft2(kernels.square().mean(dim=0),norm='backward')


def variance_from_amplitudes(rho,amplitudes):
    kernel_hat=variance_spectrum(amplitudes)
    return torch.fft.irfft2(torch.fft.rfft2(rho,norm='backward')*kernel_hat,
                            s=rho.shape[-2:],norm='backward')


class BeamMaternSource(nn.Module):
    def __init__(self,seed,gradient_scale,resolution=256):
        super().__init__()
        if not math.isfinite(float(gradient_scale)) or gradient_scale<=0:raise ValueError('invalid gradient scale')
        self.resolution=resolution;self.gradient_scale=float(gradient_scale)
        self.samplers={x:SourceSampler(x,resolution=resolution,seed=seed) for x in ('MS','S2','S8','S32')}
        self.extent=self.samplers['MS'].extent
        amp=np.stack([self.samplers[x].amplitude for x in ('S2','S8','S32')])
        self.register_buffer('amplitudes',torch.from_numpy(amp.copy()))
        self.register_buffer('var_kernel',variance_spectrum(torch.from_numpy(amp.copy())))
        initial=math.log(math.expm1(1.))
        self.a=nn.Parameter(torch.tensor(initial,dtype=torch.float32))
        self.d=nn.Parameter(torch.tensor(initial,dtype=torch.float32))

    def forward(self,b,metadata,namespace,uniform=False):
        if b.ndim!=4 or b.shape[1:]!=(1,self.resolution,self.resolution):raise ValueError('bad beam shape')
        with torch.autocast(device_type=b.device.type,enabled=False):
            b=b.float();alpha=F.softplus(self.a);beta=F.softplus(self.d)
            q=probability_map(b,alpha,beta,self.gradient_scale)
            if uniform:q=torch.full_like(q,1/(self.resolution*self.resolution))
            rho=q*(self.resolution*self.resolution)
            pad=self.extent-self.resolution
            rho=F.pad(rho-1.,(0,pad,0,pad))+1.
            var=torch.fft.irfft2(torch.fft.rfft2(rho,norm='backward')*self.var_kernel,
                                  s=(self.extent,self.extent),norm='backward').clamp_min(1e-8)
            raw=torch.zeros_like(rho)
            for i,key in enumerate(('S2','S8','S32')):
                eps=self.samplers[key].base_noise(metadata,b.device,namespace+'-'+key)
                raw=raw+filter_noise(rho.sqrt()*eps,self.amplitudes[i])/math.sqrt(3.)
            r=self.resolution
            guided=raw[...,:r,:r]/var[...,:r,:r].sqrt()
            base=self.samplers['MS'].sample(metadata,device=b.device,namespace=namespace+'-base')
            return math.sqrt(.75)*base+.5*guided,q
