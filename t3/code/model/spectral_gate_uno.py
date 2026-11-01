"""Bounded conditional spectral modulation; no new random fields or FM path.

The legacy rFFT boundary representation is unchanged. New residual modulation
is zero on kx=0/Nyquist columns, so it is a valid real-field spectral increment
without changing the original kernel's boundary projection at initialization.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from model.attention_multiscale_uno import AttentionMultiscaleUNO2d, AttentionConditionedFNOStage
from model.unet.basic_unet_denose import nonlinearity

VARIANTS = ('B', 'R', 'A', 'AP')


def retained(value, m1, m2):
    return torch.cat((value[..., :m1, :m2], value[..., -m1:, :m2]), -2)


class SpectralGate(nn.Module):
    def __init__(self, variant, condition_channels, width, time_channels=512):
        super().__init__()
        self.variant, self.width = variant, width
        self.eta, self.beta, self.eps = .5, math.pi/4, 1e-6
        self.capture = False
        self.diagnostic = None
        self.arrays = None
        if variant == 'R':
            self.radial_raw = nn.Parameter(torch.zeros(()))
        elif variant in ('A', 'AP'):
            self.align = nn.Conv2d(condition_channels, width, 1)
            self.time_features = nn.Linear(time_channels, 4)
            n = (4 if variant == 'AP' else 2)*width + 6
            self.shared = nn.Sequential(nn.Conv2d(n, 32, 1), nn.GELU())
            self.amplitude_head = nn.Conv2d(32, width, 1)
            nn.init.zeros_(self.amplitude_head.weight)
            nn.init.zeros_(self.amplitude_head.bias)
            if variant == 'AP':
                self.phase_head = nn.Conv2d(32, width, 1)
                nn.init.zeros_(self.phase_head.weight)
                nn.init.zeros_(self.phase_head.bias)
        else:
            raise ValueError(variant)

    def forward(self, h_hat, condition, time_embedding, spatial, modes, padding):
        height, width = spatial
        m1, m2 = modes
        ky = torch.fft.fftfreq(height, device=h_hat.device)
        kx = torch.fft.rfftfreq(width, device=h_hat.device)[:m2]
        ky = torch.cat((ky[:m1], ky[-m1:]))
        yy, xx = torch.meshgrid(ky, kx, indexing='ij')
        # Axis Nyquist normalized coordinates, not RF carrier frequencies.
        coords = torch.stack((2*yy, 2*xx))[None]
        active = (xx > 0) & (xx < .5)
        if self.variant == 'R':
            rho = torch.sqrt(xx.square()+yy.square()) / math.sqrt(.5)
            amp = 1 + .5*torch.tanh(self.radial_raw)*rho.pow(.7)
            amp = amp[None, None].expand(h_hat.shape[0], self.width, -1, -1)
            phase = torch.zeros_like(amp)
        else:
            c = self.align(condition.float())
            if padding:
                c = F.pad(c, (0, padding, 0, padding))
            c_hat = retained(torch.fft.rfft2(c), m1, m2)
            # Per-sample/per-channel scale normalization, shared across modes.
            hs = torch.sqrt(h_hat.abs().square().mean((-2,-1),keepdim=True)+self.eps)
            cs = torch.sqrt(c_hat.abs().square().mean((-2,-1),keepdim=True)+self.eps)
            hn, cn = h_hat/hs, c_hat/cs
            features = [torch.log1p(cn.abs()), torch.log1p(hn.abs())]
            if self.variant == 'AP':
                q = cn*hn.conj()/torch.sqrt((cn.abs().square()+self.eps)*(hn.abs().square()+self.eps))
                features += [q.real, q.imag]
            time = self.time_features(time_embedding.float())[:, :, None, None]
            features += [coords.expand(h_hat.shape[0], -1, -1, -1),
                         time.expand(-1,-1,2*m1,m2)]
            hidden = self.shared(torch.cat(features, 1))
            amp = 1+self.eta*torch.tanh(self.amplitude_head(hidden))
            phase = self.beta*torch.tanh(self.phase_head(hidden)) if self.variant == 'AP' else torch.zeros_like(amp)
        amp = torch.where(active[None,None], amp, 1.)
        phase = torch.where(active[None,None], phase, 0.)
        gate = torch.complex(amp*torch.cos(phase), amp*torch.sin(phase))
        if self.capture:
            with torch.no_grad():
                a, p = amp[..., active], phase[..., active]
                self.diagnostic = dict(amplitude_mean=float(a.mean()), amplitude_std=float(a.std(unbiased=False)),
                    amplitude_min=float(a.min()), amplitude_max=float(a.max()), phase_abs_mean=float(p.abs().mean()),
                    phase_abs_max=float(p.abs().max()), amplitude_saturation=float(((a-1).abs()>.49).float().mean()),
                    phase_saturation=float((p.abs()>.98*self.beta).float().mean()),
                    active_modes=int(active.sum()), boundary_modes=int((~active).sum()))
                self.arrays = dict(amplitude=amp[0].detach().cpu().numpy(), phase=phase[0].detach().cpu().numpy(),
                                   ky=ky.cpu().numpy(), kx=kx.cpu().numpy())
        return gate


class GatedStage(AttentionConditionedFNOStage):
    def forward(self, value, condition, time_embedding):
        self._validate_inputs(value, condition, time_embedding)
        attended = self.attention(value, condition)
        hidden = self.lifting(attended)
        height, width = hidden.shape[-2:]
        if self._padding:
            hidden = F.pad(hidden, (0,self._padding,0,self._padding))
        s = self.spectral
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            full = hidden.float()
            h_hat = torch.fft.rfft2(full)
            compact = retained(h_hat, s.modes1, s.modes2)
            gate = self.spectral_gate(compact, condition, time_embedding,
                                      full.shape[-2:], (s.modes1,s.modes2), self._padding)
            out = torch.zeros(full.shape[0],s.out_channels,full.shape[-2],full.shape[-1]//2+1,
                              dtype=torch.cfloat,device=full.device)
            out[..., :s.modes1,:s.modes2] = torch.einsum('bixy,ioxy->boxy',
                h_hat[..., :s.modes1,:s.modes2],s.weights1)*gate[..., :s.modes1,:]
            out[..., -s.modes1:,:s.modes2] = torch.einsum('bixy,ioxy->boxy',
                h_hat[..., -s.modes1:,:s.modes2],s.weights2)*gate[..., s.modes1:,:]
            spectral = torch.fft.irfft2(out,s=full.shape[-2:]).to(hidden.dtype)
            if self.spectral_gate.capture:
                with torch.no_grad():
                    baseline = s(hidden)
                    delta = (spectral-baseline)[0].float()
                    self.spectral_gate.arrays['spatial_delta_rms'] = delta.square().mean(0).sqrt().cpu().numpy()
                    self.spectral_gate.diagnostic['relative_output_delta'] = float(
                        (spectral.float()-baseline.float()).norm()/(baseline.float().norm()+1e-8))
        temporal = self.time_projection(nonlinearity(time_embedding))[:,:,None,None]
        delta = F.gelu(spectral+self.local(hidden)+temporal)
        return attended+self.projection(delta[...,:height,:width])


class SpectralGateUNO2d(AttentionMultiscaleUNO2d):
    def __init__(self, variant='B', **kwargs):
        if variant not in VARIANTS:
            raise ValueError(variant)
        super().__init__(**kwargs)
        self.spectral_variant = variant
        if variant != 'B':
            # Keep common layer initialization and caller RNG identical to B.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(271828)
                for stage in [*self.encoder_stages,self.bottleneck,*self.decoder_stages]:
                    stage.__class__ = GatedStage
                    stage.spectral_gate = SpectralGate(variant, stage._embedding_channels,
                                                        stage._operator_width,stage._time_channels)

    def gates(self):
        return [(name,m) for name,m in self.named_modules() if isinstance(m,SpectralGate)]
