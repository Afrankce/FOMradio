"""Localized source selection plus input/output-routed spectral experts.

The scalar Beam map is power, not a complex electromagnetic field. Any
spectral phase below is spatial-feature phase; FM time is not flight time.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from model.spectral_gate_uno import SpectralGateUNO2d


def amplitude_lift(beam_normalized, reference_db=-80.):
    # log(1 + relative amplitude); stable even for the full [-300, 0] dB range.
    return F.softplus((300 * beam_normalized.float() - 300 - reference_db) * math.log(10) / 20) / 10


class BeamPhaseGate(nn.Module):
    def __init__(self, width=16, phase=False, phase_input_only=False):
        super().__init__(); self.phase = phase
        # Supplementary input-only ablation: identical complex output heads in
        # both arms. Legacy experiments keep their original output policy.
        self.rotation_enabled = phase or phase_input_only
        self.capture = False; self.arrays = None
        self.align = nn.Conv2d(1, width, 1)
        self.shared = nn.Sequential(nn.Conv2d(4 * width, 32, 1), nn.GELU())
        self.amplitude = nn.Conv2d(32, width, 1)
        self.rotation = nn.Conv2d(32, width, 1)
        for head in (self.amplitude, self.rotation):
            nn.init.zeros_(head.weight); nn.init.zeros_(head.bias)

    def forward(self, compact, beam, pad, rows, modes):
        b = self.align(beam.float())
        bf = torch.fft.rfft2(F.pad(b, (pad,) * 4), norm='ortho').index_select(-2, rows)[..., :modes]
        hn = compact / (compact.abs().square().mean((-2, -1), keepdim=True) + 1e-6).sqrt()
        bn = bf / (bf.abs().square().mean((-2, -1), keepdim=True) + 1e-6).sqrt()
        q = bn * hn.conj() / ((bn.abs().square() + 1e-6) * (hn.abs().square() + 1e-6)).sqrt()
        features = [torch.log1p(bn.abs()), torch.log1p(hn.abs()),
                    q.real if self.phase else torch.zeros_like(q.real),
                    q.imag if self.phase else torch.zeros_like(q.imag)]
        features = self.shared(torch.cat(features, 1))
        amp = 1 + .5 * self.amplitude(features).tanh()
        phase = math.pi / 4 * self.rotation(features).tanh() if self.rotation_enabled else torch.zeros_like(amp)
        # Leave the self-conjugate kx=0 column unchanged. No Nyquist mode retained.
        active = torch.arange(modes, device=compact.device)[None, None, None, :] > 0
        result = torch.where(active, torch.polar(amp, phase), torch.ones_like(compact))
        if self.capture:
            self.arrays = {k: v[0].detach().float().cpu().numpy() for k, v in dict(
                beam_log_magnitude=torch.log1p(bn.abs()), hidden_log_magnitude=torch.log1p(hn.abs()),
                relative_phase_real=q.real if self.phase else torch.zeros_like(q.real),
                relative_phase_imag=q.imag if self.phase else torch.zeros_like(q.imag),
                gate_magnitude=result.abs(), gate_rotation=torch.angle(result)).items()}
        return result


class BoundaryResidual(nn.Module):
    def __init__(self, arm, phase=None, channels=128, condition_channels=64,
                 width=16, experts=4, modes=8, phase_input_only=False):
        super().__init__(); self.arm = arm; self.width = width
        self.experts = experts; self.modes = modes; self.capture = False; self.arrays = None
        self.input_projection = nn.Conv2d(channels, width, 1)
        self.output_projection = nn.Conv2d(width, channels, 1, bias=False)
        nn.init.zeros_(self.output_projection.weight)
        self.source_gate = nn.Sequential(nn.Conv2d(7, 24, 3, padding=1), nn.GELU(),
                                         nn.Conv2d(24, 24, 3, padding=1), nn.GELU(),
                                         nn.Conv2d(24, 1, 1))
        self.condition_projection = nn.Conv2d(condition_channels, 16, 1)
        self.time_projection = nn.Linear(512, 8)
        self.router = nn.Sequential(nn.Conv2d(width + 16 + 8, 32, 1), nn.GELU(),
                                    nn.Conv2d(32, 2 * experts, 1))
        if arm == 'E4':
            # 9*w*j + j*w = 61440 real parameters/expert at w=16,j=384,
            # exactly matching the stored real Fourier parameters at modes=8.
            hidden = (width * width * (2 * modes - 1) * modes * 2) // (10 * width)
            self.local_experts = nn.ModuleList(nn.Sequential(
                nn.Conv2d(width, hidden, 3, padding=1, bias=False),
                nn.Conv2d(hidden, width, 1, bias=False)) for _ in range(experts))
        else:
            self.weights = nn.Parameter(torch.randn(experts, width, width, 2 * modes - 1, modes, 2) / width)
        self.beam_spectral = BeamPhaseGate(width, phase == 'P-AP', phase_input_only) if phase else None

    def source_weights(self, condition, spatial):
        # Boundary support is derived from the high-resolution input, then pooled.
        height = condition[:, 1:2].float()
        dx = F.pad(height[..., 1:] - height[..., :-1], (0, 1, 0, 0))
        dy = F.pad(height[..., 1:, :] - height[..., :-1, :], (0, 0, 0, 1))
        boundary = ((dx.abs() + dy.abs()) * (445. / 3.)).clamp(0, 1)
        boundary = F.adaptive_max_pool2d(boundary, spatial)
        c = F.interpolate(condition.float(), size=spatial, mode='bilinear', align_corners=False)
        tx = F.adaptive_max_pool2d(condition[:, :1].float(), spatial)
        # Coordinates relative to the recorded Tx location (no target information).
        native = condition.shape[-1]; flat = condition[:, 0].flatten(1).argmax(1)
        xr = flat.remainder(native).float() / max(native - 1, 1)
        yr = torch.div(flat, native, rounding_mode='floor').float() / max(native - 1, 1)
        yy, xx = torch.meshgrid(torch.linspace(0, 1, spatial[0], device=c.device),
                                torch.linspace(0, 1, spatial[1], device=c.device), indexing='ij')
        relative_x = xx[None, None] - xr[:, None, None, None]
        relative_y = yy[None, None] - yr[:, None, None, None]
        beam = c[:, 2:3]; amp = amplitude_lift(beam)
        if self.arm == 'E3': beam = torch.zeros_like(beam); amp = torch.zeros_like(amp)
        if self.arm == 'E5': amp = torch.zeros_like(amp)
        features = torch.cat((c[:, 1:2], tx, beam, amp, boundary, relative_x, relative_y), 1)
        gate = boundary * self.source_gate(features).sigmoid()
        return gate.mean((-2, -1), keepdim=True).expand_as(gate) if self.arm == 'E1' else gate

    def spectral_weights(self):
        weights = torch.view_as_complex(self.weights.contiguous())
        first = .5 * (weights[..., :1] + weights[..., :1].flip(-2).conj())
        return torch.cat((first, weights[..., 1:]), -1)

    def forward(self, h, embedded, time_embedding, condition):
        z = self.input_projection(h)
        gate = self.source_weights(condition, h.shape[-2:]).to(z.dtype)
        time = self.time_projection(time_embedding)[:, :, None, None].expand(-1, -1, *h.shape[-2:])
        pi, alpha = self.router(torch.cat((z, self.condition_projection(embedded), time), 1)).chunk(2, 1)
        pi, alpha = pi.softmax(1), alpha.softmax(1)
        sources = z[:, None] * (gate * pi)[:, :, None]
        if self.arm == 'E4':
            propagated = torch.stack([expert(sources[:, m]) for m, expert in enumerate(self.local_experts)], 1)
        else:
            height, width = h.shape[-2:]; pad = height // 4; modes = self.modes
            with torch.autocast(device_type=h.device.type, enabled=False):
                sf = torch.fft.rfft2(F.pad(sources.float(), (pad,) * 4), norm='ortho')
                rows = torch.arange(-modes + 1, modes, device=h.device).remainder(sf.shape[-2])
                compact = sf.index_select(-2, rows)[..., :modes]
                filtered = torch.einsum('bmiyx,mioyx->bmoyx', compact, self.spectral_weights())
                if self.beam_spectral is not None:
                    self.beam_spectral.capture = self.capture
                    zf = torch.fft.rfft2(F.pad(z.float(), (pad,) * 4), norm='ortho').index_select(-2, rows)[..., :modes]
                    beam = F.interpolate(condition[:, 2:3].float(), size=(height, width), mode='bilinear', align_corners=False)
                    filtered = filtered * self.beam_spectral(zf, beam, pad, rows, modes)[:, None]
                output = torch.zeros_like(sf)
                output[..., rows, :modes] = filtered
                propagated = torch.fft.irfft2(output, s=(height + 2 * pad, width + 2 * pad), norm='ortho')
                propagated = propagated[..., pad:pad + height, pad:pad + width].to(z.dtype)
        combined = (alpha[:, :, None] * propagated).sum(1)
        result = self.output_projection(combined)
        if self.capture:
            self.arrays = {k: v[0].detach().float().cpu().numpy() for k, v in dict(
                source_gate=gate, pi=pi, alpha=alpha, localized_source_rms=sources.float().square().mean((1, 2)).sqrt()[:, None],
                propagated_rms=propagated.float().square().mean((1, 2)).sqrt()[:, None],
                residual_rms=result.float().square().mean(1, keepdim=True).sqrt()).items()}
            if self.beam_spectral is not None:
                self.arrays.update({f'beam_spectral_{k}': v for k, v in self.beam_spectral.arrays.items()})
        return result


class BoundaryUNO(SpectralGateUNO2d):
    def __init__(self, arm='E0', phase_base='E2', phase_input_only=False):
        super().__init__('AP', operator_width=32, cfg_drop_prob=0.)
        self.arm = arm
        if arm != 'E0':
            base, phase = (phase_base, arm) if arm in ('P-A', 'P-AP') else (arm, None)
            if base not in ('E1', 'E2', 'E3', 'E4', 'E5') or (phase and base == 'E4'):
                raise ValueError((arm, base))
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(314159)
                self.boundary_encoder = BoundaryResidual(base, phase, phase_input_only=phase_input_only)
                torch.manual_seed(314160)
                self.boundary_decoder = BoundaryResidual(base, phase, phase_input_only=phase_input_only)

    def _velocity(self, condition, state, step, embeddings=None):
        if self.arm == 'E0': return super()._velocity(condition, state, step, embeddings)
        self._validate_state_and_condition(condition, state)
        step = self._normalize_step(step, state)
        embeddings = self._validate_embeddings(self.embed_model(condition) if embeddings is None else embeddings, state)
        gx, gy = self.coordinate_grid(state)
        value = self.lifting(torch.cat((state, condition, gx, gy), 1)); time = self._time_embedding(step)
        skips = []
        for i, (stage, downsample) in enumerate(zip(self.encoder_stages, self.downsamples)):
            residual = self.boundary_encoder(value, embeddings[i], time, condition) if i == 2 else None
            value = stage(value, embeddings[i], time)
            if residual is not None: value = value + residual
            skips.append(value); value = downsample(value)
        value = self.bottleneck(value, embeddings[4], time)
        for i, level in enumerate(reversed(range(4))):
            value = self.upsample_fusions[i](value, skips[level])
            residual = self.boundary_decoder(value, embeddings[level], time, condition) if level == 2 else None
            value = self.decoder_stages[i](value, embeddings[level], time)
            if residual is not None: value = value + residual
        return self.projection_output(F.gelu(self.projection_hidden(value)))
