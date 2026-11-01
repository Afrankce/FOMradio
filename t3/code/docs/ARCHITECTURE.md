# Architecture and data flow

## 1. Inputs, target, and valid region

The principal tensors in each batch are:

| Symbol | Shape | Meaning |
|---|---|---|
| `c` | `[B,3,256,256]` | Tx mask, normalized height, normalized beam map |
| `Y=x_1` | `[B,1,256,256]` | Target radio map; valid dB values mapped linearly to `[0,1]` |
| `M` | `[B,1,256,256]` | `valid_mask`, selecting real propagation pixels |
| `x_0` | `[B,1,256,256]` | Gaussian noise with the target shape |
| `t` | `[B]` | Continuous time sampled independently per sample from `U(0,1)` |
| `x_t` | `[B,1,256,256]` | Intermediate state between noise and target |

The original height is 256x256; beam and radio maps are 128x128. Loading resizes them to 256x256 using bilinear interpolation and the applicable continuous/discrete rules. Target values `-300` and `1000` are floor/building sentinels and are excluded from valid-region loss.

## 2. Conditional flow matching

The implementation uses `ConditionalFlowMatcher(sigma=0.0)`, so the training path is a deterministic straight-line interpolation:

~~~math
x_t=(1-t)x_0+tY,\qquad u_t=Y-x_0.
~~~

The velocity network learns:

~~~math
v_\theta(x_t,t,c)\approx u_t.
~~~

Loss is computed only on valid pixels:

~~~math
\mathcal L=
\frac{\sum M\odot\lVert v_\theta(x_t,t,c)-u_t\rVert_2^2}
     {\sum M}.
~~~

`M` denotes the radio-map valid-region mask, not the number of FNO frequency modes `m`. Each forward pass samples one `t` per batch item; random samples across batches and epochs cover `[0,1]`. This is not equivalent to learning one fixed time.

## 3. Condition encoder (blue path)

`BasicUNetEncoder` produces five spatial scales from `c`:

~~~text
e0: [B,  32,256,256]
e1: [B,  32,128,128]
e2: [B,  64, 64, 64]
e3: [B, 128, 32, 32]
e4: [B, 256, 16, 16]
~~~

These are condition features, not flow-matching states. Each `e_i` is sent only to encoder/decoder FNO stages at the same resolution, while `e_4` enters the bottleneck. The condition encoder extracts multiscale features; a separate U-shaped state path propagates the current flow-matching state.

## 4. State lifting and coordinate grid

The model generates two deterministic coordinate channels:

~~~math
g_x(r,s)=s/(W-1),\qquad g_y(r,s)=r/(H-1).
~~~

It then concatenates them per pixel:

~~~math
q=[x_t,c,g_x,g_y]\in\mathbb R^{B\times6\times256\times256},
\qquad z_0=P_{lift}(q)\in\mathbb R^{B\times32\times256\times256}.
~~~

`P_lift` is a 1x1 convolution: it shares weights across pixels, mixes channels, and performs no spatial convolution. Coordinate channels distinguish absolute locations explicitly, instead of treating all locations as equivalent under a purely translation-equivariant spectral operator.

## 5. U-shaped multiscale FNO state backbone (green path)

State channels and resolutions are:

~~~text
Enc0  32 @ 256x256  --down-->  Enc1  64 @ 128x128
  |                                  |
skip0                              skip1
  |                                  |
Dec0  32 @ 256x256  <--up--    Dec1  64 @ 128x128

Enc2 128 @ 64x64 --down--> Enc3 256 @ 32x32 --down--> Bottleneck 256 @ 16x16
  |                         |                                  |
skip2                     skip3                               up
  |                         |                                  |
Dec2 128 @ 64x64 <--up-- Dec3 256 @ 32x32 <-------------------+
~~~

Each downsampling operation is `AvgPool2d(2,2)` followed by a 1x1 convolution. Each upsampling operation resizes bilinearly to the matching encoder skip, concatenates along channels, and compresses with a 1x1 convolution. Thus encoder and decoder each have an FNO stage at every level and four actual U-Net skip connections. The full model has 4+1+4=9 FNO stages, not four full-resolution blocks in series.

## 6. Attention-conditioned FNO stage

Each stage receives a same-scale state `z_i`, condition feature `e_i`, and global time embedding `tau(t)`:

~~~text
z_i: [B,C_i,H_i,W_i]
e_i: [B,E_i,H_i,W_i]
tau: [B,512]
~~~

### 6.1 CA/SA condition fusion

The code names a class `CrossAttention`, but its actual implementation is **not** Transformer Q/K/V attention. It combines channel attention, spatial attention, and a conditional 1x1 projection:

~~~math
CA(z)=\sigma(MLP(GAP(z))+MLP(GMP(z))),
~~~

~~~math
SA(z)=\sigma(Conv_{7\times7}([Mean_c(z),Max_c(z)])),
~~~

~~~math
a_i=z_i\odot CA(z_i)+z_i\odot SA(z_i)+P_i(e_i).
~~~

`P_i(e_i)` projects condition channels to `C_i`. Conditions do not generate Q/K/V; they enter the state as an additive same-scale residual. CA/SA weights themselves are computed from `z_i`.

### 6.2 Spectral, local, and time branches

First, `a_i` is projected per pixel to a fixed operator width of 24:

~~~math
h_i=L_i(a_i)\in\mathbb R^{B\times24\times H_i\times W_i}.
~~~

The stage update is:

~~~math
\Delta_i=GELU\left(
\mathcal F^{-1}(R_i\,\mathcal F(h_i))+W_i h_i+T_i(\tau(t))
\right),
~~~

~~~math
z_i^{out}=a_i+Q_i(\Delta_i).
~~~

- Spectral branch: `rfft2 -> low-frequency selection -> learnable complex weights -> irfft2`.
- Local branch: 1x1 convolution for location-wise channel mixing.
- Time branch: `t -> sinusoidal embedding(128) -> MLP(512) -> Linear(C=24)`, reshaped to `[B,24,1,1]` and broadcast spatially.
- Residual: project the 24-channel delta back to `C_i` and add it to `a_i`.

Time enters every stage so that velocity estimates at each scale depend explicitly on the current ODE time. A single injection at the network input could be overwhelmed after repeated transformations.

### 6.3 Meaning of `m=12/8/4`

`m` is the number of low-frequency Fourier modes retained at each scale, not the number of layers or channels:

| Scale | Resolution | Modes `m` | Padding |
|---|---:|---:|---:|
| level 0 | 256x256 | 12 | 9 |
| level 1 | 128x128 | 12 | 5 |
| level 2 | 64x64 | 8 | 3 |
| level 3 | 32x32 | 4 | 2 |
| level 4 | 16x16 | 4 | 1 |

The two-dimensional real FFT implementation learns separate upper and lower vertical frequency blocks and retains the first `m` horizontal rFFT frequencies. Deeper scales have smaller grids and therefore use fewer modes.

## 7. Output and two-step Euler integration

The final 256x256 decoder state passes through `1x1:32->128`, GELU, and `1x1:128->1` to produce a velocity field. Inference starts from fixed noise `x^(0)` and uses `K=2`, `Delta t=1/2`:

~~~math
t_k=k/K,\qquad
x^{(k+1)}=x^{(k)}+\frac{1}{K}v_{CFG}(x^{(k)},t_k,c),
\quad k=0,1.
~~~

The network is called at `t=0` and `t=0.5`; `x^(2)` is the predicted map. General CFG is:

~~~math
v_{CFG}=v_{uncond}+\gamma(v_{cond}-v_{uncond}).
~~~

This experiment fixes `gamma=1.0`. The implementation directly returns the conditional branch, numerically equivalent to purely conditional velocity. The two steps are ODE integration steps at inference, distinct from continuous-time sampling of one `t` per sample during training.

## 8. Differences from the original U-Net velocity backbone

- Retained: conditional `BasicUNetEncoder`, multiscale conditions, CA/SA fusion, CFM objective, EMA, CFG, and Euler sampling.
- Changed: 3x3 CNN stages in the state-velocity backbone become spectral-plus-local FNO stages.
- Downsampling: average pooling plus a 1x1 projection.
- Upsampling: bilinear resize, skip concatenation, and a 1x1 projection.
- Four encoder-to-decoder skips remain; this is not merely a replacement of one `TwoConv`.

Implementation entry point: `model/attention_multiscale_uno.py`.
