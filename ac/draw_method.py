"""Editable vector diagrams for the implemented A and C inference paths."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'],
                     'font.size': 10, 'pdf.fonttype': 42, 'ps.fonttype': 42})
OUT = Path(__file__).resolve().parent.parent / 'assets'
OUT.mkdir(exist_ok=True)
COLORS = {'input':'#eef3f8', 'env':'#dceef5', 'physics':'#fff1d9',
          'noise':'#e9f2dc', 'state':'#f8e9dd', 'flow':'#e9e5f7', 'out':'#dff1e7'}

def canvas(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set(xlim=(0, w), ylim=(0, h), aspect='equal')
    ax.axis('off')
    return fig, ax

def box(ax, x, y, w, h, text, color, fs=10):
    ax.add_patch(FancyBboxPatch((x,y), w,h, boxstyle='round,pad=0.06,rounding_size=0.11',
        facecolor=COLORS[color], edgecolor='#3d5263', linewidth=1.1))
    ax.text(x+w/2, y+h/2, text, ha='center', va='center', fontsize=fs, linespacing=1.25)

def arrow(ax, points):
    for i, (p, q) in enumerate(zip(points,points[1:])):
        ax.add_patch(FancyArrowPatch(p,q,arrowstyle='-|>' if i==len(points)-2 else '-',
            mutation_scale=10, linewidth=1.25, color='#334d61', connectionstyle='arc3',
            shrinkA=0, shrinkB=0))

def save(fig, name):
    for ext in ('pdf','png'):
        fig.savefig(OUT / f'{name}.{ext}', dpi=220, bbox_inches='tight', pad_inches=.08)
    plt.close(fig)

# Variant A: the known beam is a direct velocity condition, not source input.
fig, ax = canvas(16, 5.4)
ax.text(.3,5.05,'A  |  Beam-conditioned velocity', fontsize=14, fontweight='bold')
box(ax,.4,3.45,2.25,.85,'Multiscale Matérn\nsource $x_0^A$','noise')
box(ax,3.35,3.45,2.15,.85,'Current field $x_t$\nand time $t$','state')
box(ax,7.2,3.45,2.7,.85,'AP-gated U-shaped\nFourier velocity','flow')
box(ax,10.6,3.45,2.0,.85,'Euler integration\n8 steps','flow')
box(ax,13.3,3.45,2.15,.85,'Predicted\nradio map','out')
box(ax,.4,1.35,2.25,.85,'Transmitter $T$\nheight map $H$','input')
box(ax,3.35,1.35,2.15,.85,'Target beam\nmap $B$','input')
box(ax,7.2,1.35,2.7,.85,'Raw condition + encoder\n$[T,H,B,0,0,0]$','env')
for p in [([(2.65,3.88),(3.35,3.88)]), ([(5.5,3.88),(7.2,3.88)]),
          ([(9.9,3.88),(10.6,3.88)]), ([(12.6,3.88),(13.3,3.88)]),
          ([(2.65,1.78),(2.95,1.78),(2.95,.9),(6.8,.9),(6.8,1.78),(7.2,1.78)]),
          ([(5.5,1.78),(7.2,1.78)]), ([(8.55,2.2),(8.55,3.45)]),
          ([(11.6,4.3),(11.6,4.65),(4.42,4.65),(4.42,4.3)])]: arrow(ax,p)
ax.text(.4,.35,'Training: straight flow path; masked velocity and Haar-detail losses.',fontsize=10,color='#3d5263')
save(fig,'method_a')

# Variant C: direct scene condition to velocity; beam enters only the source.
fig, ax = canvas(19, 8.6)
ax.text(.3,8.22,'C  |  Structured source and beam-conditioned covariance',fontsize=14,fontweight='bold')
box(ax,.35,6.65,2.15,.85,'Scene $[T,H,F]$','input')
box(ax,3.05,6.65,2.35,.85,'Environmental\nAP-UNO','env')
box(ax,5.95,6.65,2.15,.85,'Four complex\nlatent bases $G_k$','env')
box(ax,8.9,6.65,2.2,.85,'Coherent field\n$U=\\sum_k c_kG_k$','physics')
box(ax,11.8,6.65,2.0,.85,'Power mean\n$\\mu$','physics')
box(ax,.35,4.8,2.15,.85,'Array $n$, frequency $f$,\nangle $\\theta$','input')
box(ax,3.05,4.8,2.35,.85,'Analytic excitation\ncoefficients $c_k$','physics')
box(ax,.35,2.9,2.15,.85,'Target beam\nmap $B$','input')
box(ax,3.05,2.9,2.35,.85,'Beam score and\nprobability $q(B)$','noise')
box(ax,5.95,2.9,2.15,.85,'Weighted Matérn\nnoise + variance','noise')
box(ax,8.9,2.9,2.2,.85,'Guided noise\n$z_{\\rm guide}$','noise')
box(ax,.35,1.05,2.15,.85,'Independent\nMatérn $z_{\\rm MS}$','noise')
box(ax,11.8,2.9,2.0,.85,'Initial field\n$x_0^C$','state')
box(ax,14.5,4.25,2.15,.95,'AP-gated U-shaped\nFourier velocity','flow')
box(ax,14.5,2.55,2.15,.85,'Euler integration\n8 steps','flow')
box(ax,14.5,.85,2.15,.85,'Predicted\nradio map','out')
for p in [([(2.5,7.08),(3.05,7.08)]), ([(5.4,7.08),(5.95,7.08)]),
          ([(8.1,7.08),(8.9,7.08)]), ([(11.1,7.08),(11.8,7.08)]),
          ([(2.5,5.23),(3.05,5.23)]), ([(5.4,5.23),(10,5.23),(10,6.65)]),
          ([(2.5,3.33),(3.05,3.33)]), ([(5.4,3.33),(5.95,3.33)]),
          ([(8.1,3.33),(8.9,3.33)]), ([(11.1,3.33),(11.8,3.33)]),
          ([(2.5,1.48),(12.8,1.48),(12.8,2.9)]), ([(12.8,6.65),(12.8,3.75)]),
          ([(13.8,3.33),(14.05,3.33),(14.05,4.72),(14.5,4.72)]),
          ([(15.58,4.25),(15.58,3.4)]), ([(15.58,2.55),(15.58,1.7)]),
          ([(16.65,2.98),(17.0,2.98),(17.0,4.5),(16.65,4.5)]),
          ([(1.42,6.65),(1.42,6.05),(17.35,6.05),(17.35,4.72),(16.65,4.72)])]: arrow(ax,p)
ax.text(.35,.35,'No auxiliary or target radio map at inference. $F=\\log(f/6.7\\,\\mathrm{GHz})$.',
        fontsize=10,color='#3d5263')
save(fig,'method_c')
