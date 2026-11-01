"""Editable, code-generated diagram of the implemented C architecture."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],
                     'font.size':11,'pdf.fonttype':42,'ps.fonttype':42})
fig,ax=plt.subplots(figsize=(15,6.5));ax.set(xlim=(0,15),ylim=(0,6.5));ax.axis('off')
def box(x,y,w,h,label,color):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.08,rounding_size=0.13',
                               facecolor=color,edgecolor='#315570',linewidth=1.3))
    ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=10.5)
def arrow(start,end):
    ax.add_patch(FancyArrowPatch(start,end,arrowstyle='-|>',mutation_scale=12,
                                 color='#28475d',linewidth=1.5,connectionstyle='arc3'))
box(.3,4.8,2.1,.7,'Scene geometry H\nTx, frequency f','#edf3f8')
box(3.0,4.8,2.1,.7,'Environmental\nFNO + AP','#d9edf5')
box(5.7,4.8,2.0,.7,'Latent complex\nbases $G_k$','#d9edf5')
box(.3,3.3,2.1,.7,'Array size, frequency\nsteering angle','#fff3df')
box(3.0,3.3,2.1,.7,'Analytic excitation\ncoefficients $c_k$','#fff3df')
box(8.5,4.0,2.1,.85,'Coherent sum\n$u=\\sum_k G_k c_k$','#ece8fa')
box(11.4,4.0,2.2,.85,'Power mean $\\mu$\n(normalized dB)','#ece8fa')
box(.3,1.3,2.1,.7,'Target beam map B','#edf4df')
box(3.0,1.3,2.1,.7,'Spatial score\n$q(B)$','#edf4df')
box(5.7,1.3,2.0,.7,'Beam-conditioned\nMatérn noise $z_B$','#edf4df')
box(8.5,1.3,2.1,.7,'Initial state\n$x_0=\\mu+z_B$','#f6e9df')
box(11.4,1.3,2.2,.7,'AP-gated FNO\nvelocity $v_\\theta$','#d9edf5')
box(11.4,.15,2.2,.55,'Predicted radio map','#e4f3ed')
for a,b in [((2.4,5.15),(3,5.15)),((5.1,5.15),(5.7,5.15)),
            ((7.7,5.15),(8.5,4.65)),((2.4,3.65),(3,3.65)),
            ((5.1,3.65),(8.5,4.15)),((10.6,4.43),(11.4,4.43)),
            ((2.4,1.65),(3,1.65)),((5.1,1.65),(5.7,1.65)),
            ((7.7,1.65),(8.5,1.65)),((12.5,4.0),(10.6,1.9)),
            ((10.6,1.65),(11.4,1.65)),((12.5,1.3),(12.5,.7)),
            ((2.4,4.9),(11.4,2.05))]:arrow(a,b)
ax.text(.3,.25,'At inference: no auxiliary or target radio map.  $k=1,\\ldots,4$.',fontsize=10,color='#315570')
fig.tight_layout();out=Path(__file__).resolve().parent.parent/'assets';out.mkdir(exist_ok=True)
fig.savefig(out/'method_c.png',dpi=200,bbox_inches='tight');fig.savefig(out/'method_c.pdf',bbox_inches='tight');plt.close(fig)
