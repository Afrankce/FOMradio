"""Editable, conceptual method illustrations of the implemented A/C models.

Maps and feature planes are schematic, not measured experimental images.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, FancyArrowPatch, Circle, Arc

plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],
 'font.size':8.4,'mathtext.fontset':'stix','pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none'})
OUT=Path(__file__).resolve().parent.parent/'assets'; OUT.mkdir(exist_ok=True)
INK='#203747'; MUTED='#536c7b'; BLUE='#377b9c'; GOLD='#c5833b'; GREEN='#65946d'; PURPLE='#75638d'

def setup(h=7.0):
    fig,ax=plt.subplots(figsize=(8.2,8.2*h/14)); ax.set_xlim(0,14);ax.set_ylim(0,h)
    ax.set_aspect('equal');ax.axis('off');fig.subplots_adjust(0,0,1,1)
    return fig,ax

def text(ax,x,y,s,**kw):
    a=dict(color=INK,ha='center',va='center',fontsize=8.4,zorder=8,linespacing=1.12);a.update(kw)
    ax.text(x,y,s,**a)

def route(ax,pts,color=INK,lw=1.05,dashed=False,head=True):
    for i,(a,b) in enumerate(zip(pts,pts[1:])):
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>' if head and i==len(pts)-2 else '-',
          mutation_scale=8,linewidth=lw,color=color,linestyle=(0,(2,2)) if dashed else '-',
          shrinkA=0,shrinkB=0,zorder=5))

def sheet(ax,x,y,w,h,fill='#deedf3',edge=BLUE,lean=.23,alpha=1,detail=True):
    p=np.array([[x,y],[x+w,y+lean],[x+w,y+h+lean],[x,y+h]])
    ax.add_patch(Polygon(p,closed=True,facecolor=fill,edgecolor=edge,linewidth=.8,alpha=alpha,zorder=2))
    if detail:
        for t in (.23,.46,.69):
            ax.plot([x+w*t,x+w*t],[y+lean*t,y+h+lean*t],color=edge,alpha=.17,lw=.4,zorder=3)
        for t in (.25,.50,.75):
            ax.plot([x,x+w],[y+h*t,y+h*t+lean],color=edge,alpha=.17,lw=.4,zorder=3)
    return p

def channel_stack(ax,x,y,kind='A'):
    colors=('#d8e8f0','#e5e0d4','#d6e9d2') if kind=='A' else ('#d8e8f0','#e5e0d4','#e4e6f4')
    for i,c in enumerate(colors): sheet(ax,x,y+i*.30,1.45,.43,fill=c,edge=BLUE,lean=.15)
    # One visual stack is one channel-concatenated tensor.
    labels=('$B$','$H$','$T$') if kind=='A' else ('$F$','$H$','$T$')
    for i,s in enumerate(labels):text(ax,x-.12,y+i*.30+.23,s,ha='right',fontsize=8)
    text(ax,x+.75,y-.25,'joint condition',fontweight='bold',fontsize=8.5)
    text(ax,x+.75,y-.48,r'$[T,H,B,0,0,0]$' if kind=='A' else r'$[T,H,F,0,0,0]$',fontsize=8)

def mini_encoder(ax,x,y,caption='condition encoder'):
    for i in range(3):sheet(ax,x+i*.24,y-i*.14,.30,.75-i*.14,fill='#d8eaf2',edge=BLUE,lean=.10,detail=False)
    text(ax,x+.36,y-.43,caption,fontsize=7.8,color=BLUE)

def uno(ax,x,y,scale=1):
    xs=np.arange(9)*.43*scale
    heights=np.array([1.28,1.12,.95,.80,.65,.80,.95,1.12,1.28])*scale
    centers=y+np.array([.36,.17,-.02,-.21,-.40,-.21,-.02,.17,.36])*scale
    for i,(dx,h,cy) in enumerate(zip(xs,heights,centers)):
        sheet(ax,x+dx,cy-h/2,.18*scale,h,fill='#e7e0f1',edge=PURPLE,lean=.12*scale,detail=False)
        if i: route(ax,[(x+xs[i-1]+.18*scale,centers[i-1]),(x+dx,cy)],color=PURPLE,lw=.65)
    for i in range(4):
        left=x+xs[i]+.09*scale;right=x+xs[8-i]+.09*scale
        yy=y+(1.12-i*.15)*scale
        ax.add_patch(FancyArrowPatch((left,centers[i]+heights[i]/2+.06*scale),
          (right,centers[8-i]+heights[8-i]/2+.06*scale),arrowstyle='->',
          connectionstyle=f'arc3,rad={-.12-i*.035}',mutation_scale=6,color=PURPLE,alpha=.8,lw=.65,zorder=4))
    text(ax,x+1.8*scale,y-1.15*scale,'AP-gated U-shaped FNO',fontweight='bold',fontsize=8.5*scale)

def noise(ax,x,y,color=GREEN):
    sheet(ax,x,y,1.25,.65,fill='#e7f0e1',edge=color,lean=.12,detail=False)
    for k in range(3):
        xx=np.linspace(0,1.1,60); yy=.33+.11*np.sin(10*xx+1.2*k)+.08*k
        ax.plot(x+.07+xx,y+yy+xx*.10,color=color,lw=.55,alpha=.72,zorder=4)

def result_tile(ax,x,y):
    sheet(ax,x,y,1.12,1.04,fill='#d9eaf1',edge=BLUE,lean=.13,detail=True)
    for i in range(3):
        ax.plot([x+.1,x+.38,x+.62,x+.92],[y+.2+i*.18,y+.35+i*.13,y+.20+i*.15,y+.45+i*.14],
                color=BLUE,lw=.65,alpha=.7,zorder=4)

def save(fig,name):
    for ext in ('pdf','svg','png'):
        fig.savefig(OUT/f'{name}.{ext}',dpi=240,bbox_inches='tight',pad_inches=.04,facecolor='white')
    plt.close(fig)

# A: all three modalities are concatenated before both condition uses.
fig,ax=setup(7.0)
channel_stack(ax,.38,4.58,'A');mini_encoder(ax,2.75,5.08)
route(ax,[(1.92,5.26),(2.62,5.26)],color=BLUE)
route(ax,[(3.65,5.26),(4.62,5.26),(4.62,4.70)],color=BLUE)
text(ax,4.17,5.48,'multi-scale features',fontsize=7.2,color=BLUE)
route(ax,[(1.15,4.55),(1.15,3.80),(3.15,3.80),(3.15,2.77)],color=BLUE)
text(ax,1.52,3.60,'raw condition',ha='left',fontsize=7.0,color=BLUE)
noise(ax,.48,2.08);text(ax,1.1,1.83,r'$x_0\sim\mathcal{GP}(0,C_{\rm MS})$',fontsize=8)
route(ax,[(1.78,2.47),(2.52,2.47)])
text(ax,2.90,2.52,r'$[x_t,c_A,p]$',fontsize=9.3)
route(ax,[(3.38,2.52),(4.25,2.52),(4.25,3.12)])
text(ax,3.00,3.15,'lifting',fontsize=7.2,color=MUTED)
uno(ax,4.5,3.55,1.0)
route(ax,[(8.15,3.92),(8.8,3.92)])
text(ax,9.17,3.93,r'$v_\theta$',fontsize=11)
route(ax,[(9.58,3.92),(10.45,3.92)])
text(ax,10.02,4.18,'Euler × 8',fontsize=8)
result_tile(ax,10.55,3.37);text(ax,11.12,3.02,r'predicted map $\hat y$',fontsize=8)
route(ax,[(10.25,3.92),(10.25,5.95),(4.00,5.95),(4.00,3.12),(3.30,3.12),(3.30,2.77)],color=PURPLE,dashed=True)
text(ax,8.38,6.15,r'evolving state $x_t$',fontsize=7.5,color=PURPLE)
text(ax,4.60,1.45,r'FFT  $\rightarrow\ (W_k\widehat h)\,A_k e^{i\phi_k}\ \rightarrow$  inverse FFT',ha='left',fontsize=9)
text(ax,4.62,1.15,'amplitude–phase spectral gate (per stage)',ha='left',fontsize=7.5,color=PURPLE)
text(ax,3.90,1.78,r'$t$',fontsize=10);route(ax,[(4.02,1.83),(4.32,1.83),(4.32,3.20),(4.50,3.20)],color=GOLD)
save(fig,'method_a')

# C: separate deterministic mean, target-beam noise, and shared flow field.
fig,ax=setup(8.35)
channel_stack(ax,.30,6.35,'C');mini_encoder(ax,2.85,6.70,'environment AP-UNO')
route(ax,[(1.84,7.00),(2.72,7.00)],color=BLUE)
for k in range(4):sheet(ax,4.28+.18*k,6.57+.07*k,.52,.72,fill='#e2edf4',edge=BLUE,lean=.08,detail=False)
text(ax,4.84,5.98,r'complex bases $G_1,\ldots,G_4$',fontsize=7.7,color=BLUE)
route(ax,[(3.76,7.00),(4.17,7.00)],color=BLUE)
route(ax,[(5.42,7.00),(6.07,7.00),(6.07,6.22)],color=BLUE)
text(ax,6.13,5.93,r'$\sum_k c_kG_k$',fontsize=10.5)
route(ax,[(6.84,5.93),(7.65,5.93)])
text(ax,8.24,5.93,r'$\mu\!:\ |\cdot|^2\!\to\!\log$',fontsize=9.2,color=GOLD)
route(ax,[(8.94,5.93),(9.40,5.93),(9.40,4.38)],color=GOLD)
for i in range(4):
    for j in range(4):ax.add_patch(Circle((.58+j*.13,4.30+i*.13),.035,facecolor=GOLD,edgecolor='none',zorder=3))
text(ax,1.1,4.92,r'array $(n,f,\theta)$',ha='left',fontsize=8,color=GOLD)
route(ax,[(1.15,4.50),(2.16,4.50)],color=GOLD)
text(ax,3.10,4.50,r'analytic coefficients $c_k$',fontsize=8.4,color=GOLD)
route(ax,[(4.10,4.50),(6.07,4.50),(6.07,5.66)],color=GOLD)
sheet(ax,.38,2.60,1.30,.58,fill='#e2f0df',edge=GREEN,lean=.13)
text(ax,1.05,2.34,r'target beam $B$',fontsize=8,color=GREEN)
route(ax,[(1.76,3.0),(2.35,3.0)],color=GREEN)
text(ax,3.25,3.0,r'spatial weights $q(B,\nabla B)$',fontsize=8.3,color=GREEN)
route(ax,[(4.42,3.0),(4.90,3.0)],color=GREEN)
noise(ax,5.05,2.60);text(ax,5.67,2.32,r'weighted Matérn $z_{\rm guide}$',fontsize=7.7,color=GREEN)
route(ax,[(6.35,3.0),(8.87,3.0),(8.87,4.06),(9.1,4.06)],color=GREEN)
noise(ax,5.05,.91);text(ax,5.67,.65,r'independent $z_{\rm MS}$',fontsize=7.8,color=GREEN)
route(ax,[(6.35,1.30),(9.42,1.30),(9.42,3.76)],color=GREEN)
text(ax,9.42,4.06,r'$\oplus$',fontsize=18)
text(ax,10.60,1.00,r'$x_0=\mu+\sqrt{.75}z_{\rm MS}+.5z_{\rm guide}$',fontsize=8.0)
route(ax,[(9.80,4.06),(10.12,4.06)])
uno(ax,10.20,4.0,.62)
text(ax,11.43,2.90,r'$\dot x_t=v_\theta(x_t,t,c_C)$',fontsize=7.7,color=PURPLE)
route(ax,[(12.55,4.23),(13.22,4.23),(13.22,3.80)])
result_tile(ax,12.79,2.62);text(ax,13.36,2.30,r'$\hat y$ (Euler × 8)',fontsize=7.9)
route(ax,[(1.10,7.62),(1.10,8.13),(10.30,8.13),(10.30,4.72)],color=BLUE,dashed=True)
text(ax,8.35,8.28,r'joint condition $c_C$ to velocity',fontsize=7.2,color=BLUE)
save(fig,'method_c')
