"""Animate transverse irradiance profiles as the beams propagate to 100 km."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import LogNorm
from scipy.special import j0, j1, airy

LAMBDA=1064e-9; K=2*np.pi/LAMBDA; P=1.0; W0=10e-3; ZR=np.pi*W0**2/LAMBDA
BESSEL_THETA=0.05e-3; BESSEL_KR=K*np.sin(BESSEL_THETA); BESSEL_CORE=2.405/BESSEL_KR; BESSEL_ZMAX=0.20/np.tan(BESSEL_THETA)
AIRY_X0=0.05; AIRY_A=0.08; AIRY_Z0=K*AIRY_X0**2; TOPHAT_R=W0
NAMES=['Bessel J0','Airy','Top-hat aperture','Laguerre-Gaussian LG01','Hermite-Gaussian HG10']
COLORS=['#2563eb','#ea580c','#16a34a','#9333ea','#0891b2']


def airy_field(x,z):
    s=x/AIRY_X0; xi=z/AIRY_Z0
    return airy(s-xi**2/4+1j*AIRY_A*xi)[0]*np.exp(AIRY_A*s-AIRY_A*xi**2/2+1j*(s*xi/2-xi**3/12+AIRY_A**2*xi/2))


def profiles(x,z):
    w=W0*np.sqrt(1+(z/ZR)**2); rho=np.abs(x)
    b=2*P/(np.pi*BESSEL_CORE**2)*j0(BESSEL_KR*rho)**2
    if z>BESSEL_ZMAX: b*=np.exp(-((z-BESSEL_ZMAX)/BESSEL_ZMAX)**2)
    a=P/(np.sqrt(2*np.pi)*AIRY_X0)*np.abs(airy_field(x,z))**2
    ztr=K*TOPHAT_R**2/2; q=K*TOPHAT_R*rho/max(z,1e-12); disk=np.ones_like(q); mask=q>1e-12; disk[mask]=(2*j1(q[mask])/q[mask])**2
    near=P/(np.pi*TOPHAT_R**2)*(rho<=TOPHAT_R); far=P/(np.pi*TOPHAT_R**2)*(ztr/max(z,1e-12))**2*disk; t=(1-np.clip(z/ztr,0,1))*near+np.clip(z/ztr,0,1)*far
    rho2=2*x**2/w**2; lg=4*P/(np.pi*w**2)*rho2*np.exp(-rho2); u=np.sqrt(2)*x/w; hg=2*(2*P/(np.pi*w**2))*u**2*np.exp(-u**2)
    return [b,a,t,lg,hg]


def beam_scales(z):
    # Visual characteristic half-widths, not a new physical definition of beam radius.
    w=W0*np.sqrt(1+(z/ZR)**2)
    return np.array([max(BESSEL_CORE,0.2 if z>BESSEL_ZMAX else BESSEL_CORE),
                     max(AIRY_X0, abs(z/AIRY_Z0)**2*AIRY_X0),
                     max(TOPHAT_R, 1.22*LAMBDA*max(z,1e-9)/(2*TOPHAT_R)),
                     w, w])


def main():
    outdir=Path(__file__).resolve().parent/'outputs_other_beams'; outdir.mkdir(exist_ok=True)
    zvals=np.linspace(0,100e3,41); x=np.linspace(-12,12,1000)
    fig=plt.figure(figsize=(15,10),constrained_layout=True)
    gs=fig.add_gridspec(3,2,height_ratios=[1,1,0.9])
    axes=[fig.add_subplot(gs[0,0]),fig.add_subplot(gs[0,1]),fig.add_subplot(gs[1,0]),fig.add_subplot(gs[1,1])]
    ax5=fig.add_subplot(gs[2,0]); ax6=fig.add_subplot(gs[2,1])
    axes.append(ax5)
    lines=[]
    for ax,name,color in zip(axes,NAMES,COLORS):
        line,=ax.plot([],[],color=color,lw=2,label=name); lines.append(line); ax.set_xlim(-12,12); ax.set_ylim(1e-10,2e4); ax.set_yscale('log'); ax.grid(alpha=.25); ax.set_title(name); ax.set_xlabel('transverse coordinate (m)'); ax.set_ylabel('I (W/m²)')
    ax6.set_xlim(0,100); ax6.set_ylim(0,40); ax6.set_xlabel('propagation distance (km)'); ax6.set_ylabel('characteristic width (m)'); ax6.grid(alpha=.3)
    scale_lines=[]
    for i,(name,color) in enumerate(zip(NAMES,COLORS)):
        sl,=ax6.plot([],[],color=color,lw=2,label=name); scale_lines.append(sl)
    marker=ax6.axvline(0,color='k',ls='--',lw=1.5); ax6.legend(fontsize=8,ncol=2,loc='upper left')
    title=fig.suptitle('Beam expansion during propagation: z = 0 km',fontsize=17)
    all_scales=np.array([beam_scales(z) for z in zvals])
    max_scale=max(12,np.max(all_scales)*1.05)
    for ax in axes: ax.set_xlim(-max_scale,max_scale)

    def update(i):
        z=zvals[i]; vals=profiles(x,z)
        for line,val in zip(lines,vals): line.set_data(x,val)
        for j,sl in enumerate(scale_lines): sl.set_data(zvals[:i+1]/1000,all_scales[:i+1,j])
        marker.set_xdata([z/1000,z/1000]); title.set_text(f'Beam expansion during propagation: z = {z/1000:.1f} km')
        return lines+scale_lines+[marker,title]
    ani=FuncAnimation(fig,update,frames=len(zvals),interval=150,blit=False)
    ani.save(outdir/'beam_expansion_animation.gif',writer=PillowWriter(fps=6))
    plt.close(fig)
    print(f'Wrote {outdir}/beam_expansion_animation.gif')

if __name__=='__main__': main()
