"""Plot all non-Gaussian beams at the initial plane z=0."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.special import j0, j1, airy

LAMBDA=1064e-9; K=2*np.pi/LAMBDA; P=1.0; W0=10e-3
BESSEL_THETA=0.05e-3; BESSEL_KR=K*np.sin(BESSEL_THETA); BESSEL_CORE=2.405/BESSEL_KR
AIRY_X0=0.05; AIRY_A=0.08; TOPHAT_R=W0
NAMES=['Bessel J0','Airy','Top-hat aperture','Laguerre-Gaussian LG01','Hermite-Gaussian HG10']
COLORS=['#2563eb','#ea580c','#16a34a','#9333ea','#0891b2']


def airy_field(x):
    s=x/AIRY_X0
    return airy(s)[0]*np.exp(AIRY_A*s)


def fields(X,Y):
    rho=np.sqrt(X**2+Y**2)
    b=2*P/(np.pi*BESSEL_CORE**2)*j0(BESSEL_KR*rho)**2
    a=P/(np.sqrt(2*np.pi)*AIRY_X0)*np.abs(airy_field(X)*airy_field(Y))**2
    t=P/(np.pi*TOPHAT_R**2)*(rho<=TOPHAT_R)
    rho2=2*rho**2/W0**2; lg=4*P/(np.pi*W0**2)*rho2*np.exp(-rho2)
    u=np.sqrt(2)*X/W0; hg=2*(2*P/(np.pi*W0**2))*u**2*np.exp(-u**2)
    return [b,a,t,lg,hg]


def main():
    outdir=Path(__file__).resolve().parent/'outputs_other_beams'; outdir.mkdir(exist_ok=True)
    axis=np.linspace(-0.15,0.15,700); X,Y=np.meshgrid(axis,axis); vals=fields(X,Y); center=[v[v.shape[0]//2,:] for v in vals]
    fig=plt.figure(figsize=(16,10),constrained_layout=True); gs=fig.add_gridspec(2,3,height_ratios=[1.15,0.85])
    for i,(name,val) in enumerate(zip(NAMES,vals)):
        ax=fig.add_subplot(gs[0, i] if i<3 else gs[1, i-3])
        im=ax.pcolormesh(axis,axis,np.maximum(val,1e-10),shading='auto',cmap='magma',norm=LogNorm(vmin=1e-10,vmax=max(1,float(np.max(val)))))
        ax.set_title(name); ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)'); ax.set_aspect('equal'); fig.colorbar(im,ax=ax,label='I (W/m², log scale)')
    # Replace unused lower-right slot with 1-D comparison.
    ax=fig.add_subplot(gs[1,2]);
    for name,c,prof in zip(NAMES,COLORS,center): ax.plot(axis*1000,prof,color=c,lw=2,label=name)
    ax.set_title('Center-line profiles at z = 0'); ax.set_xlabel('x (mm)'); ax.set_ylabel('I(x,0) (W/m²)'); ax.grid(alpha=.3); ax.legend(fontsize=8)
    fig.suptitle('Initial transverse irradiance: z = 0, λ = 1064 nm, P = 1 W',fontsize=17)
    fig.savefig(outdir/'initial_beams_z0_comparison.png',dpi=180); plt.close(fig)
    print(f'Wrote {outdir}/initial_beams_z0_comparison.png')

if __name__=='__main__': main()
