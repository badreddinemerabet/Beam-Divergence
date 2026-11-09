"""Corrected finite-power transverse beam comparison.

This version makes the source-plane comparison consistent:
- all beams are represented as 2-D irradiance fields I(x,y,0)
- every field is numerically normalized to POWER on the plotting aperture
- LG01 normalization is corrected
- HG10 is a full 2-D mode, not only a 1-D cut
- Airy is a separable 2-D finite-energy Airy beam
- Bessel uses a finite super-Gaussian aperture envelope

Propagation plots remain compact analytical approximations; use an angular-
spectrum propagator for engineering-grade propagation and divergence labels.
"""
from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.special import j0, j1, airy

LAMBDA=1064e-9; K=2*np.pi/LAMBDA; POWER=1.0; W0=10e-3; ZMAX=100e3
BESSEL_THETA=0.05e-3; BESSEL_KR=K*np.sin(BESSEL_THETA); BESSEL_APERTURE_RADIUS=0.20
BESSEL_CORE_RADIUS=2.405/BESSEL_KR
AIRY_X0=0.05; AIRY_A=0.08; TOPHAT_RADIUS=W0
NAMES=['Bessel (finite aperture)','Airy (2-D separable)','Top-hat aperture','Laguerre-Gaussian LG01','Hermite-Gaussian HG10']
SHORT=['Bessel','Airy','Top-hat','LG01','HG10']
COLORS=['#2563eb','#ea580c','#16a34a','#9333ea','#0891b2']


def normalize(I, dx, dy):
    total=float(np.sum(I)*dx*dy)
    return I*(POWER/total) if total>0 else I


def source_intensities(X,Y):
    r=np.hypot(X,Y)
    dx=float(X[0,1]-X[0,0]); dy=float(Y[1,0]-Y[0,0])
    # Finite-power Bessel approximation: a smooth finite generating aperture.
    b=j0(BESSEL_KR*r)**2*np.exp(-2*(r/(0.72*BESSEL_APERTURE_RADIUS))**8)
    # Separable 2-D finite-energy Airy source.
    sx=X/AIRY_X0; sy=Y/AIRY_X0
    afx=airy(sx)[0]*np.exp(AIRY_A*sx)
    afy=airy(sy)[0]*np.exp(AIRY_A*sy)
    a=np.abs(afx*afy)**2
    # Circular top-hat aperture.
    t=(r<=TOPHAT_RADIUS).astype(float)
    # Correct p=0, l=1 LG normalization: base Gaussian times rho^2.
    rho2=2*r*r/W0**2
    lg=(2*POWER/(np.pi*W0**2))*rho2*np.exp(-rho2)
    # Full 2-D HG10, normalized analytically then checked numerically.
    u=np.sqrt(2)*X/W0
    hg=(2*POWER/(np.pi*W0**2))*2*u*u*np.exp(-2*r*r/W0**2)
    return [normalize(b,dx,dy),normalize(a,dx,dy),normalize(t,dx,dy),normalize(lg,dx,dy),normalize(hg,dx,dy)]


def gaussian_width(z, mode_factor=1):
    zr=np.pi*W0**2/(mode_factor*LAMBDA)
    return W0*np.sqrt(1+(z/zr)**2)


def profiles_at_z0(axis):
    X,Y=np.meshgrid(axis,axis); vals=source_intensities(X,Y)
    mid=len(axis)//2
    return vals,[v[mid,:] for v in vals]


def plot_z0(outdir):
    axis=np.linspace(-0.15,0.15,700); vals,profiles=profiles_at_z0(axis)
    fig,axes=plt.subplots(2,3,figsize=(16,10),constrained_layout=True)
    for ax,name,val in zip(axes.flat,NAMES,vals):
        im=ax.pcolormesh(axis*1000,axis*1000,np.maximum(val,1e-12),shading='auto',cmap='coolwarm',norm=LogNorm(vmin=1e-8,vmax=max(1,float(val.max()))))
        ax.set_title(name); ax.set_xlabel('x (mm)'); ax.set_ylabel('y (mm)'); ax.set_aspect('equal'); fig.colorbar(im,ax=ax,label='I (W/m², log)')
    ax=axes[1,2]
    for name,c,p in zip(NAMES,COLORS,profiles): ax.plot(axis*1000,p,c=c,lw=1.8,label=name)
    ax.set_title('Center-line irradiance profiles at z = 0'); ax.set_xlabel('x (mm)'); ax.set_ylabel('I(x,0) (W/m²)'); ax.grid(alpha=.3); ax.legend(fontsize=7)
    fig.suptitle('Corrected initial transverse irradiance, z = 0, λ = 1064 nm, P = 1 W',fontsize=16)
    fig.savefig(outdir/'initial_beams_z0_corrected.png',dpi=180); plt.close(fig)
    # Linear-scale version for shape/edge visibility.
    fig,axes=plt.subplots(2,3,figsize=(16,10),constrained_layout=True)
    for ax,name,val in zip(axes.flat,NAMES,vals):
        im=ax.pcolormesh(axis*1000,axis*1000,val,shading='auto',cmap='coolwarm',vmin=0,vmax=max(1,float(val.max())))
        ax.set_title(name); ax.set_xlabel('x (mm)'); ax.set_ylabel('y (mm)'); ax.set_aspect('equal'); fig.colorbar(im,ax=ax,label='I (W/m²)')
    ax=axes[1,2]
    for name,c,p in zip(NAMES,COLORS,profiles): ax.plot(axis*1000,p,c=c,lw=1.8,label=name)
    ax.set_title('Linear center-line profiles at z = 0'); ax.set_xlabel('x (mm)'); ax.set_ylabel('I(x,0) (W/m²)'); ax.grid(alpha=.3); ax.legend(fontsize=7)
    fig.suptitle('Corrected initial irradiance, linear scale',fontsize=16)
    fig.savefig(outdir/'initial_beams_z0_corrected_linear.png',dpi=180); plt.close(fig)
    # Power check.
    with open(outdir/'z0_power_check.csv','w',newline='') as f:
        w=csv.writer(f); w.writerow(['beam','integrated_power_W'])
        dx=axis[1]-axis[0]
        for name,val in zip(SHORT,vals): w.writerow([name,float(np.sum(val)*dx*dx)])
    print('Saved corrected z=0 figures and power check')


def approximate_propagation(outdir):
    z=np.linspace(0,ZMAX,1001); zr=np.pi*W0**2/LAMBDA
    # These are diagnostic approximations, not replacements for angular-spectrum propagation.
    bessel_axis=np.full_like(z,POWER/(np.pi*BESSEL_CORE_RADIUS**2)); bessel_axis[z>BESSEL_APERTURE_RADIUS/np.tan(BESSEL_THETA)]*=np.exp(-((z[z>BESSEL_APERTURE_RADIUS/np.tan(BESSEL_THETA)]-BESSEL_APERTURE_RADIUS/np.tan(BESSEL_THETA))/(BESSEL_APERTURE_RADIUS/np.tan(BESSEL_THETA)))**2)
    xi=z/(K*AIRY_X0**2); airy_axis=POWER/(np.sqrt(2*np.pi)*AIRY_X0)*np.abs(airy(-xi**2/4+1j*AIRY_A*xi)[0])**2*np.exp(-AIRY_A*xi**2)
    top_axis=POWER/(np.pi*TOPHAT_RADIUS**2)*np.where(z<=K*TOPHAT_RADIUS**2/2,1,(K*TOPHAT_RADIUS**2/2/np.maximum(z,1e-12))**2)
    with open(outdir/'other_beams_on_axis_corrected.csv','w',newline='') as f:
        w=csv.writer(f); w.writerow(['z_m']+SHORT)
        w.writerows(zip(z,bessel_axis,airy_axis,top_axis,np.zeros_like(z),np.zeros_like(z)))
    fig,ax=plt.subplots(figsize=(10,6))
    for c,n,v in zip(COLORS,SHORT,[bessel_axis,airy_axis,top_axis,np.zeros_like(z),np.zeros_like(z)]): ax.plot(z/1000,np.maximum(v,1e-12),c=c,label=n)
    ax.set_yscale('log'); ax.set(xlabel='z (km)',ylabel='on-axis irradiance (W/m²)',title='Corrected analytical on-axis diagnostics'); ax.grid(alpha=.3); ax.legend(); fig.tight_layout(); fig.savefig(outdir/'other_beams_comparison_corrected.png',dpi=180); plt.close(fig)


def main():
    out=Path(__file__).resolve().parent/'outputs_other_beams'; out.mkdir(exist_ok=True)
    print(f'Corrected source-plane comparison: λ={LAMBDA*1e9:.0f} nm, P={POWER:g} W')
    print(f'Bessel core estimate={BESSEL_CORE_RADIUS*1e3:.2f} mm; finite aperture radius={BESSEL_APERTURE_RADIUS:g} m')
    plot_z0(out); approximate_propagation(out); print(f'Saved outputs to {out}')

if __name__=='__main__': main()
