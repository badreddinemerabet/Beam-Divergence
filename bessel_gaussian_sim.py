"""Finite-power Bessel–Gaussian beam simulation over 0–100 km.

The model uses a Bessel core multiplied by a Gaussian envelope:
    I(r,z) = A(z) J0(kr*r)^2 exp[-2 r^2 / w_env(z)^2]

The envelope radius follows Gaussian diffraction while the central Bessel
oscillation scale remains approximately fixed. A(z) is normalized numerically
so each plane carries P=1 W in this model.
"""
from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.special import j0

LAMBDA=1064e-9; K=2*np.pi/LAMBDA; POWER=1.0
CONE_ANGLE=0.05e-3; KR=K*np.sin(CONE_ANGLE)
ENVELOPE_W0=0.20; ZMAX=100e3
ZR_ENV=np.pi*ENVELOPE_W0**2/LAMBDA
CORE_RADIUS=2.405/KR
NAMES='Bessel–Gaussian'


def envelope_radius(z):
    z=np.asarray(z,dtype=float)
    return ENVELOPE_W0*np.sqrt(1+(z/ZR_ENV)**2)


def normalization(z):
    w=envelope_radius(z)
    # Integral of J0^2 exp(-2r^2/w^2) over the radial plane.
    rmax=8*np.max(np.atleast_1d(w))
    r=np.linspace(0,rmax,25000)
    # For scalar z this is inexpensive and stable.
    shape=j0(KR*r)**2*np.exp(-2*r**2/w**2)
    integral=2*np.pi*np.trapezoid(shape*r,r)
    return POWER/integral


def irradiance(r,z):
    r=np.asarray(r,dtype=float); w=envelope_radius(z); A=normalization(z)
    return A*j0(KR*np.abs(r))**2*np.exp(-2*r**2/w**2)


def main():
    out=Path(__file__).resolve().parent/'outputs_bessel_gaussian'; out.mkdir(exist_ok=True)
    z=np.linspace(0,ZMAX,1001)
    w=envelope_radius(z); Iaxis=np.array([irradiance(0,zi) for zi in z])
    # summary
    print('Bessel–Gaussian simulation')
    print(f'lambda = {LAMBDA*1e9:.0f} nm, power = {POWER:.1f} W')
    print(f'cone angle = {CONE_ANGLE*1e3:.3f} mrad, Bessel core radius ≈ {CORE_RADIUS*1e3:.2f} mm')
    print(f'Gaussian envelope radius: {ENVELOPE_W0:.3f} m at z=0; {w[-1]:.3f} m at 100 km')
    with open(out/'bessel_gaussian_propagation.csv','w',newline='') as f:
        wr=csv.writer(f); wr.writerow(['z_m','envelope_radius_m','on_axis_irradiance_W_m2'])
        wr.writerows(zip(z,w,Iaxis))
    with open(out/'bessel_gaussian_parameters.csv','w',newline='') as f:
        wr=csv.writer(f); wr.writerow(['parameter','value','units'])
        for row in [('wavelength',LAMBDA*1e9,'nm'),('power',POWER,'W'),('cone_angle',CONE_ANGLE*1e3,'mrad'),('core_radius',CORE_RADIUS*1e3,'mm'),('envelope_radius_z0',ENVELOPE_W0,'m'),('envelope_radius_100km',w[-1],'m'),('envelope_rayleigh_range',ZR_ENV/1000,'km')]: wr.writerow(row)

    # Longitudinal cross-section
    zmap=np.linspace(0,ZMAX,600); y=np.linspace(-0.6,0.6,500); Z,Y=np.meshgrid(zmap,y)
    W=envelope_radius(Z); A=np.array([normalization(zi) for zi in zmap]); field=A[None,:]*j0(KR*np.abs(Y))**2*np.exp(-2*Y**2/W**2)
    fig,ax=plt.subplots(figsize=(12,6),constrained_layout=True)
    im=ax.pcolormesh(Z/1000,Y,np.maximum(field,1e-12),shading='auto',cmap='magma',norm=LogNorm(vmin=1e-10,vmax=max(1,float(field.max())))); fig.colorbar(im,ax=ax,label='Irradiance (W/m², log scale)')
    ax.set(title='Bessel–Gaussian propagation cross-section',xlabel='z (km)',ylabel='transverse y (m)'); fig.savefig(out/'bessel_gaussian_propagation_map.png',dpi=180); plt.close(fig)

    # Transverse maps at source and 100 km
    for zi,tag,extent in [(0,'z0',0.25),(ZMAX,'z100km',0.6)]:
        axis=np.linspace(-extent,extent,600); X,Y=np.meshgrid(axis,axis); F=irradiance(np.sqrt(X**2+Y**2),zi)
        fig,ax=plt.subplots(figsize=(7,6),constrained_layout=True); im=ax.pcolormesh(axis,axis,np.maximum(F,1e-12),shading='auto',cmap='magma',norm=LogNorm(vmin=1e-10,vmax=max(1,float(F.max())))); fig.colorbar(im,ax=ax,label='Irradiance (W/m², log scale)'); ax.set_aspect('equal'); ax.set(title=f'Bessel–Gaussian transverse irradiance at z={zi/1000:g} km',xlabel='x (m)',ylabel='y (m)'); fig.savefig(out/f'bessel_gaussian_transverse_{tag}.png',dpi=180); plt.close(fig)
    print(f'Saved results to {out}')

if __name__=='__main__': main()
