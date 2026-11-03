"""Generate true transverse x-y irradiance maps for the five non-Gaussian beams."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.special import j0, j1, airy

LAMBDA = 1064e-9
K = 2*np.pi/LAMBDA
P = 1.0
W0 = 10e-3
ZR = np.pi*W0**2/LAMBDA
BESSEL_THETA = 0.05e-3
BESSEL_KR = K*np.sin(BESSEL_THETA)
BESSEL_CORE = 2.405/BESSEL_KR
BESSEL_ZMAX = 0.20/np.tan(BESSEL_THETA)
AIRY_X0 = 0.05
AIRY_A = 0.08
AIRY_Z0 = K*AIRY_X0**2
TOPHAT_R = W0


def airy_field_1d(x, z):
    s = x/AIRY_X0; xi = z/AIRY_Z0
    return airy(s-xi**2/4+1j*AIRY_A*xi)[0] * np.exp(AIRY_A*s-AIRY_A*xi**2/2 + 1j*(s*xi/2-xi**3/12+AIRY_A**2*xi/2))


def transverse_fields(X, Y, z):
    w = W0*np.sqrt(1+(z/ZR)**2)
    # Bessel finite-aperture approximation
    bessel = 2*P/(np.pi*BESSEL_CORE**2) * j0(BESSEL_KR*np.sqrt(X**2+Y**2))**2
    if z > BESSEL_ZMAX:
        bessel *= np.exp(-((z-BESSEL_ZMAX)/BESSEL_ZMAX)**2)
    # 2-D separable finite-energy Airy beam
    airy2d = P/(np.sqrt(2*np.pi)*AIRY_X0) * np.abs(airy_field_1d(X,z)*airy_field_1d(Y,z))**2
    # Circular aperture; use a Fraunhofer Airy pattern after the Fresnel transition
    ztr = K*TOPHAT_R**2/2
    rho = np.sqrt(X**2+Y**2)
    q = K*TOPHAT_R*rho/max(z, 1e-12)
    airy_disk = np.ones_like(q)
    mask = q > 1e-12
    airy_disk[mask] = (2*j1(q[mask])/q[mask])**2
    near = P/(np.pi*TOPHAT_R**2)*(rho <= TOPHAT_R)
    far = P/(np.pi*TOPHAT_R**2)*(ztr/max(z,1e-12))**2*airy_disk
    blend = np.clip(z/ztr, 0, 1)
    top_hat = (1-blend)*near + blend*far
    # LG01 vortex and HG10 mode
    rho2 = 2*(X**2+Y**2)/w**2
    lg01 = 4*P/(np.pi*w**2)*rho2*np.exp(-rho2)
    u = np.sqrt(2)*X/w
    hg10 = 2*(2*P/(np.pi*w**2))*u**2*np.exp(-u**2)
    return [bessel, airy2d, top_hat, lg01, hg10]


def make_maps():
    outdir = Path(__file__).resolve().parent/'outputs_other_beams'
    outdir.mkdir(exist_ok=True)
    names = ['Bessel J0','Airy','Top-hat aperture','Laguerre-Gaussian LG01','Hermite-Gaussian HG10']
    for z, tag, extent in [(0.0,'z0',0.15),(100e3,'z100km',10.0)]:
        n = 500
        axis = np.linspace(-extent, extent, n)
        X,Y = np.meshgrid(axis, axis)
        fields = transverse_fields(X,Y,z)
        fig, axes = plt.subplots(2,3,figsize=(15,10),constrained_layout=True)
        for ax,name,field in zip(axes.ravel(),names,fields):
            image=ax.pcolormesh(axis,axis,np.maximum(field,1e-12),shading='auto',cmap='magma',
                                norm=LogNorm(vmin=1e-10,vmax=max(1.0,float(np.max(field)))))
            ax.set_title(name); ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)'); ax.set_aspect('equal')
            fig.colorbar(image,ax=ax,label='Irradiance (W/m², log scale)')
        axes[-1,-1].axis('off')
        fig.suptitle(f'Transverse 2-D irradiance maps at z = {tag.replace("z","").replace("100km","100 km")}',fontsize=16)
        fig.savefig(outdir/f'transverse_irradiance_maps_{tag}.png',dpi=180); plt.close(fig)
        print(f'Wrote transverse_irradiance_maps_{tag}.png')

if __name__ == '__main__': make_maps()
