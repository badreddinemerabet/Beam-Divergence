"""2-D finite-energy Airy-beam propagation maps for multiple wavelengths."""
from pathlib import Path
import csv
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.special import airy

POWER=1.0; X0=50e-3; APOD=0.08; ZMAX=100e3
WAVELENGTHS=[(1064e-9,'1064 nm'),(1550e-9,'1550 nm'),(4e-6,'4 µm'),(10.6e-6,'10.6 µm')]

def field_1d(x,z,lam):
    # Standard finite-energy paraxial Airy solution.
    k=2*np.pi/lam; z0=k*X0**2; s=np.asarray(x)/X0; xi=np.asarray(z)/z0
    arg=s-xi**2/4+1j*APOD*xi
    Ai=airy(arg)[0]
    return Ai*np.exp(APOD*s-APOD*xi**2/2+1j*(s*xi/2-xi**3/12+APOD**2*xi/2))

def source_normalization(lam):
    x=np.linspace(-12*X0,12*X0,2400); X,Y=np.meshgrid(x,x); F=field_1d(X,0,lam)*field_1d(Y,0,lam)
    p=np.trapezoid(np.trapezoid(np.abs(F)**2,x,axis=1),x,axis=0)
    return np.sqrt(POWER/p)

def main():
    global X0
    parser=argparse.ArgumentParser(); parser.add_argument('--scale-mm',type=float,default=50.0); args=parser.parse_args(); X0=args.scale_mm*1e-3
    tag=f'{args.scale_mm:g}mm'.replace('.','p'); out=Path(__file__).resolve().parent/'outputs_other_beams'; out.mkdir(exist_ok=True)
    fig,axes=plt.subplots(2,2,figsize=(17,11),constrained_layout=True)
    with (out/f'airy_multiwavelength_summary_{tag}.csv').open('w',newline='') as f:
        wr=csv.writer(f); wr.writerow(['wavelength_nm','airy_scale_mm','apodization','z0_m','main_lobe_shift_at_100km_m','source_normalization'])
        for ax,(lam,label) in zip(axes.flat,WAVELENGTHS):
            k=2*np.pi/lam; z0=k*X0**2; xi_end=ZMAX/z0; shift=X0*xi_end**2/4
            # Track the accelerating lobe in a lab-coordinate window; 2 m margin for structure.
            xlim=max(0.25,abs(shift)+0.25); x=np.linspace(-xlim,xlim,900); z=np.linspace(0,ZMAX,700); Z,X=np.meshgrid(z,x)
            norm=source_normalization(lam); F=norm*field_1d(X,Z,lam)*field_1d(0,Z,lam); I=np.abs(F)**2
            im=ax.pcolormesh(Z/1000,X,np.maximum(I,1e-30),shading='auto',cmap='coolwarm',norm=LogNorm(vmin=max(1e-20,float(I.max())*1e-10),vmax=max(1e-20,float(I.max()))))
            ax.set_title(f'Airy beam, λ = {label}, scale = {args.scale_mm:g} mm'); ax.set_xlabel('z (km)'); ax.set_ylabel('transverse x (m)'); ax.grid(alpha=.15); fig.colorbar(im,ax=ax,label='I (W/m², log)')
            wr.writerow([lam*1e9,X0*1e3,APOD,z0,shift,norm])
    fig.suptitle('2-D finite-energy Airy-beam irradiance propagation',fontsize=16)
    fig.savefig(out/f'airy_2d_propagation_multiwavelength_{tag}.png',dpi=180); plt.close(fig)
    # Transverse maps, with the target window centered near the accelerating main lobe.
    fig,axes=plt.subplots(2,4,figsize=(19,9),constrained_layout=True)
    for j,(lam,label) in enumerate(WAVELENGTHS):
        norm=source_normalization(lam); z0=2*np.pi/lam*X0**2; shift=X0*(ZMAX/z0)**2/4
        for i,(zv,title) in enumerate([(0,'z = 0'),(ZMAX,'z = 100 km')]):
            center=X0*(zv/z0)**2/4; extent=max(6*X0,0.25) if zv==0 else max(0.25,6*X0)
            a=np.linspace(center-extent,center+extent,600); X2,Y2=np.meshgrid(a,a); F=norm*field_1d(X2,zv,lam)*field_1d(Y2,zv,lam); I=np.abs(F)**2; ax=axes[i,j]
            im=ax.pcolormesh(a,a,np.maximum(I,1e-30),shading='auto',cmap='coolwarm',norm=LogNorm(vmin=max(1e-20,float(I.max())*1e-10),vmax=max(1e-20,float(I.max())))); ax.set_title(f'{label}, {title}'); ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)'); ax.set_aspect('equal'); fig.colorbar(im,ax=ax,label='I (W/m², log)')
    fig.suptitle('Airy transverse irradiance at source and 100 km',fontsize=16)
    fig.savefig(out/f'airy_transverse_multiwavelength_z0_z100km_{tag}.png',dpi=180); plt.close(fig)
    print('Saved multi-wavelength Airy maps')

if __name__=='__main__': main()
