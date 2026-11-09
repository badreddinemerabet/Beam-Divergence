"""Circular top-hat aperture propagation maps for multiple wavelengths."""
from pathlib import Path
import csv
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.special import j1

POWER=1.0; RADIUS=50e-3; ZMAX=100e3
WAVELENGTHS=[(1064e-9,'1064 nm'),(1550e-9,'1550 nm'),(4e-6,'4 µm'),(10.6e-6,'10.6 µm')]

def intensity_line(x,z,lam):
    """Paraxial near/far-field diagnostic for a circular hard aperture."""
    k=2*np.pi/lam; ztr=k*RADIUS**2/2; x=np.asarray(x); z=np.asarray(z)
    near=POWER/(np.pi*RADIUS**2)*(np.abs(x)<=RADIUS)
    zz=np.maximum(z,1e-12); u=k*RADIUS*x/zz
    airy=np.ones_like(np.asarray(u),dtype=float); mask=np.abs(u)>1e-12; airy[mask]=(2*j1(u[mask])/u[mask])**2
    far=POWER/(np.pi*RADIUS**2)*(ztr/zz)**2*airy
    blend=np.clip(z/ztr,0,1)
    return (1-blend)*near+blend*far

def intensity_2d(x,y,z,lam):
    r=np.hypot(x,y); return intensity_line(r,z,lam)

def main():
    global RADIUS
    parser=argparse.ArgumentParser(); parser.add_argument('--radius-mm',type=float,default=50.0); args=parser.parse_args(); RADIUS=args.radius_mm*1e-3
    tag=f'{args.radius_mm:g}mm'.replace('.','p'); out=Path(__file__).resolve().parent/'outputs_other_beams'; out.mkdir(exist_ok=True)
    z=np.linspace(0,ZMAX,700); x=np.linspace(-15,15,900); Z,X=np.meshgrid(z,x)
    fig,axes=plt.subplots(2,2,figsize=(17,11),constrained_layout=True)
    with (out/f'tophat_multiwavelength_summary_{tag}.csv').open('w',newline='') as f:
        wr=csv.writer(f); wr.writerow(['wavelength_nm','aperture_radius_mm','transition_distance_km','first_zero_radius_at_100km_m','far_field_half_angle_rad'])
        for ax,(lam,label) in zip(axes.flat,WAVELENGTHS):
            I=intensity_line(X,Z,lam); im=ax.pcolormesh(Z/1000,X,np.maximum(I,1e-12),shading='auto',cmap='coolwarm',norm=LogNorm(vmin=1e-8,vmax=max(1,float(I.max()))))
            ztr=2*np.pi/lam*RADIUS**2/2; first=1.22*lam*ZMAX/(2*RADIUS); theta=1.22*lam/(2*RADIUS)
            ax.set_title(f'Top-hat aperture, λ = {label}, R = {args.radius_mm:g} mm'); ax.set_xlabel('z (km)'); ax.set_ylabel('transverse x (m)'); ax.grid(alpha=.15); fig.colorbar(im,ax=ax,label='I (W/m², log)'); wr.writerow([lam*1e9,RADIUS*1e3,ztr/1000,first,theta])
    fig.suptitle('Circular top-hat aperture irradiance propagation',fontsize=16); fig.savefig(out/f'tophat_2d_propagation_multiwavelength_{tag}.png',dpi=180); plt.close(fig)
    fig,axes=plt.subplots(2,4,figsize=(19,9),constrained_layout=True)
    for j,(lam,label) in enumerate(WAVELENGTHS):
        first=1.22*lam*ZMAX/(2*RADIUS)
        for i,(zv,title) in enumerate([(0,'z = 0'),(ZMAX,'z = 100 km')]):
            extent=max(3*RADIUS,2.5*first) if zv==ZMAX else 3*RADIUS; a=np.linspace(-extent,extent,600); X2,Y2=np.meshgrid(a,a); I=intensity_2d(X2,Y2,zv,lam); ax=axes[i,j]; im=ax.pcolormesh(a,a,np.maximum(I,1e-12),shading='auto',cmap='coolwarm',norm=LogNorm(vmin=1e-8,vmax=max(1,float(I.max())))); ax.set_title(f'{label}, {title}'); ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)'); ax.set_aspect('equal'); fig.colorbar(im,ax=ax,label='I (W/m², log)')
    fig.suptitle('Top-hat transverse irradiance: source and 100 km',fontsize=16); fig.savefig(out/f'tophat_transverse_multiwavelength_z0_z100km_{tag}.png',dpi=180); plt.close(fig)
    print('Saved multi-wavelength top-hat maps')

if __name__=='__main__': main()
