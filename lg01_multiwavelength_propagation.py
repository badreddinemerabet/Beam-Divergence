"""LG01 2-D propagation maps for multiple wavelengths."""
from pathlib import Path
import csv
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

POWER=1.0; M2=1.0; W0=10e-3; ZMAX=100e3
WAVELENGTHS=[(1064e-9,'1064 nm'),(1550e-9,'1550 nm'),(4e-6,'4 µm'),(10.6e-6,'10.6 µm')]

def waist(z,lam):
    zr=np.pi*W0**2/(M2*lam)
    return W0*np.sqrt(1+(np.asarray(z)/zr)**2)

def intensity(x,y,z,lam):
    w=waist(z,lam); r2=np.asarray(x)**2+np.asarray(y)**2; rho2=2*r2/w**2
    return (2*POWER/(np.pi*w**2))*rho2*np.exp(-rho2)

def main():
    global W0
    parser=argparse.ArgumentParser()
    parser.add_argument('--waist-mm',type=float,default=10.0)
    args=parser.parse_args(); W0=args.waist_mm*1e-3
    out=Path(__file__).resolve().parent/'outputs_other_beams'; out.mkdir(exist_ok=True)
    tag=f'{args.waist_mm:g}mm'.replace('.','p')
    z=np.linspace(0,ZMAX,700); x=np.linspace(-12,12,800); Z,X=np.meshgrid(z,x)
    fig,axes=plt.subplots(2,2,figsize=(16,10),constrained_layout=True,sharex=True,sharey=True)
    with (out/f'lg01_multiwavelength_summary_{tag}.csv').open('w',newline='') as f:
        wr=csv.writer(f); wr.writerow(['wavelength_nm','waist_z0_mm','rayleigh_range_m','waist_at_100km_m','half_angle_rad'])
        for ax,(lam,label) in zip(axes.flat,WAVELENGTHS):
            I=intensity(X,0,Z,lam)
            im=ax.pcolormesh(Z/1000,X,np.maximum(I,1e-12),shading='auto',cmap='coolwarm',norm=LogNorm(vmin=1e-8,vmax=max(1,float(I.max()))))
            ax.set_title(f'LG01, λ = {label}, w₀ = {args.waist_mm:g} mm'); ax.set_xlabel('z (km)'); ax.set_ylabel('transverse x (m)'); ax.grid(alpha=.15); fig.colorbar(im,ax=ax,label='I (W/m², log)')
            zr=np.pi*W0**2/(M2*lam); w100=float(waist(ZMAX,lam)); th=M2*lam/(np.pi*W0)
            wr.writerow([lam*1e9,W0*1e3,zr,w100,th])
    fig.suptitle('LG01 2-D irradiance propagation for multiple wavelengths',fontsize=16)
    fig.savefig(out/f'lg01_2d_propagation_multiwavelength_{tag}.png',dpi=180); plt.close(fig)
    # Source and target transverse maps for each wavelength.
    fig,axes=plt.subplots(2,4,figsize=(18,9),constrained_layout=True)
    for j,(lam,label) in enumerate(WAVELENGTHS):
        for i,(zv,extent,title) in enumerate([(0,max(0.06,6*W0),'z = 0'),(ZMAX,12,'z = 100 km')]):
            a=np.linspace(-extent,extent,600); X2,Y2=np.meshgrid(a,a); I=intensity(X2,Y2,zv,lam); ax=axes[i,j]
            im=ax.pcolormesh(a,a,np.maximum(I,1e-12),shading='auto',cmap='coolwarm',norm=LogNorm(vmin=1e-8,vmax=max(1,float(I.max())))); ax.set_title(f'{label}, {title}'); ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)'); ax.set_aspect('equal'); fig.colorbar(im,ax=ax,label='I (W/m², log)')
    fig.suptitle('LG01 transverse irradiance: source and 100 km',fontsize=16)
    fig.savefig(out/f'lg01_transverse_multiwavelength_z0_z100km_{tag}.png',dpi=180); plt.close(fig)
    print('Saved multi-wavelength LG01 maps')

if __name__=='__main__': main()
