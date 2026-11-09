"""2-D Laguerre-Gaussian LG01 irradiance propagation map."""
from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

LAMBDA=1064e-9; POWER=1.0; M2=1.0; W0=10e-3; ZMAX=100e3

def waist(z):
    zr=np.pi*W0**2/(M2*LAMBDA)
    return W0*np.sqrt(1+(np.asarray(z)/zr)**2)

def irradiance(x,y,z):
    w=waist(z); r2=np.asarray(x)**2+np.asarray(y)**2
    rho2=2*r2/w**2
    # Correct normalized LG_0^1 intensity; integral over x,y = POWER.
    return (2*POWER/(np.pi*w**2))*rho2*np.exp(-rho2)

def main():
    out=Path(__file__).resolve().parent/'outputs_other_beams'; out.mkdir(exist_ok=True)
    z=np.linspace(0,ZMAX,700); x=np.linspace(-5,5,700); Z,X=np.meshgrid(z,x)
    field=irradiance(X,0,Z)
    fig,ax=plt.subplots(figsize=(13,7),constrained_layout=True)
    im=ax.pcolormesh(Z/1000,X, np.maximum(field,1e-12), shading='auto', cmap='coolwarm', norm=LogNorm(vmin=1e-8,vmax=max(1,float(field.max()))))
    fig.colorbar(im,ax=ax,label='Irradiance (W/m², logarithmic scale)')
    ax.set(title='LG01 2-D irradiance propagation map',xlabel='Propagation distance z (km)',ylabel='Transverse coordinate x (m)')
    ax.grid(alpha=.2); fig.savefig(out/'lg01_2d_irradiance_propagation_map.png',dpi=180); plt.close(fig)
    # Transverse maps at source and target.
    fig,axes=plt.subplots(1,2,figsize=(13,5.5),constrained_layout=True)
    for ax,zv,title,extent in [(axes[0],0,'LG01 irradiance at z = 0 km',0.06),(axes[1],ZMAX,'LG01 irradiance at z = 100 km',5.0)]:
        a=np.linspace(-extent,extent,600); X2,Y2=np.meshgrid(a,a); I=irradiance(X2,Y2,zv)
        im=ax.pcolormesh(a,a,np.maximum(I,1e-12),shading='auto',cmap='coolwarm',norm=LogNorm(vmin=1e-8,vmax=max(1,float(I.max()))))
        ax.set(title=title,xlabel='x (m)',ylabel='y (m)',aspect='equal'); fig.colorbar(im,ax=ax,label='I (W/m², log)')
    fig.suptitle('Normalized Laguerre-Gaussian LG01 transverse irradiance',fontsize=15)
    fig.savefig(out/'lg01_transverse_z0_z100km.png',dpi=180); plt.close(fig)
    zr=np.pi*W0**2/(M2*LAMBDA); w100=float(waist(ZMAX));
    with (out/'lg01_propagation_summary.csv').open('w',newline='') as f:
        w=csv.writer(f); w.writerow(['parameter','value','units']); w.writerow(['wavelength',LAMBDA*1e9,'nm']); w.writerow(['power',POWER,'W']); w.writerow(['M2',M2,'']); w.writerow(['waist_z0',W0*1e3,'mm']); w.writerow(['rayleigh_range',zr,'m']); w.writerow(['waist_at_100km',w100,'m'])
    print('Saved LG01 propagation map and transverse maps')

if __name__=='__main__': main()
