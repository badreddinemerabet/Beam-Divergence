"""Compare Gaussian-beam propagation for 3 mm and 50 mm initial waists."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

C0=299_792_458.0; EPS0=8.854_187_8128e-12
POWER=1.0; M2=1.0; ZMAX=100_000.0
WAVELENGTHS=[(1064e-9,'Near-IR 1064 nm','#d97706'),(1550e-9,'Near-IR 1550 nm','#eab308'),(3500e-9,'Mid-IR 3.5 µm','#0891b2'),(10600e-9,'Mid-IR 10.6 µm','#dc2626')]
WAISTS=[(3e-3,'3 mm'),(50e-3,'50 mm')]

def radius(z, wavelength, waist):
    zr=np.pi*waist**2/(M2*wavelength)
    return waist*np.sqrt(1+(z/zr)**2)

def irradiance(y,z,wavelength,waist):
    w=radius(z,wavelength,waist)
    return 2*POWER/(np.pi*w**2)*np.exp(-2*y**2/w**2)

def main():
    out=Path(__file__).resolve().parent/'outputs'; out.mkdir(exist_ok=True)
    z=np.linspace(0,ZMAX,700); y=np.linspace(-120,120,600); Z,Y=np.meshgrid(z,y)
    fig,axes=plt.subplots(2,2,figsize=(15,11),constrained_layout=True,sharex=True,sharey=True)
    for ax,(waist,wlabel) in zip(axes.flat,WAISTS):
        allI=[]
        for lam,name,color in WAVELENGTHS: allI.append(irradiance(Y,Z,lam,waist))
        vmax=max(float(I.max()) for I in allI)
        for I,(lam,name,color) in zip(allI,WAVELENGTHS):
            ax.plot([],[],color=color,label=name)
        # Use a composite image: show the shortest wavelength as background is not useful;
        # instead make one panel per waist with wavelength curves over a shared transverse map.
        # The four wavelength maps are separated into columns below.
        ax.clear()
        ax.text(.5,.5,f'w₀ = {wlabel}\nSelect the wavelength panels below',ha='center',va='center',transform=ax.transAxes,fontsize=14)
        ax.axis('off')
    plt.close(fig)
    # Main figure: 2 rows (waists) x 4 columns (wavelengths).
    fig,axes=plt.subplots(2,4,figsize=(19,8),constrained_layout=True,sharex=True,sharey=True)
    for i,(waist,wlabel) in enumerate(WAISTS):
        for j,(lam,name,color) in enumerate(WAVELENGTHS):
            I=irradiance(Y,Z,lam,waist)
            ax=axes[i,j]
            im=ax.pcolormesh(Z/1000,Y,np.maximum(I,1e-12),shading='auto',cmap='magma',norm=LogNorm(vmin=1e-8,vmax=max(1.0,float(I.max()))))
            ax.set_title(f'{name}\n$w_0$ = {wlabel}')
            ax.set_xlabel('z (km)'); ax.set_ylabel('transverse y (m)'); ax.grid(alpha=.15)
            fig.colorbar(im,ax=ax,label='I (W/m², log)')
    fig.suptitle('Gaussian-beam propagation for two initial waist radii, P = 1 W, M² = 1',fontsize=16)
    fig.savefig(out/'gaussian_propagation_waists_3mm_50mm.png',dpi=180); plt.close(fig)
    # Summary table.
    with (out/'gaussian_waist_comparison.csv').open('w') as f:
        f.write('waist_mm,wavelength_nm,rayleigh_range_m,half_angle_rad,radius_at_100km_m\n')
        for waist,wlabel in WAISTS:
            for lam,name,color in WAVELENGTHS:
                zr=np.pi*waist**2/(M2*lam); th=M2*lam/(np.pi*waist)
                f.write(f'{waist*1000:.6g},{lam*1e9:.6g},{zr:.9g},{th:.9g},{radius(ZMAX,lam,waist):.9g}\n')
    print(f'Saved {out/"gaussian_propagation_waists_3mm_50mm.png"}')

if __name__=='__main__': main()
