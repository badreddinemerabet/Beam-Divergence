"""Unified comparison of non-Gaussian paraxial beams over 0–100 km.

Run: python other_beams_sim.py

This is a compact theoretical comparison at lambda=1064 nm and P=1 W.
The models are scalar and paraxial:
  Bessel: finite-aperture J0 approximation with a finite nondiffracting length.
  Airy: finite-energy 1-D Airy solution with exponential apodization.
  Top-hat: circular-aperture Fresnel/Fraunhofer transition approximation.
  LG01: Laguerre–Gaussian vortex mode.
  HG10: Hermite–Gaussian first-order mode.

For engineering propagation through atmosphere, apertures, or lenses, replace the
closed-form models with an angular-spectrum or Fresnel propagator.
"""
from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.special import j0, j1, eval_hermite, eval_genlaguerre, airy

C0 = 299_792_458.0
EPS0 = 8.8541878128e-12

LAMBDA = 1064e-9
K = 2*np.pi/LAMBDA
POWER = 1.0
W0 = 10e-3
ZMAX = 100e3


def gaussian_radius(z):
    zr = np.pi*W0**2/LAMBDA
    return W0*np.sqrt(1+(z/zr)**2)


def gaussian_peak(z):
    return 2*POWER/(np.pi*gaussian_radius(z)**2)

# Bessel parameters selected so the central lobe is approximately 10 mm.
BESSEL_THETA = 0.05e-3
BESSEL_KR = K*np.sin(BESSEL_THETA)
BESSEL_APERTURE_RADIUS = 0.20
BESSEL_ZMAX = BESSEL_APERTURE_RADIUS/np.tan(BESSEL_THETA)
BESSEL_CORE_RADIUS = 2.405/BESSEL_KR

# Finite-energy Airy scale and apodization.
AIRY_X0 = 0.05
AIRY_A = 0.08
AIRY_Z0 = K*AIRY_X0**2

# Circular top-hat aperture radius.
TOPHAT_RADIUS = W0


def bessel_intensity(r, z):
    """Finite-aperture Bessel-beam intensity approximation, W/m^2."""
    r = np.asarray(r); z = np.asarray(z)
    radial = j0(BESSEL_KR*r)**2
    finite_envelope = np.exp(-((np.maximum(z, 0)-BESSEL_ZMAX)/BESSEL_ZMAX)**2 * (z > BESSEL_ZMAX))
    return (2*POWER/(np.pi*BESSEL_CORE_RADIUS**2))*radial*finite_envelope


def airy_field(x, z):
    """Finite-energy Airy field along one transverse coordinate."""
    s = np.asarray(x)/AIRY_X0
    xi = np.asarray(z)/AIRY_Z0
    arg = s - xi**2/4 + 1j*AIRY_A*xi
    Ai = airy(arg)[0]
    return Ai*np.exp(AIRY_A*s - AIRY_A*xi**2/2 + 1j*(s*xi/2 - xi**3/12 + AIRY_A**2*xi/2))


def airy_intensity(x, z):
    field = airy_field(x, z)
    # Normalize to 1 W total in the plotted 1-D slice for comparison only.
    return POWER/(np.sqrt(2*np.pi)*AIRY_X0) * np.abs(field)**2


def tophat_intensity(r, z):
    """Circular top-hat aperture: near field to far-field Airy-pattern blend."""
    r = np.asarray(r); z = np.asarray(z)
    z_transition = K*TOPHAT_RADIUS**2/2
    q = np.sqrt((r/(TOPHAT_RADIUS+1e-30))**2 + (z/(z_transition+1e-30))**2)
    x = K*TOPHAT_RADIUS*r/(np.maximum(z, 1e-12))
    airy_ff = np.ones_like(x, dtype=float)
    nz = np.abs(x) > 1e-12
    airy_ff[nz] = (2*j1(x[nz])/x[nz])**2
    near = POWER/(np.pi*TOPHAT_RADIUS**2) * (r <= TOPHAT_RADIUS)
    far = POWER/(np.pi*TOPHAT_RADIUS**2) * (z_transition/np.maximum(z, 1e-12))**2 * airy_ff
    blend = np.clip(z/z_transition, 0, 1)
    return (1-blend)*near + blend*far


def lg01_intensity(r, z):
    w = gaussian_radius(z)
    rho = np.sqrt(2)*np.asarray(r)/w
    # Normalization is chosen so the total transverse power is approximately P.
    return 4*POWER/(np.pi*w**2) * rho**2*np.exp(-rho**2)


def hg10_intensity(x, z):
    w = gaussian_radius(z)
    u = np.sqrt(2)*np.asarray(x)/w
    # 1-D cut at y=0; normalization is relative to the Gaussian reference.
    return gaussian_peak(z)*2*u**2*np.exp(-u**2)


def sample_profiles(z_value, r):
    return {
        "Bessel J0": bessel_intensity(r, z_value),
        "Airy": airy_intensity(r, z_value),
        "Top-hat aperture": tophat_intensity(r, z_value),
        "Laguerre–Gaussian LG01": lg01_intensity(r, z_value),
        "Hermite–Gaussian HG10": hg10_intensity(r, z_value),
    }


def on_axis_curves(z):
    # For LG01 and HG10, axis intensity is zero by mode symmetry.
    return {
        "Bessel J0": bessel_intensity(0, z),
        "Airy": airy_intensity(0, z),
        "Top-hat aperture": tophat_intensity(0, z),
        "Laguerre–Gaussian LG01": np.zeros_like(z),
        "Hermite–Gaussian HG10": np.zeros_like(z),
    }


def make_plots(outdir):
    outdir.mkdir(exist_ok=True)
    z = np.linspace(0, ZMAX, 700)
    r = np.linspace(-0.15, 0.15, 900)
    colors = ['#2563eb','#ea580c','#16a34a','#9333ea','#0891b2']
    names = list(on_axis_curves(z))

    fig, ax = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    for color, name in zip(colors, names):
        ax[0,0].plot(z/1000, on_axis_curves(z)[name], color=color, lw=1.8, label=name)
        ax[0,1].plot(r*1000, sample_profiles(0, r)[name], color=color, lw=1.8, label=name)
        ax[1,0].plot(r*1000, sample_profiles(ZMAX, r)[name], color=color, lw=1.8, label=name)
    ax[0,0].set(title='On-axis irradiance versus distance', xlabel='z (km)', ylabel='I(0,z) (W/m²)', yscale='log')
    ax[0,1].set(title='Transverse profile at z = 0', xlabel='x or r (mm)', ylabel='irradiance (W/m²)')
    ax[1,0].set(title='Transverse profile at z = 100 km', xlabel='x or r (mm)', ylabel='irradiance (W/m²)', yscale='log')
    for a in ax.ravel(): a.grid(alpha=0.3); a.legend(fontsize=8)
    ax[1,1].axis('off')
    ax[1,1].text(0.03, 0.95, 'Model parameters\n\nλ = 1064 nm\nP = 1 W\nGaussian reference w₀ = 10 mm\nBessel cone angle = 0.05 mrad\nBessel aperture radius = 0.20 m\nAiry scale x₀ = 50 mm\nTop-hat radius = 10 mm\nRange = 0–100 km\n\nLG01 and HG10 are dark on axis\nbecause they are higher-order modes.', va='top', fontsize=12)
    fig.suptitle('Non-Gaussian paraxial beam comparison', fontsize=16)
    fig.savefig(outdir/'other_beams_comparison.png', dpi=180); plt.close(fig)

    # Line maps of normalized irradiance versus x and z.
    zmap = np.linspace(0, ZMAX, 500); xmap = np.linspace(-0.15, 0.15, 420)
    X,Z = np.meshgrid(xmap,zmap)
    fields = [bessel_intensity(X,Z), airy_intensity(X,Z), tophat_intensity(X,Z), lg01_intensity(X,Z), hg10_intensity(X,Z)]
    fig, axes = plt.subplots(3,2, figsize=(14,13), constrained_layout=True)
    for a, name, field in zip(axes.ravel(), names, fields):
        im = a.pcolormesh(Z/1000, X*1000, np.maximum(field,1e-10), shading='auto', cmap='magma')
        a.set_title(name); a.set_xlabel('z (km)'); a.set_ylabel('transverse coordinate (mm)')
        a.set_yscale('linear'); fig.colorbar(im, ax=a, label='W/m² (log color scale)')
        im.set_norm(LogNorm(vmin=1e-10, vmax=max(1, np.max(field))))
    axes[-1,-1].axis('off'); fig.suptitle('Beam propagation cross-sections', fontsize=16)
    fig.savefig(outdir/'other_beams_propagation_maps.png', dpi=180); plt.close(fig)


def export_csv(outdir):
    z = np.linspace(0, ZMAX, 1001); curves = on_axis_curves(z)
    with open(outdir/'other_beams_on_axis.csv','w',newline='') as f:
        w=csv.writer(f); w.writerow(['z_m']+list(curves))
        for i, zi in enumerate(z): w.writerow([zi]+[curves[n][i] for n in curves])
    with open(outdir/'other_beams_parameters.csv','w',newline='') as f:
        w=csv.writer(f); w.writerow(['parameter','value','units'])
        for row in [('wavelength',LAMBDA*1e9,'nm'),('power',POWER,'W'),('Gaussian_waist_radius',W0*1e3,'mm'),('Bessel_cone_angle',BESSEL_THETA*1e3,'mrad'),('Bessel_finite_length',BESSEL_ZMAX/1000,'km'),('Bessel_core_radius',BESSEL_CORE_RADIUS*1e3,'mm'),('Airy_scale',AIRY_X0*1e3,'mm'),('top_hat_radius',TOPHAT_RADIUS*1e3,'mm')]: w.writerow(row)


def main():
    outdir = Path(__file__).resolve().parent/'outputs_other_beams'; outdir.mkdir(exist_ok=True)
    print('Other-beam simulation: λ=1064 nm, P=1 W, range=0–100 km')
    print(f'Bessel finite nondiffracting length: {BESSEL_ZMAX/1000:.2f} km')
    print(f'Bessel central-core radius estimate: {BESSEL_CORE_RADIUS*1000:.2f} mm')
    print(f'Airy characteristic length z0: {AIRY_Z0/1000:.3f} km')
    make_plots(outdir); export_csv(outdir); print(f'Saved outputs to {outdir}')

if __name__ == '__main__': main()
