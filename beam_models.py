"""Shared finite-power diagnostic models (not a full diffraction solver)."""
import numpy as np
from scipy.special import airy, j0, j1, i0e

def positive(**values):
    for name, value in values.items():
        if not np.all(np.isfinite(value)) or np.any(np.asarray(value) <= 0):
            raise ValueError(f'{name} must be finite and positive')

def airy_norm(scale, apodization, power=1.0):
    """Exact normalization of separable Ai(s) exp(a*s) in two dimensions."""
    positive(scale=scale, apodization=apodization, power=power)
    integral=np.exp(2*apodization**3/3)/(2*np.sqrt(2*np.pi*apodization))
    return np.sqrt(power)/(scale*integral)

def airy_field(x,z,wavelength,scale,apodization):
    s=np.asarray(x)/scale; xi=np.asarray(z)/(2*np.pi/wavelength*scale**2)
    return airy(s-xi**2/4+1j*apodization*xi)[0]*np.exp(
        apodization*s-apodization*xi**2/2+1j*(s*xi/2-xi**3/12+apodization**2*xi/2))

def airy_intensity(x,y,z,wavelength,scale,apodization,power=1.):
    return np.abs(airy_norm(scale,apodization,power)*
        airy_field(x,z,wavelength,scale,apodization)*
        airy_field(y,z,wavelength,scale,apodization))**2

def top_hat(r,z,wavelength,radius,power=1.):
    """Bounded, power-preserving near/far interpolation, not Fresnel diffraction.

    Both components integrate to power on the infinite transverse plane.
    Clamping the far-field distance before blending avoids a 1/z spike.
    """
    positive(wavelength=wavelength,radius=radius,power=power)
    z=np.asarray(z)
    if np.any(~np.isfinite(z)) or np.any(z<0): raise ValueError('z must be finite and nonnegative')
    ztr=np.pi*radius**2/wavelength; zeff=np.maximum(z,ztr)
    u=2*np.pi/wavelength*radius*np.asarray(r)/zeff
    disk=np.ones_like(u,dtype=float)
    np.divide(2*j1(u),u,out=disk,where=np.abs(u)>1e-12)
    near=power/(np.pi*radius**2)*(np.abs(r)<=radius)
    far=power/(np.pi*radius**2)*(ztr/zeff)**2*disk**2
    blend=np.clip(z/ztr,0,1)
    return (1-blend)*near+blend*far

def bessel_gaussian(r,z,wavelength,cone,envelope,power=1.):
    """Normalized diagnostic ansatz with a fixed Bessel scale."""
    w=envelope*np.sqrt(1+(np.asarray(z)/(np.pi*envelope**2/wavelength))**2)
    kr=2*np.pi/wavelength*np.sin(cone)
    # Exact radial integral: pi*w^2/2 * exp(-t)*I0(t).
    norm=power/(np.pi*w**2/2*i0e(kr**2*w**2/4))
    return norm*j0(kr*np.asarray(r))**2*np.exp(-2*np.asarray(r)**2/w**2)
