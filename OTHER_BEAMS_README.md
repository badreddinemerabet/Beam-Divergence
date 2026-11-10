> Corrected edition: read [CORRECTIONS.md](CORRECTIONS.md) for current model conventions, limitations, and validation changes.

# Other beam types: unified comparison

This extension follows the requested order:

1. Bessel beam
2. Airy beam
3. Top-hat circular-aperture diffraction
4. Laguerre–Gaussian vortex beam
5. Hermite–Gaussian mode

The original Gaussian-beam simulation remains available separately.

## Baseline

The comparison uses a common wavelength of 1064 nm, total power of 1 W, and propagation range 0–100 km. Parameters are deliberately editable in both scripts.

- Bessel: cone angle 0.05 mrad, finite generating aperture radius 0.20 m.
- Airy: transverse scale `x0 = 50 mm`, exponential apodization `a = 0.08`.
- Top-hat: circular aperture radius 10 mm.
- LG01/HG10: Gaussian waist radius 10 mm.

## Run Python

```bash
cd /home/ubuntu/light_beam_divergence_sim
python3 -m pip install -r requirements.txt
python3 other_beams_sim.py
```

Outputs are written to `outputs_other_beams/`:

- `other_beams_comparison.png`
- `other_beams_propagation_maps.png`
- `other_beams_on_axis.csv`
- `other_beams_parameters.csv`

## Run MATLAB

```matlab
cd('/home/ubuntu/light_beam_divergence_sim')
other_beams_sim
```

MATLAB outputs include comparison plots, propagation maps, CSV tables, and `other_beams_results.mat`.

## Physical interpretation and limitations

The Bessel beam is modeled as a finite-aperture `J0` beam. An ideal infinite-energy Bessel beam does not exist physically; the finite aperture produces a finite nondiffracting range. With the baseline parameters, that range is approximately 4 km, not 100 km.

The Airy beam is a finite-energy paraxial solution. Its transverse acceleration and long tails are visible in the propagation map, but the model is not an atmospheric or finite-optics link-budget model.

The top-hat result uses a near-field circular aperture profile blended into a Fraunhofer Airy pattern. For accurate intermediate-distance results, replace it with a full Fresnel or angular-spectrum propagator.

LG01 and HG10 are higher-order Gaussian modes. Both have a central intensity null and therefore zero on-axis irradiance; their useful beam magnitude is the transverse ring or two-lobe structure rather than the axis value.

For future work, the next accuracy upgrade should be a common FFT-based angular-spectrum propagator with explicit source-plane fields, finite apertures, lenses, atmospheric phase screens, and detector apertures.

