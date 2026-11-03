# Gaussian beam divergence simulation: near-IR and mid-IR

This project compares free-space Gaussian beams over a propagation range of 0–100 km. It contains equivalent Python and MATLAB implementations.

## Baseline assumptions

The initial theoretical case uses:

| Beam | Wavelength | Waist radius `w0` | Power | Beam quality `M²` |
|---|---:|---:|---:|---:|
| Near-IR | 1064 nm | 10 mm | 1 W | 1 |
| Near-IR | 1550 nm | 10 mm | 1 W | 1 |
| Mid-IR | 3.5 µm | 10 mm | 1 W | 1 |
| Mid-IR | 10.6 µm | 10 mm | 1 W | 1 |

Here `w0` is the radius at which the waist irradiance has fallen to `1/e²` of its on-axis value. `M² = 1` is an ideal diffraction-limited beam. For a real laser, replace it with the measured beam-quality factor.

## Model

For wavelength `λ`, waist radius `w0`, power `P`, and beam-quality factor `M²`:

```text
Rayleigh range:          zR = π w0² / (M² λ)
Beam radius:             w(z) = w0 sqrt(1 + (z/zR)²)
Far-field half-angle:   θ = M² λ / (π w0)
Full-angle divergence:  2θ
Irradiance:              I(r,z) = 2P/[π w(z)²] exp[-2r²/w(z)²]
Electric-field magnitude |E| = sqrt[2I/(c ε0)]
Magnetic-field magnitude |H| = |E|/η0
```

The quoted divergence is the far-field half-angle. At finite distances, the exact beam radius `w(z)` should be used rather than the simple approximation `w0 + θz`.

## Python

Install dependencies:

```bash
cd /home/ubuntu/light_beam_divergence_sim
python3 -m pip install -r requirements.txt
python3 gaussian_beam_sim.py
```

The script prints a summary and creates an `outputs/` directory containing:

- `beam_comparison.png`: radius, on-axis irradiance, waist transverse profile, and on-axis electric-field magnitude.
- `beam_intensity_maps.png`: logarithmic 2-D irradiance cross-sections, which show both the narrow waist and weak 100 km tail.
- `beam_summary.csv`: one row per beam with Rayleigh range, divergence, and 100 km radius.
- `beam_propagation.csv`: 1001 samples per beam with radius, irradiance, `|E|`, and `|H|`.

Edit `make_beams()` in `gaussian_beam_sim.py` to change wavelength, waist, power, or `M²`.

## MATLAB

Open MATLAB, change to this folder, and run:

```matlab
gaussian_beam_sim
```

The MATLAB script creates matching plots and exports:

- `outputs/beam_comparison_matlab.png`
- `outputs/beam_intensity_maps_matlab.png`
- `outputs/beam_summary_matlab.csv`
- `outputs/beam_propagation_matlab.csv`
- `outputs/beam_results.mat`

Edit the parameter block near the top of `gaussian_beam_sim.m`.

## Important interpretation notes

- This is a scalar, paraxial, free-space model. It does not include atmospheric turbulence, absorption, thermal blooming, pointing jitter, lens aberration, aperture clipping, or detector noise.
- In this baseline, every beam has the same waist and power. Therefore, longer wavelengths diverge more strongly because `θ ∝ λ`.
- Total power remains constant during ideal propagation; peak irradiance decreases as the beam expands approximately as `1/w(z)²`.
- The electric and magnetic fields are inferred from the local irradiance using plane-wave relations. Near a focus, at an aperture, or for strongly nonparaxial beams, a full vector diffraction model is more appropriate.
- For an engineering link budget, the received power also requires an aperture model. The present files provide the incident irradiance and beam size needed for that extension.
