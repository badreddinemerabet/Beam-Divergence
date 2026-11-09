# Corrected beam models and z=0 comparison

The non-Gaussian source-plane comparison was corrected after an equation audit.

## Corrections

- All five comparison beams are now represented as 2-D transverse irradiance fields `I(x,y,0)`.
- Every field is numerically normalized to exactly 1 W over the plotting aperture.
- LG01 now uses the correct 1 W radial normalization.
- HG10 is now a full 2-D two-lobe mode rather than a 1-D cut.
- Airy is now a separable 2-D finite-energy Airy source.
- Bessel uses a finite super-Gaussian aperture envelope rather than an infinite-power J0 field.
- The z=0 maps are available in both logarithmic and linear color scales.

## Run

```bash
python3 other_beams_sim.py
```

or:

```bash
python3 initial_beams_z0.py
```

## Outputs

- `outputs_other_beams/initial_beams_z0_corrected.png`
- `outputs_other_beams/initial_beams_z0_corrected_linear.png`
- `outputs_other_beams/z0_power_check.csv`
- `outputs_other_beams/other_beams_comparison_corrected.png`
- `outputs_other_beams/other_beams_on_axis_corrected.csv`

The power-check file should show approximately 1 W for every beam.

## Limitation

The corrected source-plane fields are suitable for a consistent z=0 comparison. The current propagation diagnostics remain analytical approximations. For reliable post-propagation divergence, these complex source fields should next be propagated with a Fresnel or angular-spectrum solver.
