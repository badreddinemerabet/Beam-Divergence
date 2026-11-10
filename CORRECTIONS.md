# Corrections and validation

This source accompanies a fresh run of all 16 original Python entry points.
MATLAB parity fixes are included but were not
executed because MATLAB is not available in this environment.

## Changes

- Airy power normalization uses its exact one-dimensional integral, squared
  for the separable two-dimensional beam. This removes repeated 2400 x 2400
  numerical normalizations and the truncated-window power error.
- Airy diagnostics, animations, and transverse maps use the same 2-D field
  and W/m² units.
- Top-hat near/far diagnostic interpolation clamps the far-field evaluation
  distance before blending, removing the artificial 1/z source singularity.
  Both components carry the specified total power. This is still an
  interpolation, not a Fresnel diffraction solver.
- LG01 normalization and the missing y dependence of HG10 are corrected in
  the remaining Python animation/maps and MATLAB comparison scripts.
- The Bessel-Gaussian diagnostic uses an exact scaled-Bessel normalization
  integral, avoiding repeated quadrature and supporting array-valued distances.
  The common comparison now explicitly uses this finite-power diagnostic;
  it must not be interpreted as an exact hard-aperture Bessel propagator.
- Recommender training and test sets are split by complete scenario, with
  training-only scaling. Dataset selection accuracy uses test scenarios only.
  Tied minimum scores count as correct selections.
- LG/HG recommender scores use the same Gaussian envelope parameter as the
  plotting scripts; they no longer introduce an extra factor of two into zR.
  These are envelope width scores, not second-moment widths of all modes.
- CLI parameters reject nonpositive/nonfinite values and M² below one.
- Dependencies now require NumPy >=2.0 and explicitly include Pillow.

## Scope and limits

The Gaussian equations and finite-energy Airy field are paraxial free-space
models. The Bessel/Bessel-Gaussian and top-hat diagnostics retain explicitly
approximate propagation descriptions. Recommender labels remain heuristic
characteristic-width scores, not a universal common beam-quality measure.
Receiver aperture is currently an unused physical input retained for API
compatibility and excluded from neural features; the ranking is not received-power optimization. Atmospheric
effects and a common diffraction/receiver model are outside this correction.

Run `python test_numerics.py` for regression checks. Run the original scripts
as before; execute generate_beam_training_data.py before
train_beam_recommender_dataset.py, and train before recommend_beam.py.

The two training commands overwrite the same model/metrics files. The run
bundle preserves the first training outputs under initial-training/ and the
final dataset-trained model under outputs_beam_recommender/.

Older README descriptions below are superseded by this correction note where
they describe old output names, numerical normalization, or validated physics.


Turbo update: all intensity colormaps changed to Turbo and affected plotting scripts reran successfully. Numerical CSV files are byte-for-byte unchanged. The GIF uses line plots and has no colormap. MATLAB palette edits were not executed.
