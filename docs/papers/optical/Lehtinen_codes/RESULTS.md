# Verification results

Verified on 22 September 2026 with Python 3.13.11, NumPy 2.3.5, SciPy 1.17.0 and Matplotlib 3.10.8.

All six presets passed the separate solver and output-sampling criteria. The table uses the tight solution on the finest checked output grid; preset output defaults are the coarsest tested grids passing two successive refinements.

| Preset | Default samples | Finest checked samples | Peak NR (m⁻³) | Peak radio at 10 km (V/m) | Filtered peak (V/m) | Gamma yield >300 keV | Optical yield |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GRG | 100,000 | 399,997 | 0.719412 | 9.629274e-08 | 9.629274e-08 | 2.01205e+16 | 6.839475e+17 |
| FGF | 100,000 | 399,997 | 177.6157 | 0.01110725 | 0.011107 | 9.187297e+16 | 3.122999e+18 |
| FGFstrong | 100,000 | 399,997 | 4319.206 | 2.292209 | 2.280385 | 6.923462e+18 | 2.353463e+20 |
| TGF | 100,000 | 399,997 | 160411.4 | 0.1402779 | 0.1399687 | 3.377005e+15 | 1.147931e+17 |
| TGF2 | 399,997 | 1,599,985 | 5053492 | 0.2057322 | 0.204914 | 1.607165e+15 | 5.463166e+16 |
| TGF3 | 199,999 | 799,993 | 3534100 | 3.798458 | 3.703338 | 4.04639e+16 | 1.375472e+18 |

Yields are total source photons in the modeled time interval. They include the retained, provisional volume cutoff. The optical yields include the air-mixture density correction and are not calibrated to an experimental yield.

## Accuracy evidence

- Production tolerances: `rtol=1e-9`, `atol=[1e-7, 1e-14, 1e-3]` for `[Ecloud, NR, Nion]`.
- Tight reference: same method, `rtol=1e-11`, absolute tolerances divided by 100.
- Independent check: the other of LSODA and Radau at the tight tolerances; an additional solve bounds internal steps to `tau/100`.
- Sampling checks refine a single tight dense solution, starting at 100,000 points and doubling intervals. Every observable must change by less than 0.5% on two successive refinements. Runaway peaks above 1% of each run’s maximum are matched chronologically, with equal peak counts and timing within two fine-grid samples.

| Preset | Largest solver comparison error (%) | Default-grid vs finest error (%) | Significant runaway peaks |
| --- | ---: | ---: | ---: |
| GRG | 3.02996e-06 | 8.82513e-06 | 1 |
| FGF | 4.62586e-07 | 4.14501e-05 | 4 |
| FGFstrong | 8.64324e-05 | 0.0036154 | 22 |
| TGF | 9.00611e-07 | 0.127842 | 1 |
| TGF2 | 7.15845e-06 | 0.165205 | 2 |
| TGF3 | 7.75307e-06 | 0.342711 | 3 |

All solver comparisons gave matching significant-peak counts and sampled peak times. Dominant-peak shifts and individual amplitude/yield errors are retained in [convergence.json](convergence.json). These are numerical comparisons, not uncertainty estimates for the physical model.

TGF2 and TGF3 required finer default grids. At 100,000 points their unfiltered radio peaks were 0.2012827 and 3.751484 V/m; the finest checks gave 0.2057322 and 3.798458 V/m. Increasing output samples cannot replace tighter integration tolerances.

One tight Radau run emitted a SciPy internal step-size warning (division by zero in its error-ratio predictor); the solve completed with finite, admissible states and agreed with tight LSODA. The expected legacy-volume warnings occurred in threshold-crossing presets.

## Operational and equation checks

`MPLBACKEND=Agg python -m unittest -v test_model` passes 16 tests, covering current/field consistency across optional branches and both field signs, source activation, distance scaling, failed integrations, resampling bounds, mixture normalization, threshold limits, cross-section broadcasting and valid-energy masking, and streamer transport.

The streamer equation is checked against constant-growth analytic solutions (positive, zero and negative growth) in both flow directions, and against an independently integrated varying-coefficient solution. A localized-source test verifies that outgoing flux equals integrated production when growth is zero; internal steps are bounded by input-grid spacing to prevent skipping the source. Full streamer validation remains blocked by the missing companion module and excitation table.

The FGF poster runs end to end in a fresh output directory. Full and runaway-only plots save successfully; the dynamics figure was visually inspected. Final TGF2 and TGF3 default-grid runs also satisfy the 0.5% radio-peak criterion against the fine references.

Reproduce these data with `MPLBACKEND=Agg OPENBLAS_NUM_THREADS=1 python check_convergence.py`. See [README.md](README.md#physical-limitations) for unresolved physical assumptions and [the optional streamer section](README.md#optional-streamer-calculation) for missing scientific inputs.
