# Physical limitations affecting radio interpretation

These are open consistency issues, unquantified numerical uncertainties and
declared model assumptions, not the storage, timestamp or spectrum bugs already
repaired. The [architecture guide](RREA_WARPX_AMREX_ARCHITECTURE_GUIDE.tex)
defines the full model, including transport, chemistry and optical limitations;
this note focuses on interpreting the radio and screening results.

## 1. Charge injection and Gauss-law consistency

Scheduled particles add charge without a corresponding injection current or
electric-field initialization satisfying Gauss's law. The coupling deposits the
post-injection population and calls `PrimeMaxwellContinuityReference`, so its
material-continuity check does not test the injection event itself. See
[the coupling](../native_amrex_rrea/warpx_overlay/Source/Rrea/RreaWarpXCoupling.cpp)
and the [source/restart guide](RREA_RERUN_GUIDE.md).

Short local controls checked on September 5, 2026 gave these final diagnostics
in `repair_control/rrea_reduced.csv` and
`repair_subthreshold/rrea_reduced.csv`, respectively:

| Control | Simulated time | `em_gauss_residual_rel` | `em_charge_continuity_max_rel` |
| --- | ---: | ---: | ---: |
| Zero imposed field | 40 ns | 0.352 | 2.37e-16 |
| Subthreshold imposed field | 20 ns | 0.511 | 2.53e-16 |

These are normalized diagnostic residuals, **not fractional radio-amplitude
errors**, and they do not quantify the long production run's error. They show
that passing material continuity alone does not establish electromagnetic
consistency. The physical charge/current supply must be specified and checked
across injection before drawing quantitative screening conclusions; resetting
the diagnostic would not resolve the problem.

## 2. Sources outside the Huygens surface

The recording surface is inset from the computational boundary, while the
default PARMA column reaches the domain endcaps. Physical currents can therefore
exist outside the reconstruction surface. The surface-only free-space integral
does not account for those exterior sources, so its output is not yet established
as the complete exterior field. Their particle-weight fraction is not a bound
on their radio contribution. Assess those sources and the exterior-medium
assumptions; surface-position comparisons are meaningful only within a
source-free region enclosing the same physical sources.

## 3. Assumed zero prehistory

[The reducer](../scripts/compute_rrea_em_radio.py) sets each contribution to zero
before its first recorded E or B sample. This requires the omitted history to
be negligible on each field's own time axis. A nonzero initial field introduces
a discontinuity relative to that assumed history; the interval-slope operator
does not include its impulsive derivative. Timestamp and endpoint repairs cannot
recover unrecorded history. Establish initial-state compatibility or supply the
missing history before interpreting the onset.

## 4. Production resolution and boundary reflections

Analytic field tests support the implemented operator, not production accuracy.
Spatial resolution, source timestep, azimuthal quadrature, output cadence and
macroparticle sampling need separate refinement checks in the relevant regime.
In particular, particle weights affect the nonlinear density-conductivity-field
coupling. The optional azimuth check reports E/B refinement differences; it does
not establish convergence of the complete simulation or its band energies.

The first-order Silver-Mueller absorbing boundary can also leave reflections in
the recorded fields. Their effect on production waveforms has not been bounded;
late-time content needs boundary/domain sensitivity checks, not automatic removal.

## 5. Geometry, propagation and instrument scope

The simulation is axisymmetric (`m=0`) and omits the geomagnetic field. It cannot
represent azimuthally localized avalanches, asymmetric feedback or general
three-dimensional magnetic deflection. The radio reconstruction assumes free
space: ground reflection, ionosphere, terrain and antenna gain are absent.
Aircraft plate projections and nominal filter responses are instrument scenarios,
not a calibration. These are declared model choices, not repaired code defects.

## 6. Electric-field energy is a conditional estimate

The plotted E-only energy estimate uses the plane-wave relation
`S = c * epsilon_0 * E^2`; the general vacuum flux is
`S = E cross B / mu_0`. Coulomb and induction fields need not satisfy the
plane-wave relation. The exact-axis field in this axisymmetric model is
longitudinal, with zero axial Poynting flux, not propagating radio. Correct FFT
normalization does not remove these interpretation limits.

## Scientific priority

Resolve the injection/Gauss consistency and exterior-source treatment first,
then quantify production convergence, startup sensitivity and boundary effects.
Use governing constraints and independent physical checks; agreement with an
older waveform or a successful cluster job is not a substitute.
