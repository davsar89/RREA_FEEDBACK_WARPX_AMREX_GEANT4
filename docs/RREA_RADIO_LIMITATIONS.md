# Physical limitations affecting radio interpretation

These are open consistency issues, unquantified numerical uncertainties and
declared model assumptions. The [architecture guide](RREA_WARPX_AMREX_ARCHITECTURE_GUIDE.tex)
defines the full model, including transport, chemistry and optical limitations;
this note focuses on interpreting the radio and screening results.

## 1. Charge injection

Injected seeds radiate as neutral pairs created in place, current turn-on
included (model in the architecture guide): right for locally produced
secondaries. `em_gauss_residual_rel` measures their counter-charge, not a field
error (0.35-0.51 in zero-field controls, where nothing else is present).

## 2. Sources outside the Huygens surface

The recording surface is inset from the computational boundary, while the
default PARMA column reaches the domain endcaps. Physical currents can therefore
exist outside the reconstruction surface, and the surface-only free-space
integral omits them entirely: a synthetic dipole placed 1 km outside the
surface contributes 0.25 V/m of its 210 V/m direct field at 100 km. The
integral returns the enclosed sources only, so charge crossing the surface is
seen to vanish there. Their particle-weight fraction is not a bound on their
radio contribution; compare the escaped charge (`netchord_escaped_charge_e`)
times its continuation length with the charge moment integral M dt.
Surface-position comparisons are meaningful only within a source-free region
enclosing the same physical sources.

## 3. Assumed zero prehistory

[The reducer](../scripts/compute_rrea_em_radio.py) sets each contribution to zero
before its first recorded E or B sample. Fresh captures include the initial
zero scattered state; older captures may start at step one. Omitted history must
be negligible on each field's own time axis. A nonzero initial field introduces
a discontinuity relative to that assumed history; the interval-slope operator
does not include its impulsive derivative, and no record can recover the
unrecorded history. Establish initial-state compatibility or supply the
missing history before interpreting the onset.

## 4. Production resolution and boundary reflections

Analytic field tests support the implemented operator, not production accuracy.
Two limits are known in closed form and cap every figure label and spectrum:
Yee dispersion (accumulated phase error of 0.1 rad across the grid) and the
azimuthal trapezoid of the surface integral (exact only while k r_side < N).
`compute_rrea_em_radio.resolved_max_hz` returns the smaller, about 0.8-0.9 MHz
for the production grids; content above it is numerical. Source timestep,
output cadence and macroparticle sampling still need refinement checks in the
relevant regime.
In particular, particle weights affect the nonlinear density-conductivity-field
coupling. The optional azimuth check reports E/B refinement differences; it does
not establish convergence of the complete simulation or its band energies.

Reflections from the first-order Silver-Mueller boundary are source-free inside
the Huygens surface, so the equivalence integral removes them for exterior
observers to discretisation accuracy. The boundary's real effect is on the
fields themselves: for a static field it drives the tangential scattered E to
zero within a light-crossing time, i.e. it acts as a grounded surface whose
image charges weaken the screening field. The production domain spans
6-16 km: the bottom plane is 4.9 km below the runaway layer and the top
2.8 km above it (a higher top puts the curl-free taper's fringe field above
the thinner air's runaway threshold), so a top-plane bias of a few percent
remains unquantified.

## 5. Geometry, propagation and instrument scope

The simulation is axisymmetric (`m=0`) and omits the geomagnetic field. It cannot
represent azimuthally localized avalanches, asymmetric feedback or general
three-dimensional magnetic deflection. The radio reconstruction assumes free
space: ground reflection, ionosphere, terrain and antenna gain are absent. For
a vertical-dipole far-field scenario, a perfectly conducting ground image
doubles the vertical E and horizontal B (four times the plane-wave spectral
energy); this is not a universal correction for arbitrary sources. At the 100 km receiver the first ionospheric hop
arrives about 0.2-0.3 ms after the direct wave (reflection height 70-85 km,
source near 12 km), inside a sub-millisecond event. Band and instrument filters are causal Butterworth
responses, so filtered waveforms carry a real receiver's delay.
Aircraft plate projections and nominal filter responses are instrument scenarios,
not a calibration. These are declared model choices.

## 6. Electric-field energy is a conditional estimate

The plotted E-only energy estimate uses the plane-wave relation
`S = c * epsilon_0 * E^2`; the general vacuum flux is
`S = E cross B / mu_0`. Coulomb and induction fields need not satisfy the
plane-wave relation. The exact-axis field in this axisymmetric model is
longitudinal, with zero axial Poynting flux, not propagating radio. Correct FFT
normalization does not remove these interpretation limits.

## Scientific priority

Quantify source convergence first (macro count, timestep, material targets):
it sets the LF content, which the fluid conduction current dominates. Then
assess exterior sources per run, startup sensitivity and boundary effects.
Use governing constraints and independent physical checks; agreement with an
older waveform or a successful cluster job is not a substitute.
