# Optical and Gamma Emissions from FGF

Research code for runaway-electron feedback in atmospheric electric fields and associated radio, gamma-ray and optical source emission. The dynamical states are cloud electric field, runaway density and ion density. Numerical verification and its limits are recorded in [RESULTS.md](RESULTS.md).

## Run

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install numpy==2.3.5 scipy==1.17.0 matplotlib==3.10.8
python -i egu2025poster.py FGF --output-dir outputs
```

These package versions were tested with Python 3.13.11. On Windows activate with `.venv\Scripts\Activate.ps1`. Use `plt.show()` in the interactive session to display figures, or run with `MPLBACKEND=Agg` for headless execution.

The example saves dynamics and radio PDFs, plots cumulative source gamma production and approximate optical irradiance, and prints gamma yield and peak optical rate. Output directories are created automatically. Filenames include the preset, atmospheric density and initial-field ratio; use separate directories for other parameter studies to avoid overwriting figures.

The presets are `GRG`, `FGF`, `FGFstrong`, `TGF`, `TGF2` and `TGF3`. Their settings and verified sample counts are defined in `egu2025poster.PRESETS`. `configure_case(name)` resets the model settings; `get_results(name, **solver_options)` also initializes and solves. Unknown names raise `ValueError`.

Imports do not run examples, load optional streamer data or save figures. `model_helpers.py` supplies local midpoint and formatting helpers; the unavailable original `utils.py` is not required.

## How the optical emission calculation works

The code estimates light produced when very fast electrons collide with air and cause nitrogen molecules to glow. The calculation is in `get_337()` in [fgf_module.py](fgf_module.py).

The chain is:

**Fast electrons → slower secondary electrons → excited nitrogen → light**

1. **Fast electrons knock electrons out of air molecules.** These newly released electrons are called *secondary electrons*. The code estimates how many are produced from the simulated runaway-electron density. Here, “fast electrons” means runaway electrons represented by an energy of **7.3 MeV**, or **7.3 million electron-volts**.

2. **Some secondary electrons give energy to nitrogen molecules.** The nitrogen becomes *excited*: it temporarily holds extra energy. The code represents the useful secondary electrons by an energy of **16 eV** and approximates their production over a **10 eV-wide energy interval**. It does not calculate the full electron energy distribution or follow individual electrons. The representative runaway energy is about 450,000 times larger than this secondary-electron energy.

3. **Excited nitrogen can release that energy as light.** The calculation targets nominal **337 nm emission**, which is ultraviolet, just outside visible light. Some molecules lose their energy through collisions instead of emitting light. The code includes this loss, called *quenching*.

4. **The code adds up emission over an estimated cloud volume.** The result is a rate in **photons per second**—how many packets of light the source produces each second.

The example then estimates brightness at **10 km**. It assumes light spreads equally in every direction, divides by the area of a sphere at that distance, and converts photon counts into **watts per square metre**.

The secondary-electron and excited-nitrogen populations are assumed to adjust immediately, so this calculation cannot predict a separate optical delay. Absorption and scattering inside the cloud are absent. The emitting-volume estimate also remains unresolved: it currently switches emission off below the field threshold. The calculated absolute brightness is therefore provisional.

## Python interface

```python
import numpy as np
import fgf_module as FGF
from egu2025poster import configure_case

configure_case('FGFstrong')
FGF.figdir = 'outputs'
FGF.init()
solution, elapsed = FGF.solve_odes()
values = FGF.sol2values(solution)
gamma = FGF.get_Fgrays(values['NR'], values['E'], minTGFeV=300_000.)
optical = FGF.get_337(values['E'], values['NR'])
radio, filtered = FGF.get_Erad(values['t'], values['Jtot'], D=20_000.)
FGF.save_figure(FGF.plot(values), formats=['pdf'])
np.savez_compressed('outputs/FGFstrong_results.npz', **values,
                    gamma_rate=gamma, optical_rate=optical,
                    radio_field=radio, radio_field_filtered=filtered)
```

Configuration uses module attributes. Call `init()` after changing physical parameters, and keep them unchanged while postprocessing a solution. External-source activation is derived from nonzero `KEext` and `KJext`, including negative amplitudes; do not assign `external_field` or `external_current` independently.

| Setting | Meaning / units |
| --- | --- |
| `Katm` | Air number density / Loschmidt reference number density |
| `Lcloud` | Cloud length, m; transverse area is approximated as `Lcloud**2` |
| `E0Et` | Initial field / runaway threshold field |
| `KEext`, `KJext` | External-field and external-current amplitude factors |
| `Eext_shape`, `Jext_shape` | `sigmoid`, `gaussian`, `wideramp`, `ramp` or `twosigmoids` |
| `tau`, `T1`, `T2`, `tau_cloud` | Driving, integration and background relaxation times, s |
| `S` | Cosmic-ray runaway source, m⁻³ s⁻¹ |
| `do_feedback`, `use_ions` | Enable feedback / evolve ions |
| `minimal_model` | Ion-only conduction, accumulated ions, instantaneous no-feedback runaways |
| `ode_method`, `num_time_points` | SciPy solver method / output sample count |

`solve_odes(*, rtol=1e-9, atol=(1e-7, 1e-14, 1e-3), max_step=np.inf)` returns `(OdeResult, elapsed_seconds)`. Absolute tolerances follow **[Ecloud (V/m), NR (m⁻³), Nion (m⁻³)]**. They control integration independently of output sampling. `init_and_solve(**solver_options)` is the combined workflow.

Incomplete or nonfinite results raise `RuntimeError`. Densities below minus ten component absolute tolerances are rejected; smaller negative values warn and remain unclipped. `sol2values(sol)` uses returned samples. Explicit resampling with `sol2values(sol, t=new_times)` requires increasing times within the integrated interval; changing `num_time_points` does not resample an existing solution.

| Result key | Quantity | Units |
| --- | --- | --- |
| `t` | Time | s |
| `E` | Total signed field, including imposed external field | V/m |
| `NR`, `Nion`, `Nsec` | Runaway, one-polarity ion and secondary-electron densities | m⁻³ |
| `JR`, `Jsec` | Signed runaway and secondary-electron currents | A/m² |
| `Jion` | Signed ion current; background Ohmic current when ions are disabled | A/m² |
| `Jtot` | Modeled conduction current, excluding imposed driving currents | A/m² |

The field equation is `epsilon_0*dEcloud/dt = J0 + Jext - Jtot`. `Jtot=JR+Jsec+Jion`, except in `minimal_model`, where `Jtot=Jion`. Shared current construction preserves the original 0.1 V/m magnitude floor and positive polarity at zero field.

`get_Erad(t, Jtot, D=1e4, theta=pi/2)` returns unfiltered and cloud-size-filtered fields in V/m. It requires a positive finite distance and matching, uniformly sampled arrays. The original zero derivative endpoints and periodic Gaussian FFT filter remain; inspect boundary effects when changing the time interval or driving waveform.

`get_Fgrays()` and `get_337()` return source photons/s; `get_volume()` returns m³. `plot()` returns a figure, including in runaway-only mode. `save_figure(fig, formats=(), suffix='')` supports PDF, PNG and pickle.

## Physical limitations

- **Production volume is unresolved.** The legacy approximation gives zero below the signed threshold, `Lcloud**3` at threshold, and `min(UR/(E-Et), Lcloud)*Lcloud**2` above it. Subthreshold or reversed fields warn. A feedback spatial-density model is needed to replace this discontinuity; the no-feedback downstream profile in `get_NRext()` does not establish that normalization.
- The air-mixture ionization source uses total air density; nitrogen excitation uses nitrogen density. The correction increases optical output by 25% at fixed state and volume, but the 16 eV secondary approximation, its lifetime, quenching and nominal 337 nm band assignment remain unvalidated. [AIRFLY’s measurement](https://arxiv.org/abs/1210.6734) is a benchmark, not an applied calibration factor.
- At fixed density and gamma threshold, optical and gamma source rates are proportional to `NR*get_volume(E)`, so their source pulse shapes are identical. Gamma production also assumes 7.3 MeV electrons. Neither calculation includes a detector response.
- Fitted feedback is supported only at `Katm=1`. Other densities and `constant_delta=False` raise `NotImplementedError`. The original density-scaled manual model remains available through `manual_feedback_parameters=True`, with different coefficients. Optional Dwyer refitting requires `Dwyer2007_Fig3_L_Egamma1.txt` beside the module.

## Verification

```bash
MPLBACKEND=Agg python -m unittest -v test_model
MPLBACKEND=Agg OPENBLAS_NUM_THREADS=1 python check_convergence.py
# Optional subset, keeping the full report intact:
MPLBACKEND=Agg python check_convergence.py FGFstrong TGF2 --output convergence_subset.json
```

The convergence script separately checks solver tolerance, another solver, an internal-step bound and output refinement. It exits nonzero on unmet criteria and saves completed cases. Acceptance criteria, measured results and environment are in [RESULTS.md](RESULTS.md); detailed measurements are in [convergence.json](convergence.json).

## Optional streamer calculation

`python -i optical_xray.py` requires `streamer_parameters_v1_9_3.py` on the import path and `ExcitationCoef_N2_C3Pi_u.txt` beside the script. Neither is supplied, so full streamer results remain unvalidated. The missing N2+ excitation-rate implementation raises `NotImplementedError`.

Without those inputs, `solve_runaway_transport(xi, growth_rate, source_rate, velocity)` can independently solve `velocity*dn/dxi = growth_rate*n + source_rate`. Position increases in metres; rates are s⁻¹ and m⁻³ s⁻¹, and velocity is a nonzero constant in m/s. Rates are interpolated linearly; steps are bounded by grid spacing to resolve localized sources. Zero incoming density is imposed at the left boundary for positive flow and the right boundary for negative flow. The example’s outgoing flux is relative to the moving boundary: `abs(Vcor)*n_out*area`.

`bethe.py` owns the shared bremsstrahlung cross section, also exposed by `fgf_module`: m² per molecule per unit dimensionless photon energy. Its `NR_to_gamma()` returns photons per metre per runaway. `fgf_module.sigma_i_general()` uses m²/eV per molecule.
