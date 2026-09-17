# RREA Run and Validation Guide

The one simulation driver is `scripts/run_rrea_profiled_video_capture.py`.
Local and HPC runs use the same canonical values in
`config/rrea_defaults.json` and the same profiled atmosphere path.

## Before a run

Clone with the dependency forks and their exact commits:

```text
git clone --recurse-submodules <this-repository-url>
```

For an existing clone, run `git submodule update --init`. The gitlinks under
`upstream/` are the single version pins; `.gitmodules` points at the `davsar89`
forks. Local source preparation uses the WarpX submodule by default; the remote
HPC build helper uses the pinned WarpX/AMReX refs and commits.

`upstream/Geant4-10.7.4` pins the source for future schema-6 regeneration; the
generation tools still need restoration, and the existing tables' original
source commit is unverified. Download
[G4EMLOW 7.13](https://cern.ch/geant4-data/datasets/G4EMLOW.7.13.tar.gz)
separately, unpack it outside this repository, and set `G4LEDATA` to the
extracted `G4EMLOW7.13` directory. Running the shipped tables needs neither
Geant4 nor this raw dataset.

The packed schema-6 bundle is tracked and travels with every clone and
`hpc.py sync`; there is no separate table transfer. Its semantics belong to
the C++ loader (`RreaInteractionTables::Load`), which rejects a bundle the
engine cannot run.

1. Prepare a fresh source/build directory from the pinned submodules, compile
   the full production WarpX RZ executable, and run the registered RREA tests.
   `scripts/prepare_rrea_warpx_source.py` defaults to the pinned WarpX
   submodule; configure CMake with
   `-DWarpX_amrex_src=<repo>/upstream/AMReX-26.06`,
   `-DRREA_AMREX_OVERLAY=<repo>/native_amrex_rrea`, and
   `-DRREA_AMREX_BUILD_TESTS=ON`. Limit the build to 12 parallel jobs.
2. For the remote HPC run, check `hpc.py jobs` and `hpc.py quota`, then run
   `hpc.py sync`.

## Seed source

`seed_source_model` in `config/rrea_defaults.json` selects the source; the
default is `parma_continuous_column`: a continuous PARMA cosmic-ray
electron/positron column plus its stationary background at t = 0
(`scripts/rrea_parma_source.py` owns the physics; standalone audit via
`python scripts/rrea_parma_source.py`). Its `parma_continuous_source` block
sets latitude/longitude, minimum energy, `source_radius_m`, and
`macro_count` — one knob for stream and bath resolution together. The
mono-1-MeV pulse (`production_mono_1mev`) stays selectable, and Coleman-Dwyer
campaigns keep passing their model explicitly.

A FRESH PARMA start builds the vendored Fortran evaluator with gfortran and
stores the resolved flux table as `parma_source_table.txt` in the run
directory; every restart regenerates the schedule from that stored table
alone, so continuations on any machine need neither PARMA nor gfortran.
The table travels with `hpc.py pull-restart`/`push-restart`.

The first requested duration fixes the PARMA sampling interval: extending a
run preserves its bath, particle weights, initial splitting threshold and
previous events, and appends independently permuted intervals at the same
sampling rate. `--macro-count`
is the budget per initial interval, not a lifetime cap. Partial intervals
retain that resolution; their weighted counts approximate rate times duration
to within the discrete event weights. Zero/subthreshold PARMA controls are
allowed and have no Coleman-Dwyer multiplication reference.
Scheduled charge enters the continuity reference without an injection current.
Material continuity therefore does not establish Gauss-law closure across
injection; inspect the Gauss residual separately, particularly in zero-field controls.

## Local capture

Pass the executable, direct schema-6 transport configuration, output root and
field amplitude to the one driver. The remaining physics comes from the
canonical defaults.

```text
python scripts/run_rrea_profiled_video_capture.py \
  --warpx-exe <warpx.rz> \
  --transport-config rrea_transport_tables/schema6/production/transport_physics.json \
  --run-root runs_rrea \
  --output-dir runs_rrea/<case> \
  --e0-peak-kv-per-m <E0>
```

## Restart

Use the same driver with `--restart-checkpoint`. Machine, launcher and paths
change freely. Implicit checkpoints live at `<output-dir>/chkNNNNNN`; `auto`
searches only that namespace (or the explicit prefix's parent). Supply an
explicit path for an older sibling checkpoint. The native EM writer removes
torn tails and records beyond the restored state before appending, preserving
valid restart replacements. Declare a **rank-count, engine or physics-model change** with
`--restart-epoch-break --restart-epoch-reason <short_reason>`. The opt-in
permits supported rank-count and reduced-schema deviations; incompatible state
or mesh formats remain fatal. This is the sanctioned route for moving a
checkpoint between the HPC cluster and this laptop at a different rank count. The
checkpoint reader owns format, dimensions and state consistency; the recorded
configuration is provenance for interpretation, never a gate, and execution
paths, launcher and output cadences are not physics.
A reduced-CSV schema change is an epoch deviation: the engine archives the old
stream as `rrea_reduced_pre_<step>.csv` itself. Growing a capture past its
original stop time retains the stored late-frame cadence and appends new
frames, including a partial final interval, without moving existing capture
indices. Keep the early-window and warp settings unchanged. A shorter stop
time replaces the late window and is refused by the schedule check.

A restart that changes a physics closure must use an epoch break and record the
checkpoint time as a model discontinuity; pre- and post-epoch diagnostics are
not one homogeneous trajectory.

## Short convergence branches

Use fresh output directories and the same seed realization. Compare material
targets by setting conduction/reaction to `0.08`, `0.04`, then `0.02` and the
corresponding diffusion targets to `0.8`, `0.4`, then `0.2`:

```text
--material-conduction-target <target> \
--material-reaction-target <target> \
--material-diffusion-target <10*target>
```

Keep the hard guards at their resolved production values while tightening the
targets. Compare outer timesteps separately with `--dt-s 2.5e-9` and
`--dt-s 1.25e-9`. A closure
envelope branch selects its table with `--electron-closure-table <table.csv>`.
A short high-cap reference disables resampling with
`--population-target-electron-macros 0` and sets all three `--max-*-macros`
backstops to values justified by the run's memory headroom. For controller-only
RNG branches, `--population-controller-rng-salt <stream>` changes
only exact-CIC controller draws, while `--transport-rng-seed` changes the
physical transport realization.

The schema-9 reduced CSV records volume-, low-electron-, and source-weighted
E/N means and quantiles, material substeps and stiffness, controller exact
capacity/geometric floor, plus per-species effective sample size and maximum
macro weight. Time-integrate the weighted-mean series for an exposure; the
engine does not invent values between diagnostic rows.

## Coleman-Dwyer campaign

Use `cluster/hpc/submit_rrea_cd_campaign.py`. The campaign keeps its fixed
field grid, 10 us observation window, complete-realization statistics,
all-physics interactions and separate seed cohorts. The planner and driver
validate these values directly.

The primary observable is cumulative signed local-runaway plane flux, with
all-tracked signed flux as the recrossing audit. A completed endpoint also
reports the 8/9/10 us downstream-flux growth and measured plane-arrival speed.

## Products

- The standard multiband model-scope figure and separate direct off-axis
  Huygens CSV/metadata products.
- The standard six-panel MP4s.

`scripts/plot_rrea_radio_multiband.py` draws thin-slab populations, optical source
rates and the reduced Huygens observer waveforms when available. Otherwise it
uses explicitly labelled compact-source approximations; the point dipole
remains a comparison, not a ground-truth reference.
`compute_rrea_em_radio.py --observer-offset-km ...` evaluates the closed
Maxwell surface and is the direct free-space off-axis result. It applies no
ground reflection, ionosphere, terrain, or antenna gain; optional EFCM
responses are unit-gain and explicitly uncalibrated.

The standard video's left profile axis uses one fixed scale for the initial
radial mean, current radial mean, and red `r = 0` curve. The latter is the
first cell-centred radial sample, labelled `r = 0` by convention.

Reduce and render on the capture host before pulling large products. Current
HPC commands live only in `docs/HPC_COMMAND_WORKFLOW_NOTES.md`.
