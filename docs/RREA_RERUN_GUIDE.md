# RREA Run and Validation Guide

The one simulation driver is `scripts/run_rrea_profiled_video_capture.py`.
Local and OLIVIA runs use the same canonical values in
`config/rrea_defaults.json` and the same profiled atmosphere path.

## Before a run

Clone with the dependency forks and their exact commits:

```text
git clone --recurse-submodules <this-repository-url>
```

For an existing clone, run `git submodule update --init`. The gitlinks under
`upstream/` are the single version pins; `.gitmodules` points at the `davsar89`
forks. Local source preparation uses the WarpX submodule by default; the OLIVIA
build helper uses the pinned WarpX/AMReX refs and commits.

`upstream/Geant4-10.7.4` pins the reference source.
`tools/generate_charged_tables` extracts restricted ionization stopping,
hard rates and complementary secondary distributions as reference CSVs.
Assembling a runnable cutoff bundle remains manual, using the existing
`scripts/rrea_table_pack.py`; this is not a full-bundle generator.
Running the shipped tables needs no Geant4.

The packed schema-6 bundle is tracked and travels with every clone and
`olivia.py sync`; there is no separate table transfer. Its semantics belong to
the C++ loader (`RreaInteractionTables::Load`), which rejects a bundle the
engine cannot run.

1. Prepare a fresh source/build directory from the pinned submodules, compile
   the full production WarpX RZ executable, and run the registered RREA tests.
   Run `python3 -m pytest -q scripts` under Linux/WSL for Python checks;
   Windows skips POSIX launcher fixtures. These do not replace native tests.
   `scripts/prepare_rrea_warpx_source.py` defaults to the pinned WarpX
   submodule; configure CMake with `-DWarpX_DIMS=RZ`,
   `-DWarpX_amrex_src=<repo>/upstream/AMReX-26.06`,
   `-DRREA_AMREX_OVERLAY=<repo>/native_amrex_rrea`, and
   `-DRREA_AMREX_BUILD_TESTS=ON`. Limit the build to 12 parallel jobs.
2. For OLIVIA, follow the pre-submission steps in
   `OLIVIA_COMMAND_WORKFLOW_NOTES.md`.

## Seed source

The seed scenarios are described in the README. The PARMA source
(`scripts/rrea_parma_source.py` owns the physics; standalone audit via
`python scripts/rrea_parma_source.py --probe <build>/bin/rrea_interaction_table_probe`)
is configured by the `parma_continuous_source` block: latitude/longitude,
minimum energy, `source_radius_m`, and `macro_count`, one knob for stream and
bath resolution together.

Both production scenarios include this background. `parma_plus_1mev_pulse`
adds one initial 1 MeV electron pulse: `--macro-count`,
`--injected-real-electrons`, `--channel-radius-m` and `--seed-time-window-s`
describe the pulse; its PARMA macro count stays configuration-owned.
The default is still `parma_continuous_column`, where `--macro-count` controls
the PARMA interval. C&D has no background. Combined-run continuations retain
the pulse and bath once and append only future PARMA events; old pulse-only
captures are not converted. Continuous seeding does not guarantee new pulses.

A FRESH PARMA start builds the vendored Fortran evaluator with gfortran,
queries the ensemble drag through `rrea_interaction_table_probe` (built beside
`warpx.rz`), and stores both as `parma_source_table.txt` in the run
directory; every restart regenerates the schedule from that stored table
alone, so continuations on any machine need neither PARMA, gfortran nor the
probe.

The first requested duration fixes the PARMA sampling interval: extending a
run preserves its bath, particle weights, initial splitting threshold and
previous events, and appends independently permuted intervals at the same
sampling rate. `--macro-count`
is the budget per initial interval, not a lifetime cap. Partial intervals
retain that resolution; their weighted counts approximate rate times duration
to within the discrete event weights. Zero/subthreshold PARMA controls are
allowed and have no Coleman-Dwyer multiplication reference.

## Local capture

Pass the executable, output root and field amplitude to the one driver. The
tracked schema-6 bundle is the default `--transport-config`; the remaining
physics comes from the canonical defaults.

```text
python scripts/run_rrea_profiled_video_capture.py \
  --warpx-exe <warpx.rz> \
  --run-root runs_rrea \
  --output-dir runs_rrea/<case> \
  --e0-peak-kv-per-m <E0>
```

## Restart

Use the same driver with `--restart-checkpoint`. Machine, launcher and paths
change freely. Implicit checkpoints live at `<output-dir>/chkNNNNNN`; `auto`
searches only that namespace (or the explicit prefix's parent). Supply an
explicit path for an older sibling checkpoint. Declare a **rank-count, engine or physics-model change** with
`--restart-epoch-break --restart-epoch-reason <short_reason>`; pre- and
post-epoch diagnostics are then not one homogeneous trajectory. The opt-in
permits supported rank-count and reduced-schema deviations; incompatible state
or mesh formats remain fatal. This is the sanctioned route for moving a
checkpoint between OLIVIA and this laptop at a different rank count.

`command.json` preserves the **first** launch. Read the attempt named by
`command_in_progress.json` for the active command, or by `command_exit.json`
for the completed attempt and its exit status. Its record under
`command_attempts/` contains the resolved stop time, source, engine and layout.

A reduced-CSV schema change is an epoch deviation: the engine archives the old
stream as `rrea_reduced_pre_<step>.csv` itself. Growing a capture past its
original stop time retains all scheduled frames and their indices. Changing
`--video-frame-interval-s` changes only newly appended snapshots, on the new
grid from t=0 and strictly after the restart step. A shorter capture timeline
is refused. The render map follows the requested spacing throughout; older
views use the nearest retained state and record its actual capture time.
Snapshots use float32 display fields with lossless shuffle/zlib level-6
compression; the checkpoint retains full simulation precision.

Changing frame spacing retains earlier raw frames. Approved thinning must keep
the frame stream, offsets, capture indices, schedule, render map, moments and
derived series aligned at their actual times. Creation-history members and the
checkpoint's `history_bytes` must also agree, as must video write counts and
cursors. Work on a copy, preserve particle/field state and physical counters,
and verify a production restart that writes a frame and COMPLETE checkpoint
before removing the dense source. Preserve approved products and radio sampling.

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

Time-integrate the reduced CSV's weighted-mean series for an exposure; the
engine does not invent values between diagnostic rows.

## Coleman-Dwyer campaign

`cluster/olivia/submit_rrea_cd_campaign.py` plans the unit-weight campaign on
CPU nodes. It keeps its fixed field grid, 10 us observation window, complete-realization statistics,
all-physics interactions and separate seed cohorts. The planner selects cases;
the driver derives and validates the C&D seeds, transport RNG seed and
population ceilings.

The observable is defined in the architecture guide (C&D check); a completed
endpoint also reports the 8/9/10 us downstream-flux growth and measured
plane-arrival speed.

At unit weight the cost scales with M (about 1 GPU-h per realization at
90 kV/m, about 75 at 105 kV/m; 119 kV/m does not fit one GH200). M is an
expectation, so the default check holds the population at a fixed size with
the production controller and runs each case on one GH200 in the retained
hybrid layout, about 0.5 GPU-h at any field:

```text
B=/cluster/projects/nn9526k/dsarria/warpx_rrea_smoke/runs/cd_resampled
python scripts/olivia.py script cluster/olivia/submit_gpu.sh CPUS=48 QOS=devel WALLTIME=02:00:00 run \
  --transport-omp-threads 48 \
  --run-root "$B" --output-dir "$B/e105_r0" --e0-peak-kv-per-m 105 --dt-s 2.5e-9 \
  --stop-time-s 1e-5 --video-capture-final-time-s 1e-5 \
  --video-nr 64 --video-nz 128 --seed-source-model coleman_dwyer_7p2mev \
  --seed-realization-id 0 --cd-plane-diagnostics --checkpoint-interval 500 \
  --cd-allow-population-resampling --cd-population-resampling-reason bounded_population_cd_check \
  --population-target-electron-macros 100000 --overwrite-output
```

The report keeps resampled and unit-weight ensembles in separate groups. A
resampled result stands on the 90 kV/m bridge: its M must match the
unit-weight M within the quadrature sum of the two 95% CI half-widths
(`JOURNAL.md` records it).
A timestep ladder may run engineering realizations 0-2 with
`--cd-diagnostic-seed-count <n> --cd-diagnostic-reason <text>`: the report
accepts its paired dt ratio as timestep evidence but never pools it with, or
promotes it to, a scientific ensemble.

## Products

For accumulated electron-creation maps, enable
`--extra-warpx-arg rrea.electron_creation_bin_m=25` before a fresh run, then use:

```text
python3 scripts/plot_rrea_electron_creation.py <capture>/rrea_video/electron_creation_history.bin.gz --output e125_videos/creation.png
```

The architecture guide defines the observable and safe live-history retrieval;
script `--help` describes adaptive and explicit time intervals.

For linear/log electron lightcurves with time-linked near-axis field profiles,
run `python3 scripts/plot_rrea_electrons_and_field.py <capture> <figure.png>`.
An `.npz` output extracts compact data on the capture host; pass that file as
the input to render locally. `--panels` sets the snapshot count;
`--scan-step-us` sets analysis sampling only. Snapshots resolve changes in
field-profile shape, always including the initial frame, electron peak and
latest frame. The field is the saved magnitude in the first radial cell,
as in the standard video; every panel includes the 284 kV/m threshold scaled
by the run's density relative to the project's standard-air reference.

Unqualified video/radio requests use these defaults: one six-panel MP4
(electron, ion, photon, positron, absolute field, field fraction), with the
population and optical strips, and the standard multiband radio figure.
The video optical strip uses cloud-transported irradiance for completed
captures and source rates for previews; the radio figure includes cloud
transport by default. Keep the renderer's fixed scales, full scheduled timeline and
default encoding. Explicit requests can override these choices.

From the repository root, use the existing tools; no product wrapper is needed:

```bash
capture=/path/to/capture
products=/path/to/products
mkdir -p "$products"
python3 scripts/extract_rrea_band_series.py "$capture"
while read -r altitude offset tag; do
    python3 scripts/compute_rrea_em_radio.py "$capture" --time-stride 4 \
        --observer-altitude-km "$altitude" --observer-offset-km "$offset" \
        --out "$capture/rrea_video/rrea_em_observer_$tag.csv"
done <<'OBSERVERS'
0 100 ground
20 5 aircraft
20 0 nadir
OBSERVERS
python3 scripts/plot_rrea_radio_multiband.py "$capture" --output "$products/radio_multiband.png"
python3 scripts/render_rrea_profiled_video.py "$capture" --output-dir "$products/video"
```

Recompute the band and observer series after extending a capture. They must
cover the current record before drawing the figure. Check the MP4 with a full FFmpeg
decode and inspect the final frame and figure before delivery.

What the multiband figure draws is defined in the architecture guide (radio
scope); interpretation limits are in `RREA_RADIO_LIMITATIONS.md`.

Reduce and render on the capture host before pulling large products. Current
OLIVIA commands are indexed by `docs/OLIVIA_COMMAND_WORKFLOW_NOTES.md`;
GPU launch and transfer instructions are in `docs/OLIVIA_GPU.md`.
