# OLIVIA Workflow

Connection and 2FA are documented only in `OLIVIA_CONNECTION_AND_2FA.md`.
Use `python scripts/olivia.py --help` for the current command surface.
One-to-four-GH200 builds, runs and transfers are in `OLIVIA_GPU.md`.

## Before submitting

```text
python scripts/olivia.py jobs
python scripts/olivia.py quota
python scripts/olivia.py sync
```

Use one Slurm node and the module stack used for the build. The submitter loads
the MPI/thread layout from `config/rrea_defaults.json` (`olivia_layout`); the
capture keeps global AMReX OpenMP at one thread and scopes threading to the
RREA transport loops (`--transport-omp-threads`). `NTASKS=` and
`CPUS_PER_TASK=` override the layout. Re-measure scaling on a representative
current capture before changing it; results from another source, population,
mesh, or node generation are not transferable.
The requested wall time's full billing reservation must fit the remaining
quota; an over-large request remains pending even when a checkpoint exists.

## Build

```text
python scripts/olivia.py build --watch
```

The build archives committed HEAD and runs the native `ctest` physics smokes
against that commit's transport bundle.

## Capture

Submit through `cluster/olivia/submit_profiled_video.sh` or the C&D planner:
`KEY=VALUE` arguments set the submitter (`RUN_TAG`, `WALLTIME`, `MEMORY`,
`QOS`, `NTASKS`, `CPUS_PER_TASK`) and every other argument goes verbatim to
the driver, whose `--help` and `RREA_RERUN_GUIDE.md` describe its flags. Captures
under the project's persistent `runs/` directory stay there without a duplicate
archive; only node-local scratch captures are staged to another directory.

```text
python scripts/olivia.py script cluster/olivia/submit_profiled_video.sh \
  RUN_TAG=<tag> --warpx-exe <build>/bin/warpx.rz --e0-peak-kv-per-m <E0> \
  --checkpoint-interval <steps>
```

## Restart

Resubmit with the same `RUN_TAG` and the same driver flags plus
`--restart-checkpoint auto`; the submitter reuses an existing run
directory only for a restart. Epoch-break rules are in
`RREA_RERUN_GUIDE.md#restart`. Check `olivia.py jobs` first: the presence of a
run directory or checkpoint says nothing about whether a job is running.
For an intentional stop, confirm that a recent `rrea_checkpoint/COMPLETE`
exists before cancelling; `auto` will select the newest complete checkpoint
on the next submission.

## Products

Render and reduce on the capture host, then pull figures, CSVs and final MP4s.
Product definitions and interpretation limits live in `RREA_RERUN_GUIDE.md`.
Radio reduction needs NumPy and Numba; `RADIO_PYTHON` can select an environment
such as `warpx_rrea_smoke/venv_radio/bin/python`. The observer job bounds Numba
threads to its allocated CPUs and keeps BLAS serial.

To move a run here, use `olivia.py pull-restart <run>` rather than fetching the
checkpoint alone: the driver also requires the append-only streams it trims
back to the checkpoint's write counts. A long frame stream can be large and its
compressed size is data-dependent, so budget from the realized frame
metadata. Stage onto a Linux
filesystem, not `/mnt/c`, because the run steps and checkpoints there.
The reverse direction is `olivia.py push-restart <local_run> scratch_<tag>
auto`; both carry `parma_source_table.txt` when present. Epoch and stop-time
rules are in `RREA_RERUN_GUIDE.md`.
