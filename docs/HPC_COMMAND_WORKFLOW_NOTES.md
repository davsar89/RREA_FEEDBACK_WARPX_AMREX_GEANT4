# HPC Workflow

Connection and 2FA are documented only in `HPC_CONNECTION_AND_2FA.md`.
Use `python scripts/hpc.py --help` for the current command surface.

Site identity is private configuration (`HPC_LOGIN`, `REMOTE_ROOT`,
`HPC_SLURM_ACCOUNT`, `HPC_SLURM_MODULES` in credentials.env or the
environment); `scripts/hpc.py` forwards it to every remote command and sbatch
job. Nothing here names a specific cluster.

## Before submitting

```text
python scripts/hpc.py jobs
python scripts/hpc.py quota
python scripts/hpc.py sync
```

Use one Slurm node and the module stack your site uses for the build
(`HPC_SLURM_MODULES`). The submitter loads the normal MPI/thread layout from
`config/rrea_defaults.json`; it keeps global AMReX OpenMP at one thread and
scopes threading to the RREA transport loops. `NTASKS` and
`CPUS_PER_TASK` are explicit launcher overrides; `PROFILED_VIDEO_OMP_THREADS`
controls global OpenMP and `PROFILED_VIDEO_TRANSPORT_OMP_THREADS` scopes
threading to transport. Re-measure scaling on a representative current
capture before changing the layout; results from another source, population,
mesh, or node generation are not transferable.
The requested wall time's full billing reservation must fit the remaining
quota; an over-large request remains pending even when a checkpoint exists.

## Build

```text
python scripts/hpc.py build --watch
```

The build archive contains committed HEAD, including the tracked transport
bundle, and runs the semantic transport and native physics checks.

## Capture

Submit through `cluster/hpc/submit_profiled_video.sh` or the C&D planner. Do not
submit the capture sbatch without explicit layout values. There is no
short-leg chaining; if the job ends before `stop_time_s`, resubmit with the same
`RUN_TAG` and `PROFILED_VIDEO_RESTART_FROM=auto`. The C&D planner may
still schedule separate campaign cases as separate jobs.

The capture uses:

- `scripts/run_rrea_profiled_video_capture.py`;
- canonical values from `config/rrea_defaults.json`;
- one direct transport configuration path;
- one profiled field/density path.

For short material-convergence branches, the sbatch forwards
`PROFILED_MATERIAL_{CONDUCTION,DIFFUSION,REACTION}_{TARGET,GUARD}` and
`PROFILED_MATERIAL_{MIN,MAX}_SUBSTEPS` to the same driver. Outer-timestep,
closure-table, population-cap and RNG flags are described once in
`RREA_RERUN_GUIDE.md`.

## Restart

Use `PROFILED_VIDEO_RESTART_FROM=auto` for an interrupted capture; the
submitter reuses an existing scratch directory only for that literal value,
and refuses an explicit `chkNNNNNN` path there even though the driver itself
accepts one. Paths and
machine change freely. A **rank-count, engine or physics-model change** requires
`PROFILED_VIDEO_RESTART_EPOCH_REASON=<short_token>`; the submitter translates
that to the driver's epoch-break arguments.
Keep the same `RUN_TAG`, and check `hpc.py jobs` first: the presence of a
scratch directory or checkpoint says nothing about whether a job is running.
For an intentional stop, confirm that a recent `rrea_checkpoint/COMPLETE`
exists before cancelling; `auto` will select the newest complete checkpoint
on the next submission.

## Products

Render and reduce on the capture host, then pull figures, CSVs and final MP4s.
Product definitions and interpretation limits live in `RREA_RERUN_GUIDE.md`.

To move a run here, use `hpc.py pull-restart <run>` rather than fetching the
checkpoint alone: the driver also requires the append-only streams it trims
back to the checkpoint's write counts. A long frame stream can be large and its
compressed size is data-dependent, so budget from the uncompressed estimate
printed at launch and the realized frame metadata. Stage onto a Linux
filesystem, not `/mnt/c`, because the run steps and checkpoints there.
The reverse direction is `hpc.py push-restart <local_run> scratch_<tag>
auto` (run it inside WSL); both carry `parma_source_table.txt` when present.
Pass the epoch-break reason for an engine or rank-count change. Keep the same
stop time or extend it without changing any already-captured frame.

Render and observer reductions are host-side jobs
(`cluster/hpc/slurm_rrea_profiled_video_render.sbatch`,
`cluster/hpc/slurm_rrea_em_observers.sbatch`), submitted by hand with
`REMOTE_ROOT` and the job-specific input directories in the environment.