# Olivia: one to four GH200s

The project directory `warpx_rrea_smoke` also holds production runs and must
not be treated as disposable test output. GPU nodes have ARM64 hosts; a laptop or Olivia CPU-node
executable cannot run there. The job loads `NRIS/GPU`, GCC 13.3, CUDA 12.6,
CMake, Python and OpenMPI, builds for CUDA architecture 90, and runs one MPI
process per GPU on one node. `accel` consumes GPU hours, separately from the
CPU allocation. The selected OpenMPI module supports CUDA buffers.
[Sigma2 job settings](https://documentation.sigma2.no/jobs/job_scripts/olivia_job_scripts.html)
and [accounting](https://documentation.sigma2.no/jobs/projects_accounting.html).

Authentication and downloads use the existing `scripts/olivia.py` helper;
see `OLIVIA_CONNECTION_AND_2FA.md`. Complete the local production build and
native physics tests required by `AGENTS.md` before submitting a build or run.

## Build

`sync` transfers committed HEAD only. Commit intended changes before syncing,
or explicitly `push` each changed file for a working-tree experiment. The GPU
build needs the pinned upstream clones that `olivia.py build` (the CPU build)
creates.

After the checks in `OLIVIA_COMMAND_WORKFLOW_NOTES.md` (Before submitting):

```text
python scripts/olivia.py script cluster/olivia/submit_gpu.sh WARPX_SOURCE=/cluster/projects/nn9526k/dsarria/warpx_rrea_smoke/upstream_git/davsar89-WarpX-8f3ee117692d AMREX_SOURCE=/cluster/projects/nn9526k/dsarria/warpx_rrea_smoke/upstream_git/davsar89-AMReX-bf15fdce5209 build
python scripts/olivia.py status <job-id> --tail 40
```

The upstream paths carry the gitlink commit prefixes; update both after a
gitlink change. The default build is
`/cluster/projects/nn9526k/dsarria/warpx_rrea_smoke/gpu_cuda_mpi/build` (MPI+CUDA).
Build jobs need only the default one GPU.
The build job runs `scripts/build_rrea_cuda.sh` (tests: `GPU_PORT.md`).
Read the Slurm exit status and test summary before launching a capture.

## Configure and run

Settings `KEY=VALUE` precede `run`; subsequent arguments go directly to the
production driver. There is no second set of physics defaults. For example:

```text
python scripts/olivia.py script cluster/olivia/submit_gpu.sh CPUS=48 WALLTIME=00:20:00 run --transport-omp-threads 48 --run-root /cluster/projects/nn9526k/dsarria/warpx_rrea_smoke/runs --output-dir /cluster/projects/nn9526k/dsarria/warpx_rrea_smoke/runs/gpu_short/capture --e0-peak-kv-per-m 120 --stop-time-s 1e-7 --dt-s 2.5e-9 --macro-count 50 --diag-interval 10 --checkpoint-interval 20
```

This is a short fresh capture, not the mature population of a long discharge.
PARMA's macro count is per initial time interval, so scale it with the stop
time: the production count over a short interval makes the weights tiny.
The diagnostic interval is shorter than the 40-step run so the reduced CSV
needed by analysis and capture continuation is produced.
The wrapper sets the managed arena and defaults host transport threads to the
allocated CPUs; an explicit transport-thread argument overrides this. The deck selects host kinetics and the
engine defaults field and fluid updates to the GPU. Pass
`--extra-warpx-arg rrea.kinetic_backend=gpu` for device kinetics. It uses one
GPU by default. Set `GPUS=2`, `GPUS=3` or `GPUS=4` before `run` to
use that many MPI ranks and GPUs on the same node. Slurm binds one GPU per
rank (`--gpus-per-task=1 --gpu-bind=single:1`); the driver launches the engine
with `srun` and PMIx. GPU count does not change the physical mesh or timestep.
Some field arrays and work remain replicated, so linear scaling is not assumed.
Use driver `--help` for mesh, seed, diagnostic and material-closure settings.
Change `GPUS`, `CPUS`, `MEMORY`, `WALLTIME`, `QOS`, `GPU_WORK` or `REPO_ROOT` before
`build`/`run` (defaults in `submit_gpu.sh`); use `QOS=nn9526k` (the project
default QoS) for jobs longer than `devel` allows. `exec <command> <arguments>` runs a custom
controller command on the allocated node (for example `exec nvidia-smi`).
That command owns any `srun` launches; do not start one Python controller per
GPU when all of them would write the same run directory.

The example above is the retained single-GH200 configuration: host kinetics
with 48 transport threads and GPU field/fluid updates.

For a scaling comparison, use the MPI executable for **every** GPU count,
including one. Keep the checkpoint, mesh, physics and diagnostics fixed; vary
only ranks/GPUs. Record per-rank CUDA device identities and CPU affinity.
Report both speedup `T1/Tn` and GPU cost per step `n*Tn`, excluding the same
startup window. A four-GPU allocation used to compare smaller steps still
bills all four reserved GPUs throughout; `n*Tn` describes the cost of a
production job requesting only its needed `n` GPUs.

Measured scaling and its limits are in the architecture TeX guide, section
"Execution backend and measured GPU scaling". `JOURNAL.md` locates the raw
commands, timings, telemetry and conservation diagnostics.

## Monitor, stop, retrieve, restart

```text
python scripts/olivia.py jobs
python scripts/olivia.py status <job-id> --tail 40
python scripts/olivia.py run "tail -n 10 /cluster/projects/nn9526k/dsarria/warpx_rrea_smoke/runs/gpu_short/capture/stdout.txt"
python scripts/olivia.py pull /cluster/projects/nn9526k/dsarria/warpx_rrea_smoke/runs/gpu_short/capture/rrea_reduced.csv tmp/gpu_short_reduced.csv
python scripts/olivia.py cancel <job-id>
python scripts/olivia.py pull-restart gpu_short --dest /home/david/rrea_runs
```

A continuation needs the checkpoint **and** the capture streams; restart
rules are in `RREA_RERUN_GUIDE.md`.

## Products on a GH200 host

Use the standard render/radio commands in `RREA_RERUN_GUIDE.md` through
`exec` on the capture host, then download the compact products. The ARM stack
needs an isolated Python environment: NumPy 2.2.6, SciPy 1.15.3 and Matplotlib
3.9.2 work together; unset `PYTHONPATH` so the older system NumPy cannot shadow
it. Load `FFmpeg/7.0.2-GCCcore-13.3.0` and activate
`$PRDR/dsarria/warpx_rrea_smoke/venv_products_gh200`. Set
`MPLBACKEND=Agg OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK`.
CloudScat automatically selects `external/cloudscat_cpp/build_gh200` on ARM64,
preserving the x86 `build` directory. The ordinary Python commands in the rerun guide
then work without a custom wrapper or Python overrides.
