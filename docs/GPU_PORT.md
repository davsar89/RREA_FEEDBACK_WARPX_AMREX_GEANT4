# Experimental hybrid CUDA execution

**Olivia GH200 operation:** see [OLIVIA_GPU.md](OLIVIA_GPU.md) for the
one-to-four-GPU MPI launcher, configuration, accounting and transfers.

**Record the backend actually measured.** The CUDA code defaults the custom
fluid, Maxwell and kinetic switches to the device, but the production deck
sets `rrea.kinetic_backend=cpu` for hybrid execution. Laptop measurements in the journal describe
their stated configurations, not GH200 performance.
`gpu_fluid=0` does not make every fluid operation host code: the face solver
still uses the compiled AMReX backend.

## Validation and performance

The architecture guide owns current timing tables, validation limits and the
retained GH200 configuration (section "Execution backend and measured GPU
scaling"); `JOURNAL.md` records retained/rejected rounds and evidence locations.

CPU and CUDA run the same kinetic kernels; the native smokes exercise both
backends. Compare backends through physical observables and their statistical
spread, not trajectory identity: sorting can alter newborn ID assignment and
therefore RNG realizations even when the equations are unchanged, and agreement
on one short trajectory does not validate rare-event rates or establish
physical accuracy.

Profile the actual workload with Nsight or AMReX. Arena allocation counts are
not CUDA API-call counts, and synchronization waits overlap device work.
Set `tiny_profiler.print_threshold=0` when inspecting call counts: omitted
regions may have been folded into `Other`, rather than never executed.

## Execution scope

| Component | CUDA path in this source |
| --- | --- |
| Custom RZ Maxwell solver | Field updates, absorbing boundaries, energy/Joule/Poynting reductions |
| Low-energy carrier transport | Drift/diffusion fluxes, positivity limiting, density updates, stability and finite-state checks |
| Electron fluid closure | Immutable numerical table view and shared interpolation evaluated in device kernels |
| Main field/fluid coupling | Current assembly, field handoff, time averaging, continuity and Gauss reductions |
| Initialization | Compact host-evaluated axial/radial profiles, device mesh fill |
| Event accumulator | Device fold from scratch arrays into mesh; event scatter remains host work |
| Energetic electrons, positrons and photons | GPU histories, immutable table views, original samplers, field/density gather |
| Secondary creation and newborn transport | Device history logs, count/pack, GPU newborn waves and validation; ordered deposition/ledger replay remains host work |
| Population control | GPU live records, constrained support plans, weight/ID commit and refill; MPI routing, support ordering/capacity and insertion orchestration remain host work |
| Checkpoints, file output and several diagnostics | Host consumers with explicit synchronization |

The existing reaction/conductivity AMReX kernels remain in use.

## GPU kinetics (`rrea.kinetic_backend=gpu`)

The energetic electron, positron and photon history loops dispatch through
`RreaKineticGpu.H`; the CPU reference calls the same `RreaKineticStep.H`
equations with the same counter-keyed random stream, certified table ranges,
cutoffs, material paths, field work, competing optical depths and strict guards.
One GPU thread advances one complete history in a bounded-size batch. Each
history builds its complete variable-length birth groups, payloads, sources,
ledger operations and plane crossings on device; GPU waves then advance
newborns for the remainder of their birth timestep, including their own
secondary production. Device reductions check birth-group conservation,
payloads and final-survivor validity before the collective commit decision.
Population control maps live particles to canonical CIC support records on the
GPU and runs the original stochastic null-space thinning per independent
support, preserving the four deposited CIC moments.

Birth groups are not overwritten with end-of-step survivors: birth transactions
and insertion states have distinct roles in conservation accounting. Stable
CPU ordering is retained where it determines lineage or transaction order.
Population changes are not committed from an invalid device plan. This is the
existing fail-stop collective transaction protocol, not a whole-step rollback
after a device failure.

## Memory and ownership

`RreaGpu.H` gives the standalone physics primitives a CPU path and a CUDA path.
CUDA field/current/closure buffers use managed allocations with pointer-only
kernel views. Host owning C++ vectors, strings, maps and configuration objects are not
passed to device kernels. AMReX `ParallelFor` and reductions operate on the
mesh; the custom Maxwell solver uses the same equations in both backends.

**Every production CUDA invocation must set this before AMReX initialization:**

```text
amrex.the_arena_is_managed=1
amrex.the_arena_init_size=0
```

The retained host transaction, diagnostic, particle and mesh consumers make device-only allocation unsafe
in this intermediate implementation. A startup guard rejects it rather than
assuming that a CPU can dereference a device pointer. Component tests
initialize their test arenas accordingly.

`RreaKineticSupport.H` supplies `KineticSpan`, `KineticVector` and the
device-callable algorithms of the shared physics. A `KineticVector` created on
the CPU owns CPU heap memory; GPU histories use a private arena slice (13 KiB by
default, released after packing) whose excess spills to the device heap, with
ownership tracking keeping arena and heap deallocation distinct. Nested device
pointers are never read or freed on the CPU. Each batch:

1. constructs each log, runs its history once and counts its records in one kernel;
2. copies counts to pinned host buffers, joins, and prefixes in history order;
3. uploads offsets, packs into non-managed device buffers (`NonManagedDeviceVector`)
   and destroys the nested logs in the same second kernel;
4. copies the flat outputs to pinned host buffers, joins, and rebuilds
   independent host-owned records for ordered replay.

Counts and vector growth are checked. Exhausting the device heap traps with an
error rather than silently dropping secondaries or truncating a history. The
batch size bounds concurrent histories, **not the number of secondaries a
physical history generates**; a very long history or very large support can
still exhaust memory. Restart with a smaller batch or a larger heap after
investigating; a failed production step is not retried.

GPU work is joined before host consumers, host MPI collectives, checkpoint I/O,
and disposal of temporary arrays. The coupling singleton is released before
AMReX destroys its memory arenas. These are explicit correctness boundaries,
not a promise of overlapping CPU/GPU computation.

The full-domain Maxwell arrays and current collective are still replicated
across MPI ranks. GPU transport gathers fields directly on device, but
host source replay, particle insertion orchestration, diagnostics and I/O can
still move shared pages. Dynamic device allocation and divergent histories can
also limit performance.

## Precision

Stored state, updates and kernel argument layouts remain binary64/integer POD.
The CPU population reduction and selected delicate scalar calculations keep
their local `long double` working arithmetic; device versions use binary64.
These types never occur in shared object layouts or non-templated signatures.
A device support plan must agree with the host capacity/rank reference, meet
its requested reduction, preserve the four CIC moments within the stored-precision
checks, and produce finite nonnegative weights. Rounding can select a different
but mathematically equivalent null-space endpoint. RNG counters depend on
logical identity, timestep, interaction index and channel, not GPU scheduling;
the CPU tests verify invariance under reversed history order.

## Runtime controls

Defaults and accepted ranges are owned by `RreaKineticGpu.H` and
`RreaPopulationGpu.H`.

| Parameter | Meaning |
| --- | --- |
| `rrea.kinetic_backend` | `gpu` or `cpu` energetic transport and associated population paths; GPU in a CPU build is rejected. |
| `rrea.gpu_kinetic_batch_size` | Maximum simultaneous history logs. |
| `rrea.gpu_kinetic_heap_mib` | Requested minimum CUDA device heap, separate from AMReX arena storage. |
| `rrea.gpu_kinetic_arena_kib` | Per-history arena capacity in KiB; 0 selects heap-only. |
| `rrea.gpu_kinetic_arena_spill` | Permit allocations beyond the arena capacity to use the device heap; disabled, exhaustion aborts. |
| `rrea.gpu_kinetic_arena_audit` | Print cumulative demand, pool use and spill counts; use only for attribution. |
| `rrea.gpu_population_batch_size` | Maximum simultaneous support planners. |
| `rrea.gpu_population_cpu_fallback` | Re-evaluate a failed device rank/moment/progress plan with the CPU reference; 0 is strict device acceptance. |

Population batches additionally target at most 262144 input records, except
that an individual larger support is kept intact: splitting an independent
support would change the constrained reduction and RNG order. These settings
are starting heuristics, not measured GH200 optima. An actual numerical
fallback emits `RREA_GPU_POPULATION_FALLBACK supports=N`; it handles a rejected
numerical plan, not a memory fault. Record fallback counts when benchmarking,
since apparent GPU execution could otherwise mask host population work.

The production driver accepts these controls, except the three arena ones, plus
the `rrea.gpu_fluid` and `rrea.gpu_maxwell` 0/1 device switches (default 1,
owned by `RreaGpu.H`), as separate `--extra-warpx-arg key=value` tokens.
`rrea_run_support.py` owns the allowlist, including the creation-map diagnostic.
`rrea.kinetic_backend=cpu` selects host kinetics in a CUDA executable; the
field/fluid paths follow their own CUDA switches. Host OpenMP serves hybrid
and reference/fallback execution, not CUDA kernels.

## Build on a CUDA development machine

Use a Git checkout and its upstream gitlinks (`git submodule update --init`);
a source ZIP does **not** contain the AMReX or WarpX dependency sources, and
the `.gitmodules` branch names are not commit pins.

Supply the compatible existing source trees at `upstream/AMReX-26.06` and
`upstream/WarpX-26.06`, or pass their paths to the builder:

```bash
WARPX_SOURCE=/absolute/path/to/WarpX-26.06 \
AMREX_SOURCE=/absolute/path/to/AMReX-26.06 \
CUDA_ARCH=90 JOBS=4 \
bash scripts/build_rrea_cuda.sh
```

`90` is the intended GH200 build target. Use the actual target architecture for
another GPU. The build requires a compatible CUDA toolkit/host C++ compiler,
CMake, Python, zlib and host OpenMP support, as well as the dependencies required
by the pinned WarpX tree. The script does not install a toolchain, fetch external
sources or submit scheduler jobs.

The builder prepares the real WarpX overlay, selects CUDA with double precision,
disables fast math and IPO, and builds the application, native library and tests.
Kernel-owning translation units are explicitly compiled as CUDA with C++20.
`RREA_CUDA_HOST_OMP=ON` independently retains threading for the kinetic CPU
reference/fallback; it does not change AMReX's GPU/OpenMP launch policy.
The standalone builder defaults to `MPI=OFF`; see the script for its job
limits.

The work tree defaults to `tmp/rrea_cuda/`. The script runs real native CTest
cases serially, excluding the four-rank fixtures. It fails immediately on a failed build or test. Cluster submission is in
`OLIVIA_GPU.md`.

## First execution and validation

The native test wrapper supplies the managed-memory argument. For a manually
prepared WarpX command, append it yourself, and for GPU kinetics also
`rrea.kinetic_backend=gpu` and, for a first acceptance exercise,
`rrea.gpu_population_cpu_fallback=0` (the runtime default is 1).

Use the tested short-capture command in `OLIVIA_GPU.md`, then set the
scientifically appropriate duration, mesh and seed configuration through the
production driver. The same guide owns GPU assignment and checkpoint transfer.

Before interpreting results from this backend, run the complete native CUDA
suite, including the real WarpX transport/continuity fixtures. The
GPU-specific targets `rrea_kinetic_engine_smoke`, `rrea_kinetic_storage_smoke`
and `rrea_population_gpu_smoke` contain real GPU dispatches in a CUDA build
(complete live and newborn histories, count/pack/destruct, birth and survivor
validation, support planning with fallback disabled, on deliberately
non-aligned batch sizes). Run them under Compute Sanitizer. On
the same physical problem, inspect charge/current closure, the field-energy
ledger, carrier positivity, causal propagation, population/energy spectra,
avalanche statistics, secondary birth timing, and mesh/timestep convergence,
with several seeds. GPU reduction order can change roundoff; require
physical/numerical consistency rather than byte identity. Exercise
checkpoint/restart in both directions before relying on cross-backend
continuation. Checkpoint transfer in both directions has been exercised at one
to three GPUs; larger rank counts and long-run convergence remain unmeasured.

## Local checks and their limits

The field/fluid primitives have standalone numerical programs, run with GCC and
with Clang address/undefined-behavior sanitizers. The kinetic runner compiles
the actual numerical sources and production table loader without upstream
AMReX, through the guarded scalar adapters in
`native_amrex_rrea/tests/scalar_types/` (aliases, abort/assert helpers; no
meshes, particle containers, launches, MPI or GPU runtime; never used by the
CMake targets). It covers tables, samplers, live and newborn histories,
lossless transaction packing, storage ownership, path clipping, secondary
groups, charged interleave, CIC resampling, and a population fixture with
degenerate geometry and routing checks.

```bash
CXX=g++ bash scripts/test_rrea_gpu_primitives.sh
CXX=clang++ SANITIZE=1 bash scripts/test_rrea_gpu_primitives.sh
CXX=g++ bash scripts/test_rrea_gpu_kinetics_cpu.sh
CXX=clang++ SANITIZE=1 bash scripts/test_rrea_gpu_kinetics_cpu.sh
CXX=clang++ BINARY64=1 bash scripts/test_rrea_gpu_kinetics_cpu.sh
```

Binary64 CPU runs exercise the device arithmetic choice but do not emulate CUDA
math libraries or instruction rounding.

## Source map

| File or family | Role |
| --- | --- |
| `include/rrea/RreaInteractionPhysics.H`, `RreaInteractionTypes.H` | Immutable numerical views and shared samplers |
| `include/rrea/RreaKineticSupport.H` | Cross-execution scalar helpers, spans, owning scratch vectors and algorithms |
| `include/rrea/RreaKineticFieldView.H`, `RreaFieldGatherScalar.H` | Device field gather and shared interpolation |
| `warpx_overlay/Source/Rrea/RreaKineticStep.H` | Shared live/newborn transport equations |
| `RreaKineticGpu.H`, `RreaKineticRecords.H` | Launches, bounded batches, device ownership and packing |
| `RreaSecondaryCommit.cpp`, `RreaSecondaryValidation.H` | Newborn GPU waves, validation and collective transaction integration |
| `RreaPopulationGpu.H`, `RreaPopulationRecord.H`, `RreaPopulationParticles.H` | GPU records, constrained support planning, commit/refill |
| `RreaCouplingPopulation.cpp`, `RreaParticleSweep.H` | Production population dispatch, routing and commit integration |
| `include/rrea/RreaCicConservingResample.H` | Shared four-moment stochastic reducer |

References: the AMReX GPU guide
(https://amrex-codes.github.io/amrex/docs_html/GPU.html) and the NVIDIA CUDA
programming guide for device-heap ownership and language limits; use the
versions matching the installed toolkit and pinned AMReX tree.

## Remaining full-device and scaling work

Further device-residency work includes ordered source
and deposition replay, candidate preparation, insertion orchestration and
support routing/ordering. A distributed Maxwell/current implementation is a
separate multi-GPU design task. Tune queues/allocations and host-device boundaries using
representative profiles rather than inferring performance from kernel count.
