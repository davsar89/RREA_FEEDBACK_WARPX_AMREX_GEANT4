# Experimental CUDA energetic transport and population machinery

**Status: source implementation with CPU numerical validation. The full native
AMReX/WarpX CUDA build, GPU execution, and GH200 performance have not been
validated.** This extends the earlier hybrid field/fluid port; it is not a claim
that a complete timestep is now GPU-resident. Do not replace a validated
production installation before the native acceptance tests pass.

## What executes on the GPU in this implementation

The energetic electron, positron and photon history loops now dispatch through
`RreaKineticGpu.H` from the real WarpX overlay. The CPU reference calls the same
`RreaKineticStep.H` equations. This is not a second simplified collision model.
The same counter-keyed random stream, certified table ranges, cutoffs, material
paths, field work, competing optical depths, and strict guards remain in use.

| Operation | Implementation |
| --- | --- |
| Energetic electron/positron/photon histories | One GPU thread advances one complete history in a bounded-size batch; particle state and invalidation are updated on device. |
| Tables and samplers | Host loader builds immutable, pointer-only views into managed numerical storage; interpolation and final-state samplers are device callable. |
| Field and density lookup | Device-callable views gather the existing Maxwell/ambient fields and atmospheric profile using shared scalar equations. |
| Secondary creation | Each history builds complete variable-length birth groups, particle payloads, sources, ledger operations and plane crossings on device. |
| Transaction packing | Device count and pack passes flatten nested history records; host prefixes and replay order are stable. |
| Newborn transport | GPU waves advance newborns for the remainder of their birth timestep, including their own secondary production. |
| Validation | Device reductions check complete birth-group conservation, payloads and final-survivor validity before the collective commit decision. |
| Live population records | GPU maps live particles to canonical CIC support/phase records, excluding pending removals. |
| Constrained population reduction | Independent supports run the original stochastic null-space thinning algorithm on device, preserving four deposited CIC moments. |
| Live population commit and refill | GPU weight/ID updates after collective preflight; GPU exact-halving split decisions and parent updates. |

The charged models retain continuous losses, interleaved field work, elastic
scattering, hard electron-electron and positron-electron collisions, tracked
bremsstrahlung, and positron annihilation. Photons retain Compton,
photoelectric, nuclear-pair and triplet-pair channels. Rayleigh scattering is
not introduced. The production particle and field types remain double precision.

## What is deliberately still on the host

MPI ownership/routing and collective acceptance remain on the CPU. So do stable
support sorting and capacity preparation, the small per-history prefix pass,
ordered source/deposition/ledger replay, lineage allocation and insertion
orchestration through the existing WarpX API, candidate preparation/compaction,
several diagnostics, checkpoint serialization and I/O.

Birth groups are not overwritten with end-of-step survivors: birth transactions
and insertion states have distinct roles in conservation accounting. Stable
CPU ordering is retained where it determines lineage or transaction order.
Population changes are not committed from an invalid device plan. This retains
the existing fail-stop collective transaction protocol; it is not a whole-step
rollback mechanism after a device failure.

The Maxwell/current storage remains replicated across MPI ranks. The new kinetic
kernels do not implement a distributed multi-GPU Maxwell solver. Host replay,
managed-memory migration, variable history length, and dynamic allocation can
still dominate the application. No speedup factor has been established.

## Device allocation and transaction ownership

`RreaKineticSupport.H` supplies `KineticSpan`, `KineticVector` and device-callable
algorithms used by the shared physics. `KineticVector` created on the CPU owns
CPU heap memory; a vector created by a GPU history owns device heap memory.
They share source definitions, **not interchangeable allocation ownership**.

The implementation does not copy nested device-heap pointers to the CPU and
attempt to read or free them there. Each batch performs the following sequence:

1. Construct owning logs on the device and run each history once.
2. Count records on the device; prefix the small metadata arrays on the host.
3. Pack nested records on the device into runtime-owned flat managed buffers.
4. Synchronize and reconstruct separate host-owned records for ordered replay.
5. Destroy the original nested logs on the device before freeing the outer batch.

Counts and vector growth are checked. Exhausting the device heap traps with an
error rather than silently dropping secondaries or truncating a collision
history. The batch size bounds concurrent histories, **not the number of
secondaries generated by a physical history**. A very long history or a very
large population support can still exhaust memory. Restart with a smaller
batch or an appropriately larger heap after investigating the failure; a
failed production step is not automatically retried.

The first correctness-oriented implementation uses device `new[]`/`delete[]`
inside histories and support reductions. A measured follow-up optimization
could use pooled allocation or a work-queue design, but no untested fixed
per-particle secondary limit has been substituted here.

## Runtime controls

Set `amrex.the_arena_is_managed=1` on the command line **before AMReX is
initialized**. Retained host callbacks still access particle and mesh storage.
The runtime guard rejects device-only allocation. The native CUDA test wrappers
supply this setting.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `rrea.kinetic_backend` | `gpu` in CUDA; `cpu` otherwise | Selects energetic transport and associated GPU population paths. An explicit GPU request in a CPU build is rejected. |
| `rrea.gpu_kinetic_batch_size` | `8192` | Maximum simultaneous history logs; accepted range 1 through 1048576. |
| `rrea.gpu_kinetic_heap_mib` | `1024` | Requested minimum CUDA device heap size; minimum 8 MiB. This is separate from AMReX arena storage. |
| `rrea.gpu_population_batch_size` | `4096` | Maximum simultaneous support planners; accepted range 1 through 65536. |
| `rrea.gpu_population_cpu_fallback` | `1` | Re-evaluate a failed device rank/moment/progress plan using the CPU reference. Set 0 for strict device acceptance. |

Population batches additionally target at most 262144 input records, except
that an individual larger support is kept intact. Splitting an independent
support arbitrarily would change the constrained reduction and RNG order.
These settings are starting heuristics, not measured GH200 optima.

An actual numerical fallback emits:

```text
RREA_GPU_POPULATION_FALLBACK supports=N
```

Fallback handles a rejected numerical plan, not a CUDA memory fault or heap
exhaustion. With fallback disabled, an invalid plan aborts before weight
commit. Record fallback counts when benchmarking: otherwise apparent GPU
execution could mask host population work.

The production Python driver allows these execution-only controls via separate
`--extra-warpx-arg key=value` tokens and checks their values. Unknown `rrea.*`
keys and protected physics/model/geometry overrides remain blocked.

`rrea.kinetic_backend=cpu` in a CUDA executable is a kinetic reference mode;
the field/fluid backend is still CUDA. The independently enabled host OpenMP
support remains useful for this reference/fallback. It does not turn CUDA
kernels into OpenMP work.

## Precision and conservation

Stored state, updates and kernel argument layouts remain binary64/integer POD.
The original CPU population reduction and selected delicate scalar calculations
retain their local `long double` working arithmetic. Device versions use
binary64 local working arithmetic. These differing local types never occur in
shared object layouts or non-templated function signatures.

A device support plan must agree with the host capacity/rank reference, meet
its requested reduction, preserve the four CIC moments within the existing
stored-precision checks, and produce finite, nonnegative weights. Rounding can
change which mathematically equivalent null-space endpoint is selected; byte
identity with a long-double CPU planner is not the GPU acceptance criterion.
The explicit fallback protects ill-conditioned supports, and strict mode makes
such cases visible during validation. Independent binary64 CPU tests exercise
that arithmetic choice but do not emulate CUDA math libraries or instruction
rounding.

Transport and newborn logs retain the original energy/charge/source ledgers.
RNG counters depend on logical identity, timestep, interaction index and
channel, not GPU scheduling. The CPU tests verify invariance under reversing
history evaluation order. This does not establish bitwise CPU/GPU trajectory
identity: validate both numerical tolerances and the statistical observables
of the intended physical problem.

## Build and native GPU acceptance

See `GPU_PORT.md` for the complete dependency and managed-memory requirements.
Use a separate build directory and compatible pinned upstream source trees:

```bash
WARPX_SOURCE=/absolute/path/to/WarpX-26.06 \
AMREX_SOURCE=/absolute/path/to/AMReX-26.06 \
CUDA_ARCH=90 JOBS=4 \
bash scripts/build_rrea_cuda.sh
```

The helper builds the real production overlay, custom native library and tests.
It does not install CUDA, download missing upstreams, or submit cluster jobs.
It runs native CTest serially, initially excluding four-rank fixtures. Retain
the project requirement to pass the complete production build and native
physics tests before an OLIVIA submission.

The newly registered native targets are:

```text
rrea_kinetic_engine_smoke
rrea_kinetic_storage_smoke
rrea_population_gpu_smoke
```

In a genuine CUDA build, these contain real GPU dispatches: complete live and
newborn histories, count/pack/destruct, birth and survivor validation, and
support planning with CPU fallback disabled. The engine uses a deliberately
non-aligned batch of 67 histories; the population fixture batches 17 supports.
They have **not been compiled or run with CUDA in the implementation environment**.

After the full native suite passes, run these targets under Compute Sanitizer
and inspect a small populated production run. Use one MPI rank and one visible
GPU first. Compare population/energy spectra, avalanche growth, charge/current
continuity, four deposited CIC moments, secondary birth timing, photon flow,
energy ledgers and checkpoint/restart. Test multiple seeds and representative
high-multiplicity events. Separate kernel, allocation, packing, replay,
population, MPI and output costs in the profiler. Multi-rank/multi-GPU and
cross-backend restart require additional acceptance tests.

A manual executable invocation must include the following options; the native
fixture wrappers add their own test-specific choices:

```text
amrex.the_arena_is_managed=1
rrea.kinetic_backend=gpu
rrea.gpu_population_cpu_fallback=0
```

Strict fallback=0 is recommended for the initial acceptance exercise. The
runtime default remains 1 with explicit reporting. Do not assume that enabling
the backend alone makes a large production deck a suitable first test.

## Reproducible CPU checks

The nine-test runner compiles the **actual numerical sources and production
table loader**. Because upstream AMReX is absent, it uses the guarded adapters
in `native_amrex_rrea/tests/scalar_types/`: basic scalar/vector aliases, abort
and assertion helpers, and neutral debug configuration. They do not implement
AMReX meshes, particle containers, launches, MPI or a GPU runtime. They are used
only by this standalone script, never by native CMake targets.

```bash
CXX=g++ bash scripts/test_rrea_gpu_kinetics_cpu.sh
CXX=clang++ bash scripts/test_rrea_gpu_kinetics_cpu.sh
CXX=clang++ SANITIZE=1 bash scripts/test_rrea_gpu_kinetics_cpu.sh
CXX=clang++ BINARY64=1 bash scripts/test_rrea_gpu_kinetics_cpu.sh
```

The suite covers tables, samplers, 540 live plus 120 newborn histories,
conservation and lossless transaction packing, storage ownership, path clipping,
secondary groups, charged interleave and CIC resampling. A population fixture
checks 256 supports / 9967 records, including degenerate support geometry and
routing/identity preservation. Final-survivor tests also reject NaN weights and
the excluded upper domain boundary. The existing large statistical sampler and
CIC unbiasedness tests remain in the suite.

## Source map

| File or family | Role |
| --- | --- |
| `include/rrea/RreaInteractionPhysics.H`, `RreaInteractionTypes.H` | Immutable numerical views and shared original samplers |
| `include/rrea/RreaKineticSupport.H` | Cross-execution scalar helpers, spans, owning scratch vectors and algorithms |
| `include/rrea/RreaKineticFieldView.H`, `RreaFieldGatherScalar.H` | Device field gather and shared interpolation |
| `warpx_overlay/Source/Rrea/RreaKineticStep.H` | Shared complete live/newborn collision equations |
| `RreaKineticGpu.H`, `RreaKineticRecords.H` | Launches, bounded concurrent batches, device ownership and packing |
| `RreaSecondaryCommit.cpp`, `RreaSecondaryValidation.H` | Newborn GPU waves, validation and existing collective transaction integration |
| `RreaPopulationGpu.H`, `RreaPopulationRecord.H`, `RreaPopulationParticles.H` | GPU record creation, constrained support planning and commit/refill primitives |
| `RreaCouplingPopulation.cpp`, `RreaParticleSweep.H` | Production population dispatch, routing and commit integration |
| `include/rrea/RreaCicConservingResample.H` | Shared original four-moment stochastic reducer |

## External API references

The implementation uses AMReX's documented `ParallelFor`, reduction, managed
arena and synchronization interfaces. NVIDIA's CUDA documentation is the
reference for device heap allocation ownership and device language limitations.
Consult the documentation matching the installed toolkit and pinned AMReX tree;
reading the latest documentation is not a substitute for compiling that tree.

- AMReX GPU guide: https://amrex-codes.github.io/amrex/docs_html/GPU.html
- NVIDIA CUDA programming guide: https://docs.nvidia.com/cuda/cuda-programming-guide/index.html
- CUDA language support: https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-support.html
