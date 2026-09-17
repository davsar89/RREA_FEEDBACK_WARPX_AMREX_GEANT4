# Experimental hybrid CUDA execution

**Status: builds and runs on CUDA, and reproduces the CPU backend to
floating-point roundoff. Wall-clock on a consumer GPU is unusable for reasons
that are not the physics (see "Local checks").** The energetic
transport and population extension is described in [`GPU_KINETICS.md`](GPU_KINETICS.md).

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

The existing reaction/conductivity AMReX kernels remain in use. This port does
not replace the kinetic physics with a simplified GPU model. Rayleigh scattering
remains excluded, the production double-precision settings remain, and the
checkpoint record layout is not intentionally changed. Cross-device restart and
the complete coupled physics still need native tests.

## Memory and ownership

`RreaGpu.H` gives the standalone physics primitives a CPU path and a CUDA path.
CUDA field/current/closure buffers use managed allocations with pointer-only
kernel views. Host owning C++ vectors, strings, maps and configuration objects are not
passed to device kernels. Kinetic histories additionally own device-created
scratch vectors; these are packed and destroyed on the GPU, not read or freed
on the CPU (see `GPU_KINETICS.md`). AMReX `ParallelFor` and reductions operate on the
mesh; the custom Maxwell solver uses the same equations in both backends.

**Every production CUDA invocation must set this before AMReX initialization:**

```text
amrex.the_arena_is_managed=1
```

The retained host transaction, diagnostic, particle and mesh consumers make device-only allocation unsafe
in this intermediate implementation. A startup guard rejects it rather than
assuming that a CPU can dereference a device pointer. The production test wrapper
adds this argument for CUDA tests; component tests initialize their test arenas
accordingly. Merely adding this flag to the original CPU source is not a GPU port.

GPU work is joined before host consumers, host MPI collectives, checkpoint I/O,
and disposal of temporary arrays. The coupling singleton is released before
AMReX destroys its memory arenas. These are explicit correctness boundaries,
not a promise of overlapping CPU/GPU computation.

The full-domain Maxwell arrays and current collective are still replicated
across MPI ranks. GPU transport now gathers fields directly on device, but
host source replay, particle insertion orchestration, diagnostics and I/O can
still move shared pages. Dynamic device allocation and divergent histories can
also limit performance. No end-to-end speedup has been established.

## Build on a CUDA development machine

Use the project's original Git checkout and its recorded upstream gitlinks when
available. The delivered source ZIP, like the input ZIP, does **not** contain the
AMReX or WarpX dependency sources. A ZIP alone cannot run `git submodule update`:
that requires a real Git checkout containing those gitlinks. `.gitmodules`
records the fork URLs and the `rrea-pinned-26.06` branches, but branch names alone
are not immutable commit pins.

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
The first build uses `MPI=OFF`; `MPI=ON` is available for later MPI validation.
Compilation defaults to four jobs, with a hard maximum of twelve.

The work tree defaults to `tmp/rrea_cuda/`. The script runs real native CTest
cases serially and excludes the four-rank fixtures from that initial single-GPU
run. It fails immediately on a failed build or test. This entire build/test path
is **provided but has not completed in the implementation environment**.

## First execution and validation

Start with one MPI rank and one visible GPU. Do not reuse an eight-rank CPU
launcher without planning GPU assignment and the replicated solver memory.
The native test wrapper supplies the managed-memory argument. For a manually
prepared WarpX command, append it yourself.

The production driver already accepts the needed nonphysics override. Once a
compatible build has passed its native tests, the relevant invocation shape is:

```bash
OMP_NUM_THREADS=4 python3 scripts/run_rrea_profiled_video_capture.py \
  --warpx-exe /absolute/path/to/the/built/WarpX-executable \
  --run-root /absolute/path/to/a/new/run-root \
  --output-dir /absolute/path/to/a/new/run-root/test-run \
  --ranks 1 --transport-omp-threads 4 \
  --extra-warpx-arg amrex.the_arena_is_managed=1 \
  --extra-warpx-arg rrea.kinetic_backend=gpu \
  --extra-warpx-arg rrea.gpu_population_cpu_fallback=0
```

This is a driver example, **not a verified GH200 production run or a small-case
preset**. Set the scientifically appropriate duration, mesh and seed configuration
using the existing rerun guide. Do not begin with a long capture. Retain the
project's rule that the full production overlay and native smokes must build and
pass locally before an OLIVIA submission. No cluster submission was made for
this port.

Before interpreting results from this backend, run the complete native CUDA
suite, including the real WarpX transport/continuity fixtures. Check device
memory accesses with Compute Sanitizer. On the same physical problem, inspect
charge/current closure, the field-energy ledger, carrier positivity, causal
propagation, avalanche statistics, and mesh/timestep convergence. GPU reduction
order can change roundoff; require physical/numerical consistency rather than
byte identity. Exercise checkpoint/restart in both directions before relying on
cross-backend continuation. Multi-rank and multiple-GPU runs need separate tests.

## Local checks and their limits

The original field/fluid stage passed six standalone numerical programs with
GCC, Clang and address/undefined-behavior sanitizers. The kinetic extension
adds a nine-program CPU runner. See `GPU_KINETICS.md` for the test scope.

```bash
CXX=g++ bash scripts/test_rrea_gpu_primitives.sh
CXX=clang++ SANITIZE=1 bash scripts/test_rrea_gpu_primitives.sh
CXX=g++ bash scripts/test_rrea_gpu_kinetics_cpu.sh
CXX=clang++ SANITIZE=1 bash scripts/test_rrea_gpu_kinetics_cpu.sh
CXX=clang++ BINARY64=1 bash scripts/test_rrea_gpu_kinetics_cpu.sh
```

The CUDA build itself now compiles and runs (CUDA 12.6, sm_89, RZ double
precision). 22 of 25 native RREA tests pass; the 3 that do not are ctest
`TIMEOUT` expiries, not wrong answers -- the WarpX continuity fixture exits 0
in 166 s against a `TIMEOUT` of 120 s set for CPU timings. Against the CPU
backend on the same deck it reproduces 149 of 153 reduced-diagnostic columns
bit-identically, with identical discrete secondary and positron counts; the
two columns that differ are charge-continuity residuals at ~1e-16 e.

Wall-clock is another matter, and on this hardware it is an artifact rather
than a property of the port: consumer GPUs under WSL2 report
`concurrentManagedAccess = 0`, so the runtime bulk-migrates the managed arena
at every launch. Measured with `nsys`, each `cudaLaunchKernel` blocks ~61 ms
against 5.5 us for a bare launch, leaving the SMs 0-9% busy. Shrinking the
arena does not help. Datacenter parts report 1 and migrate on demand, so
performance has to be measured there. Cross-device checkpoint restart and
multi-GPU are still untested.

## Remaining full-device and scaling work

The collision equations, secondary histories and constrained population
kernels are now present. Further device-residency work includes ordered source
and deposition replay, candidate preparation, insertion orchestration and
support routing/ordering. A distributed Maxwell/current implementation is a
separate multi-GPU design task. Validate this correctness-oriented backend on
real CUDA first; tune queues/allocations and host-device boundaries using
representative profiles rather than inferring performance from kernel count.
