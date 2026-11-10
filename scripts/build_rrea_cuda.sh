#!/usr/bin/env bash
# Build and test the experimental hybrid CUDA overlay; no scheduler submission.
# Run on a Linux CUDA development machine, not a CPU-only login environment.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
WARPX_SOURCE=${WARPX_SOURCE:-"$ROOT/upstream/WarpX-26.06"}
AMREX_SOURCE=${AMREX_SOURCE:-"$ROOT/upstream/AMReX-26.06"}
WORK_DIR=${WORK_DIR:-"$ROOT/tmp/rrea_cuda"}
JOBS=${JOBS:-4}
CUDA_ARCH=${CUDA_ARCH:-90}
MPI=${MPI:-OFF}
if [[ ! "$JOBS" =~ ^[0-9]+$ ]] || (( JOBS < 1 || JOBS > 12 )); then
    echo 'JOBS must be an integer from 1 to 12' >&2
    exit 2
fi
if [[ "$MPI" != ON && "$MPI" != OFF ]]; then
    echo 'MPI must be ON or OFF' >&2
    exit 2
fi
for program in cmake python3; do
    command -v "$program" >/dev/null || { echo "Missing $program" >&2; exit 2; }
done
if [[ ! -f "$WARPX_SOURCE/CMakeLists.txt" || ! -f "$AMREX_SOURCE/CMakeLists.txt" ]]; then
    echo 'Populate the pinned WarpX and AMReX sources first (see docs/GPU_PORT.md).' >&2
    echo "WarpX: $WARPX_SOURCE; AMReX: $AMREX_SOURCE" >&2
    exit 2
fi
WARPX_SOURCE=$(cd -- "$WARPX_SOURCE" && pwd)
AMREX_SOURCE=$(cd -- "$AMREX_SOURCE" && pwd)
CUDACXX=${CUDACXX:-nvcc}
command -v "$CUDACXX" >/dev/null || { echo "Missing CUDA compiler: $CUDACXX" >&2; exit 2; }
export CUDACXX
mkdir -p -- "$WORK_DIR"
WORK_DIR=$(cd -- "$WORK_DIR" && pwd)
PREPARED="$WORK_DIR/source"
BUILD="$WORK_DIR/build"
prepare_args=()
if [[ -f "$PREPARED/CMakeLists.txt" ]]; then
    prepare_args+=(--refresh)
fi
python3 "$ROOT/scripts/prepare_rrea_warpx_source.py" \
    --warpx-source "$WARPX_SOURCE" --output-dir "$PREPARED" \
    --overlay-dir "$ROOT/native_amrex_rrea" "${prepare_args[@]}"
cmake -S "$PREPARED" -B "$BUILD" \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF \
    -DWarpX_DIMS=RZ \
    -DWarpX_PRECISION=DOUBLE -DWarpX_PARTICLE_PRECISION=DOUBLE \
    -DWarpX_COMPUTE=CUDA -DWarpX_MPI="$MPI" \
    -DWarpX_MPI_THREAD_MULTIPLE=ON \
    -DWarpX_OPENPMD=OFF -DWarpX_QED=OFF -DWarpX_FFT=OFF \
    -DWarpX_PYTHON=OFF -DWarpX_EB=ON \
    -DWarpX_APP=ON -DWarpX_LIB=ON -DWarpX_IPO=OFF \
    -DWarpX_FASTMATH=OFF -DABLASTR_FASTMATH=OFF \
    -DAMReX_FASTMATH=OFF -DAMReX_CUDA_FASTMATH=OFF \
    -DAMReX_CUDA_ERROR_CAPTURE_THIS=ON \
    -DAMReX_CUDA_ERROR_CROSS_EXECUTION_SPACE_CALL=ON \
    -DWarpX_amrex_internal=ON -DWarpX_amrex_src="$AMREX_SOURCE" \
    -DRREA_AMREX_OVERLAY="$ROOT/native_amrex_rrea" \
    -DRREA_AMREX_ENABLE_IPO=OFF -DRREA_AMREX_BUILD_TESTS=ON \
    -DRREA_CUDA_HOST_OMP=ON
cmake --build "$BUILD" --parallel "$JOBS"
# Tests launch real CUDA kernels. One process/test at a time: do not share a
# GPU between concurrent tests or run the four-rank fixture on a single GPU.
# Scope to the RREA suite: the prepared tree also registers WarpX's own
# regression tests, which abort under this configuration's OPENPMD=OFF.
ctest --test-dir "$BUILD" --output-on-failure -j 1 -R '^rrea' -E mpi4
printf '\nHybrid CUDA build and selected native tests completed.\n'
printf 'Executable directory: %s/bin\n' "$BUILD"
printf 'Runtime requirement: amrex.the_arena_is_managed=1\n'
printf 'CUDA collision histories, newborn transport, and CIC support planning are enabled.\n'
printf 'MPI/transaction orchestration, ordered source replay, and I/O remain host work.\n'
