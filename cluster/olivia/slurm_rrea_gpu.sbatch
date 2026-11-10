#!/usr/bin/env bash
set -euo pipefail
export SLURM_EXPORT_ENV=ALL
module --force purge
module load NRIS/GPU
module load GCC/13.3.0 CUDA/12.6.0 CMake/3.29.3-GCCcore-13.3.0 Python/3.12.3-GCCcore-13.3.0
module load OpenMPI/5.0.3-GCC-13.3.0
export SLURM_MPI_TYPE=pmix
export OMP_NUM_THREADS=1 OMP_DYNAMIC=false
cd "${REPO_ROOT:?}"
test "$(uname -m)" = aarch64
nvidia-smi
mode=$1; shift
case "$mode" in
    build)
        export http_proxy=${http_proxy:-http://10.63.2.48:3128/}
        export https_proxy=${https_proxy:-$http_proxy}
        : "${WARPX_SOURCE:?Set the pinned WarpX source path}"
        : "${AMREX_SOURCE:?Set the pinned AMReX source path}"
        export WARPX_SOURCE AMREX_SOURCE
        ompi_info --parsable --all | grep mpi_built_with_cuda_support || true
        WORK_DIR="$GPU_WORK" CUDA_ARCH=90 MPI=ON JOBS="${JOBS:-12}" \
            bash scripts/build_rrea_cuda.sh
        ;;
    run)
        python3 scripts/run_rrea_profiled_video_capture.py \
            --warpx-exe "$GPU_WORK/build/bin/warpx.rz.MPI.CUDA.DP.PDP.EB" \
            --launcher srun --ranks "$GPUS" --cpus-per-rank "${SLURM_CPUS_PER_TASK:-12}" --omp-threads 1 \
            --transport-omp-threads "${SLURM_CPUS_PER_TASK:-12}" \
            --extra-warpx-arg amrex.the_arena_is_managed=1 \
            --extra-warpx-arg amrex.the_arena_init_size=0 "$@"
        ;;
    # One controller on the batch node; custom MPI commands launch with srun.
    exec) "$@" ;;
    *) echo "Unknown mode: $mode" >&2; exit 2;;
esac
