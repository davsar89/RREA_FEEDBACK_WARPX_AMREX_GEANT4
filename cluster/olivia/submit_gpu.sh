#!/usr/bin/env bash
# One node, one MPI process per GH200. Forward physics arguments to the driver.
# Use through olivia.py script; KEY=VALUE settings precede build/run.
set -euo pipefail
while [[ "${1:-}" == *=* ]]; do export "${1?}"; shift; done
BASE=${BASE:-/cluster/projects/nn9526k/dsarria/warpx_rrea_smoke}
REPO_ROOT=${REPO_ROOT:-$BASE/repo_amrex_latest}
GPUS=${GPUS:-1}
[[ "$GPUS" =~ ^[1-4]$ ]] || { echo 'GPUS must be 1, 2, 3 or 4' >&2; exit 2; }
GPU_WORK=${GPU_WORK:-$BASE/gpu_cuda_mpi}
case "${1:-}" in build|run|exec) ;; *) echo 'Expected build, run [driver arguments], or exec [command]' >&2; exit 2;; esac
mkdir -p "$BASE/logs"
export REPO_ROOT GPU_WORK GPUS
sbatch --parsable --account=nn9526k --partition=accel --gpus="$GPUS" \
    --nodes=1 --ntasks="$GPUS" --gpus-per-task=1 --gpu-bind=single:1 \
    --cpus-per-task="${CPUS:-12}" --mem="${MEMORY:-$((96 * GPUS))G}" \
    --qos="${QOS:-devel}" --time="${WALLTIME:-02:00:00}" \
    --job-name="rrea_gpu_$1" --output="$BASE/logs/rrea_gpu_%j.out" \
    --error="$BASE/logs/rrea_gpu_%j.err" --export=ALL \
    "$REPO_ROOT/cluster/olivia/slurm_rrea_gpu.sbatch" "$@"
