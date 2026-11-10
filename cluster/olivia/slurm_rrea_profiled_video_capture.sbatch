#!/usr/bin/env bash
#SBATCH --job-name=g_prof_vid
#SBATCH --account=nn9526k
#SBATCH --nodes=1
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --output=/cluster/projects/nn9526k/dsarria/warpx_rrea_smoke/logs/rrea_prof_vid_%j.out
#SBATCH --error=/cluster/projects/nn9526k/dsarria/warpx_rrea_smoke/logs/rrea_prof_vid_%j.err

# Canonical OLIVIA schema-6 production capture on one node. Script arguments
# are passed verbatim to scripts/run_rrea_profiled_video_capture.py (which owns
# every default and check); this job adds only the Slurm layout, the scratch
# output and checkpoint locations, and staging. Rendering is the separate
# slurm_rrea_profiled_video_render.sbatch job.

set -euo pipefail
# C&D campaign plans use an explicit sbatch export list.  Re-export that
# already-sanitized job environment to the nested srun launched by the Python
# driver (Slurm otherwise may retain the sbatch export restriction).
export SLURM_EXPORT_ENV=ALL

module --force purge >/dev/null 2>&1 || true
module load NRIS/CPU
module load foss/2025a

export OMP_DYNAMIC="${OMP_DYNAMIC:-false}"
export OMP_PLACES="${OMP_PLACES:-cores}"
export OMP_PROC_BIND="${OMP_PROC_BIND:-close}"

REMOTE_ROOT="${REMOTE_ROOT:-/cluster/projects/nn9526k/dsarria/warpx_rrea_smoke}"
REPO_ROOT="${REPO_ROOT:-${REMOTE_ROOT}/repo_amrex_latest}"
PYTHON_BIN="${PYTHON_BIN:-/cluster/projects/nn9526k/dsarria/miniconda3/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
    PYTHON_BIN="$(command -v python3)"
fi

PERMANENT_OUT="${REMOTE_ROOT}/runs/profiled_video_schema6_${SLURM_JOB_ID:-manual}"
SCRATCH_BASE="${LOCALSCRATCH:-${SCRATCH:-/localscratch/${SLURM_JOB_ID:-manual}}}"
OUT="${PROFILED_VIDEO_SCRATCH_OUT:-${SCRATCH_BASE}/profiled_video_schema6_${SLURM_JOB_ID:-manual}}"
OUT=$(realpath -m -- "${OUT}")
case "${OUT}" in
    "$(realpath -m -- "${REMOTE_ROOT}/runs")"/*) PERMANENT_OUT="${OUT}" ;;
esac
CAPTURE_OUT="${OUT}/capture"
mkdir -p "${CAPTURE_OUT}" "${PERMANENT_OUT}" "${REMOTE_ROOT}/logs"
cd "${REPO_ROOT}"

stage_outputs() {
    local status=$?
    # Best-effort scratch staging preserves the original job status.
    set +e
    mkdir -p "${PERMANENT_OUT}"
    if [[ -d "${OUT}" ]]; then
        if [[ "${OUT}" != "${PERMANENT_OUT}" ]]; then
            rsync -a --prune-empty-dirs \
                --include='*/' \
                --include='*.json' \
                --include='*.csv' \
                --include='*.txt' \
                --include='*.log' \
                --include='*.mp4' \
                --include='*.png' \
                --exclude='*.bin' \
                --exclude='*.npz' \
                --exclude='rrea_diag/***' \
                --exclude='chk*/***' \
                --exclude='*' \
                "${OUT}/" "${PERMANENT_OUT}/" || true
        fi
        # Retain the newest COMPLETE checkpoint, including after a failed run.
        local last_chk=""
        last_chk=$(find "${OUT}" -maxdepth 1 -type d -name 'chk*' 2>/dev/null \
            | grep -E '/chk[0-9]+$' | sort -V \
            | while read -r c; do [[ -f "${c}/rrea_checkpoint/COMPLETE" ]] && echo "${c}"; done \
            | tail -1) || true
        if [[ -n "${last_chk}" && -d "${last_chk}" ]]; then
            if [[ "${OUT}" == "${PERMANENT_OUT}" ]] || rsync -a "${last_chk}" "${PERMANENT_OUT}/"; then
                printf '%s\n' "$(basename "${last_chk}")" > "${PERMANENT_OUT}/final_checkpoint_name.txt"
            else
                echo "WARNING: rsync of final checkpoint ${last_chk} FAILED; no restart anchor staged" >&2
            fi
        else
            echo "WARNING: no COMPLETE checkpoint found in ${OUT}; nothing staged as restart anchor" >&2
        fi
    fi
    printf '%s\n' "${OUT}" > "${PERMANENT_OUT}/scratch_output_path.txt"
    printf '%s\n' "${status}" > "${PERMANENT_OUT}/job_exit_status.txt"
    exit "${status}"
}
trap stage_outputs EXIT

echo "Job ${SLURM_JOB_ID:-manual} on $(hostname)"
date
echo "REPO_ROOT=${REPO_ROOT}"
echo "PYTHON_BIN=${PYTHON_BIN}"
echo "OUT=${OUT}"
echo "PERMANENT_OUT=${PERMANENT_OUT}"
echo "SOURCE_COMMIT=$(cat CODEX_SOURCE_COMMIT 2>/dev/null || echo unknown)"
echo "ranks=${SLURM_NTASKS:-?} cpus_per_rank=${SLURM_CPUS_PER_TASK:-?}"

# A restart (--restart-checkpoint auto|<chk path>) resumes the SAME scratch
# run dir: pass PROFILED_VIDEO_SCRATCH_OUT. The driver trims the capture to the
# checkpoint step; the engine checks format, state and mesh.
CAPTURE_ARGS=(
    scripts/run_rrea_profiled_video_capture.py
    --output-dir "${CAPTURE_OUT}"
    --run-root "${OUT}"
    --checkpoint-prefix "${OUT}/chk"
    --launcher srun
    --ranks "${SLURM_NTASKS}"
    --cpus-per-rank "${SLURM_CPUS_PER_TASK:-1}"
    # AMReX stays globally serial; --transport-omp-threads scopes the rest.
    --omp-threads 1
)
if [[ " $* " != *" --restart-checkpoint"* ]]; then
    CAPTURE_ARGS+=(--overwrite-output)
fi
CAPTURE_ARGS+=("$@")

"${PYTHON_BIN}" "${CAPTURE_ARGS[@]}"

# Persistent captures already contain the complete render/restart streams.
if [[ "${OUT}" != "${PERMANENT_OUT}" ]]; then
    ARCHIVE="${PERMANENT_OUT}/video_capture_${SLURM_JOB_ID:-manual}.tar.gz"
    COMPRESSION=(-z)
    if command -v zstd >/dev/null 2>&1; then
        ARCHIVE="${ARCHIVE%.gz}.zst"
        COMPRESSION=(-I 'zstd -T0 -8')
    fi
    tar -C "${OUT}" --exclude='rrea_diag' --exclude='chk*' --exclude='diags' \
        "${COMPRESSION[@]}" -cf "${ARCHIVE}" capture
    echo "re-render archive: ${ARCHIVE}"
fi

ln -sfn "${PERMANENT_OUT}" "${REMOTE_ROOT}/runs/profiled_video_schema6_latest"
du -sh "${PERMANENT_OUT}" || true
find "${PERMANENT_OUT}" -type f | wc -l || true
date
echo "profiled schema-6 scratch output: ${OUT}"
echo "profiled schema-6 staged output: ${PERMANENT_OUT}"
