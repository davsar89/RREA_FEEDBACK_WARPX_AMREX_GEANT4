#!/usr/bin/env bash
# Submit one profiled RREA capture job on the remote HPC system.
#
#   python scripts/hpc.py script cluster/hpc/submit_profiled_video.sh \
#       RUN_TAG=<tag> \
#       RREA_WARPX_EXE=/.../warpx.rz.MPI.OMP.DP.PDP.EB \
#       PROFILED_VIDEO_E0_KV_PER_M=<field> \
#       RUN_PROFILE=<name from config/rrea_defaults.json>
#
# Checkpoints support a later explicit resubmission with the same RUN_TAG and
# PROFILED_VIDEO_RESTART_FROM=auto. No dependent job chain is created.
set -euo pipefail

REMOTE_ROOT="${REMOTE_ROOT:?REMOTE_ROOT must name the remote work root (REMOTE_ROOT/repo_amrex_latest, runs/, logs/)}"
REPO="${REMOTE_ROOT}/repo_amrex_latest"

for kv in "$@"; do
    case "${kv}" in
        [A-Za-z_]*=*) export "${kv?}" ;;
        *) echo "arguments must be KEY=VALUE (got: ${kv})" >&2; exit 2 ;;
    esac
done

LAYOUT_EXPORTS="$(python3 - "${REPO}/config/rrea_defaults.json" <<'PY'
import json, os, sys
layout = json.load(open(sys.argv[1]))['hpc_layout']
for key, variable in {
    'mpi_ranks': 'NTASKS',
    'cpus_per_rank': 'CPUS_PER_TASK',
    'transport_omp_threads': 'PROFILED_VIDEO_TRANSPORT_OMP_THREADS',
}.items():
    if os.environ.get(variable) is None:
        print(f'export {variable}="{layout[key]}"')
PY
)"
eval "${LAYOUT_EXPORTS}"

# RUN_PROFILE loads only run physics and checkpoint cadence from the canonical
# configuration. Explicit environment values win.
if [[ -n "${RUN_PROFILE:-}" ]]; then
    PROFILE_EXPORTS="$(python3 - "${RUN_PROFILE}" "${REPO}/config/rrea_defaults.json" <<'PY'
import json
import os
import sys

name, config_path = sys.argv[1], sys.argv[2]
profiles = json.load(open(config_path))['run_profiles']
if name not in profiles:
    sys.exit(f'unknown RUN_PROFILE {name!r}; known: {sorted(profiles)}')
env_map = {
    'e0_kv_per_m': 'PROFILED_VIDEO_E0_KV_PER_M',
    'stop_time_s': 'PROFILED_VIDEO_STOP_TIME_S',
    'dt_s': 'PROFILED_VIDEO_DT_S',
    'domain_radius_m': 'PROFILED_VIDEO_DOMAIN_RADIUS_M',
    'n_cell_r': 'PROFILED_VIDEO_N_CELL_R',
    'n_cell_z': 'PROFILED_VIDEO_N_CELL_Z',
    'checkpoint_interval': 'PROFILED_VIDEO_CHECKPOINT_INT',
}
for key, value in profiles[name].items():
    if key.startswith('_'):
        continue
    variable = env_map[key]
    if os.environ.get(variable) is None:
        print(f'export {variable}="{value}"')
        print(f'echo "profile {name}: {variable}={value}"')
PY
)" || {
        echo "RUN_PROFILE=${RUN_PROFILE} resolution failed; refusing to submit" >&2
        exit 2
    }
    eval "${PROFILE_EXPORTS}"
fi

: "${RUN_TAG:?RUN_TAG=<unique run tag> is required}"
: "${RREA_WARPX_EXE:?RREA_WARPX_EXE must name the accepted engine build}"
: "${PROFILED_VIDEO_E0_KV_PER_M:?E0 has no production default}"

export PROFILED_TRANSPORT_CONFIG="${PROFILED_TRANSPORT_CONFIG:-${REPO}/rrea_transport_tables/schema6/production/transport_physics.json}"
WALLTIME="${WALLTIME:-48:00:00}"
MEMORY="${MEMORY:-64G}"

echo "=== quota before submission ==="
if [[ -n "${HPC_SLURM_ACCOUNT:-}" ]]; then
    env -u PYTHONPATH -u PYTHONHOME cost -p "${HPC_SLURM_ACCOUNT}" 2>/dev/null || true
else
    env -u PYTHONPATH -u PYTHONHOME cost 2>/dev/null || true
fi
echo "repo copy at commit: $(cat "${REPO}/CODEX_SOURCE_COMMIT")"
test -x "${RREA_WARPX_EXE}"
test -s "${PROFILED_TRANSPORT_CONFIG}"

RUN_DIR="${REMOTE_ROOT}/runs/scratch_${RUN_TAG}"
if [[ -e "${RUN_DIR}" && "${PROFILED_VIDEO_RESTART_FROM:-}" != "auto" ]]; then
    echo "refusing to reuse existing run directory: ${RUN_DIR}" >&2
    echo "to continue it, pass PROFILED_VIDEO_RESTART_FROM=auto" >&2
    exit 2
fi
export PROFILED_VIDEO_SCRATCH_OUT="${RUN_DIR}"

SBATCH_FILE="${REPO}/cluster/hpc/slurm_rrea_profiled_video_capture.sbatch"
SBATCH_ARGS=(
    --ntasks="${NTASKS}"
    --cpus-per-task="${CPUS_PER_TASK}"
    --nodes=1
    --mem="${MEMORY}"
    --time="${WALLTIME}"
    --job-name="${RUN_TAG}"
    # Log location lives in the site configuration, not in the sbatch.
    --output="${REMOTE_ROOT}/logs/rrea_prof_vid_${RUN_TAG}_%j.out"
    --error="${REMOTE_ROOT}/logs/rrea_prof_vid_${RUN_TAG}_%j.err"
    --export=ALL
)
if [[ -n "${HPC_SLURM_ACCOUNT:-}" ]]; then
    SBATCH_ARGS+=(--account="${HPC_SLURM_ACCOUNT}")
fi
if [[ -n "${QOS:-}" ]]; then
    SBATCH_ARGS+=(--qos="${QOS}")
fi

mkdir -p "${REMOTE_ROOT}/logs"
JOB_ID="$(sbatch --parsable "${SBATCH_ARGS[@]}" "${SBATCH_FILE}")"
echo "job=${JOB_ID}"
squeue -j "${JOB_ID}" --format='%.10i %.24j %.8T %.20E'
