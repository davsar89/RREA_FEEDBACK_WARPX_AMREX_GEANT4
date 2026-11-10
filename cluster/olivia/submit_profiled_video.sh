#!/usr/bin/env bash
# Submit one profiled RREA capture job on OLIVIA.
#
#   python scripts/olivia.py script cluster/olivia/submit_profiled_video.sh \
#       RUN_TAG=<tag> --warpx-exe /.../warpx.rz.MPI.OMP.DP.PDP.EB \
#       --e0-peak-kv-per-m <field> [more driver flags]
#
# KEY=VALUE arguments set this submitter (RUN_TAG, WALLTIME, MEMORY, QOS,
# NTASKS, CPUS_PER_TASK); every other argument goes verbatim to
# scripts/run_rrea_profiled_video_capture.py. Continue a run with the same
# RUN_TAG and --restart-checkpoint auto. No dependent job chain is created.
set -euo pipefail

BASE=/cluster/projects/nn9526k/dsarria/warpx_rrea_smoke
REPO="${BASE}/repo_amrex_latest"

DRIVER_ARGS=()
for arg in "$@"; do
    case "${arg}" in
        [A-Z_]*=*) export "${arg?}" ;;
        *) DRIVER_ARGS+=("${arg}") ;;
    esac
done
: "${RUN_TAG:?RUN_TAG=<unique run tag> is required}"

# The node layout is owned by config/rrea_defaults.json (olivia_layout).
eval "$(python3 - "${REPO}/config/rrea_defaults.json" <<'PY'
import json, sys
layout = json.load(open(sys.argv[1]))['olivia_layout']
print(f"LAYOUT_NTASKS={layout['mpi_ranks']} LAYOUT_CPUS={layout['cpus_per_rank']}"
      f" LAYOUT_TRANSPORT_THREADS={layout['transport_omp_threads']}")
PY
)"
NTASKS="${NTASKS:-${LAYOUT_NTASKS}}"
CPUS_PER_TASK="${CPUS_PER_TASK:-${LAYOUT_CPUS}}"

echo "repo copy at commit: $(cat "${REPO}/CODEX_SOURCE_COMMIT")"

RUN_DIR="${BASE}/runs/scratch_${RUN_TAG}"
if [[ -e "${RUN_DIR}" && " ${DRIVER_ARGS[*]} " != *" --restart-checkpoint"* ]]; then
    echo "refusing to reuse existing run directory: ${RUN_DIR}" >&2
    echo "to continue it, pass --restart-checkpoint auto" >&2
    exit 2
fi
export PROFILED_VIDEO_SCRATCH_OUT="${RUN_DIR}"

SBATCH_ARGS=(
    --ntasks="${NTASKS}"
    --cpus-per-task="${CPUS_PER_TASK}"
    --nodes=1
    --mem="${MEMORY:-64G}"
    --time="${WALLTIME:-48:00:00}"
    --job-name="${RUN_TAG}"
    --export=ALL
)
if [[ -n "${QOS:-}" ]]; then
    SBATCH_ARGS+=(--qos="${QOS}")
fi

JOB_ID="$(sbatch --parsable "${SBATCH_ARGS[@]}" \
    "${REPO}/cluster/olivia/slurm_rrea_profiled_video_capture.sbatch" \
    --transport-omp-threads "${LAYOUT_TRANSPORT_THREADS}" "${DRIVER_ARGS[@]}")"
echo "job=${JOB_ID}"
squeue -j "${JOB_ID}" --format='%.10i %.24j %.8T %.20E'
