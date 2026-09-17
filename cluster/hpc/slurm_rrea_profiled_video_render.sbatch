#!/usr/bin/env bash
#SBATCH --job-name=g_vid_render
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=rrea_vid_render_%j.out
#SBATCH --error=rrea_vid_render_%j.err

# Production HPC-side render+encode of a profiled-video capture directory.
# Frames never leave the cluster; only the resulting .mp4 files are downloaded.
#
# Submitted by hand after capture; the capture launcher names this file in its
# completion message. It is the production host-side renderer, not an orphan.
#   RENDER_INPUT_DIR   capture dir containing rrea_video/ (required)
#   RENDER_OUTPUT_DIR  output dir (default: <input>/rendered_video)
#   RENDER_EXTRA_ARGS  extra render_rrea_profiled_video.py args (optional)
#
# Override --mem for full captures: memory and render time grow with the number
# of rows in rrea_video/video_frame_metadata.csv, and the 24G directive is a
# preview-sized floor. Billing is memory-dominated, so request measured headroom
# for the capture being rendered.
#
# If using `--qos=devel`, verify the account's current eligibility and limits
# first; scheduler policy is external to this repository.
#
# The optical strip's cloudscat transport is derived by the renderer: ON when
# the capture is complete (final render),
# OFF for a mid-run preview snapshot.  Force it with
#   RENDER_EXTRA_ARGS=--cloud-scattering   (or --no-cloud-scattering)

set -euo pipefail

# Site software stack: the module list is private per-site configuration
# (HPC_SLURM_MODULES), never hardcoded in the repository. Unset means the
# system default environment is already correct.
if [[ -n "${HPC_SLURM_MODULES:-}" ]]; then
    module --force purge >/dev/null 2>&1 || true
    for module_name in ${HPC_SLURM_MODULES}; do
        module load "${module_name}"
    done
fi

REMOTE_ROOT="${REMOTE_ROOT:?REMOTE_ROOT must name the remote work root (REMOTE_ROOT/repo_amrex_latest, runs/, logs/)}"
REPO_ROOT="${REPO_ROOT:-${REMOTE_ROOT}/repo_amrex_latest}"
: "${RENDER_INPUT_DIR:?set RENDER_INPUT_DIR to the capture directory}"
RENDER_OUTPUT_DIR="${RENDER_OUTPUT_DIR:-${RENDER_INPUT_DIR}/rendered_video}"
cd "${REPO_ROOT}"

echo "Render job ${SLURM_JOB_ID:-manual} on $(hostname)"
echo "RENDER_INPUT_DIR=${RENDER_INPUT_DIR}"
echo "RENDER_OUTPUT_DIR=${RENDER_OUTPUT_DIR}"
which python3
which ffmpeg

# The capture does not write the strip's band lightcurve. Re-reduce it from
# the frame stream before every render so a live capture cannot reuse a stale
# sidecar.
python3 scripts/extract_rrea_band_series.py "${RENDER_INPUT_DIR}"

# shellcheck disable=SC2086
python3 scripts/render_rrea_profiled_video.py "${RENDER_INPUT_DIR}" \
    --output-dir "${RENDER_OUTPUT_DIR}" \
    --jobs "${SLURM_CPUS_PER_TASK:-4}" \
    ${RENDER_EXTRA_ARGS:-}

find "${RENDER_OUTPUT_DIR}" -name '*.mp4' -size +0 | sort
date
