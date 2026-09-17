#!/usr/bin/env bash
#SBATCH --job-name=g_em_obs
#SBATCH --account=nn9526k
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --output=/cluster/projects/nn9526k/dsarria/warpx_rrea_smoke/logs/rrea_em_obs_%j.out
#SBATCH --error=/cluster/projects/nn9526k/dsarria/warpx_rrea_smoke/logs/rrea_em_obs_%j.err

# Host-side Huygens/Love reduction: the three observer series the multiband
# figure draws its waveform and spectrum panels from. It runs where the
# multi-GB rrea_em_boundary.bin lives; only the small per-observer CSVs travel.
#
# Kept out of the render sbatch deliberately: that job is ~35 min and this one
# is hours, so coupling them would make every video preview unaffordable.
#   EM_INPUT_DIR     capture dir holding rrea_em_boundary.bin (required)
#   EM_TIME_STRIDE   output cadence in record steps (default 4 = 10 ns at
#                    dt 2.5 ns; Nyquist 50 MHz against a 300 kHz LF band edge
#                    and a 1.24 MHz EFCM digitizer limit)
#   EM_AZIMUTH       azimuthal quadrature nodes, off-axis observers only
#   EM_OBSERVERS     which of the three to run (default all); one observer at a
#                    coarse stride is how you calibrate wall time before
#                    committing to the full reduction
#
# Geometry is the figure's: a 100 km ground receiver, the ER-2 at 20 km MSL
# 5 km ahead, and the nadir field mill on the axis. The nadir observer takes
# the exact-axis path, where every ring has one retarded time and no azimuthal
# quadrature is needed.

set -euo pipefail

module --force purge >/dev/null 2>&1 || true
module load NRIS/CPU
module load foss/2025a
module load matplotlib/3.10.3-gfbf-2025a

REMOTE_ROOT="${REMOTE_ROOT:-/cluster/projects/nn9526k/dsarria/warpx_rrea_smoke}"
REPO_ROOT="${REPO_ROOT:-${REMOTE_ROOT}/repo_amrex_latest}"
: "${EM_INPUT_DIR:?set EM_INPUT_DIR to the capture directory}"
EM_TIME_STRIDE="${EM_TIME_STRIDE:-4}"
EM_AZIMUTH="${EM_AZIMUTH:-256}"
EM_OBSERVERS="${EM_OBSERVERS:-ground aircraft nadir}"
OUT_DIR="${EM_INPUT_DIR}/rrea_video"
cd "${REPO_ROOT}"

echo "EM observer job ${SLURM_JOB_ID:-manual} on $(hostname)"
echo "EM_INPUT_DIR=${EM_INPUT_DIR} stride=${EM_TIME_STRIDE} azimuth=${EM_AZIMUTH}"
ls -l "${EM_INPUT_DIR}/rrea_em_boundary.bin"

# altitude_km offset_km tag
while read -r altitude offset tag; do
    [[ " ${EM_OBSERVERS} " == *" ${tag} "* ]] || continue
    echo "=== observer ${tag}: ${altitude} km MSL, ${offset} km offset ==="
    python3 scripts/compute_rrea_em_radio.py "${EM_INPUT_DIR}" \
        --observer-altitude-km "${altitude}" \
        --observer-offset-km "${offset}" \
        --azimuth-samples "${EM_AZIMUTH}" \
        --time-stride "${EM_TIME_STRIDE}" \
        --out "${OUT_DIR}/rrea_em_observer_${tag}.csv"
done <<'OBSERVERS'
0 100 ground
20 5 aircraft
20 0 nadir
OBSERVERS

ls -l "${OUT_DIR}"/rrea_em_observer_*.csv
date
