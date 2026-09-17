#!/usr/bin/env bash
#SBATCH --job-name=g_prof_vid
#SBATCH --nodes=1
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --output=rrea_prof_vid_%j.out
#SBATCH --error=rrea_prof_vid_%j.err

# Canonical schema-6 production capture. The executable, semantic
# transport configuration, and E0 are explicit required inputs. Rendering is
# local by default; raw frame data and compact diagnostics are preserved.
# Use one node only.

set -euo pipefail
# C&D campaign plans use an explicit sbatch export list.  Re-export that
# already-sanitized job environment to the nested srun launched by the Python
# driver (Slurm otherwise may retain the sbatch export restriction).
export SLURM_EXPORT_ENV=ALL

# Site software stack: the module list is private per-site configuration
# (HPC_SLURM_MODULES), never hardcoded in the repository. Unset means the
# system default environment is already correct.
if [[ -n "${HPC_SLURM_MODULES:-}" ]]; then
    module --force purge >/dev/null 2>&1 || true
    for module_name in ${HPC_SLURM_MODULES}; do
        module load "${module_name}"
    done
fi

export OMP_DYNAMIC="${OMP_DYNAMIC:-false}"
export OMP_PLACES="${OMP_PLACES:-cores}"
export OMP_PROC_BIND="${OMP_PROC_BIND:-close}"
case "${PROFILED_VIDEO_TRANSPORT_PHASE_TRACE:-0}" in
    0) unset RREA_TRANSPORT_PHASE_TRACE ;;
    1) export RREA_TRANSPORT_PHASE_TRACE=1 ;;
    *)
        echo "PROFILED_VIDEO_TRANSPORT_PHASE_TRACE must be 0 or 1." >&2
        exit 2
        ;;
esac

REMOTE_ROOT="${REMOTE_ROOT:?REMOTE_ROOT must name the remote work root (REMOTE_ROOT/repo_amrex_latest, runs/, logs/)}"
REPO_ROOT="${REPO_ROOT:-${REMOTE_ROOT}/repo_amrex_latest}"
if [[ -z "${RREA_WARPX_EXE:-}" ]]; then
    echo "RREA_WARPX_EXE must name the production schema-6 build." >&2
    exit 2
fi
if [[ -z "${PROFILED_TRANSPORT_CONFIG:-}" ]]; then
    echo "PROFILED_TRANSPORT_CONFIG is required." >&2
    exit 2
fi
if [[ -z "${PROFILED_VIDEO_E0_KV_PER_M:-}" ]]; then
    echo "PROFILED_VIDEO_E0_KV_PER_M is required; E0 has no production default." >&2
    exit 2
fi
WARPX_EXE="${RREA_WARPX_EXE}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3)}"
if [[ -z "${PYTHON_BIN}" || ! -x "${PYTHON_BIN}" ]]; then
    echo "no usable python3 found; set PYTHON_BIN explicitly" >&2
    exit 2
fi

PERMANENT_OUT="${REMOTE_ROOT}/runs/profiled_video_schema6_${SLURM_JOB_ID:-manual}"
SCRATCH_BASE="${LOCALSCRATCH:-${SCRATCH:-/localscratch/${SLURM_JOB_ID:-manual}}}"
OUT="${PROFILED_VIDEO_SCRATCH_OUT:-${SCRATCH_BASE}/profiled_video_schema6_${SLURM_JOB_ID:-manual}}"
CAPTURE_OUT="${OUT}/capture"
mkdir -p "${CAPTURE_OUT}" "${PERMANENT_OUT}" "${REMOTE_ROOT}/logs"
cd "${REPO_ROOT}"

# Single-source values the sbatch logic itself needs (completion guard,
# macro count).  Same validated loader as the driver -- a malformed or
# typo'd config/rrea_defaults.json aborts the job here, loudly.
eval "$("${PYTHON_BIN}" - <<'PY'
import sys
sys.path.insert(0, "scripts")
from rrea_run_support import CAPTURE_DEFAULTS
print(f"JSON_STOP_TIME_S={CAPTURE_DEFAULTS['stop_time_s']}")
print(f"JSON_DT_S={CAPTURE_DEFAULTS['dt_s']}")
PY
)"

# Stop the background checkpoint rotator and block until it exits. A bare kill
# returns immediately, so `wait` is the barrier preventing a rotation from
# racing the final prune/staging step.
stop_rotator() {
    if [[ -n "${ROTATE_PID:-}" ]]; then
        kill "${ROTATE_PID}" 2>/dev/null || true
        wait "${ROTATE_PID}" 2>/dev/null || true
        ROTATE_PID=""
    fi
}

# A completed run needs a readable marker and nonempty archive.
verify_finalized_archive() {
    "${PYTHON_BIN}" - "$1" <<'PY'
import json, os, sys
try:
    d = json.load(open(sys.argv[1]))
    arch = d["archive"]
except Exception:
    sys.exit(1)
sys.exit(0 if isinstance(arch, str) and os.path.isfile(arch)
         and os.path.getsize(arch) > 0 else 1)
PY
}

stage_outputs() {
    local status=$?
    # Best-effort staging: the EXIT trap must not die early, so disable errexit
    # for the handler and preserve the original job status explicitly.
    set +e
    stop_rotator
    mkdir -p "${PERMANENT_OUT}"
    if [[ -d "${OUT}" ]]; then
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
        # ALWAYS preserve the final COMPLETE checkpoint (highest step whose
        # rrea_checkpoint/COMPLETE marker exists) so the run can be extended
        # past stop_time via amr.restart without redoing from step 0.  A
        # wall-time kill can leave the newest chk dir mid-write, so the
        # completion-marker filter is required. Post-run
        # cleanup may delete intermediate chk* dirs but NEVER this one.
        local last_chk=""
        last_chk=$(find "${OUT}" -maxdepth 1 -type d -name 'chk*' 2>/dev/null \
            | grep -E '/chk[0-9]+$' | sort -V \
            | while read -r c; do [[ -f "${c}/rrea_checkpoint/COMPLETE" ]] && echo "${c}"; done \
            | tail -1) || true
        if [[ -n "${last_chk}" && -d "${last_chk}" ]]; then
            # Record the staged-checkpoint marker ONLY on rsync success --
            # a marker without the data would falsely claim a restart anchor.
            if rsync -a "${last_chk}" "${PERMANENT_OUT}/"; then
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
echo "WARPX_EXE=${WARPX_EXE}"
echo "PYTHON_BIN=${PYTHON_BIN}"
echo "OUT=${OUT}"
echo "PERMANENT_OUT=${PERMANENT_OUT}"
echo "SOURCE_COMMIT=$(cat CODEX_SOURCE_COMMIT 2>/dev/null || echo unknown)"
echo "ranks=${SLURM_NTASKS:-?} cpus_per_rank=${SLURM_CPUS_PER_TASK:-?}"
echo "Reminder: run cost and dusage before submitting this job."
echo "restart_from=${PROFILED_VIDEO_RESTART_FROM:-<fresh run>}"
echo "=== dusage (space + files) at job start ==="
env -u PYTHONPATH -u PYTHONHOME dusage 2>/dev/null \
    | grep -F -- "${HPC_SLURM_ACCOUNT:-}" || true

# A restart (PROFILED_VIDEO_RESTART_FROM = 'auto' or a chkNNNNNN path)
# resume the SAME scratch run dir; the driver trims the capture to the
# checkpoint step and the engine appends. Physics, executable and MPI layout
# are strict-matched unless an intentional change carries the explicit
# restart-epoch escalation below.
if [[ -n "${PROFILED_VIDEO_RESTART_FROM:-}" ]]; then
    if [[ -z "${PROFILED_VIDEO_SCRATCH_OUT:-}" ]]; then
        echo "PROFILED_VIDEO_RESTART_FROM requires PROFILED_VIDEO_SCRATCH_OUT (the existing run dir)" >&2
        exit 2
    fi
    # If the run already reached stop_time and finalized, exit successfully.
    RESTART_STOP_S="${PROFILED_VIDEO_STOP_TIME_S:-${JSON_STOP_TIME_S}}"
    RESTART_DT_S="${PROFILED_VIDEO_DT_S:-${JSON_DT_S}}"
    if [[ -s "${CAPTURE_OUT}/rrea_reduced.csv" ]]; then
        LAST_T=$(tail -1 "${CAPTURE_OUT}/rrea_reduced.csv" | cut -d, -f2)
        if awk -v t="${LAST_T}" -v s="${RESTART_STOP_S}" -v d="${RESTART_DT_S}" \
            'BEGIN{exit !(t+0 >= s-d)}'; then
            # A final diagnostic row is not completion: the driver summary
            # proves only a clean engine
            # exit, and run_finalized.json is written at the very end of
            # the sbatch after the stream checks and archive creation
            # succeeded. Require BOTH before declaring no-op; otherwise
            # fall through and restart from the last COMPLETE checkpoint
            # -- a near-complete restart re-runs approximately zero steps and re-executes the
            # whole finalization tail.
            _finalized_ok=0
            if [[ -s "${CAPTURE_OUT}/run_finalized.json" ]] \
               && verify_finalized_archive "${CAPTURE_OUT}/run_finalized.json"; then
                _finalized_ok=1
            fi
            if [[ -s "${CAPTURE_OUT}/profiled_video_capture_summary.json" \
                  && "${_finalized_ok}" == "1" ]]; then
                echo "run already complete (t=${LAST_T} s >= stop_time ${RESTART_STOP_S} s, summary + verified finalization archive present); this job is a no-op"
                exit 0
            fi
            if [[ ! -s "${CAPTURE_OUT}/profiled_video_capture_summary.json" ]]; then
                echo "final diagnostic row present (t=${LAST_T} s) but the capture summary is missing -- previous job died during finalization; restarting"
            elif [[ ! -s "${CAPTURE_OUT}/run_finalized.json" ]]; then
                echo "run reached stop_time (t=${LAST_T} s) but run_finalized.json is missing -- previous job died during archive/staging; restarting to retry finalization"
            else
                echo "run reached stop_time (t=${LAST_T} s) and run_finalized.json exists but its archive is missing or empty -- re-running finalization" >&2
            fi
        fi
        echo "resuming run at t=${LAST_T} s toward stop_time ${RESTART_STOP_S} s"
    fi
fi

# Checkpoint rotation: retain the newest restart anchors.
# While WarpX runs, keep the newest TWO *COMPLETE* chk dirs and delete older
# COMPLETE ones plus AMReX chk*.old.* renames. Incomplete/mid-write dirs are
# never touched and never counted here, so incomplete directories cannot cause
# the newest COMPLETE restart anchor to be pruned. After a successful job the
# scratch is pruned to the single newest COMPLETE checkpoint.
rotate_checkpoints() {
    while true; do
        sleep 600
        mapfile -t _chks < <(find "${OUT}" -maxdepth 1 -type d -name 'chk*' 2>/dev/null \
            | grep -E '/chk[0-9]+$' | sort -V \
            | while read -r c; do [[ -f "${c}/rrea_checkpoint/COMPLETE" ]] && echo "${c}"; done)
        local n=${#_chks[@]}
        if (( n > 2 )); then
            local old
            for old in "${_chks[@]:0:n-2}"; do
                { rm -rf "${old}" && echo "checkpoint rotation: removed COMPLETE ${old}"; } \
                    || echo "WARNING: checkpoint rotation failed to remove ${old}" >&2
            done
        fi
        find "${OUT}" -maxdepth 1 -type d -name 'chk*.old.*' -exec rm -rf {} + 2>/dev/null || true
    done
}

TRANSPORT_CONFIG="${PROFILED_TRANSPORT_CONFIG}"

if [[ -n "${PROFILED_VIDEO_CLOUD_CURRENT_CLOSURE:-}" ]]; then
    echo "PROFILED_VIDEO_CLOUD_CURRENT_CLOSURE was removed; unset it (PIC/fluid carrier drift is the only supported path)." >&2
    exit 2
fi

test -x "${WARPX_EXE}"
test -s EFIELD_profile/digitized_efield_profile.csv
test -s EFIELD_profile/rrea_threshold_NRLMSISE00_gulf_284kVpm.csv
test -s "${TRANSPORT_CONFIG}"

"${PYTHON_BIN}" -m py_compile \
    scripts/rrea_profiled_atmosphere.py \
    scripts/rrea_run_support.py \
    scripts/run_rrea_profiled_video_capture.py

CAPTURE_ARGS=(
    scripts/run_rrea_profiled_video_capture.py
    --warpx-exe "${WARPX_EXE}"
    --output-dir "${CAPTURE_OUT}"
    --run-root "${OUT}"
    --transport-config "${TRANSPORT_CONFIG}"
    --launcher srun
    --ranks "${SLURM_NTASKS:-32}"
    --cpus-per-rank "${SLURM_CPUS_PER_TASK:-1}"
    # Keep AMReX globally serial even when extra CPUs are allocated for
    # the scoped transport/fluid regions.
    --omp-threads "${PROFILED_VIDEO_OMP_THREADS:-1}"
    # custom mode: explicit represented electron population. Screening and
    # carrier dynamics are self-consistent.
    --e0-peak-kv-per-m "${PROFILED_VIDEO_E0_KV_PER_M}"
)
# Every numeric capture default lives in config/rrea_defaults.json
# (capture_defaults) and is resolved by the driver.  The sbatch passes a
# flag ONLY when a per-run env override is explicitly set -- it carries
# NO numeric defaults of its own.
append_if_set() {
    if [[ -n "${2:-}" ]]; then
        CAPTURE_ARGS+=("$1" "$2")
    fi
}
append_if_set --n-cell-r "${PROFILED_VIDEO_N_CELL_R:-}"
append_if_set --n-cell-z "${PROFILED_VIDEO_N_CELL_Z:-}"
append_if_set --domain-radius-m "${PROFILED_VIDEO_DOMAIN_RADIUS_M:-}"
append_if_set --video-nr "${PROFILED_VIDEO_NR:-}"
append_if_set --video-nz "${PROFILED_VIDEO_NZ:-}"
# No video-extent or energy-threshold flags: the capture grid always
# spans the full domain and the threshold is fixed by the video contract;
# crop at render time only.
append_if_set --dt-s "${PROFILED_VIDEO_DT_S:-}"
append_if_set --stop-time-s "${PROFILED_VIDEO_STOP_TIME_S:-}"
append_if_set --diag-interval "${PROFILED_VIDEO_DIAG_INTERVAL:-}"
append_if_set --seed-time-window-s "${PROFILED_VIDEO_SEED_TIME_WINDOW_S:-}"
append_if_set --channel-radius-m "${PROFILED_VIDEO_CHANNEL_RADIUS_M:-}"
append_if_set --ion-mobility-model "${PROFILED_ION_MOBILITY_MODEL:-}"
append_if_set --positive-ion-reduced-mobility-stp-m2-per-vs "${PROFILED_POS_ION_K0:-}"
append_if_set --negative-ion-reduced-mobility-stp-m2-per-vs "${PROFILED_NEG_ION_K0:-}"
append_if_set --ion-mobility-density-ratio-floor "${PROFILED_ION_DENSITY_RATIO_FLOOR:-}"
append_if_set --ion-drift-cfl "${PROFILED_ION_DRIFT_CFL:-}"
append_if_set --material-conduction-target "${PROFILED_MATERIAL_CONDUCTION_TARGET:-}"
append_if_set --material-conduction-guard "${PROFILED_MATERIAL_CONDUCTION_GUARD:-}"
append_if_set --material-diffusion-target "${PROFILED_MATERIAL_DIFFUSION_TARGET:-}"
append_if_set --material-diffusion-guard "${PROFILED_MATERIAL_DIFFUSION_GUARD:-}"
append_if_set --material-reaction-target "${PROFILED_MATERIAL_REACTION_TARGET:-}"
append_if_set --material-reaction-guard "${PROFILED_MATERIAL_REACTION_GUARD:-}"
append_if_set --material-min-substeps "${PROFILED_MATERIAL_MIN_SUBSTEPS:-}"
append_if_set --material-max-substeps "${PROFILED_MATERIAL_MAX_SUBSTEPS:-}"
append_if_set --population-controller-rng-salt "${PROFILED_VIDEO_POPULATION_CONTROLLER_RNG_SALT:-}"
# Unset derives the charged cutoff from the table's certified range. Set only for a
# cutoff A/B on one executable and one bundle: the tables are general across
# cutoffs, so the flag is the sole difference between the two runs.
append_if_set --low-energy-cutoff-eV "${PROFILED_VIDEO_LOW_ENERGY_CUTOFF_EV:-}"
append_if_set --video-frame-interval-s "${PROFILED_VIDEO_FRAME_INTERVAL_S:-}"
if [[ -n "${PROFILED_VIDEO_CAPTURE_FINAL_TIME_S:-}" ]]; then
    CAPTURE_ARGS+=(
        --video-capture-final-time-s "${PROFILED_VIDEO_CAPTURE_FINAL_TIME_S}"
    )
fi

if [[ -z "${PROFILED_VIDEO_RESTART_FROM:-}" ]]; then
    CAPTURE_ARGS+=(--overwrite-output)
fi

# Unset lets the driver resolve seed_source_model from config/rrea_defaults.json.
append_if_set --seed-source-model "${PROFILED_VIDEO_SEED_SOURCE_MODEL:-}"
CAPTURE_ARGS+=(--seed-realization-id "${PROFILED_VIDEO_SEED_REALIZATION_ID:-0}")
if [[ -n "${PROFILED_VIDEO_TRANSPORT_RNG_SEED:-}" ]]; then
    CAPTURE_ARGS+=(--transport-rng-seed "${PROFILED_VIDEO_TRANSPORT_RNG_SEED}")
fi
if [[ "${PROFILED_VIDEO_SEED_SOURCE_MODEL:-}" == "coleman_dwyer_7p2mev" ]]; then
    # The default is 100k unit-weight simultaneous seeds. The only permitted
    # escalation is an explicitly labeled 200k run after inadequate 100k
    # ensemble statistics; the driver still disables population resampling.
    CAPTURE_ARGS+=(--cd-plane-diagnostics)
    if [[ "${PROFILED_VIDEO_CD_SEED_ESCALATION_200K:-0}" == "1" ]]; then
        if [[ -z "${PROFILED_VIDEO_CD_SEED_ESCALATION_REASON:-}" ]]; then
            echo "PROFILED_VIDEO_CD_SEED_ESCALATION_REASON is required for the 200k C&D escalation." >&2
            exit 2
        fi
        CAPTURE_ARGS+=(
            --cd-seed-escalation-200k
            --cd-seed-escalation-reason "${PROFILED_VIDEO_CD_SEED_ESCALATION_REASON}"
        )
    fi
else
    # Production sources: unset lets the driver resolve macro_count from the
    # JSON per model (pulse: seed_macro_count; PARMA: parma_continuous_source
    # .macro_count). A launcher fallback here would silently override the
    # model-specific resolution.
    append_if_set --macro-count "${PROFILED_VIDEO_MACRO_COUNT:-}"
    append_if_set --injected-real-electrons \
        "${PROFILED_VIDEO_INJECTED_REAL_ELECTRONS:-}"
fi
if [[ -n "${PROFILED_VIDEO_MAX_ELECTRON_MACROS:-}" \
   || -n "${PROFILED_VIDEO_MAX_PHOTON_MACROS:-}" \
   || -n "${PROFILED_VIDEO_MAX_POSITRON_MACROS:-}" ]]; then
    : "${PROFILED_VIDEO_MAX_ELECTRON_MACROS:?set all three strict macro ceilings}"
    : "${PROFILED_VIDEO_MAX_PHOTON_MACROS:?set all three strict macro ceilings}"
    : "${PROFILED_VIDEO_MAX_POSITRON_MACROS:?set all three strict macro ceilings}"
    CAPTURE_ARGS+=(--max-electron-macros "${PROFILED_VIDEO_MAX_ELECTRON_MACROS}")
    CAPTURE_ARGS+=(--max-photon-macros "${PROFILED_VIDEO_MAX_PHOTON_MACROS}")
    CAPTURE_ARGS+=(--max-positron-macros "${PROFILED_VIDEO_MAX_POSITRON_MACROS}")
fi

# The driver derives the normal radial taper from config. Setting either
# TAPER_R_* is an explicit override and requires both; the driver audits the
# full 2-D fringe against the local threshold before launch.
if [[ -n "${TAPER_R_START_M:-}" ]]; then
    CAPTURE_ARGS+=(--field-taper-r-start-m "${TAPER_R_START_M}")
fi
if [[ -n "${TAPER_R_END_M:-}" ]]; then
    CAPTURE_ARGS+=(--field-taper-r-end-m "${TAPER_R_END_M}")
fi

# Warp-aware capture: extra frames paced by the cinematic playback schedule
# (same spec format as the renderer's --warp-spec) so the time-warped render
# interpolates only across tiny gaps.
if [[ -n "${PROFILED_VIDEO_WARP_CAPTURE_SPEC:-}" ]]; then
    CAPTURE_ARGS+=(--warp-capture-spec "${PROFILED_VIDEO_WARP_CAPTURE_SPEC}")
    append_if_set --warp-capture-fps "${PROFILED_VIDEO_WARP_CAPTURE_FPS:-}"
fi
# The rank-independent max_grid_size default lives in the driver next to the
# other JSON defaults; pass only a deliberate override.
append_if_set --max-grid-size "${PROFILED_VIDEO_MAX_GRID_SIZE:-}"
: "${PROFILED_VIDEO_TRANSPORT_OMP_THREADS:?submitter must set scoped transport threads}"
CAPTURE_ARGS+=(--transport-omp-threads "${PROFILED_VIDEO_TRANSPORT_OMP_THREADS}")
# Particle load balancing (the avalanche concentrates in the seed channel).
if [[ -n "${PROFILED_VIDEO_LOAD_BALANCE_INTERVALS:-}" ]]; then
    CAPTURE_ARGS+=(--load-balance-intervals "${PROFILED_VIDEO_LOAD_BALANCE_INTERVALS}")
fi
# WarpX checkpoints for multi-day safety (restart: amr.restart <chk dir>).
# Checkpoint v2 validates its format, state, physics configuration and mesh;
# intentional engine, physics-model or rank changes require a restart epoch.
# Checkpoint v1 is unsupported.
if [[ -n "${PROFILED_VIDEO_CHECKPOINT_INT:-}" ]]; then
    CAPTURE_ARGS+=(--checkpoint-interval "${PROFILED_VIDEO_CHECKPOINT_INT}")
    CAPTURE_ARGS+=(--checkpoint-prefix "${OUT}/chk")
fi
if [[ -n "${PROFILED_VIDEO_EXTRA_WARPX_ARGS:-}" ]]; then
    for extra in ${PROFILED_VIDEO_EXTRA_WARPX_ARGS}; do
        CAPTURE_ARGS+=(--extra-warpx-arg "${extra}")
    done
fi
if [[ -n "${PROFILED_VIDEO_RESTART_FROM:-}" ]]; then
    CAPTURE_ARGS+=(--restart-checkpoint "${PROFILED_VIDEO_RESTART_FROM}")
fi
# Continuation-epoch escalation. Setting the REASON
# arms a break restart: engine, MPI-layout or physics changes are recorded as
# deviations instead of aborting.
# Cannot ride EXTRA_WARPX_ARGS -- rrea.* is a protected prefix there.
if [[ -n "${PROFILED_VIDEO_RESTART_EPOCH_REASON:-}" ]]; then
    CAPTURE_ARGS+=(
        --restart-epoch-break
        --restart-epoch-reason "${PROFILED_VIDEO_RESTART_EPOCH_REASON}"
    )
fi
# adaptive_resample_v1 population control.  The target comes from the
# SINGLE source config/rrea_defaults.json; do not override it here except
# with PROFILED_VIDEO_POPULATION_TARGET=0 to disable control entirely
# (the strict C&D campaign).  Strict-matched across restarts.
if [[ -n "${PROFILED_VIDEO_POPULATION_TARGET:-}" ]]; then
    CAPTURE_ARGS+=(--population-target-electron-macros "${PROFILED_VIDEO_POPULATION_TARGET}")
fi
if [[ -n "${PROFILED_VIDEO_POPULATION_INTERVAL:-}" ]]; then
    CAPTURE_ARGS+=(--population-control-interval "${PROFILED_VIDEO_POPULATION_INTERVAL}")
fi
if [[ -n "${PROFILED_VIDEO_POPULATION_SPLIT_MIN_WEIGHT:-}" ]]; then
    CAPTURE_ARGS+=(--population-split-min-weight "${PROFILED_VIDEO_POPULATION_SPLIT_MIN_WEIGHT}")
fi
# A Coleman-Dwyer realization refuses resampling unless it is opted into with a
# reason, because the analysis pools realizations and resampling adds
# variance. Both variables are required together; the driver rejects either
# alone, so a half-set job fails at argument parsing rather than mid-run.
if [[ -n "${PROFILED_VIDEO_CD_ALLOW_RESAMPLING:-}" ]]; then
    CAPTURE_ARGS+=(--cd-allow-population-resampling)
fi
if [[ -n "${PROFILED_VIDEO_CD_RESAMPLING_REASON:-}" ]]; then
    CAPTURE_ARGS+=(
        --cd-population-resampling-reason "${PROFILED_VIDEO_CD_RESAMPLING_REASON}")
fi
# Reduced-statistics engine diagnostic (timestep ladders), never a
# full-statistics realization; the C&D planner keeps it out of pooled results.
if [[ -n "${PROFILED_VIDEO_CD_DIAGNOSTIC_SEEDS:-}" ]]; then
    CAPTURE_ARGS+=(--cd-diagnostic-seed-count "${PROFILED_VIDEO_CD_DIAGNOSTIC_SEEDS}")
fi
if [[ -n "${PROFILED_VIDEO_CD_DIAGNOSTIC_REASON:-}" ]]; then
    CAPTURE_ARGS+=(--cd-diagnostic-reason "${PROFILED_VIDEO_CD_DIAGNOSTIC_REASON}")
fi

rotate_checkpoints &
ROTATE_PID=$!

"${PYTHON_BIN}" "${CAPTURE_ARGS[@]}"

stop_rotator
# Job finished cleanly: prune scratch to the single newest COMPLETE
# checkpoint (keep-only-last rule; the marker filter also drops any
# incomplete newer dir so an incomplete chk can never be the survivor).
mapfile -t FINAL_CHKS < <(find "${OUT}" -maxdepth 1 -type d -name 'chk*' 2>/dev/null | grep -E '/chk[0-9]+$' | sort -V)
KEEP_CHK=""
for c in "${FINAL_CHKS[@]}"; do
    [[ -f "${c}/rrea_checkpoint/COMPLETE" ]] && KEEP_CHK="${c}"
done
for old in "${FINAL_CHKS[@]}"; do
    if [[ "${old}" != "${KEEP_CHK}" ]]; then
        { rm -rf "${old}" && echo "post-run prune: removed ${old}"; } \
            || echo "WARNING: post-run prune failed to remove ${old}" >&2
    fi
done
find "${OUT}" -maxdepth 1 -type d -name 'chk*.old.*' -exec rm -rf {} + 2>/dev/null || true

test -s "${CAPTURE_OUT}/rrea_video/video_frames.bin"
test -s "${CAPTURE_OUT}/rrea_video/video_frame_metadata.csv"
test -s "${CAPTURE_OUT}/profiled_video_capture_summary.json"

# Rendering/encoding is local, or on-cluster via the dedicated
# cluster/hpc/slurm_rrea_profiled_video_render.sbatch job.

# Archive the COMPLETE re-render inputs (raw frame stream + all metadata +
# compact evidence) straight into permanent storage. This is the product that
# gets downloaded and also the on-cluster archival copy; it must never be
# deleted by cleanup scripts. Excludes only per-step rrea_diag trees and
# checkpoints.
ARCHIVE="${PERMANENT_OUT}/video_capture_${SLURM_JOB_ID:-manual}.tar"
if command -v zstd >/dev/null 2>&1; then
    ARCHIVE="${ARCHIVE}.zst"
    tar -C "${OUT}" \
        --exclude='rrea_diag' \
        --exclude='chk*' \
        --exclude='diags' \
        -I 'zstd -T0 -8' -cf "${ARCHIVE}" capture
else
    ARCHIVE="${ARCHIVE}.gz"
    tar -C "${OUT}" \
        --exclude='rrea_diag' \
        --exclude='chk*' \
        --exclude='diags' \
        -czf "${ARCHIVE}" capture
fi
du -h "${ARCHIVE}"

# The final marker is written only after stream checks and archive creation.
printf '{"format": "rrea_run_finalized_v2", "job_id": "%s", "finalized_utc": "%s", "archive": "%s"}\n' \
    "${SLURM_JOB_ID:-manual}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    "${ARCHIVE}" > "${CAPTURE_OUT}/run_finalized.json"
echo "finalization marker written: ${CAPTURE_OUT}/run_finalized.json"

ln -sfn "${PERMANENT_OUT}" "${REMOTE_ROOT}/runs/profiled_video_schema6_latest"
du -sh "${PERMANENT_OUT}" || true
find "${PERMANENT_OUT}" -type f | wc -l || true
date
echo "profiled schema-6 scratch output: ${OUT}"
echo "profiled schema-6 staged output: ${PERMANENT_OUT}"
echo "re-render archive: ${ARCHIVE}"
