#!/usr/bin/env python3
"""Plan, or submit, the production-path RREA campaigns.

This file is orchestration only.  Every case invokes the one canonical
``slurm_rrea_profiled_video_capture.sbatch`` launcher; it never constructs a
WarpX command or implements a second simulation path.  Planning is the default
and has no cluster side effects. Submission uses ``--submit`` and performs the
required quota checks first.
"""

from __future__ import annotations

import argparse
import functools
import json
import math
import os
import re
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from rrea_cd_physics import (  # noqa: E402
    CD_ALL_PHYSICS_10US_E0_KV_PER_M,
    CD_ALL_PHYSICS_10US_DT_ANCHOR_E0_KV_PER_M,
    CD_ALL_PHYSICS_10US_LABEL,
    CD_ALL_PHYSICS_10US_VIDEO_FRAME_INTERVAL_S,
    CD_ALL_PHYSICS_10US_SCOPE,
    CD_ALL_PHYSICS_10US_STOP_TIME_S,
    CD_ALL_PHYSICS_10US_VIDEO_NR,
    CD_ALL_PHYSICS_10US_VIDEO_NZ,
    CD_ALL_PHYSICS_CONFIRM_100K_IDS,
    CD_ALL_PHYSICS_CONFIRM_200K_IDS,
    CD_ALL_PHYSICS_ENGINEERING_IDS,
    CD_COHORT_CONFIRMATORY_100K,
    CD_COHORT_CONFIRMATORY_200K,
    CD_COHORT_ENGINEERING,
    CD_ESCALATED_SEED_COUNT,
    CD_INITIAL_SEED_COUNT,
    CD_GLOBAL_OMP_THREADS, CD_HPC_CPUS_PER_RANK, CD_HPC_MPI_RANKS,
    CD_POISSON_MAX_BOTTOM_ITERATIONS,
    CD_PRODUCTION_TAPER_GEOMETRY,
    CD_RESOURCE_LIMITED_E0_KV_PER_M,
    cd_population_ceilings,
    cd_seed_design,
    cd_transport_rng_seed,
)
from rrea_run_support import (  # noqa: E402
    CAPTURE_DEFAULTS,
    DEFAULT_POPULATION_CONTROL_INTERVAL,
)
from rrea_profiled_atmosphere import (  # noqa: E402
    DEFAULT_DENSITY_PROFILE,
    DEFAULT_FIELD_PROFILE,
    coleman_dwyer_profile_reference,
    load_density_ratio_profile,
    load_field_profile,
    load_threshold_profile,
)

CANONICAL_SBATCH = ROOT / "cluster/hpc/slurm_rrea_profiled_video_capture.sbatch"
DEFAULT_REMOTE_ROOT = Path(
    os.environ.get("REMOTE_ROOT") or "~/rrea_hpc"
).expanduser()
INITIAL_DT_NS = (2.5, 1.25, 0.625)
CAMPAIGN_PLAN_FORMAT = "rrea_cd_all_physics_10us_campaign_plan_v1"
COHORT_REALIZATION_IDS = {
    CD_COHORT_ENGINEERING: CD_ALL_PHYSICS_ENGINEERING_IDS,
    CD_COHORT_CONFIRMATORY_100K: CD_ALL_PHYSICS_CONFIRM_100K_IDS,
    CD_COHORT_CONFIRMATORY_200K: CD_ALL_PHYSICS_CONFIRM_200K_IDS,
}
STAGE_DEFAULT_COHORT = {
    "dt-matrix": CD_COHORT_ENGINEERING,
    "e0-scan": CD_COHORT_CONFIRMATORY_100K,
    "stats-escalation": CD_COHORT_CONFIRMATORY_200K,
}
STAGE_ALLOWED_COHORTS = {
    "dt-matrix": frozenset({CD_COHORT_ENGINEERING}),
    "e0-scan": frozenset({CD_COHORT_CONFIRMATORY_100K}),
    "stats-escalation": frozenset({CD_COHORT_CONFIRMATORY_200K}),
}
LOCAL_TRANSPORT_CONFIG = ROOT / "rrea_transport_tables/schema6/production/transport_physics.json"


@dataclass(frozen=True)
class CampaignExecutionInputs:
    repo_root: PurePosixPath
    warpx_exe: PurePosixPath
    transport_config: PurePosixPath


@dataclass(frozen=True)
class CampaignCase:
    case_id: str
    stage: str
    e0_kv_per_m: float
    dt_ns: float
    realization_id: int
    seed_count: int
    domain_radius_m: float = CD_PRODUCTION_TAPER_GEOMETRY[0]
    n_cell_r: int = CD_PRODUCTION_TAPER_GEOMETRY[1]
    field_taper_r_start_m: float | None = CD_PRODUCTION_TAPER_GEOMETRY[2]
    field_taper_r_end_m: float | None = CD_PRODUCTION_TAPER_GEOMETRY[3]
    restart_checkpoint: str | None = None
    output_case_id: str | None = None
    seed_escalation_reason: str | None = None
    scratch_out: str | None = None
    attempt_history: tuple[str, ...] = ()


def _number_label(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def _validate_remote_root(path: Path) -> PurePosixPath:
    text = str(path).replace("\\", "/")
    pure = PurePosixPath(text)
    if (
        not text.startswith("/")
        or pure == PurePosixPath("/")
        or ".." in pure.parts
        or any(character in text for character in (",", "\n", "\r", "\x00"))
    ):
        raise ValueError("--remote-root must be an absolute, non-root POSIX path")
    return pure


def _resolve_existing_remote_file(path: Path, *, option: str) -> PurePosixPath:
    """Return the canonical target path recorded by the production runner."""

    pure = _validate_remote_root(path)
    try:
        resolved = Path(str(pure)).resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"{option} cannot be resolved: {pure}: {exc}") from exc
    if not resolved.is_file():
        raise ValueError(f"{option} is not a regular file: {resolved}")
    return _validate_remote_root(resolved)


def _validate_export_value(name: str, value: str) -> str:
    if any(character in value for character in (",", "\n", "\r", "\x00")):
        raise ValueError(f"{name} contains a character unsafe for Slurm --export")
    return value


def _execution_inputs_from_args(
    args: argparse.Namespace,
) -> CampaignExecutionInputs:
    remote_root = _validate_remote_root(args.remote_root)
    repo_root = _validate_remote_root(
        args.repo_root
        if args.repo_root is not None
        else Path(str(remote_root / "repo_amrex_latest"))
    )
    if args.warpx_exe is None:
        raise ValueError("--warpx-exe (or RREA_WARPX_EXE) is required for a plan")
    warpx_exe = _resolve_existing_remote_file(args.warpx_exe, option="--warpx-exe")
    transport_config = _resolve_existing_remote_file(
        args.transport_config, option="--transport-config")
    if not LOCAL_TRANSPORT_CONFIG.is_file():
        raise ValueError(f"schema-6 transport configuration is missing: {LOCAL_TRANSPORT_CONFIG}")
    load_field_profile(DEFAULT_FIELD_PROFILE)
    load_density_ratio_profile(DEFAULT_DENSITY_PROFILE)
    load_threshold_profile(DEFAULT_DENSITY_PROFILE)
    return CampaignExecutionInputs(
        repo_root=repo_root,
        warpx_exe=warpx_exe,
        transport_config=transport_config,
    )


def _validate_dt_rungs(dt_rungs_ns: tuple[float, ...] | None) -> tuple[float, ...]:
    """Return a canonical descending subset of the 2.5 ns halving ladder."""

    values = INITIAL_DT_NS if dt_rungs_ns is None else dt_rungs_ns
    if not values:
        raise ValueError("the timestep matrix requires at least one dt rung")
    if any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError("every dt rung must be finite and positive")
    if len(set(values)) != len(values):
        raise ValueError("dt rungs must be unique")
    for value in values:
        ratio = INITIAL_DT_NS[0] / value
        exponent = round(math.log2(ratio))
        if exponent < 0 or not math.isclose(
            ratio,
            2.0**exponent,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ValueError("dt rungs must belong to the 2.5 ns / 2^n halving ladder")
    return tuple(sorted(values, reverse=True))


def _case(
    *,
    stage: str,
    e0_kv_per_m: float,
    dt_ns: float,
    realization_id: int,
    seed_count: int = CD_INITIAL_SEED_COUNT,
    domain_radius_m: float = CD_PRODUCTION_TAPER_GEOMETRY[0],
    n_cell_r: int = CD_PRODUCTION_TAPER_GEOMETRY[1],
    field_taper_r_start_m: float | None = CD_PRODUCTION_TAPER_GEOMETRY[2],
    field_taper_r_end_m: float | None = CD_PRODUCTION_TAPER_GEOMETRY[3],
    restart_checkpoint: str | None = None,
    output_case_id: str | None = None,
    seed_escalation_reason: str | None = None,
    scratch_out: str | None = None,
    attempt_history: tuple[str, ...] = (),
) -> CampaignCase:
    prefix = {
        "dt-matrix": "dt",
        "e0-scan": "scan",
        "stats-escalation": "stats",
        "restart-resume": "restart",
    }[stage]
    source_tag = "cd"
    geometry_tag = f"_r{_number_label(domain_radius_m)}" + (
        f"_taper{_number_label(field_taper_r_start_m)}to{_number_label(field_taper_r_end_m)}"
        if field_taper_r_start_m is not None
        else ""
    )
    case_id = (
        f"{prefix}_{source_tag}_e{_number_label(e0_kv_per_m)}_"
        f"dt{_number_label(dt_ns)}ns{geometry_tag}_"
        f"seed{seed_count}_"
        f"r{realization_id:02d}"
    )
    case = CampaignCase(
        case_id=case_id,
        stage=stage,
        e0_kv_per_m=e0_kv_per_m,
        dt_ns=dt_ns,
        realization_id=realization_id,
        seed_count=seed_count,
        domain_radius_m=domain_radius_m,
        n_cell_r=n_cell_r,
        field_taper_r_start_m=field_taper_r_start_m,
        field_taper_r_end_m=field_taper_r_end_m,
        restart_checkpoint=restart_checkpoint,
        output_case_id=output_case_id,
        seed_escalation_reason=seed_escalation_reason,
        scratch_out=scratch_out,
        attempt_history=attempt_history,
    )
    return case


def build_cases(
    stage: str,
    *,
    realization_ids: tuple[int, ...] | None = None,
    cohort: str | None = None,
    accepted_dt_ns: float | None = None,
    dt_rungs_ns: tuple[float, ...] | None = None,
    target_e0_kv_per_m: float | None = None,
    seed_escalation_reason: str | None = None,
) -> list[CampaignCase]:
    """Return one explicit campaign stage; this function never submits it."""
    if stage not in STAGE_DEFAULT_COHORT:
        raise ValueError(f"unknown campaign stage {stage!r}")
    selected_cohort = cohort or STAGE_DEFAULT_COHORT[stage]
    if selected_cohort not in COHORT_REALIZATION_IDS:
        raise ValueError(f"unknown campaign cohort {selected_cohort!r}")
    if selected_cohort not in STAGE_ALLOWED_COHORTS[stage]:
        allowed = ", ".join(sorted(STAGE_ALLOWED_COHORTS[stage]))
        raise ValueError(
            f"{stage} does not permit cohort {selected_cohort!r}; allowed cohort(s): {allowed}"
        )
    cohort_ids = COHORT_REALIZATION_IDS[selected_cohort]
    if realization_ids is None:
        realization_ids = cohort_ids
    if not realization_ids or len(set(realization_ids)) != len(realization_ids):
        raise ValueError("realization ids must be a non-empty unique sequence")
    if not set(realization_ids).issubset(cohort_ids):
        raise ValueError(
            f"realization ids for {selected_cohort} must be selected from {list(cohort_ids)}"
        )
    seed_count = (
        CD_ESCALATED_SEED_COUNT
        if selected_cohort == CD_COHORT_CONFIRMATORY_200K
        else CD_INITIAL_SEED_COUNT
    )
    cd_seed_design(seed_count, seed_escalation_reason)
    cases: list[CampaignCase] = []
    if stage == "dt-matrix":
        dt_values = _validate_dt_rungs(dt_rungs_ns)
        for e0 in CD_ALL_PHYSICS_10US_DT_ANCHOR_E0_KV_PER_M:
            for dt_ns in dt_values:
                for realization in realization_ids:
                    cases.append(
                        _case(
                            stage=stage,
                            e0_kv_per_m=e0,
                            dt_ns=dt_ns,
                            realization_id=realization,
                            seed_count=seed_count,
                            seed_escalation_reason=seed_escalation_reason,
                        )
                    )
    elif stage == "e0-scan":
        if accepted_dt_ns is None or not math.isfinite(accepted_dt_ns) or accepted_dt_ns <= 0:
            raise ValueError("e0-scan requires a finite positive accepted dt")
        for e0 in CD_ALL_PHYSICS_10US_E0_KV_PER_M:
            for realization in realization_ids:
                cases.append(
                    _case(
                        stage=stage,
                        e0_kv_per_m=e0,
                        dt_ns=accepted_dt_ns,
                        realization_id=realization,
                        seed_count=seed_count,
                        seed_escalation_reason=seed_escalation_reason,
                    )
                )
    elif stage == "stats-escalation":
        if accepted_dt_ns is None or not math.isfinite(accepted_dt_ns) or accepted_dt_ns <= 0:
            raise ValueError("stats-escalation requires a finite positive accepted dt")
        if target_e0_kv_per_m is None or not math.isfinite(target_e0_kv_per_m):
            raise ValueError("stats-escalation requires a finite target E0")
        if target_e0_kv_per_m not in CD_ALL_PHYSICS_10US_E0_KV_PER_M:
            raise ValueError("stats-escalation target E0 is outside campaign scope")
        for realization in realization_ids:
            cases.append(
                _case(
                    stage=stage,
                    e0_kv_per_m=target_e0_kv_per_m,
                    dt_ns=accepted_dt_ns,
                    realization_id=realization,
                    seed_count=seed_count,
                    seed_escalation_reason=seed_escalation_reason,
                )
            )
    if len({case.case_id for case in cases}) != len(cases):
        raise AssertionError("campaign case ids are not unique")
    return cases


def build_restart_resume_cases(
    resume_document: dict,
    *,
    selected_case_ids: tuple[str, ...],
) -> list[CampaignCase]:
    if not isinstance(resume_document, dict):
        raise ValueError("restart source plan must be a JSON object")
    records = {
        record["case_id"]: record
        for record in resume_document.get("cases", [])
        if isinstance(record, dict) and isinstance(record.get("case_id"), str)
    }
    if not selected_case_ids:
        raise ValueError("restart-resume requires at least one --resume-case-id")
    if len(set(selected_case_ids)) != len(selected_case_ids):
        raise ValueError("--resume-case-id values must be unique")
    missing = sorted(set(selected_case_ids) - set(records))
    if missing:
        raise ValueError(f"restart source plan lacks case ids: {missing}")
    cases = []
    for source_id in selected_case_ids:
        record = records[source_id]
        source_environment = record.get("environment")
        if not isinstance(source_environment, dict):
            raise ValueError(f"{source_id}: source plan lacks its environment")
        scratch_out = source_environment.get("PROFILED_VIDEO_SCRATCH_OUT")
        if not isinstance(scratch_out, str):
            raise ValueError(f"{source_id}: source plan lacks exact scratch output")
        scratch_path = _validate_remote_root(Path(scratch_out))
        previous_stop_s = float(source_environment.get("PROFILED_VIDEO_STOP_TIME_S", 0.0))
        if not math.isclose(
            previous_stop_s,
            CD_ALL_PHYSICS_10US_STOP_TIME_S,
            rel_tol=0.0,
            abs_tol=1.0e-18,
        ):
            raise ValueError(f"{source_id}: restart source did not target exactly 10 us")
        prior_history = record.get("attempt_history", [])
        if not isinstance(prior_history, (list, tuple)) or any(
            not isinstance(item, str) or not item
            for item in prior_history
        ):
            raise ValueError(f"{source_id}: invalid attempt history")
        cases.append(
            _case(
                stage="restart-resume",
                e0_kv_per_m=float(record["e0_kv_per_m"]),
                dt_ns=float(record["dt_ns"]),
                realization_id=int(record["realization_id"]),
                seed_count=int(record.get("seed_count", CD_INITIAL_SEED_COUNT)),
                domain_radius_m=float(
                    record.get("domain_radius_m", CD_PRODUCTION_TAPER_GEOMETRY[0])
                ),
                n_cell_r=int(
                    record.get("n_cell_r", CD_PRODUCTION_TAPER_GEOMETRY[1])
                ),
                field_taper_r_start_m=record.get(
                    "field_taper_r_start_m", CD_PRODUCTION_TAPER_GEOMETRY[2]
                ),
                field_taper_r_end_m=record.get(
                    "field_taper_r_end_m", CD_PRODUCTION_TAPER_GEOMETRY[3]
                ),
                restart_checkpoint="auto",
                output_case_id=str(record.get("output_case_id") or source_id),
                seed_escalation_reason=record.get("seed_escalation_reason"),
                scratch_out=str(scratch_path),
                attempt_history=tuple(prior_history) + (source_id,),
            )
        )
    return cases


def case_environment(
    case: CampaignCase,
    *,
    campaign_id: str,
    remote_root: Path,
    execution_inputs: CampaignExecutionInputs | None = None,
) -> dict[str, str]:
    remote_root_posix = _validate_remote_root(remote_root)
    dt_s = case.dt_ns * 1.0e-9
    checkpoint_interval = max(1, round(CD_ALL_PHYSICS_10US_STOP_TIME_S / (2.0 * dt_s)))
    output_case_id = case.output_case_id or case.case_id
    environment = {
        "REMOTE_ROOT": str(remote_root_posix),
        "PROFILED_VIDEO_E0_KV_PER_M": f"{case.e0_kv_per_m:.17g}",
        "PROFILED_VIDEO_DT_S": f"{dt_s:.17g}",
        "PROFILED_VIDEO_STOP_TIME_S": f"{CD_ALL_PHYSICS_10US_STOP_TIME_S:.17g}",
        "PROFILED_VIDEO_CAPTURE_FINAL_TIME_S": f"{CD_ALL_PHYSICS_10US_STOP_TIME_S:.17g}",
        "PROFILED_VIDEO_SEED_SOURCE_MODEL": "coleman_dwyer_7p2mev",
        "PROFILED_VIDEO_SEED_REALIZATION_ID": str(case.realization_id),
        "PROFILED_VIDEO_CHANNEL_RADIUS_M": str(CAPTURE_DEFAULTS["channel_radius_m"]),
        "PROFILED_VIDEO_TRANSPORT_RNG_SEED": str(cd_transport_rng_seed(case.realization_id)),
        "PROFILED_VIDEO_POPULATION_TARGET": "0",
        "PROFILED_VIDEO_CHECKPOINT_INT": str(checkpoint_interval),
        "PROFILED_VIDEO_FRAME_INTERVAL_S": f"{CD_ALL_PHYSICS_10US_VIDEO_FRAME_INTERVAL_S:.17g}",
        "PROFILED_VIDEO_NR": str(CD_ALL_PHYSICS_10US_VIDEO_NR),
        "PROFILED_VIDEO_NZ": str(CD_ALL_PHYSICS_10US_VIDEO_NZ),
        "PROFILED_VIDEO_OMP_THREADS": str(CD_GLOBAL_OMP_THREADS),
        "PROFILED_VIDEO_TRANSPORT_OMP_THREADS": str(CD_HPC_CPUS_PER_RANK),
        "PROFILED_VIDEO_MAX_GRID_SIZE": str(CAPTURE_DEFAULTS["max_grid_size"]),
        "PROFILED_VIDEO_DOMAIN_RADIUS_M": f"{case.domain_radius_m:.17g}",
        "PROFILED_VIDEO_N_CELL_R": str(case.n_cell_r),
        "PROFILED_VIDEO_N_CELL_Z": str(CAPTURE_DEFAULTS["n_cell_z"]),
        "PROFILED_VIDEO_SCRATCH_OUT": (
            case.scratch_out
            if case.scratch_out is not None
            else str(
                remote_root_posix / "runs/rrea_cd_campaign_work" / campaign_id / output_case_id
            )
        ),
    }
    ceilings = _case_population_ceilings(case)
    environment.update(
        {
            "PROFILED_VIDEO_MAX_ELECTRON_MACROS": str(ceilings["electron"]),
            "PROFILED_VIDEO_MAX_PHOTON_MACROS": str(ceilings["photon"]),
            "PROFILED_VIDEO_MAX_POSITRON_MACROS": str(ceilings["positron"]),
        }
    )
    if execution_inputs is not None:
        environment.update(
            {
                "REPO_ROOT": str(execution_inputs.repo_root),
                "RREA_WARPX_EXE": str(execution_inputs.warpx_exe),
                "PROFILED_TRANSPORT_CONFIG": str(execution_inputs.transport_config),
                # Explicit exports prevent inherited state altering the case.
                "OMP_DYNAMIC": "false",
                "OMP_PLACES": "cores",
                "OMP_PROC_BIND": "close",
                # The selected table bundle owns the absent transport cutoffs.
                "PROFILED_VIDEO_DIAG_INTERVAL": str(CAPTURE_DEFAULTS["diag_interval"]),
                "PROFILED_VIDEO_SEED_TIME_WINDOW_S": str(CAPTURE_DEFAULTS["seed_time_window_s"]),
                "PROFILED_ION_MOBILITY_MODEL": str(CAPTURE_DEFAULTS["ion_mobility_model"]),
                "PROFILED_POS_ION_K0": str(CAPTURE_DEFAULTS["positive_ion_reduced_mobility_stp_m2_per_vs"]),
                "PROFILED_NEG_ION_K0": str(CAPTURE_DEFAULTS["negative_ion_reduced_mobility_stp_m2_per_vs"]),
                "PROFILED_ION_DENSITY_RATIO_FLOOR": str(CAPTURE_DEFAULTS["ion_mobility_density_ratio_floor"]),
                "PROFILED_ION_DRIFT_CFL": str(CAPTURE_DEFAULTS["ion_drift_cfl"]),
                "PROFILED_VIDEO_POISSON_MAX_BOTTOM_ITERATIONS": str(CD_POISSON_MAX_BOTTOM_ITERATIONS),
                "PROFILED_VIDEO_POPULATION_INTERVAL": str(DEFAULT_POPULATION_CONTROL_INTERVAL),
            }
        )
    if case.seed_count == CD_ESCALATED_SEED_COUNT:
        environment["PROFILED_VIDEO_CD_SEED_ESCALATION_200K"] = "1"
        environment["PROFILED_VIDEO_CD_SEED_ESCALATION_REASON"] = str(case.seed_escalation_reason)
    if case.field_taper_r_start_m is not None:
        environment["TAPER_R_START_M"] = f"{case.field_taper_r_start_m:.17g}"
        environment["TAPER_R_END_M"] = f"{case.field_taper_r_end_m:.17g}"
    if case.restart_checkpoint is not None:
        environment["PROFILED_VIDEO_RESTART_FROM"] = case.restart_checkpoint
    return environment


@functools.lru_cache(maxsize=1)
def _canonical_profiles() -> tuple[object, object, object]:
    return (
        load_field_profile(DEFAULT_FIELD_PROFILE),
        load_density_ratio_profile(DEFAULT_DENSITY_PROFILE),
        load_threshold_profile(DEFAULT_DENSITY_PROFILE),
    )


@functools.lru_cache(maxsize=None)
def _population_ceilings(e0_kv_per_m: float, seed_count: int) -> dict[str, int]:
    field, density, threshold = _canonical_profiles()
    reference = coleman_dwyer_profile_reference(
        field=field,
        density_ratio=density,
        threshold_profile=threshold,
        e0_peak_v_per_m=e0_kv_per_m * 1000.0,
    )
    if reference is None:
        raise ValueError(f"E0={e0_kv_per_m:g}: no 276-defined avalanche region")
    multiplication_276 = reference["curves"]["276"]["predicted_multiplication"]
    if multiplication_276 is None:
        raise ValueError(f"E0={e0_kv_per_m:g}: non-finite 276 multiplier")
    return cd_population_ceilings(seed_count, float(multiplication_276))


def _case_population_ceilings(case: CampaignCase) -> dict[str, int]:
    return _population_ceilings(case.e0_kv_per_m, case.seed_count)


def sbatch_command(
    case: CampaignCase,
    *,
    campaign_id: str,
    remote_root: Path,
    execution_inputs: CampaignExecutionInputs,
    canonical_sbatch: Path = CANONICAL_SBATCH,
) -> list[str]:
    if canonical_sbatch.resolve() != CANONICAL_SBATCH.resolve():
        raise ValueError("C&D campaigns may use only the canonical capture sbatch")
    environment = case_environment(
        case,
        campaign_id=campaign_id,
        remote_root=remote_root,
        execution_inputs=execution_inputs,
    )
    # An explicit export list prevents an unrelated login-shell PROFILED_*,
    # RREA_*, OMP_*, or AMReX control from silently changing a planned
    # campaign case.
    export_text = ",".join(
        f"{key}={_validate_export_value(key, value)}" for key, value in sorted(environment.items())
    )
    job_name = f"rrea_{case.case_id}"[:128]
    logs_root = _validate_remote_root(Path(str(remote_root)) / "logs")
    command = [
        "sbatch",
        "--parsable",
        "--nodes=1",
        f"--ntasks={CD_HPC_MPI_RANKS}",
        f"--cpus-per-task={CD_HPC_CPUS_PER_RANK}",
        "--mem=96G",
        "--time=48:00:00",
        f"--job-name={job_name}",
        # Log location lives in the site configuration, not in the sbatch.
        f"--output={logs_root}/rrea_{case.case_id}_%j.out",
        f"--error={logs_root}/rrea_{case.case_id}_%j.err",
        f"--export={export_text}",
        str(canonical_sbatch),
    ]
    if os.environ.get("HPC_SLURM_ACCOUNT"):
        command.insert(2, f"--account={os.environ['HPC_SLURM_ACCOUNT']}")
    return command


def _campaign_case_record(
    case: CampaignCase,
    *,
    campaign_id: str,
    remote_root: Path,
    execution_inputs: CampaignExecutionInputs,
    canonical_sbatch: Path,
) -> dict:
    case_fields = asdict(case)
    case_fields["attempt_history"] = list(case.attempt_history)
    command = sbatch_command(
        case,
        campaign_id=campaign_id,
        remote_root=remote_root,
        execution_inputs=execution_inputs,
        canonical_sbatch=canonical_sbatch,
    )
    return {
        **case_fields,
        "environment": case_environment(
            case,
            campaign_id=campaign_id,
            remote_root=remote_root,
            execution_inputs=execution_inputs,
        ),
        "sbatch_command": command,
        "sbatch_command_shell": shlex.join(command),
    }


def _run_quota_checks() -> dict[str, str]:
    evidence: dict[str, str] = {}
    cost = subprocess.run(["cost"], text=True, capture_output=True, check=False)
    if cost.returncode != 0:
        raise SystemExit(f"cost check failed before submission: {cost.stderr.strip()}")
    evidence["cost"] = cost.stdout
    dusage = subprocess.run(["dusage"], text=True, capture_output=True, check=False)
    if dusage.returncode != 0:
        clean_env = os.environ.copy()
        for key in (
            "PYTHONPATH",
            "PYTHONHOME",
            "CONDA_PREFIX",
            "CONDA_DEFAULT_ENV",
            "VIRTUAL_ENV",
        ):
            clean_env.pop(key, None)
        dusage = subprocess.run(
            ["dusage"], env=clean_env, text=True, capture_output=True, check=False
        )
    if dusage.returncode != 0:
        raise SystemExit(f"dusage check failed before submission: {dusage.stderr.strip()}")
    evidence["dusage"] = dusage.stdout
    print(cost.stdout, end="")
    print(dusage.stdout, end="")
    return evidence


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--stage",
        choices=[
            "dt-matrix",
            "e0-scan",
            "stats-escalation",
            "restart-resume",
        ],
        required=True,
    )
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--accepted-dt-ns", type=float)
    parser.add_argument(
        "--dt-rung-ns",
        action="append",
        type=float,
        help=("Explicit 2.5 ns / 2^n rung; repeat as needed. Defaults to 2.5, 1.25, and 0.625 ns."),
    )
    parser.add_argument("--target-e0-kv-per-m", type=float)
    parser.add_argument("--resume-plan", type=Path)
    parser.add_argument("--resume-case-id", action="append")
    parser.add_argument("--realization-id", action="append", type=int)
    parser.add_argument(
        "--cohort",
        choices=sorted(COHORT_REALIZATION_IDS),
        help="Fixed realization cohort; defaults by stage.",
    )
    parser.add_argument(
        "--case-id",
        action="append",
        help="Select exact generated case IDs for a canary or bounded cohort.",
    )
    parser.add_argument(
        "--e0-kv-per-m",
        action="append",
        type=float,
        help="Select exact generated E0 values; repeat for multiple fields.",
    )
    parser.add_argument(
        "--max-in-flight",
        type=int,
        choices=(1, 2),
        default=1,
        help="Maximum running jobs, enforced as one or two Slurm dependency chains.",
    )
    parser.add_argument("--seed-escalation-reason")
    parser.add_argument("--remote-root", type=Path, default=DEFAULT_REMOTE_ROOT)
    parser.add_argument(
        "--repo-root",
        type=Path,
        help="Exact absolute HPC repository root (default: REMOTE_ROOT/repo_amrex_latest).",
    )
    parser.add_argument(
        "--warpx-exe",
        type=Path,
        default=(Path(os.environ["RREA_WARPX_EXE"]) if os.environ.get("RREA_WARPX_EXE") else None),
        help="Exact absolute schema-6 WarpX executable path.",
    )
    parser.add_argument(
        "--transport-config",
        type=Path,
        required=True,
        help="Absolute schema-6 transport_physics.json path.",
    )
    parser.add_argument(
        "--resource-limited-e0",
        action="append",
        type=float,
        choices=(CD_RESOURCE_LIMITED_E0_KV_PER_M,),
        default=[],
        help="Exclude E119 from this stage as a single-node resource limit.",
    )
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", args.campaign_id):
        parser.error("--campaign-id must be a safe 1-64 character slug")
    if args.stage in {"e0-scan", "stats-escalation"} and args.accepted_dt_ns is None:
        parser.error(f"--accepted-dt-ns is required for --stage {args.stage}")
    if args.stage == "dt-matrix" and args.accepted_dt_ns is not None:
        parser.error("--accepted-dt-ns does not apply to --stage dt-matrix")
    if args.dt_rung_ns is not None and args.stage != "dt-matrix":
        parser.error("--dt-rung-ns applies only to dt-matrix")
    if args.stage == "stats-escalation" and args.target_e0_kv_per_m is None:
        parser.error("stats-escalation requires --target-e0-kv-per-m")
    if args.stage != "stats-escalation" and args.target_e0_kv_per_m is not None:
        parser.error("--target-e0-kv-per-m applies only to stats-escalation")
    if args.stage == "restart-resume":
        if (
            args.resume_plan is None
            or not args.resume_case_id
        ):
            parser.error("restart-resume requires --resume-plan and --resume-case-id")
    elif any(
        value is not None
        for value in (
            args.resume_plan,
            args.resume_case_id,
        )
    ):
        parser.error("restart options apply only to restart-resume")
    if args.stage == "restart-resume" and any(
        value is not None
        for value in (
            args.cohort,
            args.realization_id,
            args.e0_kv_per_m,
            args.case_id,
            args.seed_escalation_reason,
        )
    ):
        parser.error("restart-resume selects cases only through --resume-case-id")
    if args.stage != "restart-resume":
        args.cohort = args.cohort or STAGE_DEFAULT_COHORT[args.stage]
        if args.output_json is None and args.stage == "stats-escalation":
            parser.error(
                "stats-escalation requires an explicit unique --output-json"
            )
    try:
        if args.stage != "restart-resume":
            seed_count = (
                CD_ESCALATED_SEED_COUNT
                if args.cohort == CD_COHORT_CONFIRMATORY_200K
                else CD_INITIAL_SEED_COUNT
            )
            cd_seed_design(seed_count, args.seed_escalation_reason)
        if args.dt_rung_ns is not None:
            _validate_dt_rungs(tuple(args.dt_rung_ns))
    except ValueError as exc:
        parser.error(str(exc))
    try:
        _validate_remote_root(args.remote_root)
    except ValueError as exc:
        parser.error(str(exc))
    args.canonical_sbatch = CANONICAL_SBATCH
    if args.output_json is None:
        args.output_json = (
            ROOT / "runs_rrea/campaign_plans" / f"{args.campaign_id}_{args.stage}.json"
        )
    immutable_parent_paths = [
        path
        for path in (
            args.resume_plan,
        )
        if path is not None
    ]
    if any(args.output_json.resolve() == path.resolve() for path in immutable_parent_paths):
        parser.error("--output-json must differ from every immutable parent plan")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not CANONICAL_SBATCH.is_file():
        raise SystemExit(f"canonical sbatch not found: {CANONICAL_SBATCH}")
    try:
        execution_inputs = _execution_inputs_from_args(args)
    except (OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    resume_document: dict | None = None
    if args.resume_plan is not None:
        try:
            resume_raw = args.resume_plan.read_bytes()
        except OSError as exc:
            raise SystemExit(f"cannot read --resume-plan: {exc}") from exc
        try:
            resume_document = json.loads(resume_raw)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"restart source plan is invalid JSON: {exc}") from exc
        if not isinstance(resume_document, dict):
            raise SystemExit("restart source plan must contain one JSON object")
        if (
            resume_document.get("format") != CAMPAIGN_PLAN_FORMAT
            or resume_document.get("campaign_scope") != CD_ALL_PHYSICS_10US_SCOPE
            or Path(resume_document.get("canonical_simulation_path", "")).resolve()
            != CANONICAL_SBATCH.resolve()
        ):
            raise SystemExit("restart source plan is not a C&D campaign plan")
        source_records = {
            record.get("case_id"): record
            for record in resume_document.get("cases", [])
            if isinstance(record, dict)
        }
        for source_id in args.resume_case_id:
            source_record = source_records.get(source_id)
            if not isinstance(source_record, dict):
                raise SystemExit(f"{source_id}: restart source case is missing")
    realization_ids = tuple(args.realization_id) if args.realization_id is not None else None
    resource_limited_e0 = set(args.resource_limited_e0)
    try:
        if args.stage == "restart-resume":
            if resume_document is None:
                raise AssertionError("restart plan was not retained")
            cases = build_restart_resume_cases(
                resume_document,
                selected_case_ids=tuple(args.resume_case_id),
            )
        else:
            cases = build_cases(
                args.stage,
                realization_ids=None,
                cohort=args.cohort,
                accepted_dt_ns=args.accepted_dt_ns,
                dt_rungs_ns=(tuple(args.dt_rung_ns) if args.dt_rung_ns is not None else None),
                target_e0_kv_per_m=args.target_e0_kv_per_m,
                seed_escalation_reason=args.seed_escalation_reason,
            )
            if resource_limited_e0:
                cases = [c for c in cases if c.e0_kv_per_m not in resource_limited_e0]
            if realization_ids is not None:
                selected_realizations = set(realization_ids)
                available_realizations = {case.realization_id for case in cases}
                missing_realizations = sorted(selected_realizations - available_realizations)
                if missing_realizations:
                    raise ValueError(
                        "selected realization IDs are absent from this stage/cohort: "
                        f"{missing_realizations}"
                    )
                cases = [case for case in cases if case.realization_id in selected_realizations]
            if args.e0_kv_per_m is not None:
                selected_e0 = set(args.e0_kv_per_m)
                available_e0 = {case.e0_kv_per_m for case in cases}
                missing_e0 = sorted(selected_e0 - available_e0)
                if missing_e0:
                    raise ValueError(f"selected E0 values are absent from this stage: {missing_e0}")
                cases = [case for case in cases if case.e0_kv_per_m in selected_e0]
            if args.case_id is not None:
                selected_case_ids = set(args.case_id)
                available_case_ids = {case.case_id for case in cases}
                missing_case_ids = sorted(selected_case_ids - available_case_ids)
                if missing_case_ids:
                    raise ValueError(
                        "selected case IDs are absent after cohort/E0 selection: "
                        f"{missing_case_ids}"
                    )
                cases = [case for case in cases if case.case_id in selected_case_ids]
            if not cases:
                raise ValueError(
                    "selection contains no cases after the E119 resource exclusion"
                )
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise SystemExit(str(exc)) from exc
    case_records = [
        _campaign_case_record(
            case,
            campaign_id=args.campaign_id,
            remote_root=args.remote_root,
            execution_inputs=execution_inputs,
            canonical_sbatch=args.canonical_sbatch,
        )
        for case in cases
    ]
    commands = [record["sbatch_command"] for record in case_records]
    planned_dt_rungs_ns = (
        sorted({case.dt_ns for case in cases}, reverse=True) if args.stage == "dt-matrix" else None
    )
    payload = {
        "format": CAMPAIGN_PLAN_FORMAT,
        "campaign_scope": CD_ALL_PHYSICS_10US_SCOPE,
        "observation_label": CD_ALL_PHYSICS_10US_LABEL,
        "campaign_id": args.campaign_id,
        "stage": args.stage,
        "cohort": args.cohort,
        "dt_rungs_ns": planned_dt_rungs_ns,
        "accepted_dt_ns": args.accepted_dt_ns,
        "max_in_flight": args.max_in_flight,
        "resource_limited_e0_peak_kv_per_m": sorted(resource_limited_e0),
        "observation_stop_time_s": CD_ALL_PHYSICS_10US_STOP_TIME_S,
        "all_production_physics_enabled": True,
        "canonical_simulation_path": str(CANONICAL_SBATCH),
        "submission_layout": {
            "nodes": 1,
            "mpi_ranks": CD_HPC_MPI_RANKS,
            "cpus_per_rank": CD_HPC_CPUS_PER_RANK,
            "global_omp_threads": CD_GLOBAL_OMP_THREADS,
            "scoped_transport_threads": CD_HPC_CPUS_PER_RANK,
        },
        "execution_inputs": {
            "repo_root": str(execution_inputs.repo_root),
            "warpx_exe": str(execution_inputs.warpx_exe),
            "transport_config": str(execution_inputs.transport_config),
        },
        "required_inherited_environment": [],
        "case_count": len(cases),
        "cases": case_records,
        "submitted": False,
    }
    _write_json(args.output_json, payload)
    if not args.submit:
        print(
            json.dumps(
                {
                    "status": "PLANNED_NOT_SUBMITTED",
                    "stage": args.stage,
                    "case_count": len(cases),
                    "plan": str(args.output_json),
                },
                indent=2,
            )
        )
        return 0

    for required in payload["required_inherited_environment"]:
        if not os.environ.get(required):
            raise SystemExit(f"submission requires inherited environment {required}")
    if not args.canonical_sbatch.is_file():
        raise SystemExit(f"canonical sbatch not found: {args.canonical_sbatch}")
    payload["quota_check_output"] = _run_quota_checks()
    # sbatch opens its --output file when the job starts, so the log directory
    # must exist before the first submission lands.
    logs_root = _validate_remote_root(Path(str(args.remote_root)) / "logs")
    mkdir = subprocess.run(
        ["mkdir", "-p", str(logs_root)], text=True, capture_output=True, check=False
    )
    if mkdir.returncode != 0:
        raise SystemExit(f"cannot create {logs_root}: {mkdir.stderr.strip()}")
    submissions = []
    try:
        for index, (case, command) in enumerate(zip(cases, commands, strict=True)):
            submission_command = list(command)
            dependency_job_id = None
            if index >= args.max_in_flight:
                dependency_job_id = submissions[index - args.max_in_flight]["job_id"]
                submission_command.insert(-1, f"--dependency=afterany:{dependency_job_id}")
            result = subprocess.run(submission_command, text=True, capture_output=True, check=False)
            job_id = result.stdout.strip().split(";", maxsplit=1)[0]
            if result.returncode != 0 or not re.fullmatch(r"[0-9]+", job_id):
                raise SystemExit(
                    f"sbatch failed for {case.case_id}: "
                    f"{result.stderr.strip() or result.stdout.strip()}"
                )
            submissions.append(
                {
                    "case_id": case.case_id,
                    "job_id": job_id,
                    "dependency_job_id": dependency_job_id,
                    "sbatch_command_shell": shlex.join(submission_command),
                }
            )
            payload["submissions"] = submissions
            _write_json(args.output_json, payload)
    finally:
        payload["submissions"] = submissions
        payload["submitted"] = len(submissions) == len(cases)
        _write_json(args.output_json, payload)
    print(
        json.dumps(
            {
                "status": "SUBMITTED",
                "case_count": len(cases),
                "plan": str(args.output_json),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
