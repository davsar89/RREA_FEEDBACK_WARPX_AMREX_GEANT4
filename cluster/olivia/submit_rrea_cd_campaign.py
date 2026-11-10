#!/usr/bin/env python3
"""Plan, or submit, the production-path RREA campaigns.

This file is orchestration only.  Every case invokes the one canonical
``slurm_rrea_profiled_video_capture.sbatch`` launcher; it never constructs a
WarpX command or implements a second simulation path.  Planning is the default
and has no cluster side effects; ``--submit`` submits it (check
``olivia.py quota`` first).
"""

from __future__ import annotations

import argparse
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
    CD_GLOBAL_OMP_THREADS, CD_OLIVIA_CPUS_PER_RANK, CD_OLIVIA_MPI_RANKS,
    CD_RESOURCE_LIMITED_E0_KV_PER_M,
    cd_seed_design,
)
from rrea_profiled_atmosphere import write_json  # noqa: E402

CANONICAL_SBATCH = ROOT / "cluster/olivia/slurm_rrea_profiled_video_capture.sbatch"
DEFAULT_REMOTE_ROOT = Path(
    os.environ.get("REMOTE_ROOT", "/cluster/projects/nn9526k/dsarria/warpx_rrea_smoke")
)
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
    case_id = (
        f"{prefix}_cd_e{_number_label(e0_kv_per_m)}_"
        f"dt{_number_label(dt_ns)}ns_"
        f"seed{seed_count}_"
        f"r{realization_id:02d}"
    )
    return CampaignCase(
        case_id=case_id,
        stage=stage,
        e0_kv_per_m=e0_kv_per_m,
        dt_ns=dt_ns,
        realization_id=realization_id,
        seed_count=seed_count,
        restart_checkpoint=restart_checkpoint,
        output_case_id=output_case_id,
        seed_escalation_reason=seed_escalation_reason,
        scratch_out=scratch_out,
        attempt_history=attempt_history,
    )


def build_cases(
    stage: str,
    *,
    accepted_dt_ns: float | None = None,
    dt_rungs_ns: tuple[float, ...] | None = None,
    target_e0_kv_per_m: float | None = None,
    seed_escalation_reason: str | None = None,
) -> list[CampaignCase]:
    """Return one explicit campaign stage; this function never submits it."""
    if stage not in STAGE_DEFAULT_COHORT:
        raise ValueError(f"unknown campaign stage {stage!r}")
    selected_cohort = STAGE_DEFAULT_COHORT[stage]
    realization_ids = COHORT_REALIZATION_IDS[selected_cohort]
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
    return cases


def build_restart_resume_cases(
    resume_document: dict,
    *,
    selected_case_ids: tuple[str, ...],
) -> list[CampaignCase]:
    records = {
        record["case_id"]: record
        for record in resume_document.get("cases", [])
        if isinstance(record, dict) and isinstance(record.get("case_id"), str)
    }
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
    """The sbatch's own environment: where it runs and where the case lives."""
    remote_root_posix = _validate_remote_root(remote_root)
    output_case_id = case.output_case_id or case.case_id
    environment = {
        "REMOTE_ROOT": str(remote_root_posix),
        "PROFILED_VIDEO_SCRATCH_OUT": (
            case.scratch_out
            if case.scratch_out is not None
            else str(
                remote_root_posix / "runs/rrea_cd_campaign_work" / campaign_id / output_case_id
            )
        ),
    }
    if execution_inputs is not None:
        environment["REPO_ROOT"] = str(execution_inputs.repo_root)
    return environment


def case_driver_args(
    case: CampaignCase, execution_inputs: CampaignExecutionInputs
) -> list[str]:
    """Driver flags for one case; the driver derives and checks the rest."""
    dt_s = case.dt_ns * 1.0e-9
    checkpoint_interval = max(1, round(CD_ALL_PHYSICS_10US_STOP_TIME_S / (2.0 * dt_s)))
    args = [
        "--warpx-exe", str(execution_inputs.warpx_exe),
        "--transport-config", str(execution_inputs.transport_config),
        "--e0-peak-kv-per-m", f"{case.e0_kv_per_m:.17g}",
        "--dt-s", f"{dt_s:.17g}",
        "--stop-time-s", f"{CD_ALL_PHYSICS_10US_STOP_TIME_S:.17g}",
        "--video-capture-final-time-s", f"{CD_ALL_PHYSICS_10US_STOP_TIME_S:.17g}",
        "--video-nr", str(CD_ALL_PHYSICS_10US_VIDEO_NR),
        "--video-nz", str(CD_ALL_PHYSICS_10US_VIDEO_NZ),
        "--seed-source-model", "coleman_dwyer_7p2mev",
        "--seed-realization-id", str(case.realization_id),
        "--cd-plane-diagnostics",
        "--checkpoint-interval", str(checkpoint_interval),
        "--omp-threads", str(CD_GLOBAL_OMP_THREADS),
        "--transport-omp-threads", str(CD_OLIVIA_CPUS_PER_RANK),
    ]
    if case.seed_count == CD_ESCALATED_SEED_COUNT:
        args.append("--cd-seed-escalation-200k")
    if case.seed_escalation_reason:
        args += ["--cd-seed-escalation-reason", case.seed_escalation_reason]
    if case.restart_checkpoint is not None:
        args += ["--restart-checkpoint", case.restart_checkpoint]
    return args


def sbatch_command(
    case: CampaignCase,
    *,
    campaign_id: str,
    remote_root: Path,
    execution_inputs: CampaignExecutionInputs,
) -> list[str]:
    environment = case_environment(
        case,
        campaign_id=campaign_id,
        remote_root=remote_root,
        execution_inputs=execution_inputs,
    )
    # An explicit export list prevents an unrelated login-shell RREA_*, OMP_*,
    # or AMReX control from silently changing a planned campaign case.
    export_text = ",".join(
        f"{key}={_validate_export_value(key, value)}" for key, value in sorted(environment.items())
    )
    job_name = f"rrea_{case.case_id}"[:128]
    return [
        "sbatch",
        "--parsable",
        "--nodes=1",
        f"--ntasks={CD_OLIVIA_MPI_RANKS}",
        f"--cpus-per-task={CD_OLIVIA_CPUS_PER_RANK}",
        "--mem=96G",
        "--time=48:00:00",
        f"--job-name={job_name}",
        f"--export={export_text}",
        str(CANONICAL_SBATCH),
        *case_driver_args(case, execution_inputs),
    ]


def _campaign_case_record(
    case: CampaignCase,
    *,
    campaign_id: str,
    remote_root: Path,
    execution_inputs: CampaignExecutionInputs,
) -> dict:
    case_fields = asdict(case)
    case_fields["attempt_history"] = list(case.attempt_history)
    command = sbatch_command(
        case,
        campaign_id=campaign_id,
        remote_root=remote_root,
        execution_inputs=execution_inputs,
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
        help="Exact absolute OLIVIA repository root (default: REMOTE_ROOT/repo_amrex_latest).",
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
            args.realization_id,
            args.e0_kv_per_m,
            args.case_id,
            args.seed_escalation_reason,
        )
    ):
        parser.error("restart-resume selects cases only through --resume-case-id")
    if args.output_json is None and args.stage == "stats-escalation":
        parser.error("stats-escalation requires an explicit unique --output-json")
    if args.output_json is None:
        args.output_json = (
            ROOT / "runs_rrea/campaign_plans" / f"{args.campaign_id}_{args.stage}.json"
        )
    if args.resume_plan is not None and args.output_json.resolve() == args.resume_plan.resolve():
        parser.error("--output-json must differ from the --resume-plan it resumes")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
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
        ):
            raise SystemExit("restart source plan is not a C&D campaign plan")
    realization_ids = tuple(args.realization_id) if args.realization_id is not None else None
    resource_limited_e0 = set(args.resource_limited_e0)
    try:
        if args.stage == "restart-resume":
            cases = build_restart_resume_cases(
                resume_document,
                selected_case_ids=tuple(args.resume_case_id),
            )
        else:
            cases = build_cases(
                args.stage,
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
        "cohort": STAGE_DEFAULT_COHORT.get(args.stage),
        "dt_rungs_ns": planned_dt_rungs_ns,
        "accepted_dt_ns": args.accepted_dt_ns,
        "max_in_flight": args.max_in_flight,
        "resource_limited_e0_peak_kv_per_m": sorted(resource_limited_e0),
        "observation_stop_time_s": CD_ALL_PHYSICS_10US_STOP_TIME_S,
        "all_production_physics_enabled": True,
        "canonical_simulation_path": str(CANONICAL_SBATCH),
        "submission_layout": {
            "nodes": 1,
            "mpi_ranks": CD_OLIVIA_MPI_RANKS,
            "cpus_per_rank": CD_OLIVIA_CPUS_PER_RANK,
            "global_omp_threads": CD_GLOBAL_OMP_THREADS,
            "scoped_transport_threads": CD_OLIVIA_CPUS_PER_RANK,
        },
        "execution_inputs": {
            "repo_root": str(execution_inputs.repo_root),
            "warpx_exe": str(execution_inputs.warpx_exe),
            "transport_config": str(execution_inputs.transport_config),
        },
        "case_count": len(cases),
        "cases": case_records,
        "submitted": False,
    }
    write_json(args.output_json, payload)
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

    submissions = []
    try:
        for index, (case, command) in enumerate(zip(cases, commands, strict=True)):
            submission_command = list(command)
            dependency_job_id = None
            if index >= args.max_in_flight:
                dependency_job_id = submissions[index - args.max_in_flight]["job_id"]
                submission_command.insert(1, f"--dependency=afterany:{dependency_job_id}")
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
            write_json(args.output_json, payload)
    finally:
        payload["submissions"] = submissions
        payload["submitted"] = len(submissions) == len(cases)
        write_json(args.output_json, payload)
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
