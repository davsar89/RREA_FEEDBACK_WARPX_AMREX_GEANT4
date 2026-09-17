#!/usr/bin/env python3
"""Coleman-Dwyer campaign physics values, cohorts and derived ceilings."""

from __future__ import annotations

import math

from rrea_run_support import (
    CAPTURE_DEFAULTS,
    CELL_SIZE_M,
    TAPER_START_DOMAIN_FRACTION,
)


CD_PLANE_FLUX_V1_HEADER = (
    "format_version",
    "write_index",
    "step",
    "time_s",
    "plane_index",
    "altitude_m",
    "all_tracked_up_weight",
    "all_tracked_down_weight",
    "local_runaway_up_weight",
    "local_runaway_down_weight",
    "e_ge_1mev_up_weight",
    "e_ge_1mev_down_weight",
    "local_runaway_up_r_weighted_m",
    "local_runaway_down_r_weighted_m",
    "local_runaway_up_r2_weighted_m2",
    "local_runaway_down_r2_weighted_m2",
    "core_total_ez_v_per_m",
    "core_ambient_ez_v_per_m",
    "core_field_perturbation_fraction",
)

CD_INITIAL_SEED_COUNT = 100_000
CD_ESCALATED_SEED_COUNT = 200_000
CD_INITIAL_SEED_DESIGN = "coleman_dwyer_100k_unit_weight"
CD_ESCALATED_SEED_DESIGN = "coleman_dwyer_200k_unit_weight"
CD_PRIMARY_CI_HALF_WIDTH_FRACTION = 0.03

# Keep the single all-physics early-time field grid and realization cohorts
# here so the planner and report cannot silently drift.
CD_ALL_PHYSICS_10US_SCOPE = "early-time-all-physics-cd-comparison-10us"
CD_ALL_PHYSICS_10US_LABEL = "EARLY_TIME_ALL_PHYSICS_CD_COMPARISON_10US"
CD_ALL_PHYSICS_10US_REPORT_FORMAT = "rrea_cd_all_physics_10us_report_v1"
CD_ALL_PHYSICS_10US_STOP_TIME_S = 10.0e-6
CD_POISSON_MAX_BOTTOM_ITERATIONS = 200
CD_OLIVIA_MPI_RANKS, CD_OLIVIA_CPUS_PER_RANK, CD_GLOBAL_OMP_THREADS = 16, 2, 1
CD_ALL_PHYSICS_10US_E0_KV_PER_M = (
    82.5,
    85.0,
    87.5,
    90.0,
    95.0,
    100.0,
    105.0,
    107.5,
    110.0,
    112.5,
    115.0,
    117.5,
    119.0,
)
CD_ALL_PHYSICS_10US_DT_ANCHOR_E0_KV_PER_M = (90.0, 105.0, 119.0)
CD_ALL_PHYSICS_10US_ENDPOINT_MAX_E0_KV_PER_M = 100.0
# The single field a recorded single-node resource limit may leave uncovered.
CD_RESOURCE_LIMITED_E0_KV_PER_M = 119.0
CD_COHORT_ENGINEERING = "engineering"
CD_COHORT_CONFIRMATORY_100K = "confirmatory_100k"
CD_COHORT_CONFIRMATORY_200K = "confirmatory_200k"
CD_ALL_PHYSICS_ENGINEERING_IDS = tuple(range(3))
CD_ALL_PHYSICS_CONFIRM_100K_IDS = tuple(range(100, 110))
CD_ALL_PHYSICS_CONFIRM_200K_IDS = tuple(range(200, 210))
_PRODUCTION_RADIUS_M = float(CAPTURE_DEFAULTS["domain_radius_m"])
CD_PRODUCTION_TAPER_GEOMETRY = (
    _PRODUCTION_RADIUS_M,
    round(_PRODUCTION_RADIUS_M / CELL_SIZE_M),
    TAPER_START_DOMAIN_FRACTION * _PRODUCTION_RADIUS_M,
    _PRODUCTION_RADIUS_M,
)
CD_ALL_PHYSICS_10US_VIDEO_NR = 64
CD_ALL_PHYSICS_10US_VIDEO_NZ = 128
# C&D needs no video: one frame at t=0 and one at the stop time.
CD_ALL_PHYSICS_10US_VIDEO_FRAME_INTERVAL_S = CD_ALL_PHYSICS_10US_STOP_TIME_S


def cd_all_physics_cohort_name(realization_id: int, seed_count: int) -> str:
    """Return the fixed cohort name, rejecting unplanned IDs or seed counts."""

    if realization_id in CD_ALL_PHYSICS_ENGINEERING_IDS:
        expected_seed_count = CD_INITIAL_SEED_COUNT
        cohort = CD_COHORT_ENGINEERING
    elif realization_id in CD_ALL_PHYSICS_CONFIRM_100K_IDS:
        expected_seed_count = CD_INITIAL_SEED_COUNT
        cohort = CD_COHORT_CONFIRMATORY_100K
    elif realization_id in CD_ALL_PHYSICS_CONFIRM_200K_IDS:
        expected_seed_count = CD_ESCALATED_SEED_COUNT
        cohort = CD_COHORT_CONFIRMATORY_200K
    else:
        raise ValueError(
            f"realization id {realization_id} is outside the fixed all-physics cohorts"
        )
    if seed_count != expected_seed_count:
        raise ValueError(
            f"{cohort} realization {realization_id} requires "
            f"{expected_seed_count} unit-weight seeds"
        )
    return cohort


def cd_transport_rng_seed(realization_id: int) -> int:
    """The one campaign rule mapping a realization id to its transport seed."""

    return 7_000_001 + int(realization_id)


def cd_all_physics_result_class(e0_kv_per_m: float) -> str:
    """Classify a planned field without promoting a transient endpoint result."""

    if e0_kv_per_m not in CD_ALL_PHYSICS_10US_E0_KV_PER_M:
        raise ValueError(f"E0={e0_kv_per_m!r} is outside {CD_ALL_PHYSICS_10US_SCOPE}")
    if e0_kv_per_m <= CD_ALL_PHYSICS_10US_ENDPOINT_MAX_E0_KV_PER_M:
        return "SOURCE_CONDITIONED_ENDPOINT_10US"
    return "STEADY_STATE_ELIGIBLE_ALL_PHYSICS_10US"


def cd_diagnostic_seed_design(seed_count: int, reason: str | None) -> str:
    """Name a deliberately under-powered engine-diagnostic realization.

    Kept out of ``cd_seed_design`` on purpose: a timestep-convergence study
    needs a RATIO between rungs, which is a paired comparison where seed noise
    largely cancels, so it can run at a fraction of the full statistics.
    That makes it cheap enough to afford three rungs -- but it is not a
    scientific realization, and the design string says so in its own name so a
    reduced-statistics run can never be pooled with scientific ones by accident.
    """
    if not (isinstance(reason, str) and reason.strip()):
        raise ValueError("a diagnostic seed count requires an explicit reason")
    if not isinstance(seed_count, int) or not 0 < seed_count < CD_INITIAL_SEED_COUNT:
        raise ValueError(
            "a diagnostic seed count must be a positive integer below the "
            f"production {CD_INITIAL_SEED_COUNT}"
        )
    return f"diagnostic_reduced_{seed_count}_unit_weight"


def cd_seed_design(seed_count: int, escalation_reason: str | None) -> str:
    """Return one of the two separate unit-weight scientific cohorts."""
    if seed_count == CD_INITIAL_SEED_COUNT:
        if escalation_reason not in (None, ""):
            raise ValueError("the 100k cohort may not carry an escalation reason")
        return CD_INITIAL_SEED_DESIGN
    if seed_count == CD_ESCALATED_SEED_COUNT:
        return CD_ESCALATED_SEED_DESIGN
    raise ValueError(
        f"C&D seed count must be {CD_INITIAL_SEED_COUNT} or "
        f"{CD_ESCALATED_SEED_COUNT}")


def cd_population_ceilings(seed_count: int, multiplication_276: float) -> dict[str, int]:
    """Return the exact campaign ceilings from the conservative 276 curve."""
    if seed_count <= 0:
        raise ValueError("seed_count must be positive")
    if not math.isfinite(multiplication_276) or multiplication_276 <= 0.0:
        raise ValueError("multiplication_276 must be finite and positive")

    million = 1_000_000

    def round_up_million(value: float) -> int:
        if not math.isfinite(value) or value > (2**63 - 1):
            raise ValueError("population ceiling exceeds the supported integer range")
        return int(math.ceil(value / million) * million)

    electron = round_up_million(5.0 * seed_count * multiplication_276)
    return {
        "electron": electron,
        "photon": round_up_million(2.0 * electron),
        "positron": round_up_million(0.5 * electron),
    }
