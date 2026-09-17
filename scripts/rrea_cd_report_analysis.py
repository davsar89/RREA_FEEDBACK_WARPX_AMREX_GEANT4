"""Pure ensemble statistics for the single production C&D report path.

There is intentionally no CLI and no file parsing here.  The production
report validates artifacts, then calls these compact grouping/statistics
helpers so physical comparison rules are defined once.
"""

from __future__ import annotations

import math
import statistics
from typing import Any

from rrea_cd_physics import (
    CD_ALL_PHYSICS_10US_DT_ANCHOR_E0_KV_PER_M,
    CD_ALL_PHYSICS_CONFIRM_100K_IDS,
    CD_ALL_PHYSICS_CONFIRM_200K_IDS,
    CD_ALL_PHYSICS_ENGINEERING_IDS,
    CD_COHORT_CONFIRMATORY_100K,
    CD_COHORT_CONFIRMATORY_200K,
    CD_COHORT_ENGINEERING,
    CD_ESCALATED_SEED_COUNT,
    CD_INITIAL_SEED_COUNT,
    CD_PRODUCTION_TAPER_GEOMETRY,
    cd_seed_design,
)


T95 = {
    2: 12.706,
    3: 4.303,
    4: 3.182,
    5: 2.776,
    6: 2.571,
    7: 2.447,
    8: 2.365,
    9: 2.306,
    10: 2.262,
}

(
    PRODUCTION_RADIUS_M,
    PRODUCTION_N_CELL_R,
    PRODUCTION_TAPER_START_M,
    PRODUCTION_TAPER_END_M,
) = CD_PRODUCTION_TAPER_GEOMETRY


def _uses_production_geometry(result: dict) -> bool:
    return (
        result.get("domain_radius_m"),
        result.get("n_cell_r"),
        result.get("field_taper_r_start_m"),
        result.get("field_taper_r_end_m"),
    ) == CD_PRODUCTION_TAPER_GEOMETRY

# Every cross-ensemble grouping begins with these direct physical fields;
# a comparison explicitly omits only the coordinate it varies.  Job shape --
# rank count, threads per rank, box size -- is deliberately absent: it changes
# which rank owns a macroparticle and therefore the transport RNG stream, but
# not the physics, and a dt rung continued on the other machine of the
# OLIVIA/laptop handoff must still pair with the rungs it belongs to.  The
# paired Student-t CI measures whatever spread the decomposition adds.
COMPARISON_FIELDS = (
    "e0_peak_kv_per_m",
    "dt_s",
    "domain_radius_m",
    "domain_height_m",
    "n_cell_r",
    "n_cell_z",
    "stop_time_s",
    "seed_count",
    "seed_design",
    "seed_escalation_reason",
    "seed_permutation",
    "source_radius_m",
    "cd_plane_z_m",
    "field_taper_r_start_m",
    "field_taper_r_end_m",
    "population_policy",
    "strict_population_ceilings",
)

# summarize_group owns the local eligibility gate list and emits both its
# verdict and a FAILED_<GATE> code per failure; this module reads those codes
# rather than keeping a second copy of the list that could fall behind it.
#
# A timestep check is a paired realization-level ratio.  Its engineering cohort
# is intentionally only three runs, so an imprecise *absolute* endpoint
# estimate must not veto an otherwise precise paired comparison; the absolute
# 3% gates remain mandatory for the definitive per-E0 report.  Those two are
# the only failures a cross-ensemble comparison forgives.
COMPARISON_EXEMPT_GATES = (
    "multiplication_ci_half_width_at_most_3pct",
    "all_primary_ci_half_widths_at_most_3pct",
)
_COMPARISON_EXEMPT_REASONS = frozenset(
    f"FAILED_{gate.upper()}" for gate in COMPARISON_EXEMPT_GATES
)

def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return tuple(sorted((key, _freeze(item)) for key, item in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _comparison_key(result: dict, *excluded: str) -> tuple[tuple[str, Any], ...]:
    omit = set(excluded)
    return tuple(
        (field, _freeze(result.get(field))) for field in COMPARISON_FIELDS if field not in omit
    )


def _student_t_mean_ci(values: list[float]) -> tuple[float, tuple[float, float]]:
    """Mean and realization-level 95% CI; one value is inconclusive."""
    if not values:
        return math.nan, (math.nan, math.nan)
    mean = statistics.fmean(values)
    if len(values) < 2:
        return mean, (math.nan, math.nan)
    half = T95.get(len(values), 1.96) * statistics.stdev(values) / math.sqrt(len(values))
    return mean, (mean - half, mean + half)


def _paired_ratio(
    numerator: dict[str, float], denominator: dict[str, float]
) -> tuple[list[str], float, tuple[float, float]]:
    numerator_ids = set(numerator)
    denominator_ids = set(denominator)
    if numerator_ids != denominator_ids:
        return [], math.nan, (math.nan, math.nan)
    common = sorted(numerator_ids, key=int)
    logs = [math.log(numerator[key] / denominator[key]) for key in common]
    mean_log, log_ci = _student_t_mean_ci(logs)
    ratio = math.exp(mean_log) if math.isfinite(mean_log) else math.nan
    ratio_ci = tuple(math.exp(value) if math.isfinite(value) else math.nan for value in log_ci)
    return common, ratio, ratio_ci


def _pairing_audit(first: dict[str, float], second: dict[str, float]) -> dict:
    first_ids = set(first)
    second_ids = set(second)
    matched = sorted(first_ids & second_ids, key=int)
    return {
        "exact_realization_id_match": first_ids == second_ids,
        "matched_realizations": matched,
        "first_only_realizations": sorted(first_ids - second_ids, key=int),
        "second_only_realizations": sorted(second_ids - first_ids, key=int),
        "minimum_three_realizations": len(matched) >= 3,
    }


def sources_eligible(results: list[dict]) -> bool:
    """Every compared source ensemble passes every required transfer gate."""

    def transferable(result: dict) -> bool:
        reasons = (result.get("eligibility") or {}).get("reason_codes")
        return isinstance(reasons, list) and not set(reasons) - _COMPARISON_EXEMPT_REASONS

    return bool(results) and all(transferable(result) for result in results)


def _comparisons_eligible(comparisons: list[dict]) -> bool:
    return bool(comparisons) and all(
        isinstance(comparison, dict)
        and comparison.get("all_source_ensembles_locally_eligible") is True
        for comparison in comparisons
    )


def all_physics_threshold_deviation(
    *,
    multiplier: float,
    multiplier_ci: tuple[float, float],
    reference_multiplier: float,
    threshold_kv_per_m: int,
    tolerance: float,
    realization_count: int,
    required_realizations: int = 10,
) -> dict[str, Any]:
    """Describe, rather than pass/fail, one all-physics C&D comparison.

    The C&D expression is a reference for avalanche multiplication.  With all
    schema-6 physics active, a high-field excess can be a real early feedback
    contribution.  The labels therefore say *consistent* or *deviates* and
    never imply that a high deviation is a software failure.
    """

    if threshold_kv_per_m not in (276, 284):
        raise ValueError("C&D threshold must be 276 or 284 kV/m")
    if (
        not math.isfinite(reference_multiplier)
        or reference_multiplier <= 0.0
        or not 0.0 < tolerance < 1.0
    ):
        raise ValueError("invalid C&D reference multiplier or tolerance")

    ratio = multiplier / reference_multiplier if math.isfinite(multiplier) else math.nan
    ratio_ci = tuple(
        value / reference_multiplier if math.isfinite(value) else math.nan
        for value in multiplier_ci
    )
    suffix = str(threshold_kv_per_m)
    if realization_count != required_realizations or not all(
        math.isfinite(value) for value in ratio_ci
    ):
        label = f"CD_INCONCLUSIVE_{suffix}"
        reason = (
            "CONFIRMATORY_REALIZATION_COUNT_NOT_EXACT"
            if realization_count != required_realizations
            else "NONFINITE_CONFIDENCE_INTERVAL"
        )
    else:
        low, high = ratio_ci
        band_low, band_high = 1.0 - tolerance, 1.0 + tolerance
        if low >= band_low and high <= band_high:
            label = f"CD_CONSISTENT_{suffix}"
            reason = None
        elif low > band_high:
            label = f"CD_DEVIATES_HIGH_{suffix}"
            reason = None
        elif high < band_low:
            label = f"CD_DEVIATES_LOW_{suffix}"
            reason = None
        else:
            label = f"CD_INCONCLUSIVE_{suffix}"
            reason = "CONFIDENCE_INTERVAL_CROSSES_CONSISTENCY_BOUNDARY"
    return {
        "reference_multiplier": reference_multiplier,
        "ratio_point_estimate": ratio,
        "ratio_student_t_95pct_ci": list(ratio_ci),
        "percent_deviation": 100.0 * (ratio - 1.0) if math.isfinite(ratio) else math.nan,
        "percent_deviation_student_t_95pct_ci": [
            100.0 * (value - 1.0) if math.isfinite(value) else math.nan for value in ratio_ci
        ],
        "classification": label,
        "reason": reason,
    }


def combined_all_physics_deviation(classifications: list[str]) -> str:
    """Combine the explicitly visible 276/284 descriptive classifications."""

    if len(classifications) != 2:
        raise ValueError("both 276 and 284 C&D classifications are required")
    consistent = sum(value.startswith("CD_CONSISTENT_") for value in classifications)
    deviates = sum(value.startswith("CD_DEVIATES_") for value in classifications)
    if consistent == 2:
        return "CONSISTENT_BOTH"
    if consistent == 1:
        return "CONSISTENT_EITHER"
    if deviates == 2:
        return "DEVIATES_BOTH"
    return "INCONCLUSIVE"


def classify_all_physics_realization_cohort(
    realization_ids: list[int],
    seed_count: int,
) -> dict[str, Any]:
    """Validate the fixed engineering or confirmatory realization cohort."""

    ids = sorted(realization_ids)
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate realization id")
    allowed = {
        CD_COHORT_ENGINEERING: (
            list(CD_ALL_PHYSICS_ENGINEERING_IDS),
            CD_INITIAL_SEED_COUNT,
            False,
        ),
        CD_COHORT_CONFIRMATORY_100K: (
            list(CD_ALL_PHYSICS_CONFIRM_100K_IDS),
            CD_INITIAL_SEED_COUNT,
            True,
        ),
        CD_COHORT_CONFIRMATORY_200K: (
            list(CD_ALL_PHYSICS_CONFIRM_200K_IDS),
            CD_ESCALATED_SEED_COUNT,
            True,
        ),
    }
    candidates = [
        (name, expected_ids, expected_seed_count, definitive)
        for name, (expected_ids, expected_seed_count, definitive) in allowed.items()
        if set(ids).issubset(expected_ids) and seed_count == expected_seed_count
    ]
    if len(candidates) != 1:
        return {
            "name": "invalid",
            "complete": False,
            "definitive": False,
            "realization_ids": ids,
            "reason": "REALIZATIONS_DO_NOT_MATCH_ONE_FIXED_COHORT",
        }
    name, expected_ids, _, definitive = candidates[0]
    complete = ids == expected_ids
    return {
        "name": name,
        "complete": complete,
        "definitive": definitive and complete,
        "realization_ids": ids,
        "expected_realization_ids": expected_ids,
        "reason": None if complete else "FIXED_COHORT_INCOMPLETE",
    }


def timestep_summary(results: list[dict]) -> list[dict]:
    points: dict[tuple, dict[float, dict]] = {}
    for result in results:
        if "multipliers_by_realization" not in result:
            continue
        key = _comparison_key(result, "dt_s")
        by_dt = points.setdefault(key, {})
        if result["dt_s"] in by_dt:
            raise SystemExit("duplicate timestep ensemble for one physical comparison fields")
        by_dt[result["dt_s"]] = result
    summaries = []
    for key, by_dt in points.items():
        ordered = sorted(by_dt, reverse=True)
        if len(ordered) < 2:
            continue
        comparisons = []
        for coarse_dt, fine_dt in zip(ordered, ordered[1:], strict=False):
            pairing = _pairing_audit(
                by_dt[coarse_dt]["multipliers_by_realization"],
                by_dt[fine_dt]["multipliers_by_realization"],
            )
            common, ratio, ratio_ci = _paired_ratio(
                by_dt[coarse_dt]["multipliers_by_realization"],
                by_dt[fine_dt]["multipliers_by_realization"],
            )
            sources = {"coarse": by_dt[coarse_dt], "fine": by_dt[fine_dt]}
            eligible = sources_eligible(list(sources.values()))
            enough = pairing["exact_realization_id_match"] and pairing["minimum_three_realizations"]
            within_3 = eligible and enough and ratio_ci[0] >= 0.97 and ratio_ci[1] <= 1.03
            spatial_comparisons = {}
            coarse_spatial = by_dt[coarse_dt].get("final_spatial_moments_by_realization", {})
            fine_spatial = by_dt[fine_dt].get("final_spatial_moments_by_realization", {})
            for metric in ("axial_centroid_m", "axial_rms_m", "radial_rms_m"):
                spatial_common = sorted(set(coarse_spatial) & set(fine_spatial), key=int)
                spatial_logs = []
                all_values_valid = True
                for item in spatial_common:
                    try:
                        coarse_value = float(coarse_spatial[item][metric])
                        fine_value = float(fine_spatial[item][metric])
                    except (KeyError, TypeError, ValueError):
                        all_values_valid = False
                        continue
                    if (
                        not math.isfinite(coarse_value)
                        or not math.isfinite(fine_value)
                        or coarse_value == 0.0
                        or fine_value == 0.0
                    ):
                        all_values_valid = False
                        continue
                    spatial_logs.append(math.log(abs(coarse_value) / abs(fine_value)))
                _, spatial_log_ci = _student_t_mean_ci(spatial_logs)
                spatial_ci = tuple(
                    math.exp(value) if math.isfinite(value) else math.nan
                    for value in spatial_log_ci
                )
                all_matched_realizations_valid = (
                    all_values_valid
                    and set(coarse_spatial) == set(fine_spatial) == set(common)
                    and len(spatial_logs) == len(common)
                    and len(common) >= 3
                )
                spatial_comparisons[metric] = {
                    "matched_realizations": spatial_common,
                    "all_matched_realizations_valid": (all_matched_realizations_valid),
                    "student_t_95pct_coarse_to_fine_ratio_ci": list(spatial_ci),
                    "within_3pct": (
                        all_matched_realizations_valid
                        and spatial_ci[0] >= 0.97
                        and spatial_ci[1] <= 1.03
                    ),
                }
            final_spatial_available = all(
                item["all_matched_realizations_valid"] for item in spatial_comparisons.values()
            )
            plane_spatial_available = all(
                by_dt[dt].get("plane_crossing_spatial_moments_available", False)
                for dt in (coarse_dt, fine_dt)
            )
            coarse_planes = by_dt[coarse_dt].get(
                "plane_crossing_spatial_moments_by_realization", {}
            )
            fine_planes = by_dt[fine_dt].get("plane_crossing_spatial_moments_by_realization", {})
            plane_realizations = sorted(set(coarse_planes) & set(fine_planes), key=int)
            expected_plane_count = len(by_dt[coarse_dt].get("cd_plane_z_m") or [])
            plane_spatial_available = (
                plane_spatial_available
                and set(coarse_planes) == set(fine_planes) == set(common)
                and len(plane_realizations) >= 3
                and expected_plane_count > 0
                and all(
                    len(coarse_planes[item]) == expected_plane_count
                    and len(fine_planes[item]) == expected_plane_count
                    for item in plane_realizations
                )
            )
            plane_comparisons = {}
            if plane_spatial_available:
                for plane_index in range(expected_plane_count):
                    coarse_altitudes = {
                        float(coarse_planes[item][plane_index]["altitude_m"])
                        for item in plane_realizations
                    }
                    fine_altitudes = {
                        float(fine_planes[item][plane_index]["altitude_m"])
                        for item in plane_realizations
                    }
                    if (
                        len(coarse_altitudes) != 1
                        or coarse_altitudes != fine_altitudes
                        or any(
                            int(coarse_planes[item][plane_index]["plane_index"]) != plane_index
                            or int(fine_planes[item][plane_index]["plane_index"]) != plane_index
                            for item in plane_realizations
                        )
                    ):
                        plane_spatial_available = False
                        break
                    metric_results = {}
                    for metric in ("mean_r_m", "rms_r_m"):
                        logs = []
                        all_values_valid = True
                        for item in plane_realizations:
                            try:
                                coarse_value = float(coarse_planes[item][plane_index][metric])
                                fine_value = float(fine_planes[item][plane_index][metric])
                            except (KeyError, TypeError, ValueError):
                                all_values_valid = False
                                continue
                            if (
                                not math.isfinite(coarse_value)
                                or not math.isfinite(fine_value)
                                or coarse_value <= 0.0
                                or fine_value <= 0.0
                            ):
                                all_values_valid = False
                                continue
                            logs.append(math.log(coarse_value / fine_value))
                        _, log_ci = _student_t_mean_ci(logs)
                        metric_ci = tuple(
                            math.exp(value) if math.isfinite(value) else math.nan
                            for value in log_ci
                        )
                        all_matched_realizations_valid = (
                            all_values_valid
                            and len(logs) == len(common)
                            and len(plane_realizations) == len(common)
                            and len(common) >= 3
                        )
                        metric_results[metric] = {
                            "student_t_95pct_coarse_to_fine_ratio_ci": list(metric_ci),
                            "all_matched_realizations_valid": (all_matched_realizations_valid),
                            "within_3pct": (
                                all_matched_realizations_valid
                                and metric_ci[0] >= 0.97
                                and metric_ci[1] <= 1.03
                            ),
                        }
                        if not all_matched_realizations_valid:
                            plane_spatial_available = False
                    plane_comparisons[str(plane_index)] = {
                        "altitude_m": next(iter(coarse_altitudes)),
                        "matched_realizations": plane_realizations,
                        **metric_results,
                    }
            plane_spatial_passed = (
                plane_spatial_available
                and bool(plane_comparisons)
                and all(
                    metric["within_3pct"]
                    for plane in plane_comparisons.values()
                    for name, metric in plane.items()
                    if name in {"mean_r_m", "rms_r_m"}
                )
            )
            spatial_available = final_spatial_available and plane_spatial_available
            spatial_passed = (
                spatial_available
                and plane_spatial_passed
                and all(item["within_3pct"] for item in spatial_comparisons.values())
            )
            if not eligible:
                status = "INCONCLUSIVE_SOURCE_ENSEMBLE_INELIGIBLE"
            elif not pairing["exact_realization_id_match"]:
                status = "INCONCLUSIVE_REALIZATION_ID_MISMATCH"
            elif not enough:
                status = "INCONCLUSIVE_STATISTICS"
            elif not spatial_available:
                status = "INCONCLUSIVE_MISSING_SPATIAL_MOMENTS"
            elif within_3 and spatial_passed:
                status = "PASS_3PCT"
            else:
                status = "FAIL"
            comparisons.append(
                {
                    "coarse_dt_s": coarse_dt,
                    "fine_dt_s": fine_dt,
                    "matched_realizations": common,
                    "realization_pairing": pairing,
                    "all_source_ensembles_locally_eligible": eligible,
                    "source_output_case_ids": {
                        label: source.get("output_case_ids_by_realization", {})
                        for label, source in sources.items()
                    },
                    "coarse_to_fine_multiplier_ratio": ratio,
                    "student_t_95pct_ratio_ci": list(ratio_ci),
                    "within_3pct": within_3,
                    "final_frame_spatial_moments": spatial_comparisons,
                    "plane_crossing_spatial_moments": plane_comparisons,
                    "plane_crossing_spatial_moments_available": plane_spatial_available,
                    "required_spatial_moments_within_3pct": spatial_passed,
                    "status": status,
                }
            )
        # A direct coarse/fine comparison is not enough to accept the coarse
        # rung: the nominally finer result must itself agree with a still finer
        # result.  This produces a three-rung support chain and intentionally
        # never accepts the finest available rung.
        accepted = [
            comparisons[index]["coarse_dt_s"]
            for index in range(max(0, len(comparisons) - 1))
            if comparisons[index]["status"] == "PASS_3PCT"
            and comparisons[index + 1]["status"] == "PASS_3PCT"
        ]
        metadata = dict(key)
        all_sources_eligible = _comparisons_eligible(comparisons)
        summaries.append(
            {
                # The four fields _uses_production_geometry reads, lifted out
                # of the grouping key so build_timestep_evidence can see them.
                # Without n_cell_r and the taper the summary answered None and
                # every required E0 point was reported missing.
                "e0_peak_kv_per_m": metadata["e0_peak_kv_per_m"],
                "domain_radius_m": metadata["domain_radius_m"],
                "n_cell_r": metadata["n_cell_r"],
                "field_taper_r_start_m": metadata["field_taper_r_start_m"],
                "field_taper_r_end_m": metadata["field_taper_r_end_m"],
                "comparison_fields": key,
                "comparisons": comparisons,
                "accepted_production_dt_s": max(accepted) if accepted else None,
                "accepted_production_dt_candidates_s": sorted(accepted, reverse=True),
                "finest_rung_converged": False,
                "finest_rung_status": "INCONCLUSIVE_NEEDS_FINER_REFERENCE",
                "minimum_realizations_per_comparison": 3,
                "requires_three_rung_support_chain": True,
                "all_source_ensembles_locally_eligible": all_sources_eligible,
                "status": (
                    "PASS_HAS_ACCEPTED_PRODUCTION_DT"
                    if accepted and all_sources_eligible
                    else "INCONCLUSIVE_NEEDS_THIRD_RUNG"
                    if len(ordered) < 3
                    else comparisons[-1]["status"]
                ),
            }
        )
    return summaries


REQUIRED_DT_POINTS = CD_ALL_PHYSICS_10US_DT_ANCHOR_E0_KV_PER_M



def build_timestep_evidence(
    timestep_summaries: list[dict],
    results: list[dict],
    *,
    resource_limited_e0: set[float] | None = None,
) -> dict:
    """Summarize the configured all-physics timestep-convergence ladder."""
    resource_limited_e0 = set(resource_limited_e0 or ())
    entries = []
    missing = []
    accepted_candidate_sets: list[set[float]] = []
    source_comparisons: list[dict] = []
    selected_seed_designs: dict[float, tuple[int, str, str | None]] = {}
    for e0 in REQUIRED_DT_POINTS:
        matches = [
            item
            for item in timestep_summaries
            if item.get("e0_peak_kv_per_m") == e0
            and _uses_production_geometry(item)
        ]
        if not matches:
            if e0 in resource_limited_e0:
                entries.append(
                    {
                        "e0_peak_kv_per_m": e0,
                        "status": "INCONCLUSIVE_RESOURCE_LIMIT",
                        "accepted_dt_s": None,
                        "seed_count": None,
                        "seed_design": None,
                        "seed_escalation_reason": None,
                        "source_output_case_ids": [],
                        "all_source_ensembles_locally_eligible": False,
                        "comparison_fields": None,
                    }
                )
                continue
            missing.append(f"E{e0:g}")
            continue
        candidates: list[tuple[dict, list[dict], str]] = []
        for item in matches:
            accepted = item.get("accepted_production_dt_s")
            metadata = dict(item.get("comparison_fields", ()))
            design = metadata.get("seed_design")
            comparisons = item.get("comparisons")
            if (
                item.get("status") == "PASS_HAS_ACCEPTED_PRODUCTION_DT"
                and accepted is not None
                and metadata.get("seed_count") == CD_INITIAL_SEED_COUNT
                and design == cd_seed_design(CD_INITIAL_SEED_COUNT, None)
                and metadata.get("seed_escalation_reason") in (None, "")
                and item.get("all_source_ensembles_locally_eligible") is True
                and _comparisons_eligible(comparisons if isinstance(comparisons, list) else [])
                and isinstance(design, str)
            ):
                accepted_candidates = item.get("accepted_production_dt_candidates_s")
                if (
                    isinstance(accepted_candidates, list)
                    and accepted_candidates
                    and all(
                        isinstance(value, (int, float))
                        and math.isfinite(float(value))
                        and float(value) > 0.0
                        for value in accepted_candidates
                    )
                ):
                    candidates.append((item, comparisons, design))
        if not candidates:
            if e0 in resource_limited_e0:
                contradictory = [
                    item
                    for item in matches
                    if item.get("all_source_ensembles_locally_eligible") is not True
                    or any(
                        comparison.get("status")
                        not in {
                            "PASS_3PCT",
                            "INCONCLUSIVE_NEEDS_FINER_REFERENCE",
                        }
                        for comparison in item.get("comparisons", [])
                        if isinstance(comparison, dict)
                    )
                ]
                if contradictory:
                    missing.append(
                        "E119_all_physics_resource_limit_contradicts_"
                        "completed_failed_timestep_evidence"
                    )
                    continue
                entries.append(
                    {
                        "e0_peak_kv_per_m": e0,
                        "status": "INCONCLUSIVE_RESOURCE_LIMIT",
                        "accepted_dt_s": None,
                        "seed_count": None,
                        "seed_design": None,
                        "seed_escalation_reason": None,
                        "source_output_case_ids": [],
                        "all_source_ensembles_locally_eligible": False,
                        "comparison_fields": None,
                    }
                )
                continue
            source_comparisons.extend(
                comparison
                for item in matches
                for comparison in (
                    item.get("comparisons") if isinstance(item.get("comparisons"), list) else []
                )
                if isinstance(comparison, dict)
            )
            missing.append(f"E{e0:g}_not_converged_or_source_ineligible")
            continue
        if len(candidates) != 1:
            missing.append(f"E{e0:g}_ambiguous_engineering_cohort")
            continue
        item, comparisons, design = candidates[0]
        seed_count = CD_INITIAL_SEED_COUNT
        escalation_reason = None
        accepted_candidates = {
            float(value) for value in item["accepted_production_dt_candidates_s"]
        }
        accepted_candidate_sets.append(accepted_candidates)
        selected_seed_designs[e0] = (
            seed_count,
            design,
            escalation_reason,
        )
        source_comparisons.extend(comparisons)
        source_output_case_ids = sorted(
            {
                output_id
                for comparison in item.get("comparisons", [])
                for mapping in comparison.get("source_output_case_ids", {}).values()
                for output_id in mapping.values()
                if isinstance(output_id, str)
            }
        )
        entries.append(
            {
                "e0_peak_kv_per_m": e0,
                "status": "PASS",
                "accepted_dt_s": None,
                "accepted_dt_candidates_s": sorted(accepted_candidates, reverse=True),
                "seed_count": seed_count,
                "seed_design": design,
                "seed_escalation_reason": escalation_reason,
                "source_output_case_ids": source_output_case_ids,
                "all_source_ensembles_locally_eligible": True,
                "comparison_fields": item.get("comparison_fields"),
            }
        )
    all_sources_eligible = _comparisons_eligible(source_comparisons)
    common_accepted = (
        set.intersection(*accepted_candidate_sets) if accepted_candidate_sets else set()
    )
    common_accepted_dt = max(common_accepted) if common_accepted else None
    for entry in entries:
        if entry.get("status") == "PASS":
            entry["accepted_dt_s"] = common_accepted_dt
    passed_entries = [entry for entry in entries if entry.get("status") == "PASS"]
    partial = bool(resource_limited_e0) and any(
        entry.get("status") == "INCONCLUSIVE_RESOURCE_LIMIT" for entry in entries
    )
    common_fields_passed = (
        not missing
        and len(entries) == len(REQUIRED_DT_POINTS)
        and common_accepted_dt is not None
        and all_sources_eligible
        and {entry["e0_peak_kv_per_m"] for entry in passed_entries}
        >= set(REQUIRED_DT_POINTS) - resource_limited_e0
    )
    status = (
        "PASS_PARTIAL_RESOURCE_LIMIT"
        if common_fields_passed and partial
        else "PASS"
        if common_fields_passed
        else "INCONCLUSIVE"
    )
    accepted_dt = common_accepted_dt if status in {"PASS", "PASS_PARTIAL_RESOURCE_LIMIT"} else None
    reused = [
        {
            "e0_peak_kv_per_m": result.get("e0_peak_kv_per_m"),
            "output_case_ids_by_realization": result.get("output_case_ids_by_realization"),
        }
        for result in results
        if result.get("e0_peak_kv_per_m") in REQUIRED_DT_POINTS
        and result.get("dt_s") == accepted_dt
        and _uses_production_geometry(result)
        and (
            result.get("seed_count"),
            result.get("seed_design"),
            result.get("seed_escalation_reason"),
        )
        == selected_seed_designs.get(float(result.get("e0_peak_kv_per_m")))
    ]
    return {
        "format": "rrea_cd_timestep_evidence_v1",
        "status": status,
        "accepted_dt_s": accepted_dt,
        "required_points": [{"e0_peak_kv_per_m": e0} for e0 in REQUIRED_DT_POINTS],
        "entries": entries,
        "covered_e0_peak_kv_per_m": [entry["e0_peak_kv_per_m"] for entry in passed_entries],
        "resource_limited_e0_peak_kv_per_m": sorted(resource_limited_e0),
        "reused_accepted_dt_ensembles": reused,
        "all_source_ensembles_locally_eligible": all_sources_eligible,
        "reason_codes": (
            sorted(missing)
            + (["E119_ALL_PHYSICS_RESOURCE_LIMIT"] if partial else [])
            + (
                ["NO_COMMON_SUPPORTED_TIMESTEP_ACROSS_REQUIRED_POINTS"]
                if common_accepted_dt is None
                else []
            )
            + (
                ["SOURCE_ENSEMBLE_LOCAL_ELIGIBILITY_FAILED"]
                if not all_sources_eligible
                else []
            )
        ),
    }
