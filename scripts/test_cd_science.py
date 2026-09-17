"""Focused Coleman-Dwyer analytic and all-physics campaign checks."""

from __future__ import annotations

from pathlib import Path

import pytest

from cluster.olivia.submit_rrea_cd_campaign import build_cases
from rrea_cd_report_analysis import (
    COMPARISON_EXEMPT_GATES,
    all_physics_threshold_deviation,
    build_timestep_evidence,
    combined_all_physics_deviation,
    timestep_summary,
)
from report_rrea_coleman_dwyer_production import (
    _is_exact_10us_output_time,
    classify_all_physics_campaign_status,
    signed_plane_multiplier,
    signed_plane_multipliers_by_selection,
)
from run_rrea_profiled_video_capture import require_production_geometry
from rrea_cd_physics import (
    CD_ALL_PHYSICS_10US_DT_ANCHOR_E0_KV_PER_M,
    CD_ALL_PHYSICS_10US_E0_KV_PER_M,
    CD_ALL_PHYSICS_10US_STOP_TIME_S,
    CD_INITIAL_SEED_COUNT,
    CD_PRODUCTION_TAPER_GEOMETRY,
    cd_all_physics_cohort_name,
    cd_seed_design,
)
from rrea_run_support import CAPTURE_DEFAULTS, CELL_SIZE_M, TAPER_START_DOMAIN_FRACTION
from rrea_profiled_atmosphere import ALTITUDE_AT_Z0_M, DOMAIN_HEIGHT_M, SOURCE_RADIUS_M
from rrea_profiled_atmosphere import (
    Profile,
    coleman_dwyer_profile_reference,
    find_lowest_above_threshold_region,
    load_density_ratio_profile,
    load_field_profile,
    load_threshold_profile,
    validate_density_scaled_threshold_profile,
)

ROOT = Path(__file__).resolve().parents[1]


def test_default_geometry_is_the_real_production_taper() -> None:
    radius, nr, taper_start, taper_end = CD_PRODUCTION_TAPER_GEOMETRY
    assert radius == float(CAPTURE_DEFAULTS["domain_radius_m"])
    assert nr == round(radius / CELL_SIZE_M)
    assert taper_start == pytest.approx(TAPER_START_DOMAIN_FRACTION * radius)
    assert taper_end == radius


def test_production_driver_rejects_every_other_geometry() -> None:
    radius, nr, taper_start, taper_end = CD_PRODUCTION_TAPER_GEOMETRY
    require_production_geometry(
        domain_radius_m=radius,
        n_cell_r=nr,
        taper_start_m=taper_start,
        taper_end_m=taper_end,
    )
    with pytest.raises(SystemExit, match="one geometry"):
        require_production_geometry(
            domain_radius_m=radius - CELL_SIZE_M,
            n_cell_r=nr,
            taper_start_m=taper_start,
            taper_end_m=taper_end - CELL_SIZE_M,
        )


def test_every_cd_stage_uses_only_production_geometry() -> None:
    cases = [
        *build_cases("dt-matrix", realization_ids=(0,)),
            *build_cases("e0-scan", realization_ids=(100,), accepted_dt_ns=1.25),
        *build_cases(
            "stats-escalation",
            realization_ids=(200,),
            accepted_dt_ns=1.25,
            target_e0_kv_per_m=105.0,
            seed_escalation_reason="fresh cohort after CI exceeded target",
        ),
    ]
    assert cases
    assert all(
        (
            case.domain_radius_m,
            case.n_cell_r,
            case.field_taper_r_start_m,
            case.field_taper_r_end_m,
        )
        == CD_PRODUCTION_TAPER_GEOMETRY
        for case in cases
    )
    with pytest.raises(ValueError, match="unknown campaign stage"):
        build_cases("geometry")


def test_campaign_status_uses_definitive_scan_not_timestep_engineering() -> None:
    radius, nr, taper_start, taper_end = CD_PRODUCTION_TAPER_GEOMETRY

    def result(e0: float, *, definitive: bool) -> dict:
        return {
            "e0_peak_kv_per_m": e0,
            "dt_s": 1.25e-9,
            "domain_radius_m": radius,
            "n_cell_r": nr,
            "field_taper_r_start_m": taper_start,
            "field_taper_r_end_m": taper_end,
            "realization_cohort": {"definitive": definitive},
            "seed_count": CD_INITIAL_SEED_COUNT,
            "eligibility": {"primary_transit_complete_by_10us": True},
            "statistical_recommendation": {"action": "STATISTICS_SUFFICIENT"},
        }

    results = [result(e0, definitive=True) for e0 in CD_ALL_PHYSICS_10US_E0_KV_PER_M]
    results.append(result(90.0, definitive=False))
    assert classify_all_physics_campaign_status(
        results=results,
        timestep_evidence={"status": "PASS", "accepted_dt_s": 1.25e-9},
        resource_limit_reason=None,
        incomplete_input_count=0,
    ) == "COMPLETE_ALL_PHYSICS_10US_CD_DEVIATION_CAMPAIGN"


def test_campaign_grid_stays_below_the_feedback_field() -> None:
    """The scan is strictly ascending and stays under 120 kV/m at sea level."""
    grid = CD_ALL_PHYSICS_10US_E0_KV_PER_M
    assert grid and all(a < b < 120.0 for a, b in zip(grid, grid[1:]))


def test_cohort_identity_pins_seed_weight() -> None:
    assert cd_all_physics_cohort_name(0, 100_000) == "engineering"
    assert cd_all_physics_cohort_name(209, 200_000) == "confirmatory_200k"
    with pytest.raises(ValueError):
        cd_all_physics_cohort_name(200, 100_000)


_DTS = (5e-9, 2.5e-9, 1.25e-9)
_IDS = ("0", "1", "2")
_RADIUS, _NR, _TAPER_START, _TAPER_END = CD_PRODUCTION_TAPER_GEOMETRY
_MOMENTS = {"axial_centroid_m": 2500.0, "axial_rms_m": 300.0, "radial_rms_m": 40.0}
_PLANE = {"plane_index": 0, "altitude_m": 3000.0, "mean_r_m": 30.0, "rms_r_m": 45.0}
# Everything a dt rung needs to group, pair and pass; the varying part is added
# per rung below.  Identical multipliers and moments at every dt, so the ladder
# must converge and any refusal is the gate under test, not the statistics.
_RUNG = {
    "domain_radius_m": _RADIUS, "domain_height_m": DOMAIN_HEIGHT_M, "n_cell_r": _NR,
    "n_cell_z": int(CAPTURE_DEFAULTS["n_cell_z"]),
    "stop_time_s": CD_ALL_PHYSICS_10US_STOP_TIME_S, "seed_count": CD_INITIAL_SEED_COUNT,
    "seed_design": cd_seed_design(CD_INITIAL_SEED_COUNT, None),
    "seed_escalation_reason": None, "seed_permutation": None,
    "source_radius_m": SOURCE_RADIUS_M, "cd_plane_z_m": [3000.0],
    "field_taper_r_start_m": _TAPER_START, "field_taper_r_end_m": _TAPER_END,
    "ranks": 12, "cpus_per_rank": 1, "max_grid_size": int(CAPTURE_DEFAULTS["max_grid_size"]),
    "population_policy": "adaptive_resample_v1",
    "strict_population_ceilings": {"electron": 1},
    "multipliers_by_realization": {i: 12.0 for i in _IDS},
    "final_spatial_moments_by_realization": {i: dict(_MOMENTS) for i in _IDS},
    "plane_crossing_spatial_moments_available": True,
    "plane_crossing_spatial_moments_by_realization": {i: [dict(_PLANE)] for i in _IDS},
    "output_case_ids_by_realization": {i: i for i in _IDS},
}


def _ladder(*, failing: tuple[str, ...] = (), **rung) -> tuple[str, float | None]:
    rows = [
        {**_RUNG, "e0_peak_kv_per_m": e0, "dt_s": dt,
         "eligibility": {"reason_codes": [f"FAILED_{g.upper()}" for g in failing]},
         **({"ranks": 8} if rung.get("handoff") and dt == 2.5e-9 else {}),
         **{k: v for k, v in rung.items() if k != "handoff"}}
        for e0 in CD_ALL_PHYSICS_10US_DT_ANCHOR_E0_KV_PER_M
        for dt in _DTS
    ]
    evidence = build_timestep_evidence(timestep_summary(rows), rows)
    return evidence["status"], evidence["accepted_dt_s"]


def test_dt_ladder_verdict_gates() -> None:
    """A converging ladder is accepted; the gates that must still refuse do.

    Only the two absolute-CI gates are forgiven, because a paired ratio does
    not need a precise absolute endpoint.  A rung continued at another rank
    count is a job shape, not a physics difference: the OLIVIA/laptop handoff
    must not split a ladder.
    """
    assert _ladder() == ("PASS", max(_DTS))
    assert _ladder(handoff=True) == ("PASS", max(_DTS))
    assert _ladder(failing=COMPARISON_EXEMPT_GATES) == ("PASS", max(_DTS))
    assert _ladder(n_cell_r=1000) == ("INCONCLUSIVE", None)
    assert _ladder(failing=("no_population_resampling",)) == ("INCONCLUSIVE", None)


def test_cd_threshold_classification_reports_which_threshold_fits() -> None:
    """The headline comparison: the CI against the band at BOTH thresholds.

    Representative ratios exercise both threshold labels. A high deviation is
    never a failure -- it can be real early feedback -- so the labels say
    consistent or deviates, nothing more.
    """
    def label(ratio: float, threshold: int, *, n: int = 10) -> tuple[str, str | None]:
        entry = all_physics_threshold_deviation(
            multiplier=ratio * 100.0, multiplier_ci=(ratio * 97.0, ratio * 103.0),
            reference_multiplier=100.0, threshold_kv_per_m=threshold,
            tolerance=0.10, realization_count=n)
        return entry["classification"], entry["reason"]

    assert label(0.95, 276) == ("CD_CONSISTENT_276", None)
    assert label(1.13, 284) == (
        "CD_INCONCLUSIVE_284", "CONFIDENCE_INTERVAL_CROSSES_CONSISTENCY_BOUNDARY")
    assert label(1.50, 284)[0] == "CD_DEVIATES_HIGH_284"
    assert label(0.50, 276)[0] == "CD_DEVIATES_LOW_276"
    assert label(0.95, 276, n=3) == (
        "CD_INCONCLUSIVE_276", "CONFIRMATORY_REALIZATION_COUNT_NOT_EXACT")
    assert combined_all_physics_deviation(
        ["CD_CONSISTENT_276", "CD_INCONCLUSIVE_284"]) == "CONSISTENT_EITHER"
    assert combined_all_physics_deviation(
        ["CD_CONSISTENT_276", "CD_CONSISTENT_284"]) == "CONSISTENT_BOTH"


def _profiles():
    return load_field_profile(), load_density_ratio_profile(), load_threshold_profile()


def test_no_avalanche_well_below_threshold() -> None:
    field, density, threshold = _profiles()
    assert coleman_dwyer_profile_reference(
        field=field, density_ratio=density, threshold_profile=threshold,
        e0_peak_v_per_m=30_000.0) is None


def test_threshold_crossings_are_solved_at_profile_knots() -> None:
    field = Profile(Path("field"), [9000.0, 9100.0], [1.0, 1.0], "field")
    threshold = Profile(Path("threshold"), [9000.0, 9050.0, 9100.0],
                        [500_000.0, 300_000.0, 500_000.0], "threshold")
    density = Profile(Path("density"), [9000.0, 9100.0], [1.0, 1.0],
                      "density")
    region = find_lowest_above_threshold_region(
        e0_peak_v_per_m=400_000.0, field=field, threshold=threshold,
        density_ratio=density, altitude_min_m=9000.0,
        altitude_max_m=9100.0)
    assert region is not None
    assert region.bottom_altitude_m == pytest.approx(9025.0)
    assert region.top_altitude_m == pytest.approx(9075.0)


def test_threshold_profile_values_follow_276_density_reference() -> None:
    _, density, threshold = _profiles()
    audit = validate_density_scaled_threshold_profile(density, threshold)
    assert audit["maximum_relative_error"] < 5.0e-13


def test_current_profile_reference_uses_only_276_curve() -> None:
    field, density, threshold = _profiles()
    for e0 in (90.0, 105.0, 110.0):
        result = coleman_dwyer_profile_reference(
            field=field, density_ratio=density, threshold_profile=threshold,
            e0_peak_v_per_m=e0 * 1000.0)
        assert result is not None
        assert set(result["curves"]) == {"276"}
        assert result["curves"]["276"]["predicted_multiplication"] > 1.0


def test_signed_plane_flux_cancels_recrossings() -> None:
    def row(write, plane, up, down):
        return {"write_index": str(write), "time_s": str(write + 1),
                "plane_index": str(plane), "altitude_m": str(10_000 + plane * 1000),
                "local_runaway_up_weight": str(up),
                "local_runaway_down_weight": str(down),
                "local_runaway_up_r_weighted_m": str(30 * up),
                "local_runaway_down_r_weighted_m": str(30 * down),
                "local_runaway_up_r2_weighted_m2": str(900 * up),
                "local_runaway_down_r2_weighted_m2": str(900 * down)}
    multiplier, planes = signed_plane_multiplier([
        row(0, 0, 120, 20), row(0, 1, 70, 20),
        row(1, 0, 120, 20), row(1, 1, 330, 30)])
    assert multiplier == 3.0
    assert [plane["net"] for plane in planes] == [100.0, 300.0]
    assert signed_plane_multipliers_by_selection(planes)["local_runaway"] == 3.0


def test_campaign_output_time_is_exactly_10us() -> None:
    assert _is_exact_10us_output_time(10.0e-6)
    assert not _is_exact_10us_output_time(9.9975e-6)


def _production_ambient_audit(e0_v_per_m: float) -> dict:
    from rrea_profiled_atmosphere import (
        audit_ambient_field,
        load_field_profile,
        load_threshold_profile,
    )

    radius, _, taper_start, taper_end = CD_PRODUCTION_TAPER_GEOMETRY
    return audit_ambient_field(
        field=load_field_profile(),
        threshold=load_threshold_profile(),
        e0_peak_v_per_m=e0_v_per_m,
        domain_radius_m=radius,
        taper_r_start_m=taper_start,
        taper_r_end_m=taper_end,
    )


def test_ambient_audit_production_taper_passes_with_stated_margins() -> None:
    from rrea_profiled_atmosphere import EN_TD_PER_THRESHOLD_RATIO

    audit = _production_ambient_audit(125.0e3)
    assert audit["passed"]
    assert audit["n_super_threshold_pockets"] == 0
    # Curl-free construction: max fringe |Er| = |f'|max * span/2 exactly
    # (phi_ref is the midrange, |f'|max = 1.875/width at the fade midpoint).
    expected_fringe = 1.875 / (_TAPER_END - _TAPER_START) * audit["column_potential_span_v"] / 2.0
    assert audit["max_fringe_er_v_per_m"] == pytest.approx(
        expected_fringe, rel=2e-3)
    # The worst fringe-to-threshold ratio lives above the field top (phi
    # plateaus while the threshold falls), at the domain top and mid-fade.
    loc = audit["max_fringe_er_over_threshold_location"]
    assert loc["z_m"] == pytest.approx(DOMAIN_HEIGHT_M)
    assert 5000.0 < loc["r_m"] < 7000.0
    assert 0.7 < audit["max_fringe_er_over_threshold"] < 1.0
    # E/N is the threshold ratio in other units (both anchored at 1.293):
    # the production peak sits inside the directly measured three-body band.
    assert audit["worst_en_td"] == pytest.approx(
        audit["max_e_over_threshold"] * EN_TD_PER_THRESHOLD_RATIO)
    assert 15.0 < audit["worst_en_td"] < 25.0


def test_ambient_audit_flags_only_taper_created_pockets_when_hot() -> None:
    audit = _production_ambient_audit(400.0e3)
    assert not audit["passed"]
    assert audit["n_super_threshold_pockets"] > 0
    for pocket in audit["pockets"]:
        # Every pocket is the taper's own fringe reaching threshold, in the
        # fade region (the gate never fires on the intended axial column).
        assert abs(pocket["er_v_per_m"]) >= pocket["threshold_v_per_m"]
        assert pocket["r_m"] > _TAPER_START


def test_avalanche_integral_matches_closed_form_on_a_slab() -> None:
    # Constant |E| = E0 over the region with constant density ratio r:
    # N_lambda = (E0_kV - Eth*r) * L / 7300 exactly, M = e^N.  Pins the
    # project's one external-anchor law against its closed form, for BOTH
    # threshold conventions.
    from rrea_profiled_atmosphere import (
        AboveThresholdRegion,
        Profile,
        estimate_integrated_avalanche_lengths,
    )

    ratio = 0.25
    e0_v_per_m = 125.0e3
    z_inj, z_top = 11_000.0, 13_500.0
    field = Profile(Path("synthetic"), [0.0, 20_000.0], [1.0, 1.0], "shape")
    density = Profile(Path("synthetic"), [0.0, 20_000.0], [ratio, ratio],
                      "ratio")
    region = AboveThresholdRegion(
        bottom_altitude_m=z_inj, top_altitude_m=z_top,
        injection_altitude_m=z_inj, injection_z_m=z_inj - ALTITUDE_AT_Z0_M,
        width_m=z_top - z_inj, e0_peak_v_per_m=e0_v_per_m,
        field_at_injection_v_per_m=e0_v_per_m,
        threshold_at_injection_v_per_m=276.0e3 * ratio,
        density_ratio_at_injection=ratio,
    )
    for threshold_kv in (276.0, 284.0):
        estimate = estimate_integrated_avalanche_lengths(
            field=field, density_ratio=density, region=region,
            e0_peak_v_per_m=e0_v_per_m,
            runaway_threshold_stp_kv_per_m=threshold_kv,
        )
        expected_n = (
            (e0_v_per_m / 1000.0 - threshold_kv * ratio)
            * (z_top - z_inj) / 7300.0
        )
        assert estimate.predicted_avalanche_lengths == pytest.approx(
            expected_n, rel=1e-12)
        assert estimate.predicted_multiplication_log == pytest.approx(
            expected_n, rel=1e-12)
        assert estimate.max_field_to_threshold_ratio == pytest.approx(
            (e0_v_per_m / 1000.0) / (threshold_kv * ratio), rel=1e-12)


def test_closure_table_declares_the_100_td_ceiling() -> None:
    from rrea_profiled_atmosphere import (
        EN_TD_PER_THRESHOLD_RATIO,
        closure_table_max_en_td,
    )

    table = Path(__file__).resolve().parents[1] / (
        "rrea_fluid_closure/dry_air_swarm_hybrid_v3/"
        "electron_flux_mobility_attachment_diffusion.csv")
    assert closure_table_max_en_td(table) == 100.0
    # 276 kV/m at 1.293 kg/m^3 over the Loschmidt number density: 10.27 Td.
    assert EN_TD_PER_THRESHOLD_RATIO == pytest.approx(10.272, rel=1e-3)
    # A hot E0 that would abort the engine on its first advance is caught
    # by the same numbers the preflight uses.
    hot = _production_ambient_audit(1.0e6)
    assert hot["worst_en_td"] > closure_table_max_en_td(table)
