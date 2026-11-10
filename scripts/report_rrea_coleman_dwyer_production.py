#!/usr/bin/env python3
"""Report 10 us all-physics signed-flux multiplication vs Coleman-Dwyer.

Post-processing ONLY (launches nothing): reads capture dirs produced by the
production driver `run_rrea_profiled_video_capture.py`. A quantitative verdict
requires `rrea_cd_plane_flux.csv`: population peaks and trough-to-peak ratios
are transient diagnostics, not C&D multiplication observables.

Physics reminders (interpretation, not gating):
Both denominator curves use the same interval selected by the
276-kV/m-density-scaled onset. Neither curve is silently selected as truth.
Photon and positron physics remain enabled. A high-field excess is therefore
reported as an all-physics deviation, not diagnosed as a transport failure.

Usage (typically on OLIVIA):
  python3 scripts/report_rrea_coleman_dwyer_production.py RUN_DIR [RUN_DIR ...]
      [--tolerance 0.10] [--output-json REPORT.json]
Each RUN_DIR is a capture dir (contains rrea_reduced.csv + command.json) or
its parent. """

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rrea_profiled_atmosphere import (  # noqa: E402
    AboveThresholdRegion,
    ALTITUDE_AT_Z0_M,
    coleman_dwyer_profile_reference,
    estimate_integrated_avalanche_lengths,
    load_density_ratio_profile,
    load_field_profile,
    load_threshold_profile,
    profiled_abs_field_v_per_m,
)
from rrea_cd_physics import (  # noqa: E402
    CD_ALL_PHYSICS_10US_E0_KV_PER_M,
    CD_ALL_PHYSICS_10US_LABEL,
    CD_ALL_PHYSICS_10US_REPORT_FORMAT,
    CD_ALL_PHYSICS_10US_SCOPE,
    CD_ALL_PHYSICS_10US_STOP_TIME_S,
    CD_COHORT_DIAGNOSTIC,
    CD_INITIAL_SEED_COUNT,
    CD_PRIMARY_CI_HALF_WIDTH_FRACTION,
    CD_RESOURCE_LIMITED_E0_KV_PER_M,
    cd_all_physics_cohort_name,
    cd_all_physics_result_class,
)
from rrea_run_support import required_time_steps  # noqa: E402
from rrea_cd_report_analysis import (  # noqa: E402
    _student_t_mean_ci,
    all_physics_threshold_deviation,
    build_timestep_evidence,
    combined_all_physics_deviation,
    classify_all_physics_realization_cohort,
    timestep_summary,
)

RESAMPLE_ACTIVITY_COLUMNS = (
    "resample_killed_count",
    "resample_split_count",
    "resample_thin_rounds",
    "resample_killed_weight",
    "resample_boost_weight",
    "resample_split_weight",
    "resample_killed_energy_eV",
    "resample_boost_energy_eV",
)


# A production run reaches the requested stop by repeated floating-point
# additions of dt.  At the 10 us target this can accumulate roughly 1e-18 s of
# roundoff even though the final step is exactly the configured terminal step.
# Keep the allowance orders of magnitude below every practical campaign dt so
# an additional or missing transport step is never accepted as serialization
# noise.
CD_ALL_PHYSICS_10US_OUTPUT_TIME_TOLERANCE_S = (
    1.0e-12 * CD_ALL_PHYSICS_10US_STOP_TIME_S
)

AGGREGATE_PROCESS_CONTEXT_COLUMNS = {
    "photon": (
        "photon_interaction_count",
        "compton_interaction_count",
        "photoelectric_interaction_count",
    ),
    "pair": (
        "pair_production_count",
        "pair_nuclear_count",
        "pair_triplet_count",
    ),
    "positron": (
        "positron_bhabha_event_count",
        "positron_brems_event_count",
        "positron_annihilation_count",
        "positron_annihilation_at_rest_count",
        "positron_annihilation_in_flight_count",
    ),
    "charged_secondary": (
        "hard_moller_event_count",
        "electron_brems_photon_count",
        "photon_seeded_kinetic_electron_weight",
        "photon_seeded_low_electron_weight",
        "photon_seeded_kinetic_positron_weight",
        "photon_seeded_below_cutoff_positron_weight",
    ),
}

LOW_ENERGY_BREMSSTRAHLUNG_WARNING = (
    "The schema-6 transport model represents bremsstrahlung photons below "
    "the configuration's tracked-photon cut as continuous drag, substantially "
    "undercounting the spectrum below that cut; metrics well above it, much less."
)


def json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def _capture_dir(path: Path) -> Path:
    return path if (path / "rrea_reduced.csv").exists() else path / "capture"


def _csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    return list(reader.fieldnames or []), rows


def _command_value(command: list[str], key: str) -> str | None:
    prefix = key + "="
    values = [token[len(prefix) :] for token in command if token.startswith(prefix)]
    if len(values) > 1:
        raise SystemExit(f"command contains duplicate {key} overrides")
    return values[0] if values else None


def _nonzero_columns(rows: list[dict[str, str]], columns: tuple[str, ...]) -> dict[str, float]:
    result = {}
    available = set(rows[0]) if rows else set()
    for column in columns:
        if column not in available:
            continue
        maximum = max(abs(float(row[column] or 0.0)) for row in rows)
        if maximum != 0.0:
            result[column] = maximum
    return result


def _flux_at_time(
    series: list[tuple[float, float]],
    target_s: float,
) -> float:
    if not series:
        return math.nan
    time_s, flux = min(series, key=lambda item: abs(item[0] - target_s))
    tolerance = max(1.0e-12, 64.0 * math.ulp(target_s))
    return flux if abs(time_s - target_s) <= tolerance else math.nan


def _is_exact_10us_output_time(value: object) -> bool:
    """Allow accumulated floating-point time noise around the 10 us target."""

    try:
        observed = float(value)
    except (TypeError, ValueError):
        return False
    return (
        math.isfinite(observed)
        and abs(observed - CD_ALL_PHYSICS_10US_STOP_TIME_S)
        <= CD_ALL_PHYSICS_10US_OUTPUT_TIME_TOLERANCE_S
    )


def all_physics_8_9_10us_flux_summary(
    series: list[tuple[float, float]],
) -> dict:
    """Return descriptive late-window growth; this is deliberately not a gate."""

    fluxes = {
        label: _flux_at_time(series, time_s)
        for label, time_s in (
            ("8us", 8.0e-6),
            ("9us", 9.0e-6),
            ("10us", 10.0e-6),
        )
    }

    def fractional_change(first: float, second: float) -> float:
        return (
            (second - first) / first
            if math.isfinite(first) and math.isfinite(second) and first != 0.0
            else math.nan
        )

    return {
        "signed_e_ge_1mev_downstream_flux": fluxes,
        "fractional_growth_8us_to_9us": fractional_change(fluxes["8us"], fluxes["9us"]),
        "fractional_growth_9us_to_10us": fractional_change(fluxes["9us"], fluxes["10us"]),
        "downstream_primary_flux_positive_before_10us": any(
            time_s < CD_ALL_PHYSICS_10US_STOP_TIME_S and flux > 0.0 for time_s, flux in series
        ),
        "all_requested_times_present": all(math.isfinite(value) for value in fluxes.values()),
    }


def _aggregate_process_context(final_row: dict[str, str]) -> dict:
    context = {}
    for group, columns in AGGREGATE_PROCESS_CONTEXT_COLUMNS.items():
        values = {}
        for column in columns:
            if column not in final_row:
                raise ValueError(f"missing required reduced-CSV column {column!r}")
            try:
                value = float(final_row[column])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"required reduced-CSV column {column!r} is not numeric"
                ) from exc
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(
                    f"required reduced-CSV column {column!r} must be finite and nonnegative"
                )
            values[column] = value
        context[group] = values
    return {
        "groups": context,
        "causal_interpretation_permitted": False,
        "note": "aggregate counters are non-causal context, not a feedback coefficient",
    }



def _latest_attempt_status(cap: Path) -> dict:
    try:
        attempts = sorted(
            (cap / "command_attempts").glob("attempt_*.json"),
            key=lambda path: int(path.stem.split("_")[-1]),
        )
    except ValueError:
        return {
            "clean_exit": False,
            "reason": "MALFORMED_ATTEMPT_FILENAME",
        }
    if not attempts:
        return {
            "clean_exit": False,
            "reason": "MISSING_ATTEMPT_RECORD",
        }
    latest = attempts[-1]
    try:
        exit_payload = json.loads((cap / "command_exit.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError):
        return {
            "clean_exit": False,
            "reason": "MALFORMED_OR_MISSING_LATEST_ATTEMPT_EXIT",
            "latest_attempt_record": str(latest),
        }
    clean = (
        not (cap / "command_in_progress.json").exists()
        and exit_payload.get("returncode") == 0
    )
    return {
        "clean_exit": clean,
        # Load-bearing: parse_case reads the command file this names, in
        # preference to a command.json left behind by an earlier attempt.
        "latest_attempt_record": str(latest),
        "reason": None if clean else "LATEST_ATTEMPT_NOT_CLEANLY_COMPLETED",
    }


def _final_spatial_moments(cap: Path) -> dict[str, float] | None:
    path = cap / "rrea_video" / "particle_moments.csv"
    if not path.is_file():
        return None
    _, rows = _csv_rows(path)
    required = {"write_index", "time_s", "mean_z_m", "rms_z_m", "rms_r_m"}
    if not rows or not required.issubset(rows[0]):
        return None
    latest = max(rows, key=lambda row: (float(row["time_s"]), int(row["write_index"])))
    latest_time_s = float(latest["time_s"])
    if not _is_exact_10us_output_time(latest_time_s):
        return None
    result = {
        "time_s": latest_time_s,
        "axial_centroid_m": float(latest["mean_z_m"]),
        "axial_rms_m": float(latest["rms_z_m"]),
        "radial_rms_m": float(latest["rms_r_m"]),
    }
    return result if all(math.isfinite(value) for value in result.values()) else None


def profiles_for_case(case: dict) -> dict:
    metadata = case["summary"].get("profiles", {})
    profiles = {}
    for label, loader in (
        ("field", load_field_profile),
        ("density", load_density_ratio_profile),
        ("threshold", load_threshold_profile),
    ):
        path = Path(metadata.get(f"{label}_profile_path", ""))
        profile = loader(path) if path.is_file() else loader()
        profiles[label] = profile
    return profiles


def parse_case(run_dir: Path) -> dict | None:
    """Read recorded prelaunch provenance plus completed diagnostics."""
    cap = _capture_dir(run_dir)
    if not (cap / "command.json").exists() or not (cap / "rrea_reduced.csv").exists():
        print(f"[skip] incomplete run dir (no command.json/reduced csv): {cap}", file=sys.stderr)
        return None
    attempt_status = _latest_attempt_status(cap)
    latest_attempt = attempt_status.get("latest_attempt_record")
    command_path = (
        Path(latest_attempt)
        if isinstance(latest_attempt, str) and Path(latest_attempt).is_file()
        else cap / "command.json"
    )
    command_payload = json.loads(command_path.read_text())
    command = command_payload.get("command", [])
    provenance = command_payload.get("metadata", {})
    if not isinstance(provenance.get("resolved_driver_arguments"), dict):
        raise SystemExit(f"{cap}/command.json lacks resolved_driver_arguments provenance")
    e0_text = _command_value(command, "rrea.profile_e0_peak_v_per_m")
    dt_text = _command_value(command, "warpx.const_dt")
    e0_v = float(e0_text) if e0_text is not None else None
    dt_s = float(dt_text) if dt_text is not None else None
    if e0_v is None or dt_s is None:
        raise SystemExit(f"{cap}/command.json lacks E0/dt tokens")
    summary_path = cap / "profiled_video_capture_summary.json"
    if not summary_path.exists():
        print(
            f"[skip] incomplete run dir (no {summary_path.name}): {cap}",
            file=sys.stderr,
        )
        return None
    summary = json.loads(summary_path.read_text())
    _, reduced_rows = _csv_rows(cap / "rrea_reduced.csv")
    if not reduced_rows:
        raise SystemExit(f"{cap}/rrea_reduced.csv has no rows")
    try:
        final_reduced_row = max(
            reduced_rows,
            key=lambda row: (float(row["time_s"]), int(float(row["step"]))),
        )
        final_reduced_time_s = float(final_reduced_row["time_s"])
        final_reduced_step = int(float(final_reduced_row["step"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise SystemExit(f"{cap}: reduced CSV lacks valid final step/time") from exc
    resample_activity = _nonzero_columns(reduced_rows, RESAMPLE_ACTIVITY_COLUMNS)
    clean_exit = bool(attempt_status["clean_exit"])
    expected_seed_count = summary.get("macro_count")
    try:
        final_injected_macro_count = int(float(final_reduced_row["injected_macro_count"]))
        final_injected_physical_weight = float(final_reduced_row["injected_physical_weight"])
        final_next_seed_event = int(float(final_reduced_row["next_seed_event"]))
        final_native_seed_particle_count = int(
            float(final_reduced_row["native_seed_particle_count"])
        )
        actual_source_injection_complete = (
            isinstance(expected_seed_count, int)
            and final_injected_macro_count == expected_seed_count
            and final_next_seed_event == expected_seed_count
            and final_injected_physical_weight == float(expected_seed_count)
            and final_native_seed_particle_count >= 0
        )
    except (KeyError, TypeError, ValueError):
        final_injected_macro_count = None
        final_injected_physical_weight = None
        final_next_seed_event = None
        final_native_seed_particle_count = None
        actual_source_injection_complete = False

    if not any(
        math.isclose(
            e0_v / 1.0e3,
            field,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
        for field in CD_ALL_PHYSICS_10US_E0_KV_PER_M
    ):
        raise SystemExit(f"{cap}: E0={e0_v / 1.0e3:g} kV/m is outside {CD_ALL_PHYSICS_10US_SCOPE}")
    if summary.get("seed_source_model") != "coleman_dwyer_7p2mev":
        raise SystemExit(
            f"{cap}: {CD_ALL_PHYSICS_10US_SCOPE} requires seed_source_model='coleman_dwyer_7p2mev'"
        )
    # The one fixed injection altitude, read back from the schedule the engine
    # consumed; the driver owns how that schedule was constructed.
    _, schedule_rows = _csv_rows(cap / "profiled_video_seed_schedule.csv")
    fixed_source_z_m = {float(row["z_m"]) for row in schedule_rows}
    if len(fixed_source_z_m) != 1:
        raise SystemExit(f"{cap}: C&D seeds are not at one fixed altitude")
    fixed_source_z_m = fixed_source_z_m.pop()

    realization = summary.get("seed_realization_id")
    if not isinstance(realization, int) or realization < 0:
        raise SystemExit(f"{cap}: invalid seed_realization_id {realization!r}")
    cohort_name = cd_all_physics_cohort_name(realization, int(summary.get("macro_count")))
    # Only physical and statistical dimensions relevant to pooling belong here.
    signature = (
        round(e0_v / 1e3, 9),
        dt_s,
        summary.get("domain_radius_m"),
        summary.get("domain_height_m"),
        summary.get("n_cell_r"),
        summary.get("n_cell_z"),
        summary.get("stop_time_s"),
        summary.get("seed_source_model"),
        cohort_name,
        summary.get("macro_count"),
        summary.get("population_target_electron_macros"),
        summary.get("injected_real_electrons"),
        summary.get("source_radius_m"),
        summary.get("cd_seed_design"),
        summary.get("cd_seed_permutation"),
        tuple(summary.get("cd_plane_z_m", [])),
        summary.get("cd_plane_output_interval_s"),
        summary.get("field_taper_r_start_m"),
        summary.get("field_taper_r_end_m"),
    )
    try:
        aggregate_process_context = _aggregate_process_context(final_reduced_row)
    except ValueError as exc:
        raise SystemExit(f"{cap}/rrea_reduced.csv: {exc}") from exc
    return {
        "run_dir": cap,
        "e0_kv": e0_v / 1e3,
        "dt_s": dt_s,
        "summary": summary,
        "provenance": provenance,
        "reduced_rows": reduced_rows,
        "aggregate_process_context": aggregate_process_context,
        "resample_activity": resample_activity,
        "fixed_source_z_m": fixed_source_z_m,
        "fixed_source_altitude_m": fixed_source_z_m + ALTITUDE_AT_Z0_M,
        "population_policy": _command_value(command, "rrea.population_ceiling_policy"),
        "clean_exit": clean_exit,
        "attempt_status": attempt_status,
        "final_reduced_time_s": final_reduced_time_s,
        "final_reduced_step": final_reduced_step,
        "actual_source_injection_complete": actual_source_injection_complete,
        "final_source_injection_counters": {
            "injected_macro_count": final_injected_macro_count,
            "injected_physical_weight": final_injected_physical_weight,
            "next_seed_event": final_next_seed_event,
            "native_seed_particle_count": final_native_seed_particle_count,
        },
        "final_spatial_moments": _final_spatial_moments(cap),
        # Populated only after the exact plane-v1 file is validated and its
        # signed radial moments are reduced in summarize_group.
        "plane_crossing_spatial_moments": None,
        "realization_id": realization,
        "group_signature": signature,
    }


def _validated_plane_blocks(path: Path, expected_altitudes_m: list[float]):
    """Yield the per-write blocks (one row per plane) of a plane-flux file."""

    if not path.exists():
        return
    if not expected_altitudes_m:
        raise SystemExit(f"{path}: run summary has no configured plane sequence")
    plane_count = len(expected_altitudes_m)
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or len(rows) % plane_count:
        raise SystemExit(f"{path}: {len(rows)} rows is not whole {plane_count}-plane blocks")
    for start in range(0, len(rows), plane_count):
        yield rows[start:start + plane_count]


def _plane_flux_reduction(path: Path, expected_altitudes_m: list[float]) -> dict[str, object]:
    """Reduce a production plane file to its final block and compact time series."""

    final_rows: list[dict[str, str]] = []
    final_write: tuple[int, int, float] | None = None
    downstream_series: list[tuple[float, float]] = []
    plane_series: dict[int, list[tuple[float, float]]] = {}
    max_field_perturbation = math.nan
    for block in _validated_plane_blocks(path, expected_altitudes_m):
        final_rows = block
        final_write = (
            int(block[0]["write_index"]),
            int(block[0]["step"]),
            float(block[0]["time_s"]),
        )
        downstream = block[-1]
        downstream_series.append(
            (
                float(downstream["time_s"]),
                float(downstream["e_ge_1mev_up_weight"])
                - float(downstream["e_ge_1mev_down_weight"]),
            )
        )
        for row in block:
            plane_series.setdefault(int(row["plane_index"]), []).append(
                (
                    float(row["time_s"]),
                    float(row["e_ge_1mev_up_weight"])
                    - float(row["e_ge_1mev_down_weight"]),
                )
            )
        block_max = max(float(row["core_field_perturbation_fraction"]) for row in block)
        max_field_perturbation = (
            block_max
            if not math.isfinite(max_field_perturbation)
            else max(max_field_perturbation, block_max)
        )
    return {
        "final_rows": final_rows,
        "final_write": final_write,
        "downstream_series": downstream_series,
        "plane_series": plane_series,
        "max_abs_core_field_perturbation_fraction": max_field_perturbation,
    }


def measured_front_speed_m_per_s(
    final_rows: list[dict[str, str]],
    plane_series: dict[int, list[tuple[float, float]]],
) -> float:
    """Median-arrival Theil-Sen speed through interior diagnostic planes."""
    arrivals: list[tuple[float, float]] = []
    for row in sorted(final_rows, key=lambda item: int(item["plane_index"])):
        plane = int(row["plane_index"])
        final_flux = (
            float(row["e_ge_1mev_up_weight"])
            - float(row["e_ge_1mev_down_weight"])
        )
        if not (final_flux > 0.0):
            continue
        target = 0.5 * final_flux
        arrival = next(
            (time_s for time_s, flux in plane_series.get(plane, []) if flux >= target),
            None,
        )
        if arrival is not None:
            arrivals.append((arrival, float(row["altitude_m"])))
    if len(arrivals) > 4:
        arrivals = arrivals[2:-2]
    slopes = [
        (z1 - z0) / (t1 - t0)
        for i, (t0, z0) in enumerate(arrivals)
        for t1, z1 in arrivals[i + 1:]
        if t1 > t0
    ]
    return statistics.median(slopes) if slopes else math.nan


def signed_plane_multiplier(rows: list[dict[str, str]]) -> tuple[float, list[dict[str, float]]]:
    """Endpoint multiplication from one complete final cumulative write."""
    if not rows:
        return math.nan, []
    final_write = max(int(row["write_index"]) for row in rows)
    latest = {
        int(row["plane_index"]): row for row in rows if int(row["write_index"]) == final_write
    }
    planes = []
    for plane_index, row in latest.items():
        net_by_selection = {
            selection: float(row[f"{selection}_up_weight"])
            - float(row[f"{selection}_down_weight"])
            for selection in ("all_tracked", "local_runaway", "e_ge_1mev")
        }
        # The primary count is a FIXED energy window: above 1 MeV the avalanche
        # spectrum is field-independent, so the counted fraction is the same on
        # every plane and the ratio is the avalanche growth. The local-runaway
        # window moves with the field along the profile and biases it.
        net = net_by_selection["e_ge_1mev"]
        net_r = float(row["local_runaway_up_r_weighted_m"]) - float(
            row["local_runaway_down_r_weighted_m"]
        )
        net_r2 = float(row["local_runaway_up_r2_weighted_m2"]) - float(
            row["local_runaway_down_r2_weighted_m2"]
        )
        plane = {
            "plane_index": plane_index,
            "altitude_m": float(row["altitude_m"]),
            "net": net,
            "net_all_tracked": net_by_selection["all_tracked"],
            "net_local_runaway": net_by_selection["local_runaway"],
            "net_e_ge_1mev": net_by_selection["e_ge_1mev"],
        }
        local_net = net_by_selection["local_runaway"]
        if local_net > 0.0 and net_r >= 0.0 and net_r2 >= 0.0:
            plane["local_runaway_mean_r_m"] = net_r / local_net
            plane["local_runaway_rms_r_m"] = math.sqrt(net_r2 / local_net)
        planes.append(plane)
    planes.sort(key=lambda item: item["altitude_m"])
    if len(planes) < 2 or planes[0]["net"] <= 0.0 or planes[-1]["net"] <= 0.0:
        return math.nan, planes
    return planes[-1]["net"] / planes[0]["net"], planes


def plane_multiplier_unusable_reason(planes: list[dict[str, float]]) -> str:
    """Explain a non-finite endpoint multiplier.

    ``signed_plane_multiplier`` collapses three different conditions into one
    NaN. Only one of them means the data is bad: a positive bottom plane with an
    empty top plane is an avalanche still in flight, which is a wall-clock
    result, not a corrupt file. Returns "" when a multiplier was formable.
    """
    if len(planes) < 2:
        return "MISSING_OR_INVALID_SIGNED_PLANE_FLUX"
    ordered = sorted(planes, key=lambda item: item["altitude_m"])
    if ordered[0]["net"] <= 0.0:
        return "MISSING_OR_INVALID_SIGNED_PLANE_FLUX"
    if ordered[-1]["net"] <= 0.0:
        return "PRIMARY_TRANSIT_INCOMPLETE"
    return ""


def signed_plane_multipliers_by_selection(
    planes: list[dict[str, float]],
) -> dict[str, float]:
    if len(planes) < 2:
        return {
            "all_tracked": math.nan,
            "local_runaway": math.nan,
            "e_ge_1mev": math.nan,
        }
    result = {}
    for selection, key in (
        ("all_tracked", "net_all_tracked"),
        ("local_runaway", "net_local_runaway"),
        ("e_ge_1mev", "net_e_ge_1mev"),
    ):
        upstream = float(planes[0][key])
        downstream = float(planes[-1][key])
        result[selection] = (
            downstream / upstream if upstream > 0.0 and downstream > 0.0 else math.nan
        )
    return result


def steady_state_slopes(
    planes: list[dict[str, float]], profiles: dict, e0_kv: float, reference: dict
) -> dict[str, float | int]:
    region_data = reference["above_threshold_region"]
    base = AboveThresholdRegion(**region_data)
    candidates = []
    for plane in planes:
        altitude = plane["altitude_m"]
        if not (base.injection_altitude_m <= altitude <= base.top_altitude_m):
            continue
        partial = AboveThresholdRegion(
            bottom_altitude_m=base.bottom_altitude_m,
            top_altitude_m=altitude,
            injection_altitude_m=base.injection_altitude_m,
            injection_z_m=base.injection_z_m,
            width_m=altitude - base.bottom_altitude_m,
            e0_peak_v_per_m=base.e0_peak_v_per_m,
            field_at_injection_v_per_m=base.field_at_injection_v_per_m,
            threshold_at_injection_v_per_m=base.threshold_at_injection_v_per_m,
            density_ratio_at_injection=base.density_ratio_at_injection,
        )
        s_values = {
            threshold: estimate_integrated_avalanche_lengths(
                field=profiles["field"],
                density_ratio=profiles["density"],
                region=partial,
                e0_peak_v_per_m=e0_kv * 1e3,
                runaway_threshold_stp_kv_per_m=float(threshold),
            ).predicted_avalanche_lengths
            for threshold in (276, 284)
        }
        if s_values[284] >= 2.5:
            candidates.append((s_values, float(plane["net"])))
    fit_span_284 = (
        max(point[0][284] for point in candidates)
        - min(point[0][284] for point in candidates)
        if len(candidates) >= 2
        else 0.0
    )
    all_candidate_flux_positive = all(
        math.isfinite(net_flux) and net_flux > 0.0 for _, net_flux in candidates
    )
    fit_eligible = (
        len(candidates) >= 5
        and fit_span_284 >= 1.0
        and all_candidate_flux_positive
    )
    result: dict[str, float | int] = {
        "fit_plane_count": len(candidates),
        "fit_efold_span_284": fit_span_284,
        "all_candidate_fit_plane_flux_positive": all_candidate_flux_positive,
        "actual_fit_interval_eligible": fit_eligible,
    }
    if not fit_eligible:
        return result
    for threshold in (276, 284):
        xs = [point[0][threshold] for point in candidates]
        ys = [math.log(point[1]) for point in candidates]
        x_mean = statistics.fmean(xs)
        y_mean = statistics.fmean(ys)
        denominator = sum((x - x_mean) ** 2 for x in xs)
        if denominator > 0.0:
            result[f"slope_{threshold}"] = (
                sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys, strict=False)) / denominator
            )
    return result


def measurement_interval_reference(
    profiles: dict, e0_kv: float, start_altitude_m: float, end_altitude_m: float
) -> dict[str, dict[str, float | None]]:
    density = profiles["density"]
    field = profiles["field"]
    region = AboveThresholdRegion(
        bottom_altitude_m=start_altitude_m,
        top_altitude_m=end_altitude_m,
        injection_altitude_m=start_altitude_m,
        injection_z_m=start_altitude_m - ALTITUDE_AT_Z0_M,
        width_m=end_altitude_m - start_altitude_m,
        e0_peak_v_per_m=e0_kv * 1e3,
        field_at_injection_v_per_m=profiled_abs_field_v_per_m(field, e0_kv * 1e3, start_altitude_m),
        threshold_at_injection_v_per_m=284000.0 * density.interpolate(start_altitude_m),
        density_ratio_at_injection=density.interpolate(start_altitude_m),
    )
    curves = {}
    for threshold in (276, 284):
        estimate = estimate_integrated_avalanche_lengths(
            field=field,
            density_ratio=density,
            region=region,
            e0_peak_v_per_m=e0_kv * 1e3,
            runaway_threshold_stp_kv_per_m=float(threshold),
        )
        curves[str(threshold)] = {
            "predicted_avalanche_lengths": estimate.predicted_avalanche_lengths,
            "predicted_multiplication": estimate.predicted_multiplication,
        }
    return curves


def _inconclusive_thresholds(reason: str, reference_multipliers: dict) -> dict:
    return {
        str(threshold): {
            "reference_multiplier": reference_multipliers[threshold],
            "ratio_point_estimate": math.nan,
            "ratio_student_t_95pct_ci": [math.nan, math.nan],
            "percent_deviation": math.nan,
            "percent_deviation_student_t_95pct_ci": [math.nan, math.nan],
            "classification": f"CD_INCONCLUSIVE_{threshold}",
            "verdict": f"CD_INCONCLUSIVE_{threshold}",
            "reason": reason,
        }
        for threshold in (276, 284)
    }


def summarize_group(cases: list[dict], profiles: dict, tolerance: float) -> dict:
    signatures = {case.get("group_signature") for case in cases}
    if len(signatures) != 1:
        raise SystemExit("refusing to mix C&D runs with incompatible source/run signatures")
    realization_ids = [
        int(case.get("realization_id", case["summary"].get("seed_realization_id")))
        for case in cases
    ]
    if len(set(realization_ids)) != len(realization_ids):
        raise SystemExit("duplicate seed_realization_id in one C&D ensemble")
    try:
        seed_count_for_cohort = int(cases[0]["summary"].get("macro_count"))
    except (TypeError, ValueError) as exc:
        raise SystemExit("invalid C&D ensemble seed count") from exc
    cohort = classify_all_physics_realization_cohort(
        realization_ids,
        seed_count_for_cohort,
    )
    if cohort["name"] == "invalid":
        raise SystemExit(
            "C&D ensemble realizations mix or fall outside the fixed "
            "engineering/confirmatory/diagnostic cohorts"
        )
    e0 = cases[0]["e0_kv"]
    reference = coleman_dwyer_profile_reference(
        field=profiles["field"],
        density_ratio=profiles["density"],
        threshold_profile=profiles["threshold"],
        e0_peak_v_per_m=e0 * 1e3,
    )
    if reference is None:
        thresholds = _inconclusive_thresholds(
            "NO_ABOVE_THRESHOLD_REGION", {276: 1.0, 284: 1.0})
        return {
            "e0_peak_kv_per_m": e0,
            "status": "NO_ABOVE_THRESHOLD_REGION",
            "thresholds": thresholds,
            "verdict": "INCONCLUSIVE",
        }
    # The driver built the source and planes from this same reference; the
    # group signature guarantees every case recorded the same ones.
    canonical_plane_z_m = [float(z_m) for z_m in cases[0]["summary"]["cd_plane_z_m"]]
    canonical_plane_altitudes_m = [z_m + ALTITUDE_AT_Z0_M for z_m in canonical_plane_z_m]
    canonical_geometry = {
        "source_altitude_m": cases[0]["fixed_source_altitude_m"],
        "plane_altitudes_m": canonical_plane_altitudes_m,
    }
    measurement_start = canonical_plane_altitudes_m[0]
    measurement_end = canonical_plane_altitudes_m[-1]
    measurement_curves = measurement_interval_reference(
        profiles, e0, measurement_start, measurement_end
    )
    multipliers = []
    by_realization: dict[str, float] = {}
    front_speed_by_realization: dict[str, float] = {}
    by_selection: dict[str, dict[str, float]] = {
        "all_tracked": {},
        "local_runaway": {},
        "e_ge_1mev": {},
    }
    missing = []
    transit_incomplete = []
    plane_reductions_by_case: list[dict[str, object]] = []
    final_plane_writes: list[tuple[int, int, float]] = []
    for case in cases:
        plane_path = case["run_dir"] / "rrea_cd_plane_flux.csv"
        plane_reduction = _plane_flux_reduction(plane_path, canonical_plane_altitudes_m)
        plane_reductions_by_case.append(plane_reduction)
        final_write = plane_reduction["final_write"]
        if final_write is not None:
            final_plane_writes.append(final_write)
        multiplier, planes = signed_plane_multiplier(plane_reduction["final_rows"])
        case["plane_flux"] = planes
        case["signed_multiplier"] = multiplier
        front_speed = measured_front_speed_m_per_s(
            plane_reduction["final_rows"], plane_reduction["plane_series"])
        realization = int(
            case.get("realization_id", case["summary"].get("seed_realization_id"))
        )
        if math.isfinite(front_speed) and front_speed > 0.0:
            front_speed_by_realization[str(realization)] = front_speed
        selection_multipliers = signed_plane_multipliers_by_selection(planes)
        case["signed_multipliers_by_selection"] = selection_multipliers
        case["plane_crossing_spatial_moments"] = (
            [
                {
                    "plane_index": int(plane["plane_index"]),
                    "altitude_m": plane["altitude_m"],
                    "local_runaway_mean_r_m": plane["local_runaway_mean_r_m"],
                    "local_runaway_rms_r_m": plane["local_runaway_rms_r_m"],
                }
                for plane in planes
            ]
            if planes and all(
                "local_runaway_mean_r_m" in plane
                and "local_runaway_rms_r_m" in plane
                for plane in planes)
            else None
        )
        # Late-window flux growth is reported, never used to drop a
        # realization: persistent growth can be genuine feedback.
        if multiplier > 0.0 and math.isfinite(multiplier):
            multipliers.append(multiplier)
            by_realization[str(realization)] = multiplier
            for selection, value in selection_multipliers.items():
                if math.isfinite(value) and value > 0.0:
                    by_selection[selection][str(realization)] = value
        elif plane_multiplier_unusable_reason(planes) == "PRIMARY_TRANSIT_INCOMPLETE":
            transit_incomplete.append(str(plane_path))
        else:
            missing.append(str(plane_path))
    if len(set(final_plane_writes)) > 1:
        raise SystemExit("C&D realizations do not share one common final plane write")
    if len(multipliers) != len(cases):
        all_transit_incomplete = bool(transit_incomplete) and not missing
        reason = (
            "PRIMARY_TRANSIT_INCOMPLETE"
            if all_transit_incomplete
            else "MISSING_OR_INVALID_SIGNED_PLANE_FLUX"
        )
        thresholds = _inconclusive_thresholds(reason, {
            threshold: float(measurement_curves[str(threshold)]["predicted_multiplication"])
            for threshold in (276, 284)})
        return {
            "e0_peak_kv_per_m": e0,
            "status": (
                "INCONCLUSIVE_PRIMARY_TRANSIT_INCOMPLETE"
                if all_transit_incomplete
                else "INCONCLUSIVE_MISSING_SIGNED_PLANE_FLUX"
            ),
            "missing_or_invalid_plane_flux": missing,
            "primary_transit_incomplete_plane_flux": transit_incomplete,
            "analytic_reference": reference,
            "measurement_interval_reference": {
                "start_altitude_m": measurement_start,
                "end_altitude_m": measurement_end,
                "curves": measurement_curves,
            },
            "thresholds": thresholds,
            "verdict": "INCONCLUSIVE",
        }
    slope_rows = [
        steady_state_slopes(case["plane_flux"], profiles, e0, reference)
        for case in cases
        if case.get("plane_flux")
    ]
    logs = [math.log(value) for value in multipliers]
    mean_log, log_ci = _student_t_mean_ci(logs)
    estimate = math.exp(mean_log)
    ci = tuple(math.exp(value) if math.isfinite(value) else math.nan for value in log_ci)
    multiplier_median = statistics.median(multipliers)
    if len(multipliers) >= 2:
        multiplier_q1, _, multiplier_q3 = statistics.quantiles(
            multipliers, n=4, method="inclusive")
    else:
        multiplier_q1 = multiplier_q3 = multiplier_median
    thresholds = {}
    for threshold in (276, 284):
        entry = all_physics_threshold_deviation(
            multiplier=estimate,
            multiplier_ci=ci,
            reference_multiplier=float(
                measurement_curves[str(threshold)]["predicted_multiplication"]),
            threshold_kv_per_m=threshold,
            tolerance=tolerance)
        thresholds[str(threshold)] = {
            **entry,
            "verdict": entry["classification"],
            "interpretation": "descriptive_nominal_wkb_reference"}
    steady = bool(reference["steady_state_growth_eligible"])
    result_classification = cd_all_physics_result_class(e0, steady)
    max_field_perturbation = max(
        [
            float(reduction["max_abs_core_field_perturbation_fraction"])
            for reduction in plane_reductions_by_case
            if math.isfinite(float(reduction["max_abs_core_field_perturbation_fraction"]))
        ],
        default=math.nan,
    )
    late_flux_summaries = [
        all_physics_8_9_10us_flux_summary(reduction["downstream_series"])
        for reduction in plane_reductions_by_case
    ]
    ci_half_width_fraction = (
        0.5 * (ci[1] - ci[0]) / estimate
        if estimate > 0.0 and all(math.isfinite(value) for value in ci)
        else math.nan
    )
    no_population_resampling = all(
        case["summary"].get("population_target_electron_macros") == 0
        and case.get("population_policy") == "strict_abort_v1"
        and not case.get("resample_activity", {})
        for case in cases
    )
    no_strict_ceiling_activity = all(
        case.get("clean_exit", False)
        and isinstance(case["summary"].get("strict_population_ceilings"), dict)
        for case in cases
    )
    positive_downstream_flux = all(
        bool(case.get("plane_flux"))
        and math.isfinite(float(case["plane_flux"][-1]["net"]))
        and float(case["plane_flux"][-1]["net"]) > 0.0
        for case in cases
    )
    positive_fitted_plane_flux = (
        True
        if not steady
        else len(slope_rows) == len(cases)
        and all(
            bool(row.get("all_candidate_fit_plane_flux_positive", False))
            for row in slope_rows
        )
    )
    actual_steady_fit_eligible = (
        len(slope_rows) == len(cases)
        and len(slope_rows) >= 3
        and steady
        and all(
            bool(row.get("actual_fit_interval_eligible", False))
            and "slope_276" in row
            for row in slope_rows
        )
    )
    final_plane_time_s = (
        final_plane_writes[0][2] if len(final_plane_writes) == len(cases) else math.nan
    )
    actual_reduced_complete = all(
        _is_exact_10us_output_time(case.get("final_reduced_time_s")) for case in cases
    )
    actual_plane_complete = _is_exact_10us_output_time(final_plane_time_s)
    actual_video_complete = all(
        isinstance(case.get("final_spatial_moments"), dict)
        and _is_exact_10us_output_time(case["final_spatial_moments"].get("time_s"))
        for case in cases
    )
    exact_10us_observation_window = actual_reduced_complete and actual_plane_complete
    configured_max_step_complete = all(
        isinstance(case["summary"].get("max_step"), int)
        and case["summary"]["max_step"]
        >= required_time_steps(float(case["summary"].get("stop_time_s", 0.0)), case["dt_s"])
        for case in cases
    )
    latest_attempts_clean = all(case.get("clean_exit", False) for case in cases)
    minimum_three_realizations = len(multipliers) >= 3
    primary_transit_complete = bool(plane_reductions_by_case) and all(
        any(
            time_s
            <= (
                CD_ALL_PHYSICS_10US_STOP_TIME_S
                + CD_ALL_PHYSICS_10US_OUTPUT_TIME_TOLERANCE_S
            )
            and flux > 0.0
            for time_s, flux in reduction["downstream_series"]
        )
        for reduction in plane_reductions_by_case
    )
    eligibility_checks = {
        "complete_source_injection": all(
            case.get("actual_source_injection_complete", False) for case in cases
        ),
        "realization_cohort_complete": cohort["complete"],
        "latest_attempt_complete": latest_attempts_clean,
        "no_population_resampling": no_population_resampling,
        "no_strict_ceiling_activity": no_strict_ceiling_activity,
        "exact_10us_observation_window": exact_10us_observation_window,
        "core_field_perturbation_below_1pct": (
            math.isfinite(max_field_perturbation) and max_field_perturbation < 0.01
        ),
        "multiplication_ci_half_width_at_most_3pct": (
            minimum_three_realizations
            and math.isfinite(ci_half_width_fraction)
            and ci_half_width_fraction <= CD_PRIMARY_CI_HALF_WIDTH_FRACTION
        ),
        "positive_downstream_signed_flux": positive_downstream_flux,
        "positive_final_net_flux_on_all_fitted_planes": positive_fitted_plane_flux,
        "actual_reduced_time_reaches_configured_stop": actual_reduced_complete,
        "actual_plane_time_reaches_configured_stop": actual_plane_complete,
        "actual_video_time_reaches_configured_stop": actual_video_complete,
        "max_step_can_reach_configured_stop": configured_max_step_complete,
        "primary_transit_complete_by_10us": primary_transit_complete,
    }
    realization_count = len(multipliers)
    seed_count = cases[0]["summary"].get("macro_count")
    primary_ci_half_width_fraction = ci_half_width_fraction
    primary_statistics_sufficient = (
        minimum_three_realizations
        and math.isfinite(primary_ci_half_width_fraction)
        and primary_ci_half_width_fraction <= CD_PRIMARY_CI_HALF_WIDTH_FRACTION
    )
    eligibility_reasons = [
        f"FAILED_{name.upper()}" for name, passed in eligibility_checks.items() if not passed
    ]
    if not primary_statistics_sufficient:
        target = realization_count
        if not no_population_resampling:
            # Thinning noise, not the seed count, sets a resampled ensemble's
            # CI; it narrows as 1/sqrt(realizations).
            action = "ADD_REALIZATIONS"
            if math.isfinite(primary_ci_half_width_fraction):
                target = math.ceil(realization_count * (
                    primary_ci_half_width_fraction / CD_PRIMARY_CI_HALF_WIDTH_FRACTION) ** 2)
        elif seed_count == CD_INITIAL_SEED_COUNT:
            action = "ESCALATE_TO_200K_UNIT_WEIGHT_SEEDS"
        else:
            action = "INCONCLUSIVE_INADEQUATE_STATISTICS"
        statistical_recommendation = {
            "action": action,
            "target_realization_count": target,
            "reason": (
                "the primary realization-level endpoint multiplication CI half-width exceeds 3%"
            ),
        }
    else:
        statistical_recommendation = {
            "action": "STATISTICS_SUFFICIENT",
            "target_realization_count": realization_count,
        }
    # Eligibility reasons remain visible evidence, but this benchmark does
    # not convert them into a binary agreement verdict.
    selection_summaries = {}
    for selection, values_by_realization in by_selection.items():
        values = list(values_by_realization.values())
        selection_summaries[selection] = {
            "multipliers_by_realization": values_by_realization,
            "geometric_mean_multiplier": (
                math.exp(statistics.fmean(math.log(value) for value in values))
                if values
                else math.nan
            ),
        }
    result_payload = {
        "e0_peak_kv_per_m": e0,
        "dt_s": cases[0]["dt_s"],
        "domain_radius_m": cases[0]["summary"].get("domain_radius_m"),
        "domain_height_m": cases[0]["summary"].get("domain_height_m"),
        "n_cell_r": cases[0]["summary"].get("n_cell_r"),
        "n_cell_z": cases[0]["summary"].get("n_cell_z"),
        "ranks": cases[0]["summary"].get("ranks"),
        "cpus_per_rank": cases[0]["summary"].get("cpus_per_rank"),
        "max_grid_size": cases[0]["provenance"]["resolved_driver_arguments"].get("max_grid_size"),
        "stop_time_s": cases[0]["summary"].get("stop_time_s"),
        "seed_count": cases[0]["summary"].get("macro_count"),
        "seed_design": cases[0]["summary"].get("cd_seed_design"),
        "seed_escalation_reason": cases[0]["summary"].get("cd_seed_escalation_reason"),
        "seed_permutation": cases[0]["summary"].get("cd_seed_permutation"),
        "source_radius_m": cases[0]["summary"].get("source_radius_m"),
        "cd_plane_z_m": canonical_plane_z_m,
        "field_taper_r_start_m": cases[0]["summary"].get("field_taper_r_start_m"),
        "field_taper_r_end_m": cases[0]["summary"].get("field_taper_r_end_m"),
        "campaign_scope": CD_ALL_PHYSICS_10US_SCOPE,
        "observation_label": CD_ALL_PHYSICS_10US_LABEL,
        "result_classification": result_classification,
        "realization_cohort": cohort,
        "population_policy": cases[0].get("population_policy"),
        "strict_population_ceilings": cases[0]["summary"].get("strict_population_ceilings"),
        "output_case_ids_by_realization": {
            str(case["realization_id"]): case["run_dir"].parent.name for case in cases
        },
        "realization_count": len(multipliers),
        "multipliers": multipliers,
        "multipliers_by_realization": by_realization,
        "front_speed_m_per_s_by_realization": front_speed_by_realization,
        "median_front_speed_m_per_s": (
            statistics.median(front_speed_by_realization.values())
            if front_speed_by_realization else math.nan
        ),
        "multipliers_by_selection_by_realization": by_selection,
        "signed_flux_multipliers_by_selection": selection_summaries,
        "final_spatial_moments_by_realization": {
            str(case.get("realization_id")): case.get("final_spatial_moments")
            for case in cases
            if case.get("final_spatial_moments") is not None
        },
        "plane_crossing_spatial_moments_available": all(
            case.get("plane_crossing_spatial_moments") is not None for case in cases
        ),
        "plane_crossing_spatial_moments_by_realization": {
            str(case.get("realization_id")): case["plane_crossing_spatial_moments"]
            for case in cases
            if case.get("plane_crossing_spatial_moments") is not None
        },
        "geometric_mean_multiplier": estimate,
        "median_multiplier": multiplier_median,
        "multiplier_iqr": [multiplier_q1, multiplier_q3],
        "thresholds": thresholds,
        "verdict": combined_all_physics_deviation(
            [thresholds[str(threshold)]["classification"] for threshold in (276, 284)]),
        "statistical_recommendation": statistical_recommendation,
        "analytic_reference": reference,
        "measurement_interval_reference": {
            "start_altitude_m": measurement_start,
            "end_altitude_m": measurement_end,
            "curves": measurement_curves,
        },
        "canonical_source_and_plane_geometry": canonical_geometry,
        "eligibility": {
            **eligibility_checks,
            "all_required_local_gates_pass": not eligibility_reasons,
            "reason_codes": eligibility_reasons,
            "resample_activity_by_realization": {
                str(case.get("realization_id")): case.get("resample_activity", {}) for case in cases
            },
            "strict_ceiling_activity_basis": (
                "C&D plane mode aborts at projected population >= ceiling; "
                "a completed zero-exit run proves no hit"
            ),
            "max_abs_core_field_perturbation_fraction": max_field_perturbation,
            "multiplication_ci_half_width_fraction": ci_half_width_fraction,
            "primary_ci_half_width_fraction": primary_ci_half_width_fraction,
            "actual_final_reduced_time_s_by_realization": {
                str(case["realization_id"]): case.get("final_reduced_time_s", math.nan)
                for case in cases
            },
            "actual_final_plane_time_s": final_plane_time_s,
            "exact_output_time_tolerance_s": (
                CD_ALL_PHYSICS_10US_OUTPUT_TIME_TOLERANCE_S
            ),
            "attempt_status_by_realization": {
                str(case["realization_id"]): case.get("attempt_status", {}) for case in cases
            },
            "source_injection_counters_by_realization": {
                str(case["realization_id"]): case.get("final_source_injection_counters", {})
                for case in cases
            },
            "late_flux_growth_is_not_an_eligibility_gate": True,
        },
        "late_time_flux_growth_by_realization": {
            str(case["realization_id"]): summary
            for case, summary in zip(cases, late_flux_summaries, strict=True)
        },
        "aggregate_process_context_by_realization": {
            str(case["realization_id"]): case.get("aggregate_process_context", {}) for case in cases
        },
        "steady_state_growth_compatibility": {
            "eligible": steady and actual_steady_fit_eligible,
            "analytic_interval_eligible": reference["steady_state_growth_eligible"],
            "actual_fit_interval_eligible_in_every_realization": (actual_steady_fit_eligible),
            "per_realization_fit_diagnostics": slope_rows,
            "interpretation": (
                "all-physics 10 us slope; high deviations may contain "
                "photon/positron-mediated feedback"
                if steady
                else "not applicable: source-conditioned endpoint-only field"
            ),
        },
    }
    return result_payload


def classify_all_physics_campaign_status(
    *,
    results: list[dict],
    timestep_evidence: dict,
    resource_limit_reason: str | None,
    incomplete_input_count: int,
) -> str:
    """Classify the supplied run directories from their direct physical evidence."""

    if incomplete_input_count:
        return "INCONCLUSIVE_CAMPAIGN_INCOMPLETE"
    if timestep_evidence.get("status") not in {"PASS", "PASS_PARTIAL_RESOURCE_LIMIT"}:
        return "INCONCLUSIVE_TIMESTEP"
    accepted_dt = timestep_evidence.get("accepted_dt_s")
    nonresource_results = []
    for e0 in CD_ALL_PHYSICS_10US_E0_KV_PER_M:
        if e0 == CD_RESOURCE_LIMITED_E0_KV_PER_M and resource_limit_reason:
            continue
        candidates = [
            result
            for result in results
            if result.get("e0_peak_kv_per_m") == e0
            and result.get("dt_s") == accepted_dt
            and result["realization_cohort"]["name"] != CD_COHORT_DIAGNOSTIC
        ]
        if not candidates:
            return "INCONCLUSIVE_CAMPAIGN_INCOMPLETE"
        nonresource_results.append(max(candidates, key=lambda result: (
            int(result.get("seed_count") or 0), int(result.get("realization_count") or 0))))
    if any(
        not result.get("eligibility", {}).get("primary_transit_complete_by_10us", False)
        for result in nonresource_results
    ):
        return "INCONCLUSIVE_PRIMARY_TRANSIT_INCOMPLETE"
    if any(
        result.get("statistical_recommendation", {}).get("action")
        != "STATISTICS_SUFFICIENT"
        for result in nonresource_results
    ):
        return "INCONCLUSIVE_INADEQUATE_STATISTICS"
    if resource_limit_reason:
        return "INCONCLUSIVE_RESOURCE_LIMIT"
    return "COMPLETE_ALL_PHYSICS_10US_CD_DEVIATION_CAMPAIGN"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("run_dirs", nargs="+", type=Path)
    p.add_argument("--tolerance", type=float, default=0.10)
    p.add_argument("--output-json", type=Path)
    p.add_argument("--dt-evidence-output", type=Path)
    p.add_argument(
        "--resource-limit-reason",
        help="Human reason recorded with --resource-limited-e0; required with it.",
    )
    p.add_argument(
        "--resource-limited-e0",
        action="append",
        type=float,
        choices=(CD_RESOURCE_LIMITED_E0_KV_PER_M,),
        default=[],
        help=(
            "Explicitly mark E119 as single-node resource-limited. This permits "
            "the other fields to be analyzed while keeping E119 inconclusive."
        ),
    )
    args = p.parse_args()
    resource_limited_e0 = set(args.resource_limited_e0)
    if bool(resource_limited_e0) != bool(args.resource_limit_reason):
        raise SystemExit(
            "--resource-limited-e0 and --resource-limit-reason are required together"
        )

    parsed_cases = [parse_case(path) for path in args.run_dirs]
    incomplete_input_count = sum(case is None for case in parsed_cases)
    cases = sorted(
        (case for case in parsed_cases if case is not None),
        key=lambda case: (case["e0_kv"], case["dt_s"], case["realization_id"]),
    )
    if not cases:
        raise SystemExit("no cases")
    grouped: dict[tuple, list[dict]] = {}
    for case in cases:
        grouped.setdefault(case["group_signature"], []).append(case)
    results = [
        summarize_group(group, profiles_for_case(group[0]), args.tolerance)
        for _, group in sorted(grouped.items())
    ]
    timestep_results = timestep_summary(results)
    timestep_evidence = build_timestep_evidence(
        timestep_results,
        results,
        resource_limited_e0=resource_limited_e0,
    )
    campaign_status = classify_all_physics_campaign_status(
        results=results,
        timestep_evidence=timestep_evidence,
        resource_limit_reason=args.resource_limit_reason,
        incomplete_input_count=incomplete_input_count,
    )
    payload = {
        "format": CD_ALL_PHYSICS_10US_REPORT_FORMAT,
        "campaign_scope": CD_ALL_PHYSICS_10US_SCOPE,
        "observation_label": CD_ALL_PHYSICS_10US_LABEL,
        "observation_stop_time_s": CD_ALL_PHYSICS_10US_STOP_TIME_S,
        "physics_mode": "all_schema6_production_physics_enabled",
        "observable": "time_integrated_signed_electron_plane_flux_above_1MeV",
        "comparison_policy": "DESCRIPTIVE_276_WKB_REFERENCE_ONLY",
        "avalanche_length_tolerance": args.tolerance,
        "profile_multiplier_deviation": results,
        "campaign_status": campaign_status,
        "campaign_complete": (
            campaign_status == "COMPLETE_ALL_PHYSICS_10US_CD_DEVIATION_CAMPAIGN"
        ),
        "incomplete_input_count": incomplete_input_count,
        "resource_limit_reason": args.resource_limit_reason,
        "resource_limited_e0_peak_kv_per_m": sorted(resource_limited_e0),
        "timestep_convergence": timestep_results,
        "timestep_evidence": timestep_evidence,
        "warnings": [
            (
                "All schema-6 production physics is enabled. This is not a "
                "pure-RREA or feedback-free measurement."
            ),
            (
                "A high-field positive C&D deviation may be physical early "
                "photon/positron-mediated feedback; it is not automatically a code failure."
            ),
            LOW_ENERGY_BREMSSTRAHLUNG_WARNING,
        ],
    }
    encoded = json.dumps(json_safe(payload), indent=2, sort_keys=True, allow_nan=False)
    print(encoded)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(encoded + "\n", encoding="utf-8")
    if args.dt_evidence_output is not None:
        args.dt_evidence_output.parent.mkdir(parents=True, exist_ok=True)
        args.dt_evidence_output.write_text(
            json.dumps(timestep_evidence, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
