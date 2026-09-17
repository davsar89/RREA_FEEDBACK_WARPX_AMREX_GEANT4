#!/usr/bin/env python3
"""Lean analytic delivery-physics and Maxwell observer contracts."""

from __future__ import annotations

import ast
import csv
import importlib
import inspect
import json
import math
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from render_rrea_profiled_video import (
    compute_limits,
    field_profile_curves,
    field_profile_x_max,
    shared_strip_time_limits_us,
)

SCRIPT = Path(__file__).resolve().parent / "compute_rrea_em_radio.py"

NR, NZ = 4, 6
DR = DZ = 100.0
Z_LO = 0.0
DT = 1.0e-6
C = 299792458.0
EPS0 = 8.8541878128e-12
FRAME_LEN = 1 + 4 * (NR + 1) + 2 * NR + NZ + 2 * (NZ + 1)
O_TOP_ER = 1
O_TOP_EZ = O_TOP_ER + (NR + 1)
O_TOP_BT = O_TOP_EZ + NR
O_BOT_ER = O_TOP_BT + (NR + 1)
O_BOT_EZ = O_BOT_ER + (NR + 1)
O_BOT_BT = O_BOT_EZ + NR
O_SIDE_ER = O_BOT_BT + (NR + 1)
O_SIDE_EZ = O_SIDE_ER + NZ
O_SIDE_BT = O_SIDE_EZ + (NZ + 1)

DELIVERY_CALLERS = (
    "compute_rrea_optical_emissions.py",
    "plot_rrea_radio_multiband.py",
    "render_rrea_profiled_video.py",
)
DELIVERY_API_MODULES = {
    "compute_rrea_em_radio",
    "compute_rrea_optical_emissions",
    "compute_rrea_radio_waveform",
    "extract_rrea_band_series",
    "rrea_cloud_scattering",
}


def test_field_profile_mean_axis_slice_and_shared_scale() -> None:
    field = np.array([
        [1.0, 3.0, 5.0],
        [10.0, 4.0, 1.0],
    ])
    radial_mean, axis_slice = field_profile_curves(
        field,
        symmetric_radius_display=False,
    )
    np.testing.assert_allclose(radial_mean, [3.0, 5.0])
    np.testing.assert_array_equal(axis_slice, [1.0, 10.0])
    assert field_profile_x_max(radial_mean, axis_slice) == pytest.approx(15.0)

    mirrored = np.concatenate((field[:, ::-1], field), axis=1)
    symmetric_mean, symmetric_axis = field_profile_curves(
        mirrored,
        symmetric_radius_display=True,
    )
    np.testing.assert_allclose(symmetric_mean, radial_mean)
    np.testing.assert_array_equal(symmetric_axis, axis_slice)


def test_field_profile_scale_covers_late_axis_peak() -> None:
    frames = np.zeros((2, 3, 2, 3), dtype=float)
    frames[:, 0] = 1.0  # energetic electrons
    frames[:, 1] = 1.0  # positive ions
    frames[0, 2] = 2.0  # initial |E|
    frames[1, 2] = np.array([[3.0, 2.0, 1.0], [10.0, 2.0, 1.0]])
    indices = {
        "electron": 0,
        "positive_ion": 1,
        "field": 2,
        "photon": None,
        "positron": None,
    }
    *_, profile_limit = compute_limits(
        frames,
        indices,
        slice(0, 3),
        electron_smooth_sigma=0.0,
        field_smooth_sigma=0.0,
    )
    assert profile_limit == pytest.approx(15.0)


def test_bottom_strip_time_limits_use_union_of_records() -> None:
    assert shared_strip_time_limits_us(
        np.array([0.0, 40.0, 102.0]),
        np.array([67.0, 80.0, 101.0]),
    ) == (0.0, 102.0)


def write_capture(
    tmp_path: Path,
    mutate,
    *,
    version: int = 4,
    frame_count: int = 80,
) -> Path:
    capture = tmp_path / "capture"
    (capture / "rrea_video").mkdir(parents=True)
    # The real capture layout: z0 lives in the video metadata, which is the
    # single owner every delivery path reads. A fixture carrying a key no
    # production capture writes is how the reduction came to fail on one.
    (capture / "rrea_video" / "video_capture_metadata.json").write_text(
        json.dumps({"altitude_msl_at_z0_m": 0.0}), encoding="utf-8")
    with (capture / "rrea_em_boundary.bin").open("wb") as out:
        out.write(struct.pack("<4q", 0x52454D42, version, NR, NZ))
        out.write(struct.pack("<3d", DR, DZ, Z_LO))
        for k in range(frame_count):
            frame = [0.0] * FRAME_LEN
            frame[0] = (k + (version == 3)) * DT
            mutate(k, frame)
            out.write(struct.pack(f"<{FRAME_LEN}d", *frame))
    return capture


def run_script(
    capture: Path,
    *,
    observer_altitude_km: float = 5.0,
) -> list[dict[str, str]]:
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(capture),
            "--observer-altitude-km",
            f"{observer_altitude_km:.17g}",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    with (capture / "rrea_em_on_axis_longitudinal.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        return list(csv.DictReader(handle))


@pytest.mark.parametrize("caller_name", DELIVERY_CALLERS)
def test_delivery_call_sites_match_imported_function_signatures(caller_name):
    """Catch removed/renamed arguments before a normal figure command runs."""
    caller_path = SCRIPT.parent / caller_name
    tree = ast.parse(caller_path.read_text(encoding="utf-8"), filename=caller_name)
    imported_functions = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or node.module not in DELIVERY_API_MODULES:
            continue
        module = importlib.import_module(node.module)
        for alias in node.names:
            target = getattr(module, alias.name, None)
            if inspect.isfunction(target):
                imported_functions[alias.asname or alias.name] = target

    errors = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        target = imported_functions.get(node.func.id)
        if target is None:
            continue
        # Dynamic *args/**kwargs cannot be proved statically; all ordinary
        # production calls are checked.
        if any(isinstance(arg, ast.Starred) for arg in node.args) or any(
            keyword.arg is None for keyword in node.keywords
        ):
            continue
        try:
            inspect.signature(target).bind(
                *([None] * len(node.args)),
                **{keyword.arg: None for keyword in node.keywords},
            )
        except TypeError as exc:
            errors.append(f"line {node.lineno}: {node.func.id}: {exc}")
    assert not errors, f"{caller_name} has stale delivery API calls: {errors}"


@pytest.mark.parametrize("version", [1, 2])
def test_legacy_em_boundary_versions_are_rejected(tmp_path, version):
    capture = write_capture(tmp_path, lambda _k, _frame: None, version=version)
    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(capture)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert (
        f"unsupported EM boundary record version {version}; expected 3 or 4"
        in result.stderr
    )
    assert not (capture / "rrea_em_on_axis_longitudinal.csv").exists()


@pytest.mark.parametrize("azimuths", [8, 18])
@pytest.mark.parametrize("version", [3, 4])
def test_azimuth_check_refines_both_fields(tmp_path, azimuths, version):
    def current(index, frame):
        frame[O_SIDE_BT] = index * 1e-12
    capture = write_capture(tmp_path, current, frame_count=20, version=version)
    output = capture / "observer.csv"
    subprocess.run([
        sys.executable, str(SCRIPT), str(capture), "--observer-offset-km", "1",
        "--observer-altitude-km", "0.3", "--azimuth-samples", str(azimuths),
        "--convergence-check", "--out", str(output)], check=True, capture_output=True)
    metadata = json.loads(output.with_name("observer_metadata.json").read_text())
    first_time = np.loadtxt(output, delimiter=",", skiprows=1, max_rows=1)[0]
    assert first_time == pytest.approx((1000 - NR * DR) / C - DT / 2, abs=1e-20)
    assert metadata["azimuth_samples"] == azimuths
    for field in ("e", "b"):
        error = metadata[f"azimuth_convergence_relative_{field}_error"]
        assert math.isfinite(error) and error > 0.0


def test_static_ring_charge_matches_coulomb(tmp_path):
    ez_surface = 3.0e4
    ring_index = 1  # r_s = 150 m on the top face
    r_s = (ring_index + 0.5) * DR
    z_s = Z_LO + NZ * DZ
    z_obs = 5000.0

    def mutate(_k, frame):
        frame[O_TOP_EZ + ring_index] = ez_surface

    rows = run_script(write_capture(tmp_path, mutate))
    charge = EPS0 * ez_surface * 2.0 * math.pi * r_s * DR
    r_geo = math.hypot(r_s, z_obs - z_s)
    expected = charge * (z_obs - z_s) / (
        4.0 * math.pi * EPS0 * r_geo**3
    )
    late = rows[-1]
    assert float(late["ez_on_axis_total_v_per_m"]) == pytest.approx(
        expected, rel=1e-9
    )
    assert abs(
        float(late["ez_retarded_derivative_kernel_v_per_m"])
    ) < abs(expected) * 1e-9


def test_charge_retardation_kernel_is_in_derivative_column(tmp_path):
    ez_step = 2.0e3
    ring_index = 1
    r_s = (ring_index + 0.5) * DR
    z_s = Z_LO + NZ * DZ
    z_obs = 5000.0

    def mutate(k, frame):
        frame[O_TOP_EZ + ring_index] = ez_step * k

    rows = run_script(write_capture(tmp_path, mutate))
    output_index = 30
    r_geo = math.hypot(r_s, z_obs - z_s)
    cos_alpha = (z_obs - z_s) / r_geo
    ring_over_4pi = 0.5 * r_s * DR
    expected_derivative = (
        ring_over_4pi * cos_alpha * (ez_step / DT) / (C * r_geo)
    )
    row = rows[output_index]
    assert float(row["ez_retarded_derivative_kernel_v_per_m"]) == pytest.approx(
        expected_derivative, rel=2e-9
    )
    assert float(row["ez_on_axis_total_v_per_m"]) == pytest.approx(
        float(row["ez_undifferentiated_kernel_v_per_m"])
        + expected_derivative,
        rel=2e-9,
    )


def test_static_top_er_has_love_magnetic_current_sign(tmp_path):
    er_surface = 2.0e4
    ring_index = 1  # tangential-E node at r_s = 100 m
    r_s = ring_index * DR
    z_s = Z_LO + NZ * DZ
    z_obs = 5000.0

    def mutate(_k, frame):
        frame[O_TOP_ER + ring_index] = er_surface

    rows = run_script(write_capture(tmp_path, mutate))
    r_geo = math.hypot(r_s, z_obs - z_s)
    ring_over_4pi = 0.5 * r_s * DR
    # Top M_theta = -Er and (R_hat x theta_hat)_z = -r_s/R, so
    # their product is positive above the top face.
    expected = ring_over_4pi * r_s * er_surface / r_geo**3
    late = rows[-1]
    assert float(late["ez_on_axis_total_v_per_m"]) == pytest.approx(
        expected, rel=1e-9
    )
    assert float(late["ez_on_axis_total_v_per_m"]) > 0.0
    assert abs(
        float(late["ez_retarded_derivative_kernel_v_per_m"])
    ) < abs(expected) * 1e-9


@pytest.mark.parametrize("version", [3, 4])
def test_side_btheta_pulse_is_causal_on_half_step_time_axis(tmp_path, version):
    ring_j = 3  # r = 400 m, z = 300 m side ring
    first_nonzero_frame = 20

    def mutate(k, frame):
        if first_nonzero_frame <= k <= first_nonzero_frame + 2:
            frame[O_SIDE_BT + ring_j] = 1.0e-9

    rows = run_script(write_capture(tmp_path, mutate, version=version))
    r_geo = math.hypot(NR * DR, 5000.0 - (Z_LO + ring_j * DZ))
    # Piecewise-linear B begins changing one interval before its first nonzero
    # sample.  Sample k is at (k-1/2)dt, hence the front is (k-3/2)dt.
    source_front = (first_nonzero_frame - 1.5) * DT
    t_arrival = source_front + r_geo / C
    early = [
        abs(float(row["ez_on_axis_total_v_per_m"]))
        for row in rows
        if float(row["time_s"]) < t_arrival
    ]
    late = [
        abs(float(row["ez_on_axis_total_v_per_m"]))
        for row in rows
        if float(row["time_s"]) >= t_arrival
    ]
    assert max(early, default=0.0) == 0.0
    assert max(late, default=0.0) > 0.0
    first_response = min(float(row["time_s"]) for row in rows
                         if abs(float(row["ez_on_axis_total_v_per_m"])) > 0.0)
    assert t_arrival <= first_response <= t_arrival + DT


def test_btheta_derivative_uses_leapfrog_half_step_timestamp(tmp_path):
    ring_j = 1
    z_s = Z_LO + ring_j * DZ
    r_side = NR * DR
    output_index = 40
    target_retarded_time = 19.75 * DT
    target_distance = output_index * DT * C - target_retarded_time * C
    z_obs = z_s + math.sqrt(target_distance**2 - r_side**2)
    b_scale = 1.0e-13

    def mutate(k, frame):
        frame[O_SIDE_BT + ring_j] = b_scale * k * k

    rows = run_script(
        write_capture(tmp_path, mutate),
        observer_altitude_km=z_obs / 1000.0,
    )
    # At t_r=19.75 dt, the half-step axis brackets samples k=20 and 21,
    # giving slope (21^2-20^2)*b_scale/dt = 41*b_scale/dt.  Treating B as
    # integer-time data would instead select the k=19..20 interval (39).
    d_b_dt = 41.0 * b_scale / DT
    ring_over_4pi = 0.5 * r_side * DZ
    expected = -ring_over_4pi * d_b_dt / target_distance
    row = rows[output_index]
    assert float(row["time_s"]) == pytest.approx(output_index * DT)
    assert float(row["ez_on_axis_total_v_per_m"]) == pytest.approx(
        expected, rel=2e-9
    )
    assert float(row["ez_retarded_derivative_kernel_v_per_m"]) == pytest.approx(
        expected, rel=2e-9
    )


def test_on_axis_scope_is_stated_but_does_not_block_the_run(tmp_path):
    # Exact-axis m=0 radiation is identically zero, so the output must never
    # be presented as a radio waveform.  The honesty lives in the naming, a
    # loud scope line, and the metadata -- not in refusing to compute.
    capture = write_capture(tmp_path, lambda _k, _frame: None)
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(capture),
            "--observer-altitude-km",
            "5",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "identically zero" in result.stderr
    assert "NOT" in result.stderr
    assert (capture / "rrea_em_on_axis_longitudinal.csv").exists()


def test_observer_series_extends_beyond_source_record_without_extrapolation(
    tmp_path,
):
    last_frame = 9

    def mutate(k, frame):
        # A static top-face charge is enough to prove that delayed source
        # history remains present after the source-record clock ends.
        if k >= 1:
            frame[O_TOP_EZ + 1] = 2.0e3

    capture = write_capture(
        tmp_path,
        mutate,
        version=3,
        frame_count=last_frame + 1,
    )
    rows = run_script(capture)
    source_end = last_frame * DT
    observer_end = float(rows[-1]["time_s"])
    assert observer_end > source_end
    assert float(rows[-1]["ez_on_axis_total_v_per_m"]) != 0.0

    metadata = json.loads(
        (capture / "rrea_em_on_axis_longitudinal_metadata.json").read_text(
            encoding="utf-8"
        )
    )
    assert metadata["propagating_radio_observable"] is False
    assert metadata["post_record_extrapolation"] == "forbidden"
    assert metadata["observer_time_supported_max_s"] == pytest.approx(
        observer_end
    )
    assert metadata["observer_time_supported_max_s"] > source_end


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))


def test_optical_strip_unit_travels_with_its_title() -> None:
    # The strip's y unit is part of the series, so a cloudscat irradiance
    # panel can never be labelled ph/s again.  Runs on a pulled capture when
    # one is present; cloudscat may be unavailable, in which case the
    # function falls back to source rates and the same check applies.
    from render_rrea_profiled_video import optical_strip_series
    captures = sorted((Path(__file__).resolve().parents[1] / "tmp"
                       / "hpc_captures").glob("*/rrea_reduced.csv"))
    if not captures:
        pytest.skip("no pulled capture under tmp/hpc_captures")
    for cloud in (False, True):
        strip = optical_strip_series(captures[-1].parent, cloud_scattering=cloud)
        assert strip is not None and len(strip) == 5
        _, _, _, title, unit = strip
        assert ("irradiance" in title) == (unit == "mW/m$^2$")
        assert ("source rate" in title) == (unit == "ph/s")
