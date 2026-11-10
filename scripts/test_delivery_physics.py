#!/usr/bin/env python3
"""Lean analytic delivery-physics and Maxwell observer contracts."""

from __future__ import annotations

import ast
import csv
import gzip
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
from rrea_run_support import C, EPS0
from plot_rrea_electron_creation import load_creation_map, accumulation_panels


def test_creation_density_uses_annular_volume_and_physical_weights(tmp_path):
    path = tmp_path / "electron_creation.bin"
    weights = np.array([[[2*math.pi, 6*math.pi]], [[math.pi, 0.]]], dtype="<f8")
    path.write_bytes(struct.pack("<8s2i7d", b"RREACR1\0", 2, 1,
                               0., 2., 0., 2., 1e6, 1e-3, 6000.) + weights.tobytes())
    data = load_creation_map(path)
    np.testing.assert_allclose(data["density"], [[[1., 1.]], [[.5, 0.]]])
    np.testing.assert_allclose(data["z"], [6000., 6002.])
    assert data["weights"].sum() == pytest.approx(9*math.pi)


def test_creation_intervals_follow_burst_and_partition_unique_weights(tmp_path):
    path = tmp_path / "history.bin.gz"
    with path.open("wb") as out:
        for time_us,total in zip([1,10,11,12,100],[1,2,6,10,12]):
            weights = np.array([[[total-1.,0.]],[[1.,0.]]],dtype="<f8")
            record = struct.pack("<8s2i7dQ",b"RREACR2\0",2,1,
                                 0.,2.,0.,2.,1e6,time_us*1e-6,6000.,0)
            out.write(gzip.compress(record+weights.tobytes()))
    full, intervals = accumulation_panels(path,count=3)
    np.testing.assert_allclose([(a*1e6,b*1e6) for a,b,_ in intervals],
                               [(0,11),(11,12),(12,100)])
    np.testing.assert_allclose([w.sum() for _,_,w in intervals],[6,4,2])
    np.testing.assert_allclose(sum(w for _,_,w in intervals),full["weights"].sum(axis=0))
    _, manual = accumulation_panels(path,bounds_us=[0,10,100])
    np.testing.assert_allclose([w.sum() for _,_,w in manual],[2,10])
    with pytest.raises(ValueError,match="finer"):
        accumulation_panels(path,bounds_us=[0,10,10.1,100])
    with pytest.raises(ValueError,match="recorded time range"):
        accumulation_panels(path,bounds_us=[0,101])
    with path.open("ab") as out:
        changed_grid = struct.pack("<8s2i7dQ",b"RREACR2\0",2,1,
                                   0.,4.,0.,2.,1e6,110e-6,6000.,0)
        out.write(gzip.compress(changed_grid+weights.tobytes()))
    with pytest.raises(ValueError,match="changes grid"):
        accumulation_panels(path)

SCRIPT = Path(__file__).resolve().parent / "compute_rrea_em_radio.py"

NR, NZ = 4, 6
DR = DZ = 100.0
Z_LO = 0.0
DT = 1.0e-6
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


def test_empty_population_gets_a_window_not_a_refusal() -> None:
    # Early in a run there are no photons yet: the panel exists but is empty.
    frames = np.zeros((2, 5, 2, 3), dtype=float)
    frames[:, 0] = 1.0  # energetic electrons
    frames[:, 1] = 1.0  # positive ions
    frames[:, 2] = 2.0  # |E|
    frames[:, 4] = 1.0  # positrons; photons (3) stay zero
    indices = {"electron": 0, "positive_ion": 1, "field": 2,
               "photon": 3, "positron": 4}
    _, density_limits, *_ = compute_limits(
        frames, indices, slice(0, 3),
        electron_smooth_sigma=0.0, field_smooth_sigma=0.0)
    assert density_limits["photon"] == (-3.0, 3.0)
    assert all(np.isfinite(density_limits["positron"]))


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
    # single owner every delivery path reads.
    (capture / "rrea_video" / "video_capture_metadata.json").write_text(
        json.dumps({"altitude_msl_at_z0_m": 0.0}), encoding="utf-8")
    with (capture / "rrea_em_boundary.bin").open("wb") as out:
        out.write(struct.pack("<4q", 0x52454D42, version, NR, NZ))
        out.write(struct.pack("<3d", DR, DZ, Z_LO))
        for k in range(frame_count):
            frame = [0.0] * FRAME_LEN
            frame[0] = k * DT
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


@pytest.mark.parametrize("version", [3, 5])
def test_other_em_boundary_versions_are_rejected(tmp_path, version):
    capture = write_capture(tmp_path, lambda _k, _frame: None, version=version)
    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(capture)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert (
        f"unsupported EM boundary record version {version}; expected 4"
        in result.stderr
    )
    assert not (capture / "rrea_em_on_axis_longitudinal.csv").exists()


@pytest.mark.parametrize("azimuths", [8, 18])
def test_azimuth_check_refines_both_fields(tmp_path, azimuths):
    def current(index, frame):
        frame[O_SIDE_BT] = index * 1e-12
    capture = write_capture(tmp_path, current, frame_count=20)
    output = capture / "observer.csv"
    subprocess.run([
        sys.executable, str(SCRIPT), str(capture), "--observer-offset-km", "1",
        "--observer-altitude-km", "0.3", "--azimuth-samples", str(azimuths),
        "--convergence-check", "--out", str(output)], check=True, capture_output=True)
    metadata = json.loads(output.with_name("observer_metadata.json").read_text())
    first_time = np.loadtxt(output, delimiter=",", skiprows=1, max_rows=1)[0]
    assert first_time == pytest.approx((1000 - NR * DR) / C - DT / 2, abs=1e-20)
    assert metadata["azimuth_samples"] == azimuths
    assert metadata["record_mesh"] == [NR, NZ, DR, DZ]  # the figure's resolved band
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


def test_side_btheta_pulse_is_causal_on_half_step_time_axis(tmp_path):
    ring_j = 3  # r = 400 m, z = 300 m side ring
    first_nonzero_frame = 20

    def mutate(k, frame):
        if first_nonzero_frame <= k <= first_nonzero_frame + 2:
            frame[O_SIDE_BT + ring_j] = 1.0e-9

    rows = run_script(write_capture(tmp_path, mutate))
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


@pytest.mark.parametrize("cloud_result", [False, None, True])
def test_optical_strip_units_values_and_time_axis(monkeypatch, tmp_path, cloud_result):
    import compute_rrea_optical_emissions as optical
    from render_rrea_profiled_video import optical_strip_series

    (tmp_path / "rrea_reduced.csv").touch()
    source = {"t_s": np.array([0.0, 1e-6]), "r337": [2.0, 4.0],
              "swarm": {"note": "swarm unassessed"},
              "r777_mid": [3.0, 6.0]}
    transported = {"t_s": np.array([20e-6, 23e-6]),
                   "irr337": [2000.0, 4000.0], "irr777": [3000.0, 6000.0]}
    monkeypatch.setattr(optical, "optical_series", lambda *a, **kw: source)
    monkeypatch.setattr(optical, "scattered_irradiance",
                        lambda *a, **kw: transported if cloud_result else None)
    times, blue, red, title, unit = optical_strip_series(
        tmp_path, cloud_scattering=cloud_result is not False)
    np.testing.assert_allclose(times, [20.0, 23.0] if cloud_result else [0.0, 1.0])
    np.testing.assert_allclose(blue, [2.0, 4.0])
    np.testing.assert_allclose(red, [3.0, 6.0])
    assert unit == ("mW/m$^2$" if cloud_result else "ph/s")
    assert ("irradiance" if cloud_result else "source rate") in title


@pytest.mark.parametrize("species,density", [("electron", 2.0), ("photon", 5.0)])
def test_band_population_uses_annular_volume_and_reports_selected_edges(species, density):
    from extract_rrea_band_series import band_series
    from rrea_video_stream import VIDEO_V4_FORMAT

    meta = dict(format=VIDEO_V4_FORMAT,
                arrays_per_frame=["n_energetic_e_ge_threshold_m3", "n_positive_ion_m3",
                                  "abs_E_kVpm", "n_photon_ge_threshold_m3",
                                  "n_positron_ge_threshold_m3"],
                nr=3, nz=4, r_min_m=0, r_max_m=6, z_min_m=0, z_max_m=40,
                altitude_msl_at_z0_m=12000)
    frames = np.zeros((1, 5, 4, 3))
    frames[:, 0] = 2.0
    frames[:, 3] = 5.0
    result = band_series(frames, [{"time_s": 1e-6, "step": 400}], meta,
                         species=species, z_lo_m_msl=12010, z_hi_m_msl=12030,
                         r_max_m=5)
    # Two 10-m rows and two full radial cells: radius 4 m, not requested 5 m.
    assert result["n_ge1MeV_real"][0] == pytest.approx(density * math.pi * 4**2 * 20)
    assert (result["band_z_lo_m_msl"], result["band_z_hi_m_msl"],
            result["band_r_max_m"]) == (12010, 12030, 4)
