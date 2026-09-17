"""Independent symmetry, causality and energy checks for radio analysis."""

from __future__ import annotations

import math
import struct

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

import json

import compute_rrea_em_radio
from compute_rrea_em_radio import (C, MAGIC, load_observer_series,
                                   off_axis_fields,
                                   on_axis_longitudinal_fields, read_boundary)
from compute_rrea_radio_waveform import EPS0, rc_highpass, spectral_energy_density
from plot_rrea_radio_multiband import (
    draw_scattered_overlay,
    ground_theta_field,
    efcm_plate_field,
    lip_vertical_field,
    shared_source_time_limits_us,
)


def _boundary_frames(nr: int = 2, nz: int = 2, count: int = 20, dt: float = 1e-8):
    frame_len = 1 + 4 * (nr + 1) + 2 * nr + nz + 2 * (nz + 1)
    frames = np.zeros((count, frame_len))
    frames[:, 0] = np.arange(count) * dt
    return frames


def test_axisymmetric_off_axis_field_respects_observer_plane_symmetry() -> None:
    nr = nz = 2
    frames = _boundary_frames(nr, nz)
    o_side_bt = 1 + 4 * (nr + 1) + 2 * nr + nz + (nz + 1)
    frames[:, o_side_bt:o_side_bt + nz + 1] = frames[:, 0, None] / 1e-8 * 1e-12
    times, electric, magnetic, _ = off_axis_fields(
        frames=[tuple(row) for row in frames], nr=nr, nz=nz,
        dr=10.0, dz=10.0, z_lo=0.0,
        observer_r_m=100.0, observer_z_m=15.0,
        azimuth_samples=32, time_stride=1)
    assert times[0] >= 80.0 / C - 0.5e-8 - 1e-15
    scale_e = max(np.max(np.abs(electric)), 1e-300)
    scale_b = max(np.max(np.abs(magnetic)), 1e-300)
    assert np.max(np.abs(electric[:, 1])) < 1e-12 * scale_e
    assert np.max(np.abs(magnetic[:, (0, 2)])) < 1e-12 * scale_b


def test_off_axis_static_point_charge_matches_coulomb() -> None:
    """Love integral of an interior static charge reproduces its Coulomb field.

    Btheta = 0, so this independently exercises the rho and M kernels that
    symmetry-only cases do not constrain.
    """
    nr = nz = 16
    dr = dz = 10.0
    k = 1.0e-3 / (4.0 * math.pi * EPS0)  # q/(4 pi eps0) of a 1 mC charge
    z_q = 0.5 * nz * dz                  # on the axis, at the surface centre
    frames = _boundary_frames(nr, nz, count=8, dt=1e-6)
    # Block starts, re-derived rather than imported: two readers of the layout
    # drifting apart is what this file exists to catch.
    o = np.cumsum([1, nr + 1, nr, nr + 1, nr + 1, nr, nr + 1, nz, nz + 1])

    def coulomb(r, z):
        d = np.hypot(r, z - z_q)
        return k * r / d**3, k * (z - z_q) / d**3

    node_r, cell_r = np.arange(nr + 1) * dr, (np.arange(nr) + 0.5) * dr
    for z_f, er, ez in ((nz * dz, o[0], o[1]), (0.0, o[3], o[4])):
        frames[:, er:er + nr + 1] = coulomb(node_r, z_f)[0]
        frames[:, ez:ez + nr] = coulomb(cell_r, z_f)[1]
    frames[:, o[6]:o[6] + nz] = coulomb(nr * dr, (np.arange(nz) + 0.5) * dz)[0]
    frames[:, o[7]:o[7] + nz + 1] = coulomb(nr * dr, np.arange(nz + 1) * dz)[1]

    r_obs = 600.0
    _t, electric, magnetic, _meta = off_axis_fields(
        frames=[tuple(row) for row in frames], nr=nr, nz=nz, dr=dr, dz=dz,
        z_lo=0.0, observer_r_m=r_obs, observer_z_m=z_q,
        azimuth_samples=256, time_stride=1)
    expected = k / r_obs**2
    assert abs(electric[-1][0] / expected - 1.0) < 5e-3
    assert abs(electric[-1][2]) < 1e-12 * expected
    assert float(np.max(np.abs(magnetic))) < 1e-12 * expected


def test_dynamic_dipole_fields_converge_to_the_analytic_exterior_solution():
    """Hertz dipole: value, induction and radiation terms in E; curl(A) in B."""
    def dipole(radius, height, time):
        distance = np.hypot(radius, height)
        radial, vertical = radius / distance, height / distance
        phase = (time - distance / C - 100e-9) / 20e-9
        moment = 1e-9 * np.exp(-phase**2)
        first = -2 * phase * moment / 20e-9
        second = (4 * phase**2 - 2) * moment / (20e-9)**2
        near = moment / distance**3 + first / (C * distance**2)
        far = second / (C**2 * distance)
        scale = 1 / (4 * np.pi * EPS0)
        return (scale * radial * vertical * (3 * near + far),
                scale * ((3 * vertical**2 - 1) * near + (vertical**2 - 1) * far),
                scale / C**2 * radial * (first / distance**2 + second / (C * distance)))

    errors = []
    for radial_cells, dt in ((16, 1e-9), (32, 0.5e-9)):
        axial_cells, spacing = 2 * radial_cells, 16.0 / radial_cells
        frames = _boundary_frames(radial_cells, axial_cells, round(400e-9 / dt) + 1, dt)
        offsets = np.cumsum([1, radial_cells + 1, radial_cells, radial_cells + 1,
                            radial_cells + 1, radial_cells, radial_cells + 1,
                            axial_cells, axial_cells + 1])
        electric_time, magnetic_time = frames[:, :1], frames[:, :1] - dt / 2
        nodes = np.arange(radial_cells + 1) * spacing
        cells = (np.arange(radial_cells) + 0.5) * spacing
        for height, start in ((16.0, 0), (-16.0, 3)):
            frames[:, offsets[start]:offsets[start + 1]] = dipole(nodes, height, electric_time)[0]
            frames[:, offsets[start + 1]:offsets[start + 2]] = dipole(cells, height, electric_time)[1]
            frames[:, offsets[start + 2]:offsets[start + 3]] = dipole(nodes, height, magnetic_time)[2]
        heights = np.arange(axial_cells + 1) * spacing - 16
        frames[:, offsets[6]:offsets[7]] = dipole(16.0, heights[:-1] + spacing / 2, electric_time)[0]
        frames[:, offsets[7]:offsets[8]] = dipole(16.0, heights, electric_time)[1]
        frames[:, offsets[8]:] = dipole(16.0, heights, magnetic_time)[2]
        times, electric, magnetic, _ = off_axis_fields(
            frames=frames, nr=radial_cells, nz=axial_cells, dr=spacing, dz=spacing,
            z_lo=0.0, observer_r_m=60.0, observer_z_m=40.0,
            azimuth_samples=4 * radial_cells, time_stride=1)
        exact_r, exact_z, exact_b = dipole(60.0, 24.0, times)
        expected_e = np.column_stack((exact_r, np.zeros_like(times), exact_z))
        expected_b = np.column_stack((np.zeros_like(times), exact_b, np.zeros_like(times)))
        errors.append([np.max(np.linalg.norm(actual - expected, axis=1))
                       / np.max(np.linalg.norm(expected, axis=1))
                       for actual, expected in ((electric, expected_e), (magnetic, expected_b))])
    assert np.max(errors[0]) < 0.02
    assert np.max(errors[1]) < 0.006
    assert np.all(np.array(errors[1]) < np.array(errors[0]) / 2)


def test_off_axis_azimuth_quadrature_converges() -> None:
    nr = nz = 2
    frames = _boundary_frames(nr, nz)
    o_side_bt = 1 + 4 * (nr + 1) + 2 * nr + nz + (nz + 1)
    frames[:, o_side_bt:o_side_bt + nz + 1] = frames[:, 0, None] / 1e-8 * 1e-12
    args = dict(
        frames=[tuple(row) for row in frames], nr=nr, nz=nz,
        dr=10.0, dz=10.0, z_lo=0.0,
        observer_r_m=100.0, observer_z_m=15.0, time_stride=1)
    times128, e128, _b128, _ = off_axis_fields(azimuth_samples=128, **args)
    times256, e256, _b256, _ = off_axis_fields(azimuth_samples=256, **args)
    np.testing.assert_allclose(times128, times256, rtol=0.0, atol=1e-15)
    scale = max(float(np.max(np.linalg.norm(e256, axis=1))), 1e-300)
    assert float(np.max(np.linalg.norm(e256 - e128, axis=1))) / scale < 5e-3


@pytest.mark.parametrize("count", [127, 128])
def test_physical_spectral_energy_uses_unwindowed_parseval(count) -> None:
    dt = 2.5e-9
    pulse = np.zeros(count)
    pulse[37] = 1.0
    for signal in (pulse, (-1.0) ** np.arange(count), np.ones(count),
                   np.sin(2 * np.pi * 7 * np.arange(count) / count)):
        frequency, density = spectral_energy_density(signal, dt, "none")
        spectral = np.sum(density) * (frequency[1] - frequency[0])
        y = signal - np.mean(signal)
        time_domain = C * 8.8541878128e-12 * np.sum(y * y) * dt
        np.testing.assert_allclose(spectral, time_domain, rtol=1e-12)


def test_nominal_instrument_highpass_has_no_dc_gain() -> None:
    signal = np.ones(1000)
    response = rc_highpass(signal, 100e-6, 1e-6)
    assert np.array_equal(response, np.zeros_like(response))


def test_shared_source_time_limits_use_union_of_records() -> None:
    limits = shared_source_time_limits_us(
        np.array([67.0, 80.0, 101.0]),
        np.array([0.0, 28.0, 100.0]),
        np.array([np.nan, 10.0, 102.0]),
    )
    assert limits == (0.0, 102.0)


def test_efcm_plate_reads_exactly_zero_at_nadir() -> None:
    # The plate reads sin(theta)cos(theta)(3N + S): for a vertical dipole at
    # nadir the null is structural (azimuthal symmetry), so it must be EXACT
    # for any current moment, not merely small.
    m = np.sin(np.linspace(0.0, 40.0, 512)) * 1.7e6
    plate = efcm_plate_field(m, 2.5e-9, 8.0e3, 0.0)
    assert np.array_equal(plate, np.zeros_like(plate))


def test_lip_field_mill_keeps_the_quasi_static_step() -> None:
    # A TGF transfers charge moment dp = integral M dt; a 50 S/s field mill
    # records the standing step 2 dp / (4 pi eps0 R^3).  Any high-pass in
    # this path (the y[0] = 0 recursion) would decay the step to zero, so the
    # pin is the closed-form late-time value, held flat.
    dt, r_m = 1.0e-6, 8.0e3
    m = np.zeros(4096)
    m[100:200] = 3.0e5
    lip = lip_vertical_field(m, dt, r_m, 0.0)
    dp = float(np.trapezoid(m, dx=dt))
    expected = 2.0 * dp / (4.0 * math.pi * EPS0 * r_m**3)
    assert abs(lip[-1] - expected) < 1e-6 * abs(expected)
    assert abs(lip[-1] - lip[3000]) < 1e-9 * abs(expected)


def test_scattered_overlay_lives_on_shifted_arrival_axis() -> None:
    # Draw CloudScat on its own absolute-arrival grid shifted by the geometric
    # delay, without resampling to the reduced timeline.
    fig, ax = plt.subplots()
    try:
        t_s = np.linspace(0.0, 400e-6, 137)  # != any reduced-timeline length
        flux = np.exp(-0.5 * ((t_s - 150e-6) / 40e-6) ** 2)
        scattered = {"t_s": t_s, "irr337_lo": flux, "irr337_hi": 2.0 * flux,
                     "irr777": 0.3 * flux}
        delay_us = 25.0
        fraction, end_us = draw_scattered_overlay(
            ax, scattered, (0.0, 200.0), delay_us)
        np.testing.assert_allclose(
            ax.lines[0].get_xdata(), t_s * 1e6 - delay_us)
        # Shifted Gaussian centre 125 us, sigma 40 us: mass inside [0, 200]
        # is Phi(1.875) - Phi(-3.125).
        expected = 0.5 * (math.erf(1.875 / math.sqrt(2.0))
                          - math.erf(-3.125 / math.sqrt(2.0)))
        assert abs(fraction - expected) < 5e-3
        # The panel's own axis ends where 90% of the fluence has arrived
        # (125 + 1.2816 sigma = 176 us), never shorter than the shared window.
        assert end_us == 200.0
        _, end_us = draw_scattered_overlay(
            ax, scattered, (0.0, 100.0), delay_us)
        assert abs(end_us - (125.0 + 1.2816 * 40.0)) < 4.0
        # Drawn in mW/m^2 from a uW/m^2 series.
        np.testing.assert_allclose(ax.lines[0].get_ydata(), flux * 1e-3)
    finally:
        plt.close(fig)


def _ramped_side_bt(nr=2, nz=2, count=20, dt=1e-8):
    frames = _boundary_frames(nr, nz, count, dt)
    o_side_bt = 1 + 4 * (nr + 1) + 2 * nr + nz + (nz + 1)
    frames[:, o_side_bt:o_side_bt + nz + 1] = frames[:, 0, None] / dt * 1e-12
    return frames


def _pack_record(path, frames, nr=2, nz=2, dr=10.0, dz=10.0, z_lo=0.0):
    path.write_bytes(struct.pack("<4q3d", MAGIC, 3, nr, nz, dr, dz, z_lo)
                     + np.ascontiguousarray(frames).tobytes())


@pytest.mark.parametrize("block", [1, 17, 4096])
def test_row_window_blocks_reproduce_the_whole_record_gather(monkeypatch, block):
    # The window copy is a pure data-movement change: every block size must
    # give the same field as the single-window run on both surface paths.
    frames = _ramped_side_bt(count=4103 if block == 4096 else 20)
    dt = 1e-8
    off = dict(frames=frames, nr=2, nz=2, dr=10.0, dz=10.0, z_lo=0.0,
               observer_r_m=100.0, observer_z_m=15.0, azimuth_samples=32,
               time_stride=1)
    on = dict(frames=frames, times=frames[:, 0], bt_times=frames[:, 0] - 0.5 * dt,
              nr=2, nz=2, dr=10.0, dz=10.0, z_lo=0.0, observer_z_m=100.0, dt=dt)
    monkeypatch.setattr(compute_rrea_em_radio, "_BLOCK", len(frames) + 1000)
    whole = off_axis_fields(**off), on_axis_longitudinal_fields(**on)
    monkeypatch.setattr(compute_rrea_em_radio, "_BLOCK", block)
    blocked = off_axis_fields(**off), on_axis_longitudinal_fields(**on)
    for reference, candidate in zip(whole[0][:3], blocked[0][:3]):
        np.testing.assert_allclose(candidate, reference, rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(np.array(blocked[1][1]), np.array(whole[1][1]),
                               rtol=1e-12, atol=0.0)


def test_restart_overlap_keeps_the_later_record(tmp_path) -> None:
    # A restarted capture appends frames whose times repeat the tail of the
    # record before it. The reader's row index must select the later copies,
    # and the windowed gather must read through that index, so the reduced
    # field equals the one from a clean record.
    clean = _ramped_side_bt(count=5)
    stale = clean[2:4].copy()
    stale[:, 1:] *= 0.5
    _pack_record(tmp_path / "rrea_em_boundary.bin",
                 np.vstack((clean[:2], stale, clean[2:])))
    *_, frames, _, rows = read_boundary(tmp_path / "rrea_em_boundary.bin")
    assert rows.tolist() == [0, 1, 4, 5, 6]
    args = dict(nr=2, nz=2, dr=10.0, dz=10.0, z_lo=0.0, observer_r_m=100.0,
                observer_z_m=15.0, azimuth_samples=32, time_stride=1)
    times, e_dedup, b_dedup, _ = off_axis_fields(frames=frames, rows=rows, **args)
    times_clean, e_clean, b_clean, _ = off_axis_fields(frames=clean, **args)
    np.testing.assert_allclose(times, times_clean, rtol=0.0, atol=1e-18)
    np.testing.assert_array_equal(e_dedup, e_clean)
    np.testing.assert_array_equal(b_dedup, b_clean)


def test_source_windows_reuse_one_owned_allocation(monkeypatch, tmp_path):
    path = tmp_path / "source.bin"
    np.arange(90, dtype=np.float64).tofile(path)
    data = np.memmap(path, dtype=np.float64, mode="r", shape=(9, 10))
    rows = np.array([2, 5, 6, 1, 7, 8])
    expected = [20, 50, 60, 10, 70, 80]
    monkeypatch.setattr(compute_rrea_em_radio, "_BLOCK", 1)
    previous = None
    for start, stop, origin, window in compute_rrea_em_radio._row_windows(
            data, rows, lambda time: (time, np.array([int(time)])), np.arange(5)):
        assert (stop, origin) == (start + 1, start)
        assert window.flags.c_contiguous and window.base.shape == (2, 10)
        np.testing.assert_array_equal(window[:, 0], expected[start:start + 2])
        assert not np.shares_memory(window, data)
        if previous is not None:
            assert np.shares_memory(previous, window)
        previous = window


@pytest.mark.parametrize("count,dt", [(21, 1e-8), (101, 1e-10)])
def test_off_axis_supported_end_keeps_a_constant_current(count, dt):
    frames = _boundary_frames(count=count, dt=dt)
    frames[:, -3:-1] = 1e-12
    times, _, magnetic, _ = off_axis_fields(
        frames=frames, nr=2, nz=2, dr=10.0, dz=10.0, z_lo=0.0,
        observer_r_m=100.0, observer_z_m=15.0, azimuth_samples=8, time_stride=4)
    first = min(80.0 / C, math.hypot(80.0, 5.0) / C - dt / 2)
    assert abs(times[0] - first) < 1e-20
    assert times[-1] <= first + (count - 1) * dt + 1e-20
    azimuth = np.arange(8) * math.pi / 4
    for index in (0, -1):
        expected = 0.0
        for height, width in ((0, 5), (10, 10)):
            distance = np.sqrt((100 - 20 * np.cos(azimuth))**2
                               + (20 * np.sin(azimuth))**2 + (15 - height)**2)
            live = times[index] >= distance / C - dt / 2
            expected += np.sum(1e-12 * 20 * width / 16
                               * (100 - 20 * np.cos(azimuth[live])) / distance[live]**3)
        assert magnetic[index, 1] == pytest.approx(expected, rel=1e-12, abs=1e-28)


def test_off_axis_side_current_is_causal_on_the_half_step_axis() -> None:
    # Mirrors the on-axis pin: a side ring's B ramp B_k = s max(k-10, 0),
    # with sample k at (k-1/2)dt, first changes slope at 9.5 dt, so the field
    # is exactly zero before 9.5 dt + R_min/c and nonzero from the first
    # sample after. The observer height is chosen so that sample lies 0.34 dt
    # past the front, inside the half step an integer-time B would still read
    # as zero. Once every azimuth node is past the ramp start the derivative
    # kernel alone survives: E_z = -(s/dt) sum_nodes (r dl / 2N) / R_node.
    nr = nz = 2
    dr = dz = 10.0
    dt, count, slope, n_phi = 1e-8, 60, 1e-12, 32
    frames = _boundary_frames(nr, nz, count, dt)
    o_side_bt = 1 + 4 * (nr + 1) + 2 * nr + nz + (nz + 1)
    frames[:, o_side_bt] = slope * np.maximum(np.arange(count) - 10, 0)
    r_obs, z_obs = 100.0, 18.0
    times, electric, _b, meta = off_axis_fields(
        frames=frames, nr=nr, nz=nz, dr=dr, dz=dz, z_lo=0.0,
        observer_r_m=r_obs, observer_z_m=z_obs, azimuth_samples=n_phi,
        time_stride=1)

    r_side = nr * dr
    nearest = math.hypot(r_obs - r_side, z_obs - nz * dz)     # a B node
    assert times[0] == pytest.approx(-0.5 * dt + nearest / C, abs=1e-18)
    assert times[-1] <= (count - 1.5) * dt + nearest / C + 1e-18

    phi = 2.0 * np.pi * np.arange(n_phi) / n_phi
    ring_r = np.hypot(np.hypot(r_obs - r_side * np.cos(phi), r_side * np.sin(phi)),
                      z_obs)
    front = 9.5 * dt + ring_r.min() / C
    after = times[times > front][0]
    assert front < after < front + 0.5 * dt
    assert np.all(electric[times < front] == 0.0)
    assert electric[times == after][0, 2] != 0.0

    plateau = times >= 9.5 * dt + ring_r.max() / C
    assert np.count_nonzero(plateau) > 10
    expected = -(slope / dt) * np.sum(r_side * (0.5 * dz) / (2 * n_phi) / ring_r)
    np.testing.assert_allclose(electric[plateau, 2], expected, rtol=1e-12)
    assert np.all(electric[:, 0] == 0.0)


def _write_observer(capture, tag, times, columns, *, offset_m=0.0,
                    altitude_msl_m=0.0):
    """A reduced observer sidecar, in the layout the host-side job writes."""
    video = capture / "rrea_video"
    video.mkdir(parents=True, exist_ok=True)
    names = list(columns)
    header = ",".join(["time_s"] + names)
    rows = np.column_stack([times] + [columns[n] for n in names])
    np.savetxt(video / f"rrea_em_observer_{tag}.csv", rows, delimiter=",",
               header=header, comments="")
    (video / f"rrea_em_observer_{tag}_metadata.json").write_text(
        json.dumps({"format": "rrea_em_off_axis_vector_v1",
                    "boundary_record_time_offset_s": 0.0,
                    "observer_r_m": offset_m,
                    "observer_altitude_msl_m": altitude_msl_m}),
        encoding="utf-8")


def test_observer_series_lands_on_the_source_time_axis(tmp_path) -> None:
    # The integral reports OBSERVER time; every panel of the figure is on
    # source time with the common R/c deliberately unapplied. A missed shift
    # would silently slide the ground panels by R/c = 333 us at 100 km.
    reference_m, offset_m = 100.8e3, 100e3
    delay_s = reference_m / C
    times = delay_s + np.arange(64) * 1.0e-8
    _write_observer(tmp_path, "ground", times,
                    {"ex_v_per_m": np.ones(64), "ez_v_per_m": np.zeros(64)},
                    offset_m=offset_m)

    series = load_observer_series(tmp_path, "ground", reference_m,
                                  offset_m=offset_m, altitude_msl_m=0.0)
    np.testing.assert_allclose(series["t_source_s"], times - delay_s, atol=1e-15)
    assert series["dt_s"] == pytest.approx(1.0e-8)
    assert series["t_source_s"][0] == pytest.approx(0.0, abs=1e-12)
    assert load_observer_series(tmp_path, "aircraft", reference_m,
                                offset_m=offset_m, altitude_msl_m=0.0) is None
    path = tmp_path / "rrea_video/rrea_em_observer_ground_metadata.json"
    meta = json.loads(path.read_text())
    del meta["boundary_record_time_offset_s"]
    meta["dt_source_s"] = 2.5e-9
    path.write_text(json.dumps(meta))
    legacy = load_observer_series(tmp_path, "ground", reference_m,
                                  offset_m=offset_m, altitude_msl_m=0.0)
    assert legacy["t_source_s"][0] == pytest.approx(-2.5e-9, abs=1e-15)


def test_observer_series_refuses_a_geometry_it_was_not_reduced_at(tmp_path):
    # The reduction fixes the observer hours before the figure is drawn, so a
    # --distance-km override must not relabel, re-delay and re-project a series
    # computed somewhere else. The sidecar's own metadata is the owner.
    times = 3.36e-4 + np.arange(8) * 1.0e-8
    _write_observer(tmp_path, "ground", times,
                    {"ex_v_per_m": np.ones(8), "ez_v_per_m": np.zeros(8)},
                    offset_m=100e3, altitude_msl_m=0.0)
    with pytest.raises(SystemExit, match="observer_r_m"):
        load_observer_series(tmp_path, "ground", 200.8e3,
                             offset_m=200e3, altitude_msl_m=0.0)


def test_ground_theta_projection_matches_the_dipole_convention() -> None:
    # The figure must project the Cartesian observer field onto the SAME
    # component the dipole terms produce, or the exact and compact-source
    # ground curves are different physical quantities. Drive the real helper:
    # a field built as pure theta_hat must come back as its own amplitude, and
    # a pure r_hat field must project to zero.
    zenith = math.radians(97.2)                     # ground receiver, below
    r_hat = np.array((math.sin(zenith), 0.0, math.cos(zenith)))
    theta_hat = np.array((math.cos(zenith), 0.0, -math.sin(zenith)))
    field = 3.0 * theta_hat - 7.0 * r_hat
    series = {"ex_v_per_m": np.array([field[0], r_hat[0]]),
              "ez_v_per_m": np.array([field[2], r_hat[2]])}
    projected = ground_theta_field(series, zenith)
    assert projected[0] == pytest.approx(3.0)
    assert projected[1] == pytest.approx(0.0, abs=1e-12)
