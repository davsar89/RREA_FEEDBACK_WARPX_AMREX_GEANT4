#!/usr/bin/env python3
"""Direct free-space EM field from recorded Maxwell Huygens data.

Physics: the Maxwell TM run records the SCATTERED tangential and normal fields
(Er, Ez, Btheta) on an inset closed surface every step
(`rrea_em_boundary.bin`). Love's equivalence replaces the interior by surface
sources on that closed surface S:

    J_s = n_hat x H,   M_s = -n_hat x E,   rho_s = eps0 n_hat . E

and the observer field is the retarded (Stratton-Chu / Jefimenko) surface
integral. For an ON-AXIS observer and m = 0 sources every ring has one retarded
time (R is constant on the ring) and only E_z survives:

  E_z(t) = sum over boundary rings of
      (1/4 pi eps0) 2 pi r_s dl (z_o - z_s)/R
          * ( [rho_s]/R^2 + [d_t rho_s]/(c R) )              (charge)
    - (1/4 pi) 2 pi r_s dl (r_s/R)
          * ( [M_theta]/R^2 + [d_t M_theta]/(c R) )          (magnetic)
    - (mu0/4 pi) 2 pi r_s dl [d_t H_theta]/R                 (side J_s = H_theta z_hat)

with [] at t_r = t - R/c (piecewise-linear interpolation and its slope).
The minus sign is (R_hat x M)_z = -(r_s/R) M_theta for an observer on the
axis. E samples are at the integer frame time; leapfrog Btheta samples are at
frame_time - dt/2 and are interpolated on that separate time axis. Legacy v3
labels were one outer step ahead of E; exported times remove that offset.
Radial J_s
rings on the top and bottom faces integrate to zero on axis by symmetry. The
ambient is static and absent from the scattered record.

With a nonzero --observer-offset-km the full vector Love-equivalence surface
integral is evaluated by azimuthal quadrature. No ground image, ionosphere or
instrument gain is applied. Exact-axis radiation from an axisymmetric source is
zero. The surviving E_z is longitudinal; on the axis E_r = B_theta = 0 and
there is no outward Poynting flux. The individual retarded-derivative 1/R
kernels below are therefore NOT, by themselves, a radio observable: for a
closed source they cancel in the far-zone on-axis limit. This script is an
on-axis Coulomb/induction and near-to-far consistency diagnostic. A radio
prediction requires an off-axis observer and the full transverse vector field.

Output CSV: time_s, ez_on_axis_total_v_per_m,
ez_undifferentiated_kernel_v_per_m (the undifferentiated 1/R^2 kernels), and
ez_retarded_derivative_kernel_v_per_m (all retarded time-derivative 1/R
kernels). The last two columns sum to the total; neither is labelled
"radiation" on the symmetry axis.

With --observer-offset-km, the output instead contains Cartesian E, B and
Poynting vectors plus the geometric plate-normal field and optional nominal
unit-gain EFCM fast/slow responses.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import sys
import time
from pathlib import Path

import numpy as np

from compute_rrea_optical_emissions import capture_altitude_z0_m
from compute_rrea_radio_waveform import rc_highpass
from rrea_run_support import C, MU0

MAGIC = 0x52454D42


def read_boundary(path: Path):
    """Memory-map the record as (n_frames, frame_len) float64; a production
    capture is several GB, and the reductions copy their geometry-fixed delay
    window out of the map in blocks (_row_windows)."""
    with path.open("rb") as handle:
        header = handle.read(56)
    if len(header) != 56:
        raise SystemExit(f"{path}: incomplete EM boundary header")
    magic, version, nr, nz = struct.unpack_from("<4q", header, 0)
    if magic != MAGIC:
        raise SystemExit(f"{path}: not an EM boundary record")
    if version not in (3, 4):
        raise SystemExit(
            f"{path}: unsupported EM boundary record version {version}; expected 3 or 4"
        )
    dr, dz, z_lo = struct.unpack_from("<3d", header, 32)
    frame_len = 1 + 4 * (nr + 1) + 2 * nr + nz + 2 * (nz + 1)
    n_frames, trailing_bytes = divmod(path.stat().st_size - 56, frame_len * 8)
    if trailing_bytes:
        print(f"{path}: ignoring {trailing_bytes} bytes of an incomplete trailing frame",
              file=sys.stderr)
    frames = np.memmap(path, dtype="<f8", mode="r", offset=56,
                       shape=(int(n_frames), frame_len))
    # A restarted capture may append overlapping times; keep the last record.
    # As a ROW INDEX, never a gather: the engine only ever appends, so this is
    # the ordinary production case, and materializing it would reinstate the
    # multi-GB copy this reader exists to avoid.
    times = np.asarray(frames[:, 0])
    order = np.argsort(times, kind="stable")
    rows = order[np.append(np.diff(times[order]) > 0.0, True)]
    return version, nr, nz, dr, dz, z_lo, frames, frame_len, rows


def load_observer_series(capture_dir: Path, tag: str, retard_ref_m: float,
                         *, offset_m: float, altitude_msl_m: float):
    """One reduced observer series, moved onto the figure's SOURCE-time axis.

    The surface integral returns OBSERVER time, while every other panel of the
    multiband figure is on source time with the common R/c deliberately
    unapplied. Removing the geometric delay to the observer here is what lets
    the exact and compact-source curves be compared on one axis. Returns None
    when the host-side reduction has not been run for this capture.

    The reduction fixes the observer position hours before the figure is drawn,
    so the sidecar's own metadata is the owner and a geometry the caller does
    not agree with is refused: relabelling a 100 km series as 200 km, shifting
    it by the wrong delay and projecting it with the wrong zenith would all be
    silent.
    """
    base = Path(capture_dir) / "rrea_video" / f"rrea_em_observer_{tag}"
    csv_path = base.with_name(base.name + ".csv")
    meta_path = base.with_name(base.name + "_metadata.json")
    if not (csv_path.is_file() and meta_path.is_file()):
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    for key, want in (("observer_r_m", offset_m),
                      ("observer_altitude_msl_m", altitude_msl_m)):
        got = meta.get(key)
        if got is None or abs(float(got) - want) > 1.0:
            raise SystemExit(
                f"{csv_path.name} was reduced at {key} = {got}, but the figure "
                f"asks for {want:.6g}; rerun the host-side reduction at this "
                "geometry or drop the override")
    table = np.genfromtxt(csv_path, delimiter=",", names=True)
    series = {name: np.atleast_1d(table[name]) for name in table.dtype.names}
    if "boundary_record_time_offset_s" not in meta:
        source_dt = meta.get("dt_source_s")
        if source_dt is None:
            source_dt = 2 * (meta["source_time_max_e_s"] - meta["source_time_max_b_s"])
        series["time_s"] -= source_dt
    series["metadata"] = meta
    series["t_source_s"] = series["time_s"] - retard_ref_m / C
    series["dt_s"] = float(series["time_s"][1] - series["time_s"][0])
    return series


def _progress(label: str, index: int, total: int, started: float) -> None:
    """Periodic stderr heartbeat: a multi-hour reduction must be diagnosable."""
    if index % 512 or not index:
        return
    done = index / total
    left = (time.time() - started) / done * (1.0 - done) / 60.0
    sys.stderr.write(f"\r{label} {done:6.1%} ({index}/{total}), {left:.1f} min left")
    sys.stderr.flush()


_BLOCK = 4096   # output samples per owned row window; tests shrink it


def _row_windows(data, rows, bracket, out_times):
    """Stage chronological interpolation brackets in one reusable RAM slab."""
    bounds = []
    for start in range(0, len(out_times), _BLOCK):
        stop = min(start + _BLOCK, len(out_times))
        k0 = int(bracket(out_times[start])[1].min())
        k1 = int(bracket(out_times[stop - 1])[1].max()) + 2
        bounds.append((start, stop, k0, k1))
    capacity = max((k1 - k0 for _, _, k0, k1 in bounds), default=0)
    buffer = np.empty((capacity, data.shape[1]), dtype=data.dtype)
    for start, stop, k0, k1 in bounds:
        selected = rows[k0:k1]
        breaks = np.r_[0, np.flatnonzero(np.diff(selected) != 1) + 1, selected.size]
        for left, right in zip(breaks[:-1], breaks[1:]):
            physical = int(selected[left])
            buffer[left:right] = data[physical:physical + right - left]
        yield start, stop, k0, buffer[:k1 - k0]


def off_axis_fields(
    *,
    frames,
    rows=None,
    nr: int,
    nz: int,
    dr: float,
    dz: float,
    z_lo: float,
    observer_r_m: float,
    observer_z_m: float,
    azimuth_samples: int,
    time_stride: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    """Full vector Love-equivalence field at one off-axis observer.

    Surface geometry and retarded kernel coefficients are built once. The
    time loop then performs only piecewise-linear retarded interpolation and
    vector reductions. This deliberately transparent implementation is the
    reference for any FFT-convolution accelerator.
    """
    if azimuth_samples < 8 or azimuth_samples % 2:
        raise ValueError("azimuth_samples must be an even integer >= 8")
    if time_stride < 1:
        raise ValueError("time_stride must be positive")
    data = np.asarray(frames)
    rows = np.arange(data.shape[0]) if rows is None else rows
    times = data[rows, 0]
    if times.size < 3:
        raise ValueError("at least three Huygens frames are required")
    dt = float(times[1] - times[0])
    z_hi = z_lo + nz * dz
    r_side = nr * dr
    if observer_r_m <= r_side and z_lo <= observer_z_m <= z_hi:
        raise ValueError("observer must lie outside the closed Huygens surface")

    off = _boundary_record_offsets(nr, nz)

    # kind, offset, r, z, dl, sign, basis.  rho multiplies the scalar normal
    # field, and rho_s = eps0 E_n cancels Jefimenko's 1/(4 pi eps0), so its
    # kernel carries NO eps0; J uses raw Btheta (Htheta=Btheta/mu0, mu0
    # cancels likewise); M uses tangential E.
    elements: list[tuple[str, int, float, float, float, float, str]] = []
    for i in range(nr + 1):
        r = i * dr
        w = dr * (0.5 if i in (0, nr) else 1.0)
        if r > 0.0:
            elements.extend((
                ("J", off["top_bt"] + i, r, z_hi, w, -1.0, "er"),
                ("J", off["bot_bt"] + i, r, z_lo, w, +1.0, "er"),
                ("M", off["top_er"] + i, r, z_hi, w, -1.0, "ephi"),
                ("M", off["bot_er"] + i, r, z_lo, w, +1.0, "ephi"),
            ))
    for i in range(nr):
        r = (i + 0.5) * dr
        elements.extend((
            ("rho", off["top_ez"] + i, r, z_hi, dr, +1.0, "none"),
            ("rho", off["bot_ez"] + i, r, z_lo, dr, -1.0, "none"),
        ))
    for j in range(nz):
        z = z_lo + (j + 0.5) * dz
        elements.append(("rho", off["side_er"] + j, r_side, z, dz, +1.0, "none"))
    for j in range(nz + 1):
        z = z_lo + j * dz
        w = dz * (0.5 if j in (0, nz) else 1.0)
        elements.extend((
            ("M", off["side_ez"] + j, r_side, z, w, +1.0, "ephi"),
            ("J", off["side_bt"] + j, r_side, z, w, +1.0, "ez"),
        ))

    phi = 2.0 * np.pi * np.arange(azimuth_samples) / azimuth_samples
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    er_basis = np.column_stack((cos_phi, sin_phi, np.zeros_like(phi)))
    ephi_basis = np.column_stack((-sin_phi, cos_phi, np.zeros_like(phi)))
    ez_basis = np.tile(np.array((0.0, 0.0, 1.0)), (azimuth_samples, 1))
    observer = np.array((observer_r_m, 0.0, observer_z_m))

    offsets: list[np.ndarray] = []
    is_b: list[np.ndarray] = []
    delays: list[np.ndarray] = []
    e_value: list[np.ndarray] = []
    e_derivative: list[np.ndarray] = []
    b_value: list[np.ndarray] = []
    b_derivative: list[np.ndarray] = []
    for kind, offset, r, z, dl, sign, basis_name in elements:
        source = np.column_stack((r * cos_phi, r * sin_phi, np.full_like(phi, z)))
        displacement = observer - source
        distance = np.linalg.norm(displacement, axis=1)
        rhat = displacement / distance[:, None]
        area_over_4pi = r * dl / (2.0 * azimuth_samples)
        ev = np.zeros_like(rhat)
        ed = np.zeros_like(rhat)
        bv = np.zeros_like(rhat)
        bd = np.zeros_like(rhat)
        if kind == "rho":
            ev = area_over_4pi * sign * rhat / distance[:, None] ** 2
            ed = area_over_4pi * sign * rhat / (C * distance[:, None])
        else:
            basis = {"er": er_basis, "ephi": ephi_basis, "ez": ez_basis}[basis_name]
            if kind == "J":
                cross = np.cross(basis, rhat)
                ed = -area_over_4pi * sign * basis / distance[:, None]
                bv = area_over_4pi * sign * cross / distance[:, None] ** 2
                bd = area_over_4pi * sign * cross / (C * distance[:, None])
            else:
                cross = np.cross(rhat, basis)
                ev = area_over_4pi * sign * cross / distance[:, None] ** 2
                ed = area_over_4pi * sign * cross / (C * distance[:, None])
                bd = -area_over_4pi * sign * basis / (C * C * distance[:, None])
        offsets.append(np.full(azimuth_samples, offset, dtype=np.int64))
        is_b.append(np.full(azimuth_samples, kind == "J", dtype=bool))
        delays.append(distance / C)
        e_value.append(ev)
        e_derivative.append(ed)
        b_value.append(bv)
        b_derivative.append(bd)

    offsets_a = np.concatenate(offsets)
    is_b_a = np.concatenate(is_b)
    delays_a = np.concatenate(delays)
    ev_a = np.concatenate(e_value)
    ed_a = np.concatenate(e_derivative)
    bv_a = np.concatenate(b_value)
    bd_a = np.concatenate(b_derivative)
    del offsets, is_b, delays, e_value, e_derivative, b_value, b_derivative
    sample_t0 = np.where(is_b_a, times[0] - 0.5 * dt, times[0])
    sample_t1 = np.where(is_b_a, times[-1] - 0.5 * dt, times[-1])
    arrival_times = sample_t0 + delays_a
    observer_start = float(np.min(arrival_times))
    observer_end = float(np.min(sample_t1 + delays_a))
    count = math.floor((observer_end - observer_start) / (dt * time_stride) + 1e-12)
    out_times = observer_start + dt * time_stride * np.arange(count + 1)
    e_out = np.zeros((out_times.size, 3), dtype=np.float64)
    b_out = np.zeros_like(e_out)

    def bracket(t):
        x = np.minimum((t - delays_a - sample_t0) / dt, rows.size - 1.0)
        return x, np.clip(np.floor(x).astype(np.int64), 0, rows.size - 2)

    started = time.time()
    for start, stop, k0, window in _row_windows(data, rows, bracket, out_times):
        for out_index in range(start, stop):
            _progress("off-axis", out_index, out_times.size, started)
            x, k = bracket(out_times[out_index])
            valid = out_times[out_index] >= arrival_times
            fraction = np.clip(x - k, 0.0, 1.0)
            v0 = window[k - k0, offsets_a]
            v1 = window[k - k0 + 1, offsets_a]
            value = np.where(valid, v0 + fraction * (v1 - v0), 0.0)
            derivative = np.where(valid, (v1 - v0) / dt, 0.0)
            e_out[out_index] = value @ ev_a + derivative @ ed_a
            b_out[out_index] = value @ bv_a + derivative @ bd_a
    metadata = {
        "dt_source_s": dt,
        "dt_observer_s": dt * time_stride,
        "azimuth_samples": float(azimuth_samples),
        "surface_element_count": float(len(elements)),
        "quadrature_node_count": float(offsets_a.size),
        "surface_to_observer_delay_min_s": float(np.min(delays_a)),
        "surface_to_observer_delay_max_s": float(np.max(delays_a)),
    }
    return out_times, e_out, b_out, metadata


def _boundary_record_offsets(nr: int, nz: int) -> dict[str, int]:
    """Element offsets of one frame, as AppendMaxwellBoundaryRecord writes it.

    One owner for the on-disk layout the C++ writer defines: two readers
    drifting apart would be a silent misread, not an error.  Element 0 is
    time_s, so the first face starts at 1, and each entry below is the block
    SIZE that advances the cursor to the next face.
    """
    blocks = (("top_er", nr + 1), ("top_ez", nr), ("top_bt", nr + 1),
              ("bot_er", nr + 1), ("bot_ez", nr), ("bot_bt", nr + 1),
              ("side_er", nz), ("side_ez", nz + 1), ("side_bt", nz + 1))
    offsets, cursor = {}, 1
    for name, size in blocks:
        offsets[name] = cursor
        cursor += size
    return offsets


def on_axis_longitudinal_fields(
    *,
    frames,
    rows=None,
    times,
    bt_times,
    nr: int,
    nz: int,
    dr: float,
    dz: float,
    z_lo: float,
    observer_z_m: float,
    dt: float,
    time_stride: int = 1,
):
    """Exact-axis longitudinal E_z(t) from the closed inset Huygens surface.

    Love-equivalence sources per face ring, specialized to the axis: surface
    charge rho_s = eps0 n.E (top/bottom Ez with n = +/-z, side Er with n = +r)
    contributes the undifferentiated Coulomb kernel
    rho_s (z-z_s)/(4 pi eps0 R^3) plus the retarded-derivative kernel
    (drho/dt) (z-z_s)/(4 pi eps0 c R^2); magnetic surface current
    M_s = -n x E (top: -Er theta_hat, bottom: +Er theta_hat, side Ez:
    +Ez theta_hat) enters through (R_hat x theta_hat)_z = -r_s/R with the
    same 1/R^2 + (d/dt)/(c R) split; the side electric current
    J_s = r_hat x H = H_theta z_hat contributes only the radiation term
    -mu0 (dJ_s/dt)/(4 pi R); radial J_s rings on the z faces cancel on the
    axis by symmetry.  Ring weights are trapezoid for node-centred samples
    (endpoints half) and full intervals for midpoint samples; the i = 0 ring
    has zero area and is skipped.  Retarded times interpolate linearly
    inside a sample interval, are zero before the record (the fields ARE
    zero before the run), and post-record extrapolation is FORBIDDEN: the
    observer series ends at the last time every ring's retarded time is
    still inside its own E or B record (B lives on its own half-step axis,
    bt_times).

    Exact-axis m = 0 radiation is identically zero, so this is the
    longitudinal field-change observable, NOT a radio waveform
    (off_axis_fields is).  Returns (observer_times, rows, meta): rows of
    (t, ez_total, ez_undifferentiated, ez_retarded_derivative), meta with
    the surface-to-observer delay range.
    """
    z_hi = z_lo + nz * dz
    r_side = nr * dr
    z_obs = observer_z_m

    off = _boundary_record_offsets(nr, nz)

    # Per-ring source list: (face, offset, r_s, z_s, dl). Ring radii: Er/Bt
    # samples sit at r=i*dr (the i=0 ring has zero area and is skipped), while
    # Ez sits at (i+1/2)dr. Node-centred sets include both endpoints and use
    # trapezoid weights; midpoint sets use the full interval.
    rings = []
    for i in range(nr + 1):
        r_s = i * dr
        w = dr * (0.5 if i in (0, nr) else 1.0)
        if r_s > 0.0:
            rings.append(("top_bt", off["top_bt"] + i, r_s, z_hi, w))
            rings.append(("bot_bt", off["bot_bt"] + i, r_s, z_lo, w))
            rings.append(("top_er", off["top_er"] + i, r_s, z_hi, w))
            rings.append(("bot_er", off["bot_er"] + i, r_s, z_lo, w))
    for i in range(nr):
        r_s = (i + 0.5) * dr
        rings.append(("top_ez", off["top_ez"] + i, r_s, z_hi, dr))
        rings.append(("bot_ez", off["bot_ez"] + i, r_s, z_lo, dr))
    for j in range(nz):
        rings.append(
            ("side_er", off["side_er"] + j, r_side, z_lo + (j + 0.5) * dz, dz)
        )
    for j in range(nz + 1):
        w = dz * (0.5 if j in (0, nz) else 1.0)
        rings.append(("side_ez", off["side_ez"] + j, r_side, z_lo + j * dz, w))
        rings.append(("side_bt", off["side_bt"] + j, r_side, z_lo + j * dz, w))

    ring_distances = [
        math.hypot(r_s, z_obs - z_s) for _, _, r_s, z_s, _ in rings
    ]
    delay_min_s = min(ring_distances) / C
    delay_max_s = max(ring_distances) / C

    # The source record is known to be zero before its first sample, so the
    # observer series may begin at the simulation clock origin. After the
    # record, however, no continuation is known. Stop at the latest observer
    # time for which EVERY ring's retarded time remains inside its own E or B
    # sample interval. B is half a step earlier than E.
    observer_supported_end_s = min(
        (bt_times[-1] if face.endswith("_bt") else times[-1])
        + r_geo / C
        for (face, _, _, _, _), r_geo in zip(rings, ring_distances)
    )
    observer_step_count = int(
        math.floor((observer_supported_end_s - times[0]) / dt + 1.0e-12)
    )
    observer_times = [times[0] + k * dt
                      for k in range(0, observer_step_count + 1, time_stride)]

    # Per-ring kernel coefficients, built once: ez_total = sum(c_val f +
    # c_ddt df/dt) and the reported derivative column is sum(c_ddt df/dt).
    # rho_s = eps0 (n.E) cancels the Coulomb 1/(4 pi eps0) and H = B/mu0
    # cancels the mu0 of the dJ/dt term, so neither constant appears here.
    # Radial J_s rings on the z faces cancel on axis and keep zero
    # coefficients; they stay in the list so the supported-window bound above
    # still sees their B time axis.
    data = np.asarray(frames)
    rows = np.arange(len(times)) if rows is None else rows
    ring_offsets = np.array([o for _, o, _, _, _ in rings], dtype=np.int64)
    r_geo = np.asarray(ring_distances, dtype=np.float64)
    on_b_axis = np.array([face.endswith("_bt") for face, _, _, _, _ in rings])
    c_val = np.zeros(len(rings))
    c_ddt = np.zeros(len(rings))
    for index, ((face, _, r_s, z_s, dl), r_g) in enumerate(zip(rings, r_geo)):
        ring = 0.5 * r_s * dl                      # 2 pi r_s dl / (4 pi)
        if face == "side_er" or face in ("top_ez", "bot_ez"):
            # rho_s = eps0 (n_hat . E): +-Ez on the z faces, Er on the side.
            geom = (-1.0 if face.startswith("bot") else 1.0) * (z_obs - z_s) / r_g
        elif face in ("top_er", "bot_er", "side_ez"):
            # M_s = -n_hat x E, with (R_hat x theta_hat)_z = -r_s/R.
            geom = (-1.0 if face.startswith("top") else 1.0) * (-r_s / r_g)
        elif face == "side_bt":
            # J_s = r_hat x H = H_theta z_hat: the -mu0 d_t J_s / R term only.
            c_ddt[index] = -ring / r_g
            continue
        else:
            continue
        c_val[index] = ring * geom / (r_g * r_g)
        c_ddt[index] = ring * geom / (C * r_g)

    # Linear interpolation on each ring's own axis; zero before the record (the
    # fields ARE zero before the run) and never extrapolated after it, which
    # would invent a boundary source and an artificial pulse tail.
    axis_t0 = np.where(on_b_axis, bt_times[0], times[0])
    arrival_times = axis_t0 + r_geo / C
    n_samples = len(times)

    def bracket(t):
        x = np.clip((t - r_geo / C - axis_t0) / dt, 0.0, n_samples - 1.0)
        return x, np.minimum(x.astype(np.int64), n_samples - 2)

    started = time.time()
    out_rows = []
    for start, stop, k0, window in _row_windows(data, rows, bracket, observer_times):
        for index, t in enumerate(observer_times[start:stop], start):
            _progress("on-axis", index, len(observer_times), started)
            x, k = bracket(t)
            f0 = window[k - k0, ring_offsets]
            f1 = window[k - k0 + 1, ring_offsets]
            live = t >= arrival_times
            slope = np.where(live, (f1 - f0) / dt, 0.0)
            value = np.where(live, f0 + (x - k) * (f1 - f0), 0.0)
            ez_value, ez_derivative = float(c_val @ value), float(c_ddt @ slope)
            out_rows.append((t, ez_value + ez_derivative, ez_value, ez_derivative))

    return observer_times, out_rows, {
        "surface_to_observer_delay_min_s": delay_min_s,
        "surface_to_observer_delay_max_s": delay_max_s,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("capture_dir", type=Path)
    parser.add_argument(
        "--observer-altitude-km",
        type=float,
        default=20.0,
        help="observer altitude MSL (ER-2 default)",
    )
    parser.add_argument(
        "--z0-altitude-msl-km",
        type=float,
        default=None,
        help=(
            "altitude of domain z=0; default from the capture's run json "
            "when present"
        ),
    )
    parser.add_argument(
        "--observer-offset-km", type=float, default=0.0,
        help="horizontal observer offset; nonzero selects full off-axis vector fields",
    )
    parser.add_argument("--azimuth-samples", type=int, default=256)
    parser.add_argument("--time-stride", type=int, default=1)
    parser.add_argument("--convergence-check", action="store_true")
    parser.add_argument(
        "--efcm-nominal", action="store_true",
        help="append unit-gain nominal 100 us / 150 ms EFCM high-pass channels",
    )
    parser.add_argument(
        "--plate-azimuth-deg", type=float, default=0.0,
        help="horizontal plate-normal azimuth from the source-observer plane",
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    # One owner for what altitude z = 0 sits at: the capture metadata the
    # figure and the optical chain already read. A second scan here read a
    # differently named key from a different file and simply never found it.
    z0_msl_km = capture_altitude_z0_m(
        args.capture_dir,
        None if args.z0_altitude_msl_km is None
        else args.z0_altitude_msl_km * 1000.0,
    ) / 1000.0

    version, nr, nz, dr, dz, z_lo, frames, _, rows = read_boundary(
        args.capture_dir / "rrea_em_boundary.bin"
    )
    times = np.asarray(frames[rows, 0], dtype=np.float64)
    if times.size < 3:
        raise SystemExit("fewer than 3 boundary frames; nothing to integrate")
    dt = float(times[1] - times[0])
    if not (math.isfinite(dt) and dt > 0.0):
        raise SystemExit("boundary record times must be finite and increasing")
    steps = np.diff(times)
    if not np.all(np.isfinite(steps)) or np.any(
        np.abs(steps - dt) > 1.0e-9 * dt
    ):
        raise SystemExit(
            "boundary record must have one uniform time step for retarded "
            "interpolation"
        )

    bt_times = times - 0.5 * dt
    z_hi = z_lo + nz * dz
    z_obs = (args.observer_altitude_km - z0_msl_km) * 1000.0
    r_side = nr * dr
    r_obs = abs(args.observer_offset_km) * 1000.0
    if r_obs <= r_side and z_lo <= z_obs <= z_hi:
        raise SystemExit(
            "the observer lies inside the closed Huygens surface; "
            "the exterior Love-equivalence integral used here is not valid"
        )

    print(
        f"record v{version}; mesh {nr}x{nz} dr={dr} dz={dz}; "
        f"observer at z={z_obs:.0f} m "
        f"domain-frame ({args.observer_altitude_km} km MSL, z0 at "
        f"{z0_msl_km} km); {len(times)} frames dt={dt:.3e}s"
    )
    time_offset = -dt if version == 3 else 0.0

    if r_obs > 0.0:
        try:
            out_times, e_vec, b_vec, off_axis_meta = off_axis_fields(
                frames=frames, rows=rows,
                nr=nr, nz=nz, dr=dr, dz=dz, z_lo=z_lo,
                observer_r_m=r_obs, observer_z_m=z_obs,
                azimuth_samples=args.azimuth_samples,
                time_stride=args.time_stride,
            )
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        quadrature_error = quadrature_b_error = None
        if args.convergence_check:
            fine_t, fine_e, fine_b, _ = off_axis_fields(
                frames=frames, rows=rows,
                nr=nr, nz=nz, dr=dr, dz=dz, z_lo=z_lo,
                observer_r_m=r_obs, observer_z_m=z_obs,
                azimuth_samples=2 * args.azimuth_samples, time_stride=args.time_stride,
            )
            tolerance = 32 * np.spacing(np.max(np.abs(out_times)))
            if fine_t.shape != out_times.shape or not np.allclose(
                    fine_t, out_times, rtol=0.0, atol=tolerance):
                raise SystemExit("azimuth convergence timelines differ")
            quadrature_error, quadrature_b_error = (
                float(np.max(np.linalg.norm(base - fine, axis=1)))
                / max(float(np.max(np.linalg.norm(fine, axis=1))), 1.0e-300)
                for base, fine in ((e_vec, fine_e), (b_vec, fine_b)))
        out_times += time_offset
        poynting = np.cross(e_vec, b_vec) / MU0
        plate_angle = math.radians(args.plate_azimuth_deg)
        plate_field = (
            math.cos(plate_angle) * e_vec[:, 0]
            + math.sin(plate_angle) * e_vec[:, 1]
        )
        dt_observer = float(out_times[1] - out_times[0]) if out_times.size > 1 else 0.0
        if args.efcm_nominal and dt_observer > 0.0:
            efcm_fast = rc_highpass(plate_field, 100.0e-6, dt_observer)
            efcm_slow = rc_highpass(plate_field, 150.0e-3, dt_observer)
        else:
            efcm_fast = np.full_like(plate_field, np.nan)
            efcm_slow = np.full_like(plate_field, np.nan)
        out_path = args.out or (args.capture_dir / "rrea_em_off_axis.csv")
        matrix = np.column_stack((
            out_times, e_vec, b_vec, poynting,
            plate_field, efcm_fast, efcm_slow))
        np.savetxt(
            out_path, matrix, delimiter=",", comments="",
            header=("time_s,ex_v_per_m,ey_v_per_m,ez_v_per_m,"
                    "bx_T,by_T,bz_T,sx_w_per_m2,sy_w_per_m2,sz_w_per_m2,"
                    "plate_normal_e_v_per_m,efcm_fast_nominal_v_per_m,"
                    "efcm_slow_nominal_v_per_m"),
        )
        # First-order Silver-Mueller absorption is imperfect; the native smoke
        # measures the residual, and late-time waveform content carries it.
        absorber_note = (
            "first-order open-boundary absorber; late-time content may carry "
            "residual reflection")
        print(f"scope: {absorber_note}", file=sys.stderr)
        metadata_path = out_path.with_name(out_path.stem + "_metadata.json")
        metadata_path.write_text(json.dumps({
            "format": "rrea_em_off_axis_vector_v1",
            "boundary_record_time_offset_s": time_offset,
            "observer_r_m": r_obs,
            "observer_z_domain_m": z_obs,
            "observer_altitude_msl_m": args.observer_altitude_km * 1000.0,
            "free_space_direct_field": True,
            "ground_reflection": False,
            "ionosphere": False,
            "instrument_calibration": "none",
            "absorber_residual_note": absorber_note,
            "nominal_efcm_enabled": args.efcm_nominal,
            "plate_azimuth_deg": args.plate_azimuth_deg,
            "azimuth_convergence_relative_e_error": quadrature_error,
            "azimuth_convergence_relative_b_error": quadrature_b_error,
            **off_axis_meta,
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote {out_path} and {metadata_path} ({len(out_times)} samples)")
        return 0

    observer_times, out_rows, on_axis_meta = on_axis_longitudinal_fields(
        frames=frames, rows=rows, times=times, bt_times=bt_times,
        nr=nr, nz=nz, dr=dr, dz=dz, z_lo=z_lo,
        observer_z_m=z_obs, dt=dt, time_stride=args.time_stride,
    )
    delay_min_s = on_axis_meta["surface_to_observer_delay_min_s"]
    delay_max_s = on_axis_meta["surface_to_observer_delay_max_s"]
    observer_times = np.asarray(observer_times) + time_offset

    print(
        "SCOPE: this is the on-axis LONGITUDINAL field (what an overhead "
        "field mill / nadir field-change meter sees). Propagating m=0 "
        "radiation on the symmetry axis is identically zero, so this is NOT "
        "a radio waveform; that needs an off-axis vector integral.",
        file=sys.stderr,
    )
    print(
        f"surface-to-observer delay range {delay_min_s:.3e}.."
        f"{delay_max_s:.3e}s; fully supported observer interval "
        f"{observer_times[0]:.3e}..{observer_times[-1]:.3e}s"
    )

    out_path = args.out or (
        args.capture_dir / "rrea_em_on_axis_longitudinal.csv"
    )
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(
            "time_s,ez_on_axis_total_v_per_m,"
            "ez_undifferentiated_kernel_v_per_m,"
            "ez_retarded_derivative_kernel_v_per_m\n"
        )
        for t, total, undifferentiated, derivative in out_rows:
            handle.write(
                f"{t + time_offset:.9e},{total:.9e},{undifferentiated:.9e},"
                f"{derivative:.9e}\n"
            )

    metadata_path = out_path.with_name(out_path.stem + "_metadata.json")
    metadata_path.write_text(
        json.dumps(
            {
                "format": "rrea_em_on_axis_longitudinal_v1",
                "boundary_record_time_offset_s": time_offset,
                "observer_geometry": "exact_axis",
                "observer_r_m": 0.0,
                "observer_altitude_msl_m": args.observer_altitude_km * 1000.0,
                "field_split": "scattered_only_static_ambient_omitted",
                "propagating_radio_observable": False,
                "interpretation": (
                    "longitudinal Coulomb/induction and closed-surface "
                    "consistency diagnostic; exact-axis m=0 radiation is zero"
                ),
                "source_time_min_s": times[0] + time_offset,
                "source_time_max_e_s": times[-1] + time_offset,
                "source_time_max_b_s": bt_times[-1] + time_offset,
                "observer_time_min_s": observer_times[0],
                "observer_time_supported_max_s": observer_times[-1],
                "surface_to_observer_delay_min_s": delay_min_s,
                "surface_to_observer_delay_max_s": delay_max_s,
                "post_record_extrapolation": "forbidden",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    peak = max((abs(v) for _, v, _, _ in out_rows), default=0.0)
    print(
        f"wrote {out_path} and {metadata_path} "
        f"({len(out_rows)} samples, peak |Ez| {peak:.3e} V/m)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
