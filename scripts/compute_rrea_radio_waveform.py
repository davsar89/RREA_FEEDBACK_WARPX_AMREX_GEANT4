"""Far-field radio (sferic) waveform from a profiled-video capture directory.

Method (Dwyer & Cummer, JGR 118, 3769, 2013, doi:10.1002/jgra.50188; repo
copy docs/papers/tgf_radio_dwyer_cummer.pdf): for the azimuthally symmetric
RZ source only the vertical current moment M(t) = integral Jz dV [A*m] radiates, and
the radiation field of a compact vertical current element is

    E_theta(R, theta, t) = mu0 * sin(theta) / (4 pi R) * dM/dt  (retarded),

doubled for the perfectly conducting ground image when the receiver is at
the ground (their section 8).  dipole_field_terms() below carries that term
together with the near-field pair and the E_r component, so a near-field
consumer (the ALOFT EFCM panels of plot_rrea_radio_multiband.py) shares one
implementation with the far-field one.

The current capture contract records the current moments every diagnostic
interval in global_jz_kinetic_e_A_m (all kinetic electrons),
global_jz_positron_A_m, and global_jz_fluid_A_m (sigma*Ez conduction,
including low-energy electron and both ion drift channels). M(t) is
maxwell_jz_moment_A_m, the moment Ampere's law is actually sourced from
(chord-averaged kinetic + fluid face current, averaged over substeps), and
falls back to the sum of those three columns when it is absent. The companion
global_jz_kinetic_e_ge_1MeV_A_m column is the >= 1 MeV runaway moment, used
for N_re and Omega. All four columns are required and must be finite.

M(t) is the moment of the TRUNCATED source: an electron absorbed at the top
boundary stops contributing, so an escape burst injects a spurious dM/dt.
Valid while the avalanche stays in the box; netchord_escaped_charge_e bounds it.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from rrea_video_contract import dedup_reduced_rows_by_step
from rrea_run_support import (  # SI constants have one owner
    C as C_LIGHT, ELEMENTARY_CHARGE as ELEMENTARY_CHARGE_C, EPS0, MU0)

# Dwyer & Cummer reference values used to normalize this run against an
# average TGF: the RREA front speed (Coleman & Dwyer 2006, their section 10.4),
# the runaway count of a 13 km source (section 10.3, k*N_TGF = 8.8e19 at
# k = 220 m), and the total runaway path length Omega (section 10.1).
BETA_AVALANCHE = 0.89
DC_N_TGF = 4.0e17
DC_OMEGA_M = 2.2e20

EXACT_JZ_COLUMNS = (
    "global_jz_kinetic_e_A_m",
    "global_jz_kinetic_e_ge_1MeV_A_m",
    "global_jz_positron_A_m",
    "global_jz_fluid_A_m",
)

# Local CSV conversion for the radio-waveform inputs.
def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise SystemExit(f"missing input file: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"{path} has no data rows")
    return rows


def as_float(row: dict[str, str], key: str, default: float = math.nan) -> float:
    raw = (row.get(key) or "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def load_reduced(path: Path) -> dict:
    ordered = dedup_reduced_rows_by_step(read_csv_rows(path), source=path)
    time_s = np.array([as_float(r, "time_s") for r in ordered])
    missing = [key for key in EXACT_JZ_COLUMNS if key not in ordered[0]]
    if missing:
        raise SystemExit(
            "rrea_reduced.csv is missing required exact current-moment columns: "
            + ", ".join(missing)
        )
    out: dict = {
        "time_s": time_s,
        "n_ge1MeV": np.nan_to_num(
            np.array([as_float(r, "N_kinetic_e_ge_1MeV_real", 0.0) for r in ordered])
        ),
        "n_kinetic": np.nan_to_num(
            np.array([as_float(r, "N_kinetic_e_real", 0.0) for r in ordered])
        ),
        "n_positron": np.nan_to_num(
            np.array([as_float(r, "N_positron_real", 0.0) for r in ordered])
        ),
    }
    for key in EXACT_JZ_COLUMNS:
        values = np.array([as_float(row, key) for row in ordered])
        if not np.all(np.isfinite(values)):
            raise SystemExit(
                f"rrea_reduced.csv exact current-moment column {key} "
                "contains a missing, malformed, or non-finite value"
            )
        out[key] = values
    if "maxwell_jz_moment_A_m" in ordered[0]:
        values = np.array([as_float(row, "maxwell_jz_moment_A_m") for row in ordered])
        if np.all(np.isfinite(values)):
            out["maxwell_jz_moment_A_m"] = values
    # Fluid Maxwell time (domain min) is optional; eps_QE degrades rather than
    # failing when the column is absent.
    out["min_tau_m"] = np.array([as_float(r, "min_tau_m") for r in ordered])
    # sigma|Ez|-weighted conduction-current moments; nan-tolerant for captures
    # that do not carry these optional columns.
    for key in ("fluid_current_rms_r_m", "fluid_current_rms_z_m"):
        out[key] = np.array([as_float(r, key) for r in ordered])
    return out


def load_source_geometry(path: Path, min_macros: int) -> dict:
    """Read the energetic-source dimensions for smearing and eps_QE."""
    rows = read_csv_rows(path)
    peak = {"w": -1.0, "rms_r": math.nan, "rms_z": math.nan, "span_z": math.nan}
    frames: dict[str, list[float]] = {"time_s": [], "span_z_m": [], "max_r_m": []}
    for row in rows:
        w = as_float(row, "energetic_weight", 0.0)
        macros = as_float(row, "energetic_macro_count", 0.0)
        if not (w > 0.0) or macros < min_macros:
            continue  # zero-weight frames carry 0.0 placeholders for all means
        if w > peak["w"]:
            peak = {
                "w": w,
                "rms_r": as_float(row, "rms_r_m"),
                "rms_z": as_float(row, "rms_z_m"),
                "span_z": as_float(row, "max_z_m") - as_float(row, "min_z_m"),
            }
        # Per-frame extent series (eps_QE): as_float defaults to NaN and the
        # peak-only finiteness check below does not cover these rows.
        t_row = as_float(row, "time_s")
        span = as_float(row, "max_z_m") - as_float(row, "min_z_m")
        r_ext = as_float(row, "max_r_m")
        if all(math.isfinite(v) and v >= 0.0 for v in (t_row, span, r_ext)):
            frames["time_s"].append(t_row)
            frames["span_z_m"].append(span)
            frames["max_r_m"].append(r_ext)
    if peak["w"] < 0.0:
        raise SystemExit(
            "particle_moments.csv has no usable source-geometry frame "
            f"(>= {min_macros} macros)"
        )
    geometry_values = (peak["rms_r"], peak["rms_z"], peak["span_z"])
    if not all(math.isfinite(value) and value >= 0.0 for value in geometry_values):
        raise SystemExit("particle_moments.csv source geometry is missing or non-finite")
    return {
        "peak_rms_r_m": peak["rms_r"],
        "peak_rms_z_m": peak["rms_z"],
        "peak_span_z_m": peak["span_z"],
        "frames": {key: np.array(vals) for key, vals in frames.items()},
    }


def load_current_profile(path: Path) -> dict:
    """Read the optional dM/dz side-car (rrea_video/current_profile.csv).

    Returns z-bin centers and per-component bin-integral series [A*m]: the
    bins of one row sum to the corresponding global_jz_* value, so the
    finite-source retarded integral can treat each bin as a vertical current
    element at its bin-center altitude.  Restarted captures can overlap the
    reduced CSV, so rows are deduplicated by step, keeping the last.
    """
    rows = read_csv_rows(path)
    by_step: dict[int, dict] = {}
    for row in rows:
        by_step[int(float(row["step"]))] = row
    ordered = [by_step[key] for key in sorted(by_step)]
    first = ordered[0]
    nbins = int(float(first["nbins"]))
    z_lo, z_hi = float(first["z_lo_m"]), float(first["z_hi_m"])
    out = {
        "time_s": np.array([as_float(r, "time_s") for r in ordered]),
        "z_centers_m": z_lo + (np.arange(nbins) + 0.5) * (z_hi - z_lo) / nbins,
    }
    for comp in ("jz_fluid_A_m", "jz_kinetic_e_A_m", "jz_positron_A_m"):
        cols = [f"{comp}_b{b}" for b in range(nbins)]
        values = np.array([[as_float(r, c) for c in cols] for r in ordered])
        if not np.all(np.isfinite(values)):
            raise SystemExit(
                f"current_profile.csv component {comp} has a missing or "
                "non-finite value")
        out[comp] = values
    return out


def load_capture_extents(path: Path) -> dict:
    """Domain extents from rrea_video/video_capture_metadata.json.

    Surface-field and span checks both use this resolved geometry.
    """
    if not path.exists():
        raise SystemExit(f"missing capture metadata: {path}")
    try:
        meta = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise SystemExit(f"unreadable capture metadata {path}: {exc}")
    out = {}
    for key in ("r_max_m", "z_max_m"):
        value = float(meta.get(key, math.nan))
        if not (math.isfinite(value) and value > 0.0):
            raise SystemExit(f"capture metadata {path} lacks finite {key}")
        out[key] = value
    return out


def regrid_uniform(t: np.ndarray, series: list[np.ndarray]) -> tuple[np.ndarray, list[np.ndarray], float]:
    dt = float(np.median(np.diff(t)))
    jitter = float(np.max(np.abs(np.diff(t) - dt)))
    if jitter > 0.01 * dt:
        print(
            f"WARNING: non-uniform reduced sampling (max deviation {jitter:.3g} s "
            f"vs dt {dt:.3g} s); regridding by linear interpolation"
        )
    tu = np.arange(t[0], t[-1] + 0.5 * dt, dt)
    return tu, [np.interp(tu, t, y) for y in series], dt


def current_moment_series(
    capture_dir: Path,
    min_frame_macros: int = 10,
) -> dict:
    """Return the recorded vertical current moment on a uniform time grid [A*m]."""
    reduced = load_reduced(capture_dir / "rrea_reduced.csv")
    geometry = None
    try:
        geometry = load_source_geometry(
            capture_dir / "rrea_video" / "particle_moments.csv", min_frame_macros
        )
    except SystemExit as exc:
        print(f"note: particle moments unavailable ({exc}); smearing estimate skipped")
    extents = None
    try:
        extents = load_capture_extents(
            capture_dir / "rrea_video" / "video_capture_metadata.json"
        )
    except SystemExit as exc:
        print(f"note: capture metadata unavailable ({exc}); "
              "eps_QE and span-saturation checks skipped")

    proxy_total = (
        reduced["global_jz_kinetic_e_A_m"]
        + reduced["global_jz_positron_A_m"]
        + reduced["global_jz_fluid_A_m"]
    )
    if "maxwell_jz_moment_A_m" in reduced:
        authoritative = reduced["maxwell_jz_moment_A_m"]
        moment_label = "Maxwell face-current moment"
    else:
        authoritative = proxy_total
        moment_label = "component-sum current proxy"
    t_u, (jk, jg, jp, jf, n1, nk, ja), dt = regrid_uniform(
        reduced["time_s"],
        [
            reduced["global_jz_kinetic_e_A_m"],
            reduced["global_jz_kinetic_e_ge_1MeV_A_m"],
            reduced["global_jz_positron_A_m"],
            reduced["global_jz_fluid_A_m"],
            reduced["n_ge1MeV"],
            reduced["n_kinetic"],
            authoritative,
        ],
    )
    return {
        "t_s": t_u,
        "dt_s": dt,
        "m_moment": ja,
        "label": moment_label,
        "proxy_total_moment": jk + jp + jf,
        "components": {
            "kinetic_e": jk,
            "kinetic_e_ge_1MeV": jg,
            "positron": jp,
            "fluid": jf,
        },
        "n_ge1MeV": n1,
        "n_kinetic": nk,
        "reduced": reduced,
        "source_geometry": geometry,
        "capture_extents": extents,
    }


def boxcar_kernel_taps(dt: float, window_s: float, n_samples: int) -> int:
    """Odd tap count boxcar will REALLY use (0 when it smooths nothing).

    The realized window is not the requested one: rounding to samples and then
    forcing an odd length can drop a sample, so a 0.5 us request at 125 ns
    cadence realizes as 3 taps = 0.375 us. Labels use the realized window.
    """
    n = int(round(window_s / dt))
    # np.convolve(..., "same") returns the KERNEL length when the kernel is
    # longer than the signal -- clamp (odd, <= len(y)) so short captures
    # keep their length.
    n = min(n, n_samples)
    if n % 2 == 0:
        n -= 1
    return n if n >= 3 else 0


def boxcar_corner_hz(dt: float, taps: int) -> float:
    """Exact -3 dB frequency of the realized `taps`-point moving average.

    |H(f)| = |sin(pi f N dt) / (N sin(pi f dt))| -- the discrete Dirichlet
    kernel, NOT the continuous 0.443/window approximation, which is wrong by
    40% once N is as small as 3.  Monotone below the first null at
    f = 1/(N dt), so bisect there.  Infinite when nothing is smoothed.
    """
    if taps < 3:
        return math.inf

    def magnitude(f: float) -> float:
        x = math.pi * f * dt
        return abs(math.sin(taps * x) / (taps * math.sin(x)))

    lo, hi = 0.0, 1.0 / (taps * dt)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if magnitude(mid) > 1.0 / math.sqrt(2.0):
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def boxcar(y: np.ndarray, dt: float, window_s: float) -> np.ndarray:
    n = boxcar_kernel_taps(dt, window_s, len(y))
    if n == 0:
        return y
    kernel = np.ones(n)
    # Edge-corrected normalization so the ends are unbiased.
    return np.convolve(y, kernel, "same") / np.convolve(np.ones_like(y), kernel, "same")


def rc_highpass(x: np.ndarray, tau_s: float, dt_s: float) -> np.ndarray:
    """Single-pole RC field-change response (decay constant tau).

    y[0] = 0 takes the first sample as the pre-record baseline: any static
    offset or initialization step already present at t = 0 is deliberately
    suppressed. This instrument-model convention is independent of the selected
    source model.
    Single owner for every instrument high-pass (EFCM channels here and in
    the off-axis radio deliverable); do not reimplement the recursion.
    """
    alpha = tau_s / (tau_s + dt_s)
    y = np.empty_like(x)
    y[0] = 0.0
    for i in range(1, len(x)):
        y[i] = alpha * (y[i - 1] + x[i] - x[i - 1])
    return y


def butterworth_band_mask(freq: np.ndarray, f_lo: float, f_hi: float, order: int = 4) -> np.ndarray:
    # Zero-phase magnitude-only band-pass applied in the frequency domain
    # (acausal; fine for analysis, avoids Gibbs ringing of a hard mask and
    # avoids a scipy dependency).
    h = np.zeros_like(freq)
    positive = freq > 0
    f = freq[positive]
    h[positive] = 1.0 / np.sqrt(1.0 + (f_lo / f) ** (2 * order)) / np.sqrt(
        1.0 + (f / f_hi) ** (2 * order)
    )
    return h


def cumulative_charge_moment(m_moment: np.ndarray, dt: float) -> np.ndarray:
    """Trapezoid integral of a current moment, zero at the first sample."""
    return np.concatenate(
        [[0.0], np.cumsum(0.5 * (m_moment[1:] + m_moment[:-1]) * dt)]
    )


def dipole_field_terms(
    m_moment: np.ndarray,
    dt: float,
    distance_m: float,
    zenith_rad: float,
    image_factor: float = 1.0,
    *,
    p_moment: np.ndarray | None = None,
    dm_dt: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Retarded vertical-dipole field terms [V/m] for one observer geometry.

    p_moment/dm_dt override the copies derived here.  A caller working on a
    RETARDED time axis must pass them, derived on the SOURCE axis: integrating
    a time-shifted current from zero silently discards whatever charge
    accumulated before that element's first retarded sample, which is a
    constant static offset per element. The RC high-pass hides it on EFCM panels;
    the field-mill and static panels wear it.

    The retarded Hertzian-dipole field of a compact vertical current element
    (the standard result, e.g. Jackson, Classical Electrodynamics sec. 9.2;
    Dwyer & Cummer's eq 12 is the companion B, induction + radiation only,
    with no electrostatic term), with p(t) = integral M dt:

        E_theta = sin(theta)/(4 pi eps0) [p/R^3 + M/(c R^2)] + radiation
        E_r     = 2 cos(theta)/(4 pi eps0) [p/R^3 + M/(c R^2)]

    The bracket is common to both components; E_r has no radiation term.
    Callers take the vertical component with

        e_z = e_r cos(theta) - e_theta sin(theta),

    which degenerates to E_r for an observer directly overhead (the LIP
    field-mill case, where the radiation term is exactly zero) and to
    -E_theta for a receiver on the ground at theta = 90 deg.  The EFCM's
    flat plate points along the aircraft heading and reads the HORIZONTAL
    in-plane component e_r sin(theta) + e_theta cos(theta) instead, which
    is sin(theta) cos(theta) (3N + S) in terms of the near bracket N and
    radiation magnitude S -- zero on axis AND at theta = 90 deg, maximal
    at 45 deg.  image_factor = 2 adds the co-directed perfectly conducting
    ground image (their section 8).
    """
    near = image_factor / (4.0 * math.pi * EPS0)
    if p_moment is None:
        p_moment = cumulative_charge_moment(m_moment, dt)
    if dm_dt is None:
        dm_dt = np.gradient(m_moment, dt)
    static = near * p_moment / distance_m**3
    induction = near * m_moment / (C_LIGHT * distance_m**2)
    radiation = image_factor * MU0 * dm_dt / (4.0 * math.pi * distance_m)
    sin_theta, cos_theta = math.sin(zenith_rad), math.cos(zenith_rad)
    return {
        "p_moment": p_moment,
        "e_theta_static": sin_theta * static,
        "e_theta_induction": sin_theta * induction,
        "e_theta_radiation": sin_theta * radiation,
        "e_r_static": 2.0 * cos_theta * static,
        "e_r_induction": 2.0 * cos_theta * induction,
        # The full components; the terms above are the decomposition.
        "e_theta": sin_theta * (static + induction + radiation),
        "e_r": 2.0 * cos_theta * (static + induction),
    }


def finite_source_fields(
    profile: dict,
    t_s: np.ndarray,
    dt: float,
    observer_alt_msl_m: float,
    offset_m: float,
    z0_alt_msl_m: float,
    smooth_window_s: float = 0.0,
    retard_ref_m: float | None = None,
) -> dict[str, np.ndarray]:
    """Aircraft fields from the z-resolved current profile, bin by bin.

    A z-RESOLVED LINE DIPOLE, radially collapsed: each of the dM/dz side-car's
    z-bins is a vertical current element at its own slant range and polar
    angle, but the whole radial current distribution of a bin sits on the
    symmetry axis. That is the binding approximation: source_context derives
    its unresolved radial smearing from each run's recorded moments. Resolving
    it requires an (r,z) current diagnostic, which the capture does not record.

    Within that approximation the axial treatment is the edge-charge model
    (lambda = -int dI/dz dt): adjacent bins' implicit edge charges telescope,
    and the residual per-bin multipole error is O((dz/R_b)^2), so
    it removes the (L_z/R)^2 point-dipole geometry error that is of order
    unity at the aircraft standoff.  Current that exits the domain top has its
    charge left at the top edge: for an OPEN boundary that is a closure
    assumption, not charge conservation -- p = int M dt holds exactly only up
    to the boundary flux term, and the escaping charge physically keeps going.

    Returns e_z (vertical, +z up), e_h (horizontal in-plane, positive AWAY
    from the source axis -- instrument sign conventions belong to the
    caller), e_z_static (static terms only: what survives a field mill's
    quasi-static sampling), and m_check (the bin-summed moment, which must
    reproduce the reduced-CSV M(t) -- callers gate on it).

    retard_ref_m applies the DIFFERENTIAL retardation (R_b - ref)/c per
    bin. The common ref/c stays unapplied so
    every consumer keeps the repo's source-time axis convention.
    """
    m_bins = (
        profile["jz_fluid_A_m"]
        + profile["jz_kinetic_e_A_m"]
        + profile["jz_positron_A_m"]
    )
    t_prof = profile["time_s"]
    dt_prof = float(np.median(np.diff(t_prof))) if t_prof.size > 1 else dt
    smooth_taps = boxcar_kernel_taps(dt_prof, smooth_window_s, t_prof.size)
    e_z = np.zeros_like(t_s)
    e_h = np.zeros_like(t_s)
    e_z_static = np.zeros_like(t_s)
    m_check = np.zeros_like(t_s)
    delays_s: list[float] = []
    for i_bin, z_center in enumerate(profile["z_centers_m"]):
        vertical_m = observer_alt_msl_m - (z0_alt_msl_m + z_center)
        r_b = math.hypot(offset_m, vertical_m)
        theta_b = math.atan2(offset_m, vertical_m)
        t_bin = t_s
        delay_s = 0.0
        if retard_ref_m is not None:
            delay_s = (r_b - retard_ref_m) / C_LIGHT
            t_bin = t_s - delay_s
        delays_s.append(delay_s)
        # Smooth, integrate and differentiate on the SOURCE axis, then carry
        # all three to the retarded axis.  Interpolating the current alone and
        # rebuilding p from it there restarts the charge integral at whatever
        # sample the shift happens to land on.
        m_src = m_bins[:, i_bin]
        if smooth_taps:
            m_src = boxcar(m_src, dt_prof, smooth_window_s)
        p_src = cumulative_charge_moment(m_src, dt_prof)
        dm_src = np.gradient(m_src, dt_prof)
        # right=: values past the record only ever reach samples the caller
        # masks (see valid_interval_s); nothing inside the supported
        # window reads them, because p(t) depends on the current at t' <= t.
        m_b = np.interp(t_bin, t_prof, m_src, left=0.0, right=m_src[-1])
        p_b = np.interp(t_bin, t_prof, p_src, left=0.0, right=p_src[-1])
        dm_b = np.interp(t_bin, t_prof, dm_src, left=0.0, right=0.0)
        terms = dipole_field_terms(
            m_b, dt, r_b, theta_b, p_moment=p_b, dm_dt=dm_b)
        e_r, e_theta = terms["e_r"], terms["e_theta"]
        sin_t, cos_t = math.sin(theta_b), math.cos(theta_b)
        e_z += e_r * cos_t - e_theta * sin_t
        e_h += e_r * sin_t + e_theta * cos_t
        e_z_static += (
            terms["e_r_static"] * cos_t - terms["e_theta_static"] * sin_t
        )
        # Un-shifted, un-smoothed sum: the conservation gate compares the
        # bins against the reduced-CSV moment, not against a filtered copy.
        m_check += np.interp(t_s, t_prof, m_bins[:, i_bin],
                             left=0.0, right=m_bins[-1, i_bin])
    # Observer times every bin can answer from recorded source, without
    # extrapolation: the farthest bin sets how late the window opens, the
    # nearest how early it must close.
    valid_lo = float(t_prof[0] + max(delays_s))
    valid_hi = float(t_prof[-1] + min(delays_s))
    return {"e_z": e_z, "e_h": e_h, "e_z_static": e_z_static,
            "m_check": m_check,
            "valid_interval_s": (valid_lo, valid_hi),
            "smooth_taps": smooth_taps,
            "smooth_window_s": smooth_taps * dt_prof,
            "smooth_corner_hz": boxcar_corner_hz(dt_prof, smooth_taps)}


def source_context(
    series: dict,
    distance_m: float,
    zenith_rad: float,
) -> dict:
    """Measured validity limits and TGF normalization from the recorded run.

    * source-size smearing, carrying the Dwyer & Cummer (1 - beta cos theta)
      front-motion compression on the axial term (their eqs 18-19; section
      13.4 notes it matters at small angles, i.e. for an overhead observer),
      and the corner frequency 1/(2 pi t_smear) above which a point-source
      waveform overestimates the emission;
    * the point-dipole error (L/R)^2 at the plotted standoff;
    * separate energetic-beam and conductivity-weighted fluid extents, so the
      component dominating the recorded current supplies its own corner;
    * the runaway count and path length the recorded current implies
      (their eq B2, I_re = e v N_re on the >= 1 MeV moment, which is the
      population their N_TGF counts, and Omega = int |I_re| dt / e), against an
      average TGF, because the absolute field amplitudes scale with the seed
      normalization of the run and not with any observed TGF.
    """
    m_moment = series["m_moment"]
    i_peak = int(np.argmax(np.abs(m_moment)))
    m_kinetic = series["components"]["kinetic_e_ge_1MeV"]
    j_peak = int(np.argmax(np.abs(m_kinetic)))
    # D&C's Omega ([53], eq. 36) is the total runaway path length
    # int N_re v dt = int |I_re| dt / e: positive definite, so abs goes inside.
    omega_m = float(cumulative_charge_moment(
        np.abs(m_kinetic), series["dt_s"])[-1] / ELEMENTARY_CHARGE_C)
    n_re = abs(m_kinetic[j_peak]) / (
        ELEMENTARY_CHARGE_C * BETA_AVALANCHE * C_LIGHT
    )
    context = {
        "m_peak_A_m": float(m_moment[i_peak]),
        "t_peak_s": float(series["t_s"][i_peak]),
        "fluid_fraction": float(
            series["components"]["fluid"][i_peak] / m_moment[i_peak]
        )
        if m_moment[i_peak] != 0.0
        else math.nan,
        "n_re": float(n_re),
        "n_re_ratio": float(n_re / DC_N_TGF),
        "dipole_error": math.nan,
        "dipole_error_rms": math.nan,
        "omega_m": omega_m,
        "omega_ratio": omega_m / DC_OMEGA_M,
    }
    geometry = series["source_geometry"]
    if geometry is not None:
        context.update(
            dipole_error=(geometry["peak_span_z_m"] / distance_m) ** 2,
            # The span is an extremal statistic that can saturate at the mesh,
            # so also quote the rms-based convention.
            dipole_error_rms=(2.0 * geometry["peak_rms_z_m"] / distance_m) ** 2,
            omega_m=omega_m,
            omega_ratio=omega_m / DC_OMEGA_M,
        )
        extents = series.get("capture_extents")
        if extents and geometry["peak_span_z_m"] > 0.9 * extents["z_max_m"]:
            print(
                f"WARN: energetic span_z {geometry['peak_span_z_m']:.0f} m is "
                f"{geometry['peak_span_z_m'] / extents['z_max_m']:.0%} of the "
                "domain height -- a domain-saturated extremal statistic; "
                "prefer the (2*rms_z/R)^2 convention"
            )
    context["summary"] = (
        f"source context: |M| peak {context['m_peak_A_m']:.3e} A*m "
        f"({context['m_peak_A_m'] / 1e6:.0f} kA*km, fluid conduction "
        f"{context['fluid_fraction'] * 100.0:.0f}% of it) at "
        f"t = {context['t_peak_s'] * 1e6:.1f} us; implied N_re = "
        f"{context['n_re']:.2e} ({context['n_re_ratio']:.0f}x the average-TGF "
        f"N_TGF = {DC_N_TGF:.0e}), Omega = {context['omega_m']:.2e} m "
        f"({context['omega_ratio']:.0f}x {DC_OMEGA_M:.1e} m) -- absolute "
        "amplitudes scale with this run's seed normalization, not with an "
        "observed TGF.\n"
        f"  at R = {distance_m / 1e3:.1f} km, theta = "
        f"{math.degrees(zenith_rad):.1f} deg: point-dipole error "
        f"(L/R)^2 = {context['dipole_error']:.3g} (span-based), "
        f"(2*rms_z/R)^2 = {context['dipole_error_rms']:.3g}"
    )
    # Use the sigma|Ez|-weighted fluid-current moments when available.
    red = series.get("reduced") or {}
    fluid_z = red.get("fluid_current_rms_z_m")
    if fluid_z is not None and np.any(np.isfinite(fluid_z)):
        j = int(np.argmin(np.abs(red["time_s"] - context["t_peak_s"])))
        if np.isfinite(fluid_z[j]) and fluid_z[j] > 0.0:
            context["fluid_rms_z_m"] = float(fluid_z[j])
            context["fluid_rms_r_m"] = float(red["fluid_current_rms_r_m"][j])
            context["dipole_error_fluid"] = (
                2.0 * context["fluid_rms_z_m"] / distance_m) ** 2
            # Beam and conduction-current extents can differ, so the component
            # dominating a run's recorded M must supply its own smearing scale.
            # The fluid current is assigned no advancing-front compression: it
            # flows in the ionized channel rather than with the runaway front.
            context["summary"] += (
                f"; FLUID-current rms_z = {context['fluid_rms_z_m'] / 1e3:.2f} km "
                f"(rms_r {context['fluid_rms_r_m'] / 1e3:.2f} km) at the |M| "
                f"peak -> (2*rms_z/R)^2 = {context['dipole_error_fluid']:.3g}"
            )
            # Collapsed onto the axis the current loses its broadside phase
            # spread k rms_r sin(theta); the compact source overestimates above.
            context["radial_corner_hz"] = C_LIGHT / (
                2.0 * math.pi * context["fluid_rms_r_m"] * math.sin(zenith_rad))
            context["summary"] += (
                f"; radial corner c/(2 pi rms_r sin theta) = "
                f"{context['radial_corner_hz'] / 1e3:.0f} kHz, above which the "
                "collapsed dipole overestimates the radiation")
    return context


def amplitude_spectrum(
    e_field: np.ndarray,
    dt: float,
    window: str,
    gain: str = "coherent",
) -> tuple[np.ndarray, np.ndarray]:
    """One-sided |E(f)| [V/m/Hz].

    gain="coherent" preserves peak amplitude and is what the waveform panels
    want; gain="power" preserves energy (Parseval) and is what the spectral
    energy density needs.
    """
    y = e_field - np.mean(e_field)
    if window == "hann":
        w = np.hanning(len(y))
        y = y * w / (np.mean(w) if gain == "coherent" else math.sqrt(np.mean(w**2)))
    freq = np.fft.rfftfreq(len(y), dt)
    return freq, dt * np.abs(np.fft.rfft(y))


def spectral_energy_density(
    e_field: np.ndarray,
    dt: float,
    window: str = "none",
) -> tuple[np.ndarray, np.ndarray]:
    """Radiated energy per unit area per unit frequency [J/(Hz m^2)].

    Dwyer & Cummer eqs 30-32 with |E| = c|B|: Parseval on S = c eps0 E^2
    gives the one-sided density 2 c eps0 |E(f)|^2.  This is the quantity
    their Figures 4, 5, 12 and 13 compare, and the one that constrains the
    number of avalanche pulses.  Their extra factor 4 for a ground receiver
    comes free when E already carries image_factor = 2.
    """
    freq, spec = amplitude_spectrum(e_field, dt, window, gain="power")
    density = 2.0 * C_LIGHT * EPS0 * spec**2
    density[0] *= 0.5                     # DC, and the even-N Nyquist bin,
    if e_field.size % 2 == 0:             # occur once in the two-sided sum
        density[-1] *= 0.5
    return freq, density


def figure_argument_parser(
    description: str, *, smooth_us_default: float
) -> argparse.ArgumentParser:
    """Common CLI preamble of the three figure scripts (radio, multiband,
    optical): positional capture dir, output PNG, smoothing.  Script-specific
    flags -- including the sparse-frame floor, which only the two scripts that
    read particle_moments.csv need -- are added by the caller."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("capture_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="output PNG")
    parser.add_argument("--smooth-us", type=float, default=smooth_us_default)
    return parser


def add_cloud_arguments(parser: argparse.ArgumentParser) -> None:
    """Cloud-transport flags shared by the optical and multiband figures."""
    parser.add_argument(
        "--no-cloud-scattering", dest="cloud_scattering", action="store_false",
        help="skip the cloudscat transport and show only the free-space "
             "1/(4 pi d^2) curves",
    )
    parser.set_defaults(cloud_scattering=True)
    parser.add_argument(
        "--cloud-top-km", type=float, default=16.0,
        help="cloud top MSL for the transport (default 16; cloudscat ships "
             "12, which can leave a high-altitude source outside the cloud)",
    )
    parser.add_argument(
        "--cloud-photons", type=int, default=200_000,
        help="Monte Carlo photons per band (default 2e5; results are cached "
             "per case)",
    )


def main() -> None:
    parser = figure_argument_parser(
        __doc__.splitlines()[0], smooth_us_default=0.5
    )
    parser.add_argument("--min-frame-macros", type=int, default=10)
    parser.add_argument("--distance-km", type=float, default=100.0)
    parser.add_argument("--zenith-deg", type=float, default=90.0)
    parser.add_argument("--ground-image", dest="ground_image", action="store_true", default=False)
    parser.add_argument("--no-ground-image", dest="ground_image", action="store_false")
    parser.add_argument("--band", type=float, nargs=2, default=[1e3, 5e5], metavar=("F_LO", "F_HI"))
    parser.add_argument("--spectrum-window", choices=["hann", "none"], default="hann")
    parser.add_argument("--full-field", action="store_true",
                        help="also plot electrostatic + induction terms")
    args = parser.parse_args()

    series = current_moment_series(args.capture_dir, args.min_frame_macros)
    t_u = series["t_s"]
    dt = series["dt_s"]
    m_moment = series["m_moment"]
    reduced = series["reduced"]
    print(f"current-moment source: exact -- {series['label']}")

    i_peak = int(np.argmax(reduced["n_ge1MeV"]))
    n_peak = reduced["n_ge1MeV"][i_peak]
    pos_ratio = reduced["n_positron"][i_peak] / max(n_peak, 1.0)
    sub_ratio = (reduced["n_kinetic"][i_peak] - n_peak) / max(n_peak, 1.0)
    print(f"at N>=1MeV peak: positron/electron = {pos_ratio:.2e}, sub-MeV/ge-1MeV = {sub_ratio:.2f}")

    zenith_rad = math.radians(args.zenith_deg)
    distance_m = args.distance_km * 1e3
    context = source_context(series, distance_m, zenith_rad)
    print(context["summary"])

    m_smooth = boxcar(m_moment, dt, args.smooth_us * 1e-6)
    image = 2.0 if args.ground_image else 1.0
    terms = dipole_field_terms(m_smooth, dt, distance_m, zenith_rad, image)
    e_raw = terms["e_theta"]
    freq_mask = np.fft.rfftfreq(len(e_raw), dt)
    mask_h = butterworth_band_mask(freq_mask, args.band[0], args.band[1])
    e_band = np.fft.irfft(np.fft.rfft(e_raw) * mask_h, n=len(e_raw))
    freq, spec = amplitude_spectrum(e_raw, dt, args.spectrum_window)
    dmdt = np.gradient(m_smooth, dt)

    print(
        f"|E| peak = {np.max(np.abs(e_raw)):.2f} V/m at R={args.distance_km:.0f} km, "
        f"theta={args.zenith_deg:.0f} deg, image x{image:.0f}. Dwyer & Cummer 2013: "
        "~14 kA*km for the relativistic-feedback model (Fig B1; their lightning-"
        "source models reach 50-100+), peak current ~10-100 kA (section 13); the "
        "two measured LF events give ~0.45-1.2 V/m at 470-500 km (Figs 8-9, E = cB "
        "from 1.5/4 nT) after sensor response and ground-wave attenuation, so that "
        "amplitude is not a raw radiation field"
    )

    # --- outputs --------------------------------------------------------------
    args.output.parent.mkdir(parents=True, exist_ok=True)
    csv_path = args.output.with_suffix(".waveform.csv")
    comps = series["components"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["t_s", "M_Am", "M_smooth_Am", "dMdt_Am_per_s",
             "E_raw_Vpm", "E_band_Vpm", "M_kinetic_e_Am",
             "M_fluid_Am", "M_positron_Am"]
        )
        for i in range(len(t_u)):
            writer.writerow(
                [f"{t_u[i]:.9e}", f"{m_moment[i]:.6e}", f"{m_smooth[i]:.6e}",
                 f"{dmdt[i]:.6e}", f"{e_raw[i]:.6e}", f"{e_band[i]:.6e}",
                 f"{comps['kinetic_e'][i]:.6e}",
                 f"{comps['fluid'][i]:.6e}",
                 f"{comps['positron'][i]:.6e}"]
            )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t_us = t_u * 1e6
    fig, axes = plt.subplots(3, 1, figsize=(11.0, 12.5), dpi=180)
    ax = axes[0]
    ax.plot(t_us, m_moment / 1e6, lw=0.9, color="#1f77b4", label=f"M(t): {series['label']}")
    ax.plot(t_us, comps["kinetic_e"] / 1e6, lw=0.7, ls="--", color="#ff7f0e",
            label="kinetic e-")
    ax.plot(t_us, comps["fluid"] / 1e6, lw=0.7, ls="--", color="#2ca02c",
            label="fluid conduction")
    ax.plot(t_us, comps["positron"] / 1e6, lw=0.7, ls=":", color="#d62728",
            label="positrons")
    ax.set_ylabel("current moment (kA km)")
    ax.legend(loc="lower left", fontsize=8)
    ax.set_title("Vertical current moment (negative: upward electrons, conventional current down)")

    ax = axes[1]
    ax.plot(t_us, e_raw, lw=0.7, color="#d62728", label="radiation term (raw)")
    ax.plot(t_us, e_band, lw=0.9, color="#2ca02c",
            label=f"band {args.band[0]/1e3:.0f}-{args.band[1]/1e3:.0f} kHz")
    if args.full_field:
        ax.plot(t_us, terms["e_theta_static"], lw=0.7, ls="--",
                label="electrostatic term")
        ax.plot(t_us, terms["e_theta_induction"], lw=0.7, ls=":",
                label="induction term")
    ax.set_ylabel(r"$E_\theta$ (V/m)")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title(
        rf"$E_\theta$ at R={args.distance_km:.0f} km, $\theta$={args.zenith_deg:.0f}$^\circ$"
        rf"{', PEC ground image x2' if args.ground_image else ''} "
        rf"(source time; add R/c = {distance_m/C_LIGHT*1e6:.0f} $\mu$s; "
        r"ground receiver sees $E_z \approx -E_\theta$)"
    )

    ax = axes[2]
    positive = freq > 0
    ax.loglog(freq[positive], np.maximum(spec[positive], 1e-30), lw=0.8, color="#9467bd")
    for f_edge in args.band:
        ax.axvline(f_edge, color="grey", ls="--", lw=0.7)
    ax.axvline(0.5 / dt, color="red", ls=":", lw=0.8)
    ax.text(0.5 / dt, ax.get_ylim()[0], " Nyquist", fontsize=7, color="red",
            rotation=90, va="bottom")
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("spectral amplitude (V/m/Hz)")
    ax.set_title(f"Amplitude spectrum ({args.spectrum_window} window)")

    for ax in axes[:2]:
        ax.set_xlabel(r"source time ($\mu$s)")
        ax.grid(alpha=0.2)
    axes[2].grid(alpha=0.2, which="both")
    caveats = (
        "Source: recorded per-step current moments (kinetic e- incl. sub-MeV, "
        "positrons, fluid conduction incl. ion drift). Remaining approximations: "
        "free-space leading current-dipole field with no ground or ionosphere; "
        f"point-dipole error (L/R)^2 = {context['dipole_error']:.3g}. "
        "Amplitudes scale with this run's seed normalization: "
        f"implied N_re = {context['n_re']:.2e} = {context['n_re_ratio']:.0f}x the "
        "average-TGF value. Method: Dwyer & Cummer, JGR 2013, "
        "doi:10.1002/jgra.50188."
    )
    fig.text(0.01, 0.005, caveats, fontsize=6.5, va="bottom")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(args.output)
    plt.close(fig)
    print(f"Wrote {args.output}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
