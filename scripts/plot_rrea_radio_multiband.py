#!/usr/bin/env python3
"""Multi-band view of the RREA radio waveform from one capture directory.

Panels, in order: the electron source lightcurve (linear, then log) and the
photon lightcurve, both everything above 12 km MSL, the 337.1 / 777.4 nm optical source
rates and aircraft irradiance, the two ALOFT EFCM channels, the LIP
field-mill panel, the ground-receiver LF and VLF waveforms, and the
amplitude plus radiated-energy spectra -- all on one source-time axis. Row
indices come from the PANEL_* block below; never hand-count axes[n].

The three band lightcurves need scripts/extract_rrea_band_series.py to run on
the capture host first; the video renderer's strip draws the SAME CSVs, so the
two products cannot disagree. The bands belong to that script's BAND_SPECIES.

The instrument assumptions and interpretation limits are documented in
docs/RREA_WARPX_AMREX_ARCHITECTURE_GUIDE.tex. The two rules for reading a band
curve (population not flux; a leading zero run is an occupancy floor, not an
empty band) ARE printed on the figure, because a reader of the PNG has only
the PNG -- that copy is deliberate, and the guide is where the reasoning
lives.

Usage:
  python3 scripts/plot_rrea_radio_multiband.py CAPTURE_DIR --output out.png
"""

from __future__ import annotations

import math
import textwrap
from pathlib import Path

import numpy as np

from rrea_cloud_scattering import OBSERVER_ALTITUDE_M_MSL
from extract_rrea_band_series import band_label, load_band_series
from compute_rrea_em_radio import load_observer_series
from compute_rrea_optical_emissions import (
    E_PHOTON_337_J,
    E_PHOTON_777_J,
    capture_altitude_z0_m,
    observer_geometry,
    optical_series,
    scattered_irradiance,
)
from compute_rrea_radio_waveform import (
    add_cloud_arguments,
    figure_argument_parser,
    C_LIGHT,
    amplitude_spectrum,
    boxcar,
    boxcar_corner_hz,
    boxcar_kernel_taps,
    butterworth_band_mask,
    current_moment_series,
    dipole_field_terms,
    finite_source_fields,
    load_current_profile,
    rc_highpass,
    source_context,
    spectral_energy_density,
)

BANDS = [
    ("LF 30-300 kHz", 3e4, 3e5, "#2ca02c"),
    ("VLF 3-30 kHz", 3e3, 3e4, "#1f77b4"),
]
EFCM_TAU_SLOW_S = 0.150   # ALOFT EFCM slow channel decay constant (nominal)
EFCM_TAU_FAST_S = 100e-6  # ALOFT EFCM fast channel decay constant (nominal)
EFCM_FNY_SLOW_HZ = 0.5e6  # slow channel digitizer Nyquist (1 MHz sampling)
EFCM_FNY_FAST_HZ = 5.0e6  # fast channel digitizer Nyquist (10 MHz sampling)
# (name, RC decay constant, digitizer Nyquist).  Kochkin et al. 2021 states
# the EFCM bandwidth "is not sufficiently understood beyond fast and slow";
# these values are the nominal ALOFT instrument-page figures.
EFCM_CHANNELS = (
    ("slow", EFCM_TAU_SLOW_S, EFCM_FNY_SLOW_HZ),
    ("fast", EFCM_TAU_FAST_S, EFCM_FNY_FAST_HZ),
)
# The dM/dz side-car's per-species bin columns, in the order the field model
# sums them.  Named once so the per-species decomposition below cannot drift
# from what finite_source_fields actually adds up.
PROFILE_SPECIES = ("jz_fluid_A_m", "jz_kinetic_e_A_m", "jz_positron_A_m")
LIP_SAMPLE_HZ = 50.0      # LIP rotating-vane field mills, samples per second
HUYGENS_NOTE = "Huygens/Love surface integral"
# Derive every panel offset and the subplot count from this row order.
(
    PANEL_LIGHTCURVE,
    PANEL_LIGHTCURVE_LOG,
    PANEL_PHOTON_BAND,
    PANEL_OPTICAL_RATES,
    PANEL_OPTICAL_IRRADIANCE,
    PANEL_EFCM_FIRST,
) = range(6)
PANEL_LIP = PANEL_EFCM_FIRST + len(EFCM_CHANNELS)
PANEL_RADIO_FIRST = PANEL_LIP + 1
PANEL_SPECTRA = PANEL_RADIO_FIRST + len(BANDS)
PANEL_COUNT = PANEL_SPECTRA + 1


def mask_unsupported(
    y: np.ndarray, t_s: np.ndarray, interval: tuple[float, float] | None
) -> np.ndarray:
    """NaN outside the observer window every source bin can answer.

    Applied AFTER filtering, never before: rc_highpass is a recursion, so a
    leading NaN would poison the whole trace instead of blanking an edge.
    Outside this window the per-bin retardation would be reading source the
    capture never recorded.
    """
    if interval is None:
        return y
    lo, hi = interval
    out = np.array(y, dtype=float, copy=True)
    out[(t_s < lo) | (t_s > hi)] = np.nan
    return out


def shared_source_time_limits_us(*time_arrays: np.ndarray) -> tuple[float, float]:
    """Union of finite source-time records for every time-domain panel."""
    finite = [
        np.asarray(values, dtype=float)[np.isfinite(values)]
        for values in time_arrays
        if values is not None
    ]
    finite = [values for values in finite if values.size]
    if not finite:
        raise ValueError("shared source-time limits require finite samples")
    lo = min(float(np.min(values)) for values in finite)
    hi = max(float(np.max(values)) for values in finite)
    if not hi > lo:
        raise ValueError("shared source-time limits require a positive span")
    return lo, hi


def last_supported_index(
    t_s: np.ndarray, interval: tuple[float, float] | None
) -> int:
    """Index of the last sample inside the supported observer window.

    A headline read off ``[-1]`` lands on the most extrapolated sample in the
    record, which is exactly where the model is least defensible.
    """
    if interval is None:
        return len(t_s) - 1
    inside = np.flatnonzero(t_s <= interval[1])
    return int(inside[-1]) if inside.size else len(t_s) - 1


def butterworth_lowpass_mask(freq: np.ndarray, f_ny: float, order: int = 4) -> np.ndarray:
    """Zero-phase low-pass magnitude mask with h(0) = 1.

    compute_rrea_radio_waveform.butterworth_band_mask zeroes the DC bin
    (harmless for its band-pass callers, wrong for a low-pass: it would
    subtract the slow channel's mean), hence this dedicated mask.
    """
    return 1.0 / np.sqrt(1.0 + (freq / f_ny) ** (2 * order))


def apply_lowpass(x: np.ndarray, f_ny: float, dt_s: float) -> np.ndarray:
    """Zero-phase Butterworth low-pass at f_ny, edge-padded.

    FFT filtering is circular; the EFCM slow trace ends far from where it
    starts, so an unpadded filter smears the wrap discontinuity into spikes
    at both record edges. Constant edge padding of a few filter time
    constants keeps the ringing inside the cropped pads.
    """
    n_pad = max(int(round(4.0 / (f_ny * dt_s))), 8)
    x_pad = np.concatenate(
        [np.full(n_pad, x[0]), x, np.full(n_pad, x[-1])]
    )
    freq = np.fft.rfftfreq(len(x_pad), dt_s)
    y_pad = np.fft.irfft(
        np.fft.rfft(x_pad) * butterworth_lowpass_mask(freq, f_ny), n=len(x_pad)
    )
    return y_pad[n_pad:n_pad + len(x)]


def lip_vertical_field(
    m_moment: np.ndarray,
    dt: float,
    distance_m: float,
    zenith_rad: float,
) -> np.ndarray:
    """Vertical field E_z at the aircraft, all three retarded dipole terms.

    The LIP field mills read this component; +z up, so the negative moment
    of a downward-TGF charge transfer gives a negative step.
    """
    terms = dipole_field_terms(m_moment, dt, distance_m, zenith_rad)
    e_r, e_theta = terms["e_r"], terms["e_theta"]
    return e_r * math.cos(zenith_rad) - e_theta * math.sin(zenith_rad)


def efcm_plate_field(
    m_moment: np.ndarray,
    dt: float,
    distance_m: float,
    zenith_rad: float,
) -> np.ndarray:
    """Field along the EFCM plate normal: horizontal, along heading.

    The EFCM "is a flat-plate antenna which is pointed straight ahead"
    (Kochkin et al. 2021, JGR 126, e2020JD033467), so it reads the
    horizontal in-plane component -- sin(theta)cos(theta)(3N + S), with an
    EXACT null for a vertical dipole at nadir (azimuthal symmetry, not a
    point-dipole artifact) and another at theta = 90 deg, maximal at 45.
    Sign: the source sits AHEAD of the aircraft, so the plate normal points
    toward the source azimuth = MINUS the away-from-axis direction;
    positive output = field pointing forward along heading.
    """
    terms = dipole_field_terms(m_moment, dt, distance_m, zenith_rad)
    e_r, e_theta = terms["e_r"], terms["e_theta"]
    return -(e_r * math.sin(zenith_rad) + e_theta * math.cos(zenith_rad))


def draw_scattered_overlay(
    ax,
    scattered: dict,
    window_us: tuple[float, float],
    geometric_delay_us: float,
) -> float:
    """Draw the CloudScat curves on the shared source-time axis.

    CloudScat output is ABSOLUTE observer-arrival time on its own grid
    (pinned by test_optical_physics.py, and generally not the reduced
    timeline's length).  Only the constant geometric delay d/c is removed
    here, so the curves land on this figure's shared source-time axis while
    the in-cloud excess delay -- the physical scattering lag -- stays
    visible.  Returns the fraction of the 337 nm fluence inside the shared
    window_us and the source time by which 90% of it has arrived, so this
    panel alone can run past the shared axis instead of clipping the tail.
    """
    t_us = scattered["t_s"] * 1e6 - geometric_delay_us
    # The series is uW/m^2 (scattered_irradiance); this panel reads mW/m^2.
    lo, hi = scattered["irr337_lo"] * 1e-3, scattered["irr337_hi"] * 1e-3
    ax.fill_between(t_us, np.clip(lo, 1e-9, None), np.clip(hi, 1e-9, None),
                    color="#0d366b", alpha=0.55, label="337.1 nm scattered")
    ax.plot(t_us, lo, lw=1.0, color="#0d366b")
    ax.plot(t_us, scattered["irr777"] * 1e-3, lw=1.0, color="#7c1a19",
            label="777.4 nm scattered")
    flux = np.clip(scattered["irr337_lo"], 0.0, None)
    total = float(np.trapezoid(flux, t_us))
    if total <= 0.0:
        return 1.0, window_us[1]
    inside = (t_us >= window_us[0]) & (t_us <= window_us[1])
    fraction = float(np.trapezoid(np.where(inside, flux, 0.0), t_us)) / total
    cumulative = np.concatenate(
        [[0.0], np.cumsum(0.5 * (flux[1:] + flux[:-1]) * np.diff(t_us))])
    # 90%, not 99%: the diffusion tail is quasi-exponential, so the last
    # decade of fluence would squeeze the peak into a sliver.
    t90_us = float(t_us[np.searchsorted(cumulative, 0.90 * total)])
    if fraction < 0.9:
        print(f"WARN: only {fraction * 100.0:.0f}% of the cloud-scattered "
              "337 nm fluence arrives inside the shared source-time window; "
              f"the irradiance panel runs to {t90_us:.0f} us (90% arrival)")
    return fraction, max(t90_us, window_us[1])


def run_label(capture_dir: Path) -> str:
    """Name the run a capture directory belongs to, in either layout.

    On the cluster the run name is one level up (``<run>/capture``); locally
    ``olivia.py pull-capture`` flattens that to ``<run>/``.
    """
    return (capture_dir.parent
            if capture_dir.name == "capture" else capture_dir).name


def draw_band_panel(
    ax, band: dict | None, *, noun: str, color: str, log: bool
) -> None:
    """One band source lightcurve.

    `band` is None for a capture that has not been reduced (frames stay on the
    cluster): the fixed layout keeps and labels the row.
    """
    if band is None:
        print(f"WARN: no {noun} band series; run "
              "scripts/extract_rrea_band_series.py on the capture host")
        ax.set_title(f"{noun} band lightcurve: not extracted for this capture",
                     loc="left")
        ax.text(0.5, 0.5, "run scripts/extract_rrea_band_series.py\n"
                          "on the capture to fill this panel",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=9, color="0.45")
        ax.set_yticks([])
        return
    t_band_us = band["time_s"] * 1e6
    n_band = band["n_ge1MeV_real"]
    # A peak-relative floor preserves a log panel's useful dynamic range.
    if log:
        peak = float(np.max(n_band)) if n_band.size else 0.0
        floor = peak / 1.0e10 if peak > 0.0 else 1.0
        ax.plot(t_band_us, np.clip(n_band, floor, None), lw=1.1, color=color)
        ax.set_ylim(bottom=floor)
    else:
        ax.plot(t_band_us, n_band, lw=1.1, color=color)
    i_pk = int(np.argmax(n_band))
    ax.axvline(t_band_us[i_pk], color=color, ls=":", lw=0.8)
    # A leading run of exactly-zero frames is a MACROPARTICLE OCCUPANCY floor,
    # not an empty band: the grid holds the density of the SAMPLED population,
    # so the outermost nonzero cell sits wherever a macroparticle happened to
    # reach, and a band stays empty while it is in the far tail.  Reading the
    # turn-on as a physical arrival would be wrong -- 1 MeV photons cross the
    # domain ballistically in tens of ns -- so shade it and say so.
    # Titles stay one line: the population-not-flux caveat and the occupancy
    # explanation are stated ONCE in the figure caption instead of three times
    # here, where they overran the canvas and were clipped.
    note = ""
    filled = np.flatnonzero(n_band > 0.0)
    if filled.size and filled[0] > 0:
        ax.axvspan(t_band_us[0], t_band_us[filled[0]], color="0.6", alpha=0.18)
        note = f"; shaded {int(filled[0])}/{n_band.size} frames empty"
    if log:
        ax.set_yscale("log")
    title, y_label = band_label(band, "photon" if noun == "photons" else "electron")
    ax.set_title(
        title
        + (", log" if log else "")
        + f"; peak {n_band[i_pk]:.2e} at t = {t_band_us[i_pk]:.1f} us"
        f"{note})",
        loc="left",
    )
    ax.set_ylabel(y_label)


def ground_theta_field(series: dict, zenith_rad: float) -> np.ndarray:
    """E_theta at the observer from the reduced Cartesian field.

    theta_hat = (cos, 0, -sin) with theta from +z and the observer on +x, the
    same angle convention dipole_field_terms uses, so the exact and
    compact-source ground curves are the SAME component of the same field.
    """
    return (series["ex_v_per_m"] * math.cos(zenith_rad)
            - series["ez_v_per_m"] * math.sin(zenith_rad))


def draw_lip_panel(
    ax,
    t_us: np.ndarray,
    e_z: np.ndarray,
    e_z_static: np.ndarray,
    note: str,
    reference: np.ndarray | None = None,
    static_label: str = "static term",
) -> None:
    """The LIP field-mill panel: vertical field at nadir, raw and static.

    A rotating-vane mill reads the ABSOLUTE field including DC, so nothing
    here passes through rc_highpass (whose y[0] = 0 deliberately kills
    steps) or any low-pass: the event is far shorter than one 20 ms sample,
    so what the 50 S/s record shows is the quasi-static step between the
    samples bracketing it -- the final static level, drawn as the dotted
    line.  The raw curve is the summed incident field the mill cannot
    resolve.  Sign is the physics convention (+z up): a negative moment
    steps the field down.  `reference`, when given, is the point-dipole
    E_z overlaid so the finite-source geometry correction is visible.
    """
    ax.plot(t_us, e_z, lw=1.0, color="#17becf", label="E$_z$ (raw)")
    ax.plot(t_us, e_z_static, lw=0.8, ls="--", color="#1f77b4",
            label=static_label)
    if reference is not None:
        ax.plot(t_us, reference, lw=0.8, ls="--", color="0.55",
                label="point dipole")
    # Read the step at the last finite supported sample; the masked tail is NaN.
    finite = np.flatnonzero(np.isfinite(e_z_static))
    step = float(e_z_static[finite[-1]]) if finite.size else float("nan")
    ax.axhline(step, ls=":", lw=1.0, color="#d62728")
    ax.set_ylabel("E$_z$ (V/m)")
    ax.legend(loc="upper left", fontsize=8, ncols=3)
    ax.set_title(
        f"LIP field mills, vertical E$_z$ at nadir: raw peak "
        f"{np.nanmax(np.abs(e_z)):.2g} V/m; quasi-static step {step:.2g} V/m "
        f"= what a {LIP_SAMPLE_HZ:.0f} S/s mill records (event << one "
        f"sample){note}",
        loc="left",
    )


def main() -> None:
    parser = figure_argument_parser(__doc__, smooth_us_default=0.5)
    parser.add_argument("--min-frame-macros", type=int, default=10)
    parser.add_argument("--title", default=None)
    parser.add_argument("--layout", choices=("tall", "print"), default="print",
                        help="'print' (default) uses two columns; 'tall' uses one")
    parser.add_argument("--distance-km", type=float, default=100.0,
                        help="HORIZONTAL distance along the ground to the "
                             "LF/VLF receiver (slant range and zenith angle "
                             "are derived from the source altitude)")
    parser.add_argument("--observer-altitude-km-msl", type=float,
                        default=OBSERVER_ALTITUDE_M_MSL / 1e3,
                        help="observer MSL altitude for the EFCM and optical "
                             "panels (default %(default)g km: the ER-2 "
                             "aircraft)")
    parser.add_argument("--efcm-offset-km", type=float, default=5.0,
                        help="HORIZONTAL distance the source sits AHEAD of "
                             "the aircraft along heading (default "
                             "%(default)g km, the illustrative geometry). "
                             "The EFCM plate normal points along heading "
                             "(Kochkin et al. 2021), so the instrument has "
                             "an EXACT null for a vertical dipole at nadir "
                             "-- 0 shows that null, not a prediction")
    parser.add_argument("--optical-distance-km", type=float, default=None,
                        help="observer distance for the optical irradiance "
                             "panel ONLY; by default it uses the nadir/LIP "
                             "geometry (ER-2 directly overhead)")
    add_cloud_arguments(parser)
    args = parser.parse_args()

    series = current_moment_series(args.capture_dir, args.min_frame_macros)
    t_s, dt = series["t_s"], series["dt_s"]
    m_smooth = boxcar(series["m_moment"], dt, args.smooth_us * 1e-6)
    t_us = t_s * 1e6
    optical = optical_series(args.capture_dir)
    scattered = (
        scattered_irradiance(
            optical,
            observer_alt_m=args.observer_altitude_km_msl * 1e3,
            n_photons=args.cloud_photons,
            cloud_top_m=args.cloud_top_km * 1e3)
        if args.cloud_scattering else None)

    # One emission-weighted source altitude sets all three geometries: the
    # EFCM (aircraft, source ahead along heading), the LIP + optical panels
    # (aircraft at nadir -- LIP reads the vertical field there and FEGS is
    # nadir-viewing), and the ground receiver's slant path.
    src_alt_m, efcm_r_m, efcm_zenith, efcm_label = observer_geometry(
        optical["alt_m"], optical["r337_lo"],
        args.observer_altitude_km_msl * 1e3, args.efcm_offset_km * 1e3,
    )
    _, lip_r_m, lip_zenith, lip_label = observer_geometry(
        optical["alt_m"], optical["r337_lo"],
        args.observer_altitude_km_msl * 1e3, 0.0,
    )
    _, ground_r_m, ground_zenith, _ = observer_geometry(
        optical["alt_m"], optical["r337_lo"], 0.0, args.distance_km * 1e3,
    )
    efcm_label = f"ER-2 {efcm_label}"
    lip_label = f"ER-2 {lip_label}"
    # Waveforms and spectra come from the Huygens/Love surface integral: the
    # exact retarded field of the finite source, with no compact-source
    # assumption. The point dipole stays as the dashed reference on the LF
    # panel, and carries the panels outright for a capture whose host-side
    # reduction has not been run -- every title names which one is drawn.
    aircraft_alt_m = args.observer_altitude_km_msl * 1e3
    observers = {
        tag: load_observer_series(args.capture_dir, tag, reference_m,
                                  offset_m=offset_m, altitude_msl_m=alt_m)
        for tag, reference_m, offset_m, alt_m in (
            ("ground", ground_r_m, args.distance_km * 1e3, 0.0),
            ("aircraft", efcm_r_m, args.efcm_offset_km * 1e3, aircraft_alt_m),
            ("nadir", lip_r_m, 0.0, aircraft_alt_m))
    }
    e_dipole = dipole_field_terms(
        m_smooth, dt, ground_r_m, ground_zenith, 1.0
    )["e_theta"]
    ground_obs = observers["ground"]
    if ground_obs is None:
        t_ground_us, dt_ground, e_raw = t_us, dt, e_dipole
        ground_note = "point dipole"
    else:
        # E_theta from the Cartesian field: theta_hat = (cos, 0, -sin) with
        # theta from +z, the same angle convention the dipole terms use.
        t_ground_us = ground_obs["t_source_s"] * 1e6
        dt_ground = ground_obs["dt_s"]
        e_raw = ground_theta_field(ground_obs, ground_zenith)
        ground_note = "Huygens/Love surface integral"
    spec_f, spec_a = amplitude_spectrum(e_raw, dt_ground, "hann")
    _, spec_phi = spectral_energy_density(e_raw, dt_ground, "none")
    freq = np.fft.rfftfreq(len(e_raw), dt_ground)

    ground = source_context(series, ground_r_m, ground_zenith)
    aircraft = source_context(series, efcm_r_m, efcm_zenith)
    print(ground["summary"])
    print(f"  at the aircraft ({efcm_label}): point-dipole error "
          f"(L/R)^2 = {aircraft['dipole_error']:.3g}")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Panel titles carry the validity block, so they are long: at 9 pt they
    # overrun a half-width panel and collide with the next column.  One knob,
    # not eight literals -- the titles inherit it.
    plt.rcParams["axes.titlesize"] = 5.4 if args.layout == "print" else 9.0
    if args.layout == "print":
        rows = max(PANEL_EFCM_FIRST, PANEL_COUNT - PANEL_EFCM_FIRST)
        fig, grid = plt.subplots(rows, 2, figsize=(16.0, 1.7 * rows),
                                 constrained_layout=True)
        axes = list(grid[:PANEL_EFCM_FIRST, 0]) + list(grid[:, 1])
        for spare in grid[PANEL_EFCM_FIRST:, 0]:
            spare.set_axis_off()
        # The odd panel count leaves one cell empty; the footnote goes THERE
        # rather than under the figure, where two columns leave it no room and
        # it lands on the bottom panel's axis labels.
        caption_ax = grid[PANEL_EFCM_FIRST, 0]
    else:
        caption_ax = None
        fig, axes = plt.subplots(
            PANEL_COUNT, 1, figsize=(11.5, 2.6 * PANEL_COUNT),
            constrained_layout=True,
        )
    fig.suptitle(
        (args.title or run_label(args.capture_dir))
        + f"\nEFCM source {args.efcm_offset_km:.0f} km ahead; LIP + optical "
        f"at the {lip_label}; LF/VLF at ground, "
        f"{args.distance_km:.0f} km horizontal; source time throughout",
        fontsize=11,
    )

    # All three source lightcurves are band populations, read from the same
    # CSVs the video's strip draws, so figure and video cannot show different
    # quantities under the same name.  Each is a POPULATION inside the band,
    # never a flux through it: the frame stream stores densities with no
    # direction, so a crossing rate is not recoverable from it.
    electron_band = load_band_series(args.capture_dir, "electron")
    photon_band = load_band_series(args.capture_dir, "photon")
    t_opt_us = optical["t_s"] * 1e6
    shared_time_xlim_us = shared_source_time_limits_us(
        t_us,
        t_opt_us,
        None if electron_band is None else electron_band["time_s"] * 1e6,
        None if photon_band is None else photon_band["time_s"] * 1e6,
    )
    draw_band_panel(axes[PANEL_LIGHTCURVE], electron_band,
                    noun="electrons", color="#ff7f0e", log=False)
    draw_band_panel(axes[PANEL_LIGHTCURVE_LOG], electron_band,
                    noun="electrons", color="#ff7f0e", log=True)
    draw_band_panel(axes[PANEL_PHOTON_BAND], photon_band,
                    noun="photons", color="#2ca02c", log=True)

    ax = axes[PANEL_OPTICAL_RATES]
    ax.fill_between(t_opt_us, np.clip(optical["r337_lo"], 1.0, None),
                    np.clip(optical["r337_hi"], 1.0, None),
                    color="#1f77b4", alpha=0.35)
    ax.plot(t_opt_us, optical["r337_lo"], lw=1.0, color="#1f77b4",
            label="337.1 nm N$_2$ 2P(0,0)")
    ax.fill_between(t_opt_us, np.clip(optical["r777_lo"], 1.0, None),
                    np.clip(optical["r777_hi"], 1.0, None),
                    color="#d62728", alpha=0.25)
    ax.plot(t_opt_us, optical["r777_mid"], lw=1.0, color="#d62728",
            label="777.4 nm OI (x/3)")
    ax.set_yscale("log")
    lo_floor = max(np.max(optical["r337_hi"]) / 1e8, 1.0)
    ax.set_ylim(bottom=lo_floor)
    ax.set_ylabel("ph/s")
    ax.legend(loc="lower right", fontsize=8, ncols=2)
    i_opt = int(np.argmax(optical["r337_lo"]))
    ax.set_title(
        f"optical source rates (ASIM/MMIA bands; peak 337 = "
        f"{optical['r337_lo'][i_opt]:.2g} ph/s at {t_opt_us[i_opt]:.1f} us, "
        f"337/777 ~ {optical['r337_lo'][i_opt]/max(optical['r777_mid'][i_opt],1.0):.0f}; "
        "degradation term + Xu-2015 field-enhancement estimate)",
        loc="left",
    )

    ax = axes[PANEL_OPTICAL_IRRADIANCE]
    if args.optical_distance_km is not None:
        dist_m = args.optical_distance_km * 1e3
        dist_label = f"{args.optical_distance_km:.0f} km"
    else:
        # Nadir, NOT the EFCM's ahead-of-source geometry: FEGS looks straight
        # down; coupling this to --efcm-offset-km would make an unrelated
        # instrument geometry change the optical panel.
        dist_m, dist_label = lip_r_m, lip_label
    irr_337 = E_PHOTON_337_J / (4.0 * math.pi * dist_m**2) * 1e3  # mW/m^2 per ph/s
    irr_777 = E_PHOTON_777_J / (4.0 * math.pi * dist_m**2) * 1e3
    ax.fill_between(t_opt_us, np.clip(optical["r337_lo"] * irr_337, 1e-9, None),
                    np.clip(optical["r337_hi"] * irr_337, 1e-9, None),
                    color="#1f77b4", alpha=0.35, label="337.1 nm")
    ax.fill_between(t_opt_us, np.clip(optical["r777_lo"] * irr_777, 1e-9, None),
                    np.clip(optical["r777_hi"] * irr_777, 1e-9, None),
                    color="#d62728", alpha=0.25, label="777.4 nm")
    if scattered is not None:
        # The transport observer is the nadir aircraft geometry (lip_r_m),
        # regardless of any --optical-distance-km override on the free-space
        # curves, so the constant delay removed must be lip_r_m/c.
        captured_fraction, irradiance_end_us = draw_scattered_overlay(
            ax, scattered, shared_time_xlim_us,
            lip_r_m / C_LIGHT * 1e6)
    ax.set_ylim(bottom=0.0)
    ax.set_ylabel("mW/m$^2$")
    ax.legend(loc="upper right", fontsize=7, ncols=2)  # over the low tail only
    ax.set_title(
        f"optical irradiance at {dist_label}, isotropic source, "
        "no atmosphere/scattering"
        if scattered is None else
        f"optical irradiance at {dist_label}: free space (light) vs cloudscat "
        f"transport (dark); arrival minus d/c = "
        f"{lip_r_m / C_LIGHT * 1e6:.0f} us, scattering lag kept; "
        f"{captured_fraction * 100.0:.0f}% of scattered fluence within "
        f"{shared_time_xlim_us[1]:.0f} us; axis runs to {irradiance_end_us:.0f} us "
        "(90% arrived) -- this panel's time axis is longer than the others",
        loc="left",
    )

    # Aircraft fields.  When the z-resolved dM/dz side-car exists, sum the
    # finite source bin by bin -- removing the (L_z/R)^2 point-dipole
    # geometry error that is of order unity at the aircraft standoff -- and
    # gate on charge conservation: the bins must reproduce the reduced-CSV
    # moment, or the figure refuses rather than drawing an unsupported
    # source model. Captures without the side-car fall back to the point dipole
    # with its error quoted.
    profile_path = args.capture_dir / "rrea_video" / "current_profile.csv"
    z0_m = capture_altitude_z0_m(args.capture_dir, None)
    if profile_path.is_file():
        profile = load_current_profile(profile_path)
        fs_kwargs = dict(
            t_s=t_s, dt=dt, z0_alt_msl_m=z0_m,
            observer_alt_msl_m=args.observer_altitude_km_msl * 1e3,
            smooth_window_s=args.smooth_us * 1e-6,
        )
        fs_efcm = finite_source_fields(
            profile, offset_m=args.efcm_offset_km * 1e3,
            retard_ref_m=efcm_r_m, **fs_kwargs)
        fs_lip = finite_source_fields(
            profile, offset_m=0.0, retard_ref_m=lip_r_m, **fs_kwargs)
        m_err = float(np.max(np.abs(fs_lip["m_check"] - series["m_moment"])))
        m_ref = float(np.max(np.abs(series["m_moment"])))
        # 1%, not tighter: the profile and the reduced CSV sit on different
        # cadences, so interpolation error is legitimate -- a broken bin sum
        # is not.
        if m_ref > 0.0 and m_err > 0.01 * m_ref:
            raise SystemExit(
                "current_profile.csv bins do not reproduce the reduced-CSV "
                f"moment (max error {m_err:.3g} vs peak {m_ref:.3g} A*m); "
                "refusing the finite-source model")
        # Each species' own plate field, through the SAME geometry: which
        # current dominates a channel is then a measurement rather than an
        # assertion, and it is band-resolved because a channel IS a band.
        efcm_species = {}
        for name in PROFILE_SPECIES:
            solo = dict(profile)
            for other in PROFILE_SPECIES:
                if other != name:
                    solo[other] = np.zeros_like(profile[other])
            efcm_species[name.split("_")[1]] = -finite_source_fields(
                solo, offset_m=args.efcm_offset_km * 1e3,
                retard_ref_m=efcm_r_m, **fs_kwargs)["e_h"]
        e_efcm = -fs_efcm["e_h"]  # plate normal along heading, source ahead
        e_lip, e_lip_static = fs_lip["e_z"], fs_lip["e_z_static"]
        lip_reference = lip_vertical_field(m_smooth, dt, lip_r_m, lip_zenith)
        model_note = "z-resolved line dipole"
        efcm_valid = fs_efcm["valid_interval_s"]
        lip_valid = fs_lip["valid_interval_s"]
        smooth_taps = fs_efcm["smooth_taps"]
        smooth_corner_hz = fs_efcm["smooth_corner_hz"]
    else:
        e_efcm = efcm_plate_field(m_smooth, dt, efcm_r_m, efcm_zenith)
        terms = dipole_field_terms(m_smooth, dt, lip_r_m, lip_zenith)
        e_lip = lip_vertical_field(m_smooth, dt, lip_r_m, lip_zenith)
        e_lip_static = (
            terms["e_r_static"] * math.cos(lip_zenith)
            - terms["e_theta_static"] * math.sin(lip_zenith)
        )
        lip_reference = None
        model_note = "point dipole"
        efcm_species = {}
        # No per-bin retardation, so every sample is answerable from the
        # record and there is nothing to mask.
        efcm_valid = lip_valid = None
        smooth_taps = boxcar_kernel_taps(dt, args.smooth_us * 1e-6, len(t_s))
        smooth_corner_hz = boxcar_corner_hz(dt, smooth_taps)

    # The line-dipole model keeps ONE job here: the per-species decomposition.
    # The Huygens surface carries only the TOTAL field, so which current drives
    # a channel is a question only a modelled source can answer.
    e_efcm_dipole, dipole_valid, dt_dipole = e_efcm, efcm_valid, dt
    efcm_t_s, dt_efcm, lip_t_s = t_s, dt, t_s
    efcm_note = lip_note = model_note
    lip_static_label, lip_relaxation = "static term", None
    if observers["aircraft"] is not None:
        aircraft_obs = observers["aircraft"]
        efcm_t_s, dt_efcm = aircraft_obs["t_source_s"], aircraft_obs["dt_s"]
        # Plate normal along heading, source ahead: the CSV column is the
        # horizontal field at plate azimuth 0, and the panel sign convention is
        # the negative of it, as the line-dipole path also takes.
        e_efcm = -aircraft_obs["ex_v_per_m"]
        # Already truncated to the window every surface element can answer.
        efcm_valid = None
        efcm_note = HUYGENS_NOTE
    if observers["nadir"] is not None:
        nadir_obs = observers["nadir"]
        lip_t_s = nadir_obs["t_source_s"]
        e_lip = nadir_obs["ez_on_axis_total_v_per_m"]
        e_lip_static = nadir_obs["ez_undifferentiated_kernel_v_per_m"]
        lip_valid = None
        lip_note = HUYGENS_NOTE
        lip_static_label = "1/R$^2$ kernel"
        lip_relaxation = nadir_obs["ez_retarded_derivative_kernel_v_per_m"]
        if lip_reference is not None:
            lip_reference = np.interp(lip_t_s, t_s, lip_reference)

    lip_last = last_supported_index(lip_t_s, lip_valid)
    print(
        f"aircraft field model: {efcm_note}; EFCM theta = "
        f"{math.degrees(efcm_zenith):.1f} deg, plate factor sin*cos = "
        f"{math.sin(efcm_zenith) * math.cos(efcm_zenith):.2f} (max 0.5 at "
        f"45 deg); LIP nadir R = {lip_r_m / 1e3:.1f} km, quasi-static step = "
        f"{float(e_lip_static[lip_last]):.3g} V/m at t = "
        f"{lip_t_s[lip_last] * 1e6:.1f} us"
        + ("" if lip_relaxation is None else
           f" (retarded-derivative kernels are "
           f"{abs(float(lip_relaxation[lip_last])) / max(abs(float(e_lip[lip_last])), 1e-300):.1%}"
           " of E_z there, so the step is quasi-static only to that much)")
        + "".join(
            f"; {name} retardation-supported window "
            f"{window[0] * 1e6:.1f}-{window[1] * 1e6:.1f} us"
            for name, window in (("EFCM", efcm_valid), ("LIP", lip_valid))
            if window is not None)
    )

    # EFCM (ALOFT field-change meter): the ALONG-HEADING plate component at
    # the aircraft through each channel's RC response and digitizer-Nyquist
    # low-pass, then differentiated -- the instrument records the derivative
    # of the field impulse. The t = 0 baseline (p = 0, rc_highpass y[0] = 0)
    # deliberately drops the artificial seed-injection step (~1e-6 of peak M).
    on_axis = args.efcm_offset_km == 0.0
    record_floor_hz = 1.0 / (efcm_t_s[-1] - efcm_t_s[0])
    for i_ax, (chan, tau, f_ny) in enumerate(EFCM_CHANNELS):
        ax = axes[PANEL_EFCM_FIRST + i_ax]
        # Y is the field-change response the instrument physically produces:
        # a flat plate reads the field, and the RC leaks its baseline away.
        # dY/dt is a derived view of it, NOT the incident dE/dt -- recovering
        # that needs dE/dt = dY/dt + Y/tau, and neither is a hardware voltage
        # because gain G = 1 here (uncalibrated, field-equivalent units).
        y_chan = apply_lowpass(
            rc_highpass(e_efcm, tau, dt_efcm), f_ny, dt_efcm)
        dydt_chan = np.gradient(y_chan, dt_efcm)
        y_plot = mask_unsupported(y_chan, efcm_t_s, efcm_valid)
        dydt_plot = mask_unsupported(dydt_chan, efcm_t_s, efcm_valid)
        color = "#8c564b" if chan == "slow" else "#e377c2"
        ax.plot(efcm_t_s * 1e6, y_plot, lw=0.9, color=color)
        ax.set_ylabel("Y$_\\parallel$ (V/m, G=1)")
        twin = ax.twinx()
        twin.plot(efcm_t_s * 1e6, dydt_plot, lw=0.7, ls="--", color="0.45")
        twin.set_ylabel("dY$_\\parallel$/dt (V/m/s)", fontsize=8, color="0.45")
        twin.tick_params(labelsize=7, colors="0.45")
        f_hp = 1.0 / (2.0 * math.pi * tau)
        # Name the numerical limit that actually governs this nominal channel.
        limits = [("digitizer Nyquist", f_ny),
                  ("diagnostic cadence", 0.5 / dt_efcm)]
        if efcm_note != HUYGENS_NOTE:
            # The boxcar lives on M(t); the surface integral never saw it.
            limits.append((f"{smooth_taps}-tap smoothing", smooth_corner_hz))
        top_name, f_top = min(limits, key=lambda limit: limit[1])
        if efcm_species:
            # Inside the supported observer window only, and as an rms share: a
            # peak-time share answers a different question, because the peak
            # of a 150 us record is a low-frequency feature whichever channel
            # carries it.  Shares need not sum to 100% -- the components
            # partly cancel, which is itself worth seeing.
            y_ref = apply_lowpass(
                rc_highpass(e_efcm_dipole, tau, dt_dipole), f_ny, dt_dipole)
            inside = np.isfinite(mask_unsupported(y_ref, t_s, dipole_valid))

            def share(signal: np.ndarray, total: np.ndarray) -> float:
                ref = float(np.sqrt(np.mean(total[inside] ** 2)))
                if ref <= 0.0:
                    return float("nan")
                return 100.0 * float(np.sqrt(np.mean(signal[inside] ** 2))) / ref

            parts = []
            for sp, e_sp in efcm_species.items():
                y_sp = apply_lowpass(
                    rc_highpass(e_sp, tau, dt_dipole), f_ny, dt_dipole)
                parts.append(
                    f"{sp} {share(y_sp, y_ref):.0f}%/"
                    f"{share(np.gradient(y_sp, dt_dipole), np.gradient(y_ref, dt_dipole)):.0f}%")
            print(f"  EFCM {chan} channel, per-species rms share of "
                  f"Y/(dY/dt), line-dipole model: " + ", ".join(parts))
        tau_label = (f"{tau * 1e6:.4g} us" if tau < 1e-3
                     else f"{tau * 1e3:.4g} ms")
        # Two lines, deliberately: one long line clipped mid-word at the
        # canvas edge, silently dropping the model attribution.
        ax.set_title(
            f"EFCM {chan} channel: field-change response Y, plate along "
            f"heading (dY/dt dashed; G = 1, uncalibrated)\n"
            f"nominal RC tau {tau_label} ({f_hp:.3g} Hz) | record floor "
            f"{record_floor_hz / 1e3:.1f} kHz | usable to "
            f"{f_top / 1e3:.0f} kHz ({top_name}) | peak "
            f"{np.nanmax(np.abs(y_plot)):.2g} V/m, "
            f"{np.nanmax(np.abs(dydt_plot)):.2g} V/m/s | "
            + ("plate component identically 0 at nadir (model null)"
               if on_axis and efcm_note != HUYGENS_NOTE else
               f"theta {math.degrees(efcm_zenith):.0f} deg, {efcm_note}"),
            fontsize=8, loc="left",
        )
    draw_lip_panel(
        axes[PANEL_LIP], lip_t_s * 1e6,
        mask_unsupported(e_lip, lip_t_s, lip_valid),
        mask_unsupported(e_lip_static, lip_t_s, lip_valid),
        note=f"; {lip_note}", reference=lip_reference,
        static_label=lip_static_label,
    )
    for i_band, (label, f_lo, f_hi, color) in enumerate(BANDS):
        e_band = np.fft.irfft(
            np.fft.rfft(e_raw) * butterworth_band_mask(freq, f_lo, f_hi), n=len(e_raw)
        )
        ax = axes[PANEL_RADIO_FIRST + i_band]
        ax.plot(t_ground_us, e_band, lw=0.9, color=color)
        note = ""
        if ground_obs is not None:
            # Compact-source reference through the same band: a geometry and
            # time-shift check below the radial corner in the source context
            # (~20 kHz, inside VLF); above it the gap is physics, not a fault.
            e_ref = np.fft.irfft(
                np.fft.rfft(e_dipole)
                * butterworth_band_mask(np.fft.rfftfreq(len(e_dipole), dt),
                                        f_lo, f_hi), n=len(e_dipole))
            ax.plot(t_us, e_ref, lw=0.7, ls="--", color="0.45",
                    label=f"point dipole (|E|_peak = {np.max(np.abs(e_ref)):.2g} V/m)")
            ax.legend(loc="upper right", fontsize=7)
        if (t_ground_us[-1] - t_ground_us[0]) * 1e-6 < 2.0 / f_lo:
            note += "  [record < 2 periods of band floor]"
        ax.set_title(
            f"{label} free-space at ground geometry, slant R = {ground_r_m/1e3:.1f} km "
            f"(|E|_peak = {np.max(np.abs(e_band)):.2g} V/m; {ground_note}){note}",
            loc="left",
        )
        ax.set_ylabel("E (V/m)")
    for ax in axes[:-1]:
        ax.set_xlim(*shared_time_xlim_us)
        ax.grid(alpha=0.25)
    if scattered is not None:  # the scattering lag outlives the source
        axes[PANEL_OPTICAL_IRRADIANCE].set_xlim(right=irradiance_end_us)
    axes[-2].set_xlabel("source time (us)")

    ax = axes[PANEL_SPECTRA]
    valid = spec_f > 0
    ax.loglog(spec_f[valid], spec_a[valid], lw=0.9, color="#444444",
              label="|E(f)| (left)")
    ax.set_ylabel("|E(f)| (V/m/Hz)")
    twin = ax.twinx()
    twin.loglog(spec_f[valid], np.maximum(spec_phi[valid], 1e-30), lw=1.1,
                ls="--", color="#8c564b",
                label="$\\Phi$ = 2 c $\\epsilon_0$ |E(f)|$^2$ (right)")
    # Dwyer & Cummer Figs 4-5: their model curves span 1e-19..1e-15 J/Hz/m^2
    # at 500 km, which is what makes Phi the model discriminator. Phi ~ 1/R^2,
    # so scale their decades to this receiver range to make the band readable
    # against our curve.
    dc_scale = (500e3 / ground_r_m) ** 2
    twin.axhspan(1e-19 * dc_scale, 1e-15 * dc_scale, color="#8c564b", alpha=0.10)
    twin.set_ylabel("$\\Phi$(f) (J/Hz/m$^2$)", color="#8c564b")
    twin.tick_params(colors="#8c564b")
    for _label, f_lo, f_hi, color in BANDS:
        ax.axvspan(f_lo, f_hi, color=color, alpha=0.12)
    ax.set_xlim(spec_f[valid][0], min(4e6, 0.5 / dt_ground))
    ax.set_xlabel("frequency (Hz)")
    handles = ax.get_legend_handles_labels()
    twin_handles = twin.get_legend_handles_labels()
    ax.legend(handles[0] + twin_handles[0], handles[1] + twin_handles[1],
              loc="lower left", fontsize=8)
    ax.set_title(
        "spectra of the ground E_theta: amplitude (Hann) and plane-wave energy "
        "estimate; shaded band = the D&C Figs 4/5 model decades (their eq. 32, "
        "free space, as the curves are), scaled 1/R$^2$ to "
        f"{ground_r_m/1e3:.0f} km; {ground_note}",
        loc="left",
    )
    ax.grid(alpha=0.25, which="both")

    # constrained_layout does not know about fig.text or, once a rect is set,
    # about the suptitle: reserve both strips explicitly.
    fig.get_layout_engine().set(rect=(0.0, 0.022, 1.0, 0.982))
    # The field-model scope belongs on the figure for the same reason the
    # point-dipole and R/c corners already do: it bounds when M(t) itself is
    # trustworthy, not merely how it maps onto a receiver.  The ONE scope
    # object gated at startup backs this caption, the terminal report, and
    # the sidecar -- never recompute a second status here.
    band_note = (
        " Source lightcurves are the POPULATION above 12 km MSL (the whole "
        "domain there), never a flux through a surface: the frame stream "
        "stores densities with no direction. A shaded leading run is a "
        "macroparticle OCCUPANCY floor, not an empty region -- the outermost "
        "nonzero cell sits wherever a macroparticle reached, so the count reads "
        "exactly zero until the population grows into the region, and is "
        "shot-noise dominated just after. Each panel title carries its own "
        "altitude range, radius and row count."
    )
    caption = (
        "M(t) from exact recorded currents (kinetic e- incl. sub-MeV, "
        f"positrons, fluid conduction: {ground['fluid_fraction']*100:.0f}% of "
        "the peak). EFCM is a flat plate pointed along heading (Kochkin et "
        "al. 2021, doi:10.1029/2020JD033467): it reads the along-heading "
        "horizontal component (+ = forward, source ahead), with an EXACT "
        "null for a vertical dipole at nadir -- the vertical-at-nadir "
        "prediction is the LIP panel, and its nominal RC constants carry "
        "Kochkin's caveat that the bandwidth is not understood beyond "
        f"fast/slow. Aircraft field model: {efcm_note}"
        + (", the exact retarded field of the finite source: no compact-source "
           "geometry error remains on the aircraft panels"
           if efcm_note == HUYGENS_NOTE else
           ", which removes the (L_z/R)^2 point-dipole geometry error; the "
           "radial collapse to the axis "
           f"((2*rms_r/R)^2 = "
           f"{(2.0 * series['source_geometry']['peak_rms_r_m'] / lip_r_m) ** 2:.2g}"
           ") remains"
           if model_note == "z-resolved line dipole" else
           f": (L/R)^2 = {aircraft['dipole_error']:.2g} (span; (2*rms_z/R)^2 "
           f"= {aircraft['dipole_error_rms']:.2g}) at the aircraft vs "
           f"{ground['dipole_error']:.2g} at the ground receiver -- near 1 "
           "means an order-unity geometry error on the aircraft panels")
        + ". The line-source cross-check resolves z but collapses radial "
        "extent. Retardation is NOT applied to the current-moment panels: add R/c = "
        f"{efcm_r_m/C_LIGHT*1e6:.0f} us (aircraft) or "
        f"{ground_r_m/C_LIGHT*1e6:.0f} us (ground); the finite-source model "
        "applies the per-bin DIFFERENTIAL delay only. Amplitudes scale with "
        f"this run's seed normalization: implied N_re = {ground['n_re']:.2e} "
        f"= {ground['n_re_ratio']:.0f}x an average TGF. EFCM constants: ALOFT "
        "instrument page; method: Dwyer & Cummer, JGR 2013, "
        "doi:10.1002/jgra.50188."
        if series["source_geometry"] is not None else
        "M(t) from recorded currents; source geometry unavailable, so the "
        "point-dipole limit is not quantified here."
    )
    if args.layout == "print":
        # A half-width panel cannot hold the longest validity lines on one
        # line.  Re-wrap here, once, rather than at the eight title sites --
        # and per existing line, so the deliberate breaks survive.
        for ax in axes:
            title = ax.get_title(loc="left")
            if title:
                ax.set_title("\n".join(textwrap.fill(line, 140)
                                       for line in title.split("\n")),
                             loc="left")
    if caption_ax is not None:
        # wrap=True wraps to the FIGURE width, which in two columns runs the
        # text under the other column and squeezes the panels; wrap to the
        # cell explicitly instead.
        caption_ax.text(0.0, 1.0,
                        textwrap.fill(caption + band_note, 155),
                        transform=caption_ax.transAxes, fontsize=5.5,
                        va="top", ha="left", linespacing=1.35)
    else:
        fig.text(
            0.005, 0.001, caption + band_note,
            fontsize=6.5, va="bottom", wrap=True,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
