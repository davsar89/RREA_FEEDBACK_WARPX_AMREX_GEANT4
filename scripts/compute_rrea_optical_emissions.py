"""Optical source and CloudScat observer curves for one RREA capture.

The 337.1-nm baseline is the Rosado dry-air fluorescence yield, scaled by
local pressure and temperature.  Xu et al.'s 300--430-nm enhancement is kept
only as a heuristic scenario inside its published domain (E/N >= 16 Td);
below 16 Td the baseline is used unchanged.  The 777.4-nm estimate remains an
order-of-magnitude cold-air scenario.

The native positive-ion creation source supplies both the ion-pair rate and
its source-weighted altitude.  Positive-ion density changes are deliberately
not used: drift or recombination can change a density without creating light.

CloudScat is used unchanged.  Its observer timeline is retained as absolute
arrival time, including each source bin's own propagation delay.  No global
emission-weighted travel-time shift is applied.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

from rrea_cloud_scattering import (
    OBSERVER_ALTITUDE_M_MSL,
    emission_weighted_altitude,
)
from compute_rrea_radio_waveform import (
    add_cloud_arguments,
    as_float,
    boxcar,
    figure_argument_parser,
    read_csv_rows,
)
from rrea_video_contract import dedup_reduced_rows_by_step
from rrea_profiled_atmosphere import ALTITUDE_AT_Z0_M
from rrea_run_support import C, K_BOLTZ

# ---------------------------------------------------------------------------
# Constants (provenance in the module docstring)
# Keep numerically identical to rrea::mean_energy_per_ion_pair_air_eV; the
# native constant owns the physics and this analysis copy only converts units.
W_AIR_EV = 33.97  # eV per ion pair in dry air, ICRU Report 31 (1979)
X_N2 = 0.7809
X_O2 = 0.2095

# 337.1 nm band (N2 2P(0,0))
FY337_REF_PH_PER_MEV = 7.04  # at 800 hPa, 293 K, dry air
FY337_REF_P_HPA = 800.0
FY337_REF_T_K = 293.0
P_PRIME_337_HPA = 15.9  # AIRFLY quenching reference pressure at 293 K
E_PHOTON_337_J = 5.894e-19  # h*c / 337.1 nm
# 337.1 nm PHOTON share of the 2P+1N 300-430 nm complex: 1.021 of 4.05 ph/m
# at 1013 hPa, 20 C (Nagano et al., Astropart. Phys. 22 (2004) 235, Table 2).
# Turns the 337 nm photon yield into the band total used by the Xu anchor.
FRACTION_337_OF_300_430 = 0.252
# Minimum-ionizing dynamic friction force in air at STP.
DEDX_STP_MEV_PER_M = 0.220

# Xu et al. (2015) section 4.1: total fluorescence photons (300-430 nm,
# quenched, ground pressure) per runaway electron per meter in steady RREA.
# Xu's first point is 16 Td; there is deliberately no invented 0-Td anchor.
XU_EN_TD = np.array([16.0, 46.5, 69.9])

# 777.4 nm OI multiplet
SIGMA_777_CM2 = 4.23e-18  # e + O2 -> O(3p5P), 100 eV (Erdman & Zipf 1987)
SIGMA_ION_AIR_CM2 = 2.5e-16  # air ionization cross section at 100 eV
EXC_PER_PAIR_777 = X_O2 * SIGMA_777_CM2 / SIGMA_ION_AIR_CM2  # ~3.5e-3
UNCERTAINTY_FACTOR_777 = 3.0
A_777_PER_S = 3.69e7  # radiative rate of the OI 3p5P multiplet (tau ~ 27 ns)
KQ_N2_777_M3_S = 5.9e-16  # O(3p3P) proxies (Niemi et al. 2005)
KQ_O2_777_M3_S = 9.4e-16
E_PHOTON_777_J = 2.556e-19  # h*c / 777.4 nm


def load_reduced(path: Path) -> dict:
    ordered = dedup_reduced_rows_by_step(read_csv_rows(path), source=path)
    time_s = np.array([as_float(r, "time_s") for r in ordered])

    def col(key: str) -> np.ndarray:
        return np.nan_to_num(np.array([as_float(r, key, 0.0) for r in ordered]))

    required_source_columns = (
        "ion_pair_source_weight_cumulative",
        "ion_pair_source_z_weighted_m_cumulative",
    )
    missing = [key for key in required_source_columns if key not in ordered[0]]
    if missing:
        raise SystemExit(
            f"{path} predates the native optical source observable; missing "
            + ", ".join(missing)
        )
    source_weight = np.array([
        as_float(row, "ion_pair_source_weight_cumulative") for row in ordered
    ])
    source_z_weighted = np.array([
        as_float(row, "ion_pair_source_z_weighted_m_cumulative")
        for row in ordered
    ])
    if (not np.all(np.isfinite(source_weight))
            or np.any(np.diff(source_weight) < 0.0)
            or not np.all(np.isfinite(source_z_weighted))):
        raise SystemExit(f"{path}: cumulative native ion-pair source is invalid")

    out = {
        "time_s": time_s,
        "ion_pair_source_weight_cumulative": source_weight,
        "ion_pair_source_z_weighted_m_cumulative": source_z_weighted,
        "n_ge1MeV": col("N_kinetic_e_ge_1MeV_real"),
        "energy_loss_eV_cum": col("transport_energy_loss_eV"),
        "photon_absorbed_eV_cum": col("photon_absorbed_energy_eV"),
        "positron_loss_eV_cum": col("positron_transport_energy_loss_eV"),
        "soft_brems_eV_cum": col("photon_continuous_soft_brems_energy_eV"),
    }
    # ne_low-weighted |E|/E0_peak: prefer the largest recorded channel radius.
    # Missing column -> None: the caller must NOT assume an unscreened field.
    frac_key = None
    for key in ordered[0]:
        if key.startswith("profile_active_abs_e_over_e0_peak_ne_low_weighted_r"):
            radius = float(key.rsplit("_r", 1)[1].rstrip("m"))
            if frac_key is None or radius > frac_key[1]:
                frac_key = (key, radius)
    out["field_fraction"] = col(frac_key[0]) if frac_key else None
    out["field_fraction_column"] = (
        frac_key[0] if frac_key
        else "(missing; field screening unknown, Xu enhancement NOT applied)")
    # Ambient peak field as the engine recorded it (0/absent -> None).
    e0 = as_float(ordered[-1], "profile_e0_peak_v_per_m", 0.0)
    out["e0_peak_v_per_m"] = e0 if e0 > 0.0 else None
    return out


def load_profile(path: Path) -> dict:
    rows = read_csv_rows(path)
    alt_m = np.array([as_float(r, "altitude_km_MSL") * 1000.0 for r in rows])
    n_m3 = np.array([as_float(r, "neutral_number_density_cm3") * 1e6 for r in rows])
    temp_k = np.array([as_float(r, "temperature_K") for r in rows])
    if not np.all(np.diff(alt_m) > 0):
        raise SystemExit(f"{path}: altitudes not strictly increasing")
    return {"alt_m": alt_m, "n_m3": n_m3, "temp_k": temp_k}


def capture_altitude_z0_m(capture_dir: Path, override: float | None) -> float:
    if override is not None:
        return override
    meta = capture_dir / "rrea_video" / "video_capture_metadata.json"
    if meta.exists():
        return float(json.loads(meta.read_text())["altitude_msl_at_z0_m"])
    print(f"WARN: no capture altitude metadata; using configured z=0 altitude {ALTITUDE_AT_Z0_M:g} m")
    return ALTITUDE_AT_Z0_M


def fy337_ph_per_mev(p_hpa: np.ndarray, temp_k: np.ndarray) -> np.ndarray:
    """Quenched 337.1 nm yield: FY = Y0 / (1 + P/P'(T)).

    Quenching rate ~ N sigma v ~ P sigma(T) T^-0.5, so P'(T) ~ T^(0.5 - alpha)
    for sigma ~ T^alpha.  AIRFLY measured alpha_337.1 = -0.36 +/- 0.08 (Ave et
    al., NIM A 597, 50, 2008, Table 1), giving P' ~ T^0.86. A sqrt(T) law
    assumes alpha = 0 and overestimates the yield at altitude (their Fig. 4).
    """
    y0 = FY337_REF_PH_PER_MEV * (1.0 + FY337_REF_P_HPA / P_PRIME_337_HPA)
    p_prime = P_PRIME_337_HPA * (temp_k / FY337_REF_T_K) ** 0.86
    return y0 / (1.0 + p_hpa / p_prime)


# Field-free degradation baseline and the 0 Td anchor, derived from the yield
# model above so a change there cannot silently desynchronize the enhancement.
XU_FLOOR_PH_PER_M = float(
    DEDX_STP_MEV_PER_M
    * fy337_ph_per_mev(np.array([1013.25]), np.array([FY337_REF_T_K]))[0]
    / FRACTION_337_OF_300_430)
XU_YIELD_PH_PER_M = np.array([6.2, 10.7, 38.3])


def quench_factor_777(n_m3: np.ndarray) -> np.ndarray:
    quench = KQ_N2_777_M3_S * X_N2 * n_m3 + KQ_O2_777_M3_S * X_O2 * n_m3
    return A_777_PER_S / (A_777_PER_S + quench)


def enhancement_337(en_td: np.ndarray) -> np.ndarray:
    """Xu heuristic in its published domain; baseline below 16 Td."""
    ratio = XU_YIELD_PH_PER_M / XU_FLOOR_PH_PER_M
    en = np.asarray(en_td, dtype=float)
    result = np.ones_like(en)
    valid = en >= XU_EN_TD[0]
    result[valid] = np.exp(np.interp(
        np.clip(en[valid], XU_EN_TD[0], XU_EN_TD[-1]),
        XU_EN_TD,
        np.log(ratio),
    ))
    return result


def native_pair_source_series(
    t_s: np.ndarray,
    cumulative_weight: np.ndarray,
    cumulative_z_weighted_m: np.ndarray,
    smooth_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return source rate and mean z from cumulative native event moments."""
    t = np.asarray(t_s, dtype=float)
    weight = np.asarray(cumulative_weight, dtype=float)
    z_weighted = np.asarray(cumulative_z_weighted_m, dtype=float)
    if not (t.shape == weight.shape == z_weighted.shape) or t.size < 2:
        raise ValueError("native pair-source arrays must share a shape of at least two")
    dt_interval = np.diff(t)
    delta_weight = np.diff(weight)
    delta_z_weighted = np.diff(z_weighted)
    if (not np.all(np.isfinite(t)) or not np.all(dt_interval > 0.0)
            or not np.all(np.isfinite(weight))
            or not np.all(np.isfinite(z_weighted))
            or np.any(delta_weight < 0.0)):
        raise ValueError("native cumulative pair-source moments are invalid")
    midpoint = 0.5 * (t[:-1] + t[1:])
    interval_rate = delta_weight / dt_interval
    valid = delta_weight > 0.0
    if not np.any(valid):
        raise SystemExit("capture contains no positive native ion-pair source")
    mean_z_interval = delta_z_weighted[valid] / delta_weight[valid]
    mean_z = np.interp(
        t,
        midpoint[valid],
        mean_z_interval,
        left=mean_z_interval[0],
        right=mean_z_interval[-1],
    )
    rate = np.interp(
        t,
        midpoint,
        interval_rate,
        left=interval_rate[0],
        right=interval_rate[-1],
    )
    return np.clip(boxcar(rate, float(np.median(dt_interval)), smooth_s), 0.0, None), mean_z


def optical_series(
    capture_dir: Path,
    *,
    profile_csv: Path | None = None,
    z0_altitude_m: float | None = None,
    e0_peak_kv_per_m: float | None = None,
    smooth_us: float = 1.0,
    verbose: bool = True,
) -> dict:
    """Compute the 337.1 / 777.4 nm source-rate series for one capture.

    e0_peak_kv_per_m=None (the default) uses the ambient peak field the
    capture itself recorded (profile_e0_peak_v_per_m in rrea_reduced.csv);
    an explicit value overrides it (a WARN is printed if they disagree).

    Returns the source, atmosphere, yield, and diagnostic series. The video
    renderer imports this function so optical outputs share one implementation.
    """
    if profile_csv is None:
        profile_csv = (Path(__file__).resolve().parent.parent
                       / "EFIELD_profile" / "rrea_threshold_NRLMSISE00_gulf_284kVpm.csv")
    profile = load_profile(profile_csv)
    z0_m = capture_altitude_z0_m(capture_dir, z0_altitude_m)

    reduced = load_reduced(capture_dir / "rrea_reduced.csv")

    t = reduced["time_s"]
    dt = float(np.median(np.diff(t)))

    # The native source is already a rate.  It is created at the interaction
    # point before fluid drift/recombination and is therefore the optical
    # driver; no state difference appears anywhere in this calculation.
    pairs_rate, source_mean_z_m = native_pair_source_series(
        t,
        reduced["ion_pair_source_weight_cumulative"],
        reduced["ion_pair_source_z_weighted_m_cumulative"],
        smooth_us * 1e-6,
    )
    dep_mev_s = pairs_rate * W_AIR_EV * 1e-6

    # Compact scalar bookkeeping check; it does not define the source.
    pair_count = float(np.trapezoid(pairs_rate, t))
    energy_loss = float(
        reduced["energy_loss_eV_cum"][-1] - reduced["energy_loss_eV_cum"][0]
    )
    ev_per_pair = energy_loss / max(pair_count, 1.0)
    if verbose:
        print(f"ledger check: transport energy loss / native ion-pair source = "
              f"{ev_per_pair:.2f} eV/pair (physical W = {W_AIR_EV}; the excess "
              f"is tracked-secondary kinetic energy booked at its Moller "
              f"event and again where it deposits, NOT bremsstrahlung, which "
              f"never enters this tally)")
    # Absorbed photons are already folded into the pair count; brems below the
    # table-defined cut is continuous drag, never re-deposited, so it is the one channel
    # genuinely MISSING from the fluorescence driver.
    total_loss_eV = max(energy_loss, 1.0)
    photon_share = (
        reduced["photon_absorbed_eV_cum"][-1]
        - reduced["photon_absorbed_eV_cum"][0]
    ) / total_loss_eV
    soft_brems_share = (
        reduced["soft_brems_eV_cum"][-1]
        - reduced["soft_brems_eV_cum"][0]
    ) / total_loss_eV
    if verbose:
        print(f"photon-deposited share of energy: {photon_share:.2e} "
              f"(folded into pairs); soft-brems drag share: "
              f"{soft_brems_share:.2e} (radiated, never re-deposited -- "
              f"yields are low by at most this much)")

    # Source altitude is the native positive-ion creation-rate centroid.
    alt_m = z0_m + source_mean_z_m
    n_m3 = np.interp(alt_m, profile["alt_m"], profile["n_m3"])
    temp_k = np.interp(alt_m, profile["alt_m"], profile["temp_k"])
    p_hpa = n_m3 * K_BOLTZ * temp_k / 100.0

    fy337 = fy337_ph_per_mev(p_hpa, temp_k)
    q777 = quench_factor_777(n_m3)

    # Ambient peak field: explicit argument > capture-recorded value.
    e0_csv_kv = (reduced["e0_peak_v_per_m"] / 1e3
                 if reduced["e0_peak_v_per_m"] is not None else None)
    if e0_peak_kv_per_m is None:
        if e0_csv_kv is not None:
            e0_kv = e0_csv_kv
            e0_origin = "from capture profile_e0_peak_v_per_m"
        else:
            raise ValueError(
                "capture does not record profile_e0_peak_v_per_m; pass "
                "e0_peak_kv_per_m explicitly instead of inventing a case field"
            )
    else:
        e0_kv = e0_peak_kv_per_m
        e0_origin = "explicit argument"
        if e0_csv_kv is not None and abs(e0_kv - e0_csv_kv) > 0.01 * e0_csv_kv:
            print(f"WARN: explicit E0 = {e0_kv:g} kV/m disagrees with the "
                  f"capture's recorded {e0_csv_kv:g} kV/m -- E/N and the Xu "
                  f"enhancement follow the explicit value")
    if verbose:
        print(f"ambient peak field E0 = {e0_kv:g} kV/m ({e0_origin})")

    if reduced["field_fraction"] is not None:
        en_td = e0_kv * 1e3 * reduced["field_fraction"] / n_m3 / 1e-21
        f_enh = enhancement_337(en_td)
    else:
        # No screening diagnostic: do NOT fabricate an unscreened field.
        print("WARN: no ne_low-weighted field-fraction column in "
              "rrea_reduced.csv -- field screening unknown, Xu enhancement "
              "NOT applied (r337_hi = r337_lo)")
        en_td = np.zeros_like(t)
        f_enh = np.ones_like(t)

    f_enh_source = "scalar " + reduced["field_fraction_column"]

    r337_lo = dep_mev_s * fy337  # degradation term only
    r337_hi = r337_lo * f_enh  # x Xu field-enhancement estimate (not a bound)
    r777_mid = pairs_rate * EXC_PER_PAIR_777 * q777

    return {
        "t_s": t,
        "dt_s": dt,
        "pairs_rate": pairs_rate,
        "dep_mev_s": dep_mev_s,
        "n_ge1MeV": reduced["n_ge1MeV"],
        "alt_m": alt_m,
        "n_m3": n_m3,
        "p_hpa": p_hpa,
        "fy337": fy337,
        "q777": q777,
        "en_td": en_td,
        "f_enh": f_enh,
        "r337_lo": r337_lo,
        "r337_hi": r337_hi,
        "r777_lo": r777_mid / UNCERTAINTY_FACTOR_777,
        "r777_mid": r777_mid,
        "r777_hi": r777_mid * UNCERTAINTY_FACTOR_777,
        "ev_per_pair": ev_per_pair,
        "field_fraction_column": reduced["field_fraction_column"],
        "f_enh_source": f_enh_source,
        "e0_peak_kv_per_m_used": e0_kv,
        "e0_origin": e0_origin,
    }


def scattered_irradiance(
    series: dict,
    *,
    observer_alt_m: float = OBSERVER_ALTITUDE_M_MSL,
    n_photons: int = 200_000,
    seed: int = 20200,
    cloud: dict | None = None,
    cloud_top_m: float | None = None,
    verbose: bool = True,
) -> dict | None:
    """Observed irradiance [uW/m^2] at the observer through cloud transport.

    Hands the source rates to scripts/rrea_cloud_scattering.py (cloudscat) and
    returns {t_s, irr337_lo, irr337_hi, irr777, label} on CloudScat's absolute
    observer-arrival timeline, or None when CloudScat is unavailable.

    This REPLACES the geometric curve rather than correcting it: cloudscat's
    next-event estimator already carries the direct term through exp(-tau).

    The 337 upper scenario is the transported baseline multiplied by the
    source-weighted mean Xu factor, avoiding an unnecessary third Monte Carlo
    run for a deliberately heuristic correction.
    """
    try:
        from rrea_cloud_scattering import ALOFT_CLOUD, scattered_flux
    except ImportError as exc:  # pragma: no cover - environment-dependent
        print(f"cloud scattering unavailable ({exc}); geometric panel only")
        return None
    if cloud_top_m is not None:
        cloud = {**(cloud or {}), "top_m": float(cloud_top_m)}
    t, alt_m = series["t_s"], series["alt_m"]
    out = {}
    arrival_t = None
    try:
        for key, rate, band, photon_j in (
            ("irr337_lo", series["r337_lo"], "337", E_PHOTON_337_J),
            ("irr777", series["r777_mid"], "777", E_PHOTON_777_J),
        ):
            t_obs, flux, _ = scattered_flux(
                t, rate, alt_m, band=band, observer_alt_m=observer_alt_m,
                n_photons=n_photons, seed=seed, cloud=cloud)
            if arrival_t is None:
                arrival_t = t_obs
            elif not np.array_equal(arrival_t, t_obs):
                raise SystemExit("CloudScat band timelines do not match")
            out[key] = flux * photon_j * 1e6  # ph m^-2 s^-1 -> uW/m^2
    except SystemExit as exc:
        # Default-on must degrade to the geometric panel, not kill the figure.
        print(f"cloud scattering skipped: {exc}")
        return None
    weight = np.clip(series["r337_lo"], 0.0, None)
    mean_enh = (float(np.average(series["f_enh"], weights=weight))
                if float(np.sum(weight)) > 0.0 else 1.0)
    out["irr337_hi"] = out["irr337_lo"] * mean_enh
    out["t_s"] = np.asarray(arrival_t)
    effective_cloud = {**ALOFT_CLOUD, **(cloud or {})}
    out["label"] = (
        f"CloudScat liquid-water cloud "
        f"{effective_cloud['bottom_m']/1e3:g}-"
        f"{effective_cloud['top_m']/1e3:g} km MSL, "
        f"{effective_cloud['droplet_n_cm3']:g} cm^-3, "
        f"{effective_cloud['droplet_radius_m']*1e6:g} um droplets; "
        f"observer {observer_alt_m/1e3:g} km MSL; {n_photons:,} histories"
    )
    if verbose:
        i = int(np.argmax(out["irr337_lo"]))
        print(f"cloud-scattered 337 nm peak: {out['irr337_lo'][i]:.3e} uW/m^2 "
              f"at absolute arrival time {out['t_s'][i]*1e6:.1f} us")
    return out


def observer_geometry(
    alt_m: np.ndarray,
    weights: np.ndarray,
    observer_alt_m: float,
    offset_m: float = 0.0,
) -> tuple[float, float, float, str]:
    """Emission-weighted source altitude and the geometry seen by an observer.

    Returns (source altitude MSL, slant range, polar angle from +z, label).
    The angle is atan2(offset, observer - source), so it is correct on both
    sides of the source: an observer overhead gets theta -> 0 (cos > 0), one
    on the ground below gets theta > 90 deg (cos < 0), and the vertical field
    component is e_r cos(theta) - e_theta sin(theta) in either case.

    Used by the optical irradiance calculation so source altitude and observer
    range are derived in one place.
    """
    src_alt_m = emission_weighted_altitude(alt_m, weights)
    vertical_m = observer_alt_m - src_alt_m
    r_m = math.hypot(offset_m, vertical_m)
    if r_m <= 0.0:
        raise SystemExit(
            "observer coincides with the emission-weighted source altitude "
            f"({src_alt_m:.0f} m MSL); give a distance or offset"
        )
    if offset_m == 0.0 and vertical_m < 0.0:
        print(
            f"WARN: emission-weighted source ({src_alt_m/1e3:.1f} km MSL) is "
            f"ABOVE the observer ({observer_alt_m/1e3:.1f} km MSL); the "
            "observer is inside or below the emitting column and the point "
            "dipole is meaningless there"
        )
    label = (
        f"{observer_alt_m/1e3:.0f} km MSL, overhead, d = {r_m/1e3:.1f} km"
        if offset_m == 0.0
        else (
            f"{observer_alt_m/1e3:.0f} km MSL, {offset_m/1e3:.0f} km "
            f"horizontal, R = {r_m/1e3:.1f} km, "
            f"theta = {math.degrees(math.atan2(offset_m, vertical_m)):.0f} deg"
        )
    )
    return src_alt_m, r_m, math.atan2(offset_m, vertical_m), label


def main() -> int:
    parser = figure_argument_parser(__doc__.splitlines()[0], smooth_us_default=1.0)
    parser.add_argument("--title", default=None)
    parser.add_argument("--profile-csv", type=Path, default=None,
                        help="NRLMSISE profile CSV (default: repo EFIELD_profile/"
                             "rrea_threshold_NRLMSISE00_gulf_284kVpm.csv)")
    parser.add_argument("--z0-altitude-m", type=float, default=None,
                        help="MSL altitude of z=0 (default: capture metadata JSON)")
    parser.add_argument("--e0-peak-kv-per-m", type=float, default=None,
                        help="ambient peak field (default: the capture's "
                             "recorded profile_e0_peak_v_per_m; explicit "
                             "values override it with a WARN on mismatch)")
    parser.add_argument("--observer-altitude-km-msl", type=float,
                        default=OBSERVER_ALTITUDE_M_MSL / 1e3,
                        help="observer MSL altitude for the irradiance panel "
                             "(default %(default)g km: the ER-2 aircraft "
                             "overhead; distance = observer altitude - mean "
                             "emission altitude)")
    parser.add_argument("--distance-km", type=float, default=None,
                        help="explicit observer distance; overrides "
                             "--observer-altitude-km-msl when given")
    add_cloud_arguments(parser)
    args = parser.parse_args()

    series = optical_series(
        args.capture_dir,
        profile_csv=args.profile_csv,
        z0_altitude_m=args.z0_altitude_m,
        e0_peak_kv_per_m=args.e0_peak_kv_per_m,
        smooth_us=args.smooth_us,
    )
    t, dt = series["t_s"], series["dt_s"]
    pairs_rate, dep_mev_s = series["pairs_rate"], series["dep_mev_s"]
    alt_m, n_m3, p_hpa = series["alt_m"], series["n_m3"], series["p_hpa"]
    fy337, q777 = series["fy337"], series["q777"]
    en_td, f_enh = series["en_td"], series["f_enh"]
    r337_lo, r337_hi = series["r337_lo"], series["r337_hi"]
    r777_lo, r777_mid, r777_hi = (
        series["r777_lo"], series["r777_mid"], series["r777_hi"])

    if args.distance_km is not None:
        direct_distance_m = np.full_like(t, args.distance_km * 1e3)
        dist_label = f"{args.distance_km:.0f} km"
    else:
        # ER-2 geometry: observer directly overhead at the given MSL altitude.
        _, _, _, label = observer_geometry(
            alt_m, r337_lo, args.observer_altitude_km_msl * 1e3
        )
        direct_distance_m = np.abs(
            args.observer_altitude_km_msl * 1e3 - alt_m
        )
        dist_label = f"ER-2 {label}"
    if np.any(direct_distance_m <= 0.0):
        raise SystemExit("free-space observer coincides with an optical source bin")
    to_irradiance_337 = E_PHOTON_337_J / (4.0 * math.pi * direct_distance_m**2) * 1e6
    to_irradiance_777 = E_PHOTON_777_J / (4.0 * math.pi * direct_distance_m**2) * 1e6
    direct_arrival_us = (t + direct_distance_m / C) * 1e6

    i_peak = int(np.argmax(r337_lo))
    total_337_lo = float(np.trapezoid(r337_lo, t))
    total_337_hi = float(np.trapezoid(r337_hi, t))
    total_777 = float(np.trapezoid(r777_mid, t))
    print(f"emission altitude at peak: {alt_m[i_peak]/1e3:.2f} km MSL, "
          f"P = {p_hpa[i_peak]:.0f} hPa, FY337 = {fy337[i_peak]:.1f} ph/MeV, "
          f"q777 = {q777[i_peak]:.2e}")
    print(f"E/N (ne_low-weighted, {series['field_fraction_column']}): "
          f"{en_td[i_peak]:.1f} Td at peak -> 337 enhancement x{f_enh[i_peak]:.2f}")
    print(f"peak 337.1 nm source rate: {r337_lo[i_peak]:.3e} - {r337_hi[i_peak]:.3e} ph/s "
          f"({r337_lo[i_peak]*E_PHOTON_337_J:.3e} - {r337_hi[i_peak]*E_PHOTON_337_J:.3e} W)")
    print(f"peak 777.4 nm source rate: {r777_mid[i_peak]:.3e} ph/s "
          f"(x/{UNCERTAINTY_FACTOR_777:.0f} band)")
    print(f"totals so far: 337 nm {total_337_lo:.3e}-{total_337_hi:.3e} ph, "
          f"777 nm {total_777:.3e} ph; 337/777 ratio ~ "
          f"{total_337_lo/max(total_777,1.0):.0f}")

    scattered = (
        scattered_irradiance(
            series,
            observer_alt_m=args.observer_altitude_km_msl * 1e3,
            n_photons=args.cloud_photons,
            cloud_top_m=args.cloud_top_km * 1e3)
        if args.cloud_scattering else None)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t_us = t * 1e6
    fig, axes = plt.subplots(3, 1, figsize=(11, 11), sharex=False,
                             constrained_layout=True)
    title = args.title or f"RREA optical emissions: {args.capture_dir}"
    fig.suptitle(title, fontsize=13)

    ax = axes[0]
    ax.plot(t_us, series["n_ge1MeV"], color="tab:orange", lw=1.4,
            label="N(electrons > 1 MeV)")
    ax.plot(t_us, pairs_rate * dt, color="0.5", lw=1.0,
            label=f"ion pairs per {dt*1e9:.0f} ns")
    ax.set_yscale("log")
    ax.set_ylabel("count")
    ax.axvline(t_us[i_peak], color="0.7", ls=":", lw=1)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_title("Lightcurve and ionization driver", fontsize=10)

    ax = axes[1]
    ax.fill_between(t_us, np.clip(r337_lo, 1e10, None), np.clip(r337_hi, 1e10, None),
                    color="tab:blue", alpha=0.35,
                    label="337.1 nm N$_2$ 2P(0,0) [degradation .. x Xu field-enhancement estimate]")
    ax.plot(t_us, r337_lo, color="tab:blue", lw=1.2)
    ax.fill_between(t_us, np.clip(r777_lo, 1e10, None), np.clip(r777_hi, 1e10, None),
                    color="tab:red", alpha=0.30,
                    label=f"777.4 nm OI [x/{UNCERTAINTY_FACTOR_777:.0f} nominal; order-of-magnitude model]")
    ax.plot(t_us, r777_mid, color="tab:red", lw=1.2)
    ax.set_yscale("log")
    ax.set_ylabel("source photon rate (ph/s)")
    ax.set_xlabel("source time (us)")
    ax.axvline(t_us[i_peak], color="0.7", ls=":", lw=1)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_title(
        f"Source emission at {alt_m[i_peak]/1e3:.1f} km MSL "
        f"(FY337 = {fy337[i_peak]:.0f} ph/MeV, 777 quench = {q777[i_peak]:.1e})",
        fontsize=10)

    ax = axes[2]
    ax.fill_between(direct_arrival_us,
                    np.clip(r337_lo * to_irradiance_337, 1e-12, None),
                    np.clip(r337_hi * to_irradiance_337, 1e-12, None),
                    color="tab:blue", alpha=0.35,
                    label="337.1 nm, free space")
    ax.fill_between(direct_arrival_us,
                    np.clip(r777_lo * to_irradiance_777, 1e-12, None),
                    np.clip(r777_hi * to_irradiance_777, 1e-12, None),
                    color="tab:red", alpha=0.30, label="777.4 nm, free space")
    if scattered is not None:
        arrival_us = scattered["t_s"] * 1e6
        ax.fill_between(arrival_us,
                        np.clip(scattered["irr337_lo"], 1e-12, None),
                        np.clip(scattered["irr337_hi"], 1e-12, None),
                        color="#0d366b", alpha=0.55, label="337.1 nm, scattered")
        ax.plot(arrival_us, scattered["irr337_lo"], color="#0d366b", lw=1.2)
        ax.plot(arrival_us, scattered["irr777"], color="#7c1a19", lw=1.2,
                label="777.4 nm, scattered")
    ax.set_ylim(bottom=0.0)
    ax.set_ylabel(f"irradiance ({dist_label}) (uW/m$^2$)")
    ax.set_xlabel("absolute observer arrival time (us)")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_title(
        "Isotropic-source irradiance (no atmosphere/scattering)"
        if scattered is None else
        f"Absolute-arrival irradiance: free space vs "
        f"cloudscat transport ({scattered['label']})",
        fontsize=10)

    caveat = (
        "337 baseline anchored to Rosado dry-air fluorescence yield; "
        "Xu-2015 enhancement is a heuristic used only for E/N >= 16 Td; "
        "777.4 nm is an independent order-of-magnitude model (O2 dissociative excitation, "
        "Erdman & Zipf 1987; x/3 band nominal, O(3p3P)-proxy quenching; not Xu-validated). "
        "Native pair-source-weighted emission altitude; "
        "no hot-leader channel (real events add leader 777.4 nm). "
        + ("No radiative transfer: free-space 1/(4 pi d^2) only."
           if scattered is None else
           f"Transported curves: {scattered['label']}; direct term included, "
           "so they replace rather than correct the free-space reference.")
    )
    fig.text(0.02, -0.015, caveat, fontsize=7.5, style="italic", wrap=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight")
    print(f"wrote {args.output}")

    companion = args.output.with_suffix(".optical.csv")
    with companion.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        header = [
            "time_s", "pairs_per_s", "dep_MeV_per_s", "alt_m", "n_m3", "p_hPa",
            "fy337_ph_per_MeV", "quench777", "en_td", "f_enh_337",
            "r337_lo_ph_s", "r337_hi_ph_s", "r777_mid_ph_s",
            "free_space_arrival_time_s"]
        writer.writerow(header)
        for i in range(len(t)):
            row = [t[i], pairs_rate[i], dep_mev_s[i], alt_m[i], n_m3[i],
                   p_hpa[i], fy337[i], q777[i], en_td[i], f_enh[i],
                   r337_lo[i], r337_hi[i], r777_mid[i],
                   direct_arrival_us[i] * 1e-6]
            writer.writerow([f"{v:.8e}" for v in row])
    print(f"wrote {companion}")
    if scattered is not None:
        arrival_companion = args.output.with_suffix(".optical_arrival.csv")
        with arrival_companion.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow([
                "absolute_arrival_time_s",
                "scat337_lo_uW_m2",
                "scat337_hi_uW_m2",
                "scat777_uW_m2",
            ])
            for values in zip(
                scattered["t_s"],
                scattered["irr337_lo"],
                scattered["irr337_hi"],
                scattered["irr777"],
                strict=True,
            ):
                writer.writerow([f"{value:.8e}" for value in values])
        print(f"wrote {arrival_companion}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
