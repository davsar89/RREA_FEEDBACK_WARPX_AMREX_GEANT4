"""Optical source and CloudScat observer curves for one RREA capture.

The 337.1-nm deposition term is the Rosado dry-air fluorescence yield, scaled
by local pressure and temperature. A separate Phelps/BOLSIG swarm envelope
assesses the omitted field-driven contribution; it is not added as a prediction.
The 777.4-nm estimate remains an order-of-magnitude cold-air scenario.

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
from textwrap import fill

import numpy as np

from rrea_cloud_scattering import (
    OBSERVER_ALTITUDE_M_MSL,
    OBSERVER_FOV_HALF_DIAGONAL_DEG,
    emission_weighted_altitude,
)
from compute_rrea_radio_waveform import (
    add_cloud_arguments,
    as_float,
    boxcar,
    figure_argument_parser,
)
from rrea_video_stream import dedup_reduced_rows_by_step
from rrea_profiled_atmosphere import ALTITUDE_AT_Z0_M, read_csv_rows
from rrea_run_support import C, K_BOLTZ

# ---------------------------------------------------------------------------
# Constants
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
# Gilmore, Laher & Espy (1992), doi:10.1063/1.555910: C(v'=0) -> B(v''=0).
C0_BRANCH_337 = 1.31e7 * 3.71e-8  # A00 times the radiative lifetime
SWARM_TABLE = (Path(__file__).resolve().parent.parent / "rrea_fluid_closure"
               / "dry_air_swarm_hybrid_v3/n2_c_excitation.csv")

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
        "soft_brems_eV_cum": col("photon_continuous_soft_brems_energy_eV"),
    }
    # Missing assessment data remains missing; it does not imply zero light.
    for key in ("N_low_e_real", "closure_realized_max_en_td"):
        out[key] = np.array([as_float(row, key, float("nan")) for row in ordered])
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


def quench_factor_777(n_m3: np.ndarray) -> np.ndarray:
    """OI survival A/(A + N sum(x_s k_s)); Niemi et al. (2005) proxy rates."""
    quench = KQ_N2_777_M3_S * X_N2 * n_m3 + KQ_O2_777_M3_S * X_O2 * n_m3
    return A_777_PER_S / (A_777_PER_S + quench)


def photon_probability_337(n_m3, temperature_K):
    """Photons per C(v'=0) excitation, with the same AIRFLY quenching as FY."""
    pressure = np.asarray(n_m3) * K_BOLTZ * np.asarray(temperature_K) / 100
    return C0_BRANCH_337 * fy337_ph_per_mev(pressure, temperature_K) / fy337_ph_per_mev(
        np.zeros_like(pressure), temperature_K)


def source_yield_bounds(altitude, density, temperature, mean_altitude):
    """Bound E[FY(z)] from the native source centroid, without assuming its width.

    Jensen's inequality applied to a convex minorant / concave majorant.
    FY decreases with N*T**0.14. Independent endpoint extrema bound each
    linearly interpolated atmosphere segment; the hulls then bound any
    positive source distribution with the recorded first moment.
    """
    z, n, temp = (np.asarray(a) for a in (altitude, density, temperature))
    bounds = []
    for sign, extreme in ((1, np.maximum), (-1, np.minimum)):
        nn, tt = extreme(n[:-1], n[1:]), extreme(temp[:-1], temp[1:])
        segment = sign * fy337_ph_per_mev(nn*K_BOLTZ*tt/100, tt)
        y = np.minimum(np.r_[segment[0], segment], np.r_[segment, segment[-1]])
        hull = []
        for i in range(len(z)):
            while len(hull) > 1:
                a, b = hull[-2:]
                if (y[b]-y[a])*(z[i]-z[b]) < (y[i]-y[b])*(z[b]-z[a]):
                    break
                hull.pop()
            hull.append(i)
        bound = np.interp(mean_altitude, z[hull], y[hull])
        # Tiny cumulative-source differences can lose their centroid to
        # roundoff; in that case only the whole-domain bound is supported.
        bound = np.where((mean_altitude >= z[0]) & (mean_altitude <= z[-1]),
                         bound, y.min())
        bounds.append(sign * bound)
    return np.array(bounds)


def swarm_emission_envelope(time_s, electrons, en_max, pair_count, atmosphere,
                            table_path=SWARM_TABLE, deposition_yield_lower=None):
    """P_sw <= N_e max(N_N2*k_C*b337); deposition >= S_pair min(W*FY).

    Maximize over the tabulated gas/model cases and every E/N knot up to the
    next enclosing node. This bounds a nonmonotone interpolant, without treating
    the source-weighted field as the electron-weighted field or cancelling
    differently weighted quenching. Fraction C0=1 is an upper spectral choice.
    The numerical tail and molecular-source assumptions remain model limits.
    """
    t, ne, en = (np.asarray(a, dtype=float) for a in (time_s, electrons, en_max))
    missing = dict(upper_ph_s=np.full(t.shape, np.nan), fraction=np.nan,
                   interval_fraction=np.full(t.shape, np.nan),
                   interval_photons=np.full(t.shape, np.nan), note="swarm unassessed")
    if not Path(table_path).is_file() or not np.all(np.isfinite(ne + en)):
        return missing
    with Path(table_path).open() as stream:
        rows = list(csv.DictReader(line for line in stream if not line.startswith("#")))
    def column(key):
        return np.array([float(r[key]) for r in rows])
    grid, temp, density = (column(k) for k in ("en_td", "temperature_K", "density_m3"))
    n, temperature = atmosphere
    if (np.any(ne < 0) or np.any(en < 0) or np.any(en > grid.max())
            or np.min(n) < density.min() or np.max(n) > density.max()
            or np.min(temperature) < temp.min() or np.max(temperature) > temp.max()):
        missing["note"] = "swarm unassessed: outside coefficient domain"
        return missing
    rate = (column("k_c_m3_s") + column("tail_envelope_m3_s"))
    x_n2 = (1 - column("argon_fraction")) * X_N2 / (X_N2 + X_O2)
    rate *= x_n2 * density * photon_probability_337(density, temp)
    allowed = np.ones(grid.shape, dtype=bool)
    for values, physical in ((density, n), (temp, temperature)):
        nodes = np.unique(values)
        lo = nodes[np.searchsorted(nodes, np.min(physical), side="right") - 1]
        hi = nodes[np.searchsorted(nodes, np.max(physical), side="left")]
        allowed &= (values >= lo) & (values <= hi)
    knots = np.unique(grid)
    upper = np.maximum.accumulate([rate[(grid == e) & allowed].max() for e in knots])
    per_electron = upper[np.searchsorted(knots, en, side="left")]
    photon_rate = ne * per_electron
    # Use exact native pair-count differences, not a smoothed source integral.
    if deposition_yield_lower is None:
        deposition_yield_lower = np.min(fy337_ph_per_mev(
            np.asarray(n) * K_BOLTZ * np.asarray(temperature) / 100, temperature))
    eta_min = W_AIR_EV * 1e-6 * deposition_yield_lower
    dep = np.diff(pair_count) * eta_min
    emitted = .5 * (photon_rate[:-1] + photon_rate[1:]) * np.diff(t)
    ratios = np.full(t.shape, np.nan)
    np.divide(emitted, dep, out=ratios[1:], where=dep > 0)
    fraction = float(emitted.sum()/dep.sum()) if dep.sum() > 0 else np.nan
    note = (f"sampled swarm model envelope: {fraction:.3%} of deposition fluence"
            if np.isfinite(fraction) else "swarm envelope reported as absolute photons")
    return dict(upper_ph_s=photon_rate, fraction=fraction,
                interval_fraction=ratios, interval_photons=np.r_[np.nan, emitted], note=note)


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
    rate = np.clip(boxcar(rate, float(np.median(dt_interval)), smooth_s), 0.0, None)
    # Smoothing and irregular diagnostic spacing must not create or lose pairs.
    rate *= delta_weight.sum() / np.trapezoid(rate, t)
    return rate, mean_z


def optical_series(
    capture_dir: Path,
    *,
    profile_csv: Path | None = None,
    z0_altitude_m: float | None = None,
    smooth_us: float = 1.0,
    verbose: bool = True,
) -> dict:
    """Compute the 337.1 / 777.4 nm source-rate series for one capture.

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
    soft_brems_share = (
        reduced["soft_brems_eV_cum"][-1]
        - reduced["soft_brems_eV_cum"][0]
    ) / max(pair_count * W_AIR_EV, 1.0)
    if verbose:
        print(f"soft-brems drag / native deposition energy: "
              f"{soft_brems_share:.2e} (untransported radiative energy; "
              f"its fluorescence is unassessed)")

    # Source altitude is the native positive-ion creation-rate centroid.
    alt_m = z0_m + source_mean_z_m
    n_m3 = np.interp(alt_m, profile["alt_m"], profile["n_m3"])
    temp_k = np.interp(alt_m, profile["alt_m"], profile["temp_k"])
    p_hpa = n_m3 * K_BOLTZ * temp_k / 100.0

    fy337 = fy337_ph_per_mev(p_hpa, temp_k)
    q777 = quench_factor_777(n_m3)

    r337 = dep_mev_s * fy337  # degradation term only
    r777_mid = pairs_rate * EXC_PER_PAIR_777 * q777

    z = profile["alt_m"]
    command_file = capture_dir / "command.json"
    if command_file.exists():
        command = json.loads(command_file.read_text())["command"]
        # Recorded runtime geometry, including any final command-line override.
        limits = {arg.split("=", 1)[0]: [float(x) for x in arg.split("=", 1)[1].split()]
                  for arg in command if arg.startswith(("geometry.prob_lo=", "geometry.prob_hi="))}
        if "geometry.prob_hi" in limits:
            lo = z0_m + limits.get("geometry.prob_lo", [0., 0.])[-1]
            hi = z0_m + limits["geometry.prob_hi"][-1]
            z = np.r_[lo, profile["alt_m"][(profile["alt_m"] > lo) & (profile["alt_m"] < hi)], hi]
    atmosphere = tuple(np.interp(z, profile["alt_m"], profile[k]) for k in ("n_m3", "temp_k"))
    delta_pairs = np.diff(reduced["ion_pair_source_weight_cumulative"])
    source_z = np.full(delta_pairs.shape, np.nan)
    np.divide(np.diff(reduced["ion_pair_source_z_weighted_m_cumulative"]),
              delta_pairs, out=source_z, where=delta_pairs > 0)
    yield_bounds = source_yield_bounds(z, *atmosphere, source_z + z0_m)
    swarm = swarm_emission_envelope(
        t, reduced["N_low_e_real"], reduced["closure_realized_max_en_td"],
        reduced["ion_pair_source_weight_cumulative"], atmosphere,
        deposition_yield_lower=yield_bounds[0])
    if verbose:
        print(swarm["note"])

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
        "r337": r337,
        "swarm": swarm,
        "deposition_fluence_bounds": W_AIR_EV*1e-6*(yield_bounds @ delta_pairs),
        "r777_lo": r777_mid / UNCERTAINTY_FACTOR_777,
        "r777_mid": r777_mid,
        "r777_hi": r777_mid * UNCERTAINTY_FACTOR_777,
        "ev_per_pair": ev_per_pair,
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
    returns {t_s, irr337, irr777, label} on CloudScat's absolute
    observer-arrival timeline, or None when CloudScat is unavailable.

    This REPLACES the geometric curve rather than correcting it: cloudscat's
    next-event estimator already carries the direct term through exp(-tau).

    """
    from rrea_cloud_scattering import ALOFT_CLOUD, scattered_flux
    if cloud_top_m is not None:
        cloud = {**(cloud or {}), "top_m": float(cloud_top_m)}
    t, alt_m = series["t_s"], series["alt_m"]
    out = {}
    arrival_t = None
    try:
        for key, rate, band, photon_j in (
            ("irr337", series["r337"], "337", E_PHOTON_337_J),
            ("irr777", series["r777_mid"], "777", E_PHOTON_777_J),
        ):
            t_obs, flux = scattered_flux(
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
    out["t_s"] = np.asarray(arrival_t)
    effective_cloud = {**ALOFT_CLOUD, **(cloud or {})}
    out["label"] = (
        f"CloudScat liquid-water cloud "
        f"{effective_cloud['bottom_m']/1e3:g}-"
        f"{effective_cloud['top_m']/1e3:g} km MSL, "
        f"{effective_cloud['droplet_n_cm3']:g} cm^-3, "
        f"{effective_cloud['droplet_radius_m']*1e6:g} um droplets; "
        f"observer {observer_alt_m/1e3:g} km MSL, photometer "
        f"{2 * OBSERVER_FOV_HALF_DIAGONAL_DEG:g} deg full diagonal; "
        f"{n_photons:,} histories"
    )
    if verbose:
        i = int(np.argmax(out["irr337"]))
        print(f"cloud-scattered 337 nm peak: {out['irr337'][i]:.3e} uW/m^2 "
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
        smooth_us=args.smooth_us,
    )
    t, dt = series["t_s"], series["dt_s"]
    pairs_rate, dep_mev_s = series["pairs_rate"], series["dep_mev_s"]
    alt_m, n_m3, p_hpa = series["alt_m"], series["n_m3"], series["p_hpa"]
    fy337, q777 = series["fy337"], series["q777"]
    r337 = series["r337"]
    r777_lo, r777_mid, r777_hi = (
        series["r777_lo"], series["r777_mid"], series["r777_hi"])

    if args.distance_km is not None:
        direct_distance_m = np.full_like(t, args.distance_km * 1e3)
        dist_label = f"{args.distance_km:.0f} km"
    else:
        # ER-2 geometry: observer directly overhead at the given MSL altitude.
        _, _, _, label = observer_geometry(
            alt_m, r337, args.observer_altitude_km_msl * 1e3
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

    i_peak = int(np.argmax(r337))
    total_337 = float(np.trapezoid(r337, t))
    total_777 = float(np.trapezoid(r777_mid, t))
    print(f"emission altitude at peak: {alt_m[i_peak]/1e3:.2f} km MSL, "
          f"P = {p_hpa[i_peak]:.0f} hPa, FY337 = {fy337[i_peak]:.1f} ph/MeV, "
          f"q777 = {q777[i_peak]:.2e}")
    print(f"peak 337.1 nm deposition rate: {r337[i_peak]:.3e} ph/s "
          f"({r337[i_peak]*E_PHOTON_337_J:.3e} W)")
    print(f"peak 777.4 nm source rate: {r777_mid[i_peak]:.3e} ph/s "
          f"(x/{UNCERTAINTY_FACTOR_777:.0f} band)")
    print(f"totals so far: 337 nm {total_337:.3e} ph, "
          f"777 nm {total_777:.3e} ph; 337/777 ratio ~ "
          f"{total_337/max(total_777,1.0):.0f}")
    lo, hi = series["deposition_fluence_bounds"]
    print(f"337 nm spatial-quenching bound from native source centroid: "
          f"{lo:.3e} .. {hi:.3e} ph (source width unresolved)")

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
    title = args.title or f"RREA optical emissions: {args.capture_dir.name}"
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
    ax.plot(t_us, r337, color="tab:blue", lw=1.2,
            label="337.1 nm N$_2$ 2P(0,0), deposition")
    if np.any(np.isfinite(series["swarm"]["upper_ph_s"])):
        ax.plot(t_us, series["swarm"]["upper_ph_s"], "--", color="0.4", lw=.8,
                label="omitted swarm: model envelope")
    ax.fill_between(t_us, np.clip(r777_lo, 1e10, None), np.clip(r777_hi, 1e10, None),
                    color="tab:red", alpha=0.30,
                    label=f"777.4 nm OI [x/{UNCERTAINTY_FACTOR_777:.0f} nominal; order-of-magnitude model]")
    ax.plot(t_us, r777_mid, color="tab:red", lw=1.2)
    ax.set_yscale("log")
    ax.set_ylim(max(r337.max()/1e8, 1.),
                2*np.nanmax([r337, r777_hi, series["swarm"]["upper_ph_s"]]))
    ax.set_ylabel("source photon rate (ph/s)")
    ax.set_xlabel("source time (us)")
    ax.axvline(t_us[i_peak], color="0.7", ls=":", lw=1)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_title(
        f"Source emission at {alt_m[i_peak]/1e3:.1f} km MSL "
        f"(FY337 = {fy337[i_peak]:.0f} ph/MeV, 777 quench = {q777[i_peak]:.1e})",
        fontsize=10)

    ax = axes[2]
    ax.plot(direct_arrival_us, r337 * to_irradiance_337,
            color="tab:blue", label="337.1 nm, free space")
    ax.fill_between(direct_arrival_us,
                    np.clip(r777_lo * to_irradiance_777, 1e-12, None),
                    np.clip(r777_hi * to_irradiance_777, 1e-12, None),
                    color="tab:red", alpha=0.30, label="777.4 nm, free space")
    if scattered is not None:
        arrival_us = scattered["t_s"] * 1e6
        ax.plot(arrival_us, scattered["irr337"], color="#0d366b", lw=1.2,
                label="337.1 nm, scattered")
        ax.plot(arrival_us, scattered["irr777"], color="#7c1a19", lw=1.2,
                label="777.4 nm, scattered")
    ax.set_ylim(bottom=0.0)
    ax.set_ylabel("irradiance (uW/m$^2$)")
    ax.set_xlabel("absolute observer arrival time (us)")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_title(fill(
        f"Isotropic-source irradiance at {dist_label} (no atmosphere/scattering)"
        if scattered is None else
        f"Absolute-arrival irradiance: free space vs "
        f"cloudscat transport ({scattered['label']})", 110),
        fontsize=10)

    caveat = (
        "337 baseline anchored to Rosado dry-air fluorescence yield; "
        f"{series['swarm']['note']}; "
        "777.4 nm is an independent order-of-magnitude model (O2 dissociative excitation, "
        "Erdman & Zipf 1987; x/3 band nominal, O(3p3P)-proxy quenching). "
        "Native pair-source-weighted emission altitude; "
        "no hot-leader channel (real events add leader 777.4 nm). "
        + ("No radiative transfer: free-space 1/(4 pi d^2) only."
           if scattered is None else
           f"Transported curves: {scattered['label']}; direct term included, "
           "so they replace rather than correct the free-space reference.")
    )
    fig.supxlabel(fill(caveat, 140), fontsize=7.5, style="italic", ha="left", x=.02)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight")
    print(f"wrote {args.output}")

    companion = args.output.with_suffix(".optical.csv")
    with companion.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        header = [
            "time_s", "pairs_per_s", "dep_MeV_per_s", "alt_m", "n_m3", "p_hPa",
            "fy337_ph_per_MeV", "quench777",
            "r337_ph_s", "r777_mid_ph_s", "swarm_model_upper_ph_s",
            "swarm_interval_fraction", "swarm_interval_upper_ph",
            "free_space_arrival_time_s"]
        writer.writerow(header)
        for i in range(len(t)):
            row = [t[i], pairs_rate[i], dep_mev_s[i], alt_m[i], n_m3[i],
                   p_hpa[i], fy337[i], q777[i], r337[i], r777_mid[i],
                   series["swarm"]["upper_ph_s"][i], series["swarm"]["interval_fraction"][i],
                   series["swarm"]["interval_photons"][i],
                   direct_arrival_us[i] * 1e-6]
            writer.writerow([f"{v:.8e}" for v in row])
    print(f"wrote {companion}")
    if scattered is not None:
        arrival_companion = args.output.with_suffix(".optical_arrival.csv")
        with arrival_companion.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow([
                "absolute_arrival_time_s",
                "scat337_uW_m2",
                "scat777_uW_m2",
            ])
            for values in zip(
                scattered["t_s"],
                scattered["irr337"],
                scattered["irr777"],
                strict=True,
            ):
                writer.writerow([f"{value:.8e}" for value in values])
        print(f"wrote {arrival_companion}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
