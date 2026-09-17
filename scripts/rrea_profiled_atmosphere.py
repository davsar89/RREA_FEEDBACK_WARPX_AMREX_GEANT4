#!/usr/bin/env python3
"""Shared helpers for profiled-atmosphere RREA/WarpX runs."""

from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path



ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIELD_PROFILE = ROOT / "EFIELD_profile/digitized_efield_profile.csv"
DEFAULT_DENSITY_PROFILE = ROOT / "EFIELD_profile/rrea_threshold_NRLMSISE00_gulf_284kVpm.csv"
# Run geometry/time values come from the single source
# config/rrea_defaults.json (capture_defaults).  This is a minimal direct
# read because rrea_run_support imports THIS module (a loader import here
# would cycle); the strict schema validation still guards every driver
# and sbatch path through rrea_run_support.
_CAPTURE_DEFAULTS = json.loads(
    (ROOT / "config/rrea_defaults.json").read_text(encoding="utf-8")
)["capture_defaults"]
ALTITUDE_AT_Z0_M = float(_CAPTURE_DEFAULTS["altitude_at_z0_m"])
DOMAIN_RADIUS_M = float(_CAPTURE_DEFAULTS["domain_radius_m"])
DOMAIN_HEIGHT_M = float(_CAPTURE_DEFAULTS["domain_height_m"])
SOURCE_RADIUS_M = float(_CAPTURE_DEFAULTS["channel_radius_m"])
SEED_ENERGY_EV = float(_CAPTURE_DEFAULTS["seed_energy_eV"])
ELECTRON_REST_ENERGY_EV = 510998.95000
_SEED_GAMMA = 1.0 + SEED_ENERGY_EV / ELECTRON_REST_ENERGY_EV
SEED_UZ = math.sqrt(_SEED_GAMMA * _SEED_GAMMA - 1.0)
RREA_AVALANCHE_NUMERATOR_KV = 7300.0
RREA_THRESHOLD_STP_KV_PER_M = 276.0

# Three consumers scale by air density, and they need three different reference
# densities. Every anchor below is named at its point of use; none is a bare
# ratio column.
#
#   RREA reference E_th = 276 kV/m * rho/RREA_THRESHOLD_REFERENCE_DENSITY_KG_M3
#   transport rates          rho / (configuration reference_density_kg_m3, 1.20479)
#   reduced ion mobility     rho / Loschmidt STP  (C++ fluid; see
#                            LowEnergyFluidFaceTransport)
#
# The profile CSV's own `*_ratio_to_sea_level` columns describe this particular
# Gulf sounding, not a physical reference, so they are deliberately not read:
# every ratio here is derived from `mass_density_g_cm3` against an explicit
# reference.
#
# The Coleman-Dwyer reference uses the published 276-kV/m denominator at the
# explicit 1.293-kg/m^3 STP density.  It is a nominal local-density/WKB
# comparison, not an exact solution for the nonuniform field.
RREA_THRESHOLD_REFERENCE_DENSITY_KG_M3 = 1.293
COLEMAN_DWYER_MEAN_ENERGY_EV = 7.2e6
SCHEMA6_CHARGED_ENERGY_MIN_EV = 1.0e3
SCHEMA6_CHARGED_ENERGY_MAX_EV = 1.0e10


@dataclass(frozen=True)
class Profile:
    path: Path
    altitude_m: list[float]
    values: list[float]
    value_column: str

    @property
    def max_abs(self) -> float:
        return max(abs(value) for value in self.values)

    def interpolate(self, altitude_m: float) -> float:
        if altitude_m < self.altitude_m[0] or altitude_m > self.altitude_m[-1]:
            raise ValueError(
                f"altitude {altitude_m:.6g} m outside {self.path} range "
                f"[{self.altitude_m[0]:.6g}, {self.altitude_m[-1]:.6g}] m"
            )
        lo = 0
        hi = len(self.altitude_m) - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if self.altitude_m[mid] <= altitude_m:
                lo = mid
            else:
                hi = mid
        span = self.altitude_m[hi] - self.altitude_m[lo]
        if span <= 0:
            return self.values[lo]
        t = (altitude_m - self.altitude_m[lo]) / span
        return (1.0 - t) * self.values[lo] + t * self.values[hi]

    def interpolate_zero_outside(self, altitude_m: float) -> float:
        if altitude_m < self.altitude_m[0] or altitude_m > self.altitude_m[-1]:
            return 0.0
        return self.interpolate(altitude_m)


@dataclass(frozen=True)
class AboveThresholdRegion:
    bottom_altitude_m: float
    top_altitude_m: float
    injection_altitude_m: float
    injection_z_m: float
    width_m: float
    e0_peak_v_per_m: float
    field_at_injection_v_per_m: float
    threshold_at_injection_v_per_m: float
    density_ratio_at_injection: float



@dataclass(frozen=True)
class AvalancheEstimate:
    predicted_avalanche_lengths: float
    predicted_multiplication: float | None
    predicted_multiplication_log: float
    integrated_distance_m: float
    max_field_to_threshold_ratio: float
    min_field_to_threshold_ratio: float
    runaway_threshold_stp_kv_per_m: float


def read_profile(
    path: Path,
    *,
    altitude_column: str,
    value_column: str,
    altitude_scale: float,
    value_scale: float,
    require_positive_values: bool = False,
) -> Profile:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"missing CSV header: {path}")
        for column in [altitude_column, value_column]:
            if column not in reader.fieldnames:
                raise ValueError(f"missing column {column!r} in {path}")
        altitude_m: list[float] = []
        values: list[float] = []
        for row_number, row in enumerate(reader, start=2):
            try:
                altitude = float(row[altitude_column]) * altitude_scale
                value = float(row[value_column]) * value_scale
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"invalid numeric value in {path} row {row_number} "
                    f"for columns {altitude_column!r}/{value_column!r}"
                ) from exc
            if not math.isfinite(altitude) or not math.isfinite(value):
                raise ValueError(
                    f"non-finite value in {path} row {row_number} "
                    f"for columns {altitude_column!r}/{value_column!r}"
                )
            if require_positive_values and value <= 0.0:
                raise ValueError(
                    f"non-positive value in {path} row {row_number} "
                    f"for column {value_column!r}"
                )
            altitude_m.append(altitude)
            values.append(value)
    if len(altitude_m) < 2:
        raise ValueError(f"profile needs at least two rows: {path}")
    for lhs, rhs in zip(altitude_m, altitude_m[1:], strict=False):
        if rhs <= lhs:
            raise ValueError(f"profile altitudes must be sorted and unique: {path}")
    if max(abs(value) for value in values) <= 0.0:
        raise ValueError(f"profile values are all zero: {path}")
    return Profile(
        path=path,
        altitude_m=altitude_m,
        values=values,
        value_column=value_column,
    )


def load_field_profile(path: Path = DEFAULT_FIELD_PROFILE) -> Profile:
    return read_profile(
        path,
        altitude_column="altitude_km_MSL",
        value_column="minus_Ez_kV_per_m",
        altitude_scale=1000.0,
        value_scale=1000.0,
    )


def load_air_density_profile(path: Path = DEFAULT_DENSITY_PROFILE) -> Profile:
    """Absolute air mass density in kg/m^3 -- the only unambiguous column."""
    return read_profile(
        path,
        altitude_column="altitude_km_MSL",
        value_column="mass_density_g_cm3",
        altitude_scale=1000.0,
        value_scale=1000.0,
        require_positive_values=True,
    )


def _rescaled(profile: Profile, factor: float, value_column: str) -> Profile:
    """Derive a profile from another, keeping the source file's identity."""
    return Profile(
        path=profile.path,
        altitude_m=profile.altitude_m,
        values=[value * factor for value in profile.values],
        value_column=value_column,
    )


def load_density_ratio_profile(
    path: Path = DEFAULT_DENSITY_PROFILE,
    *,
    reference_density_kg_m3: float = RREA_THRESHOLD_REFERENCE_DENSITY_KG_M3,
) -> Profile:
    """Air density relative to the density the RREA threshold is quoted at.

    The CSV's sea-level ratios use its sounding surface; this function divides
    the absolute density by the stated reference instead.
    """
    if not reference_density_kg_m3 > 0.0:
        raise ValueError("reference density must be positive")
    return _rescaled(
        load_air_density_profile(path),
        1.0 / reference_density_kg_m3,
        f"mass_density_g_cm3 / {reference_density_kg_m3:g} kg m^-3",
    )


def load_threshold_profile(
    path: Path = DEFAULT_DENSITY_PROFILE,
    *,
    threshold_stp_kv_per_m: float = RREA_THRESHOLD_STP_KV_PER_M,
) -> Profile:
    """Onset field in V/m: ``T * rho(z) / rho_STP``.

    Derived from absolute density rather than the CSV's sounding-relative
    ``RREA_Eth_abs_kV_per_m`` column, which is about 8.8 percent higher.
    """
    return _rescaled(
        load_density_ratio_profile(path),
        threshold_stp_kv_per_m * 1000.0,
        f"{threshold_stp_kv_per_m:g} kV/m * mass_density_g_cm3 / "
        f"{RREA_THRESHOLD_REFERENCE_DENSITY_KG_M3:g} kg m^-3",
    )


def validate_density_scaled_threshold_profile(
    density_ratio: Profile,
    threshold_profile: Profile,
    *,
    threshold_stp_kv_per_m: float = RREA_THRESHOLD_STP_KV_PER_M,
) -> dict[str, float | str]:
    """Validate the derived ``T * rho(z) / rho_STP`` reference profile."""
    if density_ratio.altitude_m != threshold_profile.altitude_m:
        raise ValueError("density and threshold profiles use different altitude knots")
    if not density_ratio.values or len(density_ratio.values) != len(
        threshold_profile.values
    ):
        raise ValueError("density and threshold profiles have incompatible lengths")
    maximum_relative_error = 0.0
    maximum_absolute_error_v_per_m = 0.0
    for altitude_m, density, threshold in zip(
        density_ratio.altitude_m,
        density_ratio.values,
        threshold_profile.values,
        strict=True,
    ):
        expected = threshold_stp_kv_per_m * 1000.0 * density
        absolute_error = abs(threshold - expected)
        relative_error = absolute_error / expected
        maximum_absolute_error_v_per_m = max(
            maximum_absolute_error_v_per_m, absolute_error
        )
        maximum_relative_error = max(maximum_relative_error, relative_error)
        if not math.isclose(threshold, expected, rel_tol=5.0e-13, abs_tol=1.0e-9):
            raise ValueError(
                "loaded threshold profile is not the expected "
                f"{threshold_stp_kv_per_m:g} kV/m * density profile at "
                f"altitude {altitude_m:g} m"
            )
    return {
        "threshold_stp_kv_per_m": threshold_stp_kv_per_m,
        "threshold_reference_density_kg_m3": RREA_THRESHOLD_REFERENCE_DENSITY_KG_M3,
        "maximum_relative_error": maximum_relative_error,
        "maximum_absolute_error_v_per_m": maximum_absolute_error_v_per_m,
    }


def profiled_abs_field_v_per_m(field: Profile, e0_peak_v_per_m: float, altitude_m: float) -> float:
    return abs(e0_peak_v_per_m * field.interpolate_zero_outside(altitude_m) / field.max_abs)


def profiled_minus_ez_v_per_m(field: Profile, e0_peak_v_per_m: float, altitude_m: float) -> float:
    return e0_peak_v_per_m * field.interpolate_zero_outside(altitude_m) / field.max_abs


def taper_weight(r_m: float, r_start_m: float, r_end_m: float) -> float:
    """f(r) of the C2 quintic radial taper, f = 1 - S(t), S(t)=6t^5-15t^4+10t^3
    (ports RreaFieldInitializer::taper_f): 1 inside r_start, 0 outside r_end."""
    if r_m <= r_start_m:
        return 1.0
    if r_m >= r_end_m:
        return 0.0
    t = (r_m - r_start_m) / (r_end_m - r_start_m)
    return 1.0 - t * t * t * (10.0 + t * (6.0 * t - 15.0))


def taper_weight_deriv(r_m: float, r_start_m: float, r_end_m: float) -> float:
    """df/dr of the C2 quintic radial taper f = 1 - S(t), S(t)=6t^5-15t^4+10t^3
    (ports RreaFieldInitializer::taper_f).  S'(t) = 30 t^2 (1-t)^2, so
    f'(r) = -30 t^2 (1-t)^2 / (end-start); |f'|max = 1.875/width at t=0.5."""
    if r_end_m <= r_start_m or r_m <= r_start_m or r_m >= r_end_m:
        return 0.0
    width = r_end_m - r_start_m
    t = (r_m - r_start_m) / width
    return -30.0 * t * t * (1.0 - t) * (1.0 - t) / width


# E/N in townsends is PROPORTIONAL to the local field-to-threshold ratio,
# because threshold and neutral density scale together: E_th(z) =
# RREA_THRESHOLD_STP_KV_PER_M * rho(z)/1.293 and the runtime closure's
# chi(z) = rho(z)/1.293 with N(z) = chi * n_Loschmidt (RreaConstants.H
# n_loschmidt_m3 = 2.6868e25; RreaWarpXCoupling.H
# kReducedMobilityReferenceDensityKgM3 = 1.293 -- same 1.293 as
# RREA_THRESHOLD_REFERENCE_DENSITY_KG_M3 above, so the two ratios are one).
# Hence E/N [Td] = (|E|/E_th,local) * E_th,STP0C/n_Loschmidt = ratio * 10.27 Td.
N_LOSCHMIDT_M3 = 2.6868e25
EN_TD_PER_THRESHOLD_RATIO = (
    RREA_THRESHOLD_STP_KV_PER_M * 1.0e3 * 1.0e21 / N_LOSCHMIDT_M3
)


def closure_table_max_en_td(table_path: Path) -> float:
    """The certified E/N ceiling of an en_table closure CSV, from its header.

    The bundle declares `valid_en_td=LO..HI` in a comment line (above HI the
    engine aborts by declared policy rather than extrapolate).  A table
    without the declaration fails closed here: the preflight cannot certify
    a range nobody stated.
    """
    pattern = re.compile(r"valid_en_td\s*=\s*([0-9.]+)\s*\.\.\s*([0-9.]+)")
    with table_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.startswith("#"):
                break
            match = pattern.search(line)
            if match:
                return float(match.group(2))
    raise ValueError(
        f"{table_path} declares no `valid_en_td=LO..HI` header; cannot "
        "preflight the closure range")


def audit_ambient_field(
    *,
    field: Profile,
    threshold: Profile,
    e0_peak_v_per_m: float,
    domain_radius_m: float,
    taper_r_start_m: float | None = None,
    taper_r_end_m: float | None = None,
    domain_height_m: float = DOMAIN_HEIGHT_M,
    altitude_at_z0_m: float = ALTITUDE_AT_Z0_M,
    nr: int = 240,
    nz: int = 480,
    pocket_cap: int = 64,
) -> dict:
    """2-D (r,z) audit of the FULL ambient field vs the local RREA threshold.

    Field construction (matches RreaFieldInitializer): Ez = f(r)*E_axial(z),
    Er = -f'(r)*(phi_z(z) - phi_ref), phi_ref = midrange of phi_z, with the
    C2 quintic taper f(r).  The intended axial field is deliberately ABOVE
    threshold in the avalanche column -- that is the experiment -- so the
    GATE is the taper-created fringe alone: a pocket is any point where
    |Er| reaches the local threshold, scanned over the WHOLE domain.  That
    includes above the field top, where phi_z plateaus while the threshold
    keeps falling. RREA is direction-agnostic, so any super-threshold radial
    pocket there can seed an avalanche.

    The TOTAL field |E| = hypot(Er, Ez) is reported, never gated: wherever
    the tapered Ez falls continuously through threshold with any fringe
    present, a thin shell with |E| >= E_th > |Ez| necessarily exists -- that
    shell is the intended column's edge, not a defect, and a total-field
    gate could never pass for any taper.

    Because E_th and the neutral density scale together (both anchored at
    1.293 kg/m^3), the same scan yields the run's worst initial E/N:
    worst_en_td = max(|E|/E_th,local) * EN_TD_PER_THRESHOLD_RATIO -- the
    quantity the en_table closure certifies (abort above its table max).
    """
    has_taper = taper_r_start_m is not None or taper_r_end_m is not None
    if has_taper:
        if taper_r_start_m is None or taper_r_end_m is None:
            raise ValueError("taper audit needs both r_start and r_end")
        if not (0.0 <= taper_r_start_m < taper_r_end_m <= domain_radius_m):
            raise ValueError(
                "taper audit requires 0 <= r_start < r_end <= domain radius")
    nr = max(2, int(nr))
    nz = max(2, int(nz))
    z_grid = [domain_height_m * k / nz for k in range(nz + 1)]
    e_axial = [
        profiled_minus_ez_v_per_m(field, e0_peak_v_per_m, altitude_at_z0_m + z) for z in z_grid
    ]
    phi = [0.0] * (nz + 1)
    for k in range(1, nz + 1):
        phi[k] = phi[k - 1] + 0.5 * (e_axial[k - 1] + e_axial[k]) * (z_grid[k] - z_grid[k - 1])
    phi_ref = 0.5 * (min(phi) + max(phi))
    threshold_z = [threshold.interpolate(altitude_at_z0_m + z) for z in z_grid]
    if min(threshold_z) <= 0.0:
        raise ValueError("threshold profile must be strictly positive over the domain")

    max_fringe_er = 0.0
    max_fringe_loc = {"r_m": 0.0, "z_m": 0.0, "er_v_per_m": 0.0}
    max_ratio = 0.0
    max_ratio_loc: dict = {"r_m": 0.0, "z_m": 0.0}
    max_fringe_ratio = 0.0
    max_fringe_ratio_loc: dict = {"r_m": 0.0, "z_m": 0.0}
    pockets: list[dict] = []
    n_pockets = 0
    for i in range(nr + 1):
        r = domain_radius_m * i / nr
        if has_taper:
            f = taper_weight(r, taper_r_start_m, taper_r_end_m)
            fp = taper_weight_deriv(r, taper_r_start_m, taper_r_end_m)
        else:
            f, fp = 1.0, 0.0
        for k in range(nz + 1):
            ez = f * e_axial[k]
            er = fp * (phi[k] - phi_ref)
            abs_er = abs(er)
            if abs_er > max_fringe_er:
                max_fringe_er = abs_er
                max_fringe_loc = {"r_m": r, "z_m": z_grid[k], "er_v_per_m": abs_er}
            e_total = math.hypot(er, ez)
            ratio = e_total / threshold_z[k]
            if ratio > max_ratio:
                max_ratio = ratio
                max_ratio_loc = {
                    "r_m": r, "z_m": z_grid[k], "e_v_per_m": e_total,
                    "threshold_v_per_m": threshold_z[k],
                }
            fringe_ratio = abs_er / threshold_z[k]
            if fringe_ratio > max_fringe_ratio:
                max_fringe_ratio = fringe_ratio
                max_fringe_ratio_loc = {
                    "r_m": r, "z_m": z_grid[k], "er_v_per_m": abs_er,
                    "threshold_v_per_m": threshold_z[k],
                }
            if abs_er >= threshold_z[k]:
                n_pockets += 1
                if len(pockets) < pocket_cap:
                    pockets.append({
                        "r_m": r,
                        "z_m": z_grid[k],
                        "e_v_per_m": e_total,
                        "er_v_per_m": er,
                        "ez_v_per_m": ez,
                        "threshold_v_per_m": threshold_z[k],
                        "er_over_threshold": fringe_ratio,
                    })

    params = {
        "e0_peak_v_per_m": e0_peak_v_per_m,
        "taper_r_start_m": taper_r_start_m,
        "taper_r_end_m": taper_r_end_m,
        "domain_radius_m": domain_radius_m,
        "domain_height_m": domain_height_m,
        "altitude_at_z0_m": altitude_at_z0_m,
        "nr": nr,
        "nz": nz,
    }
    return {
        "audit": "rrea_ambient_field_v2",
        "passed": n_pockets == 0,
        "parameters": params,
        "column_potential_span_v": max(phi) - min(phi),
        "max_fringe_er_v_per_m": max_fringe_er,
        "max_fringe_over_e0": (max_fringe_er / e0_peak_v_per_m if e0_peak_v_per_m else 0.0),
        "max_fringe_location": max_fringe_loc,
        "max_e_over_threshold": max_ratio,
        "max_e_over_threshold_location": max_ratio_loc,
        "max_fringe_er_over_threshold": max_fringe_ratio,
        "max_fringe_er_over_threshold_location": max_fringe_ratio_loc,
        "worst_en_td": max_ratio * EN_TD_PER_THRESHOLD_RATIO,
        "n_super_threshold_pockets": n_pockets,
        "pockets_truncated": n_pockets > len(pockets),
        "pockets": pockets,
    }


def estimate_integrated_avalanche_lengths(
    *,
    field: Profile,
    density_ratio: Profile,
    region: AboveThresholdRegion,
    e0_peak_v_per_m: float,
    runaway_threshold_stp_kv_per_m: float = RREA_THRESHOLD_STP_KV_PER_M,
) -> AvalancheEstimate:
    """Coleman-Dwyer avalanche-length integral over the realistic profile.

    Governing law: the local avalanche e-folding length is
    lambda(z) = 7300 kV / (|E(z)| - E_th(z)) with E in kV/m and
    E_th(z) = runaway_threshold_stp_kv_per_m * rho(z)/rho_ref (the WKB
    local-density form), so the number of e-foldings over the run is

        N_lambda = integral_{z_inj}^{z_top} max(|E(z)| - E_th(z), 0)
                   / 7300 kV  dz,      M_CD = exp(N_lambda).

    Integrated exactly on the piecewise-linear profile (breakpoints at every
    profile node plus field sign crossings). Comparisons evaluate both 276 and
    284 kV/m conventions on one physical interval; they are external anchors,
    not ground truth or software gates (Coleman & Dwyer 2006).
    """
    if runaway_threshold_stp_kv_per_m <= 0.0:
        raise ValueError("runaway_threshold_stp_kv_per_m must be positive")
    z_start = region.injection_altitude_m
    z_stop = region.top_altitude_m
    breakpoints = sorted(
        {
            z_start,
            z_stop,
            *(z for z in field.altitude_m if z_start < z < z_stop),
            *(z for z in density_ratio.altitude_m if z_start < z < z_stop),
        }
    )
    # |linear field| is piecewise linear only after splitting at a sign
    # crossing. The production profile does not cross inside this interval,
    # but keeping the helper exact makes synthetic-profile tests meaningful.
    signed_scale = e0_peak_v_per_m / field.max_abs / 1000.0
    for left, right in list(zip(breakpoints, breakpoints[1:], strict=False)):
        f0 = signed_scale * field.interpolate_zero_outside(left)
        f1 = signed_scale * field.interpolate_zero_outside(right)
        if f0 * f1 < 0.0:
            breakpoints.append(left + (right - left) * abs(f0) / (abs(f0) + abs(f1)))
    breakpoints.sort()

    predicted = 0.0
    max_ratio = 0.0
    min_ratio = math.inf
    integrated_distance = 0.0
    for z0, z1 in zip(breakpoints, breakpoints[1:], strict=False):
        dz = z1 - z0
        e0 = profiled_abs_field_v_per_m(field, e0_peak_v_per_m, z0) / 1000.0
        e1 = profiled_abs_field_v_per_m(field, e0_peak_v_per_m, z1) / 1000.0
        t0 = runaway_threshold_stp_kv_per_m * density_ratio.interpolate(z0)
        t1 = runaway_threshold_stp_kv_per_m * density_ratio.interpolate(z1)
        for e_value, threshold_value in ((e0, t0), (e1, t1)):
            if threshold_value > 0.0:
                ratio = e_value / threshold_value
                max_ratio = max(max_ratio, ratio)
                min_ratio = min(min_ratio, ratio)
        margin0 = e0 - t0
        margin1 = e1 - t1
        if margin0 >= 0.0 and margin1 >= 0.0:
            predicted += 0.5 * dz * (margin0 + margin1) / RREA_AVALANCHE_NUMERATOR_KV
            integrated_distance += dz
        elif margin0 > 0.0 or margin1 > 0.0:
            positive_fraction = (
                margin0 / (margin0 - margin1)
                if margin0 > 0.0
                else margin1 / (margin1 - margin0)
            )
            positive_distance = dz * positive_fraction
            positive_margin = max(margin0, margin1)
            predicted += 0.5 * positive_distance * positive_margin / RREA_AVALANCHE_NUMERATOR_KV
            integrated_distance += positive_distance
    multiplication = math.exp(predicted) if predicted <= 700.0 else None
    return AvalancheEstimate(
        predicted_avalanche_lengths=predicted,
        predicted_multiplication=multiplication,
        predicted_multiplication_log=predicted,
        integrated_distance_m=integrated_distance,
        max_field_to_threshold_ratio=max_ratio,
        min_field_to_threshold_ratio=min_ratio if math.isfinite(min_ratio) else math.nan,
        runaway_threshold_stp_kv_per_m=runaway_threshold_stp_kv_per_m,
    )


def _partial_positive_linear_integral(
    value0: float, value1: float, distance_m: float, fraction_stop: float = 1.0
) -> float:
    """Integral of max(linear(value0,value1), 0) over a segment prefix."""
    fraction_stop = min(max(fraction_stop, 0.0), 1.0)
    if fraction_stop == 0.0:
        return 0.0
    end_value = value0 + fraction_stop * (value1 - value0)
    segment_distance = distance_m * fraction_stop
    if value0 >= 0.0 and end_value >= 0.0:
        return 0.5 * segment_distance * (value0 + end_value)
    if value0 <= 0.0 and end_value <= 0.0:
        return 0.0
    crossing = -value0 / (end_value - value0)
    if value0 > 0.0:
        return 0.5 * segment_distance * crossing * value0
    return 0.5 * segment_distance * (1.0 - crossing) * end_value


def coleman_dwyer_profile_reference(
    *,
    field: Profile,
    density_ratio: Profile,
    threshold_profile: Profile,
    e0_peak_v_per_m: float,
    onset_threshold_stp_kv_per_m: float = 276.0,
    denominator_thresholds_stp_kv_per_m: tuple[float, ...] = (276.0,),
    injection_buffer_m: float = 50.0,
) -> dict[str, object] | None:
    """Return C&D curves on one onset-defined physical altitude interval.

    The onset threshold chooses the interval exactly once. Denominator
    thresholds change only the avalanche-length integrand and can therefore be
    compared without silently changing propagation distance.
    """
    if onset_threshold_stp_kv_per_m != 276.0:
        raise ValueError("the RREA profile benchmark uses only the 276 kV/m reference")
    threshold_validation = validate_density_scaled_threshold_profile(
        density_ratio,
        threshold_profile,
        threshold_stp_kv_per_m=onset_threshold_stp_kv_per_m,
    )
    region = find_lowest_above_threshold_region(
        e0_peak_v_per_m=e0_peak_v_per_m,
        field=field,
        threshold=threshold_profile,
        density_ratio=density_ratio,
        injection_buffer_m=injection_buffer_m,
    )
    if region is None:
        return None
    z0 = region.injection_altitude_m
    z1 = region.top_altitude_m
    knots = sorted(
        {
            z0,
            z1,
            *(z for z in field.altitude_m if z0 < z < z1),
            *(z for z in density_ratio.altitude_m if z0 < z < z1),
        }
    )
    profile_segments: list[
        tuple[float, float, float, float, float, float, list[float]]
    ] = []
    for left, right in zip(knots, knots[1:], strict=False):
        e_left = profiled_abs_field_v_per_m(field, e0_peak_v_per_m, left) / 1000.0
        e_right = profiled_abs_field_v_per_m(field, e0_peak_v_per_m, right) / 1000.0
        n_left = density_ratio.interpolate(left)
        n_right = density_ratio.interpolate(right)
        splits = [0.0, 1.0]
        for reduced_threshold in (286.0, 300.0):
            a = e_left - reduced_threshold * n_left
            b = e_right - reduced_threshold * n_right
            if a * b < 0.0:
                splits.append(a / (a - b))
        profile_segments.append(
            (left, right, e_left, e_right, n_left, n_right, sorted(set(splits)))
        )

    near_threshold_distance_m = 0.0
    near_threshold_points: list[tuple[float, float, float]] = []
    for left, right, e_left, e_right, n_left, n_right, splits in profile_segments:
        distance = right - left
        for fa, fb in zip(splits, splits[1:], strict=False):
            fm = 0.5 * (fa + fb)
            em = e_left + fm * (e_right - e_left)
            nm = n_left + fm * (n_right - n_left)
            reduced = em / nm
            if not 286.0 <= reduced <= 300.0:
                continue
            near_threshold_distance_m += distance * (fb - fa)
            for fraction in (fa, fb):
                e_value = e_left + fraction * (e_right - e_left)
                n_value = n_left + fraction * (n_right - n_left)
                denominator_value = e_value - 285.0 * n_value
                if denominator_value > 0.0:
                    near_threshold_points.append(
                        (
                            5100.0 / denominator_value,
                            left + fraction * distance,
                            e_value / n_value,
                        )
                    )

    if near_threshold_points:
        shortest_5100 = min(near_threshold_points)
        longest_5100 = max(near_threshold_points)
        near_threshold_diagnostic: dict[str, object] = {
            "applicability": "diagnostic_only_not_an_acceptance_curve",
            "applicability_reduced_field_min_kv_per_m": 286.0,
            "applicability_reduced_field_max_kv_per_m": 300.0,
            "expression": "5100 kV / (E - 285 n kV/m)",
            "altitude_span_in_applicability_band_m": near_threshold_distance_m,
            "minimum_local_length_m": shortest_5100[0],
            "minimum_local_length_altitude_m": shortest_5100[1],
            "minimum_local_length_reduced_field_kv_per_m": shortest_5100[2],
            "maximum_local_length_m": longest_5100[0],
            "maximum_local_length_altitude_m": longest_5100[1],
            "maximum_local_length_reduced_field_kv_per_m": longest_5100[2],
        }
    else:
        near_threshold_diagnostic = {
            "applicability": "diagnostic_only_not_an_acceptance_curve",
            "applicability_reduced_field_min_kv_per_m": 286.0,
            "applicability_reduced_field_max_kv_per_m": 300.0,
            "expression": "5100 kV / (E - 285 n kV/m)",
            "altitude_span_in_applicability_band_m": 0.0,
            "minimum_local_length_m": None,
            "minimum_local_length_altitude_m": None,
            "minimum_local_length_reduced_field_kv_per_m": None,
            "maximum_local_length_m": None,
            "maximum_local_length_altitude_m": None,
            "maximum_local_length_reduced_field_kv_per_m": None,
        }

    curves: dict[str, object] = {}
    for denominator in denominator_thresholds_stp_kv_per_m:
        estimate = estimate_integrated_avalanche_lengths(
            field=field,
            density_ratio=density_ratio,
            region=region,
            e0_peak_v_per_m=e0_peak_v_per_m,
            runaway_threshold_stp_kv_per_m=denominator,
        )
        total = estimate.predicted_avalanche_lengths
        high_reduced_integral = 0.0
        near_threshold_integral = 0.0
        local_lengths_m: list[float] = []
        minimum_scale_separation: dict[str, float | None] | None = None
        for left, right, e_left, e_right, n_left, n_right, splits in profile_segments:
            distance = right - left
            margin_left = e_left - denominator * n_left
            margin_right = e_right - denominator * n_right
            for fa, fb in zip(splits, splits[1:], strict=False):
                fm = 0.5 * (fa + fb)
                em = e_left + fm * (e_right - e_left)
                nm = n_left + fm * (n_right - n_left)
                reduced = em / nm
                ma = margin_left + fa * (margin_right - margin_left)
                mb = margin_left + fb * (margin_right - margin_left)
                contribution = (
                    _partial_positive_linear_integral(ma, mb, distance * (fb - fa))
                    / RREA_AVALANCHE_NUMERATOR_KV
                )
                if reduced >= 300.0:
                    high_reduced_integral += contribution
                    slope = (margin_right - margin_left) / distance
                    for fraction, margin_value in ((fa, ma), (fb, mb)):
                        if margin_value <= 0.0:
                            continue
                        local_length_m = RREA_AVALANCHE_NUMERATOR_KV / margin_value
                        local_lengths_m.append(local_length_m)
                        variation_scale_m = (
                            math.inf if slope == 0.0 else abs(margin_value / slope)
                        )
                        scale_to_length = variation_scale_m / local_length_m
                        if (
                            minimum_scale_separation is None
                            or scale_to_length
                            < float(minimum_scale_separation["scale_to_length_ratio"])
                        ):
                            point_e = e_left + fraction * (e_right - e_left)
                            point_n = n_left + fraction * (n_right - n_left)
                            minimum_scale_separation = {
                                "scale_to_length_ratio": scale_to_length,
                                "local_avalanche_length_m": local_length_m,
                                "profile_variation_scale_m": variation_scale_m,
                                "altitude_m": left + fraction * distance,
                                "reduced_field_kv_per_m": point_e / point_n,
                            }
                elif 286.0 <= reduced < 300.0:
                    near_threshold_integral += contribution
        minimum_ratio = (
            minimum_scale_separation["scale_to_length_ratio"]
            if minimum_scale_separation is not None
            and math.isfinite(
                float(minimum_scale_separation["scale_to_length_ratio"])
            )
            else None
        )
        if minimum_scale_separation is not None:
            variation_scale = minimum_scale_separation["profile_variation_scale_m"]
            if variation_scale is not None and not math.isfinite(float(variation_scale)):
                minimum_scale_separation["profile_variation_scale_m"] = None
            if minimum_ratio is None:
                minimum_scale_separation["scale_to_length_ratio"] = None
        curves[str(int(denominator))] = {
            **avalanche_estimate_to_dict(estimate),
            "denominator_threshold_stp_kv_per_m": denominator,
            "fraction_exponent_at_reduced_field_ge_300": (
                high_reduced_integral / total if total > 0.0 else 0.0
            ),
            "fraction_exponent_in_near_threshold_286_to_300": (
                near_threshold_integral / total if total > 0.0 else 0.0
            ),
            "minimum_profile_variation_scale_to_avalanche_length_ratio_at_reduced_field_ge_300": minimum_ratio,
            "local_avalanche_length_m_range_at_reduced_field_ge_300": {
                "minimum_m": min(local_lengths_m) if local_lengths_m else None,
                "maximum_m": max(local_lengths_m) if local_lengths_m else None,
            },
            "minimum_scale_separation_at_reduced_field_ge_300": minimum_scale_separation,
            "profile_variation_scale_definition": "abs((E - T n) / d(E - T n)/dz), in m",
            "local_avalanche_length_definition": "7300 kV / (E - T n), in m",
        }
    conservative = curves.get("276", {})
    conservative_lengths = float(conservative.get("predicted_avalanche_lengths", 0.0))
    fit_distance = max(0.0, region.top_altitude_m - region.injection_altitude_m)

    def altitude_at_efolds(target: float) -> float | None:
        if conservative_lengths < target:
            return None
        low = region.injection_altitude_m
        high = region.top_altitude_m
        for _ in range(56):
            mid = 0.5 * (low + high)
            partial_region = AboveThresholdRegion(
                bottom_altitude_m=region.bottom_altitude_m,
                top_altitude_m=mid,
                injection_altitude_m=region.injection_altitude_m,
                injection_z_m=region.injection_z_m,
                width_m=mid - region.bottom_altitude_m,
                e0_peak_v_per_m=region.e0_peak_v_per_m,
                field_at_injection_v_per_m=region.field_at_injection_v_per_m,
                threshold_at_injection_v_per_m=region.threshold_at_injection_v_per_m,
                density_ratio_at_injection=region.density_ratio_at_injection,
            )
            partial = estimate_integrated_avalanche_lengths(
                field=field,
                density_ratio=density_ratio,
                region=partial_region,
                e0_peak_v_per_m=e0_peak_v_per_m,
                runaway_threshold_stp_kv_per_m=276.0,
            ).predicted_avalanche_lengths
            if partial < target:
                low = mid
            else:
                high = mid
        return 0.5 * (low + high)

    return {
        "model": "rrea_profile_benchmark_reference_v1",
        "onset_threshold_stp_kv_per_m": onset_threshold_stp_kv_per_m,
        "threshold_profile_validation": threshold_validation,
        "interval_start_altitude_m": region.injection_altitude_m,
        "interval_end_altitude_m": region.top_altitude_m,
        "interval_distance_m": fit_distance,
        "curves": curves,
        "steady_state_growth_eligible": (
            conservative_lengths >= 3.5 and math.floor(fit_distance / 50.0) + 1 >= 5
        ),
        "steady_state_equilibration_efolds": 2.5,
        "steady_state_required_fit_efolds": 1.0,
        "altitude_at_s276_2p5_m": altitude_at_efolds(2.5),
        "altitude_at_s276_3p75_m": altitude_at_efolds(3.75),
        "near_threshold_5100_over_e_minus_285_is_diagnostic_only": True,
        "near_threshold_5100_over_e_minus_285": near_threshold_diagnostic,
        "above_threshold_region": region_to_dict(region),
    }


def build_cd_plane_z_m(
    reference: dict[str, object],
    n_cell_z: int,
    *,
    domain_height_m: float = DOMAIN_HEIGHT_M,
    altitude_at_z0_m: float = ALTITUDE_AT_Z0_M,
) -> list[float]:
    """Return the canonical production C&D plane sequence in simulation z."""
    if not isinstance(n_cell_z, int) or isinstance(n_cell_z, bool) or n_cell_z <= 0:
        raise ValueError("n_cell_z must be a positive integer")
    if not math.isfinite(domain_height_m) or domain_height_m <= 0.0:
        raise ValueError("domain_height_m must be finite and positive")
    dz = domain_height_m / n_cell_z
    start_altitude = float(reference["interval_start_altitude_m"])
    end_altitude = float(reference["interval_end_altitude_m"])
    candidates = [
        start_altitude - altitude_at_z0_m + 0.5 * dz,
        end_altitude - altitude_at_z0_m - 0.5 * dz,
    ]
    count = int(math.floor((end_altitude - start_altitude) / 50.0))
    candidates.extend(
        start_altitude - altitude_at_z0_m + 50.0 * index
        for index in range(1, count + 1)
        if start_altitude + 50.0 * index < end_altitude - 0.5 * dz
    )
    clipped = sorted(value for value in candidates if 0.0 < value < domain_height_m)
    unique: list[float] = []
    for value in clipped:
        if not unique or not math.isclose(
            value, unique[-1], rel_tol=0.0, abs_tol=1.0e-8
        ):
            unique.append(value)
    return unique


def avalanche_estimate_to_dict(estimate: AvalancheEstimate | None) -> dict[str, float | None]:
    if estimate is None:
        return {
            "predicted_avalanche_lengths": None,
            "predicted_multiplication": None,
            "predicted_multiplication_log": None,
            "integrated_distance_m": None,
            "max_field_to_threshold_ratio": None,
            "min_field_to_threshold_ratio": None,
            "runaway_threshold_stp_kv_per_m": None,
        }
    return {
        "predicted_avalanche_lengths": estimate.predicted_avalanche_lengths,
        "predicted_multiplication": estimate.predicted_multiplication,
        "predicted_multiplication_log": estimate.predicted_multiplication_log,
        "integrated_distance_m": estimate.integrated_distance_m,
        "max_field_to_threshold_ratio": estimate.max_field_to_threshold_ratio,
        "min_field_to_threshold_ratio": estimate.min_field_to_threshold_ratio,
        "runaway_threshold_stp_kv_per_m": estimate.runaway_threshold_stp_kv_per_m,
    }


def find_lowest_above_threshold_region(
    *,
    e0_peak_v_per_m: float,
    field: Profile,
    threshold: Profile,
    density_ratio: Profile,
    altitude_min_m: float = ALTITUDE_AT_Z0_M,
    altitude_max_m: float = ALTITUDE_AT_Z0_M + DOMAIN_HEIGHT_M,
    injection_buffer_m: float = 50.0,
) -> AboveThresholdRegion | None:
    if e0_peak_v_per_m < 0.0 or not math.isfinite(e0_peak_v_per_m):
        raise ValueError(f"E0 peak must be finite and non-negative, got {e0_peak_v_per_m!r}")
    if altitude_max_m <= altitude_min_m:
        raise ValueError(
            f"invalid altitude search range [{altitude_min_m}, {altitude_max_m}] m"
        )
    if injection_buffer_m < 0.0 or not math.isfinite(injection_buffer_m):
        raise ValueError(
            f"injection_buffer_m must be finite and non-negative, got {injection_buffer_m!r}"
        )

    def margin(altitude_m: float) -> float:
        # Positive minus-Ez means Ez is negative, so electrons accelerate upward
        # when the seed momentum is +z.
        return profiled_minus_ez_v_per_m(field, e0_peak_v_per_m, altitude_m) - threshold.interpolate(altitude_m)

    # On each interval in the union of the two profile knot sets, both the
    # imposed field and threshold are linear.  Their difference is therefore
    # linear too, so its roots and the first contiguous positive interval are
    # available analytically; a fixed-distance scan can miss a narrow interval
    # or shift an endpoint by its sampling resolution.
    knots = sorted(
        {
            altitude_min_m,
            altitude_max_m,
            *(
                altitude
                for altitude in field.altitude_m
                if altitude_min_m < altitude < altitude_max_m
            ),
            *(
                altitude
                for altitude in threshold.altitude_m
                if altitude_min_m < altitude < altitude_max_m
            ),
        }
    )
    positive_intervals: list[tuple[float, float]] = []
    for left, right in zip(knots, knots[1:], strict=False):
        left_margin = margin(left)
        right_margin = margin(right)
        if not (math.isfinite(left_margin) and math.isfinite(right_margin)):
            raise ValueError("field/threshold margin is non-finite at a profile knot")
        if left_margin == right_margin:
            if left_margin > 0.0:
                positive_intervals.append((left, right))
            continue
        root_fraction = -left_margin / (right_margin - left_margin)
        root_inside = 0.0 < root_fraction < 1.0
        root = left + root_fraction * (right - left) if root_inside else None
        if left_margin >= 0.0 and right_margin >= 0.0 and (
            left_margin > 0.0 or right_margin > 0.0
        ):
            positive_intervals.append((left, right))
        elif left_margin > 0.0 and right_margin < 0.0 and root is not None:
            positive_intervals.append((left, root))
        elif left_margin < 0.0 and right_margin > 0.0 and root is not None:
            positive_intervals.append((root, right))

    if not positive_intervals:
        return None
    bottom, top = positive_intervals[0]
    for next_bottom, next_top in positive_intervals[1:]:
        tolerance = 32.0 * math.ulp(max(abs(top), abs(next_bottom), 1.0))
        if next_bottom > top + tolerance:
            break
        top = max(top, next_top)
    width = top - bottom
    injection_altitude = 0.5 * (bottom + top) if width < 100.0 else bottom + injection_buffer_m
    injection_z = injection_altitude - ALTITUDE_AT_Z0_M
    return AboveThresholdRegion(
        bottom_altitude_m=bottom,
        top_altitude_m=top,
        injection_altitude_m=injection_altitude,
        injection_z_m=injection_z,
        width_m=width,
        e0_peak_v_per_m=e0_peak_v_per_m,
        field_at_injection_v_per_m=profiled_abs_field_v_per_m(field, e0_peak_v_per_m, injection_altitude),
        threshold_at_injection_v_per_m=threshold.interpolate(injection_altitude),
        density_ratio_at_injection=density_ratio.interpolate(injection_altitude),
    )


def deterministic_source_point(index: int, macro_count: int, source_radius_m: float) -> tuple[float, float, float, float, float]:
    golden = 0.6180339887498948482
    u = (index + 0.5) / max(macro_count, 1)
    phi = 2.0 * math.pi * ((index * golden) % 1.0)
    rho = source_radius_m * math.sqrt(u)
    x = rho * math.cos(phi)
    y = rho * math.sin(phi)
    r = math.hypot(x, y)
    z_offset = source_radius_m * (1.0 - u)
    return x, y, r, z_offset, phi


SPLITMIX64_STREAM_BASE = 0xD1B54A32D192ED03


def splitmix64_permutation(count: int, seed: int) -> list[int]:
    """Version-stable SplitMix64 Fisher-Yates permutation of ``range(count)``.

    Deliberately avoids Python's implementation-defined RNG so the permutation
    order is stable across interpreter versions.
    """
    if count <= 0:
        raise ValueError("count must be positive")
    mask = (1 << 64) - 1
    state = seed & mask

    def next_u64() -> int:
        nonlocal state
        state = (state + 0x9E3779B97F4A7C15) & mask
        value = state
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
        return (value ^ (value >> 31)) & mask

    def randbelow(bound: int) -> int:
        # Rejection removes the modulo bias for every Fisher-Yates range.
        limit = (1 << 64) - ((1 << 64) % bound)
        while True:
            value = next_u64()
            if value < limit:
                return value % bound

    indices = list(range(count))
    for index in range(count - 1, 0, -1):
        swap_index = randbelow(index + 1)
        indices[index], indices[swap_index] = indices[swap_index], indices[index]
    return indices


def conditioned_exponential_seed_energies(
    macro_count: int,
    *,
    realization_id: int,
    mean_energy_eV: float = COLEMAN_DWYER_MEAN_ENERGY_EV,
    minimum_energy_eV: float = SCHEMA6_CHARGED_ENERGY_MIN_EV,
    maximum_energy_eV: float = SCHEMA6_CHARGED_ENERGY_MAX_EV,
) -> list[float]:
    """Deterministic stratified samples from a bounded exponential law.

    The SplitMix64 Fisher-Yates shuffle keeps energy quantiles independent of
    the source's deterministic radial ordering.
    """
    if macro_count <= 0:
        raise ValueError("macro_count must be positive")
    if realization_id < 0:
        raise ValueError("realization_id must be non-negative")
    if not (0.0 <= minimum_energy_eV < maximum_energy_eV):
        raise ValueError("conditioned energy bounds must satisfy 0 <= min < max")
    if mean_energy_eV <= 0.0 or not math.isfinite(mean_energy_eV):
        raise ValueError("mean_energy_eV must be finite and positive")
    survival_min = math.exp(-minimum_energy_eV / mean_energy_eV)
    survival_max = math.exp(-maximum_energy_eV / mean_energy_eV)
    quantile_indices = splitmix64_permutation(
        macro_count, SPLITMIX64_STREAM_BASE ^ realization_id
    )

    energies: list[float] = []
    for quantile_index in quantile_indices:
        q = (quantile_index + 0.5) / macro_count
        survival = survival_min - q * (survival_min - survival_max)
        energy = -mean_energy_eV * math.log(max(survival, math.ldexp(1.0, -1074)))
        energies.append(min(max(energy, minimum_energy_eV), maximum_energy_eV))
    return energies


def proper_velocity_from_kinetic_energy(
    kinetic_energy_eV: float, *, direction: float = 1.0
) -> float:
    """Return signed ``u_z/c = gamma*beta`` for an axial electron seed."""
    if kinetic_energy_eV < 0.0 or not math.isfinite(kinetic_energy_eV):
        raise ValueError("kinetic_energy_eV must be finite and non-negative")
    if not math.isfinite(direction) or direction == 0.0:
        raise ValueError("direction must be finite and non-zero")
    gamma = 1.0 + kinetic_energy_eV / ELECTRON_REST_ENERGY_EV
    magnitude = math.sqrt(max((gamma - 1.0) * (gamma + 1.0), 0.0))
    return math.copysign(magnitude, direction)


def write_seed_schedule(
    path: Path,
    *,
    case_id: str,
    injection_z_m: float,
    macro_count: int,
    total_weight_real_electrons: float,
    seed_time_window_s: float = 0.0,
    source_radius_m: float = SOURCE_RADIUS_M,
    kinetic_energy_eV: float = SEED_ENERGY_EV,
    kinetic_energies_eV: list[float] | None = None,
    uz: float = SEED_UZ,
    fixed_axial_position: bool = False,
    source_region: str = "profiled_bottom_above_threshold_disk",
    direction_model: str = "field_aligned_positive_z",
) -> float:
    if macro_count <= 0:
        raise ValueError(f"macro_count must be positive, got {macro_count}")
    if total_weight_real_electrons <= 0.0 or not math.isfinite(total_weight_real_electrons):
        raise ValueError(
            "total_weight_real_electrons must be finite and positive, "
            f"got {total_weight_real_electrons!r}"
        )
    if seed_time_window_s < 0.0 or not math.isfinite(seed_time_window_s):
        raise ValueError(
            f"seed_time_window_s must be finite and non-negative, got {seed_time_window_s!r}"
        )
    if source_radius_m <= 0.0 or not math.isfinite(source_radius_m):
        raise ValueError(f"source_radius_m must be finite and positive, got {source_radius_m!r}")
    if kinetic_energy_eV <= 0.0 or not math.isfinite(kinetic_energy_eV):
        raise ValueError(f"kinetic_energy_eV must be finite and positive, got {kinetic_energy_eV!r}")
    if kinetic_energies_eV is not None:
        if len(kinetic_energies_eV) != macro_count:
            raise ValueError("kinetic_energies_eV length must equal macro_count")
        if any(value <= 0.0 or not math.isfinite(value) for value in kinetic_energies_eV):
            raise ValueError("every kinetic_energies_eV value must be finite and positive")
    if not math.isfinite(injection_z_m):
        raise ValueError(f"injection_z_m must be finite, got {injection_z_m!r}")
    if not math.isfinite(uz):
        raise ValueError(f"uz must be finite, got {uz!r}")
    path.parent.mkdir(parents=True, exist_ok=True)
    weight = total_weight_real_electrons / macro_count
    fieldnames = [
        "case_id",
        "macro_index",
        "time_s",
        "x_m",
        "y_m",
        "r_m",
        "z_m",
        "phi_rad",
        "source_radius_m",
        "kinetic_energy_eV",
        "ux",
        "uy",
        "uz",
        "weight_real_electrons",
        "represented_real_electrons",
        "species_id",
        "generation_id",
        "source_region",
        "direction_model",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index in range(macro_count):
            x, y, r, z_offset, phi = deterministic_source_point(index, macro_count, source_radius_m)
            if fixed_axial_position:
                z_offset = 0.0
            if seed_time_window_s > 0.0:
                time_s = seed_time_window_s * (index + 0.5) / macro_count
            else:
                time_s = 0.0
            row_energy_eV = (
                kinetic_energies_eV[index]
                if kinetic_energies_eV is not None
                else kinetic_energy_eV
            )
            row_uz = (
                proper_velocity_from_kinetic_energy(row_energy_eV, direction=uz)
                if kinetic_energies_eV is not None
                else uz
            )
            writer.writerow(
                {
                    "case_id": case_id,
                    "macro_index": index,
                    "time_s": f"{time_s:.17e}",
                    "x_m": f"{x:.17e}",
                    "y_m": f"{y:.17e}",
                    "r_m": f"{r:.17e}",
                    "z_m": f"{injection_z_m + z_offset:.17e}",
                    "phi_rad": f"{phi:.17e}",
                    "source_radius_m": f"{source_radius_m:.17e}",
                    "kinetic_energy_eV": f"{row_energy_eV:.17e}",
                    "ux": "0.0",
                    "uy": "0.0",
                    "uz": f"{row_uz:.17e}",
                    "weight_real_electrons": f"{weight:.17e}",
                    "represented_real_electrons": f"{weight:.17e}",
                    "species_id": "1",
                    "generation_id": "0",
                    "source_region": source_region,
                    "direction_model": direction_model,
                }
            )
    return total_weight_real_electrons


def launcher_prefix(
    kind: str, ranks: int, cpus_per_rank: int, nice: int = 0
) -> list[str]:
    """Argv prefix that starts the engine, optionally at reduced priority.

    `nice` is applied once, in front of the launcher, and every rank inherits
    it: a long local campaign then yields the machine to anything at nice 0
    instead of making it unusable for days.  Set at launch and never changed
    afterwards -- renicing a running MPI job mid-flight skews the ranks
    against each other, which is exactly what a lockstep field solve must not
    have.
    """
    if not 0 <= nice <= 19:
        raise SystemExit("--nice must be in [0, 19]; a negative value needs root")
    lead = ["nice", "-n", str(nice)] if nice > 0 else []
    if kind == "none":
        if ranks != 1:
            raise SystemExit("--launcher=none only supports one rank")
        return lead
    if kind == "srun":
        prefix = lead + ["srun", "--kill-on-bad-exit=1", "-n", str(ranks)]
        if cpus_per_rank > 0:
            prefix.extend(["-c", str(cpus_per_rank)])
        return prefix
    if kind == "mpiexec":
        return lead + ["mpiexec", "-n", str(ranks)]
    raise SystemExit(f"unknown launcher: {kind}")


def read_reduced(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise SystemExit(f"missing reduced diagnostics: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"empty reduced diagnostics: {path}")
    return rows


def profile_metadata(field: Profile, density: Profile, threshold: Profile) -> dict[str, object]:
    return {
        "altitude_mapping": "altitude_m = altitude_at_z0_m + z_m",
        "domain_altitude_min_m": ALTITUDE_AT_Z0_M,
        "domain_altitude_max_m": ALTITUDE_AT_Z0_M + DOMAIN_HEIGHT_M,
        "field_profile_path": str(field.path),
        "field_profile_peak_abs_v_per_m": field.max_abs,
        "field_profile_zero_outside_range": True,
        "field_profile_min_altitude_m": field.altitude_m[0],
        "field_profile_max_altitude_m": field.altitude_m[-1],
        "density_profile_path": str(density.path),
        "threshold_profile_path": str(threshold.path),
    }


def region_to_dict(region: AboveThresholdRegion | None) -> dict[str, float | None]:
    if region is None:
        return {
            "bottom_altitude_m": None,
            "top_altitude_m": None,
            "injection_altitude_m": None,
            "injection_z_m": None,
            "width_m": None,
            "e0_peak_v_per_m": None,
            "field_at_injection_v_per_m": None,
            "threshold_at_injection_v_per_m": None,
            "density_ratio_at_injection": None,
        }
    return {
        "bottom_altitude_m": region.bottom_altitude_m,
        "top_altitude_m": region.top_altitude_m,
        "injection_altitude_m": region.injection_altitude_m,
        "injection_z_m": region.injection_z_m,
        "width_m": region.width_m,
        "e0_peak_v_per_m": region.e0_peak_v_per_m,
        "field_at_injection_v_per_m": region.field_at_injection_v_per_m,
        "threshold_at_injection_v_per_m": region.threshold_at_injection_v_per_m,
        "density_ratio_at_injection": region.density_ratio_at_injection,
    }


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def ensure_profiles_cover_domain(field: Profile, density: Profile, threshold: Profile) -> None:
    domain_min = ALTITUDE_AT_Z0_M
    domain_max = ALTITUDE_AT_Z0_M + DOMAIN_HEIGHT_M
    if field.altitude_m[0] > domain_min:
        raise ValueError(
            f"{field.path} does not cover profiled domain lower altitude "
            f"{domain_min} m; field is zero only above its maximum altitude"
        )
    for profile in [density, threshold]:
        if profile.altitude_m[0] > domain_min or profile.altitude_m[-1] < domain_max:
            raise ValueError(
                f"{profile.path} does not cover profiled domain "
                f"[{domain_min}, {domain_max}] m"
            )
