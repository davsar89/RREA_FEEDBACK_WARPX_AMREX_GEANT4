#!/usr/bin/env python3
"""Check the active low-energy closure against published dry-air swarm data.

Production reads mobility, tensor diffusion and attachment from the zero--100
Td hybrid dry-air bundle; this checks the shipped table against cited source
rows using unit-explicit conversions.

REFERENCE values are cited measurements; DERIVED values are recomputed here.
The primary source is J. de Urquijo, O. Gonzalez-Magana, E. Basurto and
A. M. Juarez, J. Phys. D 57, 125205 (2024), DOI 10.1088/1361-6463/ad164a.
Its dry-air flux data reach 3 Td.  The low-field extension uses Hegerberg &
Reid's measured dry-air drift velocities, Milloy et al.'s zero-field mobility,
Shimamori--Hatano's thermal attachment, Morrow--Lowke's attachment fit, and
Dutton's diffusion data.

Usage:
    python scripts/verify_fluid_closure_anchors.py
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# --------------------------------------------------------------------------
# REFERENCE -- published, cited, taken as true.
# --------------------------------------------------------------------------

# de Urquijo, Gonzalez-Magana, Basurto & Juarez, J. Phys. D 57, 125205
# (2024), DOI 10.1088/1361-6463/ad164a.  Pulsed-Townsend measurements at room
# temperature; dry-air (0% H2O) columns of Table 2 (flux drift velocity W_F,
# ~1-1.5% uncertainty) and Table 5 (density-squared-normalized three-body
# attachment coefficient eta/N^2, 6-10% for dry air).  The paper defines the
# three-body rate coefficient as k3 = W_F * eta/N^2.
#
# W_F is the FLUX drift velocity, which is the correct velocity for the
# number-flux/conductivity closure this repo runs, and it is measured down to
# 3 Td.  Tables 2 and 5 do NOT print values at the same E/N nodes.  In
# particular, the dry-air Table-5 cells at 3.3, 3.6, 4.5, 5.5, and 6.5 Td are
# blank; values from the humid columns must not be substituted.  Keep the two
# primary-source transcriptions separate.
DE_URQUIJO_DRY_AIR_DRIFT = (
    # E/N [Td], W_F [1e5 cm/s], Table 2 dry-air (0% H2O)
    (3.0, 10.9), (3.3, 11.2), (3.6, 11.8), (4.0, 12.4),
    (4.5, 13.0), (5.0, 13.6), (5.5, 14.3), (6.0, 15.0),
    (6.5, 15.6), (7.0, 16.3), (8.0, 17.5), (9.0, 18.9),
    (10.0, 20.1), (12.0, 22.8), (14.0, 25.3), (16.0, 27.8),
    (18.0, 30.2), (20.0, 32.6), (23.0, 36.1), (26.0, 39.4),
    (30.0, 43.7), (33.0, 46.8), (36.0, 50.1), (40.0, 54.1),
    (45.0, 59.1), (50.0, 63.8), (55.0, 68.6), (60.0, 73.4),
    (65.0, 78.1), (70.0, 83.3), (80.0, 93.2), (90.0, 103.0),
    (100.0, 114.0),
)
DE_URQUIJO_DRY_AIR_ATTACHMENT = (
    # E/N [Td], eta/N^2 [1e-39 cm^5], Table 5 dry-air (0% H2O)
    (3.0, 50.30), (4.0, 38.40), (5.0, 26.40), (6.0, 20.20),
    (7.0, 16.40), (8.0, 13.60), (9.0, 12.20), (10.0, 10.30),
    (12.0, 8.62), (13.0, 7.78), (14.0, 7.07), (15.0, 6.69),
    (16.0, 6.81), (17.0, 5.64), (18.0, 5.79), (20.0, 5.08),
    (23.0, 4.51), (26.0, 4.16), (30.0, 3.54),
)

# Same paper, Table 3: measured dry-air density-normalized longitudinal
# diffusion N*D_L in 1e21 cm^-1 s^-1.  The printed 7-Td value is 0.00,
# which cannot be a physical diffusion coefficient and is excluded rather
# than silently treated as a zero.  Blank cells (including 10 Td) are not
# source data.  Below the first usable 6-Td node the project holds that node,
# matching the closure's declared low-field policy.
DE_URQUIJO_DRY_AIR_LONGITUDINAL_DIFFUSION = (
    # E/N [Td], N*D_L [1e21 cm^-1 s^-1]
    (6.0, 8.27), (6.5, 8.10), (8.0, 7.11), (9.0, 7.65),
    (12.0, 8.29), (14.0, 7.64), (16.0, 7.80), (18.0, 7.67),
    (20.0, 7.71), (23.0, 6.15), (26.0, 5.51), (30.0, 6.92),
    (33.0, 5.79), (40.0, 6.24), (45.0, 9.23), (50.0, 8.41),
    (55.0, 8.39), (60.0, 10.6), (80.0, 13.0), (100.0, 18.4),
)
# Dutton, J. Phys. Chem. Ref. Data 4, 577 (1975), Table 2.19 [docs/papers/1.555525.pdf]:
# experimental transverse characteristic energy D_T/mu for dry CO2-free air.
# Units are volts.  The full bracketing source grid is retained so the 30-Td
# closure endpoint is interpolated, never extrapolated.
DUTTON_DRY_AIR_TRANSVERSE_DIFFUSION_TO_MOBILITY = (
    # E/N [Td], D_T/mu [V]
    (0.2980, 0.099), (0.600, 0.146), (0.894, 0.179),
    (1.192, 0.206), (1.490, 0.233), (1.788, 0.261),
    (2.086, 0.293), (2.384, 0.320), (2.682, 0.350),
    (2.980, 0.390), (3.58, 0.450), (4.47, 0.540),
    (5.36, 0.610), (5.96, 0.660), (8.94, 0.820),
    (11.92, 0.930), (14.90, 1.00), (29.80, 1.18),
    (44.7, 1.32), (59.6, 1.48), (60.6, 1.50),
    (75.8, 1.73), (90.9, 1.97), (106.1, 2.24), (121.2, 2.53),
)

# Hegerberg & Reid, Aust. J. Phys. 33, 227 (1980), Table 1 [docs/papers/ph800227a.pdf]: best estimates
# of attachment-corrected drift velocity in dry CO2-free air at 293 K.
# Units are 1e5 cm/s.  Their uncertainty is 3% above 0.2 Td and 5% at 0.1 Td.
HEGERBERG_REID_DRY_AIR_DRIFT = (
    (0.10, 2.55), (0.12, 2.71), (0.14, 2.85), (0.17, 2.90),
    (0.20, 3.06), (0.25, 3.26), (0.30, 3.46), (0.35, 3.66),
    (0.40, 3.87), (0.50, 4.34), (0.60, 4.77), (0.80, 5.63),
    (1.00, 6.35),
)
# Milloy, Reid & Crompton, Aust. J. Phys. 28, 231 (1975), DOI 10.1071/PH750231:
# lim(E/N->0) W/(E/N) = (4.7 +/- 0.2)e6 cm/s/Td in dry air.
ZERO_FIELD_DRIFT_CM_S_PER_TD = 4.7e6
ZERO_FIELD_PROXY_TD = 1.0e-6
# Morrow & Lowke, J. Phys. D 30, 614 (1997), DOI 10.1088/0022-3727/30/4/017.
MORROW_LOWKE_ETA3_PREF_CM5 = 4.7778e-59
MORROW_LOWKE_ETA3_EXPONENT = -1.2749
MORROW_LOWKE_ETA2_SLOPE = 6.089e-4
MORROW_LOWKE_ETA2_INTERCEPT_CM2 = -2.893e-19
MORROW_LOWKE_CHARACTERISTIC_ENERGY_PREF_V = 0.3341e9
MORROW_LOWKE_CHARACTERISTIC_ENERGY_EXPONENT = 0.54069
# Shimamori & Hatano, Chem. Phys. 12, 439 (1976), DOI 10.1016/0301-0104(76)87082-6:
# e+O2+O2 and e+O2+N2.  Convert to a coefficient multiplying total N^2.
THERMAL_K3_O2_THIRD_BODY_CM6_S = 2.4e-30
THERMAL_K3_N2_THIRD_BODY_CM6_S = 8.5e-32
DRY_AIR_O2_FRACTION = 0.21
DRY_AIR_N2_FRACTION = 0.79


def _primary_log_interp(rows: tuple[tuple[float, float], ...], x: float) -> float:
    """Log-linear interpolation inside one positive primary-source table."""
    for x0, y0 in rows:
        if x == x0:
            return y0
    for (x0, y0), (x1, y1) in zip(rows, rows[1:]):
        if x0 < x < x1:
            f = (math.log(x) - math.log(x0)) / (math.log(x1) - math.log(x0))
            return math.exp(math.log(y0) + f * (math.log(y1) - math.log(y0)))
    raise ValueError(f"{x:g} Td is outside the transcribed source range")


# Engine-facing closure grid: union of all source nodes inside 3--30 Td.  Only
# a missing quantity is interpolated; every usable printed dry-air value
# remains unchanged.  Adding exact interpolated nodes does not change the
# existing log-log mobility/attachment curves.
DE_URQUIJO_DRY_AIR = tuple(
    (
        en_td,
        _primary_log_interp(DE_URQUIJO_DRY_AIR_DRIFT, en_td),
        _primary_log_interp(DE_URQUIJO_DRY_AIR_ATTACHMENT, en_td),
    )
    for en_td in sorted(
        {row[0] for row in DE_URQUIJO_DRY_AIR_DRIFT if row[0] <= 30.0}
        | {row[0] for row in DE_URQUIJO_DRY_AIR_ATTACHMENT}
        | {row[0] for row in DE_URQUIJO_DRY_AIR_LONGITUDINAL_DIFFUSION
           if row[0] <= 30.0}
        | {
            row[0]
            for row in DUTTON_DRY_AIR_TRANSVERSE_DIFFUSION_TO_MOBILITY
            if 3.0 <= row[0] <= 30.0
        }
    )
)
LOSCHMIDT_CM3 = 2.6868e19
LOSCHMIDT_M3 = 2.6868e25
TOWNSEND_V_M2 = 1.0e-21


def low_field_nodes_td() -> tuple[float, ...]:
    """Source-node union for the source-anchored zero--3 Td extension."""
    return tuple(sorted(
        {ZERO_FIELD_PROXY_TD, 3.0}
        | {row[0] for row in HEGERBERG_REID_DRY_AIR_DRIFT}
        | {
            row[0]
            for row in DUTTON_DRY_AIR_TRANSVERSE_DIFFUSION_TO_MOBILITY
            if 0.1 <= row[0] < 3.0
        }
    ))


def low_field_drift_m_s(en_td: float) -> float:
    """Measured W below 1 Td; source-bracketed interpolation to 3 Td."""
    if en_td == ZERO_FIELD_PROXY_TD:
        return ZERO_FIELD_DRIFT_CM_S_PER_TD * en_td * 1.0e-2
    measured = [(td, w * 1.0e3) for td, w in HEGERBERG_REID_DRY_AIR_DRIFT]
    if en_td <= 1.0:
        return _primary_log_interp(tuple(measured), en_td)
    return _primary_log_interp(
        ((1.0, measured[-1][1]), (3.0, 10900.0)), en_td)


def low_field_eta_over_n2_cm5(en_td: float, w_m_s: float) -> float:
    """Dry-air three-body attachment: thermal limit then Appendix-A5 fit."""
    if en_td == 3.0:
        return dict(DE_URQUIJO_DRY_AIR_ATTACHMENT)[3.0] * 1.0e-39
    if en_td == ZERO_FIELD_PROXY_TD:
        k3 = DRY_AIR_O2_FRACTION * (
            DRY_AIR_O2_FRACTION * THERMAL_K3_O2_THIRD_BODY_CM6_S
            + DRY_AIR_N2_FRACTION * THERMAL_K3_N2_THIRD_BODY_CM6_S
        )
        return k3 / (w_m_s * 100.0)
    return MORROW_LOWKE_ETA3_PREF_CM5 * (en_td * 1.0e-17) ** (
        MORROW_LOWKE_ETA3_EXPONENT)


def low_field_diffusion_ref(
    en_td: float, k0_m2_per_v_s: float
) -> tuple[float, float]:
    """D_L,D_T at N0; thermal/isotropic limit and measured D_T/mu."""
    if en_td == ZERO_FIELD_PROXY_TD:
        d_t = k0_m2_per_v_s * (8.617333262e-5 * 294.0)
    else:
        if en_td < 0.298:
            dt_over_mu = MORROW_LOWKE_CHARACTERISTIC_ENERGY_PREF_V * (
                en_td * 1.0e-17) ** MORROW_LOWKE_CHARACTERISTIC_ENERGY_EXPONENT
        else:
            dt_over_mu = _primary_log_interp(
                DUTTON_DRY_AIR_TRANSVERSE_DIFFUSION_TO_MOBILITY, en_td)
        d_t = k0_m2_per_v_s * dt_over_mu
    if en_td <= 1.0:
        return d_t, d_t
    k0_1 = low_field_drift_m_s(1.0) / (
        TOWNSEND_V_M2 * LOSCHMIDT_M3)
    d_l_1 = low_field_diffusion_ref(1.0, k0_1)[1]
    d_l_3 = diffusion_coefficients_ref(3.0, de_urquijo_row(3.0)[0])[0]
    return _primary_log_interp(((1.0, d_l_1), (3.0, d_l_3)), en_td), d_t


def high_field_nodes_td() -> tuple[float, ...]:
    """Source-node union for the dry-air 30--100 Td transition."""
    return tuple(sorted(
        {row[0] for row in DE_URQUIJO_DRY_AIR_DRIFT if 30.0 < row[0] <= 100.0}
        | {row[0] for row in DE_URQUIJO_DRY_AIR_LONGITUDINAL_DIFFUSION
           if 30.0 < row[0] <= 100.0}
        | {row[0] for row in DUTTON_DRY_AIR_TRANSVERSE_DIFFUSION_TO_MOBILITY
           if 30.0 < row[0] <= 100.0}
    ))


def high_field_drift_m_s(en_td: float) -> float:
    """de Urquijo Table-2 dry-air flux drift, log-interpolated."""
    return 1.0e3 * _primary_log_interp(DE_URQUIJO_DRY_AIR_DRIFT, en_td)


def morrow_lowke_eta3_over_n2_cm5(en_td: float) -> float:
    """Morrow--Lowke Appendix A5 three-body attachment coefficient."""
    return MORROW_LOWKE_ETA3_PREF_CM5 * (en_td * 1.0e-17) ** (
        MORROW_LOWKE_ETA3_EXPONENT)


def morrow_lowke_eta2_over_n_cm2(en_td: float) -> float:
    """Morrow--Lowke Appendix A4 two-body coefficient, clipped at onset."""
    return max(
        0.0,
        MORROW_LOWKE_ETA2_SLOPE * en_td * 1.0e-17
        + MORROW_LOWKE_ETA2_INTERCEPT_CM2,
    )


def two_body_coefficients_ref(
    en_td: float, w_m_s: float
) -> tuple[float, float, float]:
    """DERIVED: eta/N [cm2], k2 [cm3/s], nu2(N0) [1/s]."""
    eta_over_n = morrow_lowke_eta2_over_n_cm2(en_td)
    k2 = w_m_s * 100.0 * eta_over_n
    return eta_over_n, k2, k2 * LOSCHMIDT_CM3


# --------------------------------------------------------------------------
# DERIVED -- unit-explicit conversions from the de Urquijo dry-air columns.
#
#   W_F                 cm/s     (printed unit 1e5 cm/s)
#   eta/N^2             cm^5     (printed unit 1e-39 cm^5)
#   k3 = W_F * eta/N^2  cm^6/s
#   N0^2                cm^-6
#   nu3_ref = k3*N0^2   s^-1     (frequency at the reference density N0)
#   K0 = W_F/(E/N * N0) m^2/V/s  (reduced flux mobility at N0)
# --------------------------------------------------------------------------


def derived_coefficients(
    en_td: float,
    w_units_1e5_cm_s: float,
    eta_over_n2_units_1e39_cm5: float,
) -> tuple[float, float, float]:
    """DERIVED: (K0 [m^2/V/s], k3 [cm^6/s], nu3_ref [1/s]) from printed rows."""
    if not (en_td > 0.0):
        raise ValueError("E/N must be positive")
    w_cm_s = w_units_1e5_cm_s * 1.0e5
    w_m_s = w_cm_s * 1.0e-2
    eta_over_n2_cm5 = eta_over_n2_units_1e39_cm5 * 1.0e-39
    k0_m2_per_v_s = w_m_s / (en_td * TOWNSEND_V_M2 * LOSCHMIDT_M3)
    k3_cm6_per_s = w_cm_s * eta_over_n2_cm5
    nu3_ref_s = k3_cm6_per_s * LOSCHMIDT_CM3**2
    return k0_m2_per_v_s, k3_cm6_per_s, nu3_ref_s


def diffusion_coefficients_ref(
    en_td: float,
    k0_m2_per_v_s: float,
) -> tuple[float, float]:
    """DERIVED: (D_L, D_T) at Loschmidt density in m^2/s.

    D_L follows de Urquijo Table 3, with the first usable node held below
    6 Td.  D_T = K0*(D_T/mu), using Dutton Table 2.19.  Both are reduced
    transport data and therefore scale locally as 1/(N/N0).
    """
    if not (en_td > 0.0 and k0_m2_per_v_s > 0.0):
        raise ValueError("diffusion inputs must be positive")
    if en_td <= DE_URQUIJO_DRY_AIR_LONGITUDINAL_DIFFUSION[0][0]:
        nd_l_units = DE_URQUIJO_DRY_AIR_LONGITUDINAL_DIFFUSION[0][1]
    else:
        nd_l_units = _primary_log_interp(
            DE_URQUIJO_DRY_AIR_LONGITUDINAL_DIFFUSION, en_td
        )
    # [1e21 cm^-1 s^-1] / [cm^-3] -> cm^2/s, then cm^2 -> m^2.
    d_l_ref_m2_per_s = nd_l_units * 1.0e21 / LOSCHMIDT_CM3 * 1.0e-4
    dt_over_mu_v = _primary_log_interp(
        DUTTON_DRY_AIR_TRANSVERSE_DIFFUSION_TO_MOBILITY, en_td
    )
    d_t_ref_m2_per_s = k0_m2_per_v_s * dt_over_mu_v
    return d_l_ref_m2_per_s, d_t_ref_m2_per_s


def de_urquijo_row(en_td: float) -> tuple[float, float, float]:
    """DERIVED at an exact table node: (K0, k3, nu3_ref)."""
    for td, w_u, eta_u in DE_URQUIJO_DRY_AIR:
        if td == en_td:
            return derived_coefficients(td, w_u, eta_u)
    raise KeyError(f"{en_td} Td is not a de Urquijo dry-air node")


def _read_closure_rows() -> dict[float, dict[str, str]]:
    path = (
        REPO
        / "rrea_fluid_closure"
        / "dry_air_swarm_hybrid_v3"
        / "electron_flux_mobility_attachment_diffusion.csv"
    )
    with path.open(newline="", encoding="ascii") as handle:
        data = [line for line in handle if line.strip() and not line.startswith("#")]
    return {float(row["en_td"]): row for row in csv.DictReader(data)}


def main() -> int:
    rows = _read_closure_rows()
    failures: list[str] = []
    checks = 0

    def check(en_td: float, column: str, expected: float, source: str) -> None:
        nonlocal checks
        checks += 1
        row = rows.get(en_td)
        if row is None:
            failures.append(f"{source}: missing {en_td:g} Td row")
            return
        observed = float(row[column])
        if not math.isclose(observed, expected, rel_tol=2.0e-12, abs_tol=1.0e-30):
            failures.append(
                f"{source}: {en_td:g} Td {column}={observed:.12g}, "
                f"expected {expected:.12g}"
            )

    for en_td, drift_units in HEGERBERG_REID_DRY_AIR_DRIFT:
        check(
            en_td,
            "flux_drift_velocity_m_per_s",
            drift_units * 1.0e3,
            "Hegerberg-Reid Table 1",
        )

    for en_td, drift_units in DE_URQUIJO_DRY_AIR_DRIFT:
        check(
            en_td,
            "flux_drift_velocity_m_per_s",
            drift_units * 1.0e3,
            "de Urquijo Table 2",
        )

    for en_td, eta_units in DE_URQUIJO_DRY_AIR_ATTACHMENT:
        check(
            en_td,
            "eta_over_n2_cm5",
            eta_units * 1.0e-39,
            "de Urquijo Table 5",
        )

    for en_td, nd_l_units in DE_URQUIJO_DRY_AIR_LONGITUDINAL_DIFFUSION:
        check(
            en_td,
            "longitudinal_diffusion_ref_m2_per_s",
            nd_l_units * 1.0e21 / LOSCHMIDT_CM3 * 1.0e-4,
            "de Urquijo Table 3",
        )

    for en_td, dt_over_mu_v in DUTTON_DRY_AIR_TRANSVERSE_DIFFUSION_TO_MOBILITY:
        if en_td > 100.0:
            continue
        row = rows.get(en_td)
        if row is None:
            failures.append(f"Dutton Table 2.19: missing {en_td:g} Td row")
            continue
        checks += 1
        observed = (
            float(row["transverse_diffusion_ref_m2_per_s"])
            / float(row["reduced_flux_mobility_ref_m2_per_v_s"])
        )
        if not math.isclose(
            observed, dt_over_mu_v, rel_tol=2.0e-12, abs_tol=1.0e-30
        ):
            failures.append(
                f"Dutton Table 2.19: {en_td:g} Td D_T/mu={observed:.12g}, "
                f"expected {dt_over_mu_v:.12g}"
            )

    if min(rows) != ZERO_FIELD_PROXY_TD or max(rows) != 100.0:
        failures.append(
            f"closure range is {min(rows):g}--{max(rows):g} Td, "
            f"expected {ZERO_FIELD_PROXY_TD:g}--100 Td"
        )
    if float(rows[7.0]["longitudinal_diffusion_ref_m2_per_s"]) <= 0.0:
        failures.append("excluded nonphysical 7 Td diffusion zero was retained")

    if failures:
        print("\n".join(f"FAIL {failure}" for failure in failures))
        return 1
    print(f"OK {checks} published dry-air anchors across {len(rows)} closure rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
