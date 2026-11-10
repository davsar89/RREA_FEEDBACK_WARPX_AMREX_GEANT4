#!/usr/bin/env python3
"""Arithmetic checks for the reviewed 337 nm RREA calculation (stdlib only).

This is NOT electron transport, a fluorescence calibration, or a FEGS simulator.
The original in-field case and an optional contained, field-free exit component
are separate. The peak sum assumes the same Gaussian width, time and distance.

Sources and conventions are documented in rrea_337_back_of_envelope_reviewed.tex.
Run: python rrea_337_review_check.py --distance-m 8000
"""
from __future__ import annotations
import argparse
import json
import math

H = 6.62607015e-34       # J s
C = 299792458.0         # m/s
KB = 1.380649e-23       # J/K
E_CHARGE = 1.602176634e-19
PHOTON_J = H * C / 337.1e-9
N0 = 2.688e25           # Xu reference molecule density, m^-3
Q = 0.24               # rounded 12 km/reference density ratio
AC = 2.0e7             # Xu effective C-state radiative rate, s^-1
Q0 = N0 * (0.79e-17 + 0.21 * 3.0e-16)
FIT_U_V = 7.3e6
FIT_DRAG_V_PER_M = 2.76e5
FIELD12_V_PER_M = Q * 4.3e5


def positive(text: str) -> float:
    value = float(text)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("value must be finite and positive")
    return value


def fraction(text: str) -> float:
    value = float(text)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise argparse.ArgumentTypeError("fraction must lie between 0 and 1")
    return value


def airfly_yield(p_hpa: float, t_k: float, alpha: float) -> float:
    """Photons/MeV deposited; dry air, AIRFLY pressure/temperature parametrization.

    alpha=0: constant quenching cross section, NOT constant rate coefficient.
    alpha=-0.36: measured 337 nm exponent; use at 217 K extrapolates the
    240--310 K measurement interval. Does not calibrate the field-driven yield.
    """
    pprime_hpa = 15.89 * (t_k / 293.0) ** (0.5 - alpha)
    return 5.61 * (1.0 + 1013.25 / 15.89) / (1.0 + p_hpa / pprime_hpa)


def signal(photons: float, sigma_s: float, distance_m: float) -> dict[str, float]:
    energy_j = photons * PHOTON_J
    peak_w = energy_j / (math.sqrt(2 * math.pi) * sigma_s)
    area_m2 = 4 * math.pi * distance_m**2
    return dict(photons=photons, energy_J=energy_j, peak_source_W=peak_w,
                peak_irradiance_W_m2=peak_w / area_m2,
                fluence_J_m2=energy_j / area_m2)


def original_arithmetic_check() -> dict[str, float]:
    """Checks rounded numbers only; these assertions do not validate physics."""
    lam = FIT_U_V / (FIELD12_V_PER_M - FIT_DRAG_V_PER_M * Q)
    prad0 = AC / (AC + Q0)
    prad12 = AC / (AC + Q * Q0)
    correction = Q * prad12 / prad0
    s = signal(3e18, 50e-6, 1000.0)
    dt, sigma = 10e-6, 50e-6
    boxcar = (math.sqrt(2 * math.pi) * sigma / dt
              * math.erf(dt / (2 * math.sqrt(2) * sigma)))
    computed = dict(lambda_m=lam, Q0=Q0, Prad0=prad0, Prad12=prad12,
                    dens_corr=correction, Ngam=s['photons'], W_J=s['energy_J'],
                    Ppk_W=s['peak_source_W'], Fpk_1km=s['peak_irradiance_W_m2'],
                    tauC=1 / (AC + Q * Q0), boxcar=boxcar)
    rounded = dict(lambda_m=197.5, Q0=1.906e9, Prad0=0.0104, Prad12=0.0419,
                   dens_corr=0.968, Ngam=3e18, W_J=1.768, Ppk_W=1.411e4,
                   Fpk_1km=1.122e-3, tauC=2.09e-9, boxcar=0.99834)
    for name, expected in rounded.items():
        if abs(computed[name] / expected - 1) >= 3e-3:
            raise AssertionError((name, computed[name], expected))
    return computed


def legacy_coefficient(relative_density: float) -> dict[str, float]:
    """Replicate the scalar coefficient of supplied fgf_module.py:get_337.

    NOT a repaired model. Uses that routine's monoenergetic 7.3 MeV parent,
    16 eV representative secondary, 10 eV window and range-based lifetime.
    This is whole modeled C-state radiative emission, NOT a 337 nm band yield.
    The 337/(300--430) fraction cannot be reused as its whole-2P branching.
    Source: docs/papers/optical/Lehtinen_codes/fgf_module.py.
    """
    n0_code = 101325.0 / (KB * 273.15)
    nm = n0_code * relative_density
    mc2_j = 9.1093837139e-31 * C**2
    rclass = 2.8179403205e-15
    e0, es, width = 7.3e6, 16.0, 10.0
    gamma = 1 + e0 * E_CHARGE / mc2_j
    v0 = C * math.sqrt(1 - gamma**-2)
    kappa = 2 * math.pi * rclass**2 * mc2_j * C**2

    def cross_section(z: float, ei: float) -> float:
        return kappa * z / v0**2 / ((ei + es) * E_CHARGE)**2 * E_CHARGE

    differential = 0.8 * cross_section(14, 13) + 0.2 * cross_section(16, 17.4)
    range_m = (8.93e19 + 1.11e21 * (es / 1e3)**1.67) / nm
    birth_per_m = differential * width * nm
    alpha_c = 0.2 * 3e-16 + 0.8 * 0.13e-16
    a_c = 2.38e7
    radiative_fraction = a_c / (a_c + nm * alpha_c)
    eta_c = range_m * birth_per_m * 3.8e-21 * 0.8 * nm * radiative_fraction
    return dict(whole_C_photons_per_electron_m=eta_c,
                direct_births_in_representative_window_per_m=birth_per_m)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--distance-m', type=positive, default=1000.0)
    parser.add_argument('--electrons', type=positive, default=1e16)
    parser.add_argument('--sigma-us', type=positive, default=50.0)
    parser.add_argument('--eta337', type=positive, default=1.5,
                        help='already-quenched in-field 337 band photons/electron-m')
    parser.add_argument('--leff-m', type=positive, default=200.0)
    parser.add_argument('--pressure-hpa', type=positive, default=194.0)
    parser.add_argument('--temperature-k', type=positive, default=216.65)
    parser.add_argument('--exit-mean-mev', type=positive, default=8.3,
                        help='mean crossing energy for electrons counted above 1 MeV')
    parser.add_argument('--deposition-fraction', type=fraction, default=1.0,
                        help='fraction of exiting kinetic energy deposited in chosen post-field volume')
    args = parser.parse_args()
    baseline = original_arithmetic_check()
    sigma_s = args.sigma_us * 1e-6
    y_const_xsec = airfly_yield(args.pressure_hpa, args.temperature_k, 0.0)
    y_measured_alpha = airfly_yield(args.pressure_hpa, args.temperature_k, -0.36)
    nin = args.eta337 * args.electrons * args.leff_m
    nexit = (y_measured_alpha * args.deposition_fraction * args.electrons
             * args.exit_mean_mev)
    in_signal = signal(nin, sigma_s, args.distance_m)
    exit_signal = signal(nexit, sigma_s, args.distance_m)
    total_signal = signal(nin + nexit, sigma_s, args.distance_m)
    lam = baseline['lambda_m']
    work = FIELD12_V_PER_M * lam / 1e6
    simplified_loss = work - 7.3
    rho_g_cm3 = Q * N0 * 28.96 * 1.66054e-27 * 1e-3
    approx_stopping_MeV_m = 1.8 * rho_g_cm3 * 100
    plot_base = signal(3e18, 50e-6, 7400.0)
    # Use unscattered fluences only. The 50 mW/m2 peak is the scattered curve.
    old_plot_fluence = 6e-6
    output = dict(
        assumptions=vars(args), arithmetic_assertions_pass=True,
        original_rounded_example=baseline,
        inferred_Xu_eta337_12km=6.2 * 0.25 * baseline['dens_corr'],
        AIRFLY_cross_check=dict(
            constant_cross_section_ph_MeV=y_const_xsec,
            fitted_alpha_extrapolation_ph_MeV=y_measured_alpha,
            fixed_density_k217_over_k293=(args.temperature_k / 293.0)**0.14,
            field_free_ground_ph_e_m=5.61 * 1.8 * 1.205e-3 * 100,
            field_free_12km_constant_xsec_ph_e_m=y_const_xsec * approx_stopping_MeV_m,
            field_free_12km_fitted_alpha_ph_e_m=y_measured_alpha * approx_stopping_MeV_m,
            stopping_power_1p8_is_review_approximation_not_new_ESTAR_table=True),
        in_field=in_signal, post_exit=exit_signal, combined=total_signal,
        extra_photons_relative_to_in_field=nexit / nin,
        energy_scale_check=dict(
            field_work_MeV=work, simplified_loss_MeV=simplified_loss,
            photons_using_constant_xsec_Y=simplified_loss * y_const_xsec,
            photons_using_fitted_alpha_Y=simplified_loss * y_measured_alpha,
            independently_validates_transport=False),
        supplied_code_scalar_replication={str(q): legacy_coefficient(q) for q in [1.0, 0.24]},
        earlier_plot_comparison=dict(
            original_baseline_unscattered_fluence_7p4km=plot_base['fluence_J_m2'],
            old_plot_quoted_unscattered_fluence_J_m2=old_plot_fluence,
            fluence_ratio_to_original_baseline=old_plot_fluence / plot_base['fluence_J_m2'],
            same_source_normalization_established=False),
        warnings=[
            'Peaks add only under the same assumed Gaussian envelope and distance.',
            'Post-exit light is not to be added if its deposits are already counted.',
            'Density-only Xu baseline and AIRFLY temperature check use different quenching models.',
            'The mean energy above 1 MeV is 8.3 MeV only for the assumed ideal exponential tail.',
            'No cloud propagation, finite-source geometry, or FEGS response is simulated.'
        ])
    print(json.dumps(output, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
