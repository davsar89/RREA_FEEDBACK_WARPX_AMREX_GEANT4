"""Independent, focused checks for the optical-source coupling."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import rrea_cloud_scattering as cloud
from compute_rrea_optical_emissions import (
    E_PHOTON_337_J,
    E_PHOTON_777_J,
    enhancement_337,
    fy337_ph_per_mev,
    native_pair_source_series,
)


def test_photon_energies_are_hc_over_lambda() -> None:
    h, c = 6.62607015e-34, 299792458.0
    assert E_PHOTON_337_J == pytest.approx(h * c / 337.1e-9, rel=1e-3)
    assert E_PHOTON_777_J == pytest.approx(h * c / 777.4e-9, rel=1e-3)


def test_fy337_quench_follows_the_airfly_t086_law() -> None:
    # FY = Y0 / (1 + P/P'(T)) with P'(T) ~ T^0.86 (AIRFLY alpha = -0.36).
    # Recover P'(T) from two pressures at fixed T (1/FY is linear in P),
    # then pin the exponent from two temperatures -- a shape test using only
    # the public function, no transcribed internals.
    def p_prime(temp_k: float) -> float:
        p1, p2 = 200.0, 800.0
        inv1 = 1.0 / fy337_ph_per_mev(np.array([p1]), np.array([temp_k]))[0]
        inv2 = 1.0 / fy337_ph_per_mev(np.array([p2]), np.array([temp_k]))[0]
        slope = (inv2 - inv1) / (p2 - p1)
        intercept = inv1 - slope * p1
        return intercept / slope

    t1, t2 = 220.0, 290.0
    assert p_prime(t2) / p_prime(t1) == pytest.approx(
        (t2 / t1) ** 0.86, rel=1e-9)
    # More pressure quenches harder, monotonically.
    yields = fy337_ph_per_mev(
        np.array([100.0, 400.0, 1013.25]), np.full(3, 260.0))
    assert yields[0] > yields[1] > yields[2] > 0.0


def test_xu_enhancement_is_not_extrapolated_below_16_td() -> None:
    values = enhancement_337(np.array([0.0, 10.0, 15.999, 16.0, 46.5]))
    np.testing.assert_array_equal(values[:3], np.ones(3))
    assert values[3] > 1.0
    assert values[4] > values[3]


def test_conservative_ion_translation_is_not_an_optical_source() -> None:
    # A translated ion packet has zero signed inventory change; reconstructing
    # a source as clip(delta n_+, 0) would invent light.
    before = np.array([2.0, 1.0, 0.0, 0.0])
    after = np.array([0.0, 2.0, 1.0, 0.0])
    delta = after - before
    assert np.sum(delta) == 0.0
    assert np.sum(np.clip(delta, 0.0, None)) > 0.0

    # The native event source correctly remains zero and smoothing cannot
    # turn transport of existing ions into pair creation.
    native_cumulative = np.zeros(5)
    with pytest.raises(SystemExit, match="no positive native ion-pair source"):
        native_pair_source_series(
            np.arange(5.0), native_cumulative, native_cumulative, 3.0
        )


def test_native_source_moments_recover_rate_and_altitude() -> None:
    t = np.array([0.0, 1.0, 2.0, 3.0])
    cumulative_weight = np.array([0.0, 2.0, 2.0, 6.0])
    cumulative_z_weighted = np.array([0.0, 200.0, 200.0, 1400.0])
    rate, mean_z = native_pair_source_series(
        t, cumulative_weight, cumulative_z_weighted, 0.0
    )
    np.testing.assert_allclose(
        rate,
        [2.0, 1.0, 2.0, 4.0],
    )
    np.testing.assert_allclose(mean_z, [100.0, 150.0, 250.0, 300.0])


def test_varying_altitude_clear_air_timing_stays_on_absolute_clock(
    monkeypatch,
) -> None:
    c = 299792458.0
    emission_t = np.array([0.0, 10.0e-6])
    altitude_m = np.array([10_000.0, 12_000.0])
    observer_m = 20_000.0
    expected_arrival = emission_t + (observer_m - altitude_m) / c

    monkeypatch.setattr(cloud, "run_cached", lambda config: Path("unused"))
    monkeypatch.setattr(
        cloud,
        "read_timeline",
        lambda out_dir: (expected_arrival.copy(), np.array([3.0, 4.0])),
    )
    returned_t, flux, captured = cloud.scattered_flux(
        emission_t,
        np.array([1.0e12, 2.0e12]),
        altitude_m,
        band="337",
        observer_alt_m=observer_m,
        cloud={"droplet_n_cm3": 0.0},
        rayleigh=False,
    )

    np.testing.assert_array_equal(returned_t, expected_arrival)
    assert returned_t[1] - emission_t[1] != returned_t[0] - emission_t[0]
    assert np.all(flux > 0.0)
    assert captured == 1.0
