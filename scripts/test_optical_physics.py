"""Independent, focused checks for the optical-source coupling."""

from __future__ import annotations

from pathlib import Path
import csv

import numpy as np
import pytest

import rrea_cloud_scattering as cloud
from compute_rrea_optical_emissions import (
    E_PHOTON_337_J,
    E_PHOTON_777_J,
    fy337_ph_per_mev,
    native_pair_source_series,
    photon_probability_337,
    quench_factor_777,
    A_777_PER_S,
    KQ_N2_777_M3_S,
    KQ_O2_777_M3_S,
    source_yield_bounds,
    swarm_emission_envelope,
    C0_BRANCH_337,
    X_N2,
    X_O2,
    W_AIR_EV,
)


def test_photon_energies_are_hc_over_lambda() -> None:
    h, c = 6.62607015e-34, 299792458.0
    assert E_PHOTON_337_J == pytest.approx(h * c / 337.1e-9, rel=1e-3)
    assert E_PHOTON_777_J == pytest.approx(h * c / 777.4e-9, rel=1e-3)


def test_fy337_matches_the_airfly_absolute_yield() -> None:
    # AIRFLY final absolute 337 nm yield in dry air: 5.61 +/- 0.06 (stat)
    # +/- 0.21 (syst) photons/MeV at 1013 hPa, 293 K (Ave et al., Astropart.
    # Phys. 42 (2013) 90) -- independent of the 800 hPa anchor the code uses.
    assert fy337_ph_per_mev(np.array([1013.25]), np.array([293.0]))[0] == (
        pytest.approx(5.61, abs=0.22))
    # More pressure quenches harder, monotonically.
    yields = fy337_ph_per_mev(
        np.array([100.0, 400.0, 1013.25]), np.full(3, 260.0))
    assert yields[0] > yields[1] > yields[2] > 0.0


def test_c0_branching_and_quenching_limits():
    assert photon_probability_337(0., 293.) == pytest.approx(C0_BRANCH_337)
    n = np.array([1e24, 1e25, 1e29, 1e30])
    b = photon_probability_337(n, 293.)
    assert np.all(np.diff(b) < 0) and np.all((b > 0) & (b < 1))
    assert n[-1]*b[-1] == pytest.approx(n[-2]*b[-2], rel=1e-5)


def test_oi777_quenching_limits():
    assert quench_factor_777(0.) == 1.
    n = np.array([1e24, 1e25, 1e30])
    q = quench_factor_777(n)
    assert np.all(np.diff(q) < 0) and np.all((q > 0) & (q < 1))
    high_density_limit = A_777_PER_S / (X_N2*KQ_N2_777_M3_S + X_O2*KQ_O2_777_M3_S)
    assert n[-1]*q[-1] == pytest.approx(high_density_limit, rel=1e-6)


def test_phelps_source_retains_threshold_units_and_positive_elastic_remainder():
    from generate_rrea_fluid_closure_bundle import phelps_collisions, excitation_rates
    records = phelps_collisions()
    for kind, header, data in records:
        assert np.all(np.diff(data[:, 0]) > 0) and np.all(data[:, 1] >= 0)
        if kind == "ELASTIC":
            assert np.all(data[:, 1] > 0)
        if header.splitlines()[0] == "N2 -> N2(C3)":
            # Original Phelps C-state knots: 11.03-eV onset, area in m^2.
            np.testing.assert_allclose(data[0], [11.03, 0])
            assert np.interp(14., *data.T) == pytest.approx(6.5e-21, abs=1e-25)
    with pytest.raises(ValueError, match="machine precision"):
        excitation_rates("unused", fields=[20], temperature=230, density=8e24,
                         argon=.0093, c_scale=1, points=800, precision=1e-18)


def test_source_centroid_bounds_heterogeneous_fluorescence():
    z = np.linspace(0, 10000, 101)
    n, temp = 2e25*np.exp(-z/7000), 280-z*.005
    rng = np.random.default_rng(43)
    for _ in range(30):
        positions, weights = rng.uniform(0, 10000, 20), rng.uniform(0, 1, 20)
        nn, tt = np.interp(positions, z, n), np.interp(positions, z, temp)
        true_yield = np.average(fy337_ph_per_mev(nn*1.380649e-23*tt/100, tt), weights=weights)
        lo, hi = source_yield_bounds(z, n, temp, np.average(positions, weights=weights))
        assert lo <= true_yield <= hi
    # A concentrated source approaches its local yield as atmosphere bins shrink.
    lo, hi = source_yield_bounds(z, n, temp, z[0])
    assert hi/lo < 1.03


def test_bolsig_rate_uses_native_process_and_active_domain(monkeypatch):
    import generate_rrea_fluid_closure_bundle as generator
    from scipy.integrate import quad
    from scipy.constants import elementary_charge, electron_mass

    # The exported EEDF ends below excitation threshold; neither its endpoint
    # nor its apparent zero excitation may replace the native solver results.
    def solve(command, *, cwd, **kwargs):
        (cwd / "out.dat").write_text(
            "R1\nMaximum energy 200\nMaximum energy 20\n"
            "# of iterations 8\nMaximum # of iterations 2000\n"
            "Mobility *N (1/m/V/s) 1.5e24\nMean energy (eV) 1.1\n"
            "C21 N2 Excitation 11.03 eV 2e-23\n"
            "Energy (eV) EEDF (eV-3/2) Anisotropy\n0 1 0\n10 1e-15 .5\n")
    monkeypatch.setattr(generator.subprocess, "run", solve)
    row, = generator.excitation_rates(
        "bolsig", fields=[28.], temperature=230., density=8e24,
        argon=.0093, c_scale=1., points=800, precision=1e-15)
    assert row["k_c_m3_s"] == 2e-23 and row["active_max_eV"] == 20.
    assert row["flux_drift_m_s"] == pytest.approx(42000.)
    data, = [a for _, h, a in generator.phelps_collisions()
             if h.splitlines()[0] == "N2 -> N2(C3)"]
    integral = quad(lambda e: e*np.interp(e, *data.T), 20., data[-1, 0],
                    points=data[data[:, 0] > 20., 0], epsabs=1e-30)[0]
    expected = 1e-15*np.sqrt(2*elementary_charge/electron_mass)*integral
    assert row["tail_envelope_m3_s"] == pytest.approx(expected, rel=1e-12, abs=0.)


def test_swarm_envelope_preserves_heterogeneous_quenching_and_field_peak(tmp_path):
    table = tmp_path / "rates.csv"
    with table.open("w") as f:
        writer = csv.writer(f)
        writer.writerow(["en_td", "temperature_K", "density_m3", "argon_fraction",
                         "k_c_m3_s", "tail_envelope_m3_s"])
        for en, k in [(1., 0.), (20., 1e-22), (30., 1e-24)]:
            for n, temp in [(5e24, 200.), (1e25, 300.)]:
                writer.writerow([en, temp, n, 0., k, 1e-29])
    n, temp = np.array([5e24, 1e25]), np.array([200., 300.])
    t, ne, pairs = np.array([0., 1., 2.]), np.full(3, 10.), np.array([0., 100., 100.])
    result = swarm_emission_envelope(t, ne, np.full(3, 30.), pairs, (n, temp), table)
    # The strongest rate is at an INTERIOR field; different cells quench
    # differently. Their true local sum cannot exceed the population envelope.
    local_swarm = np.sum(np.array([9., 1.]) * n * X_N2/(X_N2+X_O2)
                         * np.array([1e-22, 1e-24]) * photon_probability_337(n, temp))
    assert np.all(result["upper_ph_s"] >= local_swarm)
    local_dep = 100 * W_AIR_EV * 1e-6 * fy337_ph_per_mev(
        n[0]*1.380649e-23*temp[0]/100, temp[0])
    assert result["fraction"] >= 2*local_swarm/local_dep
    assert np.isnan(result["interval_fraction"][-1])  # afterglow: no false ratio
    assert result["upper_ph_s"][-1] > 0
    assert result["interval_photons"][-1] == pytest.approx(
        result["upper_ph_s"][-1] * (t[-1]-t[-2]))
    # A solver-truncated zero remains an explicitly nonzero numerical envelope.
    low = swarm_emission_envelope(t, ne, np.zeros(3), pairs, (n, temp), table)
    assert np.all(low["upper_ph_s"] > 0)
    for fields in (np.full(3, np.nan), np.full(3, 31.)):
        unavailable = swarm_emission_envelope(t, ne, fields, pairs, (n, temp), table)
        assert "unassessed" in unavailable["note"]
        assert np.all(np.isnan(unavailable["upper_ph_s"]))


def test_zero_native_pair_source_is_rejected() -> None:
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


def test_native_source_smoothing_preserves_pair_count():
    t = np.array([0., .1, .5, 1., 2., 3.])
    count = np.array([0., 2., 3., 8., 8., 10.])
    for smoothing in (0., 1., 4.):
        rate, z = native_pair_source_series(t, count, 250*count, smoothing)
        assert np.trapezoid(rate, t) == pytest.approx(10.)
        np.testing.assert_allclose(z, 250.)


def test_cloudscat_boundary_preserves_timeline_geometry_and_normalization(
    monkeypatch,
) -> None:
    c = 299792458.0
    emission_t = np.array([0.0, 10.0e-6])
    altitude_m = np.array([10_000.0, 12_000.0])
    observer_m = 20_000.0
    expected_arrival = emission_t + (observer_m - altitude_m) / c

    def record_config(config):
        points = config["source"]["points"]
        assert [p["time"] for p in points] == list(emission_t)
        assert [p["position"][2] for p in points] == list(altitude_m)
        assert config["params"]["sigma_ray"] == 0
        return Path("unused")

    monkeypatch.setattr(cloud, "run_cached", record_config)
    monkeypatch.setattr(
        cloud,
        "read_timeline",
        lambda out_dir: (expected_arrival.copy(), np.array([3.0, 4.0])),
    )
    returned_t, flux = cloud.scattered_flux(
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
    # Trapezoidal emitted photons = (1e12 + 2e12) * 10 us / 2.
    np.testing.assert_allclose(flux, np.array([3.0, 4.0]) * 1.5e7)
