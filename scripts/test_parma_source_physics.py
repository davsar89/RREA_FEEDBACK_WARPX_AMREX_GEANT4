"""Committed pin for the PARMA continuous-column e-/e+ source module.

Synthetic tables only -- no Fortran, no upstream clone.  The governing
conversion q_s = phi_s/H (Ndot_s = A_cm2 * <int phi_s dE>_z, including the
cm <-> m conversion) is verified analytically per species; macro allocation
between the species is proportional to physical rate; every sampling channel
is checked against its own constructed CDF as an exact stratified quantile
set per species; grids are compared with doubled resolution; identical local
inputs generate deterministic output.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from rrea_parma_source import (
    ParmaSourceTable,
    PreparedParmaSource,
    SPECIES_SEED_ID,
    SPEED_OF_LIGHT_CM_PER_S,
    _build_grids,
    ensure_parma_source_table,
    parma_source_audit,
    relativistic_beta,
    write_parma_seed_schedule,
    write_parma_source_table,
)
from rrea_profiled_atmosphere import ALTITUDE_AT_Z0_M, DOMAIN_HEIGHT_M
from rrea_run_support import load_rrea_defaults, DEFAULTS_PATH

MIN_EV = 1.0e5
MAX_EV = 1.0e10


def synthetic_table(
    *,
    electron_fn=lambda energy_eV, altitude_km: 1.0,
    positron_fn=lambda energy_eV, altitude_km: 0.5,
    ang_fn=lambda mu, energy_eV, altitude_km: 1.0,
    refine: int = 1,
) -> ParmaSourceTable:
    altitude_km, energy_eV, angular_energy_eV, mu = _build_grids(
        MIN_EV, MAX_EV, refine
    )
    return ParmaSourceTable(
        latitude_deg=28.7,
        longitude_deg=-80.8,
        minimum_energy_eV=MIN_EV,
        maximum_energy_eV=MAX_EV,
        solar_w_index=50.0,
        solar_status=1,
        rigidity_gv=2.0,
        altitude_km=altitude_km,
        depth_g_cm2=[100.0 for _ in altitude_km],
        energy_eV=energy_eV,
        electron_spec_cm2_s_mev=[
            [electron_fn(e, a) for e in energy_eV] for a in altitude_km
        ],
        positron_spec_cm2_s_mev=[
            [positron_fn(e, a) for e in energy_eV] for a in altitude_km
        ],
        angular_energy_eV=angular_energy_eV,
        mu=mu,
        angular=[
            [[ang_fn(m, e, a) for m in mu] for e in angular_energy_eV]
            for a in altitude_km
        ],
    )


def read_schedule(path: Path) -> list[dict]:
    import csv

    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_constant_spectra_pin_column_normalization_per_species(tmp_path):
    # phi_s(E,z) = C_s per MeV everywhere: <int phi_s dE>_z = C_s * span_MeV
    # exactly under trapezoid, so Ndot_s = pi*(100*R_m)^2 * C_s * span --
    # the cm <-> m conversion pin, per species; total is their sum.
    c_electron, c_positron = 3.0e-4, 1.0e-4
    table = synthetic_table(
        electron_fn=lambda e, a: c_electron,
        positron_fn=lambda e, a: c_positron,
    )
    prepared = PreparedParmaSource(table)
    span_mev = (MAX_EV - MIN_EV) / 1.0e6
    radius_m = 2000.0
    audit = parma_source_audit(prepared, source_radius_m=radius_m)
    area_cm2 = math.pi * (radius_m * 100.0) ** 2
    expected_e = area_cm2 * c_electron * span_mev
    expected_p = area_cm2 * c_positron * span_mev
    assert math.isclose(
        audit["physical_electrons_per_s"], expected_e, rel_tol=1e-12
    )
    assert math.isclose(
        audit["physical_positrons_per_s"], expected_p, rel_tol=1e-12
    )
    assert math.isclose(
        audit["physical_particles_per_s_total"],
        expected_e + expected_p,
        rel_tol=1e-12,
    )
    half = parma_source_audit(prepared, source_radius_m=radius_m / 2.0)
    assert math.isclose(
        half["physical_particles_per_s_total"],
        (expected_e + expected_p) / 4.0,
        rel_tol=1e-12,
    )
    # Proportional macro allocation and exact per-species weights.
    n = 400
    stop_time_s = 2.0e-4
    total, audit = write_parma_seed_schedule(
        tmp_path / "s.csv",
        table=table,
        case_id="pin",
        macro_count=n,
        realization_id=0,
        stop_time_s=stop_time_s,
        source_radius_m=radius_m,
    )
    expected_stream_total = (expected_e + expected_p) * stop_time_s
    assert math.isclose(
        total,
        expected_stream_total + audit["bath_injected_real_particles"],
        rel_tol=1e-12,
    )
    counts = audit["species_macro_counts"]
    assert counts["electron"] + counts["positron"] == n
    assert counts["electron"] == round(
        n * expected_e / (expected_e + expected_p)
    )
    weights = audit["species_macro_weights"]
    assert math.isclose(
        weights["electron"],
        expected_e * stop_time_s / counts["electron"],
        rel_tol=1e-12,
    )
    assert math.isclose(
        weights["positron"],
        expected_p * stop_time_s / counts["positron"],
        rel_tol=1e-12,
    )
    # Bath pin: n = phi/(beta c) integrated over the same grids, computed
    # here with an independently coded trapezoid.
    _, energy_eV, _, _ = _build_grids(MIN_EV, MAX_EV)
    energy_mev = [e / 1.0e6 for e in energy_eV]
    for name, c_species in (
        ("electron", c_electron), ("positron", c_positron)
    ):
        density_integral = 0.0
        for i in range(1, len(energy_mev)):
            lo = c_species / (
                relativistic_beta(energy_eV[i - 1]) * SPEED_OF_LIGHT_CM_PER_S
            )
            hi = c_species / (
                relativistic_beta(energy_eV[i]) * SPEED_OF_LIGHT_CM_PER_S
            )
            density_integral += 0.5 * (lo + hi) * (
                energy_mev[i] - energy_mev[i - 1]
            )
        expected_bath = area_cm2 * density_integral * 7.0e5  # 7 km column
        assert math.isclose(
            audit["species"][name]["physical_bath_particles"],
            expected_bath,
            rel_tol=1e-12,
        ), name
        assert audit["bath_macro_counts"][name] == round(
            expected_bath / audit["species_macro_weights"][name]
        )
    # Per-species injected weight sums reproduce each species' physics
    # (stream) plus the macro-quantized bath.
    rows = read_schedule(tmp_path / "s.csv")
    for name, expected_rate in (
        ("electron", expected_e), ("positron", expected_p)
    ):
        stream_rows = [
            r for r in rows
            if int(r["species_id"]) == SPECIES_SEED_ID[name]
            and r["source_region"] == "parma_column_cylinder"
        ]
        bath_rows = [
            r for r in rows
            if int(r["species_id"]) == SPECIES_SEED_ID[name]
            and r["source_region"] == "parma_column_bath"
        ]
        assert len(stream_rows) == counts[name]
        assert len(bath_rows) == audit["bath_macro_counts"][name]
        assert all(float(r["time_s"]) == 0.0 for r in bath_rows)
        injected = sum(float(r["weight_real_electrons"]) for r in stream_rows)
        assert math.isclose(
            injected, expected_rate * stop_time_s, rel_tol=1e-9
        )


def test_every_channel_is_an_exact_stratified_quantile_set(tmp_path):
    # z-constant table so conditional CDFs equal their node CDFs exactly;
    # species spectra differ so the two Latin hypercubes are distinct.
    table = synthetic_table(
        electron_fn=lambda e, a: 1.0e6 / e,
        positron_fn=lambda e, a: 0.5e6 / e * (e / 1.0e6) ** -0.2,
        ang_fn=lambda m, e, a: 1.0 + 0.5 * m,
    )
    n = 512
    stop_time_s = 2.0e-4
    radius_m = 2000.0
    path = tmp_path / "schedule.csv"
    _, audit = write_parma_seed_schedule(
        path,
        table=table,
        case_id="quantiles",
        macro_count=n,
        realization_id=3,
        stop_time_s=stop_time_s,
        source_radius_m=radius_m,
    )
    rows = read_schedule(path)
    counts = audit["species_macro_counts"]
    bath_counts = audit["bath_macro_counts"]
    assert len(rows) == n + sum(bath_counts.values())
    prepared = PreparedParmaSource(table)
    for name in ("electron", "positron"):
        for region, block, altitude_cdf, energy_cdf in (
            (
                "parma_column_cylinder", counts[name],
                prepared.altitude_cdf, prepared.energy_cdf_at_node,
            ),
            (
                "parma_column_bath", bath_counts[name],
                prepared.bath_altitude_cdf, prepared.bath_energy_cdf_at_node,
            ),
        ):
            block_rows = [
                r for r in rows
                if int(r["species_id"]) == SPECIES_SEED_ID[name]
                and r["source_region"] == region
            ]
            assert len(block_rows) == block
            expected = [(i + 0.5) / block for i in range(block)]

            def assert_quantiles(values, label):
                assert len(values) == block, label
                for got, want in zip(sorted(values), expected):
                    assert math.isclose(
                        got, want, rel_tol=0.0, abs_tol=1e-9
                    ), f"{name} {region} {label}"

            if region == "parma_column_bath":
                assert all(float(r["time_s"]) == 0.0 for r in block_rows)
            else:
                times = [float(r["time_s"]) / stop_time_s for r in block_rows]
                assert times == sorted(times)
                for got, want in zip(times, expected):
                    assert math.isclose(got, want, rel_tol=1e-12)
            assert_quantiles(
                [
                    altitude_cdf(
                        name, (float(r["z_m"]) + ALTITUDE_AT_Z0_M) / 1000.0
                    )
                    for r in block_rows
                ],
                "altitude",
            )
            assert_quantiles(
                [
                    energy_cdf(name, 0, float(r["kinetic_energy_eV"]))
                    for r in block_rows
                ],
                "energy",
            )
            assert_quantiles(
                [
                    prepared.mu_cdf_at_node(0, 0, -float(r["uz"]))
                    for r in block_rows
                ],
                "mu",
            )
            assert_quantiles(
                [(float(r["r_m"]) / radius_m) ** 2 for r in block_rows],
                "radius",
            )
            assert_quantiles(
                [float(r["phi_rad"]) / (2.0 * math.pi) for r in block_rows],
                "position azimuth",
            )
            assert_quantiles(
                [
                    (math.atan2(float(r["uy"]), float(r["ux"]))
                     / (2.0 * math.pi)) % 1.0
                    for r in block_rows
                ],
                "direction azimuth",
            )
    # A different realization permutes differently but keeps the marginals.
    other = tmp_path / "other.csv"
    write_parma_seed_schedule(
        other,
        table=table,
        case_id="quantiles",
        macro_count=n,
        realization_id=4,
        stop_time_s=stop_time_s,
        source_radius_m=radius_m,
    )
    energies = [float(r["kinetic_energy_eV"]) for r in rows]
    other_energies = [
        float(r["kinetic_energy_eV"]) for r in read_schedule(other)
    ]
    assert energies != other_energies
    assert sorted(energies) == pytest.approx(sorted(other_energies))


def test_extensions_keep_the_original_bath_weights_and_events(tmp_path):
    table = synthetic_table(electron_fn=lambda energy, altitude: 3e-4,
                            positron_fn=lambda energy, altitude: 1e-4)
    interval, radius, count = 2e-4, 2000.0, 40
    arguments = dict(table=table, case_id="extension", macro_count=count,
                     realization_id=7, source_radius_m=radius)
    original = tmp_path / "original.csv"
    write_parma_seed_schedule(original, stop_time_s=interval, **arguments)
    previous = read_schedule(original)
    bath = [row for row in previous if row["source_region"] == "parma_column_bath"]
    expected_weight = (math.pi * (100 * radius)**2 * 4e-4
                       * (MAX_EV - MIN_EV) / 1e6 * interval / count)
    for duration in (1.37 * interval, 2 * interval, 5 * interval):
        target = tmp_path / "extended.csv"
        total, audit = write_parma_seed_schedule(
            target, stop_time_s=duration, sampling_interval_s=interval, **arguments)
        rows = read_schedule(target)
        assert rows[:len(previous)] == previous
        assert [row for row in rows if row["source_region"] == "parma_column_bath"] == bath
        assert len({row["macro_index"] for row in rows}) == len(rows)
        assert all(float(row["time_s"]) < duration for row in rows)
        stream = [row for row in rows if row["source_region"] == "parma_column_cylinder"]
        assert abs(len(stream) - count * duration / interval) <= 1
        assert all(math.isclose(float(row["weight_real_electrons"]), expected_weight,
                                rel_tol=1e-12) for row in rows)
        assert math.isclose(total, expected_weight * len(rows), rel_tol=1e-12)
        assert audit["sampling_interval_s"] == interval
        previous = rows
    electrons = [row for row in stream if row["species_id"] == "1"]
    first_energies = [row["kinetic_energy_eV"] for row in electrons
                      if float(row["time_s"]) < interval]
    second_energies = [row["kinetic_energy_eV"] for row in electrons
                       if interval <= float(row["time_s"]) < 2 * interval]
    assert first_energies != second_energies


def test_domain_bounds_directions_and_species_columns(tmp_path):
    table = synthetic_table(
        electron_fn=lambda e, a: (1.0e6 / e) * (1.0 + 0.1 * (a - 9.0)),
        positron_fn=lambda e, a: (0.4e6 / e) * (1.0 + 0.05 * (a - 9.0)),
        ang_fn=lambda m, e, a: 1.0 + 0.5 * m,
    )
    n = 256
    stop_time_s = 2.0e-4
    radius_m = 2000.0
    path = tmp_path / "schedule.csv"
    total, audit = write_parma_seed_schedule(
        path,
        table=table,
        case_id="bounds",
        macro_count=n,
        realization_id=0,
        stop_time_s=stop_time_s,
        source_radius_m=radius_m,
    )
    rows = read_schedule(path)
    weights = audit["species_macro_weights"]
    seen_indices = set()
    for index, r in enumerate(rows):
        assert int(r["macro_index"]) == index
        assert int(r["macro_index"]) not in seen_indices
        seen_indices.add(int(r["macro_index"]))
        species_id = int(r["species_id"])
        assert species_id in (1, 2)
        assert r["generation_id"] == "0"
        assert MIN_EV <= float(r["kinetic_energy_eV"]) <= MAX_EV
        assert 0.0 <= float(r["time_s"]) < stop_time_s
        x, y = float(r["x_m"]), float(r["y_m"])
        assert math.isclose(float(r["r_m"]), math.hypot(x, y), rel_tol=1e-15)
        assert float(r["r_m"]) < radius_m
        assert 0.0 <= float(r["z_m"]) < DOMAIN_HEIGHT_M
        ux, uy, uz = float(r["ux"]), float(r["uy"]), float(r["uz"])
        assert math.isclose(
            math.sqrt(ux * ux + uy * uy + uz * uz), 1.0, rel_tol=1e-12
        )
        assert -1.0 <= uz <= 1.0
        name = "electron" if species_id == 1 else "positron"
        assert math.isclose(
            float(r["weight_real_electrons"]), weights[name], rel_tol=1e-15
        )
    assert math.isclose(
        sum(float(r["weight_real_electrons"]) for r in rows),
        total,
        rel_tol=1e-9,
    )


def test_grid_refinement_converges_sub_percent():
    def electron(e, a):
        return (e / 1.0e6) ** -1.8 * (1.0 + 0.08 * (a - 9.0))

    def positron(e, a):
        return 0.5 * (e / 1.0e6) ** -1.9 * (1.0 + 0.06 * (a - 9.0))

    def ang(m, e, a):
        return math.exp(1.2 * m)

    base = PreparedParmaSource(
        synthetic_table(electron_fn=electron, positron_fn=positron, ang_fn=ang)
    )
    fine = PreparedParmaSource(
        synthetic_table(
            electron_fn=electron, positron_fn=positron, ang_fn=ang, refine=2
        )
    )
    for name in ("electron", "positron"):
        for coarse_value, fine_value, label in (
            (
                base.mean_integrated_flux[name],
                fine.mean_integrated_flux[name],
                "column mean flux",
            ),
            (
                base.mean_energy_eV_per_altitude(name)[0],
                fine.mean_energy_eV_per_altitude(name)[0],
                "mean energy",
            ),
            (
                base.mean_mu_per_altitude(name)[0],
                fine.mean_mu_per_altitude(name)[0],
                "mean mu",
            ),
        ):
            assert abs(coarse_value - fine_value) <= 0.01 * abs(fine_value), (
                f"{name} {label}"
            )


def test_generation_is_deterministic_and_echo_fails_closed(tmp_path):
    table = synthetic_table(
        electron_fn=lambda e, a: 1.0e6 / e,
        positron_fn=lambda e, a: 0.3e6 / e,
    )
    stored = tmp_path / "parma_source_table.txt"
    write_parma_source_table(table, stored)
    loaded = ensure_parma_source_table(
        stored,
        latitude_deg=28.7,
        longitude_deg=-80.8,
        minimum_energy_eV=MIN_EV,
        maximum_energy_eV=MAX_EV,
    )
    assert loaded.energy_eV == table.energy_eV
    assert loaded.electron_spec_cm2_s_mev == table.electron_spec_cm2_s_mev
    assert loaded.positron_spec_cm2_s_mev == table.positron_spec_cm2_s_mev
    assert loaded.angular == table.angular
    first, second = tmp_path / "a.csv", tmp_path / "b.csv"
    for path in (first, second):
        write_parma_seed_schedule(
            path,
            table=loaded,
            case_id="determinism",
            macro_count=64,
            realization_id=7,
            stop_time_s=2.0e-4,
            source_radius_m=2000.0,
        )
    assert first.read_bytes() == second.read_bytes()
    with pytest.raises(SystemExit, match="latitude_deg"):
        ensure_parma_source_table(
            stored,
            latitude_deg=60.0,
            longitude_deg=-80.8,
            minimum_energy_eV=MIN_EV,
            maximum_energy_eV=MAX_EV,
        )


def test_defaults_validation_rejects_bad_parma_blocks(tmp_path):
    good = json.loads(DEFAULTS_PATH.read_text(encoding="utf-8"))
    # The PARMA continuous e-/e+ column with its t=0 bath is the production
    # default source; the pulse stays selectable.
    assert good["seed_source_model"] == "parma_continuous_column"
    parma = load_rrea_defaults(DEFAULTS_PATH)["parma_continuous_source"]
    assert parma["macro_count"] == 100000
    assert parma["source_radius_m"] == 2000.0

    def rejected(mutate, match):
        payload = json.loads(DEFAULTS_PATH.read_text(encoding="utf-8"))
        mutate(payload)
        candidate = tmp_path / "defaults.json"
        candidate.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(SystemExit, match=match):
            load_rrea_defaults(candidate)

    rejected(
        lambda p: p.update(seed_source_model="coleman_dwyer_7p2mev"),
        "seed_source_model",
    )
    rejected(
        lambda p: p["parma_continuous_source"].update(latitude_deg=95.0),
        "latitude_deg",
    )
    rejected(
        lambda p: p["parma_continuous_source"].update(minimum_energy_eV=1.0e4),
        "minimum_energy_eV",
    )
    rejected(
        lambda p: p["parma_continuous_source"].update(source_radius_m=-1.0),
        "source_radius_m",
    )
    rejected(
        lambda p: p["parma_continuous_source"].update(macro_count=0),
        "macro_count",
    )
