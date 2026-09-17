"""Lean production-transport and fluid-closure physics contracts."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

import generate_rrea_fluid_closure_bundle as closure_bundle
import rrea_table_pack
from rrea_table_pack import main as pack_main
import verify_fluid_closure_anchors as closure_anchors
from rrea_run_support import SCHEMA6_TRANSPORT_MODEL, require_schema6_config

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = Path(os.environ.get("RREA_BUNDLE") or (
    ROOT / "rrea_transport_tables/schema6/production"))


def test_driver_requires_the_schema6_model_string(tmp_path: Path) -> None:
    """The driver checks identity only; RreaInteractionTables::Load owns semantics."""
    config = tmp_path / "transport_physics.json"
    identity = {"schema_version": 6, "transport_model": SCHEMA6_TRANSPORT_MODEL}
    config.write_text(json.dumps(identity), encoding="utf-8")
    assert require_schema6_config(config) == config.resolve()
    for key, wrong in (("schema_version", 5), ("transport_model", "other_v1")):
        config.write_text(json.dumps({**identity, key: wrong}), encoding="utf-8")
        with pytest.raises(SystemExit, match=key):
            require_schema6_config(config)


def _transport_entries() -> dict[str, dict]:
    config_path = BUNDLE / "transport_physics.json"
    if not config_path.is_file():
        if os.environ.get("RREA_BUNDLE"):
            raise AssertionError(f"RREA_BUNDLE names no bundle: {config_path}")
        pytest.skip(f"no installed bundle at {BUNDLE}")
    found: dict[str, dict] = {}

    def walk(node: object) -> None:
        if isinstance(node, dict):
            if "name" in node and "path" in node:
                found.setdefault(str(node["name"]), node)
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(json.loads(config_path.read_text(encoding="utf-8")))
    return found


def _csv_rows(path: Path) -> list[dict[str, object]]:
    """Rows of a packed table.  Numeric cells arrive as float, not text."""
    columns = rrea_table_pack.load_columns(path)
    names = list(columns)
    return [
        {name: columns[name][row] for name in names}
        for row in range(len(columns[names[0]]))
    ]


CDF_TO_RATE = {
    "brems_photon_energy_cdf_air": "electron_tracked_brems_inverse_length_air",
    "positron_brems_photon_energy_cdf_air": "positron_tracked_brems_inverse_length_air",
    "electron_moller_secondary_cdf_air": "electron_hard_moller_inverse_length_air",
    "positron_hard_bhabha_secondary_cdf_air": "positron_hard_bhabha_inverse_length_air",
}


def test_conditional_cdfs_cover_exact_active_rate_domain() -> None:
    entries = _transport_entries()
    for cdf_name, rate_name in CDF_TO_RATE.items():
        cdf_entry, rate_entry = entries[cdf_name], entries[rate_name]
        rate_rows = _csv_rows(BUNDLE / str(rate_entry["path"]))
        rate_x = next(name for name in ("energy_eV", "kinetic_energy_eV")
                      if name in rate_rows[0])
        rate = [(float(row[rate_x]), float(row[str(rate_entry["value_column"])]))
                for row in rate_rows]
        first_positive = next(i for i, (_, value) in enumerate(rate) if value > 0.0)
        start = rate[max(first_positive - 1, 0)][0]
        cdf_rows = _csv_rows(BUNDLE / str(cdf_entry["path"]))
        x_name = str(cdf_entry.get("x_column") or next(
            name for name in ("primary_energy_eV", "parent_energy_eV")
            if name in cdf_rows[0]))
        energies = sorted({float(row[x_name]) for row in cdf_rows})
        assert energies[0] == start
        assert energies[-1] == 1.0e10


def test_conditional_cdf_axes_are_ascending() -> None:
    entries = _transport_entries()
    for cdf_name in CDF_TO_RATE:
        entry = entries[cdf_name]
        rows = _csv_rows(BUNDLE / str(entry["path"]))
        x_name = str(entry.get("x_column") or next(
            name for name in ("primary_energy_eV", "parent_energy_eV")
            if name in rows[0]))
        seen: list[float] = []
        for row in rows:
            value = float(row[x_name])
            if not seen or value != seen[-1]:
                seen.append(value)
        assert seen == sorted(seen)


def _cdf_by_parent(path: Path) -> dict[float, list[tuple[float, float]]]:
    groups: dict[float, list[tuple[float, float]]] = {}
    for row in _csv_rows(path):
        groups.setdefault(float(row["parent_energy_eV"]), []).append(
            (float(row["cdf_u"]), float(row["photon_energy_fraction"])))
    for rows in groups.values():
        rows.sort()
    return groups


def _mean_fraction(nodes: list[tuple[float, float]], samples: int = 20001) -> float:
    total = 0.0
    j = 0
    for i in range(samples):
        u = (i + 0.5) / samples
        while j + 2 < len(nodes) and nodes[j + 1][0] < u:
            j += 1
        span = nodes[j + 1][0] - nodes[j][0]
        t = 0.0 if span <= 0.0 else (u - nodes[j][0]) / span
        total += nodes[j][1] + t * (nodes[j + 1][1] - nodes[j][1])
    return total / samples


def _active_holdout() -> list[dict]:
    path = BUNDLE / "validation/brems_holdout_metrics.json.rtb"
    if not path.is_file():
        pytest.skip(f"no installed bundle at {BUNDLE}")
    metrics = json.loads(rrea_table_pack.load_bytes(path).decode("utf-8"))
    return [group for group in metrics["groups"]
            if not group.get("inactive_zero_rate")
            and group.get("holdout_mean_photon_fraction") is not None]


def test_brems_inverse_cdf_matches_independent_holdout() -> None:
    cdfs = {
        species: _cdf_by_parent(
            BUNDLE / f"final_state/schema6_brems_quantiles_{species}.csv.rtb")
        for species in ("electron", "positron")
    }
    worst = 0.0
    compared = 0
    for group in _active_holdout():
        nodes = cdfs[str(group["species"])].get(float(group["parent_energy_eV"]))
        if nodes is None:
            continue
        want = float(group["holdout_mean_photon_fraction"])
        worst = max(worst, abs(_mean_fraction(nodes) - want) / want)
        compared += 1
    assert compared >= 80
    assert worst <= 0.02


def test_brems_holdout_active_groups_exist_in_cdf() -> None:
    cdfs = {
        species: set(_cdf_by_parent(
            BUNDLE / f"final_state/schema6_brems_quantiles_{species}.csv.rtb"))
        for species in ("electron", "positron")
    }
    for group in _active_holdout():
        assert float(group["parent_energy_eV"]) in cdfs[str(group["species"])]


def _closure_rows() -> list[dict[str, str]]:
    text = (closure_bundle.BUNDLE_DIR / closure_bundle.CSV_NAME).read_text(
        encoding="ascii")
    return closure_bundle.parsed_rows(text)


def test_fluid_closure_generator_matches_parsed_rows() -> None:
    assert _closure_rows() == closure_bundle.parsed_rows(closure_bundle.csv_text())


def test_fluid_closure_rows_match_independent_anchor_arithmetic() -> None:
    rows = {float(row["en_td"]): row for row in _closure_rows()}
    for en_td, w_u, eta_u in closure_anchors.DE_URQUIJO_DRY_AIR:
        row = rows[en_td]
        k0, _k3, nu3 = closure_anchors.derived_coefficients(en_td, w_u, eta_u)
        assert float(row["en_td"]) == pytest.approx(en_td)
        assert float(row["reduced_flux_mobility_ref_m2_per_v_s"]) == pytest.approx(k0)
        assert float(row["three_body_frequency_ref_s"]) == pytest.approx(nu3)
    for en_td in closure_anchors.high_field_nodes_td():
        row = rows[en_td]
        w = closure_anchors.high_field_drift_m_s(en_td)
        _eta2, _k2, nu2 = closure_anchors.two_body_coefficients_ref(en_td, w)
        assert float(row["two_body_frequency_ref_s"]) == pytest.approx(nu2)


def test_fluid_closure_primary_source_sentinels() -> None:
    """Literal sentinels reviewed directly against the printed source tables."""
    low_drift = dict(closure_anchors.HEGERBERG_REID_DRY_AIR_DRIFT)
    assert {td: low_drift[td] for td in (0.1, 0.3, 0.6, 1.0)} == {
        0.1: 2.55, 0.3: 3.46, 0.6: 4.77, 1.0: 6.35,
    }
    rows = {float(row["en_td"]): row for row in _closure_rows()}
    assert min(rows) == closure_anchors.ZERO_FIELD_PROXY_TD
    assert float(rows[0.1]["flux_drift_velocity_m_per_s"]) == 2550.0
    drift = dict(closure_anchors.DE_URQUIJO_DRY_AIR_DRIFT)
    diffusion = dict(closure_anchors.DE_URQUIJO_DRY_AIR_LONGITUDINAL_DIFFUSION)
    attachment = dict(closure_anchors.DE_URQUIJO_DRY_AIR_ATTACHMENT)
    assert {td: drift[td] for td in (3.0, 10.0, 40.0, 100.0)} == {
        3.0: 10.9, 10.0: 20.1, 40.0: 54.1, 100.0: 114.0,
    }
    assert {td: diffusion[td] for td in (33.0, 40.0, 100.0)} == {
        33.0: 5.79, 40.0: 6.24, 100.0: 18.4,
    }
    assert {td: attachment[td] for td in (3.0, 13.0, 15.0, 17.0, 30.0)} == {
        3.0: 50.30, 13.0: 7.78, 15.0: 6.69, 17.0: 5.64, 30.0: 3.54,
    }
    assert all(td not in attachment for td in (3.3, 3.6, 4.5, 5.5, 6.5))


def test_fluid_closure_rows_are_positive_and_ordered() -> None:
    rows = _closure_rows()
    en = [float(row["en_td"]) for row in rows]
    assert en == sorted(set(en))
    assert all(float(row["reduced_flux_mobility_ref_m2_per_v_s"]) > 0 for row in rows)
    assert all(float(row["three_body_frequency_ref_s"]) > 0 for row in rows)
    assert float(rows[0]["two_body_frequency_ref_s"]) == 0.0
    assert float(rows[-1]["two_body_frequency_ref_s"]) > 0.0


def test_table_container_round_trips_every_kind() -> None:
    """RREATBL restores what it packed, and refuses what it cannot decode."""
    import struct

    import rrea_table_pack as pack

    csv_bytes = b"e_eV,shell,p\n1000,K,0.5\n50000.000000000015,L1,0.5\n"
    blob = pack.pack_bytes("t.csv", csv_bytes)
    header, payload = pack.read_container(blob)
    assert header["kind"] == "columns" and header["rows"] == "2"
    assert [c[1] for c in pack.header_columns(header)] == ["f8", "str", "f8"]
    # Value-exact, including the sub-ulp anchors the loader needs distinct.
    assert pack._columns_from(header, payload)["e_eV"] == [1000.0, 50000.000000000015]
    assert pack._same_values(csv_bytes, pack.unpack_bytes(blob))

    record = bytearray(pack.PAIR_RECORD_BYTES)
    for name, off, dtype in pack.PAIR_FIELDS:
        struct.pack_into(pack._STRUCT[dtype], record, off, 7 if dtype != "f8" else 0.25)
    raw = b"RREAPAIR" + bytes(56) + bytes(record) * 3
    assert pack.unpack_bytes(pack.pack_bytes("p.bin", raw)) == raw

    opaque = b"\x00\x01\x02not text"
    assert pack.unpack_bytes(pack.pack_bytes("x.json", opaque)) == opaque

    for broken in (b"NOTRREA" + blob[7:], blob[:8] + struct.pack("<I", 99) + blob[12:], blob[:-1]):
        with pytest.raises(ValueError):
            pack.read_container(broken)


def test_table_verify_refuses_to_check_nothing(tmp_path: Path) -> None:
    """A verify that examines zero files reads exactly like a pass."""
    (tmp_path / "empty").mkdir()
    assert pack_main(["verify", str(tmp_path / "empty")]) == 1
