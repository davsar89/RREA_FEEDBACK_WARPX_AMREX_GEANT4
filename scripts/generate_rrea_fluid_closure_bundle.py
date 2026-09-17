#!/usr/bin/env python3
"""Generate the dry-air zero--100 Td electron closure table.

The low-field part combines measured drift/diffusion anchors with cited
attachment fits and a thermal limit; transport through 100 Td is de Urquijo.
Runtime validates
the unit-bearing table semantically, never through a digest.

Scope and policy, deliberately explicit:

- Dry air, room temperature (293-300 K).  Not a storm-atmosphere closure.
- Valid range includes the thermal limit through 100 Td.
- Three-body attachment is measured through 30 Td; the Morrow--Lowke A4/A5
  transition supplies separate density-linear and density-squared channels.
- The first positive node represents the measured zero-field asymptote; smaller
  realized fields hold that limit rather than extrapolating through log(0).
- ABOVE 100 Td evaluation is forbidden: the engine must abort, not clamp.
- Blank dry-air Table-5 cells are filled by log-linear interpolation between
  adjacent printed dry-air nodes.  Values in humid columns are never used.
- The uncertainty columns are broad source-wide ranges (~1.5% W_F,
  ~10% dry-air eta/N^2), not a point-by-point uncertainty model.

Usage:
    python scripts/generate_rrea_fluid_closure_bundle.py [--check]

--check compares parsed rows and metadata, never file bytes.
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from verify_fluid_closure_anchors import (  # noqa: E402
    DE_URQUIJO_DRY_AIR,
    DE_URQUIJO_DRY_AIR_ATTACHMENT,
    DE_URQUIJO_DRY_AIR_DRIFT,
    DE_URQUIJO_DRY_AIR_LONGITUDINAL_DIFFUSION,
    DUTTON_DRY_AIR_TRANSVERSE_DIFFUSION_TO_MOBILITY,
    derived_coefficients,
    diffusion_coefficients_ref,
    low_field_diffusion_ref,
    low_field_drift_m_s,
    low_field_eta_over_n2_cm5,
    low_field_nodes_td,
    high_field_drift_m_s,
    high_field_nodes_td,
    morrow_lowke_eta3_over_n2_cm5,
    two_body_coefficients_ref,
)

REPO = Path(__file__).resolve().parent.parent
BUNDLE_DIR = REPO / "rrea_fluid_closure" / "dry_air_swarm_hybrid_v3"
CSV_NAME = "electron_flux_mobility_attachment_diffusion.csv"
CSV_COLUMNS = (
    "en_td",
    "flux_drift_velocity_m_per_s",
    "reduced_flux_mobility_ref_m2_per_v_s",
    "longitudinal_diffusion_ref_m2_per_s",
    "transverse_diffusion_ref_m2_per_s",
    "eta_over_n2_cm5",
    "three_body_rate_cm6_per_s",
    "three_body_frequency_ref_s",
    "eta_over_n_cm2",
    "two_body_rate_cm3_per_s",
    "two_body_frequency_ref_s",
    "relative_uncertainty_mobility",
    "relative_uncertainty_three_body",
    "relative_uncertainty_two_body",
    "source_row",
)


def csv_text() -> str:
    lines = [
        "# citation=Milloy et al., Aust. J. Phys. 28, 231 (1975); Shimamori and Hatano, Chem. Phys. 12, 439 (1976); Hegerberg and Reid, Aust. J. Phys. 33, 227 (1980); Morrow and Lowke, J. Phys. D 30, 614 (1997); de Urquijo et al., J. Phys. D 57, 125205 (2024); Dutton, J. Phys. Chem. Ref. Data 4, 577 (1975)",
        "# gas=dry air (0% H2O); measurement_temperature_K=293..300",
        "# valid_en_td=0.000001..100; interpolation=positive log-linear plus nonnegative two-body onset; below_range=hold zero-field asymptote; above_range=forbidden",
        ",".join(CSV_COLUMNS),
    ]
    printed_drift = {row[0] for row in DE_URQUIJO_DRY_AIR_DRIFT}
    printed_attachment = {row[0] for row in DE_URQUIJO_DRY_AIR_ATTACHMENT}
    printed_longitudinal = {
        row[0] for row in DE_URQUIJO_DRY_AIR_LONGITUDINAL_DIFFUSION
    }
    printed_transverse = {
        row[0] for row in DUTTON_DRY_AIR_TRANSVERSE_DIFFUSION_TO_MOBILITY
    }
    for en_td in low_field_nodes_td():
        w_m_s = low_field_drift_m_s(en_td)
        eta_cm5 = low_field_eta_over_n2_cm5(en_td, w_m_s)
        k0, k3, nu3_ref = derived_coefficients(
            en_td, w_m_s / 1.0e3, eta_cm5 / 1.0e-39)
        d_l_ref, d_t_ref = low_field_diffusion_ref(en_td, k0)
        source = (
            "thermal_limit_milloy_shimamori_hatano"
            if en_td < 0.1 else
            "hegerberg_reid__morrow_lowke__dutton"
            if en_td < 1.0 else
            "measured_endpoint_bridge__morrow_lowke__dutton"
            if en_td < 3.0 else
            "de_urquijo__dutton"
        )
        values = (
            en_td, w_m_s, k0, d_l_ref, d_t_ref, eta_cm5, k3,
            nu3_ref, 0.0, 0.0, 0.0,
            0.10 if 1.0 < en_td < 3.0 else (0.05 if en_td <= 0.1 else 0.03),
            0.20 if en_td < 3.0 else 0.10, 0.20,
        )
        lines.append(",".join(repr(value) for value in values) + f",{source}")
    for en_td, w_u, eta_u in DE_URQUIJO_DRY_AIR:
        if en_td <= 3.0:
            continue
        k0, k3, nu3_ref = derived_coefficients(en_td, w_u, eta_u)
        d_l_ref, d_t_ref = diffusion_coefficients_ref(en_td, k0)
        values = (
            en_td,
            w_u * 1.0e5 * 1.0e-2,
            k0,
            d_l_ref,
            d_t_ref,
            eta_u * 1.0e-39,
            k3,
            nu3_ref,
            0.0, 0.0, 0.0,
            0.015,
            0.10,
            0.20,
        )
        source = (
            f"table2_{'printed' if en_td in printed_drift else 'interpolated'}"
            f"__table5_{'printed' if en_td in printed_attachment else 'interpolated'}"
            f"__table3_{'printed' if en_td in printed_longitudinal else ('held' if en_td < 6.0 else 'interpolated')}"
            f"__dutton2p19_{'printed' if en_td in printed_transverse else 'interpolated'}"
        )
        lines.append(",".join(repr(value) for value in values) + f",{source}")
    for en_td in high_field_nodes_td():
        w_m_s = high_field_drift_m_s(en_td)
        eta3 = morrow_lowke_eta3_over_n2_cm5(en_td)
        k0, k3, nu3_ref = derived_coefficients(
            en_td, w_m_s / 1.0e3, eta3 / 1.0e-39)
        eta2, k2, nu2_ref = two_body_coefficients_ref(en_td, w_m_s)
        d_l_ref, d_t_ref = diffusion_coefficients_ref(en_td, k0)
        values = (
            en_td, w_m_s, k0, d_l_ref, d_t_ref,
            eta3, k3, nu3_ref, eta2, k2, nu2_ref,
            0.015, 0.20, 0.20,
        )
        source = "de_urquijo_table2_table3__dutton2p19__morrow_a4_a5"
        lines.append(",".join(repr(value) for value in values) + f",{source}")
    return "\n".join(lines) + "\n"


def parsed_rows(text: str) -> list[dict[str, str]]:
    lines = [line for line in text.splitlines() if line and not line.startswith("#")]
    return list(csv.DictReader(io.StringIO("\n".join(lines))))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    output = BUNDLE_DIR / CSV_NAME
    if args.check:
        if not output.is_file():
            print(f"MISSING {output}")
            return 1
        expected = parsed_rows(csv_text())
        observed = parsed_rows(output.read_text(encoding="ascii"))
        if observed != expected:
            print(f"MISMATCH parsed closure rows in {output}")
            return 1
        print(f"OK closure rows and units: {output}")
    else:
        BUNDLE_DIR.mkdir(parents=True, exist_ok=True)
        output.write_text(csv_text(), encoding="ascii", newline="\n")
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
