#!/usr/bin/env python3
"""Generate the dry-air zero--60 Td electron closure table.

The low-field part combines measured drift/diffusion anchors with cited
attachment fits and a thermal limit; transport through 60 Td is de Urquijo.
Runtime validates
the unit-bearing table semantically, never through a digest.

Scope and policy, deliberately explicit:

- Dry air, room temperature (293-300 K).  Not a storm-atmosphere closure.
- Valid range includes the thermal limit through 60 Td (CLOSURE_MAX_EN_TD:
  the fluid has no impact ionization, which is 8% of attachment there).
- Three-body attachment is measured through 30 Td; below 1 Td it is the
  thermal rate with the Kossyi electron-temperature dependence; the
  Morrow--Lowke A4/A5 transition supplies separate density-linear and
  density-squared channels above.
- The first positive node represents the measured zero-field asymptote; smaller
  realized fields hold that limit rather than extrapolating through log(0).
- ABOVE 60 Td evaluation is forbidden: the engine must abort, not clamp.
- Blank dry-air Table-5 cells are filled by log-linear interpolation between
  adjacent printed dry-air nodes.  Values in humid columns are never used.
- The uncertainty columns are broad source-wide ranges (~1.5% W_F,
  ~10% dry-air eta/N^2), not a point-by-point uncertainty model.

Usage:
    python scripts/generate_rrea_fluid_closure_bundle.py [--check]
    python scripts/generate_rrea_fluid_closure_bundle.py --excitation rates.csv --bolsig /path/to/bolsigminus

--check compares parsed data rows (the '#' metadata lines are not compared),
never file bytes.
"""

from __future__ import annotations

import argparse
import csv
import io
import re
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from scipy.constants import elementary_charge, electron_mass

sys.path.insert(0, str(Path(__file__).resolve().parent))

from verify_fluid_closure_anchors import (  # noqa: E402
    DE_URQUIJO_DRY_AIR,
    DE_URQUIJO_DRY_AIR_ATTACHMENT,
    DE_URQUIJO_DRY_AIR_DRIFT,
    DE_URQUIJO_DRY_AIR_LONGITUDINAL_DIFFUSION,
    CLOSURE_MAX_EN_TD,
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


def phelps_collisions():
    """Phelps complete sets: sigma_el = sigma_effective - sum(sigma_inelastic).

    Subtract the ORIGINAL set before any C-state sensitivity. The normalized
    three-body record is not a binary area and is excluded from this sum.
    LXCat conventions: Pitchford et al. (2017), doi:10.1002/ppap.201600098.
    """
    source = (REPO / "rrea_fluid_closure/phelps_swarm.txt").read_text()
    blocks = [(kind, header, np.loadtxt(io.StringIO(data))) for kind, header, data in
              re.findall(r"(?ms)^(EFFECTIVE|EXCITATION|IONIZATION|ATTACHMENT)\n"
                         r"(.*?)^-{5,}[ \t]*\n(.*?)^-{5,}[ \t]*$", source)]
    for i, (kind, header, data) in enumerate(blocks):
        if kind != "EFFECTIVE":
            continue
        gas = header.splitlines()[0]
        inelastic = [a for k, h, a in blocks if k != "EFFECTIVE"
                     and h.splitlines()[0].split()[0] == gas and "3-body" not in h]
        x = np.unique(np.concatenate([data[:, 0], *[a[:, 0] for a in inelastic]]))
        x = x[(x >= data[0, 0]) & (x <= data[-1, 0])]
        area = np.interp(x, *data.T) - sum(
            np.interp(x, *a.T, left=0., right=0.) for a in inelastic)
        if np.any(area < 0):
            raise ValueError(f"Negative elastic remainder for {gas}")
        blocks[i] = ("ELASTIC", header, np.c_[x, area])
    return blocks


def excitation_rates(bolsig, *, fields, temperature, density, argon, c_scale,
                     points, precision):
    """BOLSIG process rates, not integrals of its truncated EEDF export.

    Two-term Boltzmann model: Hagelaar & Pitchford, PSST 14, 722 (2005).
    Phelps O2 attachment is normalized to 1 cm^-3 O2 (phelps5 notes): multiply
    its area by n_O2/1e6, so nu_3 = k3*n_O2**2. No N2-assisted term is implied.
    Phelps's 0.67 recommendation concerns the older Report 26. The current
    export matches his posted table; applying it again remains a sensitivity.
    """
    if precision < np.finfo(float).eps:
        raise ValueError("BOLSIG precision must not be below machine precision (07/2024 manual)")
    fractions = np.array([.7809, .2095, argon])
    fractions /= fractions.sum()
    blocks = phelps_collisions()
    lines = []
    c_data = None
    for kind, header, raw in blocks:
        data = raw.copy()
        if "3-body" in header:
            data[:, 1] *= density * fractions[1] / 1e6
        if header.splitlines()[0] == "N2 -> N2(C3)":
            data[:, 1] *= c_scale
            c_data = data
        lines.extend([kind, header.rstrip(), "-----"])
        lines.extend(f"{e:.12e} {a:.12e}" for e, a in data)
        lines.extend(["-----", ""])
    conditions = ["VAR", 0, 0, temperature, temperature, 0, 0, density,
                  1, 1, 0, 1, 1, 0, points, 3, 200, precision, 1e-8, 2000,
                  " ".join(map(str, fractions)), 1]
    script = ("NOSCREEN\nREADCOLLISIONS\ncollisions.txt\nN2 O2 Ar\n0\n\nCONDITIONS\n"
              + "\n".join(map(str, conditions)) + "\n\nRUN\n"
              + "\n".join(map(str, fields))
              + "\n\nSAVERESULTS\nout.dat\n1\n1\n1\n1\n0\n0\n1\n1\n\nEND\n")
    with TemporaryDirectory(prefix="rrea_swarm_") as temporary:
        folder = Path(temporary)
        (folder / "collisions.txt").write_text("\n".join(lines))
        (folder / "input.dat").write_text(script)
        subprocess.run([str(Path(bolsig).resolve()), "input.dat"], cwd=folder,
                       check=True, capture_output=True, text=True)
        if not (folder / "out.dat").is_file():
            raise RuntimeError((folder / "bolsiglog.txt").read_text())
        output = (folder / "out.dat").read_text()
    sections = re.split(r"^R\d+\s*$", output, flags=re.M)[1:]
    if len(sections) != len(fields):
        raise ValueError("BOLSIG did not return every requested field")
    results = []
    for en, section in zip(fields, sections, strict=True):
        def scalar(label, section=section):
            return float(re.findall(r"^" + re.escape(label) + r"\s+(\S+)",
                                    section, re.M)[-1])
        if scalar("# of iterations") >= scalar("Maximum # of iterations"):
            raise ValueError(f"BOLSIG iteration limit at {en:g} Td")
        rate = float(re.findall(r"^C\d+\s+N2\s+Excitation\s+11\.03 eV\s+(\S+)",
                                section, re.M)[0])
        eedf = np.loadtxt(io.StringIO(section.split(
            "Energy (eV) EEDF (eV-3/2) Anisotropy")[1]))
        maximum = scalar("Maximum energy")
        # Numerical tail envelope for the declared Phelps C-state model:
        # f0 <= precision*max(f0) beyond the active grid. With v=gamma*sqrt(E),
        # delta k <= precision*max(f0)*gamma*integral E*sigma_C(E) dE.
        # This is conditional on the decreasing source-free swarm tail, not
        # a statement that a truncated rate or the source's 150-eV zero is exact.
        x = np.unique(np.r_[maximum, c_data[c_data[:, 0] > maximum, 0]])
        y = np.interp(x, *c_data.T, left=0., right=0.)
        integral = np.sum(np.diff(x) / 6 * (
            (2*x[:-1]+x[1:])*y[:-1] + (x[:-1]+2*x[1:])*y[1:]))
        tail = (precision * eedf[:, 1].max()
                * np.sqrt(2 * elementary_charge / electron_mass) * integral)
        if not np.isfinite(rate + tail) or min(rate, tail) < 0:
            raise ValueError("Invalid excitation rate")
        results.append(dict(en_td=en, temperature_K=temperature,
                            density_m3=density, argon_fraction=fractions[2],
                            c_scale=c_scale, grid_points=points, precision=precision,
                            k_c_m3_s=rate, tail_envelope_m3_s=tail,
                            flux_drift_m_s=scalar("Mobility *N (1/m/V/s)")*en*1e-21,
                            mean_energy_eV=scalar("Mean energy (eV)"),
                            active_max_eV=maximum))
    return results


def write_excitation(args):
    """One small rate table; original cross sections stay in phelps_swarm.txt."""
    rows = []
    for temperature in args.temperatures:
        for density in args.densities:
            for argon in args.argon:
                for scale in args.c_scales:
                    rows.extend(excitation_rates(
                        args.bolsig, fields=args.fields, temperature=temperature,
                        density=density, argon=argon, c_scale=scale,
                        points=args.points, precision=args.precision))
    with args.excitation.open("w") as stream:
        stream.write("# Phelps/LXCat N2 O2 Ar; BOLSIG+ 07/2024, temporal growth; "
                     "gas/excitation temperatures equal; O2-assisted attachment only.\n"
                     "# C(v) total: sigma_C0/sigma_C in [0.545,1] is a spectral "
                     "sensitivity, not measured vibrational branching.\n")
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({key: format(value, ".9g") for key, value in row.items()}
                         for row in rows)
    print(args.excitation, len(rows), "rates", flush=True)


def csv_text() -> str:
    lines = [
        "# citation=Milloy et al., Aust. J. Phys. 28, 231 (1975); Shimamori and Hatano, Chem. Phys. 12, 439 (1976); Hegerberg and Reid, Aust. J. Phys. 33, 227 (1980); Morrow and Lowke, J. Phys. D 30, 614 (1997); de Urquijo et al., J. Phys. D 57, 125205 (2024); Dutton, J. Phys. Chem. Ref. Data 4, 577 (1975)",
        "# gas=dry air (0% H2O); measurement_temperature_K=293..300",
        f"# valid_en_td=0.000001..{CLOSURE_MAX_EN_TD:g}; interpolation=positive log-linear plus nonnegative two-body onset; below_range=hold zero-field asymptote; above_range=forbidden",
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
            "hegerberg_reid__kossyi_shimamori_hatano__dutton"
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
    parser.add_argument("--excitation", type=Path, help="Write an offline C-state rate CSV")
    parser.add_argument("--bolsig", type=Path, help="Installed bolsigminus executable")
    parser.add_argument("--fields", type=float, nargs="+", default=np.r_[
        [1e-6, .1, 1, 3, 6, 10, 12, 14, 16, 18, 19], np.arange(20, 30.01, .25), [35, 40, 50, 60]])
    parser.add_argument("--temperatures", type=float, nargs="+", default=[200, 230, 266])
    parser.add_argument("--densities", type=float, nargs="+", default=[3.9e24, 8e24, 1.35e25])
    parser.add_argument("--argon", type=float, nargs="+", default=[.0093])
    parser.add_argument("--c-scales", type=float, nargs="+", default=[1., .67])
    parser.add_argument("--points", type=int, default=800)
    parser.add_argument("--precision", type=float, default=1e-15)
    args = parser.parse_args()
    if args.excitation:
        if args.bolsig is None:
            parser.error("--excitation requires --bolsig")
        write_excitation(args)
        return 0
    output = BUNDLE_DIR / CSV_NAME
    if args.check:
        if not output.is_file():
            print(f"MISSING {output}")
            return 1
        expected = parsed_rows(csv_text())
        observed = parsed_rows(output.read_text(encoding="ascii"))
        def numeric(rows):
            return [[float(row[key]) for key in CSV_COLUMNS[:-1]] for row in rows]
        if (len(observed) != len(expected)
                or not np.allclose(numeric(observed), numeric(expected), rtol=5e-13, atol=0)):
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
