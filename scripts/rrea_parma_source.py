#!/usr/bin/env python3
"""PARMA continuous cosmic-ray e-/e+ source (``parma_continuous_column``).

PHYSICS MODULE: single owner of the cosmic-ray seed source, applied to each
species s in {electron, positron}.  PARMA gives the steady omnidirectional
flux phi_s(E, z) (cm^-2 s^-1 MeV^-1, ``getSpec`` IDs 31 and 32).  The stream
is the birth rate that keeps that flux alive against slowing down; CSDA
continuity in energy gives the births above E and their spectrum,

    Q_s(>E, z) = phi_s(E, z) S(E) chi(z)       [cm^-3 s^-1]
    q_s(E, z)  = -dQ_s/dE,   Ndot_s = A int Q_s(>E_min, z) dz,

with S the ensemble mean drag at the transport tables' reference density
(``RreaInteractionTablesView::ElectronEnsembleDrag`` through the table probe;
positrons use it too, the tables differing by under 5% above 100 keV) and
chi = rho/rho_ref from the run's density profile.  Q is made non-increasing
in E (a local rise of phi S would be a negative birth rate).  The rate does
not depend on the column height.

Sampling factorizes per Sato (2016): altitude and energy from q, direction
zenith from the normalized
angular law ``getSpecAngFinal`` (class 5, the shared electron/positron
parametrization, a stated model limitation), uniform azimuth.  PARMA uses
``mu = cos(theta) = 1`` for vertically DOWNWARD motion while simulation +z is
up, so the emitted direction has ``d_z = -mu``.  Macro resolution is split
between the species in proportion to their physical rates, so all macro
weights agree to O(1/N) and the population controller sees one weight scale.

The schedule also carries the STATIONARY BACKGROUND at t = 0, with the field
on throughout. Interpreting PARMA's omnidirectional fluence as scalar flux
gives the stationary differential density

    n_s(E, z) = phi_s(E, z) / (beta(E) c)      [cm^-3 MeV^-1]
    N_bath,s  = A_cm2 * int dz_cm int dE  phi_s / (beta c),

sampled per species with the 1/beta weight (slightly softer than the flux
spectrum -- slow particles accumulate) and the same angular law, at the same
per-species macro weight as the stream, which replaces it within its
slowing-down time.

Citations: Sato (2015) PLoS ONE 10:e0144679; Sato (2016) PLoS ONE
11:e0160390; https://phits.jaea.go.jp/expacs/ (research use with citation;
commercial use requires JAEA agreement).  Fixed model assumptions: solar reference
2019-05-27 (most recent bundled daily W index), geometry parameter g = 0.15;
only latitude/longitude are configurable.  PARMA is evaluated at the
atmospheric depth of the run's own density profile
(``atmospheric_depth_g_cm2``), not at a standard atmosphere.

Determinism: a fresh run queries the Fortran adapter once and stores the
resulting flux/angle/drag table (text, %.17e) inside the run directory; schedule
generation reads only that table, so restart regeneration needs neither PARMA
nor gfortran. Across machines the platform libm (exp/log/pow) can shift values
by last-ulp amounts, so the driver compares numeric cells with a tight relative
tolerance.
Sampling is a Latin hypercube: stratified quantiles per channel,
decorrelated by independent SplitMix64 Fisher-Yates permutations keyed by
the source realization ID.

The engine replicates the full schedule on every MPI rank
(uniqueparticles=0 partitioning), so schedule memory and parsing cost scale
with the total macro count on every rank. Production resolution is a JSON
tuning parameter and requires an explicit convergence/performance study.

Standalone audit:  python scripts/rrea_parma_source.py --help
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import shutil
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path

from rrea_profiled_atmosphere import (
    ALTITUDE_AT_Z0_M,
    atmospheric_depth_g_cm2,
    load_air_density_profile,
    DOMAIN_HEIGHT_M,
    ELECTRON_REST_ENERGY_EV,
    SCHEMA6_CHARGED_ENERGY_MAX_EV,
    SEED_SCHEDULE_FIELDS,
    SPLITMIX64_STREAM_BASE,
    splitmix64_permutation,
)
from rrea_run_support import C, PARMA_CONTINUOUS_SOURCE_DEFAULTS


def relativistic_beta(kinetic_energy_eV: float) -> float:
    """Exact v/c for an electron or positron of the given kinetic energy."""
    gamma = 1.0 + kinetic_energy_eV / ELECTRON_REST_ENERGY_EV
    return math.sqrt(max(1.0 - 1.0 / (gamma * gamma), 0.0))

ROOT = Path(__file__).resolve().parents[1]
# Vendored PARMA model code + data (standalone, tracked): subroutines.f90,
# input/, LICENSE and README from github.com/davsar89/COSMIC_RAY_FLUXES.
# No external clone is needed at run time.
UPSTREAM_DIR = ROOT / "rrea_parma_upstream"
ADAPTER_SOURCE = ROOT / "scripts/parma_electron_query.f90"

TABLE_FORMAT = "PARMA_SOURCE_TABLE_V3"
SOLAR_REFERENCE = "2019-05-27"
GEOMETRY_PARAMETER = 0.15
ELECTRON_PARTICLE_ID = 31
POSITRON_PARTICLE_ID = 32
ANGULAR_PARTICLE_CLASS = 5
# Seed-schedule species ids (the engine injects 1 -> rrea_electrons,
# 2 -> rrea_positrons).
SPECIES = ("electron", "positron")
SPECIES_SEED_ID = {"electron": 1, "positron": 2}
# Extra per-species salt decorrelating the two Latin hypercubes.
_SPECIES_SALTS = {"electron": 0, "positron": 0x9BD3F27A51C86E15}
# Extra salt for the t=0 bath block's own hypercube.
_BATH_SALT = 0x7C55E1A94D3B26F7
SPEED_OF_LIGHT_CM_PER_S = 100.0 * C

# Baseline grids; the committed refinement test doubles them.
ALTITUDE_NODE_COUNT = 29
SPECTRAL_ENERGY_NODE_COUNT = 256
ANGULAR_ENERGY_NODE_COUNT = 33
MU_NODE_COUNT = 129

# Distinct per-channel salts decorrelate the Latin-hypercube permutations;
# time is strictly stratified and unpermuted.
_CHANNEL_SALTS = {
    "altitude": 0xA511E9B3D3C0FF11,
    "energy": 0x8F0C2B7A6E5D4C3B,
    "mu": 0xC3A5F1D2B4E60789,
    "radius": 0x5D2E8C4A9F1B3E67,
    "position_azimuth": 0xE7B9D1F3A5C70B2D,
    "direction_azimuth": 0x1F3E5D7C9BA0D2E5,
}


@dataclass(frozen=True)
class ParmaSourceTable:
    latitude_deg: float
    longitude_deg: float
    minimum_energy_eV: float
    maximum_energy_eV: float
    solar_w_index: float
    solar_status: int
    rigidity_gv: float
    altitude_km: list[float]
    depth_g_cm2: list[float]
    energy_eV: list[float]
    electron_spec_cm2_s_mev: list[list[float]]  # [altitude][energy]
    positron_spec_cm2_s_mev: list[list[float]]  # [altitude][energy]
    angular_energy_eV: list[float]
    mu: list[float]
    angular: list[list[list[float]]]  # shared e-/e+ law [alt][ang energy][mu]
    stopping_eV_per_m_stp: list[float]  # ensemble drag at rho_ref [energy]
    density_ratio: list[float]  # rho/rho_ref of the transport tables [altitude]

    def species_spec(self, species: str) -> list[list[float]]:
        if species == "electron":
            return self.electron_spec_cm2_s_mev
        if species == "positron":
            return self.positron_spec_cm2_s_mev
        raise ValueError(f"unknown species {species!r}")


def _log_grid(minimum: float, maximum: float, count: int) -> list[float]:
    ratio = maximum / minimum
    return [minimum * ratio ** (i / (count - 1)) for i in range(count)]


def _build_grids(
    minimum_energy_eV: float, maximum_energy_eV: float, refine: int = 1
) -> tuple[list[float], list[float], list[float], list[float]]:
    nz = (ALTITUDE_NODE_COUNT - 1) * refine + 1
    ne = SPECTRAL_ENERGY_NODE_COUNT * refine
    nea = (ANGULAR_ENERGY_NODE_COUNT - 1) * refine + 1
    nmu = (MU_NODE_COUNT - 1) * refine + 1
    alt_min_km = ALTITUDE_AT_Z0_M / 1000.0
    alt_max_km = (ALTITUDE_AT_Z0_M + DOMAIN_HEIGHT_M) / 1000.0
    altitude_km = [
        alt_min_km + (alt_max_km - alt_min_km) * i / (nz - 1) for i in range(nz)
    ]
    energy_eV = _log_grid(minimum_energy_eV, maximum_energy_eV, ne)
    angular_energy_eV = _log_grid(minimum_energy_eV, maximum_energy_eV, nea)
    mu = [-1.0 + 2.0 * i / (nmu - 1) for i in range(nmu)]
    return altitude_km, energy_eV, angular_energy_eV, mu


def _ensure_adapter_binary() -> Path:
    if not (UPSTREAM_DIR / "subroutines.f90").is_file():
        raise SystemExit(
            f"vendored PARMA model missing at {UPSTREAM_DIR}; it is tracked "
            "in this repository; restore it from Git"
        )
    binary = UPSTREAM_DIR / "parma_electron_query"
    sources = (UPSTREAM_DIR / "subroutines.f90", ADAPTER_SOURCE)
    if binary.is_file() and all(
        binary.stat().st_mtime >= source.stat().st_mtime for source in sources
    ):
        return binary
    if shutil.which("gfortran") is None:
        raise SystemExit(
            "gfortran is required to build the PARMA adapter (run under "
            "WSL/Linux); restarts of an existing run do not need it"
        )
    # Same flags as the upstream Makefile; subroutines first so its modules
    # exist when the adapter compiles.
    command = [
        "gfortran", "-O2", "-std=legacy", "-ffree-line-length-none",
        "-fno-backtrace", "-o", str(binary),
        str(sources[0]), str(sources[1]),
    ]
    result = subprocess.run(
        command, cwd=UPSTREAM_DIR, capture_output=True, text=True
    )
    if result.returncode != 0:
        raise SystemExit(
            f"PARMA adapter build failed:\n{' '.join(command)}\n{result.stderr}"
        )
    return binary


def query_parma_table(
    *,
    latitude_deg: float,
    longitude_deg: float,
    minimum_energy_eV: float,
    maximum_energy_eV: float,
    probe: Path,
    transport_config: Path,
    refine: int = 1,
) -> ParmaSourceTable:
    """Evaluate PARMA on the module grids via the batch adapter (fresh runs),
    and the ensemble drag there via the transport tables' own probe."""
    altitude_km, energy_eV, angular_energy_eV, mu = _build_grids(
        minimum_energy_eV, maximum_energy_eV, refine
    )
    lines = [
        f"{latitude_deg!r} {longitude_deg!r}",
        str(len(altitude_km)),
        " ".join(repr(v) for v in atmospheric_depth_g_cm2(
            [1000.0 * a for a in altitude_km])),
        str(len(energy_eV)),
        " ".join(repr(v / 1.0e6) for v in energy_eV),
        str(len(angular_energy_eV)),
        " ".join(repr(v / 1.0e6) for v in angular_energy_eV),
        str(len(mu)),
        " ".join(repr(v) for v in mu),
    ]
    drag = subprocess.run(
        [str(probe), str(transport_config), "drag"],
        input="\n".join(repr(e) for e in energy_eV) + "\n",
        capture_output=True, text=True, check=True).stdout.splitlines()
    stopping = [float(line.split("=", 1)[1]) for line in drag
                if line.startswith("drag_eV_per_m=")]
    if len(stopping) != len(energy_eV):
        raise SystemExit(f"{probe}: expected {len(energy_eV)} drag values")
    binary = _ensure_adapter_binary()
    result = subprocess.run(
        [str(binary)],
        cwd=UPSTREAM_DIR,  # the model opens its data as 'input/...'
        input="\n".join(lines) + "\n",
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise SystemExit(f"PARMA adapter failed: {result.stderr.strip()}")
    tokens = result.stdout.split()
    cursor = 0

    def take(count: int) -> list[str]:
        nonlocal cursor
        chunk = tokens[cursor:cursor + count]
        if len(chunk) != count:
            raise SystemExit("PARMA adapter output truncated")
        cursor += count
        return chunk

    def expect(label: str) -> None:
        got = take(1)[0]
        if got != label:
            raise SystemExit(f"PARMA adapter output: expected {label}, got {got}")

    expect("PARMA_QUERY_V2")
    expect("SOLAR")
    solar_date = "-".join(f"{int(v):02d}" for v in take(3))
    if solar_date != SOLAR_REFERENCE:
        raise SystemExit(f"PARMA adapter reports foreign solar date {solar_date}")
    solar_w_index = float(take(1)[0])
    solar_status = int(take(1)[0])
    expect("RIGIDITY_GV")
    rigidity_gv = float(take(1)[0])
    expect("DEPTH_G_CM2")
    nz = int(take(1)[0])
    depth = [float(v) for v in take(nz)]
    specs: dict[str, list[list[float]]] = {}
    for label, name in (
        (f"SPEC{ELECTRON_PARTICLE_ID}", "electron"),
        (f"SPEC{POSITRON_PARTICLE_ID}", "positron"),
    ):
        expect(label)
        nz2, ne = int(take(1)[0]), int(take(1)[0])
        if nz2 != len(altitude_km) or ne != len(energy_eV):
            raise SystemExit("PARMA adapter grid sizes disagree with the request")
        flat = [float(v) for v in take(nz2 * ne)]
        specs[name] = [flat[iz * ne:(iz + 1) * ne] for iz in range(nz2)]
    expect("ANG")
    nz3, nea, nmu = int(take(1)[0]), int(take(1)[0]), int(take(1)[0])
    ang_flat = [float(v) for v in take(nz3 * nea * nmu)]
    if not (nz == nz3 == len(altitude_km)
            and nea == len(angular_energy_eV) and nmu == len(mu)):
        raise SystemExit("PARMA adapter grid sizes disagree with the request")
    angular = [
        [
            ang_flat[(iz * nea + ie) * nmu:(iz * nea + ie + 1) * nmu]
            for ie in range(nea)
        ]
        for iz in range(nz)
    ]
    return ParmaSourceTable(
        latitude_deg=latitude_deg,
        longitude_deg=longitude_deg,
        minimum_energy_eV=minimum_energy_eV,
        maximum_energy_eV=maximum_energy_eV,
        solar_w_index=solar_w_index,
        solar_status=solar_status,
        rigidity_gv=rigidity_gv,
        altitude_km=altitude_km,
        depth_g_cm2=depth,
        energy_eV=energy_eV,
        electron_spec_cm2_s_mev=specs["electron"],
        positron_spec_cm2_s_mev=specs["positron"],
        angular_energy_eV=angular_energy_eV,
        mu=mu,
        angular=angular,
        stopping_eV_per_m_stp=stopping,
        density_ratio=[
            load_air_density_profile().interpolate(1000.0 * a)
            / json.loads(transport_config.read_text())["reference_density_kg_m3"]
            for a in altitude_km
        ],
    )


def write_parma_source_table(table: ParmaSourceTable, path: Path) -> None:
    def block(name: str, values: list[float]) -> list[str]:
        return [f"{name} {len(values)}"] + [f"{v:.17e}" for v in values]

    lines = [
        TABLE_FORMAT,
        f"latitude_deg {table.latitude_deg:.17e}",
        f"longitude_deg {table.longitude_deg:.17e}",
        f"minimum_energy_eV {table.minimum_energy_eV:.17e}",
        f"maximum_energy_eV {table.maximum_energy_eV:.17e}",
        f"solar_reference {SOLAR_REFERENCE}",
        f"solar_w_index {table.solar_w_index:.17e}",
        f"solar_status {table.solar_status}",
        f"rigidity_gv {table.rigidity_gv:.17e}",
        f"geometry_parameter {GEOMETRY_PARAMETER:.17e}",
        f"electron_particle_id {ELECTRON_PARTICLE_ID}",
        f"positron_particle_id {POSITRON_PARTICLE_ID}",
        f"angular_particle_class {ANGULAR_PARTICLE_CLASS}",
        "atmosphere_model run_density_profile",
        *block("altitude_km", table.altitude_km),
        *block("depth_g_cm2", table.depth_g_cm2),
        *block("energy_eV", table.energy_eV),
        *block(
            "spec_electron_cm2_s_mev",
            [v for row in table.electron_spec_cm2_s_mev for v in row],
        ),
        *block(
            "spec_positron_cm2_s_mev",
            [v for row in table.positron_spec_cm2_s_mev for v in row],
        ),
        *block("angular_energy_eV", table.angular_energy_eV),
        *block("mu", table.mu),
        *block(
            "angular",
            [v for plane in table.angular for row in plane for v in row],
        ),
        *block("stopping_eV_per_m_stp", table.stopping_eV_per_m_stp),
        *block("density_ratio", table.density_ratio),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def load_parma_source_table(path: Path) -> ParmaSourceTable:
    tokens = path.read_text(encoding="utf-8").split()
    cursor = 0

    def take(count: int) -> list[str]:
        nonlocal cursor
        chunk = tokens[cursor:cursor + count]
        if len(chunk) != count:
            raise SystemExit(f"{path}: truncated PARMA source table")
        cursor += count
        return chunk

    def scalar(name: str) -> str:
        got = take(1)[0]
        if got != name:
            raise SystemExit(f"{path}: expected {name}, got {got}")
        return take(1)[0]

    def block(name: str) -> list[float]:
        count = int(scalar(name))
        return [float(v) for v in take(count)]

    if take(1)[0] != TABLE_FORMAT:
        raise SystemExit(f"{path}: not a {TABLE_FORMAT} file")
    latitude_deg = float(scalar("latitude_deg"))
    longitude_deg = float(scalar("longitude_deg"))
    minimum_energy_eV = float(scalar("minimum_energy_eV"))
    maximum_energy_eV = float(scalar("maximum_energy_eV"))
    if scalar("solar_reference") != SOLAR_REFERENCE:
        raise SystemExit(f"{path}: foreign solar reference")
    solar_w_index = float(scalar("solar_w_index"))
    solar_status = int(scalar("solar_status"))
    rigidity_gv = float(scalar("rigidity_gv"))
    if float(scalar("geometry_parameter")) != GEOMETRY_PARAMETER:
        raise SystemExit(f"{path}: foreign geometry parameter")
    if int(scalar("electron_particle_id")) != ELECTRON_PARTICLE_ID:
        raise SystemExit(f"{path}: foreign electron particle id")
    if int(scalar("positron_particle_id")) != POSITRON_PARTICLE_ID:
        raise SystemExit(f"{path}: foreign positron particle id")
    if int(scalar("angular_particle_class")) != ANGULAR_PARTICLE_CLASS:
        raise SystemExit(f"{path}: foreign angular particle class")
    if scalar("atmosphere_model") != "run_density_profile":
        raise SystemExit(f"{path}: foreign atmosphere model")
    altitude_km = block("altitude_km")
    depth = block("depth_g_cm2")
    energy_eV = block("energy_eV")
    electron_flat = block("spec_electron_cm2_s_mev")
    positron_flat = block("spec_positron_cm2_s_mev")
    angular_energy_eV = block("angular_energy_eV")
    mu = block("mu")
    ang_flat = block("angular")
    stopping = block("stopping_eV_per_m_stp")
    density_ratio = block("density_ratio")
    nz, ne, nea, nmu = (
        len(altitude_km), len(energy_eV), len(angular_energy_eV), len(mu)
    )
    if (
        len(depth) != nz
        or len(electron_flat) != nz * ne
        or len(positron_flat) != nz * ne
        or len(ang_flat) != nz * nea * nmu
        or len(stopping) != ne
        or len(density_ratio) != nz
    ):
        raise SystemExit(f"{path}: inconsistent PARMA table block sizes")
    return ParmaSourceTable(
        latitude_deg=latitude_deg,
        longitude_deg=longitude_deg,
        minimum_energy_eV=minimum_energy_eV,
        maximum_energy_eV=maximum_energy_eV,
        solar_w_index=solar_w_index,
        solar_status=solar_status,
        rigidity_gv=rigidity_gv,
        altitude_km=altitude_km,
        depth_g_cm2=depth,
        energy_eV=energy_eV,
        electron_spec_cm2_s_mev=[
            electron_flat[iz * ne:(iz + 1) * ne] for iz in range(nz)
        ],
        positron_spec_cm2_s_mev=[
            positron_flat[iz * ne:(iz + 1) * ne] for iz in range(nz)
        ],
        angular_energy_eV=angular_energy_eV,
        mu=mu,
        angular=[
            [
                ang_flat[(iz * nea + ie) * nmu:(iz * nea + ie + 1) * nmu]
                for ie in range(nea)
            ]
            for iz in range(nz)
        ],
        stopping_eV_per_m_stp=stopping,
        density_ratio=density_ratio,
    )


def ensure_parma_source_table(
    table_path: Path,
    *,
    latitude_deg: float,
    longitude_deg: float,
    minimum_energy_eV: float,
    maximum_energy_eV: float,
    probe: Path,
    transport_config: Path,
) -> ParmaSourceTable:
    """Load the run's stored table, or generate it once on a fresh start.

    The stored table is the sole sampling input, so restarts regenerate the
    schedule without PARMA present (to the platform libm's last ulp; the
    driver's restart compare is numeric).  A parameter mismatch against the
    currently resolved configuration is the one drift channel the schedule
    compare cannot see, so it fails closed here.
    """
    if table_path.exists():
        table = load_parma_source_table(table_path)
        for name, stored, requested in (
            ("latitude_deg", table.latitude_deg, latitude_deg),
            ("longitude_deg", table.longitude_deg, longitude_deg),
            ("minimum_energy_eV", table.minimum_energy_eV, minimum_energy_eV),
            ("maximum_energy_eV", table.maximum_energy_eV, maximum_energy_eV),
        ):
            if stored != requested:
                raise SystemExit(
                    f"{table_path}: stored {name}={stored!r} disagrees with the "
                    f"resolved configuration {requested!r}; a PARMA run cannot "
                    "change source parameters across a restart"
                )
        return table
    table = query_parma_table(
        latitude_deg=latitude_deg,
        longitude_deg=longitude_deg,
        minimum_energy_eV=minimum_energy_eV,
        maximum_energy_eV=maximum_energy_eV,
        probe=probe,
        transport_config=transport_config,
    )
    write_parma_source_table(table, table_path)
    return table


def _production_density(
    energy_mev: list[float], phi: list[float], stopping_mev_per_cm: list[float],
    chi: float,
) -> list[float]:
    """q = -dQ/dE on the energy nodes, Q(>E) = chi phi S made non-increasing."""
    rate_above = [chi * f * s for f, s in zip(phi, stopping_mev_per_cm)]
    for i in range(len(rate_above) - 2, -1, -1):
        rate_above[i] = max(rate_above[i], rate_above[i + 1])
    last = len(rate_above) - 1
    return [
        (rate_above[max(i - 1, 0)] - rate_above[min(i + 1, last)])
        / (energy_mev[min(i + 1, last)] - energy_mev[max(i - 1, 0)])
        for i in range(len(rate_above))
    ]


def _trapezoid_cumulative(x: list[float], y: list[float]) -> list[float]:
    cumulative = [0.0]
    for i in range(1, len(x)):
        cumulative.append(
            cumulative[-1] + 0.5 * (y[i] + y[i - 1]) * (x[i] - x[i - 1])
        )
    return cumulative


def _invert_piecewise_linear_density(
    x: list[float], y: list[float], cumulative: list[float], q: float
) -> float:
    """Invert the CDF of a piecewise-linear density at quantile q in (0, 1).

    Grid endpoints are integration BOUNDS, not samples; q never reaches 0/1
    (stratified quantiles are interior), so results are strictly inside.
    """
    total = cumulative[-1]
    target = q * total
    k = min(bisect_right(cumulative, target), len(cumulative) - 1) - 1
    k = max(k, 0)
    segment = cumulative[k + 1] - cumulative[k]
    if segment <= 0.0:
        return x[k]
    dx = x[k + 1] - x[k]
    y0, y1 = y[k], y[k + 1]
    a = target - cumulative[k]
    if y1 == y0:
        t = a / segment
    else:
        # Solve y0*t + (y1-y0)*t^2/2 = a/dx for t in [0, 1].
        slope = (y1 - y0) / dx
        disc = y0 * y0 + 2.0 * slope * a
        t = (math.sqrt(max(disc, 0.0)) - y0) / (slope * dx)
    return x[k] + min(max(t, 0.0), 1.0) * dx


def _forward_piecewise_linear_cdf(
    x: list[float], y: list[float], cumulative: list[float], value: float
) -> float:
    total = cumulative[-1]
    k = min(max(bisect_right(x, value) - 1, 0), len(x) - 2)
    t = (value - x[k]) / (x[k + 1] - x[k])
    y0, y1 = y[k], y[k + 1]
    partial = (y0 * t + 0.5 * (y1 - y0) * t * t) * (x[k + 1] - x[k])
    return (cumulative[k] + partial) / total


class PreparedParmaSource:
    """CDF machinery precomputed from one stored table (single owner).

    Spectrum-dependent parts (column birth rate, altitude marginal, energy
    CDFs) are held per species; the angular machinery is shared because the
    class-5 law is one e-/e+ parametrization.
    """

    def __init__(self, table: ParmaSourceTable):
        self.table = table
        energy_mev = [e / 1.0e6 for e in table.energy_eV]
        self._energy_mev = energy_mev
        stopping_mev_per_cm = [v * 1.0e-8 for v in table.stopping_eV_per_m_stp]
        # Birth density q_s [altitude][energy], its rate per altitude node
        # (cm^-3 s^-1) and the column rate (cm^-2 s^-1).
        self.production: dict[str, list[list[float]]] = {}
        self.production_rate: dict[str, list[float]] = {}
        self.column_rate_cm2_s: dict[str, float] = {}
        self._altitude_cumulative: dict[str, list[float]] = {}
        self._energy_cumulative: dict[str, list[list[float]]] = {}
        # Bath machinery: density n = phi/(beta c), so every bath CDF is the
        # stream CDF reweighted by 1/beta.
        inverse_speed = [
            1.0 / (relativistic_beta(e) * SPEED_OF_LIGHT_CM_PER_S)
            for e in table.energy_eV
        ]
        self._bath_density: dict[str, list[list[float]]] = {}
        self.bath_density_cm3: dict[str, list[float]] = {}  # [altitude]
        self.bath_column_cm2: dict[str, float] = {}
        self._bath_altitude_cumulative: dict[str, list[float]] = {}
        self._bath_energy_cumulative: dict[str, list[list[float]]] = {}
        for name in SPECIES:
            spec = [
                _production_density(energy_mev, row, stopping_mev_per_cm, chi)
                for row, chi in zip(table.species_spec(name), table.density_ratio)
            ]
            rate = [_trapezoid_cumulative(energy_mev, row)[-1] for row in spec]
            alt_cum = _trapezoid_cumulative(table.altitude_km, rate)
            self.production[name] = spec
            self.production_rate[name] = rate
            self._altitude_cumulative[name] = alt_cum
            self.column_rate_cm2_s[name] = alt_cum[-1] * 1.0e5  # dz km -> cm
            self._energy_cumulative[name] = [
                _trapezoid_cumulative(energy_mev, row) for row in spec
            ]
            density = [
                [phi * inv for phi, inv in zip(row, inverse_speed)]
                for row in table.species_spec(name)
            ]
            self._bath_density[name] = density
            self.bath_density_cm3[name] = [
                _trapezoid_cumulative(energy_mev, row)[-1] for row in density
            ]
            bath_alt_cum = _trapezoid_cumulative(
                table.altitude_km, self.bath_density_cm3[name]
            )
            self._bath_altitude_cumulative[name] = bath_alt_cum
            # Column bath content per unit area [cm^-2]: dz in km -> cm.
            self.bath_column_cm2[name] = bath_alt_cum[-1] * 1.0e5
            self._bath_energy_cumulative[name] = [
                _trapezoid_cumulative(energy_mev, row) for row in density
            ]
        if max(self.column_rate_cm2_s.values()) <= 0.0:
            raise SystemExit("PARMA table carries no positive production rate")
        # Shared angular CDFs at (altitude, angular-energy) nodes.
        self._mu_cumulative = [
            [_trapezoid_cumulative(table.mu, row) for row in plane]
            for plane in table.angular
        ]
        self._log_ang_energy = [math.log(e) for e in table.angular_energy_eV]

    # -- forward CDFs (exposed for the committed marginal tests) ----------
    def altitude_cdf(self, species: str, altitude_km: float) -> float:
        return _forward_piecewise_linear_cdf(
            self.table.altitude_km, self.production_rate[species],
            self._altitude_cumulative[species], altitude_km,
        )

    def energy_cdf_at_node(self, species: str, iz: int, energy_eV: float) -> float:
        return _forward_piecewise_linear_cdf(
            self._energy_mev, self.production[species][iz],
            self._energy_cumulative[species][iz], energy_eV / 1.0e6,
        )

    def mu_cdf_at_node(self, iz: int, ie: int, mu: float) -> float:
        return _forward_piecewise_linear_cdf(
            self.table.mu, self.table.angular[iz][ie],
            self._mu_cumulative[iz][ie], mu,
        )

    def bath_altitude_cdf(self, species: str, altitude_km: float) -> float:
        return _forward_piecewise_linear_cdf(
            self.table.altitude_km, self.bath_density_cm3[species],
            self._bath_altitude_cumulative[species], altitude_km,
        )

    def bath_energy_cdf_at_node(
        self, species: str, iz: int, energy_eV: float
    ) -> float:
        return _forward_piecewise_linear_cdf(
            self._energy_mev, self._bath_density[species][iz],
            self._bath_energy_cumulative[species][iz], energy_eV / 1.0e6,
        )

    # -- inversions -------------------------------------------------------
    def sample_altitude(self, species: str, q: float) -> float:
        return _invert_piecewise_linear_density(
            self.table.altitude_km, self.production_rate[species],
            self._altitude_cumulative[species], q,
        )

    def sample_bath_altitude(self, species: str, q: float) -> float:
        return _invert_piecewise_linear_density(
            self.table.altitude_km, self.bath_density_cm3[species],
            self._bath_altitude_cumulative[species], q,
        )

    def sample_bath_energy_eV(
        self, species: str, q: float, altitude_km: float
    ) -> float:
        return self._sample_energy(
            self._bath_density[species], self._bath_energy_cumulative[species],
            q, altitude_km)

    def _altitude_bracket(self, altitude_km: float) -> tuple[int, float]:
        grid = self.table.altitude_km
        k = min(max(bisect_right(grid, altitude_km) - 1, 0), len(grid) - 2)
        return k, (altitude_km - grid[k]) / (grid[k + 1] - grid[k])

    def sample_energy_eV(
        self, species: str, q: float, altitude_km: float
    ) -> float:
        return self._sample_energy(
            self.production[species], self._energy_cumulative[species],
            q, altitude_km)

    def _sample_energy(
        self, spec: list[list[float]], cums: list[list[float]], q: float,
        altitude_km: float,
    ) -> float:
        # Quantile blending between altitude nodes; geometric blend because
        # the spectrum spans five decades and quantiles live in log E.
        k, w = self._altitude_bracket(altitude_km)
        lo = _invert_piecewise_linear_density(
            self._energy_mev, spec[k], cums[k], q,
        )
        hi = _invert_piecewise_linear_density(
            self._energy_mev, spec[k + 1], cums[k + 1], q,
        )
        blended = math.exp((1.0 - w) * math.log(lo) + w * math.log(hi)) * 1.0e6
        # FP-boundary guard only (mirrors the C&D sampler); quantiles are
        # interior so no physical mass is ever redistributed by this.
        return min(
            max(blended, self.table.minimum_energy_eV),
            self.table.maximum_energy_eV,
        )

    def sample_mu(self, q: float, altitude_km: float, energy_eV: float) -> float:
        k, wz = self._altitude_bracket(altitude_km)
        log_e = math.log(energy_eV)
        grid = self._log_ang_energy
        j = min(max(bisect_right(grid, log_e) - 1, 0), len(grid) - 2)
        we = (log_e - grid[j]) / (grid[j + 1] - grid[j])
        we = min(max(we, 0.0), 1.0)
        corners = 0.0
        for iz, wa in ((k, 1.0 - wz), (k + 1, wz)):
            for ie, wb in ((j, 1.0 - we), (j + 1, we)):
                corners += wa * wb * _invert_piecewise_linear_density(
                    self.table.mu, self.table.angular[iz][ie],
                    self._mu_cumulative[iz][ie], q,
                )
        return min(max(corners, -1.0), 1.0)

    # -- audit moments ----------------------------------------------------
    def mean_energy_eV_per_altitude(self, species: str) -> list[float]:
        result = []
        for iz, row in enumerate(self.production[species]):
            weighted = [e * phi for e, phi in zip(self._energy_mev, row)]
            total = self._energy_cumulative[species][iz][-1]
            result.append(
                _trapezoid_cumulative(self._energy_mev, weighted)[-1]
                / total * 1.0e6 if total > 0.0 else 0.0
            )
        return result

    def mean_mu_per_altitude(self, species: str) -> list[float]:
        """Spectrum-weighted mean zenith cosine (mu=1 is downward)."""
        result = []
        ang_e_mev = [e / 1.0e6 for e in self.table.angular_energy_eV]
        spec = self.production[species]
        for iz, plane in enumerate(self.table.angular):
            mu_means = []
            for ie, row in enumerate(plane):
                norm = self._mu_cumulative[iz][ie][-1]
                weighted = [m * v for m, v in zip(self.table.mu, row)]
                mu_means.append(
                    _trapezoid_cumulative(self.table.mu, weighted)[-1] / norm
                    if norm > 0.0 else 0.0
                )
            spec_at = [
                _interpolate_log(self._energy_mev, spec[iz], e)
                for e in ang_e_mev
            ]
            norm = _trapezoid_cumulative(ang_e_mev, spec_at)[-1]
            weighted = [m * s for m, s in zip(mu_means, spec_at)]
            result.append(
                _trapezoid_cumulative(ang_e_mev, weighted)[-1] / norm
                if norm > 0.0 else 0.0
            )
        return result


def _interpolate_log(x: list[float], y: list[float], value: float) -> float:
    k = min(max(bisect_right(x, value) - 1, 0), len(x) - 2)
    t = (math.log(value) - math.log(x[k])) / (math.log(x[k + 1]) - math.log(x[k]))
    return (1.0 - t) * y[k] + t * y[k + 1]


def parma_source_audit(
    prepared: PreparedParmaSource, *, source_radius_m: float
) -> dict:
    table = prepared.table
    area_cm2 = math.pi * (source_radius_m * 100.0) ** 2
    species = {
        name: {
            "physical_particles_per_s": (
                area_cm2 * prepared.column_rate_cm2_s[name]
            ),
            "column_rate_cm2_s": prepared.column_rate_cm2_s[name],
            "production_rate_cm3_s_per_altitude": list(
                prepared.production_rate[name]
            ),
            "mean_energy_eV_per_altitude": (
                prepared.mean_energy_eV_per_altitude(name)
            ),
            "mean_mu_per_altitude": prepared.mean_mu_per_altitude(name),
            # Stationary bath n = phi/(beta c): exact kinetic identity.
            "bath_density_cm3_per_altitude": list(
                prepared.bath_density_cm3[name]
            ),
            "physical_bath_particles": (
                area_cm2 * prepared.bath_column_cm2[name]
            ),
        }
        for name in SPECIES
    }
    return {
        "model": "parma_continuous_column",
        "governing_equation": "Q_s(>E,z)=phi_s(E,z)*S(E)*chi(z); q_s=-dQ_s/dE; Ndot_s=A*int Q_s(>E_min,z) dz",
        "latitude_deg": table.latitude_deg,
        "longitude_deg": table.longitude_deg,
        "solar_reference": SOLAR_REFERENCE,
        "solar_w_index": table.solar_w_index,
        "rigidity_gv": table.rigidity_gv,
        "minimum_energy_eV": table.minimum_energy_eV,
        "maximum_energy_eV": table.maximum_energy_eV,
        "source_radius_m": source_radius_m,
        "source_area_cm2": area_cm2,
        "altitude_km": list(table.altitude_km),
        "species": species,
        "physical_electrons_per_s": (
            species["electron"]["physical_particles_per_s"]
        ),
        "physical_positrons_per_s": (
            species["positron"]["physical_particles_per_s"]
        ),
        "physical_particles_per_s_total": sum(
            entry["physical_particles_per_s"] for entry in species.values()
        ),
        "physical_bath_particles_total": sum(
            entry["physical_bath_particles"] for entry in species.values()
        ),
        "grid_sizes": {
            "altitude": len(table.altitude_km),
            "energy": len(table.energy_eV),
            "angular_energy": len(table.angular_energy_eV),
            "mu": len(table.mu),
        },
    }


def write_parma_seed_schedule(
    path: Path,
    *,
    table: ParmaSourceTable,
    case_id: str,
    macro_count: int,
    realization_id: int,
    stop_time_s: float,
    source_radius_m: float,
    sampling_interval_s: float | None = None,
    append_offset: int = 0,
) -> tuple[float, dict]:
    """Stratified source intervals with weight = rate * interval / count.

    The first interval fixes weights and the stationary bath. Extensions use
    independent permutations at the same resolution, never resampling history.
    A positive append_offset appends after that many existing pulse macros;
    their header and rows remain untouched, and PARMA indices start there.
    Returns the scheduled weight (bath included) and its physical-rate audit.
    The driver validates the counts, times and radius.
    """
    if sampling_interval_s is None:
        sampling_interval_s = stop_time_s
    if not 0.0 < sampling_interval_s <= stop_time_s:
        raise ValueError("sampling_interval_s must be positive and no longer than stop_time_s")
    prepared = PreparedParmaSource(table)
    audit = parma_source_audit(prepared, source_radius_m=source_radius_m)
    ndot = {
        name: audit["species"][name]["physical_particles_per_s"]
        for name in SPECIES
    }
    ndot_total = audit["physical_particles_per_s_total"]
    positive = [name for name in SPECIES if ndot[name] > 0.0]
    counts = dict.fromkeys(SPECIES, 0)
    if len(positive) == 1:
        counts[positive[0]] = macro_count
    else:
        if macro_count < 2:
            raise ValueError(
                "macro_count must be at least 2 for a two-species source"
            )
        electron_count = int(round(macro_count * ndot["electron"] / ndot_total))
        counts["electron"] = min(max(electron_count, 1), macro_count - 1)
        counts["positron"] = macro_count - counts["electron"]
    weights = {
        name: (ndot[name] * sampling_interval_s / counts[name]) if counts[name] else 0.0
        for name in SPECIES
    }
    # Stationary bath at t=0, at the same per-species weight as the stream.
    bath_physical = {
        name: audit["species"][name]["physical_bath_particles"]
        for name in SPECIES
    }
    bath_counts = {
        name: (
            int(round(bath_physical[name] / weights[name]))
            if counts[name]
            else 0
        )
        for name in SPECIES
    }

    def channel_permutations(block_counts: dict[str, int], extra_salt: int):
        return {
            (name, channel): splitmix64_permutation(
                block_counts[name],
                SPLITMIX64_STREAM_BASE
                ^ realization_id
                ^ channel_salt
                ^ _SPECIES_SALTS[name]
                ^ extra_salt,
            )
            for name in SPECIES
            if block_counts[name]
            for channel, channel_salt in _CHANNEL_SALTS.items()
        }

    bath_perms = channel_permutations(bath_counts, _BATH_SALT)
    stream_counts = dict.fromkeys(SPECIES, 0)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a" if append_offset else "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SEED_SCHEDULE_FIELDS)
        if not append_offset:
            writer.writeheader()
        macro_index = append_offset

        def emit_row(
            name: str,
            time_s: float,
            index: int,
            block: int,
            permutations: dict,
            *,
            bath: bool = False,
        ) -> None:
            nonlocal macro_index
            q = {channel: (permutations[(name, channel)][index] + 0.5) / block
                 for channel in _CHANNEL_SALTS}
            sample_altitude = prepared.sample_bath_altitude if bath else prepared.sample_altitude
            sample_energy = prepared.sample_bath_energy_eV if bath else prepared.sample_energy_eV
            altitude_km = sample_altitude(name, q["altitude"])
            energy_eV = sample_energy(name, q["energy"], altitude_km)
            mu = prepared.sample_mu(q["mu"], altitude_km, energy_eV)
            radius = source_radius_m * math.sqrt(q["radius"])
            position_phi = 2.0 * math.pi * q["position_azimuth"]
            direction_phi = 2.0 * math.pi * q["direction_azimuth"]
            x = radius * math.cos(position_phi)
            y = radius * math.sin(position_phi)
            z_m = altitude_km * 1000.0 - ALTITUDE_AT_Z0_M
            if not 0.0 <= z_m < DOMAIN_HEIGHT_M:
                raise ValueError(
                    f"sampled altitude {altitude_km} km leaves the half-open "
                    "simulation domain; the table altitude grid must span "
                    "exactly the domain column"
                )
            sin_theta = math.sqrt(max(1.0 - mu * mu, 0.0))
            # PARMA mu=1 is downward; simulation +z is up.  Unit direction
            # hint; the injection primitive derives proper velocity from the
            # kinetic energy (the electron and positron rest masses agree).
            writer.writerow(
                {
                    "case_id": case_id,
                    "macro_index": macro_index,
                    "time_s": f"{time_s:.17e}",
                    "x_m": f"{x:.17e}",
                    "y_m": f"{y:.17e}",
                    "r_m": f"{math.hypot(x, y):.17e}",
                    "z_m": f"{z_m:.17e}",
                    "phi_rad": f"{position_phi:.17e}",
                    "source_radius_m": f"{source_radius_m:.17e}",
                    "kinetic_energy_eV": f"{energy_eV:.17e}",
                    "ux": f"{sin_theta * math.cos(direction_phi):.17e}",
                    "uy": f"{sin_theta * math.sin(direction_phi):.17e}",
                    "uz": f"{-mu:.17e}",
                    "weight_real_electrons": f"{weights[name]:.17e}",
                    "represented_real_electrons": f"{weights[name]:.17e}",
                    "species_id": str(SPECIES_SEED_ID[name]),
                    "generation_id": "0",
                    "source_region": "parma_column_bath" if bath else "parma_column_cylinder",
                    "direction_model": "parma_sato2016_zenith_uniform_azimuth",
                }
            )
            macro_index += 1

        # Bath block: the stationary population, all at t = 0.
        for name in SPECIES:
            block = bath_counts[name]
            for index in range(block):
                emit_row(name, 0.0, index, block, bath_perms, bath=True)
        for interval in range(math.ceil(stop_time_s / sampling_interval_s)):
            stream_perms = channel_permutations(counts, interval * 0x9E3779B97F4A7C15)
            events = [
                (interval * sampling_interval_s
                 + sampling_interval_s * (index + 0.5) / counts[name], name, index)
                for name in SPECIES for index in range(counts[name])
            ]
            if interval:
                events.sort()
            for event_time, name, index in events:
                if event_time >= stop_time_s:
                    continue
                emit_row(name, event_time, index, counts[name], stream_perms)
                stream_counts[name] += 1
    bath_total = sum(
        bath_counts[name] * weights[name] for name in SPECIES
    )
    injected_total = sum(stream_counts[name] * weights[name] for name in SPECIES) + bath_total
    audit.update(
        {
            "macro_count": macro_count,
            "realization_id": realization_id,
            "stop_time_s": stop_time_s,
            "sampling_interval_s": sampling_interval_s,
            "species_macro_counts": stream_counts,
            "expected_continuous_real_particles": ndot_total * stop_time_s,
            "species_macro_weights": dict(weights),
            "bath_macro_counts": dict(bath_counts),
            "bath_injected_real_particles": bath_total,
            "bath_replacement_time_s": (
                bath_total / ndot_total if ndot_total > 0.0 else 0.0
            ),
            "injected_real_particles_total": injected_total,
            "seed_permutation": "splitmix64_fisher_yates_v1",
        }
    )
    return injected_total, audit


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "PARMA continuous-column source audit: print the birth-rate "
            "profile, spectral/angular moments, and the physical injection "
            "rate without launching WarpX."
        )
    )
    defaults = PARMA_CONTINUOUS_SOURCE_DEFAULTS
    parser.add_argument("--latitude-deg", type=float, default=defaults["latitude_deg"])
    parser.add_argument("--longitude-deg", type=float, default=defaults["longitude_deg"])
    parser.add_argument("--minimum-energy-eV", type=float,
                        default=defaults["minimum_energy_eV"])
    parser.add_argument("--source-radius-m", type=float,
                        default=defaults["source_radius_m"])
    parser.add_argument(
        "--table", type=Path, default=None,
        help="load this stored table instead of querying PARMA",
    )
    parser.add_argument(
        "--write-table", type=Path, default=None,
        help="also store the queried table at this path",
    )
    parser.add_argument(
        "--refine", type=int, default=1,
        help="grid refinement factor for convergence checks",
    )
    parser.add_argument("--probe", type=Path, help="rrea_interaction_table_probe "
                        "built next to warpx.rz (fresh queries)")
    args = parser.parse_args()
    if args.table is not None:
        table = load_parma_source_table(args.table)
    else:
        table = query_parma_table(
            latitude_deg=args.latitude_deg,
            longitude_deg=args.longitude_deg,
            minimum_energy_eV=args.minimum_energy_eV,
            maximum_energy_eV=SCHEMA6_CHARGED_ENERGY_MAX_EV,
            probe=args.probe,
            transport_config=ROOT / "rrea_transport_tables/schema6/production/transport_physics.json",
            refine=args.refine,
        )
        if args.write_table is not None:
            write_parma_source_table(table, args.write_table)
    prepared = PreparedParmaSource(table)
    audit = parma_source_audit(prepared, source_radius_m=args.source_radius_m)
    print(
        f"PARMA e-/e+ source, lat {table.latitude_deg:g} deg, "
        f"lon {table.longitude_deg:g} deg, solar {SOLAR_REFERENCE} "
        f"(W={table.solar_w_index:.4g}), rigidity {table.rigidity_gv:.4g} GV"
    )
    print(
        f"energy window [{table.minimum_energy_eV:.3e}, "
        f"{table.maximum_energy_eV:.3e}] eV (truncated to the RREA charged "
        "tables; PARMA's unsupported tail is excluded by construction)"
    )
    electron = audit["species"]["electron"]
    positron = audit["species"]["positron"]
    print(
        "altitude_km   birth_e-     birth_e+    mean_E_MeV(e-/e+)  "
        "mean_mu_down(e-/e+)"
    )
    for iz, alt in enumerate(audit["altitude_km"]):
        print(
            f"{alt:11.3f}  {electron['production_rate_cm3_s_per_altitude'][iz]:.5e}"
            f"  {positron['production_rate_cm3_s_per_altitude'][iz]:.5e}"
            f"  {electron['mean_energy_eV_per_altitude'][iz] / 1.0e6:8.3f}/"
            f"{positron['mean_energy_eV_per_altitude'][iz] / 1.0e6:<8.3f}"
            f"  {electron['mean_mu_per_altitude'][iz]:7.4f}/"
            f"{positron['mean_mu_per_altitude'][iz]:<7.4f}"
        )
    for name in SPECIES:
        entry = audit["species"][name]
        print(
            f"{name}: column birth rate "
            f"{entry['column_rate_cm2_s']:.6e} cm^-2 s^-1 -> "
            f"Ndot = {entry['physical_particles_per_s']:.6e} /s at radius "
            f"{args.source_radius_m:g} m"
        )
    total = audit["physical_particles_per_s_total"]
    print(f"total: Ndot = {total:.6e} particles/s")
    for name in SPECIES:
        entry = audit["species"][name]
        print(
            f"{name} bath at t=0 (n = phi/(beta c)): "
            f"{entry['physical_bath_particles']:.6e} particles"
        )
    bath_total = audit["physical_bath_particles_total"]
    print(
        f"bath total: {bath_total:.6e} particles; replaced by the stream in "
        f"{bath_total / total:.3e} s"
    )


if __name__ == "__main__":
    main()
