"""Runnable examples shown in the RREA quiz ("Try it" boxes).

Run them all from the repository root:

    PYTHONPATH=scripts:quiz/tools RREA_PROBE=<build>/bin/rrea_interaction_table_probe \
        python3 quiz/tools/examples.py

quiz/tools/build.py runs each block below (split at '# --- id: title') in a fresh
namespace from the repository root and embeds the code and its output in the page.
"""

# --- drag: Where the ensemble drag is weakest
import os
import subprocess

probe = os.environ["RREA_PROBE"]  # <build>/bin/rrea_interaction_table_probe
config = "rrea_transport_tables/schema6/production/transport_physics.json"
energies = [1e5 * 10 ** (k / 200) for k in range(601)]  # 100 keV ... 100 MeV
out = subprocess.run([probe, config, "drag"], input="\n".join(map(repr, energies)),
                     capture_output=True, text=True, check=True).stdout.splitlines()
drag = [float(line.split("=")[1]) for line in out if line.startswith("drag_eV_per_m=")]  # eV/m, 1.20479 kg/m3
k = min(range(len(drag)), key=drag.__getitem__)
print(f"minimum ensemble drag: {drag[k] / 1e3:.1f} keV/m at {energies[k] / 1e6:.2f} MeV")
print(f"the same air at 1.293 kg/m3: {drag[k] / 1e3 * 1.293 / 1.20479:.0f} kV/m")
print(f"at 100 keV: {drag[0] / 1e3:.0f} keV/m, at 100 MeV: {drag[-1] / 1e3:.0f} keV/m")

# --- courant: The production time step against the mesh
import math

from rrea_run_support import load_rrea_defaults

cap = load_rrea_defaults()["capture_defaults"]  # the driver writes geometry and dt from here
nr, nz = round(cap["domain_radius_m"] / cap["cell_size_m"]), cap["n_cell_z"]
dr, dz, dt = cap["domain_radius_m"] / nr, cap["domain_height_m"] / nz, cap["dt_s"]
c = 299792458.0
print(f"{nr} x {nz} cells: dr = {dr:g} m, dz = {dz:.3f} m, c*dt = {c * dt:.3f} m")
print(f"2D Yee Courant number c*dt*sqrt(1/dr^2 + 1/dz^2) = {c * dt * math.hypot(1 / dr, 1 / dz):.3f}")
alpha = 0.2105  # WarpX CylindricalYeeAlgorithm::ComputeMaxDt, one azimuthal mode (m = 0)
dt_rz = 1 / (c * math.sqrt((1 + alpha) / dr**2 + 1 / dz**2))
print(f"WarpX RZ Yee limit {dt_rz * 1e9:.2f} ns; the run uses {dt * 1e9:g} ns")

# --- boxes: How AMReX chops the production mesh into boxes
from collections import Counter

from build import amrex_base_grids  # port of AmrMesh::MakeBaseGrids + BoxList::maxSize

r, z = amrex_base_grids([2000, 1440], max_grid_size=64, nprocs=8)
print("radial box lengths:", dict(Counter(r)), " axial:", dict(Counter(z)))
print(f"{len(r)} x {len(z)} = {len(r) * len(z)} boxes, {len(r) * len(z) / 8:g} per rank on average over 8 ranks")
print("cells covered:", sum(r) * sum(z), "=", 2000 * 1440)

# --- deck: What the production deck asks of WarpX
deck = {}
for line in open("cases/rrea/production/inputs"):
    line = line.split("#", 1)[0]
    if "=" in line:
        key, value = line.split("=", 1)
        deck[key.strip()] = " ".join(value.split())
for name in deck["particles.species_names"].split():
    print(f"{name:15s} type={deck[name + '.species_type']:8s} injection_style={deck[name + '.injection_style']}")
for key in ("warpx.do_electrostatic", "warpx.poisson_solver", "algo.particle_pusher",
            "algo.particle_shape", "algo.current_deposition", "boundary.field_hi"):
    print(f"{key} = {deck[key]}")

# --- ordering: Patch a copy of WarpXEvolve.cpp and find the hooks
import shutil
import tempfile
from pathlib import Path

from prepare_rrea_warpx_source import patch_evolve

src = Path("upstream/WarpX-26.06/Source/Evolve/WarpXEvolve.cpp")
with tempfile.TemporaryDirectory() as tmp:
    copy = Path(tmp) / src.name
    shutil.copy(src, copy)
    patch_evolve(copy)  # also asserts the ordering contract
    once = copy.read_text()
    patch_evolve(copy)  # idempotent: the second run must change nothing
    print("second run changed the file:", copy.read_text() != once)
lines = once.splitlines()
for marker in ("InjectScheduledSeeds(", "CaptureRreaPositionsBeforePush(", "OneStep(cur_time",
               "AfterParticlePushBeforeBoundaries(", "mypc->doResampling(",
               "HandleParticlesAtBoundaries(step", "SpaceChargeSolvePurpose::Evolution"):
    print(f"line {next(i for i, t in enumerate(lines, 1) if marker in t):4d}: {marker}")

# --- tau: Optical depth of one step, and the substep guard
import math

import rrea_table_pack as tp

table = tp.load_columns("rrea_transport_tables/schema6/production/mc_tables/"
                        "electron_transport_coefficients_air.csv.rtb")
i = table["kinetic_energy_eV"].index(1e6)
mu = table["hard_moller_inverse_m_stp"][i] + table["tracked_brems_inverse_m_stp"][i]  # 1/m at 1.20479 kg/m3
chord = 299792458.0 * 0.941 * 2.5e-9  # metres a 1 MeV electron (beta = 0.941) covers in 2.5 ns
for chi in (1.0, 0.28):  # rho/rho0: table density, and near 12 km
    tau = mu * chi * chord
    print(f"rho/rho0 = {chi}: tau = {tau:.4f} -> {max(1, math.ceil(tau / 0.05))} substep(s) at 0.05 per substep, "
          f"P(at least one hard event) = {1 - math.exp(-tau):.4f}")

# --- photon: How far a 1 MeV photon flies, and which channel wins
import rrea_table_pack as tp

ph = tp.load_columns("rrea_transport_tables/schema6/production/mc_tables/"
                     "photon_process_inverse_length_air.csv.rtb")
i = ph["energy_eV"].index(1e6)
rates = {c.split("_inverse")[0]: ph[c][i] for c in ph if c.endswith("_inverse_m_stp")}
total = sum(rates.values())
print(f"1 MeV photon, mean free path {1 / total:.1f} m at 1.20479 kg/m3")
print("channel shares:", {name: round(rate / total, 4) for name, rate in rates.items()})
print(f"at rho/rho0 = 0.28 (about 12 km): {1 / (total * 0.28):.0f} m")

# --- taper: The circulation constraint of the radial taper
from rrea_profiled_atmosphere import taper_weight, taper_weight_deriv

r0, r1 = 2000.0, 10000.0  # production: taper from 0.2 R to R, R = 10 km
slopes = [-taper_weight_deriv(r0 + (r1 - r0) * k / 1000, r0, r1) for k in range(1001)]
print(f"max |f'| x width = {max(slopes) * (r1 - r0):.4f} (quintic smoothstep peak slope)")
print("f at 2, 4, 6, 8, 10 km:", [round(taper_weight(r, r0, r1), 4) for r in (2e3, 4e3, 6e3, 8e3, 1e4)])
span = 318e6  # V, column potential span of the 125 kV/m profile
print(f"fringe |Er| ~ 1.875 * (span/2) / width = {1.875 * span / 2 / (r1 - r0) / 1e3:.1f} kV/m")

# --- ceilings: Population-control keys the driver writes
from rrea_cd_physics import cd_population_ceilings
from rrea_run_support import population_overrides

# split_min_weight is illustrative: the driver passes the mean weight of a source macroparticle.
for key in population_overrides(population_target=400000, control_interval=10,
                                split_min_weight=4909.8, strict_ceilings=None):
    print(key)
print("C&D unit-weight cohort, 1e5 seeds, M_276 = 5.2e4:", cd_population_ceilings(100000, 5.2e4))

# --- cd: Coleman-Dwyer e-foldings along the real profile
import rrea_profiled_atmosphere as atm

field = atm.load_field_profile()
density = atm.load_density_ratio_profile(atm.DEFAULT_DENSITY_PROFILE, reference_density_kg_m3=1.293)
threshold = atm.load_threshold_profile(atm.DEFAULT_DENSITY_PROFILE, threshold_stp_kv_per_m=276.0)
ref = atm.coleman_dwyer_profile_reference(field=field, density_ratio=density,
                                          threshold_profile=threshold, e0_peak_v_per_m=125e3)
region = atm.AboveThresholdRegion(**ref["above_threshold_region"])
print(f"interval {region.injection_altitude_m:.0f} -> {region.top_altitude_m:.0f} m MSL")
for t in (276.0, 284.0):
    est = atm.estimate_integrated_avalanche_lengths(field=field, density_ratio=density, region=region,
                                                    e0_peak_v_per_m=125e3, runaway_threshold_stp_kv_per_m=t)
    print(f"T = {t:.0f} kV/m: {est.predicted_avalanche_lengths:.2f} e-folds, M = {est.predicted_multiplication:.3g}")

# --- optics: 337 nm yield along the column and the radio band limit
import numpy as np

import compute_rrea_em_radio as radio
import compute_rrea_optical_emissions as optics
import rrea_profiled_atmosphere as atm

prof = optics.load_profile(atm.DEFAULT_DENSITY_PROFILE)
for alt in (6000.0, 9000.0, 12000.0, 15000.0):
    n = np.interp(alt, prof["alt_m"], prof["n_m3"])
    t = np.interp(alt, prof["alt_m"], prof["temp_k"])
    p_hpa = n * optics.K_BOLTZ * t / 100
    print(f"{alt / 1e3:4.0f} km: {p_hpa:5.0f} hPa, {t:5.1f} K -> {float(optics.fy337_ph_per_mev(p_hpa, t)):5.2f} photons/MeV")
print(f"resolved radio band: {radio.resolved_max_hz((1992, 1424, 5.0, 10000 / 1440), 2.5e-9, 256) / 1e3:.0f} kHz")
