#!/usr/bin/env python3
"""Regenerate the generated block of quiz/rrea-quiz.html from this repository.

    python3 quiz/tools/build.py --probe <path to rrea_interaction_table_probe>

Everything the page shows that is not prose comes from here:
1. figure data, computed with the project's own functions (scripts/*) and the native
   table probe, which owns the ensemble drag (RreaInteractionTablesView::ElectronEnsembleDrag);
2. an excerpt for every 'file:line' or 'file:a-b' the authored page cites, repository files
   and the pinned upstream/WarpX-26.06 and upstream/AMReX-26.06 sources alike;
3. the output of every example in quiz/tools/examples.py, each run in a fresh namespace;
4. facts: values computed here, compared with the numbers the page quotes, each displayed
   string required in the questions that show it; quoted document text must still exist;
5. deterministic JSON between the generated markers, with provenance.
Any unresolved reference, failing example or drifted fact exits non-zero and names it.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAGE = ROOT / "quiz" / "rrea-quiz.html"
EXAMPLES = ROOT / "quiz" / "tools" / "examples.py"
TABLES = ROOT / "rrea_transport_tables" / "schema6" / "production"
TRANSPORT_CONFIG = TABLES / "transport_physics.json"
CLOSURE_CSV = (ROOT / "rrea_fluid_closure" / "dry_air_swarm_hybrid_v3"
               / "electron_flux_mobility_attachment_diffusion.csv")
BEGIN, END = "<!-- generated:begin -->", "<!-- generated:end -->"

sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "quiz" / "tools")]
import compute_rrea_em_radio as radio  # noqa: E402
import compute_rrea_optical_emissions as optics  # noqa: E402
import rrea_profiled_atmosphere as atm  # noqa: E402
import rrea_table_pack as tp  # noqa: E402
from rrea_run_support import load_rrea_defaults  # noqa: E402

C = 299792458.0
RHO_TABLE = 1.20479      # transport_physics.json reference_density_kg_m3 (checked below)
RHO_THRESHOLD = 1.293    # density the 276/284 kV/m thresholds are quoted at


def fail(msg: str) -> None:
    raise SystemExit(f"build.py: {msg}")


def logspace(a: float, b: float, n: int) -> list[float]:
    return [a * (b / a) ** (k / (n - 1)) for k in range(n)]


# ------------------------------------------------------------------ AMReX box layout
def amrex_base_grids(n_cell: list[int], max_grid_size: int, nprocs: int) -> list[list[int]]:
    """Level-0 box lengths per direction, as AMReX builds them.

    AmrMesh::MakeBaseGrids (upstream/AMReX-26.06/Src/AmrCore/AMReX_AmrMesh.cpp:524) coarsens
    an even-length domain by 2, applies BoxList::maxSize with max_grid_size/2
    (Src/Base/AMReX_BoxList.cpp:765) and refines back; ChopGrids only acts when there are
    fewer boxes than ranks. maxSize factors out the powers of two that chunk and length share,
    then splits into numblk nearly equal blocks, the first `extra` ones one unit longer.
    One domain box splits as a tensor product, so per-direction lengths describe it fully.
    This reproduces the 736-box BoxArray of a production WarpXHeader (32 x 23 boxes)."""
    out = []
    for length in n_cell:
        fac = 2 if length % 2 == 0 else 1
        nlen, bs, ratio = length // fac, max_grid_size // fac, 1
        if nlen <= bs:
            out.append([length])
            continue
        while bs % 2 == 0 and nlen % 2 == 0:
            ratio, bs, nlen = ratio * 2, bs // 2, nlen // 2
        numblk = (nlen + bs - 1) // bs
        sz = nlen // numblk
        extra = nlen - sz * numblk
        out.append([((sz + 1) if k < extra else sz) * ratio * fac for k in range(numblk)])
    if math.prod(len(s) for s in out) < nprocs:
        fail("ChopGrids would split further; not modelled")
    return out


# ------------------------------------------------------------------ figure data and facts
def probe_drag(probe: Path, energies_eV: list[float]) -> list[float]:
    """Ensemble electron drag (eV/m) at the table density, from the native probe."""
    run = subprocess.run([str(probe), str(TRANSPORT_CONFIG), "drag"],
                         input="\n".join(repr(e) for e in energies_eV) + "\n",
                         capture_output=True, text=True, check=True)
    vals = [float(l.split("=", 1)[1]) for l in run.stdout.splitlines() if l.startswith("drag_eV_per_m=")]
    if len(vals) != len(energies_eV) or not all(math.isfinite(v) and v > 0 for v in vals):
        fail(f"{probe}: bad drag output")
    return vals


def minimum(xs: list[float], ys: list[float]) -> tuple[float, float]:
    i = min(range(len(ys)), key=ys.__getitem__)
    return xs[i], ys[i]


def compute(probe: Path) -> tuple[dict, dict]:
    F: dict[str, float] = {}
    P: dict[str, object] = {}
    cfg = json.loads(TRANSPORT_CONFIG.read_text())
    if cfg["reference_density_kg_m3"] != RHO_TABLE:
        fail("table reference density changed")
    F["photon_cut_keV"] = cfg["brems_absolute_cutoff_eV"] / 1e3
    F["moller_threshold_keV"] = cfg["hard_moller_secondary_threshold_eV"] / 1e3
    F["charged_floor_keV"] = cfg["charged_energy_range_eV"][0] / 1e3

    # Drag: ensemble (probe) and realized = restricted collision + soft radiative (table).
    K = logspace(1e3, 1e9, 151)
    ens = probe_drag(probe, K)
    el = tp.load_columns(TABLES / "mc_tables" / "electron_transport_coefficients_air.csv.rtb")
    Kt = el["kinetic_energy_eV"]
    realized = [a + b for a, b in zip(el["collision_stopping_eV_per_m_stp"],
                                      el["soft_radiative_stopping_eV_per_m_stp"])]
    pick = [i for i in range(len(Kt)) if 1e3 <= Kt[i] <= 1e9][::12]
    P["drag"] = {"K_keV": [k / 1e3 for k in K], "ens_keV_m": [d / 1e3 for d in ens],
                 "Kt_keV": [Kt[i] / 1e3 for i in pick], "real_keV_m": [realized[i] / 1e3 for i in pick]}
    fine = logspace(4e5, 4e6, 801)
    kmin, dmin = minimum(fine, probe_drag(probe, fine))
    F["ens_min_MeV"] = kmin / 1e6
    F["ens_min_keV_m"] = dmin / 1e3
    F["ens_min_kV_m_1293"] = dmin / 1e3 * RHO_THRESHOLD / RHO_TABLE
    band = [k for k, d in zip(fine, probe_drag(probe, fine)) if d <= 1.005 * dmin]
    F["ens_flat_lo_MeV"], F["ens_flat_hi_MeV"] = band[0] / 1e6, band[-1] / 1e6
    kr, rmin = minimum(Kt, el["collision_stopping_eV_per_m_stp"])
    F["coll_min_MeV"], F["coll_min_keV_m"] = kr / 1e6, rmin / 1e3
    F["ens_at_50keV_kV_m"] = probe_drag(probe, [5e4])[0] / 1e3
    F["ens_100keV_keV_m"], F["ens_100MeV_keV_m"] = (d / 1e3 for d in probe_drag(probe, [1e5, 1e8]))
    # 50 keV force balance in reduced-field units, at the engine's number density
    # (Loschmidt n_L scaled by rho/rho_L), as RreaConstants.H reduced_field_td does.
    n_l, rho_l = 2.6868e25, 2.6868e25 * 28.96546e-3 / 6.02214076e23
    F["balance_50keV_Td"] = probe_drag(probe, [5e4])[0] / (1e-21 * n_l * RHO_TABLE / rho_l)
    F["rho_loschmidt"] = rho_l
    F["table_vs_loschmidt_pct"] = 100 * (rho_l / RHO_TABLE - 1)

    # Photon channels (inverse lengths at the table density).
    ph = tp.load_columns(TABLES / "mc_tables" / "photon_process_inverse_length_air.csv.rtb")
    E = ph["energy_eV"]
    chans = ["compton_inverse_m_stp", "photoelectric_inverse_m_stp",
             "pair_nuclear_inverse_m_stp", "pair_triplet_inverse_m_stp"]
    tot = [sum(ph[c][i] for c in chans) for i in range(len(E))]
    keep = [i for i in range(len(E)) if 1e4 <= E[i] <= 1e10]
    P["photon"] = {"E_keV": [E[i] / 1e3 for i in keep], "mfp_m": [1 / tot[i] for i in keep],
                   "compton": [ph[chans[0]][i] / tot[i] for i in keep],
                   "pe": [ph[chans[1]][i] / tot[i] for i in keep],
                   "pair": [(ph[chans[2]][i] + ph[chans[3]][i]) / tot[i] for i in keep]}
    i1 = E.index(1e6)
    F["photon_mfp_1MeV_m"] = 1 / tot[i1]
    F["photon_mfp_1MeV_12km_m"] = None  # filled after the density profile is loaded
    pair = [ph[chans[2]][i] + ph[chans[3]][i] for i in range(len(E))]

    def crossing(a: list[float], b: list[float]) -> float:
        """Energy where channel a stops exceeding channel b (log-log interpolation)."""
        for i in range(len(E) - 1):
            if a[i] > 0 and b[i] > 0 and a[i + 1] > 0 and b[i + 1] > 0:
                g0, g1 = math.log(a[i] / b[i]), math.log(a[i + 1] / b[i + 1])
                if g0 > 0 >= g1:
                    return math.exp(math.log(E[i]) + g0 / (g0 - g1) * math.log(E[i + 1] / E[i]))
        fail("no channel crossing")
    F["pe_compton_cross_keV"] = crossing(ph[chans[1]], ph[chans[0]]) / 1e3
    F["compton_pair_cross_MeV"] = crossing(ph[chans[0]], pair) / 1e6

    # Atmosphere, thresholds and the Coleman-Dwyer integral (276-based interval, both T).
    field = atm.load_field_profile()
    dens = atm.load_density_ratio_profile(atm.DEFAULT_DENSITY_PROFILE, reference_density_kg_m3=RHO_THRESHOLD)
    thr = atm.load_threshold_profile(atm.DEFAULT_DENSITY_PROFILE, threshold_stp_kv_per_m=276.0)
    z_lo, z_hi = atm.ALTITUDE_AT_Z0_M, atm.ALTITUDE_AT_Z0_M + atm.DOMAIN_HEIGHT_M
    # Field and density are both piecewise linear: their knots plus the domain ends sample them exactly.
    zs = sorted({z_lo, z_hi, *(z for z in (*field.altitude_m, *dens.altitude_m) if z_lo < z < z_hi)})
    P["profile"] = {"z_km": [z / 1e3 for z in zs],
                    "shape": [atm.profiled_abs_field_v_per_m(field, 1.0, z) for z in zs],
                    "n": [dens.interpolate(z) for z in zs]}
    for alt in (6000, 10000, 12000, 16000):
        F[f"n_{alt // 1000}km"] = dens.interpolate(alt)
    F["thr276_16km"] = 276.0 * F["n_16km"]
    F["dragmin_16km"] = F["ens_min_keV_m"] * F["n_16km"] * RHO_THRESHOLD / RHO_TABLE
    F["chi_12km"] = F["n_12km"] * RHO_THRESHOLD / RHO_TABLE
    F["photon_mfp_1MeV_12km_m"] = F["photon_mfp_1MeV_m"] / F["chi_12km"]
    F["lambda_12km_140_m"] = 7300.0 / (140.0 - 276.0 * F["n_12km"])
    F["field_peak_km"] = max(field.altitude_m, key=lambda z: abs(field.interpolate(z))) / 1e3

    cd = []
    for e0 in range(100, 161):
        ref = atm.coleman_dwyer_profile_reference(field=field, density_ratio=dens, threshold_profile=thr,
                                                  e0_peak_v_per_m=e0 * 1e3)
        if ref is None:
            cd.append(None)
            continue
        region = atm.AboveThresholdRegion(**ref["above_threshold_region"])
        s276 = ref["curves"]["276"]["predicted_avalanche_lengths"]
        s284 = atm.estimate_integrated_avalanche_lengths(
            field=field, density_ratio=dens, region=region, e0_peak_v_per_m=e0 * 1e3,
            runaway_threshold_stp_kv_per_m=284.0).predicted_avalanche_lengths
        cd.append({"e0": e0, "z0": region.injection_altitude_m / 1e3, "z1": region.top_altitude_m / 1e3,
                   "s276": s276, "s284": s284, "ratio": ref["curves"]["276"]["max_field_to_threshold_ratio"]})
        if e0 == 125:
            F["cd125_s276"], F["cd125_s284"] = s276, s284
            F["cd125_m276"], F["cd125_m284"] = math.exp(s276), math.exp(s284)
            F["cd125_z0_km"], F["cd125_z1_km"] = cd[-1]["z0"], cd[-1]["z1"]
            F["cd125_ratio"] = cd[-1]["ratio"]
            F["cd125_inj_n"] = region.density_ratio_at_injection
            F["cd125_inj_lambda_m"] = 7300.0 / (region.field_at_injection_v_per_m / 1e3
                                               - 276.0 * region.density_ratio_at_injection)
    P["cd"] = cd
    F["cd_first_e0"] = next(r["e0"] for r in cd if r)

    # Radial taper: production r_start = 0.2 R, r_end = R.
    cap = load_rrea_defaults()["capture_defaults"]
    R = cap["domain_radius_m"]
    r0, r1 = cap["field_taper_start_fraction"] * R, R
    rs = [R * k / 200 for k in range(201)]
    P["taper"] = {"r_km": [r / 1e3 for r in rs], "f": [atm.taper_weight(r, r0, r1) for r in rs],
                  "slope": [-atm.taper_weight_deriv(r, r0, r1) * (r1 - r0) for r in rs]}
    F["taper_peak_slope"] = max(P["taper"]["slope"])
    audit = atm.audit_ambient_field(field=field, threshold=thr, e0_peak_v_per_m=125e3,
                                    domain_radius_m=R, taper_r_start_m=r0, taper_r_end_m=r1)
    F["taper_span_MV"] = audit["column_potential_span_v"] / 1e6
    F["taper_fringe_kV_m"] = audit["max_fringe_er_v_per_m"] / 1e3
    F["fringe_thr_ratio"] = audit["max_fringe_er_over_threshold"]

    # Fluid closure (reduced coefficients at Loschmidt density).
    rows = [l for l in CLOSURE_CSV.read_text().splitlines() if l.strip() and not l.startswith("#")]
    head = rows[0].split(",")
    tab = [dict(zip(head, r.split(","))) for r in rows[1:]]
    tab = [t for t in tab if float(t["en_td"]) >= 0.1]
    P["closure"] = {"en": [float(t["en_td"]) for t in tab],
                    "w": [float(t["flux_drift_velocity_m_per_s"]) for t in tab],
                    "nu2": [float(t["two_body_frequency_ref_s"]) for t in tab],
                    "nu3": [float(t["three_body_frequency_ref_s"]) for t in tab]}
    F["closure_max_td"] = max(P["closure"]["en"])

    # Optical yields along the column (dry-air AIRFLY law, Niemi 777.4 nm quenching).
    prof = optics.load_profile(atm.DEFAULT_DENSITY_PROFILE)
    alts = [6000 + 250 * k for k in range(41)]
    import numpy as np
    n = np.interp(alts, prof["alt_m"], prof["n_m3"])
    T = np.interp(alts, prof["alt_m"], prof["temp_k"])
    p_hpa = n * optics.K_BOLTZ * T / 100
    P["optics"] = {"z_km": [a / 1e3 for a in alts],
                   "fy337": [float(v) for v in optics.fy337_ph_per_mev(p_hpa, T)],
                   "q777": [float(v) for v in optics.quench_factor_777(n)]}
    i12 = alts.index(12000)
    F["fy337_12km"] = P["optics"]["fy337"][i12]
    F["fy337_6km"] = P["optics"]["fy337"][0]
    F["q777_12km"] = P["optics"]["q777"][i12]
    F["p_12km_hpa"], F["p_6km_hpa"] = float(p_hpa[i12]), float(p_hpa[0])

    # Mesh, time step, Courant numbers, box layout, radio band. The deck's geometry is a
    # placeholder; the driver writes these from the defaults (run_rrea_profiled_video_capture.py).
    nr, nz = round(R / cap["cell_size_m"]), int(cap["n_cell_z"])
    dr, dz, dt = R / nr, cap["domain_height_m"] / nz, cap["dt_s"]
    mgs = int(cap["max_grid_size"])
    ranks = int(load_rrea_defaults()["olivia_layout"]["mpi_ranks"])
    F.update(dr_m=dr, dz_m=dz, c_dt_m=C * dt, nr=nr, nz=nz,
             courant_2d=C * dt * math.sqrt(1 / dr**2 + 1 / dz**2),
             warpx_rz_dtmax_ns=1e9 / (C * math.sqrt((1 + 0.2105) / dr**2 + 1 / dz**2)))
    segs = amrex_base_grids([nr, nz], mgs, ranks)
    P["boxes"] = {"r": segs[0], "z": segs[1], "nr": nr, "nz": nz}
    F["boxes_total"] = len(segs[0]) * len(segs[1])
    F["boxes_r"], F["boxes_z"] = len(segs[0]), len(segs[1])
    F["boxes_per_rank"] = F["boxes_total"] / ranks
    F["ranks"] = ranks
    F["radio_max_kHz"] = radio.resolved_max_hz((nr - 8, nz - 16, dr, dz), dt, 256) / 1e3
    # Hard-event optical depth of a 1 MeV electron over one step (straight chord of length v*dt).
    i1 = Kt.index(1e6)
    gamma = 1 + 1e6 / 510998.95
    beta = math.sqrt(1 - 1 / gamma**2)
    mu = el["hard_moller_inverse_m_stp"][i1] + el["tracked_brems_inverse_m_stp"][i1]
    F["tau_1MeV"] = mu * C * beta * dt
    F["tau_1MeV_12km"] = F["tau_1MeV"] * F["chi_12km"]
    F["u_1MeV_m_s"] = gamma * beta * C
    F["max_grid_size"] = mgs
    F["stop_time_us"] = cap["stop_time_s"] * 1e6
    F["light_cross_domain_us"] = R / C * 1e6
    return F, P


# ------------------------------------------------------------------ excerpts
REF = re.compile(r"""['"]((?:[\w.+-]+/)+[\w.+-]+|[\w-]+\.(?:md|py|json|txt)):(\d+)(?:-(\d+))?['"]""")
CPP_HEAD = re.compile(r"^\s{0,4}(?!if\b|for\b|while\b|switch\b|else\b|return\b|do\b|case\b)"
                      r"[\w\[\]:<>~&*,\s]*?(?:\b(?:struct|class|enum)\b[^;]*|\w[\w:~<>]*\s*\([^;]*)$")


def block_end(lines: list[str], i: int, path: str) -> int:
    """Last line (0-based) of the definition starting at line i, or -1 if line i starts none."""
    src = lines[i]
    if path.endswith(".py"):
        if not re.match(r"\s*(?:async\s+)?(?:def|class)\s", src):
            return -1
        ind = len(src) - len(src.lstrip())
        end, j = i, i + 1
        while j < len(lines):
            t = lines[j]
            if t.strip():
                if len(t) - len(t.lstrip()) <= ind and not t.lstrip().startswith(")"):
                    break
                end = j
            j += 1
        return end
    if path.endswith((".H", ".cpp", ".h")) and CPP_HEAD.match(src):
        depth, opened = 0, False
        for j in range(i, min(len(lines), i + 400)):
            code = re.sub(r"//.*|\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])'", "", lines[j])
            if not opened and ";" in code and "{" not in code:
                return -1
            for ch in code:
                if ch == "{":
                    depth, opened = depth + 1, True
                elif ch == "}":
                    depth -= 1
                    if opened and depth == 0:
                        return j
        return -1
    return -1


def excerpts(authored: str) -> tuple[dict, dict]:
    ex, files = {}, {}
    for m in REF.finditer(authored):
        ref, path, a, b = m.group(0)[1:-1], m.group(1), int(m.group(2)), m.group(3)
        f = ROOT / path
        if not f.is_file():
            fail(f"reference {ref}: no such file")
        lines = f.read_text(encoding="utf-8", errors="replace").splitlines()
        if not 1 <= a <= len(lines) or (b and not a <= int(b) <= len(lines)):
            fail(f"reference {ref}: line out of range ({len(lines)} lines)")
        more, hl = 0, [a]
        if b:
            lo, hi, hl = a, int(b), []
        else:
            end = block_end(lines, a - 1, path)
            if end >= 0:
                lo, hi = a, end + 1
                if hi - lo > 29:
                    more, hi = hi - (lo + 29), lo + 29
            else:
                lo, hi = max(1, a - 2), min(len(lines), a + 2)
        ex[ref] = {"file": path, "from": lo, "to": hi, "more": more, "hl": hl}
        store = files.setdefault(path, {})
        for k in range(lo, hi + 1):
            store[str(k)] = lines[k - 1].rstrip().replace("\t", "    ")
    return ex, files


# ------------------------------------------------------------------ examples
MARK = re.compile(r"^# --- (\S+): (.+)$", re.M)


def examples(probe: Path) -> dict:
    text = EXAMPLES.read_text(encoding="utf-8")
    marks = list(MARK.finditer(text))
    out = {}
    os.environ["RREA_PROBE"] = str(probe)
    cwd = os.getcwd()
    os.chdir(ROOT)
    try:
        for k, m in enumerate(marks):
            code = text[m.end():marks[k + 1].start() if k + 1 < len(marks) else len(text)].strip("\n")
            buf = io.StringIO()
            try:
                with contextlib.redirect_stdout(buf):
                    exec(compile(code, f"examples.py:{m.group(1)}", "exec"), {"__name__": "__quiz__"})
            except Exception as err:  # noqa: BLE001
                fail(f"example {m.group(1)} failed: {err!r}")
            out[m.group(1)] = {"title": m.group(2).strip(), "code": code.rstrip(), "output": buf.getvalue().rstrip()}
    finally:
        os.chdir(cwd)
    return out


# ------------------------------------------------------------------ facts bound to the page
# (fact, quoted value, relative tolerance, "qid, qid" or "page", strings those questions must show)
CHECKS: list[tuple[str, float, float, str, list[str]]] = [
    ("ens_min_keV_m", 201, 0.005, "big2, big3", ["201 keV/m"]),
    ("ens_min_MeV", 1.15, 0.01, "big2, big3", ["1.15 MeV"]),
    ("ens_min_kV_m_1293", 216, 0.005, "big3", ["216 kV/m"]),
    ("ens_flat_lo_MeV", 0.93, 0.01, "big3", ["0.93"]),
    ("ens_flat_hi_MeV", 1.46, 0.01, "big3", ["1.46 MeV"]),
    ("ens_100keV_keV_m", 441, 0.005, "big2", ["441 keV/m"]),
    ("ens_100MeV_keV_m", 581, 0.005, "big2", ["581 keV/m"]),
    ("coll_min_MeV", 1.65, 0.01, "big3", ["1.65 MeV"]),
    ("n_12km", 0.2606, 0.001, "big10, tab4", ["0.2606"]),
    ("n_12km", 0.26, 0.01, "big9, big10, tab4", ["0.26"]),
    ("lambda_12km_140_m", 107, 0.005, "big10", ["107 m"]),
    ("courant_2d", 0.185, 0.005, "pic5", ["0.185"]),
    ("warpx_rz_dtmax_ns", 12.7, 0.005, "pic6", ["12.7 ns"]),
    ("u_1MeV_m_s", 8.3e8, 0.01, "pic8", ["8.3 × 10⁸ m/s"]),
    ("boxes_total", 736, 0.0, "amr1-4, amr1-10, amr2-1", ["736"]),
    ("boxes_per_rank", 92, 0.0, "page", ["92 per rank"]),
    ("ranks", 8, 0.0, "amr1-10", ["8 OLIVIA ranks"]),
    ("stop_time_us", 100, 0.0, "wx2-8", ["100 µs"]),
    ("light_cross_domain_us", 33, 0.02, "wx2-8", ["33 µs"]),
    ("tau_1MeV", 0.133, 0.005, "kin2", ["0.133"]),
    ("tau_1MeV_12km", 0.037, 0.02, "kin2", ["0.037"]),
    ("chi_12km", 0.28, 0.01, "kin2", ["0.28"]),
    ("chi_12km", 0.280, 0.002, "tab4", ["0.280"]),
    ("photon_mfp_1MeV_m", 130.8, 0.001, "tab4", ["130.8 m"]),
    ("photon_mfp_1MeV_12km_m", 468, 0.005, "tab4", ["468 m"]),
    ("pe_compton_cross_keV", 29, 0.02, "tab5, tab9", ["29 keV"]),
    ("compton_pair_cross_MeV", 22, 0.02, "tab5, tab9", ["22 MeV"]),
    ("table_vs_loschmidt_pct", 7, 0.05, "tab3", ["7%"]),
    ("taper_fringe_kV_m", 37, 0.02, "fld10", ["37 kV/m"]),
    ("taper_span_MV", 318, 0.002, "fld10", ["318 MV"]),
    ("taper_peak_slope", 1.875, 1e-6, "fld10", ["1.875"]),
    ("fringe_thr_ratio", 0.91, 0.01, "fld10", ["91%"]),
    ("thr276_16km", 41, 0.01, "fld10", ["41 kV/m"]),
    ("dragmin_16km", 32, 0.01, "fld10", ["32 kV/m"]),
    ("cd125_s276", 10.87, 0.002, "val2", ["10.87", "10.9"]),
    ("cd125_s284", 10.23, 0.002, "val2", ["10.23", "10.2"]),
    ("cd125_m276", 5.2e4, 0.01, "val2", ["5.2 × 10⁴"]),
    ("cd125_m284", 2.8e4, 0.02, "val2", ["2.8 × 10⁴"]),
    ("cd125_s276", 10.9, 0.005, "val4", ["10.9"]),
    ("cd125_s284", 10.2, 0.005, "val4", ["10.2"]),
    ("cd125_ratio", 1.75, 0.005, "val4", ["1.75 times"]),
    ("cd125_z0_km", 10.90, 0.001, "page", ["10.90 to 13.16 km"]),
    ("cd125_z1_km", 13.16, 0.001, "page", ["10.90 to 13.16 km"]),
    ("radio_max_kHz", 821, 0.002, "obs4", ["821 kHz"]),
    ("fy337_12km", 19.9, 0.005, "obs5", ["19.9"]),
    ("fy337_6km", 10.4, 0.005, "obs5", ["10.4"]),
    ("p_12km_hpa", 214, 0.005, "obs5", ["214 hPa"]),
    ("p_6km_hpa", 494, 0.005, "obs5", ["494 hPa"]),
    ("q777_12km", 0.008, 0.05, "obs7", ["0.8%"]),
]
# (file, verbatim text that must still be there, "qid, qid" or "page", strings shown)
QUOTES: list[tuple[str, str, str, list[str]]] = [
    ("docs/RREA_WARPX_AMREX_ARCHITECTURE_GUIDE.tex", "$218~\\mathrm{kV/m}$", "big3", ["218 kV/m"]),
    ("native_amrex_rrea/include/rrea/RreaConstants.H", "mean_energy_per_ion_pair_air_eV = 33.97;", "big8", ["33.97 eV"]),
    ("docs/RREA_WARPX_AMREX_ARCHITECTURE_GUIDE.tex", "$7.04 \\pm 0.24$ photons/MeV at 800\\,hPa, 293\\,K", "page", ["7.04 ± 0.24 photons/MeV at 800 hPa and 293 K"]),
    ("docs/RREA_WARPX_AMREX_ARCHITECTURE_GUIDE.tex", "$23.118\\pm0.118$", "val7", ["23.118 ± 0.118"]),
    ("docs/RREA_WARPX_AMREX_ARCHITECTURE_GUIDE.tex", "$23.033\\pm0.120$", "val7", ["23.033 ± 0.120"]),
    ("docs/RREA_WARPX_AMREX_ARCHITECTURE_GUIDE.tex", "are 27.605 (276~kV/m) and 26.110 (284~kV/m)", "val7", ["27.605", "26.110"]),
    ("docs/RREA_WARPX_AMREX_ARCHITECTURE_GUIDE.tex", "respectively 0.9465 and 0.9627", "val7", ["0.9465", "0.9627"]),
    ("docs/RREA_WARPX_AMREX_ARCHITECTURE_GUIDE.tex", "$m_ec^2/2 = 255$", "tab9", ["255 keV"]),
    ("docs/RREA_WARPX_AMREX_ARCHITECTURE_GUIDE.tex", "inset eight cells", "obs2", ["eight cells"]),
    ("AGENTS.md", "Kinetic photon transport deliberately excludes coherent Rayleigh scattering.", "page", ["Rayleigh scattering is deliberately excluded"]),
    ("upstream/AMReX-26.06/Src/AmrCore/AMReX_AmrMesh.H", "max_grid_size {{IntVect(AMREX_D_PICK(128,128,64))}}", "amr1-9", ["128 cells"]),
    ("upstream/AMReX-26.06/Src/AmrCore/AMReX_AmrMesh.H", "max_grid_size {{IntVect(AMREX_D_PICK(128,128,32))}}", "amr1-9", ["128 cells"]),
    ("config/rrea_defaults.json", '"video_frame_interval_s": 1E-06', "obs9", ["1 µs"]),
    ("cases/rrea/production/inputs", "rrea.transport_max_subcycles = 8192", "kin2", ["8192"]),
]


def qblock(authored: str, qid: str) -> str:
    m = re.search(r"\{ id: '" + re.escape(qid) + r"',", authored)
    if not m:
        fail(f"no question {qid}")
    nxt = re.compile(r"\{ id: '").search(authored, m.end())
    return authored[m.start(): nxt.start() if nxt else len(authored)]


def where(authored: str, scope: str) -> list[tuple[str, str]]:
    if scope == "page":
        return [("page", authored)]
    return [(q, qblock(authored, q)) for q in (s.strip() for s in scope.split(","))]


def check_facts(F: dict, authored: str) -> None:
    for name, quoted, rtol, scope, shown in CHECKS:
        if name not in F:
            fail(f"unknown fact {name}")
        if not abs(F[name] / quoted - 1) <= rtol:
            fail(f"fact {name} = {F[name]:.6g} drifted from the quoted {quoted:g} (rtol {rtol}) in {scope}")
        for qid, block in where(authored, scope):
            for s in shown:
                if s not in block:
                    fail(f"{qid}: expected to show {s!r} for fact {name}")
    for path, verbatim, scope, shown in QUOTES:
        if verbatim not in (ROOT / path).read_text(encoding="utf-8"):
            fail(f"{path} no longer contains {verbatim!r}")
        for qid, block in where(authored, scope):
            for s in shown:
                if s not in block:
                    fail(f"{qid}: expected to show {s!r} quoted from {path}")


# ------------------------------------------------------------------ output
def rounded(x):
    if isinstance(x, float):
        if not math.isfinite(x):
            fail("non-finite value in generated data")
        return float(f"{x:.6g}")
    if isinstance(x, dict):
        return {k: rounded(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [rounded(v) for v in x]
    return x


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--probe", type=Path, required=True, help="rrea_interaction_table_probe executable")
    args = ap.parse_args()
    html = PAGE.read_text(encoding="utf-8")
    if html.count(BEGIN) != 1 or html.count(END) != 1:
        fail("generated markers missing or duplicated")
    head, rest = html.split(BEGIN)
    tail = rest.split(END)[1]
    authored = head + tail
    F, P = compute(args.probe)
    ex, files = excerpts(authored)
    snippets = examples(args.probe)
    for sid in set(re.findall(r"snippet: '([\w-]+)'", authored)) - set(snippets):
        fail(f"page names example {sid}, absent from {EXAMPLES.name}")
    check_facts(F, authored)
    commit = git("rev-parse", "--short=7", "HEAD")
    dirty = bool(git("status", "--porcelain", "--", ".", ":(exclude)quiz"))
    meta = {"commit": commit + (" + local changes" if dirty else ""),
            "date": git("show", "-s", "--format=%cs", "HEAD"),
            "python": ".".join(map(str, sys.version_info[:3])),
            "transport_model": json.loads(TRANSPORT_CONFIG.read_text())["transport_model"]}
    data = rounded({"meta": meta, "facts": F, "plots": P, "excerpts": ex, "files": files, "snippets": snippets})
    blob = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    blob = blob.replace("</", "<\\/")
    block = f'{BEGIN}\n<script id="quiz-generated" type="application/json">{blob}</script>\n{END}'
    new = head + block + tail
    if new != html:
        PAGE.write_text(new, encoding="utf-8")
    print(f"build.py: {len(ex)} excerpts from {len(files)} files, {len(snippets)} examples, "
          f"{len(CHECKS)} fact checks, {len(QUOTES)} quoted texts; {len(blob) / 1024:.0f} KiB generated; "
          f"commit {meta['commit']}")


if __name__ == "__main__":
    main()
