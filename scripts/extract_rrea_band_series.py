"""Band lightcurves, channel-field and gain reductions, from the frame stream.

The capture writes ``n_photon_ge_threshold_m3`` and
``n_energetic_e_ge_threshold_m3`` on the whole RZ mesh every frame, but no
engine diagnostic reduces either by altitude: ``rrea_reduced.csv`` carries
domain totals only, and the plane-flux diagnostic is bound to the
Coleman-Dwyer seeding model, so a production profiled capture has none.  This
reduces one altitude band of the mesh (a thin slab, or everything above an
altitude) to a single number per frame, per species, and writes small CSVs
next to the other analysis products.

The normal figure workflow reduces the frame stream on the capture
host. ``olivia.py pull-capture`` fetches the small CSVs as optional sidecars;
an unreduced capture still pulls cleanly.

    python scripts/extract_rrea_band_series.py <capture_dir>

writes both species at their default bands in one pass. For the band
reductions each frame contributes one contiguous block of mesh rows per
species. The channel-field and gain
reductions (``--channel-field``, ``--gain-integral``, ``--gain-profile``,
``--gain-weighted``) read whole frames instead, and ``--gain-weighted`` reads
them twice to build its time averages, so they cost what a render costs.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

from render_rrea_profiled_video import frame_array_indices, load_frames

# One altitude band per species, echoed into every row. Keyed by the
# frame_array_indices key, which IS the species name, so there is no second
# name to keep in step: (z_lo, z_hi, r_max, output CSV) in m MSL / m, inf
# meaning the mesh edge; the rows record the edges actually integrated. Both
# grids are the >= 1 MeV populations, the same cut the renderer's density
# panels draw.
# Both species: everything above 12 km MSL over the whole radius -- the source
# an observer above the cloud sees, not a sample at one altitude. A thin slab
# is the wrong instrument for a whole-population observable; it sits in the
# macroparticle-occupancy tail, reads exactly 0.0 until the sampled population
# reaches it, and is shot-noise dominated once it fills. A large volume removes
# that noise and still sits above the emission region.
BAND_SPECIES = {
    "photon": (12_000.0, math.inf, math.inf, "photon_band_series.csv"),
    "electron": (12_000.0, math.inf, math.inf, "electron_band_series.csv"),
}

BAND_HEADER = (
    "time_s",
    "step",
    "band_z_lo_m_msl",
    "band_z_hi_m_msl",
    "band_r_max_m",
    "band_cell_count",
    "n_ge1MeV_real",
)


class CylinderCells(NamedTuple):
    rows: slice
    cols: slice
    # Centres are COORDINATES (weighted mean radius/altitude), never selection.
    r_centres_m: np.ndarray
    z_centres_m_msl: np.ndarray
    dz_m: float
    r_achieved_m: float


def cylinder_cells(
    capture_meta: dict[str, Any], *, r_max_m: float,
    z_lo_m_msl: float, z_hi_m_msl: float, what: str,
) -> CylinderCells:
    """Mesh cells of one upright cylinder, EDGE-aligned in r.

    One owner for the selection the channel mean and both gain reductions
    share, through band_column_slice, so the integrated volume is exactly
    pi r^2 dz -- the convention ring_volumes_m3 weights by.  Selecting on
    cell CENTRES instead admits the column straddling r_max and weights it
    with its whole ring, integrating to r_max + dr/2 under an r_max label.
    r_achieved_m is the outer edge actually integrated; callers report it
    rather than the request, as band_series already does.
    """
    if not (r_max_m > 0.0 and z_hi_m_msl > z_lo_m_msl):
        raise SystemExit(f"{what} needs r_max_m > 0 and z_hi > z_lo")
    cols, r_achieved_m = band_column_slice(capture_meta, r_max_m=r_max_m)
    nr, r_min = int(capture_meta["nr"]), float(capture_meta["r_min_m"])
    r_centres = (r_min + (float(capture_meta["r_max_m"]) - r_min) / nr
                 * (np.arange(nr) + 0.5))
    nz, z_min = int(capture_meta["nz"]), float(capture_meta["z_min_m"])
    dz = (float(capture_meta["z_max_m"]) - z_min) / nz
    z_centres = (float(capture_meta["altitude_msl_at_z0_m"]) + z_min
                 + (np.arange(nz) + 0.5) * dz)
    rows = np.flatnonzero((z_centres >= z_lo_m_msl) & (z_centres <= z_hi_m_msl))
    if rows.size == 0:
        raise SystemExit(
            f"{what} selects no mesh row: {z_lo_m_msl:g}-{z_hi_m_msl:g} m MSL "
            f"against a mesh spanning "
            f"{z_centres[0]:.0f}-{z_centres[-1]:.0f} m MSL")
    return CylinderCells(slice(int(rows[0]), int(rows[-1]) + 1), cols,
                         r_centres, z_centres, dz, r_achieved_m)


def ring_volumes_m3(capture_meta: dict[str, Any]) -> np.ndarray:
    """Exact RZ cell volumes of one mesh row: pi (r_out^2 - r_in^2) dz.

    The mesh is edge-aligned (crop_columns uses the same convention), so
    column j spans [r_min + j dr, r_min + (j+1) dr] -- NOT a cell-centred
    r_j * dr * 2pi, which would misweight the axis column by 2x.
    """
    nr = int(capture_meta["nr"])
    nz = int(capture_meta["nz"])
    r_min = float(capture_meta["r_min_m"])
    r_max = float(capture_meta["r_max_m"])
    dz = (float(capture_meta["z_max_m"]) - float(capture_meta["z_min_m"])) / nz
    edges = r_min + (r_max - r_min) / nr * np.arange(nr + 1, dtype=float)
    return math.pi * (edges[1:] ** 2 - edges[:-1] ** 2) * dz


def band_column_slice(
    capture_meta: dict[str, Any], *, r_max_m: float
) -> tuple[slice, float]:
    """Mesh columns fully inside r <= r_max, and the radius actually sampled.

    EDGE-aligned, matching ring_volumes_m3: a column counts only when its
    OUTER edge is within the cap, so the sampled volume is exactly
    pi r^2 dz per row and never straddles the boundary.  The achieved radius
    is returned rather than the request, because they differ whenever the cap
    falls mid-cell and the figure must label the volume it really integrated.
    """
    nr = int(capture_meta["nr"])
    r_min = float(capture_meta["r_min_m"])
    r_max_mesh = float(capture_meta["r_max_m"])
    edges = r_min + (r_max_mesh - r_min) / nr * np.arange(nr + 1, dtype=float)
    inside = np.flatnonzero(edges[1:] <= r_max_m)
    if inside.size == 0:
        raise SystemExit(
            f"no mesh column inside r <= {r_max_m:g} m; the mesh starts at "
            f"r = {edges[0]:g} m with dr = {edges[1] - edges[0]:g} m"
        )
    stop = int(inside[-1]) + 1
    return slice(0, stop), float(edges[stop])


def band_series(
    frames: np.ndarray,
    frame_meta: list[dict[str, Any]],
    capture_meta: dict[str, Any],
    *,
    species: str,
    z_lo_m_msl: float | None = None,
    z_hi_m_msl: float | None = None,
    r_max_m: float | None = None,
) -> dict[str, Any]:
    """Real >= 1 MeV particles of `species` inside the band, one per frame.

    This is an instantaneous POPULATION, not a crossing rate: the frame stream
    stores densities and carries no direction, so the flux through an altitude
    cannot be recovered from it.  Callers must label it as a population.
    """
    lo, hi, r_cap, _csv = BAND_SPECIES[species]
    cells = cylinder_cells(
        capture_meta,
        r_max_m=r_cap if r_max_m is None else r_max_m,
        z_lo_m_msl=lo if z_lo_m_msl is None else z_lo_m_msl,
        z_hi_m_msl=hi if z_hi_m_msl is None else z_hi_m_msl,
        what=f"{species} band")
    # frame_array_indices enforces the exact five-array order, so both
    # grids are always present by the time we get here.
    grid_index = frame_array_indices(capture_meta, frames)[species]
    volumes = ring_volumes_m3(capture_meta)[cells.cols]
    # One band per frame keeps peak memory at a single band regardless of
    # capture length; on a memmap it also pages in only the band's rows.
    counts = np.array(
        [
            float(np.einsum(
                "zr,r->", frames[index, grid_index, cells.rows, cells.cols],
                volumes))
            for index in range(frames.shape[0])
        ],
        dtype=float,
    )
    z_edges = cells.z_centres_m_msl[cells.rows][[0, -1]] + np.array(
        [-0.5, 0.5]) * cells.dz_m
    return {
        "time_s": np.array([float(row["time_s"]) for row in frame_meta]),
        "step": [int(float(row["step"])) for row in frame_meta],
        "n_ge1MeV_real": counts,
        # Edges actually integrated, never the request: an infinite bound reads
        # back as the mesh edge, so the label names the volume it really holds.
        "band_z_lo_m_msl": float(z_edges[0]),
        "band_z_hi_m_msl": float(z_edges[1]),
        "band_r_max_m": cells.r_achieved_m,
        # Rows, not cells: the label reads "N rows" and the radial extent is
        # reported by band_r_max_m rather than folded into this count.
        "band_cell_count": int(cells.rows.stop - cells.rows.start),
    }


# The channel-field reduction.  It answers ONE question: does the field inside
# the avalanche channel oscillate in step with the electron population?  The
# reduced CSV cannot answer it -- even `field_fraction_mean_accel_region`
# averages the whole taper-core cylinder, which dilutes a channel-confined
# modulation by orders of magnitude.  The frame stream has the field on the
# full mesh, so restrict it to the channel and read it directly.
CHANNEL_FIELD_CSV = "channel_field_series.csv"
CHANNEL_FIELD_HEADER = (
    "time_s",
    "step",
    "r_max_m",
    "z_lo_m_msl",
    "z_hi_m_msl",
    "cell_count",
    "mean_abs_e_kvpm",
    "volume_m3",
)


def channel_field_series(
    frames: np.ndarray,
    frame_meta: list[dict[str, Any]],
    capture_meta: dict[str, Any],
    *,
    r_max_m: float,
    z_lo_m_msl: float,
    z_hi_m_msl: float,
) -> dict[str, Any]:
    """Volume-weighted mean |E| inside a cylinder, one value per frame.

    VOLUME-weighted, not particle-weighted: the question is what field the
    channel is sitting in, so a cell counts for its size and nothing else.
    That is the whole point of not reusing the reduced CSV's
    ne_low-weighted column, whose weighting moves with the plasma and so
    cannot separate "the field changed" from "the plasma moved".
    """
    grid_index = frame_array_indices(capture_meta, frames)["field"]
    cyl = cylinder_cells(capture_meta, r_max_m=r_max_m, z_lo_m_msl=z_lo_m_msl,
                         z_hi_m_msl=z_hi_m_msl, what="channel")
    row_slice, col_slice = cyl.rows, cyl.cols
    weights = ring_volumes_m3(capture_meta)[col_slice]
    total_volume = float(weights.sum()) * (row_slice.stop - row_slice.start)
    means = np.array(
        [
            float(
                np.einsum(
                    "zr,r->", frames[index, grid_index, row_slice, col_slice],
                    weights,
                )
            )
            / total_volume
            for index in range(frames.shape[0])
        ],
        dtype=float,
    )
    return {
        "time_s": np.array([float(row["time_s"]) for row in frame_meta]),
        "step": [int(float(row["step"])) for row in frame_meta],
        "mean_abs_e_kvpm": means,
        "r_max_m": cyl.r_achieved_m,
        "z_lo_m_msl": float(z_lo_m_msl),
        "z_hi_m_msl": float(z_hi_m_msl),
        "cell_count": int((row_slice.stop - row_slice.start)
                          * (col_slice.stop - col_slice.start)),
        "volume_m3": total_volume,
    }


# The super-threshold gain integral. The channel |E| mean cannot answer whether
# the RREA gain oscillates, because gain is a rectified functional
#     S = integral max(E(z) - E_th(z), 0) dz / 7300      [e-folds]
# and a mean-preserving redistribution in z modulates S while leaving the band
# mean flat.  This is the one field-route hypothesis the mean cannot kill.
# S is also weighting-free: every ne_low- or runaway-weighted field mean
# confounds "the field changed" with "the plasma moved", and no single weighted
# scalar can separate the two.
GAIN_INTEGRAL_CSV = "gain_integral_series.csv"
GAIN_INTEGRAL_HEADER = (
    "time_s",
    "step",
    "r_max_m",
    "threshold_stp_kv_per_m",
    "s_efolds",
    "l_super_m",
    "m_signed_kvpm",
    "band_mean_abs_e_kvpm",
)
# lambda_RREA = RREA_AVALANCHE_NUMERATOR_KV / (E - E_th), E in kV/m, lambda in m.
# Imported, not restated: rrea_profiled_atmosphere owns it, and this module
# already imports from there.


def gain_integral_series(
    frames: np.ndarray,
    frame_meta: list[dict[str, Any]],
    capture_meta: dict[str, Any],
    *,
    r_max_m: float,
    z_lo_m_msl: float,
    z_hi_m_msl: float,
    threshold_stp_kv_per_m: float,
) -> dict[str, Any]:
    """Per-frame super-threshold gain integral over a cylinder.

    Returns S in e-folds, the super-threshold column length, the SIGNED mean
    margin, and the plain band-mean |E| -- the last one purely so the caller
    can check it against ``channel_field_series`` and catch a broken
    reduction rather than misread one as physics.
    """
    from rrea_profiled_atmosphere import (
        RREA_AVALANCHE_NUMERATOR_KV, load_threshold_profile)

    grid_index = frame_array_indices(capture_meta, frames)["field"]
    cyl = cylinder_cells(capture_meta, r_max_m=r_max_m, z_lo_m_msl=z_lo_m_msl,
                         z_hi_m_msl=z_hi_m_msl, what="gain integral")
    row_slice, col_slice = cyl.rows, cyl.cols
    z_centres, dz = cyl.z_centres_m_msl, cyl.dz_m
    # Same ring weights as channel_field_series: the mesh is edge-aligned, so
    # r*dr*2pi would misweight the axis column by 2x.
    weights = ring_volumes_m3(capture_meta)[col_slice]
    weights = weights / weights.sum()
    profile = load_threshold_profile(
        threshold_stp_kv_per_m=threshold_stp_kv_per_m)
    e_th = np.array(
        [profile.interpolate(float(z)) / 1000.0
         for z in z_centres[row_slice]], dtype=float)
    band_m = float(row_slice.stop - row_slice.start) * dz

    s_efolds, l_super, m_signed, band_mean = [], [], [], []
    margins = np.empty((frames.shape[0], e_th.size), dtype=np.float32)
    for index in range(frames.shape[0]):
        e_bar = frames[index, grid_index, row_slice, col_slice] @ weights
        margin = e_bar - e_th
        margins[index] = margin
        s_efolds.append(
            float(np.clip(margin, 0.0, None).sum() * dz
                  / RREA_AVALANCHE_NUMERATOR_KV))
        l_super.append(float((margin > 0.0).sum() * dz))
        m_signed.append(float(margin.sum() * dz / band_m))
        band_mean.append(float(e_bar.mean()))
    return {
        "altitude_m_msl": z_centres[row_slice],
        "e_th_kvpm": e_th,
        "margin_kvpm": margins,
        "dz_m": dz,
        "time_s": np.array([float(row["time_s"]) for row in frame_meta]),
        "step": [int(float(row["step"])) for row in frame_meta],
        "r_max_m": cyl.r_achieved_m,
        "threshold_stp_kv_per_m": float(threshold_stp_kv_per_m),
        "s_efolds": np.array(s_efolds),
        "l_super_m": np.array(l_super),
        "m_signed_kvpm": np.array(m_signed),
        "band_mean_abs_e_kvpm": np.array(band_mean),
    }


# Weight the avalanche margin by n_e, then separate field evolution from
# population motion:
#   m_weighted      full signal      n_e(t) . margin(t)
#   m_frozen_field  packet moves     n_e(t) . <margin>_t     <- kinematic term
#   m_frozen_packet field moves      <n_e>_t . margin(t)     <- field term
# This needs two frame-stream passes to build the time averages.
GAIN_WEIGHTED_CSV = "gain_weighted_series.csv"
GAIN_WEIGHTED_HEADER = (
    "time_s",
    "step",
    "r_max_m",
    "threshold_stp_kv_per_m",
    "weight_arb",
    "m_weighted_kvpm",
    "m_frozen_field_kvpm",
    "m_frozen_packet_kvpm",
    "mean_r_m",
    "mean_z_m_msl",
)


def gain_weighted_series(
    frames: np.ndarray,
    frame_meta: list[dict[str, Any]],
    capture_meta: dict[str, Any],
    *,
    r_max_m: float,
    z_lo_m_msl: float,
    z_hi_m_msl: float,
    threshold_stp_kv_per_m: float,
) -> dict[str, Any]:
    """Electron-density-weighted margin plus its frozen-field/frozen-packet split."""
    from rrea_profiled_atmosphere import load_threshold_profile

    idx = frame_array_indices(capture_meta, frames)
    field_i, elec_i = idx["field"], idx["electron"]
    cyl = cylinder_cells(capture_meta, r_max_m=r_max_m, z_lo_m_msl=z_lo_m_msl,
                         z_hi_m_msl=z_hi_m_msl, what="weighted gain")
    rs, cs = cyl.rows, cyl.cols
    r_centres, z_centres = cyl.r_centres_m, cyl.z_centres_m_msl
    vol = ring_volumes_m3(capture_meta)[cs]              # (r,)
    profile = load_threshold_profile(
        threshold_stp_kv_per_m=threshold_stp_kv_per_m)
    e_th = np.array([profile.interpolate(float(z)) / 1000.0
                     for z in z_centres[rs]], dtype=float)[:, None]   # (z,1)
    n_frames = frames.shape[0]

    # Pass 1: time averages of the margin field and of the density.
    margin_sum = np.zeros((rs.stop - rs.start, cs.stop - cs.start))
    dens_sum = np.zeros_like(margin_sum)
    for i in range(n_frames):
        margin_sum += frames[i, field_i, rs, cs] - e_th
        dens_sum += frames[i, elec_i, rs, cs]
    margin_bar = margin_sum / n_frames
    dens_bar = dens_sum / n_frames

    # Pass 2: the three weighted means.
    out = {k: [] for k in ("weight_arb", "m_weighted_kvpm", "m_frozen_field_kvpm",
                           "m_frozen_packet_kvpm", "mean_r_m", "mean_z_m_msl")}
    r_grid = r_centres[cs][None, :]
    z_grid = z_centres[rs][:, None]
    denom_bar = float((dens_bar * vol).sum())
    for i in range(n_frames):
        dens = frames[i, elec_i, rs, cs]
        margin = frames[i, field_i, rs, cs] - e_th
        w = dens * vol
        tot = float(w.sum())
        out["weight_arb"].append(tot)
        if tot <= 0.0:
            for k in ("m_weighted_kvpm", "m_frozen_field_kvpm",
                      "m_frozen_packet_kvpm", "mean_r_m", "mean_z_m_msl"):
                out[k].append(float("nan"))
            continue
        out["m_weighted_kvpm"].append(float((w * margin).sum() / tot))
        out["m_frozen_field_kvpm"].append(float((w * margin_bar).sum() / tot))
        out["m_frozen_packet_kvpm"].append(
            float((dens_bar * vol * margin).sum() / denom_bar))
        out["mean_r_m"].append(float((w * r_grid).sum() / tot))
        out["mean_z_m_msl"].append(float((w * z_grid).sum() / tot))
    return {
        "time_s": np.array([float(r["time_s"]) for r in frame_meta]),
        "step": [int(float(r["step"])) for r in frame_meta],
        "r_max_m": cyl.r_achieved_m,
        "threshold_stp_kv_per_m": float(threshold_stp_kv_per_m),
        **{k: np.array(v) for k, v in out.items()},
    }


def write_gain_weighted_series(
    series: dict[str, Any] | list[dict[str, Any]], path: Path
) -> None:
    all_series = [series] if isinstance(series, dict) else list(series)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(GAIN_WEIGHTED_HEADER)
        for one in all_series:
            for i, time_s in enumerate(one["time_s"]):
                writer.writerow([
                    repr(float(time_s)), one["step"][i],
                    repr(one["r_max_m"]), repr(one["threshold_stp_kv_per_m"]),
                    repr(float(one["weight_arb"][i])),
                    repr(float(one["m_weighted_kvpm"][i])),
                    repr(float(one["m_frozen_field_kvpm"][i])),
                    repr(float(one["m_frozen_packet_kvpm"][i])),
                    repr(float(one["mean_r_m"][i])),
                    repr(float(one["mean_z_m_msl"][i])),
                ])


GAIN_PROFILE_NPZ = "gain_profile.npz"


def write_gain_profile(series: dict[str, Any], path: Path) -> None:
    """z-resolved super-threshold integrand, one row per frame.

    S(t) alone says the gain modulates but not WHERE, and "where" is the
    discriminator: a disturbance that propagates through the column with a
    phase tilt in z is transport, while one that switches on across the whole
    column at once is local -- or numerical.  A scalar cannot tell those
    apart, so this keeps the altitude axis in a compact NPZ rather than CSV.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        time_s=series["time_s"].astype(np.float64),
        step=np.asarray(series["step"], dtype=np.int64),
        altitude_m_msl=series["altitude_m_msl"].astype(np.float64),
        e_th_kvpm=series["e_th_kvpm"].astype(np.float64),
        margin_kvpm=series["margin_kvpm"].astype(np.float32),
        r_max_m=float(series["r_max_m"]),
        threshold_stp_kv_per_m=float(series["threshold_stp_kv_per_m"]),
        dz_m=float(series["dz_m"]),
    )


def write_gain_integral_series(
    series: dict[str, Any] | list[dict[str, Any]], path: Path
) -> None:
    all_series = [series] if isinstance(series, dict) else list(series)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(GAIN_INTEGRAL_HEADER)
        for one in all_series:
            for index, time_s in enumerate(one["time_s"]):
                writer.writerow([
                    repr(float(time_s)),
                    one["step"][index],
                    repr(one["r_max_m"]),
                    repr(one["threshold_stp_kv_per_m"]),
                    repr(float(one["s_efolds"][index])),
                    repr(float(one["l_super_m"][index])),
                    repr(float(one["m_signed_kvpm"][index])),
                    repr(float(one["band_mean_abs_e_kvpm"][index])),
                ])


def write_channel_field_series(
    series: dict[str, Any] | list[dict[str, Any]], path: Path
) -> None:
    """Write one or several channel series to ONE csv.

    Several radii land in the same file rather than one file per radius
    because ``r_max_m`` is already a per-row column: a consumer selects a
    radius by filtering, and the alternative (encoding the radius in the
    filename) invents a second place for the same number.  Writing them
    together is also what makes a radius SCAN affordable -- the frame stream
    is read once for every radius instead of once per radius.
    """
    all_series = [series] if isinstance(series, dict) else list(series)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(CHANNEL_FIELD_HEADER)
        for one in all_series:
            for index, time_s in enumerate(one["time_s"]):
                writer.writerow([
                    repr(float(time_s)),
                    one["step"][index],
                    repr(one["r_max_m"]),
                    repr(one["z_lo_m_msl"]),
                    repr(one["z_hi_m_msl"]),
                    one["cell_count"],
                    repr(float(one["mean_abs_e_kvpm"][index])),
                    repr(float(one["volume_m3"])),
                ])


def band_label(band: dict[str, Any], species: str) -> tuple[str, str]:
    """Shared (title stem, y-axis label) for a band curve."""
    noun = "photons" if species == "photon" else "electrons"
    # Default region: everything above 12 km MSL; the parenthesis is the extent summed.
    return (
        f"source lightcurve: N {noun} E >= 1 MeV above "
        f"{band['band_z_lo_m_msl'] / 1e3:.0f} km MSL (to "
        f"{band['band_z_hi_m_msl'] / 1e3:g} km, r <= "
        f"{band['band_r_max_m'] / 1e3:g} km, {band['band_cell_count']} rows",
        f"N {noun}",
    )


def write_band_series(series: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(BAND_HEADER)
        for index, time_s in enumerate(series["time_s"]):
            writer.writerow(
                [
                    repr(float(time_s)),
                    series["step"][index],
                    repr(series["band_z_lo_m_msl"]),
                    repr(series["band_z_hi_m_msl"]),
                    repr(series["band_r_max_m"]),
                    series["band_cell_count"],
                    repr(float(series["n_ge1MeV_real"][index])),
                ]
            )


def load_band_series(capture_dir: Path, species: str) -> dict[str, Any] | None:
    """Read a written series back, or None when the capture has none.

    None is normal for an unreduced capture; consumers show a labelled empty
    panel or omit the strip.
    """
    path = Path(capture_dir) / "rrea_video" / BAND_SPECIES[species][3]
    if not path.exists():
        return None
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return None
    missing = [name for name in BAND_HEADER if name not in rows[0]]
    if missing:
        raise SystemExit(f"{path} is missing columns {missing}")
    return {
        "time_s": np.array([float(row["time_s"]) for row in rows]),
        "n_ge1MeV_real": np.array(
            [float(row["n_ge1MeV_real"]) for row in rows]),
        "band_z_lo_m_msl": float(rows[0]["band_z_lo_m_msl"]),
        "band_z_hi_m_msl": float(rows[0]["band_z_hi_m_msl"]),
        "band_r_max_m": float(rows[0]["band_r_max_m"]),
        "band_cell_count": int(rows[0]["band_cell_count"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("capture_dir", type=Path)
    parser.add_argument("--species", choices=sorted(BAND_SPECIES), action="append",
                        help="repeatable; default is every species")
    parser.add_argument("--band-z-lo-km-msl", type=float, default=None,
                        help="override the species band (see BAND_SPECIES); "
                             "the edges actually integrated are written into "
                             "every row")
    parser.add_argument("--band-z-hi-km-msl", type=float, default=None)
    parser.add_argument("--band-r-max-m", type=float, default=None)
    parser.add_argument(
        "--channel-field", action="store_true",
        help="also reduce the volume-mean |E| inside the avalanche channel "
             "(writes " + CHANNEL_FIELD_CSV + "); this is the only local way "
             "to see a channel-confined field modulation, which the reduced "
             "CSV's whole-domain mean dilutes away")
    parser.add_argument(
        "--channel-r-max-m", type=float, nargs="+", default=[150.0],
        help="repeatable in ONE run: several radii share the single frame-stream "
             "read, which is what the reduction actually costs. A scan is the "
             "only way to tell a channel-confined field modulation from one "
             "diluted by too WIDE a cylinder -- the same dilution that blurs "
             "the reduced CSV's field_fraction_mean_accel_region")
    parser.add_argument("--channel-z-lo-km-msl", type=float, default=10.887)
    parser.add_argument("--channel-z-hi-km-msl", type=float, default=13.136)
    parser.add_argument(
        "--gain-integral", action="store_true",
        help="also reduce the super-threshold gain integral S = int max(E-E_th,0) dz "
             "/ 7300 over the same cylinder (writes " + GAIN_INTEGRAL_CSV + "). "
             "S is what the avalanche actually responds to: because it is "
             "RECTIFIED, a mean-preserving redistribution in z modulates it while "
             "leaving the band mean |E| flat, so a flat channel mean does NOT imply "
             "a flat gain")
    parser.add_argument(
        "--gain-threshold-stp-kv-per-m", type=float, nargs="+", default=[284.0, 276.0],
        help="RREA onset at STP; both project thresholds by default")
    parser.add_argument(
        "--gain-weighted", action="store_true",
        help="reduce the ELECTRON-DENSITY-WEIGHTED margin and its "
             "frozen-field/frozen-packet split (writes " + GAIN_WEIGHTED_CSV + "). "
             "Use a radius that actually contains the population -- the >=1 MeV "
             "electrons sit near mean_r ~ 2 km, so an on-axis cylinder measures "
             "gain where they are not. Two passes over the frame stream")
    parser.add_argument(
        "--gain-profile", action="store_true",
        help="also write the z-RESOLVED margin cube (" + GAIN_PROFILE_NPZ + ") for "
             "the first radius/threshold. S(t) says the gain modulates but not "
             "where; a phase tilt in z means the disturbance PROPAGATES "
             "(transport), no tilt means it switches across the column at once "
             "(local, or numerical)")
    args = parser.parse_args()

    frames, frame_meta, capture_meta, _render_map = load_frames(args.capture_dir)
    if args.channel_field:
        all_series = [
            channel_field_series(
                frames, frame_meta, capture_meta,
                r_max_m=r_max,
                z_lo_m_msl=args.channel_z_lo_km_msl * 1e3,
                z_hi_m_msl=args.channel_z_hi_km_msl * 1e3,
            )
            for r_max in args.channel_r_max_m
        ]
        path = args.capture_dir / "rrea_video" / CHANNEL_FIELD_CSV
        write_channel_field_series(all_series, path)
        for series in all_series:
            means = series["mean_abs_e_kvpm"]
            print(
                f"wrote {path}: {means.size} frames, channel "
                f"r<={series['r_max_m']:g} m, "
                f"{series['z_lo_m_msl'] / 1e3:g}-{series['z_hi_m_msl'] / 1e3:g} km MSL "
                f"({series['cell_count']} cells, {series['volume_m3']:.4g} m^3), "
                f"mean |E| {means.min():.4g}-{means.max():.4g} kV/m"
                if means.size else f"wrote {path}: no frames"
            )
    if args.gain_weighted:
        weighted = [
            gain_weighted_series(
                frames, frame_meta, capture_meta, r_max_m=r_max,
                z_lo_m_msl=args.channel_z_lo_km_msl * 1e3,
                z_hi_m_msl=args.channel_z_hi_km_msl * 1e3,
                threshold_stp_kv_per_m=e_th)
            for r_max in args.channel_r_max_m
            for e_th in args.gain_threshold_stp_kv_per_m
        ]
        path = args.capture_dir / "rrea_video" / GAIN_WEIGHTED_CSV
        write_gain_weighted_series(weighted, path)
        for one in weighted:
            mw = one["m_weighted_kvpm"]
            print(
                f"wrote {path}: r<={one['r_max_m']:g} m, "
                f"E_th(STP)={one['threshold_stp_kv_per_m']:g} | "
                f"weighted margin {np.nanmin(mw):.4g} to {np.nanmax(mw):.4g} kV/m | "
                f"population mean_r {np.nanmean(one['mean_r_m']):.0f} m, "
                f"mean_z {np.nanmean(one['mean_z_m_msl']):.0f} m MSL")

    if args.gain_integral:
        all_series = [
            gain_integral_series(
                frames, frame_meta, capture_meta,
                r_max_m=r_max,
                z_lo_m_msl=args.channel_z_lo_km_msl * 1e3,
                z_hi_m_msl=args.channel_z_hi_km_msl * 1e3,
                threshold_stp_kv_per_m=e_th,
            )
            for r_max in args.channel_r_max_m
            for e_th in args.gain_threshold_stp_kv_per_m
        ]
        path = args.capture_dir / "rrea_video" / GAIN_INTEGRAL_CSV
        write_gain_integral_series(all_series, path)
        if args.gain_profile:
            # One cube only: the z-structure question does not need a radius or
            # threshold sweep.
            profile_path = args.capture_dir / "rrea_video" / GAIN_PROFILE_NPZ
            write_gain_profile(all_series[0], profile_path)
            cube = all_series[0]["margin_kvpm"]
            print(
                f"wrote {profile_path}: {cube.shape[0]} frames x {cube.shape[1]} "
                f"altitudes, r<={all_series[0]['r_max_m']:g} m, "
                f"E_th(STP)={all_series[0]['threshold_stp_kv_per_m']:g} kV/m, "
                f"margin {cube.min():.4g} to {cube.max():.4g} kV/m")
        for series in all_series:
            s, l = series["s_efolds"], series["l_super_m"]
            # Frame 0 is the unscreened ambient profile, so it is a free
            # calibration of the whole reduction: print it rather than trust it.
            print(
                f"wrote {path}: r<={series['r_max_m']:g} m, "
                f"E_th(STP)={series['threshold_stp_kv_per_m']:g} kV/m, "
                f"{s.size} frames | S {s.min():.4g}-{s.max():.4g} e-folds "
                f"(frame0 {s[0]:.4g}) | L_super {l.min():.4g}-{l.max():.4g} m "
                f"(frame0 {l[0]:.4g}) | band mean |E| frame0 "
                f"{series['band_mean_abs_e_kvpm'][0]:.4g} kV/m"
                if s.size else f"wrote {path}: no frames"
            )

    for species in (args.species or sorted(BAND_SPECIES)):
        series = band_series(
            frames, frame_meta, capture_meta, species=species,
            z_lo_m_msl=(None if args.band_z_lo_km_msl is None
                        else args.band_z_lo_km_msl * 1e3),
            z_hi_m_msl=(None if args.band_z_hi_km_msl is None
                        else args.band_z_hi_km_msl * 1e3),
            r_max_m=args.band_r_max_m,
        )
        path = args.capture_dir / "rrea_video" / BAND_SPECIES[species][3]
        write_band_series(series, path)
        counts = series["n_ge1MeV_real"]
        peak = int(np.argmax(counts)) if counts.size else 0
        print(
            f"wrote {path}: {counts.size} frames, {species} band "
            f"{series['band_z_lo_m_msl'] / 1e3:g}-"
            f"{series['band_z_hi_m_msl'] / 1e3:g} km MSL, "
            f"r<={series['band_r_max_m']:g} m "
            f"({series['band_cell_count']} mesh rows), peak "
            f"{counts[peak]:.3e} at t = {series['time_s'][peak] * 1e6:.2f} us"
            if counts.size else f"wrote {path}: no frames"
        )


if __name__ == "__main__":
    main()
