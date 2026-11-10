"""Band lightcurves from the frame stream.

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

writes both species at their default bands in one pass; each frame
contributes one contiguous block of mesh rows per species.
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
# reaches it, and is shot-noise dominated once it fills. A larger volume can reduce
# relative sampling noise; its populations are not detector fluxes.
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
    # Centres are COORDINATES, never selection.
    z_centres_m_msl: np.ndarray
    dz_m: float
    r_achieved_m: float


def cylinder_cells(
    capture_meta: dict[str, Any], *, r_max_m: float,
    z_lo_m_msl: float, z_hi_m_msl: float, what: str,
) -> CylinderCells:
    """Mesh cells of one upright cylinder, EDGE-aligned in r.

    Columns come from band_column_slice (edge-aligned); r_achieved_m is the
    outer edge actually integrated.
    """
    if not (r_max_m > 0.0 and z_hi_m_msl > z_lo_m_msl):
        raise SystemExit(f"{what} needs r_max_m > 0 and z_hi > z_lo")
    cols, r_achieved_m = band_column_slice(capture_meta, r_max_m=r_max_m)
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
                         z_centres, dz, r_achieved_m)


def ring_volumes_m3(capture_meta: dict[str, Any]) -> np.ndarray:
    """Exact RZ cell volumes of one mesh row: pi (r_out^2 - r_in^2) dz.

    The mesh is edge-aligned (crop_columns uses the same convention), so
    column j spans [r_min + j dr, r_min + (j+1) dr].
    Using the cell-centre radius gives the equivalent 2*pi*r*dr*dz; using
    a radial edge in that expression would misweight the axis cell.
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
    # One frame at a time keeps peak memory at one inflated frame regardless
    # of capture length.
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
    args = parser.parse_args()

    frames, frame_meta, capture_meta, _render_map = load_frames(args.capture_dir)
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
