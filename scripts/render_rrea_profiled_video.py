#!/usr/bin/env python3
"""Render compact RREA profiled-video frames to high-quality MP4s and previews."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Iterable, NamedTuple

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from rrea_cloud_scattering import OBSERVER_ALTITUDE_M_MSL
from rrea_run_support import VIDEO_DISPLAY_DOMAIN_FRACTION
from rrea_video_contract import (
    VIDEO_ARRAYS_PER_FRAME,
    VIDEO_V4_FORMAT,
    decode_v4_frame,
    parse_warp_control_points,
    warp_rate_us_per_s_at,
)


PANEL_CHOICES = ("abs", "fraction", "delta", "both", "all")
PREVIEW_TIMES_S = [0.0, 0.25e-6, 0.5e-6, 1.0e-6, 10.0e-6, 50.0e-6, 100.0e-6]
# Pooled samples retained per colour-limit accumulator (see compute_limits).
# 8e6 resolves the 99.7th percentile far finer than the +0.2 dex pad the
# window itself applies while bounding memory independently of capture length.
LIMITS_SAMPLE_BUDGET = 8_000_000
# Default nonlinear playback: brisk through sparse growth, slower around the
# active interval, then accelerating through a long relaxation tail.
DEFAULT_WARP_SPEC = "0:6,70:6,95:2,150:8,300:25,1000:60"
def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_frame_metadata(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"no frame metadata rows found: {path}")
    return rows


def require_video_metadata(capture_meta: dict[str, Any]) -> tuple[str, ...]:
    """Fail closed when a capture loses or drifts its binary contract."""
    if capture_meta.get("format") != VIDEO_V4_FORMAT:
        raise SystemExit(
            f"video renderer requires {VIDEO_V4_FORMAT}; "
            f"got {capture_meta.get('format')!r}"
        )
    arrays = capture_meta.get("arrays_per_frame")
    if not isinstance(arrays, list) or tuple(arrays) != VIDEO_ARRAYS_PER_FRAME:
        raise SystemExit(
            "video renderer requires the exact five-array order; "
            f"got {arrays!r}"
        )
    return VIDEO_ARRAYS_PER_FRAME


class CompressedFrameStream:
    """Lazy [frame, array, z, r] view over a video-v4 stream.

    v4 frames are individually deflated and variable length, so they cannot be
    memory-mapped as one array.  This presents the same indexing surface the
    render loop already uses and inflates one frame at a time, keeping the most
    recent few: the loop walks frames in order and touches several arrays of the
    same frame per iteration, so a tiny cache removes essentially all repeats.
    """

    def __init__(
        self,
        path: Path,
        frame_meta: list[dict[str, Any]],
        *,
        n_arrays: int,
        nz: int,
        nr: int,
        cache: int = 2,
    ) -> None:
        self._path = path
        self._n_arrays, self._nz, self._nr = n_arrays, nz, nr
        self._raw_bytes = n_arrays * nz * nr * 4
        self._cache: dict[int, np.ndarray] = {}
        self._cache_max = max(1, cache)
        self._index: list[tuple[int, int]] = []
        for row_number, row in enumerate(frame_meta, start=2):
            try:
                self._index.append(
                    (int(row["byte_offset"]), int(row["byte_length"]))
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise SystemExit(
                    f"{path}: video-v4 frame metadata row {row_number} lacks a "
                    f"usable byte_offset/byte_length ({exc})"
                ) from exc
        self.shape = (len(self._index), n_arrays, nz, nr)
        # Array consumers inspect these before indexing.
        self.ndim = 4
        self.dtype = np.dtype("<f4")

    def __len__(self) -> int:
        return len(self._index)

    def _frame(self, index: int) -> np.ndarray:
        cached = self._cache.get(index)
        if cached is not None:
            return cached
        offset, length = self._index[index]
        with self._path.open("rb") as handle:
            handle.seek(offset)
            blob = handle.read(length)
        if len(blob) != length:
            raise SystemExit(
                f"{self._path}: frame {index} is truncated "
                f"({len(blob)} of {length} bytes at offset {offset})"
            )
        values = np.frombuffer(
            decode_v4_frame(blob, expected_bytes=self._raw_bytes), dtype="<f4"
        ).reshape(self._n_arrays, self._nz, self._nr)
        if len(self._cache) >= self._cache_max:
            self._cache.pop(next(iter(self._cache)))
        self._cache[index] = values
        return values

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            return self._frame(int(key))
        return self._frame(int(key[0]))[key[1:]]


def compact_npz_candidates(input_dir: Path) -> list[Path]:
    return [
        input_dir / "rendered_video" / "profiled_video_frames_compact.npz",
        input_dir / "compact_frames" / "profiled_video_frames_compact.npz",
        input_dir / "profiled_video_frames_compact.npz",
    ]


def load_frames(input_dir: Path) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    video_dir = input_dir / "rrea_video"
    raw_path = video_dir / "video_frames.bin"
    render_map_path = input_dir / "video_render_map.json"
    if raw_path.exists():
        capture_meta = load_json(video_dir / "video_capture_metadata.json")
        frame_meta = load_frame_metadata(video_dir / "video_frame_metadata.csv")
        nr = int(capture_meta["nr"])
        nz = int(capture_meta["nz"])
        arrays = require_video_metadata(capture_meta)
        # A live capture can be one frame ahead of its metadata (or behind it,
        # between the two appends); the index only covers complete pairs, so
        # trailing partial work is simply not addressable and is skipped.
        frames = CompressedFrameStream(
            raw_path, frame_meta, n_arrays=len(arrays), nz=nz, nr=nr
        )
        return frames, frame_meta, capture_meta, load_json(render_map_path)

    for npz_path in compact_npz_candidates(input_dir):
        if not npz_path.exists():
            continue
        with np.load(npz_path, allow_pickle=False) as archive:
            frames = np.asarray(archive["frames"])
            frame_meta = json.loads(str(archive["frame_metadata"]))
            capture_meta = json.loads(str(archive["capture_metadata"]))
            render_map = json.loads(str(archive["render_map"]))
        arrays = require_video_metadata(capture_meta)
        if frames.ndim != 4 or frames.shape[1] != len(arrays):
            raise SystemExit(f"invalid compact frame archive shape: {frames.shape}")
        return frames, frame_meta, capture_meta, render_map

    raise SystemExit(
        f"no video_frames.bin or compact NPZ found below {input_dir}; "
        "rerun capture or keep the compact archive"
    )


def gaussian_kernel1d(sigma: float) -> np.ndarray:
    if sigma <= 0.0:
        return np.array([1.0], dtype=float)
    radius = max(1, int(math.ceil(3.0 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-(x * x) / (2.0 * sigma * sigma))
    kernel /= kernel.sum()
    return kernel


def smooth2d(values: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0.0:
        return values
    kernel = gaussian_kernel1d(sigma)
    radius = kernel.size // 2
    # Separable convolution, zero-padded like np.convolve(..., mode="same").
    padded = np.pad(values, ((0, 0), (radius, radius)))
    out = sliding_window_view(padded, kernel.size, axis=1) @ kernel
    padded = np.pad(out, ((radius, radius), (0, 0)))
    return sliding_window_view(padded, kernel.size, axis=0) @ kernel


def robust_limits(values: np.ndarray) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    lo = float(np.nanpercentile(finite, 0.5))
    hi = float(np.nanpercentile(finite, 99.5))
    if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
        lo, hi = float(np.nanmin(finite)), float(np.nanmax(finite) + 1.0)
    return lo, hi


def parse_limits(text: str | None, name: str) -> tuple[float, float] | None:
    if text is None:
        return None
    parts = [part.strip() for part in text.replace(",", " ").split() if part.strip()]
    if len(parts) != 2:
        raise SystemExit(f"{name} must contain exactly two numbers, got: {text!r}")
    lo, hi = float(parts[0]), float(parts[1])
    if not (math.isfinite(lo) and math.isfinite(hi)) or hi <= lo:
        raise SystemExit(f"{name} must be finite with hi > lo, got: {text!r}")
    return lo, hi


def crop_columns(capture_meta: dict[str, Any], r_max_display_m: float | None) -> tuple[slice, list[float]]:
    nr = int(capture_meta["nr"])
    r_min = float(capture_meta["r_min_m"])
    r_max = float(capture_meta["r_max_m"])
    dr = (r_max - r_min) / nr
    requested_r_max = r_max if r_max_display_m is None else min(r_max, float(r_max_display_m))
    if requested_r_max <= r_min:
        raise SystemExit("display radius must exceed captured r_min")
    last_col = int(math.ceil((requested_r_max - r_min) / dr))
    last_col = max(1, min(nr, last_col))
    extent_r_max = r_min + last_col * dr
    z_min = float(capture_meta["z_min_m"])
    z_max = float(capture_meta["z_max_m"])
    altitude0_km = float(capture_meta["altitude_msl_at_z0_m"]) / 1000.0
    extent = [r_min, extent_r_max, altitude0_km + z_min / 1000.0, altitude0_km + z_max / 1000.0]
    return slice(0, last_col), extent


def display_array(values: np.ndarray, *, symmetric_radius_display: bool) -> np.ndarray:
    if not symmetric_radius_display:
        return values
    return np.concatenate((values[:, ::-1], values), axis=1)


def field_profile_curves(
    values: np.ndarray,
    *,
    symmetric_radius_display: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (radial mean, user-labelled r=0 slice) of displayed |E|(z,r).

    The captured field is cell-centred, so the ``r = 0`` curve is the first
    radial cell. In a mirrored display it is the centre-right column.
    """
    field = np.asarray(values, dtype=float)
    if field.ndim != 2 or field.shape[1] == 0:
        raise ValueError("field profile input must be a nonempty 2-D array")
    axis_column = field.shape[1] // 2 if symmetric_radius_display else 0
    return np.nanmean(field, axis=1), field[:, axis_column]


def field_profile_x_max(*profiles: np.ndarray) -> float:
    """One fixed |E| scale with 50% headroom for every profile curve."""
    finite = [
        values[np.isfinite(values)]
        for values in (np.asarray(profile, dtype=float) for profile in profiles)
    ]
    finite = [values for values in finite if values.size]
    peak = max((float(np.max(values)) for values in finite), default=0.0)
    return 1.5 * peak if peak > 0.0 else 1.0


def shared_strip_time_limits_us(*time_arrays: np.ndarray) -> tuple[float, float]:
    """Union of finite source-time records drawn in the bottom strips."""
    finite = [
        np.asarray(values, dtype=float)[np.isfinite(values)]
        for values in time_arrays
        if values is not None
    ]
    finite = [values for values in finite if values.size]
    if not finite:
        raise ValueError("shared strip time limits require finite samples")
    lo = min(float(np.min(values)) for values in finite)
    hi = max(float(np.max(values)) for values in finite)
    if not hi > lo:
        raise ValueError("shared strip time limits require a positive span")
    return lo, hi


def display_extent_and_xlabel(
    extent: list[float],
    *,
    symmetric_radius_display: bool,
) -> tuple[list[float], str]:
    if not symmetric_radius_display:
        return extent, "radius r (m)"
    if abs(extent[0]) > 1.0e-12:
        raise SystemExit("symmetric radius display requires captured/cropped r_min = 0")
    return [-extent[1], extent[1], extent[2], extent[3]], "mirrored horizontal coordinate (m)"


def frame_array_indices(capture_meta: dict[str, Any], frames: np.ndarray) -> dict[str, int | None]:
    arrays = require_video_metadata(capture_meta)
    if frames.ndim != 4 or frames.shape[1] != len(arrays):
        raise SystemExit(
            "frame tensor does not match the five-array contract: "
            f"shape={frames.shape}"
        )
    names = {str(name): index for index, name in enumerate(arrays)}
    electron_index = names.get("n_energetic_e_ge_threshold_m3", 0)
    field_index = names.get("abs_E_kVpm", frames.shape[1] - 1)

    def optional(name: str) -> int | None:
        index = names.get(name)
        return None if index is None else int(index)

    # Photon and positron grids use the same energy threshold as electrons;
    # they are not total-population grids.
    return {
        "electron": int(electron_index),
        "positive_ion": optional("n_positive_ion_m3"),
        "photon": optional("n_photon_ge_threshold_m3"),
        "positron": optional("n_positron_ge_threshold_m3"),
        "field": int(field_index),
    }


class DensityPanel(NamedTuple):
    source: str    # frame_array_indices key holding the grid
    title: str
    quantity: str  # colourbar reads "log10 <quantity> density (m^-3)"
    cmap: str


# Every density panel renders identically (log10 of a per-volume density), so
# they share one code path rather than each growing a slot of its own -- but
# labels and colours travel with each source key to prevent cross-species
# mislabelling.
# Only the ion grid is unthresholded: photon/positron are the >= 1 MeV
# populations, the same cut as the electron map.
#
# One species, one colour ramp.  All four (magma for electrons, below) are
# perceptually uniform and colourblind-safe, and they separate by mid-range
# hue -- red, orange, green, blue -- which is where most pixels sit.  They do
# share a near-black low end, so an empty panel looks the same in all of them.
DENSITY_PANELS = {
    "ion": DensityPanel(
        "positive_ion", "Positive ions on AMReX mesh", "positive ion", "plasma"),
    "photon": DensityPanel(
        "photon", "Photons, E >= 1 MeV", "photon", "viridis"),
    "positron": DensityPanel(
        "positron", "Positrons, E >= 1 MeV", "positron", "cividis"),
}


def parse_density_panels(text: str) -> list[str]:
    kinds = [item.strip() for item in text.split(",") if item.strip()]
    unknown = [kind for kind in kinds if kind not in DENSITY_PANELS]
    if unknown:
        raise SystemExit(
            f"--density-panels: unknown {unknown}; choose from {sorted(DENSITY_PANELS)}")
    return kinds


def iter_cropped_arrays(
    frames: np.ndarray,
    indices: dict[str, int | None],
    r_slice: slice,
    *,
    electron_smooth_sigma: float,
    field_smooth_sigma: float,
) -> Iterable[tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]]:
    electron_index = int(indices["electron"])
    field_index = int(indices["field"])
    density_indices = {
        kind: int(indices[panel.source])
        for kind, panel in DENSITY_PANELS.items()
        if indices.get(panel.source) is not None
    }
    for frame_index in range(frames.shape[0]):
        electron = np.asarray(frames[frame_index, electron_index, :, r_slice], dtype=float)
        field = np.asarray(frames[frame_index, field_index, :, r_slice], dtype=float)
        densities = {
            kind: smooth2d(
                np.asarray(frames[frame_index, source, :, r_slice], dtype=float),
                electron_smooth_sigma)
            for kind, source in density_indices.items()
        }
        electron = smooth2d(electron, electron_smooth_sigma)
        field = smooth2d(field, field_smooth_sigma)
        yield electron, densities, field


def compute_limits(
    frames: np.ndarray,
    indices: dict[str, int | None],
    r_slice: slice,
    *,
    electron_smooth_sigma: float,
    field_smooth_sigma: float,
) -> tuple[
    tuple[float, float],
    tuple[float, float] | None,
    tuple[float, float],
    tuple[float, float],
    tuple[float, float],
    float,
]:
    electron_logs: list[np.ndarray] = []
    density_logs: dict[str, list[np.ndarray]] = {kind: [] for kind in DENSITY_PANELS}
    field_values: list[np.ndarray] = []
    fraction_values: list[np.ndarray] = []
    field_profile_limit_kvpm = 1.0
    initial_field = None
    # Only percentiles come out of this pass, so a fixed sample budget makes its
    # memory independent of frame count.  Sampling, not
    # binning: a fixed-range histogram would quantise |fraction-1| against its
    # own 1e-6 clamp and visibly distort the delta panel in the quiet regime,
    # while decimation is unbiased and has no resolution floor.
    nframes = max(1, int(frames.shape[0]))
    ncols = len(range(*r_slice.indices(int(frames.shape[3]))))
    per_frame_budget = max(1, LIMITS_SAMPLE_BUDGET // nframes)

    def thin(values: np.ndarray) -> np.ndarray:
        if values.size > per_frame_budget:
            keep = -(-values.size // per_frame_budget)
            # The flatten is z-major/r-minor, so a stride sharing a factor with
            # the row length would resample the same r-columns on every row.
            while ncols > 1 and math.gcd(keep, ncols) != 1:
                keep += 1
            values = values[::keep]
        # Anything that is still a view -- a strided sample, or a ravel of the
        # frame -- would pin its whole frame-sized base array in the list and
        # defeat the budget entirely.  Only materialise in that case.
        return values if values.base is None else values.copy()

    for electron, densities, field in iter_cropped_arrays(
        frames,
        indices,
        r_slice,
        electron_smooth_sigma=electron_smooth_sigma,
        field_smooth_sigma=field_smooth_sigma,
    ):
        positive = electron[electron > 0.0]
        if positive.size:
            electron_logs.append(thin(np.log10(positive)))
        for kind, grid in densities.items():
            grid_positive = grid[grid > 0.0]
            if grid_positive.size:
                density_logs[kind].append(thin(np.log10(grid_positive)))
        # |E| is defined in every cell and the fraction's denominator is floored
        # above zero, so both isfinite masks were all-True and each allocated a
        # full-size copy per frame for nothing; nanpercentile ignores NaN anyway.
        field_values.append(thin(field.ravel()))
        profile_mean, profile_axis = field_profile_curves(
            field, symmetric_radius_display=False)
        field_profile_limit_kvpm = max(
            field_profile_limit_kvpm,
            field_profile_x_max(profile_mean, profile_axis),
        )
        if initial_field is None:
            initial_field = np.array(field, copy=True)
        fraction_values.append(
            thin(field_for_panel(field, initial_field, "fraction").ravel()))
    # Anchor the color scale to the peak and show a bounded number of decades
    # below it. A plain low percentile can span 10+ decades and flatten the
    # useful structure into saturated blobs.
    def log_window(flat: np.ndarray, decades: float) -> tuple[float, float]:
        hi = float(np.nanpercentile(flat, 99.7)) + 0.2
        lo = max(float(np.nanpercentile(flat, 30.0)), hi - decades)
        return (lo, hi)

    # Release each accumulator before allocating the next concatenation.
    def drain(chunks: list[np.ndarray]) -> np.ndarray:
        flat = np.concatenate(chunks)
        chunks.clear()
        return flat

    if electron_logs:
        electron_limits = log_window(drain(electron_logs), 6.5)
    else:
        electron_limits = (-3.0, 3.0)
    density_limits: dict[str, tuple[float, float] | None] = {
        kind: (log_window(drain(logs), 7.0) if logs else None)
        for kind, logs in density_logs.items()
    }
    field_limits = robust_limits(drain(field_values))
    fraction_flat = drain(fraction_values)
    if fraction_flat.size:
        delta_abs = np.abs(fraction_flat - 1.0)
        half_width = float(np.nanpercentile(delta_abs, 99.5))
        if not math.isfinite(half_width) or half_width <= 0.0:
            half_width = float(np.nanmax(delta_abs)) if delta_abs.size else 0.0
        # Use a symmetric scale around no-change.  This prevents a nearly
        # unchanged field from collapsing into one flat color while avoiding
        # wildly amplifying pure roundoff noise.
        half_width = min(max(half_width, 1.0e-6), 0.25)
    else:
        half_width = 1.0e-3
    fraction_limits = (1.0 - half_width, 1.0 + half_width)
    delta_percent_limits = (-100.0 * half_width, 100.0 * half_width)
    return (
        electron_limits,
        density_limits,
        field_limits,
        fraction_limits,
        delta_percent_limits,
        field_profile_limit_kvpm,
    )


def panel_names(field_panel: str) -> list[str]:
    if field_panel == "all":
        return ["abs", "fraction", "delta"]
    if field_panel == "both":
        return ["abs", "fraction"]
    return [field_panel]


def output_name(
    segment_name: str,
    panel: str,
    r_max_display_m: float | None,
    fps: int,
    symmetric_radius_display: bool,
    grid_panel_count: int = 0,
) -> str:
    """Output filename, deriving the grid tag from the panel count."""
    r_tag = "fullr" if r_max_display_m is None else f"r{int(round(r_max_display_m))}"
    if symmetric_radius_display:
        r_tag = f"sym{r_tag}"
    panel_tag = {
        "abs": "absE",
        "fraction": "Efraction",
        "delta": "EdeltaPct",
        "grid": f"{grid_panel_count}panel",
    }[panel]
    if segment_name == "combined":
        prefix = "profiled_rrea_combined_18s"
    else:
        prefix = f"profiled_rrea_{segment_name}"
    return f"{prefix}_{r_tag}_{panel_tag}_{fps}fps.mp4"


def apply_style(style: str):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if style == "dark":
        plt.style.use("dark_background")
    return plt


def panel_config(
    panel: str,
    field_limits: tuple[float, float],
    fraction_limits: tuple[float, float],
    delta_percent_limits: tuple[float, float],
) -> dict[str, Any]:
    if panel == "fraction":
        return {
            "label": "1 + (|E| - |E$_0$|) / max|E$_0$|",
            "title_prefix": (
                "Field change / peak ambient |E$_0$|, autoscaled around 1"),
            "limits": fraction_limits,
            "cmap": "coolwarm",
        }
    if panel == "delta":
        return {
            "label": "100 * (|E| - |E$_0$|) / max|E$_0$| (%)",
            "title_prefix": "Field change, % of peak ambient |E$_0$|",
            "limits": delta_percent_limits,
            "cmap": "coolwarm",
        }
    return {
        "label": "|E| (kV/m)",
        "title_prefix": "Electric-field magnitude |E|",
        "limits": field_limits,
        "cmap": "viridis",
    }


def draw_panel(axis, array, *, cmap, limits, extent):
    """One density or field image, on the contract every panel shares.

    Six call sites spelled out the same origin/aspect/extent/interpolation
    kwargs, so the preview still and the encoded frame could drift apart on
    colormap or limits while looking like the same picture.
    """
    return axis.imshow(
        array, origin="lower", aspect="auto", extent=extent, cmap=cmap,
        vmin=limits[0], vmax=limits[1], interpolation="bilinear",
    )


def make_frame_plot(
    *,
    electron_density: np.ndarray,
    positive_ion_density: np.ndarray | None,
    field: np.ndarray,
    initial_field: np.ndarray,
    panel: str,
    extent: list[float],
    x_label: str,
    time_us: float,
    total_e: float,
    title: str,
    electron_log_limits: tuple[float, float],
    density_log_limits: dict[str, tuple[float, float] | None],
    field_limits: tuple[float, float],
    fraction_limits: tuple[float, float],
    delta_percent_limits: tuple[float, float],
    output_path: Path,
    dpi: int,
    style: str = "default",
) -> None:
    plt = apply_style(style)

    electron_masked = np.ma.masked_less_equal(electron_density, 0.0)
    electron_log = np.ma.log10(electron_masked)
    cmap_e = plt.get_cmap("magma").copy()
    cmap_e.set_bad("black")
    # This is the 2-panel preview plotter, so of the density kinds it draws only
    # ion; the grid renderer is the one that consumes the whole dict.
    ion_log_limits = density_log_limits.get("ion")
    has_ion_panel = positive_ion_density is not None and ion_log_limits is not None
    if has_ion_panel:
        ion_masked = np.ma.masked_less_equal(positive_ion_density, 0.0)
        ion_log = np.ma.log10(ion_masked)
        cmap_i = plt.get_cmap(DENSITY_PANELS["ion"].cmap).copy()
        cmap_i.set_bad("black")

    config = panel_config(panel, field_limits, fraction_limits, delta_percent_limits)
    field_plot = field_for_panel(field, initial_field, panel)
    field_title = f"{config['title_prefix']}\ntime = {time_us:.3f} us"

    fig, axes = plt.subplots(
        1,
        3 if has_ion_panel else 2,
        figsize=(21.5, 6.2) if has_ion_panel else (15.5, 6.4),
        constrained_layout=True,
    )
    im0 = draw_panel(axes[0], electron_log, cmap=cmap_e,
                     limits=electron_log_limits, extent=extent)
    axes[0].set_title(f"Energetic electrons, E >= 1 MeV\nN_view = {total_e:.3e}")
    axes[0].set_xlabel(x_label)
    axes[0].set_ylabel("altitude MSL (km)")
    cb0 = fig.colorbar(im0, ax=axes[0])
    cb0.set_label("log10 smoothed weighted density (m^-3)")

    field_axis = axes[2] if has_ion_panel else axes[1]
    if has_ion_panel:
        im1 = draw_panel(axes[1], ion_log, cmap=cmap_i,
                         limits=ion_log_limits, extent=extent)
        axes[1].set_title(DENSITY_PANELS["ion"].title)
        axes[1].set_xlabel(x_label)
        axes[1].set_ylabel("altitude MSL (km)")
        cb1 = fig.colorbar(im1, ax=axes[1])
        cb1.set_label(f"log10 {DENSITY_PANELS['ion'].quantity} density (m^-3)")

    im2 = draw_panel(field_axis, field_plot, cmap=config["cmap"],
                     limits=config["limits"], extent=extent)
    field_axis.set_title(field_title)
    field_axis.set_xlabel(x_label)
    field_axis.set_ylabel("altitude MSL (km)")
    cb2 = fig.colorbar(im2, ax=field_axis)
    cb2.set_label(config["label"])

    fig.suptitle(title, fontsize=13)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def static_field_spike_mask(field0: np.ndarray) -> np.ndarray:
    # Box-boundary sampling artifacts sit at fixed cells and are already present
    # at t = 0, far above the imposed profile; a robust percentile of the
    # pre-physics frame separates them cleanly.
    p99 = float(np.percentile(field0, 99.0))
    threshold = max(1.2 * p99, 1.0e-6)
    return field0 > threshold


def repair_masked_cells(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return values
    out = np.array(values, copy=True)
    nz, nr = out.shape
    for z, r in zip(*np.nonzero(mask), strict=False):
        z0, z1 = max(0, z - 2), min(nz, z + 3)
        r0, r1 = max(0, r - 2), min(nr, r + 3)
        patch = values[z0:z1, r0:r1]
        good = patch[~mask[z0:z1, r0:r1]]
        if good.size:
            out[z, r] = float(np.median(good))
    return out


def electron_log_for_plot(electron_density: np.ndarray) -> np.ma.MaskedArray:
    electron_masked = np.ma.masked_less_equal(electron_density, 0.0)
    return np.ma.log10(electron_masked)


def field_for_panel(field: np.ndarray, initial_field: np.ndarray, panel: str) -> np.ndarray:
    """Field grid as the named panel draws it. One definition, three callers.

    Fraction and delta panels divide field change by the peak initial field,
    one scalar for the frame. This avoids division by near-zero ambient cells
    in the taper and field-free domain top; +1 centres fraction on no change.
    """
    if panel == "abs":
        return field
    # nanmax over a profile that is zero everywhere would still divide by a
    # positive number rather than produce inf.
    scale = max(float(np.nanmax(initial_field)), 1.0e-12)
    relative = (field - initial_field) / scale
    if panel == "fraction":
        return 1.0 + relative
    return 100.0 * relative


def build_frame_plan(frames: list[dict[str, Any]], frame_meta: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # The frames array and frame_meta are in WRITE order; the render map speaks
    # capture_index. The two differ whenever a scheduled capture was skipped
    # (band-boundary steps that coalesce) or the capture is still in flight
    # (mid-run preview), so map capture_index -> row position and drop schedule
    # entries that were never captured; do not index positionally.
    row_by_capture_index = {
        int(row["capture_index"]): position for position, row in enumerate(frame_meta)
    }
    plan: list[dict[str, Any]] = []
    skipped = 0
    for mapping in frames:
        capture_index = int(mapping["capture_index"])
        position = row_by_capture_index.get(capture_index)
        if position is None:
            skipped += 1
            continue
        meta = frame_meta[position]
        plan.append(
            {
                "i0": position,
                "i1": position,
                "alpha": 0.0,
                "time_us": float(meta["time_s"]) * 1.0e6,
                "n_view": float(meta["total_energetic_weight"]),
                "rate_us_per_s": None,
            }
        )
    if skipped:
        print(
            f"frame plan: skipped {skipped} scheduled frames without captured data "
            f"({len(plan)} rendered)",
            file=sys.stderr,
        )
    return plan


def capture_is_complete(
    render_map: dict[str, Any], frame_meta: list[dict[str, Any]]
) -> bool:
    """True when the LAST scheduled capture was actually written, i.e. the
    simulation reached the end of its frame schedule (a final render), as
    opposed to a mid-run preview snapshot of a still-running capture."""
    last_scheduled = max(int(mapping["capture_index"]) for mapping in render_map["frames"])
    captured = {int(row["capture_index"]) for row in frame_meta}
    return last_scheduled in captured


def build_warp_plan(
    frame_meta: list[dict[str, Any]],
    warp_points: list[tuple[float, float]],
    fps: int,
) -> list[dict[str, Any]]:
    times = np.array([float(row["time_s"]) * 1.0e6 for row in frame_meta], dtype=float)
    weights = np.array([float(row["total_energetic_weight"]) for row in frame_meta], dtype=float)
    if np.any(np.diff(times) < 0.0):
        raise SystemExit("frame metadata times are not ascending; cannot build warp plan")
    t = float(times[0])
    t_end = float(times[-1])
    plan: list[dict[str, Any]] = []
    while True:
        upper = int(np.searchsorted(times, t, side="left"))
        if upper <= 0:
            i0, i1, alpha = 0, 0, 0.0
        elif upper >= times.size:
            i0 = i1 = int(times.size - 1)
            alpha = 0.0
        else:
            i0, i1 = upper - 1, upper
            span = times[i1] - times[i0]
            alpha = 0.0 if span <= 0.0 else float((t - times[i0]) / span)
        rate = warp_rate_us_per_s_at(t, warp_points)
        n_view = (1.0 - alpha) * weights[i0] + alpha * weights[i1]
        plan.append(
            {
                "i0": int(i0),
                "i1": int(i1),
                "alpha": float(alpha),
                "time_us": float(t),
                "n_view": float(n_view),
                "rate_us_per_s": float(rate),
            }
        )
        if t >= t_end:
            break
        if len(plan) >= 100000:
            raise SystemExit("warp plan exceeds 100000 movie frames; increase --warp-spec rates")
        t = min(t + rate / float(fps), t_end)
    return plan


def playback_text(rate_us_per_s: float) -> str:
    slowdown = 1.0 / (rate_us_per_s * 1.0e-6)
    magnitude = 10.0 ** math.floor(math.log10(slowdown))
    rounded = round(slowdown / magnitude, 1) * magnitude
    return f"playback {rate_us_per_s:.3g} us/s  (~{rounded:,.0f}x slow motion)"


def start_ffmpeg_rawvideo(
    *,
    width: int,
    height: int,
    fps: int,
    crf: int,
    preset: str,
    output_path: Path,
) -> subprocess.Popen:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise SystemExit("ffmpeg is required to render MP4 videos")
    cmd = [
        ffmpeg,
        "-y",
        "-loglevel",
        "warning",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgba",
        "-s",
        f"{width}x{height}",
        "-framerate",
        str(fps),
        "-i",
        "-",
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-vf",
        "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


# The video's lightcurve strip uses the same band CSVs as the multiband figure,
# so the two deliverables cannot disagree.  The bands are per species and live
# in extract_rrea_band_series.BAND_SPECIES, not here, so there is one owner
# and no transcribed altitude to drift.
LIGHTCURVE_SPECIES = ("photon", "electron")
DEFAULT_LIGHTCURVE_SPECIES = "photon"


# A band series stopping short of the frames is stale. The tolerance is 1% of the
# rendered span, which absorbs the only legitimate shortfall (a live capture
# writing a frame or two between the reduction and the render) while rejecting
# staleness that would misinform.
LIGHTCURVE_STALE_FRACTION = 0.01


def lightcurve_strip_series(
    input_dir: Path,
    species: str = DEFAULT_LIGHTCURVE_SPECIES,
    frames_end_s: float | None = None,
) -> tuple[np.ndarray, np.ndarray, str, str] | None:
    """Band population lightcurve for the strip panel.

    Returns (times_us, population, title, y_label), or None when the capture
    has no band series; a missing strip is not replaced with another quantity.

    ``frames_end_s`` is the last frame time this render will draw; pass it and
    a series that does not reach it is dropped too, for the same reason.
    """
    # Imported here, not at module scope: extract_rrea_band_series imports
    # load_frames from THIS module, so a top-level import would close the cycle.
    from extract_rrea_band_series import (
        BAND_SPECIES,
        load_band_series,
        band_label,
    )

    band = load_band_series(input_dir, species)
    if band is None:
        print(
            f"lightcurve strip disabled: this capture has no "
            f"{BAND_SPECIES[species][3]} -- run "
            f"scripts/extract_rrea_band_series.py on the capture host",
            file=sys.stderr)
        return None
    if frames_end_s is not None and len(band["time_s"]):
        band_end_s = float(band["time_s"][-1])
        span_s = frames_end_s - float(band["time_s"][0])
        if span_s > 0.0 and (
            frames_end_s - band_end_s > LIGHTCURVE_STALE_FRACTION * span_s
        ):
            print(
                f"lightcurve strip disabled: {BAND_SPECIES[species][3]} stops "
                f"at t={band_end_s * 1.0e6:.3f} us but the render draws frames "
                f"to t={frames_end_s * 1.0e6:.3f} us -- re-run "
                f"scripts/extract_rrea_band_series.py on the capture host",
                file=sys.stderr)
            return None
    title, y_label = band_label(band, species)
    return (
        band["time_s"] * 1.0e6,
        band["n_ge1MeV_real"],
        f"{title}, linear)",
        y_label,
    )


def optical_strip_series(
    input_dir: Path,
    cloud_scattering: bool = False,
    observer_alt_m: float = OBSERVER_ALTITUDE_M_MSL,
    cloud_photons: int = 1_000_000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str] | None:
    """337.1 / 777.4 nm optical series for the video's strip panel.

    Reuses compute_rrea_optical_emissions.optical_series on the capture
    directory's own CSV. Returns (times_us, blue_337, red_777, title, y_unit)
    or None when the capture has no reduced CSV (synthetic smokes).
    Presentation: 337 = degradation term (the solid curve of the figures),
    777 = x/3-band center; the strip is drawn on a LINEAR y scale.

    With cloud_scattering the panel switches from source rate (ph/s) to the
    irradiance an observer at observer_alt_m actually records (mW/m^2) after
    cloudscat transport, which is a different quantity on a different scale --
    the title and y unit travel with the series so they cannot disagree. The transported curve uses its
    absolute observer-arrival time axis.
    """
    try:
        from compute_rrea_optical_emissions import (
            optical_series, scattered_irradiance)
    except ImportError as exc:
        print(f"optical panel disabled (import failed: {exc})", file=sys.stderr)
        return None
    if not (input_dir / "rrea_reduced.csv").exists():
        return None
    try:
        series = optical_series(input_dir, verbose=False)
    except SystemExit as exc:
        print(f"optical panel disabled: {exc}", file=sys.stderr)
        return None
    times_us = np.asarray(series["t_s"]) * 1.0e6
    if cloud_scattering:
        scattered = scattered_irradiance(
            series, observer_alt_m=observer_alt_m, n_photons=cloud_photons,
            verbose=False)
        if scattered is not None:
            return (  # scattered_irradiance is uW/m^2; the strip reads mW/m^2
                np.asarray(scattered["t_s"]) * 1.0e6,
                np.asarray(scattered["irr337_lo"]) * 1e-3,
                np.asarray(scattered["irr777"]) * 1e-3,
                f"optical irradiance at {observer_alt_m/1e3:.0f} km MSL "
                "(linear, cloudscat cloud transport): 337.1 nm blue "
                "+ 777.4 nm red",
                "mW/m$^2$",
            )
        print("optical strip: falling back to source rates", file=sys.stderr)
    return (
        times_us,
        np.asarray(series["r337_lo"]),
        np.asarray(series["r777_mid"]),
        "optical source rate (linear): 337.1 nm N2 2P(0,0) blue "
        "(degradation term) + 777.4 nm OI red (x/3-band center)",
        "ph/s",
    )


def render_plan_to_mp4(
    *,
    frames: np.ndarray,
    frame_meta: list[dict[str, Any]],
    indices: dict[str, int | None],
    plan: list[dict[str, Any]],
    panels_grid: list[list[str]],
    title: str,
    output_path: Path,
    fps: int,
    r_slice: slice,
    extent: list[float],
    x_label: str,
    symmetric_radius_display: bool,
    electron_log_limits: tuple[float, float],
    density_log_limits: dict[str, tuple[float, float] | None],
    field_limits: tuple[float, float],
    fraction_limits: tuple[float, float],
    delta_percent_limits: tuple[float, float],
    field_profile_limit_kvpm: float,
    electron_smooth_sigma: float,
    field_smooth_sigma: float,
    crf: int,
    preset: str,
    dpi: int,
    style: str,
    lightcurve_inset: bool,
    repair_field_spikes: bool,
    optical_strip: tuple[np.ndarray, np.ndarray, np.ndarray, str, str] | None = None,
    lightcurve_strip: tuple[np.ndarray, np.ndarray, str] | None = None,
) -> None:
    plt = apply_style(style)

    if not plan:
        raise SystemExit(f"empty frame plan for {output_path}")
    kinds = [kind for row in panels_grid for kind in row]
    electron_index = int(indices["electron"])
    field_index = int(indices["field"])
    # Every density panel resolves the same way; a requested one whose grid the
    # capture lacks is an error, not a silently blank panel.
    density_indices: dict[str, int] = {}
    for kind, panel in DENSITY_PANELS.items():
        if kind not in kinds:
            continue
        index = indices.get(panel.source)
        if index is None or density_log_limits.get(kind) is None:
            raise SystemExit(
                f"{kind} panel requested but the capture has no "
                f"{panel.source} array")
        density_indices[kind] = int(index)
    field0_raw = np.asarray(frames[0, field_index, :, r_slice], dtype=float)
    spike_mask = static_field_spike_mask(field0_raw) if repair_field_spikes else None
    if spike_mask is not None:
        field0_raw = repair_masked_cells(field0_raw, spike_mask)
    initial_field = smooth2d(field0_raw, field_smooth_sigma)
    initial_field = display_array(initial_field, symmetric_radius_display=symmetric_radius_display)
    cmap_e = plt.get_cmap("magma").copy()
    cmap_e.set_bad("black")
    density_cmaps = {}
    for kind, panel in DENSITY_PANELS.items():
        density_cmaps[kind] = plt.get_cmap(panel.cmap).copy()
        density_cmaps[kind].set_bad("black")

    # Smoothed-frame cache: plans advance monotonically in capture index, so a
    # handful of entries covers the current interpolation bracket.
    cache: dict[int, tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]] = {}

    def smoothed_triplet(index: int) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
        cached = cache.get(index)
        if cached is not None:
            return cached
        electron = display_array(
            smooth2d(np.asarray(frames[index, electron_index, :, r_slice], dtype=float), electron_smooth_sigma),
            symmetric_radius_display=symmetric_radius_display,
        )
        densities = {
            kind: display_array(
                smooth2d(np.asarray(frames[index, source, :, r_slice], dtype=float), electron_smooth_sigma),
                symmetric_radius_display=symmetric_radius_display,
            )
            for kind, source in density_indices.items()
        }
        field_raw = np.asarray(frames[index, field_index, :, r_slice], dtype=float)
        if spike_mask is not None:
            field_raw = repair_masked_cells(field_raw, spike_mask)
        field = display_array(
            smooth2d(field_raw, field_smooth_sigma),
            symmetric_radius_display=symmetric_radius_display,
        )
        cache[index] = (electron, densities, field)
        if len(cache) > 4:
            for stale in sorted(cache)[: len(cache) - 4]:
                if stale != index:
                    del cache[stale]
        return cache[index]

    def compose(entry: dict[str, Any]) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
        i0, i1, alpha = int(entry["i0"]), int(entry["i1"]), float(entry["alpha"])
        e0, d0, f0 = smoothed_triplet(i0)
        if i1 == i0 or alpha <= 0.0:
            return e0, d0, f0
        e1, d1, f1 = smoothed_triplet(i1)
        blend = 1.0 - alpha
        electron = blend * e0 + alpha * e1
        densities = {kind: blend * d0[kind] + alpha * d1[kind] for kind in d0}
        field = blend * f0 + alpha * f1
        return electron, densities, field

    def suptitle_for(entry: dict[str, Any]) -> str:
        rate = entry.get("rate_us_per_s")
        if rate is None:
            return title
        return f"{title}   |   {playback_text(float(rate))}"

    first = plan[0]
    electron, densities, field = compose(first)

    config_by_kind = {
        kind: panel_config(kind, field_limits, fraction_limits, delta_percent_limits)
        for kind in kinds
        if kind in ("abs", "fraction", "delta")
    }

    def panel_data(kind: str, electron: np.ndarray, densities: dict[str, np.ndarray], field: np.ndarray) -> np.ndarray:
        if kind == "electron":
            return electron_log_for_plot(electron)
        if kind in DENSITY_PANELS:
            return electron_log_for_plot(densities[kind])
        return field_for_panel(field, initial_field, kind)

    nrows = len(panels_grid)
    ncols = max(len(row) for row in panels_grid)
    if nrows == 1:
        base_w = 21.5 if ncols == 3 else (15.5 if ncols == 2 else 7.6 * ncols)
        figsize = (base_w + 3.4, 6.2 if ncols == 3 else 6.4)
    else:
        # Wide per-column budget because aspect="auto" gives extra width to maps.
        figsize = (10.0 * ncols + 2.6, 6.1 * nrows)
    n_strips = (
        (1 if lightcurve_strip is not None else 0)
        + (1 if optical_strip is not None else 0)
    )
    lightcurve_axis = None
    optical_axis = None
    if n_strips:
        # Extra full-width strips along the bottom, time-synced: the selected
        # band population and optical (337.1/777.4 nm) source rates.
        figsize = (figsize[0], figsize[1] + 1.9 * n_strips)
        fig = plt.figure(figsize=figsize, dpi=dpi, constrained_layout=True)
        gs = fig.add_gridspec(
            nrows + n_strips,
            ncols + 1,
            width_ratios=[0.32] + [1.0] * ncols,
            height_ratios=[1.0] * nrows + [0.30] * n_strips,
        )
        strip_row = nrows
        if lightcurve_strip is not None:
            lightcurve_axis = fig.add_subplot(gs[strip_row, :])
            strip_row += 1
        if optical_strip is not None:
            optical_axis = fig.add_subplot(gs[strip_row, :])
            strip_row += 1
    else:
        fig = plt.figure(figsize=figsize, dpi=dpi, constrained_layout=True)
        gs = fig.add_gridspec(nrows, ncols + 1, width_ratios=[0.32] + [1.0] * ncols)
    # Leftmost column, spanning the image rows: on-axis |E|(z) altitude
    # profile, updated every frame alongside the image panels.
    profile_axis = fig.add_subplot(gs[:nrows, 0])
    axs = [[fig.add_subplot(gs[i, j + 1]) for j in range(ncols)] for i in range(nrows)]
    artists: list[tuple[str, Any, Any]] = []
    electron_axis = None
    for i, row in enumerate(panels_grid):
        for j in range(ncols):
            axis = axs[i][j]
            if j >= len(row):
                axis.set_axis_off()
                continue
            kind = row[j]
            if kind == "electron":
                im = draw_panel(
                    axis, panel_data(kind, electron, densities, field),
                    cmap=cmap_e, limits=electron_log_limits, extent=extent)
                title_obj = axis.set_title(
                    f"Energetic electrons, E >= 1 MeV\nN_view = {first['n_view']:.3e}"
                )
                fig.colorbar(im, cax=axis.inset_axes([1.012, 0.0, 0.022, 1.0])).set_label("log10 smoothed weighted density (m^-3)")
                electron_axis = axis
            elif kind in DENSITY_PANELS:
                kind_limits = density_log_limits[kind]
                panel = DENSITY_PANELS[kind]
                im = draw_panel(
                    axis, panel_data(kind, electron, densities, field),
                    cmap=density_cmaps[kind], limits=kind_limits,
                    extent=extent)
                axis.set_title(panel.title)
                title_obj = None
                fig.colorbar(im, cax=axis.inset_axes([1.012, 0.0, 0.022, 1.0])).set_label(
                    f"log10 {panel.quantity} density (m^-3)")
            else:
                config = config_by_kind[kind]
                im = draw_panel(
                    axis, panel_data(kind, electron, densities, field),
                    cmap=config["cmap"], limits=config["limits"],
                    extent=extent)
                title_obj = axis.set_title(
                    f"{config['title_prefix']}\ntime = {first['time_us']:.3f} us"
                )
                fig.colorbar(im, cax=axis.inset_axes([1.012, 0.0, 0.022, 1.0])).set_label(config["label"])
            axis.set_xlabel(x_label)
            axis.set_ylabel("altitude MSL (km)")
            artists.append((kind, im, title_obj))

    # Field profile panel: the radial mean and the user-labelled r=0 slice
    # share one |E| scale. With symmetric display the mirrored half duplicates
    # the physical columns, leaving the mean unchanged; field_profile_curves
    # selects the centre-right copy of the first physical radial cell.
    nz_display = field.shape[0]
    dz_km = (extent[3] - extent[2]) / nz_display
    alt_centers_km = extent[2] + (np.arange(nz_display) + 0.5) * dz_km
    profile_initial, profile_axis_initial = field_profile_curves(
        initial_field,
        symmetric_radius_display=symmetric_radius_display,
    )
    profile_now, profile_axis_now = field_profile_curves(
        field,
        symmetric_radius_display=symmetric_radius_display,
    )
    dark_style = style == "dark"
    profile_axis.plot(
        profile_initial,
        alt_centers_km,
        lw=0.9,
        ls="--",
        color="0.6",
        label="t = 0 radial mean",
    )
    (profile_line,) = profile_axis.plot(
        profile_now,
        alt_centers_km,
        lw=1.4,
        color="#40c4ff" if dark_style else "tab:blue",
        label="radial mean now",
    )
    (axis_profile_line,) = profile_axis.plot(
        profile_axis_now,
        alt_centers_km,
        lw=1.4,
        color="#ff5252" if dark_style else "tab:red",
        label="r = 0 now",
    )
    # One fixed scale, measured over every captured profile before rendering,
    # prevents late compression fronts from clipping or making the axis shift.
    profile_axis.set_xlim(
        0.0,
        field_profile_limit_kvpm,
    )
    profile_axis.set_ylim(extent[2], extent[3])
    profile_axis.set_xlabel("|E| (kV/m)")
    profile_axis.set_ylabel("altitude MSL (km)")
    profile_axis.set_title("Field magnitude profiles |E|(z)")
    profile_axis.legend(loc="lower right", fontsize=7, framealpha=0.4)

    suptitle = fig.suptitle(suptitle_for(first), fontsize=13)

    cursor_min_us = None
    cursors: list[Any] = []
    # Draw the inset only when the full-width lightcurve strip is absent.
    if lightcurve_inset and lightcurve_strip is None:
        times_us = np.array([float(row["time_s"]) * 1.0e6 for row in frame_meta], dtype=float)
        dark = style == "dark"
        line_color = "#40c4ff" if dark else "tab:blue"
        cursor_color = "white" if dark else "black"
        inset_host = electron_axis if electron_axis is not None else axs[0][0]
        cursor_min_us = float(times_us.min()) if times_us.size else 0.0

        # Whole-domain >=1 MeV population (always available from the frame metadata).
        n_series = np.array([float(row["total_energetic_weight"]) for row in frame_meta], dtype=float)
        n_positive = n_series > 0.0
        # Common x-range for every inset (both panels share one time scale): the span
        # over which the whole-domain energetic population is nonzero -- it outlives
        # the thin-plane crossing, so it bounds the activity for both panels.
        if n_positive.any():
            last_active_us = float(times_us[n_positive].max())
        elif times_us.size:
            last_active_us = float(times_us.max())
        else:
            last_active_us = cursor_min_us + 1.0
        span = max(last_active_us - cursor_min_us, 1.0e-3)
        cursor_max_us = last_active_us + 0.08 * span

        def finish_inset(inset_ax: Any, ylabel: str) -> None:
            inset_ax.patch.set_facecolor("black" if dark else "white")
            inset_ax.patch.set_alpha(0.6)
            # Visible frame so the panel reads as a box even where its data is a
            # near-zero flat line on an otherwise-black upper region of the panel.
            for spine in inset_ax.spines.values():
                spine.set_edgecolor("0.75" if dark else "0.3")
                spine.set_linewidth(0.6)
            inset_ax.set_xlim(cursor_min_us, cursor_max_us)
            inset_ax.tick_params(labelsize=6, colors="0.85" if dark else "0.15")
            inset_ax.set_xlabel("t (us)", fontsize=6, color="0.9" if dark else "0.1")
            inset_ax.set_ylabel(ylabel, fontsize=6, color="0.9" if dark else "0.1")
            cursors.append(
                (
                    inset_ax.axvline(
                        min(max(first["time_us"], cursor_min_us), cursor_max_us),
                        lw=0.8,
                        color=cursor_color,
                        alpha=0.85,
                    ),
                    float(cursor_min_us),
                    float(cursor_max_us),
                )
            )

        # Single inset: whole-domain N(>=1 MeV) on a linear scale. The log-scale
        # plane-crossing-rate panel was dropped -- the linear population curve reads
        # more clearly (the giant relaxation pulse dominates as one sharp spike).
        inset = inset_host.inset_axes([0.55, 0.74, 0.43, 0.24])
        if n_positive.any():
            inset.plot(
                times_us, np.where(n_positive, n_series, np.nan), lw=0.9, color=line_color
            )
        finish_inset(inset, "N >= 1 MeV (linear)")

    strip_time_xlim_us = shared_strip_time_limits_us(
        None if lightcurve_strip is None else lightcurve_strip[0],
        None if optical_strip is None else optical_strip[0],
    ) if lightcurve_strip is not None or optical_strip is not None else None

    if lightcurve_axis is not None and lightcurve_strip is not None:
        lc_t_us, lc_n, lc_label, lc_y_label = lightcurve_strip
        dark = style == "dark"
        lightcurve_axis.plot(
            lc_t_us, lc_n, lw=0.8, color="#40c4ff" if dark else "tab:blue"
        )
        assert strip_time_xlim_us is not None
        lightcurve_axis.set_xlim(*strip_time_xlim_us)
        lightcurve_axis.set_ylim(bottom=0.0)
        if optical_axis is None:
            lightcurve_axis.set_xlabel("t (us)", fontsize=8)
        lightcurve_axis.set_ylabel(lc_y_label, fontsize=8)
        lightcurve_axis.tick_params(labelsize=7)
        lightcurve_axis.set_title(lc_label, fontsize=8)
        lightcurve_axis.grid(alpha=0.2)
        cursors.append(
            (
                lightcurve_axis.axvline(
                    min(max(first["time_us"], strip_time_xlim_us[0]), strip_time_xlim_us[1]),
                    lw=0.9,
                    color="white" if dark else "black",
                    alpha=0.85,
                ),
                strip_time_xlim_us[0],
                strip_time_xlim_us[1],
            )
        )

    if optical_axis is not None and optical_strip is not None:
        opt_t_us, opt_337, opt_777, opt_label, opt_unit = optical_strip
        dark = style == "dark"
        optical_axis.plot(
            opt_t_us, opt_337, lw=0.8,
            color="#64b5f6" if dark else "tab:blue", label="337.1 nm",
        )
        optical_axis.plot(
            opt_t_us, opt_777, lw=0.8,
            color="#ef5350" if dark else "tab:red", label="777.4 nm",
        )
        assert strip_time_xlim_us is not None
        optical_axis.set_xlim(*strip_time_xlim_us)
        optical_axis.set_ylim(bottom=0.0)
        optical_axis.set_xlabel("t (us)", fontsize=8)
        optical_axis.set_ylabel(opt_unit, fontsize=8)
        optical_axis.tick_params(labelsize=7)
        optical_axis.set_title(opt_label, fontsize=8)
        optical_axis.grid(alpha=0.2)
        optical_axis.legend(loc="upper right", fontsize=7)
        cursors.append(
            (
                optical_axis.axvline(
                    min(max(first["time_us"], strip_time_xlim_us[0]), strip_time_xlim_us[1]),
                    lw=0.9,
                    color="white" if dark else "black",
                    alpha=0.85,
                ),
                strip_time_xlim_us[0],
                strip_time_xlim_us[1],
            )
        )

    fig.canvas.draw()
    buffer = np.asarray(fig.canvas.buffer_rgba())
    height, width = buffer.shape[0], buffer.shape[1]
    encoder = start_ffmpeg_rawvideo(
        width=width,
        height=height,
        fps=fps,
        crf=crf,
        preset=preset,
        output_path=output_path,
    )
    assert encoder.stdin is not None
    try:
        encoder.stdin.write(buffer.data)
        for entry in plan[1:]:
            electron, densities, field = compose(entry)
            profile_now, profile_axis_now = field_profile_curves(
                field,
                symmetric_radius_display=symmetric_radius_display,
            )
            profile_line.set_data(profile_now, alt_centers_km)
            axis_profile_line.set_data(profile_axis_now, alt_centers_km)
            for kind, im, title_obj in artists:
                im.set_data(panel_data(kind, electron, densities, field))
                if title_obj is None:
                    continue
                if kind == "electron":
                    title_obj.set_text(
                        f"Energetic electrons, E >= 1 MeV\nN_view = {entry['n_view']:.3e}"
                    )
                else:
                    title_obj.set_text(
                        f"{config_by_kind[kind]['title_prefix']}\ntime = {entry['time_us']:.3f} us"
                    )
            suptitle.set_text(suptitle_for(entry))
            for cursor, x_lo, x_hi in cursors:
                cursor_x = min(max(float(entry["time_us"]), x_lo), x_hi)
                cursor.set_xdata([cursor_x, cursor_x])
            fig.canvas.draw()
            buffer = np.asarray(fig.canvas.buffer_rgba())
            if buffer.shape[0] != height or buffer.shape[1] != width:
                raise SystemExit(
                    f"canvas size changed mid-render ({width}x{height} -> "
                    f"{buffer.shape[1]}x{buffer.shape[0]}); cannot stream to ffmpeg"
                )
            encoder.stdin.write(buffer.data)
    except BrokenPipeError:
        # ffmpeg exited early; fall through to wait() so its real exit code
        # (surfaced below) is what the user sees -- not this pipe write or a
        # second BrokenPipeError from close().
        pass
    finally:
        try:
            encoder.stdin.close()
        except (BrokenPipeError, OSError):
            pass
        return_code = encoder.wait()
    plt.close(fig)
    if return_code != 0:
        raise SystemExit(
            f"ffmpeg failed with exit code {return_code} for {output_path} "
            "(if frames stopped early this is usually ffmpeg dying mid-stream; "
            "see its stderr above)"
        )
    if not output_path.exists() or output_path.stat().st_size <= 0:
        raise SystemExit(f"failed to render nonempty video: {output_path}")


def _render_output_job(
    spec: dict[str, Any],
    preloaded: tuple[np.ndarray, list[dict[str, Any]], dict[str, Any], dict[str, Any]] | None = None,
) -> str:
    input_dir = Path(spec["input_dir"])
    if preloaded is None:
        frames, frame_meta, capture_meta, _render_map = load_frames(input_dir)
    else:
        frames, frame_meta, capture_meta, _render_map = preloaded
    r_slice, base_extent = crop_columns(capture_meta, spec["r_max_display_m"])
    extent, x_label = display_extent_and_xlabel(
        base_extent,
        symmetric_radius_display=spec["symmetric_radius_display"],
    )
    indices = frame_array_indices(capture_meta, frames)
    ion_limits = spec["ion_log_limits"]
    # A plan without per-species density windows uses its ion window for the
    # available density panels.
    density_limits = {
        kind: (tuple(value) if value is not None else None)
        for kind, value in (spec.get("density_log_limits")
                            or {"ion": ion_limits}).items()
    }
    # The spec carries strips built once in render(), avoiding duplicate
    # cloudscat work across output jobs.
    optical_strip = spec.get("optical_strip")
    lightcurve_strip = spec.get("lightcurve_strip")
    render_plan_to_mp4(
        frames=frames,
        frame_meta=frame_meta,
        indices=indices,
        plan=spec["plan"],
        panels_grid=spec["panels_grid"],
        title=spec["title"],
        output_path=Path(spec["output_path"]),
        fps=int(spec["fps"]),
        r_slice=r_slice,
        extent=extent,
        x_label=x_label,
        symmetric_radius_display=spec["symmetric_radius_display"],
        electron_log_limits=tuple(spec["electron_log_limits"]),
        density_log_limits=density_limits,
        field_limits=tuple(spec["field_limits"]),
        fraction_limits=tuple(spec["fraction_limits"]),
        delta_percent_limits=tuple(spec["delta_percent_limits"]),
        field_profile_limit_kvpm=float(spec["field_profile_limit_kvpm"]),
        electron_smooth_sigma=float(spec["electron_smooth_sigma"]),
        field_smooth_sigma=float(spec["field_smooth_sigma"]),
        crf=int(spec["crf"]),
        preset=spec["preset"],
        dpi=int(spec["dpi"]),
        style=spec["style"],
        lightcurve_inset=bool(spec["lightcurve_inset"]),
        repair_field_spikes=bool(spec["repair_field_spikes"]),
        optical_strip=optical_strip,
        lightcurve_strip=lightcurve_strip,
    )
    return spec["output_path"]


def write_preview_pngs(
    *,
    frames: np.ndarray,
    frame_meta: list[dict[str, Any]],
    capture_meta: dict[str, Any],
    indices: dict[str, int | None],
    output_dir: Path,
    panels: list[str],
    r_slice: slice,
    extent: list[float],
    x_label: str,
    symmetric_radius_display: bool,
    electron_log_limits: tuple[float, float],
    density_log_limits: dict[str, tuple[float, float] | None],
    field_limits: tuple[float, float],
    fraction_limits: tuple[float, float],
    delta_percent_limits: tuple[float, float],
    electron_smooth_sigma: float,
    field_smooth_sigma: float,
    dpi: int,
    style: str,
    repair_field_spikes: bool,
) -> None:
    captured_times = np.array([float(row["time_s"]) for row in frame_meta])
    electron_index = int(indices["electron"])
    field_index = int(indices["field"])
    ion_index = indices.get("positive_ion")
    field0_raw = np.asarray(frames[0, field_index, :, r_slice], dtype=float)
    spike_mask = static_field_spike_mask(field0_raw) if repair_field_spikes else None
    if spike_mask is not None:
        field0_raw = repair_masked_cells(field0_raw, spike_mask)
    initial_field = smooth2d(field0_raw, field_smooth_sigma)
    initial_field = display_array(initial_field, symmetric_radius_display=symmetric_radius_display)
    for panel in panels:
        preview_dir = output_dir / {
            "abs": "previews_absE",
            "fraction": "previews_Efraction",
            "delta": "previews_EdeltaPct",
        }[panel]
        preview_dir.mkdir(parents=True, exist_ok=True)
        for target_time in PREVIEW_TIMES_S:
            index = int(np.argmin(np.abs(captured_times - target_time)))
            electron_density = smooth2d(
                np.asarray(frames[index, electron_index, :, r_slice], dtype=float),
                electron_smooth_sigma,
            )
            electron_density = display_array(
                electron_density,
                symmetric_radius_display=symmetric_radius_display,
            )
            positive_ion_density = (
                smooth2d(
                    np.asarray(frames[index, int(ion_index), :, r_slice], dtype=float),
                    electron_smooth_sigma,
                )
                if ion_index is not None
                else None
            )
            if positive_ion_density is not None:
                positive_ion_density = display_array(
                    positive_ion_density,
                    symmetric_radius_display=symmetric_radius_display,
                )
            field_raw = np.asarray(frames[index, field_index, :, r_slice], dtype=float)
            if spike_mask is not None:
                field_raw = repair_masked_cells(field_raw, spike_mask)
            field = smooth2d(field_raw, field_smooth_sigma)
            field = display_array(field, symmetric_radius_display=symmetric_radius_display)
            time_us = float(frame_meta[index]["time_s"]) * 1.0e6
            total_e = float(frame_meta[index]["total_energetic_weight"])
            make_frame_plot(
                electron_density=electron_density,
                positive_ion_density=positive_ion_density,
                field=field,
                initial_field=initial_field,
                panel=panel,
                extent=extent,
                x_label=x_label,
                time_us=time_us,
                total_e=total_e,
                title="Profiled RREA preview",
                electron_log_limits=electron_log_limits,
                density_log_limits=density_log_limits,
                field_limits=field_limits,
                fraction_limits=fraction_limits,
                delta_percent_limits=delta_percent_limits,
                output_path=preview_dir / f"preview_{target_time * 1e6:08.3f}us.png",
                dpi=dpi,
                style=style,
            )


def render(
    input_dir: Path,
    output_dir: Path,
    *,
    write_compact_npz: bool,
    r_max_display_m: float | None,
    symmetric_radius_display: bool,
    electron_smooth_sigma: float,
    field_smooth_sigma: float,
    field_panel: str,
    density_panels: list[str],
    layout: str,
    crf: int,
    preset: str,
    dpi: int,
    preview_only: bool,
    style: str,
    lightcurve_species: str = DEFAULT_LIGHTCURVE_SPECIES,
    time_warp: bool,
    warp_only: bool,
    warp_spec: str,
    lightcurve_inset: bool,
    repair_field_spikes: bool,
    cloud_scattering: bool | None,
    weak_tgf_threshold: float,
    jobs: int,
    electron_log_limits_override: tuple[float, float] | None,
    positive_ion_log_limits_override: tuple[float, float] | None,
    field_limits_kvpm_override: tuple[float, float] | None,
    field_fraction_limits_override: tuple[float, float] | None,
    field_delta_percent_limits_override: tuple[float, float] | None,
    field_profile_limit_kvpm_override: float | None,
) -> dict[str, Any]:
    frames, frame_meta, capture_meta, render_map = load_frames(input_dir)
    if cloud_scattering is None:
        # Cloud transport is expensive, so mid-run preview snapshots skip it and
        # show the raw source rate; a COMPLETE capture (simulation finished,
        # final render) gets the ER-2 irradiance panel.  Force either way
        # with --cloud-scattering / --no-cloud-scattering.
        cloud_scattering = capture_is_complete(render_map, frame_meta)
        print(
            "cloud scattering auto: "
            + ("ON (capture complete -- final render)" if cloud_scattering
               else "OFF (mid-run preview; --cloud-scattering forces it)"),
            file=sys.stderr,
        )
    # The strip series depend only on input_dir and cloud_scattering. Compute
    # them once before per-segment work; the external cache has no writer lock.
    optical_strip = optical_strip_series(input_dir, cloud_scattering=cloud_scattering)
    lightcurve_strip = lightcurve_strip_series(
        input_dir,
        lightcurve_species,
        frames_end_s=max(float(row["time_s"]) for row in frame_meta),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    if r_max_display_m is None:
        # The capture spans the full domain; the default display crop focuses
        # on the strong-field region. Pass --r-max-display-m (up to the
        # captured radius) for another window -- re-render, never rerun.
        r_max_display_m = VIDEO_DISPLAY_DOMAIN_FRACTION * float(
            capture_meta["r_max_m"]
        )
    r_slice, base_extent = crop_columns(capture_meta, r_max_display_m)
    extent, x_label = display_extent_and_xlabel(
        base_extent,
        symmetric_radius_display=symmetric_radius_display,
    )
    indices = frame_array_indices(capture_meta, frames)
    panels = panel_names(field_panel)
    have_ion = indices["positive_ion"] is not None
    # A requested density map the capture lacks is an error, not a blank panel.
    for kind in density_panels:
        if indices.get(DENSITY_PANELS[kind].source) is None:
            raise SystemExit(
                f"--density-panels asked for {kind} but this capture has no "
                f"{DENSITY_PANELS[kind].source} array")
    if layout == "grid":
        if not have_ion:
            raise SystemExit(
                "--layout grid needs the positive-ion capture array; "
                "use --layout row for captures without it"
            )
        needed_field_kinds = {"abs", "fraction"}
    else:
        needed_field_kinds = set(panels)
    missing_limits = (
        electron_log_limits_override is None
        or (have_ion and positive_ion_log_limits_override is None)
        or ("abs" in needed_field_kinds and field_limits_kvpm_override is None)
        or ("fraction" in needed_field_kinds and field_fraction_limits_override is None)
        or ("delta" in needed_field_kinds and field_delta_percent_limits_override is None)
        or field_profile_limit_kvpm_override is None
    )
    if missing_limits:
        (
            electron_limits,
            density_limits,
            field_limits,
            fraction_limits,
            delta_percent_limits,
            field_profile_limit_kvpm,
        ) = compute_limits(
            frames,
            indices,
            r_slice,
            electron_smooth_sigma=electron_smooth_sigma,
            field_smooth_sigma=field_smooth_sigma,
        )
    else:
        print("all needed color limits fixed on the command line; skipping the limits pass", file=sys.stderr)
        electron_limits = (-3.0, 3.0)
        density_limits = {kind: None for kind in DENSITY_PANELS}
        field_limits = (0.0, 1.0)
        fraction_limits = (0.9, 1.1)
        delta_percent_limits = (-10.0, 10.0)
        field_profile_limit_kvpm = 1.0
    if electron_log_limits_override is not None:
        electron_limits = electron_log_limits_override
    if positive_ion_log_limits_override is not None:
        density_limits["ion"] = positive_ion_log_limits_override
    if not have_ion:
        density_limits["ion"] = None
    ion_limits = density_limits["ion"]
    if field_limits_kvpm_override is not None:
        field_limits = field_limits_kvpm_override
    if field_fraction_limits_override is not None:
        fraction_limits = field_fraction_limits_override
    if field_delta_percent_limits_override is not None:
        delta_percent_limits = field_delta_percent_limits_override
    if field_profile_limit_kvpm_override is not None:
        field_profile_limit_kvpm = field_profile_limit_kvpm_override
    fps = int(render_map.get("fps", 24))
    peak_energetic = max(
        (float(row["total_energetic_weight"]) for row in frame_meta),
        default=0.0,
    )
    weak_tgf = weak_tgf_threshold > 0.0 and 0.0 < peak_energetic < weak_tgf_threshold
    strong_tgf = weak_tgf_threshold > 0.0 and peak_energetic >= weak_tgf_threshold
    if weak_tgf:
        tgf_tag = " -- weak TGF"
    elif strong_tgf:
        tgf_tag = " -- TGF"
    else:
        tgf_tag = ""

    def job_spec(
        plan: list[dict[str, Any]],
        panels_grid: list[list[str]],
        title: str,
        output_path: Path,
    ) -> dict[str, Any]:
        return {
            "input_dir": str(input_dir),
            "plan": plan,
            "panels_grid": panels_grid,
            "title": title,
            "output_path": str(output_path),
            "fps": fps,
            "r_max_display_m": r_max_display_m,
            "symmetric_radius_display": symmetric_radius_display,
            "electron_log_limits": list(electron_limits),
            "ion_log_limits": None if ion_limits is None else list(ion_limits),
            "density_log_limits": {
                kind: (None if limits is None else list(limits))
                for kind, limits in density_limits.items()
            },
            "field_limits": list(field_limits),
            "fraction_limits": list(fraction_limits),
            "delta_percent_limits": list(delta_percent_limits),
            "field_profile_limit_kvpm": field_profile_limit_kvpm,
            "electron_smooth_sigma": electron_smooth_sigma,
            "field_smooth_sigma": field_smooth_sigma,
            "crf": crf,
            "preset": preset,
            "dpi": dpi,
            "style": style,
            "lightcurve_inset": lightcurve_inset,
            "repair_field_spikes": repair_field_spikes,
            "cloud_scattering": cloud_scattering,
            "optical_strip": optical_strip,
            "lightcurve_strip": lightcurve_strip,
        }

    density_row = [kind for kind in density_panels
                   if indices.get(DENSITY_PANELS[kind].source) is not None]
    if layout == "grid":
        # Wrap all selected density and field panels into the nearest-square grid.
        ordered = ["electron"] + density_row + ["abs", "fraction"]
        ncols_grid = math.ceil(math.sqrt(len(ordered)))
        grid_rows = [ordered[start:start + ncols_grid]
                     for start in range(0, len(ordered), ncols_grid)]
        panel_jobs: list[tuple[str, list[list[str]]]] = [("grid", grid_rows)]
    else:
        panel_jobs = [
            (panel, [["electron"] + density_row + [panel]])
            for panel in panels
        ]

    job_specs: list[dict[str, Any]] = []
    warp_movie_frames = None
    if not preview_only:
        for panel_name, panels_grid_for_job in panel_jobs:
            if not warp_only:
                # A video is ONE file (deliverable-outputs rule): every
                # captured frame in schedule order, at the fixed capture spacing.
                combined_plan = build_frame_plan(render_map["frames"], frame_meta)
                if combined_plan:
                    combined_path = output_dir / output_name(
                        "combined",
                        panel_name,
                        r_max_display_m,
                        fps,
                        symmetric_radius_display,
                        sum(len(row) for row in panels_grid_for_job),
                    )
                    job_specs.append(
                        job_spec(
                            combined_plan,
                            panels_grid_for_job,
                            f"Profiled RREA{tgf_tag}, nonlinear playback time",
                            combined_path,
                        )
                    )
            if time_warp:
                warp_points = parse_warp_control_points(warp_spec, flag_name="--warp-spec")
                warp_plan = build_warp_plan(frame_meta, warp_points, fps)
                warp_movie_frames = len(warp_plan)
                warp_path = output_dir / output_name(
                    "cinematic",
                    panel_name,
                    r_max_display_m,
                    fps,
                    symmetric_radius_display,
                    sum(len(row) for row in panels_grid_for_job),
                )
                job_specs.append(
                    job_spec(
                        warp_plan,
                        panels_grid_for_job,
                        f"Profiled RREA{tgf_tag}, time-warped playback",
                        warp_path,
                    )
                )

    outputs: list[str] = []
    if job_specs:
        if jobs > 1 and len(job_specs) > 1:
            with ProcessPoolExecutor(max_workers=min(jobs, len(job_specs))) as pool:
                for finished in pool.map(_render_output_job, job_specs):
                    print(f"rendered {finished}", file=sys.stderr)
                    outputs.append(finished)
        else:
            preloaded = (frames, frame_meta, capture_meta, render_map)
            for spec in job_specs:
                outputs.append(_render_output_job(spec, preloaded=preloaded))
                print(f"rendered {outputs[-1]}", file=sys.stderr)

    write_preview_pngs(
        frames=frames,
        frame_meta=frame_meta,
        capture_meta=capture_meta,
        indices=indices,
        output_dir=output_dir,
        panels=["abs", "fraction"] if layout == "grid" else panels,
        r_slice=r_slice,
        extent=extent,
        x_label=x_label,
        symmetric_radius_display=symmetric_radius_display,
        electron_log_limits=electron_limits,
        density_log_limits=density_limits,
        field_limits=field_limits,
        fraction_limits=fraction_limits,
        delta_percent_limits=delta_percent_limits,
        electron_smooth_sigma=electron_smooth_sigma,
        field_smooth_sigma=field_smooth_sigma,
        dpi=dpi,
        style=style,
        repair_field_spikes=repair_field_spikes,
    )
    compact_npz = None
    if write_compact_npz:
        compact_npz = output_dir / "profiled_video_frames_compact.npz"
        np.savez_compressed(
            compact_npz,
            frames=np.asarray(frames),
            frame_metadata=np.array(json.dumps(frame_meta)),
            capture_metadata=np.array(json.dumps(capture_meta)),
            render_map=np.array(json.dumps(render_map)),
        )
    summary = {
        "status": "rendered_previews" if preview_only else "rendered",
        "frame_count": int(frames.shape[0]),
        "arrays_per_frame": capture_meta.get("arrays_per_frame"),
        "fps": fps,
        "render_settings": {
            "r_max_display_m": r_max_display_m,
            "symmetric_radius_display": symmetric_radius_display,
            "electron_smooth_sigma": electron_smooth_sigma,
            "field_smooth_sigma": field_smooth_sigma,
            "field_panel": field_panel,
            "layout": layout,
            "crf": crf,
            "preset": preset,
            "dpi": dpi,
            "preview_only": preview_only,
            "style": style,
            "time_warp": time_warp,
            "warp_only": warp_only,
            "warp_spec": warp_spec if time_warp else None,
            "warp_movie_frames": warp_movie_frames,
            "lightcurve_inset": lightcurve_inset,
            "repair_field_spikes": repair_field_spikes,
            "cloud_scattering": cloud_scattering,
            "weak_tgf_threshold": weak_tgf_threshold,
            "peak_energetic_weight": peak_energetic,
            "weak_tgf": weak_tgf,
            "tgf_class": "weak" if weak_tgf else ("strong" if strong_tgf else "none"),
            "jobs": jobs,
            "fixed_color_limits": {
                "electron_log_limits": electron_log_limits_override,
                "positive_ion_log_limits": positive_ion_log_limits_override,
                "field_limits_kVpm": field_limits_kvpm_override,
                "field_fraction_limits": field_fraction_limits_override,
                "field_delta_percent_limits": field_delta_percent_limits_override,
                "field_profile_limit_kVpm": field_profile_limit_kvpm_override,
            },
            "field_profile_limit_kVpm": field_profile_limit_kvpm,
        },
        "display_extent": extent,
        "base_display_extent": base_extent,
        "electron_log_limits": electron_limits,
        "positive_ion_log_limits": ion_limits,
        "field_limits_kVpm": field_limits,
        "field_fraction_limits": fraction_limits,
        "field_delta_percent_limits": delta_percent_limits,
        "outputs": outputs,
        "compact_npz": str(compact_npz) if compact_npz else None,
    }
    (output_dir / "profiled_video_render_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary


def _band_help() -> str:
    """Bands for --help, read from their owner so they cannot drift."""
    from extract_rrea_band_series import BAND_SPECIES

    return ", ".join(f"{name} {lo / 1e3:g}-{hi / 1e3:g} km MSL r<={r:g} m"
                     for name, (lo, hi, r, _csv) in sorted(BAND_SPECIES.items()))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    # Opt-in because this duplicates the existing compressed frame stream.
    parser.add_argument("--compact-npz", action="store_true")
    # None shows VIDEO_DISPLAY_DOMAIN_FRACTION of the captured domain; an
    # explicit radius changes only the render window.
    parser.add_argument("--r-max-display-m", type=float, default=None)
    parser.add_argument(
        "--symmetric-radius-display",
        action="store_true",
        help="Mirror the RZ r>=0 arrays for display as -r..+r without changing physics data.",
    )
    parser.add_argument("--electron-smooth-sigma", type=float, default=0.7)
    parser.add_argument("--field-smooth-sigma", type=float, default=0.3)
    parser.add_argument("--field-panel", choices=PANEL_CHOICES, default="both")
    parser.add_argument(
        "--density-panels", default="ion,photon,positron",
        help="comma-separated density maps to render from "
        f"{sorted(DENSITY_PANELS)} (default: %(default)s). photon and positron "
        "are the >= 1 MeV populations, same threshold as the electron map.")
    parser.add_argument(
        "--layout",
        choices=("grid", "row"),
        default="grid",
        help="Video panel layout. 'grid' (default) packs the electron map, all "
        "selected density maps, |E| and field fraction into one near-square MP4 "
        "per segment. 'row' writes one horizontal MP4 per --field-panel choice.",
    )
    parser.add_argument("--crf", type=int, default=14)
    parser.add_argument("--preset", default="slow")
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument(
        "--lightcurve-species", choices=sorted(LIGHTCURVE_SPECIES),
        default=DEFAULT_LIGHTCURVE_SPECIES,
        help="species whose band lightcurve the strip draws "
             "(default %(default)s; bands " + _band_help() + "). "
             "The strip is dropped, never refilled from another quantity, if "
             "that species has no band CSV")
    parser.add_argument("--preview-only", action="store_true")
    parser.add_argument(
        "--cloud-scattering", dest="cloud_scattering", action="store_true",
        help="force the optical strip to the irradiance an ER-2 observer "
             "records after cloudscat cloud transport (mW/m2)")
    parser.add_argument(
        "--no-cloud-scattering", dest="cloud_scattering", action="store_false",
        help="force the optical strip to the raw source photon rate (ph/s)")
    # Neither flag: derived in render() -- ON when the capture is complete
    # (final render), OFF for a mid-run preview snapshot.
    parser.set_defaults(cloud_scattering=None)
    parser.add_argument(
        "--time-warp",
        action="store_true",
        help="Additionally render one time-warped 'cinematic' MP4 per panel with "
        "nonlinear playback speed and data-space frame interpolation.",
    )
    parser.add_argument(
        "--warp-only",
        action="store_true",
        help="Render only the time-warped MP4s (implies --time-warp).",
    )
    parser.add_argument(
        "--warp-spec",
        default=DEFAULT_WARP_SPEC,
        help="Comma-separated 'sim_time_us:playback_rate_us_per_s' control points; "
        "rates are interpolated linearly in simulation time between points. "
        f"Default: {DEFAULT_WARP_SPEC}",
    )
    parser.add_argument(
        "--weak-tgf-threshold",
        type=float,
        default=1.0e16,
        help="Tag titles with 'weak TGF' when the peak N(>=1 MeV) over the run "
        "stays below this count (default 1e16; 0 disables the tag).",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Render this many MP4 outputs in parallel worker processes.",
    )
    parser.add_argument(
        "--electron-log-limits",
        help="Fixed log10 color limits for energetic-electron density, e.g. '-12,2'.",
    )
    parser.add_argument(
        "--positive-ion-log-limits",
        help="Fixed log10 color limits for positive-ion density, e.g. '-9,6.2'.",
    )
    parser.add_argument(
        "--field-limits-kvpm",
        help="Fixed absolute-field color limits in kV/m, e.g. '0,81'.",
    )
    parser.add_argument(
        "--field-fraction-limits",
        help="Fixed |E|/|E_initial| color limits, e.g. '0.9995,1.0005'.",
    )
    parser.add_argument(
        "--field-delta-percent-limits",
        help="Fixed percent-change color limits, e.g. '-0.05,0.05'.",
    )
    parser.add_argument(
        "--field-profile-limit-kvpm",
        type=float,
        help=(
            "Fixed upper x-axis limit for the |E|(z) profile in kV/m. "
            "By default it is measured over every captured frame with 50%% headroom."
        ),
    )
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = args.input_dir / "rendered_video"
    if args.r_max_display_m is not None and args.r_max_display_m <= 0.0:
        raise SystemExit("--r-max-display-m must be positive")
    if args.electron_smooth_sigma < 0.0 or args.field_smooth_sigma < 0.0:
        raise SystemExit("smoothing sigmas must be nonnegative")
    if not (0 <= args.crf <= 51):
        raise SystemExit("--crf must be in [0, 51]")
    if args.dpi <= 0:
        raise SystemExit("--dpi must be positive")
    if args.jobs < 1:
        raise SystemExit("--jobs must be >= 1")
    if (
        args.field_profile_limit_kvpm is not None
        and args.field_profile_limit_kvpm <= 0.0
    ):
        raise SystemExit("--field-profile-limit-kvpm must be positive")
    if args.warp_only:
        args.time_warp = True
    parse_warp_control_points(args.warp_spec, flag_name="--warp-spec")
    args.electron_log_limits = parse_limits(args.electron_log_limits, "--electron-log-limits")
    args.positive_ion_log_limits = parse_limits(
        args.positive_ion_log_limits,
        "--positive-ion-log-limits",
    )
    args.field_limits_kvpm = parse_limits(args.field_limits_kvpm, "--field-limits-kvpm")
    args.field_fraction_limits = parse_limits(
        args.field_fraction_limits,
        "--field-fraction-limits",
    )
    args.field_delta_percent_limits = parse_limits(
        args.field_delta_percent_limits,
        "--field-delta-percent-limits",
    )
    return args


def main() -> None:
    args = parse_args()
    summary = render(
        args.input_dir,
        args.output_dir,
        write_compact_npz=args.compact_npz,
        r_max_display_m=args.r_max_display_m,
        symmetric_radius_display=args.symmetric_radius_display,
        electron_smooth_sigma=args.electron_smooth_sigma,
        field_smooth_sigma=args.field_smooth_sigma,
        field_panel=args.field_panel,
        density_panels=parse_density_panels(args.density_panels),
        layout=args.layout,
        crf=args.crf,
        preset=args.preset,
        dpi=args.dpi,
        preview_only=args.preview_only,
        style="dark",
        lightcurve_species=args.lightcurve_species,
        time_warp=args.time_warp,
        warp_only=args.warp_only,
        warp_spec=args.warp_spec,
        lightcurve_inset=True,
        repair_field_spikes=True,
        cloud_scattering=args.cloud_scattering,
        weak_tgf_threshold=args.weak_tgf_threshold,
        jobs=args.jobs,
        electron_log_limits_override=args.electron_log_limits,
        positive_ion_log_limits_override=args.positive_ion_log_limits,
        field_limits_kvpm_override=args.field_limits_kvpm,
        field_fraction_limits_override=args.field_fraction_limits,
        field_delta_percent_limits_override=args.field_delta_percent_limits,
        field_profile_limit_kvpm_override=args.field_profile_limit_kvpm,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
