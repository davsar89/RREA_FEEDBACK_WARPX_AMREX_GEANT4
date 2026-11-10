#!/usr/bin/env python3
"""Electron lightcurves with time-linked, near-axis |E|(z) snapshots.

Run on the capture host; an NPZ output extracts only small plotting data:
  python scripts/plot_rrea_electrons_and_field.py <capture> field_series.npz
Then render anywhere, or render directly from the capture:
  python scripts/plot_rrea_electrons_and_field.py field_series.npz figure.png

The population is the whole-domain >=1 MeV electron count. Field extraction
uses the video's existing first-radial-cell primitive (not signed Ez).
--scan-step-us controls analysis sampling only; capture cadence is untouched.
Panels resolve changes in profile shape, always including the first frame,
electron peak and last frame, connected to a uniformly scaled time axis.
The RREA threshold uses 284 kV/m at the project's standard-air density.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path

import numpy as np


def extract(capture: Path, scan_step_us: float) -> dict[str, np.ndarray]:
    from render_rrea_profiled_video import (
        field_profile_curves, frame_array_indices, load_frames,
    )
    from rrea_video_stream import dedup_reduced_rows_by_step
    from rrea_profiled_atmosphere import load_threshold_profile

    frames, rows, meta, _ = load_frames(capture)
    frame_times = np.array([float(row["time_s"]) for row in rows])
    # Ignore a live writer's unfinished CSV line; complete frame rows index
    # already-written frames. Stop both observables at the last saved frame.
    raw = (capture / "rrea_reduced.csv").read_bytes()
    raw = raw[:raw.rfind(b"\n") + 1]
    reduced = dedup_reduced_rows_by_step(
        list(csv.DictReader(io.StringIO(raw.decode()))), source=capture)
    reduced = [r for r in reduced if float(r["time_s"]) <= frame_times[-1] + 1e-14]
    times = np.array([float(r["time_s"]) for r in reduced])
    counts = np.array([float(r["N_kinetic_e_ge_1MeV_real"]) for r in reduced])
    targets = np.r_[np.arange(frame_times[0], frame_times[-1], scan_step_us * 1e-6),
                    times[np.argmax(counts)], frame_times[-1]]
    indices = sorted({int(np.argmin(abs(frame_times - t))) for t in targets})
    field_index = frame_array_indices(meta, frames)["field"]
    profiles = np.array([
        field_profile_curves(frames[i, field_index, :, :1],
                             symmetric_radius_display=False)[1] for i in indices
    ])
    dz = (meta["z_max_m"] - meta["z_min_m"]) / meta["nz"]
    altitude = (meta["altitude_msl_at_z0_m"] + meta["z_min_m"]
                + (np.arange(meta["nz"]) + .5) * dz)
    run = json.loads((capture / "command.json").read_text())["metadata"]
    threshold = load_threshold_profile(Path(run["density_profile"]), threshold_stp_kv_per_m=284.)
    label = f"E₀ = {run['e0_peak_kv_per_m']:g} kV/m"
    if run["seed_source_model"] == "parma_plus_1mev_pulse":
        label += f"  •  Initial {run['pulse_real_electrons']:g}-electron pulse + continuous PARMA"
    return dict(time_s=times, n_electrons=counts, profile_time_s=frame_times[indices],
                altitude_m=altitude, field_kvpm=profiles,
                threshold_kvpm=np.array([threshold.interpolate(z) / 1000 for z in altitude]),
                source_label=np.array(label),
                axis_cell_width_m=np.array((meta["r_max_m"] - meta["r_min_m"]) / meta["nr"]))


def profile_indices(data: dict, panels: int) -> list[int]:
    """Resolve the largest RMS_z error from time-interpolated shown profiles.

    Linear interpolation is a display-selection estimate, not a physical
    prediction. The uniform mesh gives each altitude equal weight. Static
    profiles favor gaps in time; first, peak and last frames stay pinned.
    """
    fields, times = data["field_kvpm"], data["profile_time_s"]
    panels = min(panels, len(times))
    peak = data["time_s"][np.argmax(data["n_electrons"])]
    selected = {0, int(np.argmin(abs(times - peak))), len(times) - 1}
    while len(selected) < panels:
        remaining = np.array([i for i in range(len(times)) if i not in selected])
        anchors = np.array(sorted(selected))
        right = np.searchsorted(anchors, remaining)
        lo, hi = anchors[right - 1], anchors[right]
        fraction = ((times[remaining] - times[lo]) / (times[hi] - times[lo]))[:, None]
        estimate = fields[lo] + fraction * (fields[hi] - fields[lo])
        error = np.sqrt(np.mean((fields[remaining] - estimate)**2, axis=1))
        gap = np.minimum(times[remaining] - times[lo], times[hi] - times[remaining])
        selected.add(int(max(zip(error, gap, remaining))[2]))
    return sorted(selected)


def plot(data: dict, output: Path, panels: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import ConnectionPatch
    from matplotlib.ticker import LogLocator, MaxNLocator
    from rrea_profiled_atmosphere import RREA_THRESHOLD_REFERENCE_DENSITY_KG_M3

    t, n = data["time_s"] * 1e3, data["n_electrons"]
    fields, ft = data["field_kvpm"], data["profile_time_s"] * 1e3
    if (not np.all(np.isfinite(fields)) or not np.all(np.isfinite(n))
            or np.any(n < 0) or np.any(np.diff(t) <= 0)):
        raise ValueError("Need finite fields and nonnegative counts at increasing times")
    selected = profile_indices(data, panels)
    colors = plt.get_cmap("tab10")(np.linspace(0, .9, len(selected)))
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.titlelocation": "left"})
    fig = plt.figure(figsize=(16, 10.5))
    grid = fig.add_gridspec(4, len(selected), height_ratios=[1, 1, .30, 1.45],
                           left=.065, right=.98, bottom=.13, top=.87,
                           hspace=.4, wspace=.2)
    linear = fig.add_subplot(grid[0, :])
    logarithmic = fig.add_subplot(grid[1, :], sharex=linear)
    exponent = int(np.floor(np.log10(max(float(n.max()), 1.))))
    linear.plot(t, n / 10.**exponent, color="#176d9c", lw=1.2)
    linear.fill_between(t, 0, n / 10.**exponent, color="#176d9c", alpha=.08)
    linear.set(ylabel=fr"Electron count ($10^{{{exponent}}}$)", ylim=(0, n.max()/10.**exponent*1.2))
    logarithmic.plot(t, np.ma.masked_less_equal(n, 0), color="#176d9c", lw=.7)
    logarithmic.set(yscale="log", ylabel="Electron count", xlabel="Simulation time (ms)")
    logarithmic.xaxis.label.set_bbox(dict(facecolor="white", edgecolor="none", pad=2))
    logarithmic.yaxis.set_major_locator(LogLocator(base=10, numticks=6))
    for ax, name in [(linear, "Linear"), (logarithmic, "Logarithmic")]:
        ax.set_title(name, fontsize=12, fontweight="bold")
        ax.set_xlim(0, ft[-1]); ax.grid(alpha=.17)
    linear.tick_params(labelbottom=False)

    # A zoom makes clustered early snapshot markers readable while retaining
    # the full, linear-in-time axis in both main lightcurve panels.
    peak_time = t[np.argmax(n)]
    zoom_end = min(t[-1], max(2 * peak_time, t[-1] * .08))
    inset = linear.inset_axes([.44, .20, .36, .68])
    inset.plot(t, n / 10.**exponent, color="#176d9c", lw=1)
    inset.set_xlim(0, zoom_end); inset.set_ylim(linear.get_ylim())
    inset.set_title("Initial pulse · time (ms)", fontsize=9)
    inset.tick_params(labelsize=8); inset.grid(alpha=.15)

    altitude = data["altitude_m"] / 1e3
    threshold = data["threshold_kvpm"]
    xmax = np.ceil(max(fields[selected].max(), fields[0].max(), threshold.max()) / 25) * 25
    for column, (i, color) in enumerate(zip(selected, colors)):
        label = chr(65 + column)
        ts = ft[i]
        for ax, values in [(linear, n/10.**exponent), (logarithmic, n), (inset, n/10.**exponent)]:
            ax.axvline(ts, color=color, ls=":", lw=1, alpha=.75)
            if t[0] <= ts <= t[-1]:
                ax.plot(ts, np.interp(ts, t, values), "o", color=color, ms=4, clip_on=True)
        if ts <= zoom_end:
            inset.text(ts, .92, label, color=color, ha="center", fontsize=8,
                       transform=inset.get_xaxis_transform())
        ax = fig.add_subplot(grid[3, column])
        ax.plot(fields[0], altitude, "--", color=".55", lw=1)
        ax.plot(threshold, altitude, "-.", color=".15", lw=1.1)
        ax.plot(fields[i], altitude, color=color, lw=1.6)
        ax.set(xlim=(0, xmax), ylim=(altitude[0], altitude[-1]), xlabel="|E| (kV/m)")
        ax.xaxis.set_major_locator(MaxNLocator(3)); ax.tick_params(labelsize=9)
        ax.set_title(f"{label} · {ts*1000:,.4g} µs", color=color, fontsize=12, fontweight="bold")
        if column == 0:
            ax.set_ylabel("Altitude MSL (km)")
        else:
            ax.tick_params(labelleft=False)
        ax.grid(alpha=.17)
        fig.add_artist(ConnectionPatch(
            xyA=(ts, 0), coordsA=logarithmic.get_xaxis_transform(),
            xyB=(.5, 1.17), coordsB=ax.transAxes,
            color=color, lw=.85, alpha=.6, clip_on=False, zorder=-1))
    fig.suptitle("Energetic electrons and on-axis electric-field evolution",
                 x=.065, y=.965, ha="left", fontsize=21, fontweight="bold")
    fig.text(.065, .925, str(data.get("source_label", "")), fontsize=12)
    fig.text(.065, .899, f"Whole-domain electrons ≥1 MeV  •  Through {ft[-1]:.3f} ms  •  "
             "Colored markers and connectors identify each field snapshot", fontsize=12)
    width = float(data["axis_cell_width_m"])
    fig.text(.065, .078, f"Field panels: |E|(z) in the first radial cell (0–{width:g} m); "
             "gray dashed = initial profile. All panels share the same field scale.", fontsize=10)
    fig.text(.065, .055, "Black dash-dot = RREA threshold: 284 × ρ(z)/ρ₀ kV/m; "
             f"ρ₀ = {RREA_THRESHOLD_REFERENCE_DENSITY_KG_M3:g} kg/m³ (standard air), "
             "using the run's density profile.", fontsize=10)
    fig.text(.065, .032, "Snapshots resolve changes in field-profile shape, with "
             "the initial frame, electron peak and latest frame included. Lightcurves are unsmoothed.", fontsize=10)
    if np.any(n == 0):
        fig.text(.065, .010, "Zero-count samples are omitted on the logarithmic axis.", fontsize=9, color=".4")
    fig.savefig(output, dpi=180, facecolor="white")
    plt.close(fig)
    print("Profile times (µs):", ", ".join(f"{ft[i]*1000:.1f}" for i in selected))
    print(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", type=Path, help="capture directory or previously extracted .npz")
    parser.add_argument("output", type=Path, help=".npz for extraction only; .png/.svg for a figure")
    parser.add_argument("--panels", type=int, default=8)
    parser.add_argument("--scan-step-us", type=float, default=10., help="field analysis sampling interval (default: 10 µs)")
    args = parser.parse_args()
    if not 3 <= args.panels <= 10 or not np.isfinite(args.scan_step_us) or args.scan_step_us <= 0:
        parser.error("need 3–10 panels and a finite, positive scan step")
    if args.source.is_dir():
        data = extract(args.source, args.scan_step_us)
    else:
        with np.load(args.source, allow_pickle=False) as archive:
            data = dict(archive)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.suffix == ".npz":
        np.savez_compressed(args.output, **data)
    else:
        plot(data, args.output, args.panels)


if __name__ == "__main__":
    main()
