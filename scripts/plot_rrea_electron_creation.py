#!/usr/bin/env python3
r"""Plot full and interval densities where electrons first reach >=1 MeV.

Count each particle weight once, at an energetic birth or first upward crossing.

Enable capture with --extra-warpx-arg rrea.electron_creation_bin_m=25.
Example: python scripts/plot_rrea_electron_creation.py \
    RUN/capture/rrea_video/electron_creation_history.bin.gz \
    --output e125_videos/pulse_parma_creation_1000us.png

Default: up to five intervals with approximately equal weighted counts.
--bounds-us 0,20,50,100,300,1000 selects in-range bounds snapped to saved frames.
A single checkpoint electron_creation.bin gives only the full accumulation.
Density is in electrons/m^3 accumulated during each interval; it is not a rate.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import gzip
from pathlib import Path
import struct

import numpy as np


def creation_snapshots(path: Path):
    """Stream raw checkpoint or concatenated, lossless gzip snapshots."""
    header = struct.Struct("<8s2i7d")
    geometry = None
    with path.open("rb") as raw:
        opener = gzip.GzipFile(fileobj=raw) if raw.peek(2)[:2] == b"\x1f\x8b" else nullcontext(raw)
        with opener as stream:
            while block := stream.read(header.size):
                magic, nr, nz, r0, r1, z0, z1, threshold, time, altitude = header.unpack(block)
                if (magic not in (b"RREACR1\0", b"RREACR2\0") or min(nr, nz) <= 0
                        or not (0 <= r0 < r1 and z0 < z1)
                        or not np.isfinite([r0,r1,z0,z1,threshold,time,altitude]).all()):
                    raise ValueError("Invalid electron creation map header")
                current = (nr,nz,r0,r1,z0,z1,threshold,altitude)
                if geometry is not None and not np.allclose(current,geometry,rtol=1e-12,atol=1e-12):
                    raise ValueError("Creation history changes grid or threshold")
                geometry = current
                history_bytes = struct.unpack("<Q", stream.read(8))[0] if magic == b"RREACR2\0" else 0
                weights = np.frombuffer(stream.read(2*nz*nr*8), dtype="<f8").reshape(2, nz, nr)
                if not np.isfinite(weights).all() or (weights < 0).any():
                    raise ValueError("Creation weights must be finite and nonnegative")
                r, z = np.linspace(r0,r1,nr+1), np.linspace(z0,z1,nz+1)
                # n_created = sum(w_first)/[pi*(r_hi^2-r_lo^2)*dz], in m^-3.
                volume = np.diff(z)[:,None] * (np.pi*np.diff(r**2))[None,:]
                yield dict(density=weights/volume, weights=weights, volume=volume,
                           r=r, z=z+altitude, threshold_eV=threshold, time_s=time,
                           history_bytes=history_bytes)


def load_creation_map(path: Path) -> dict:
    return next(creation_snapshots(path))


def accumulation_panels(path: Path, count: int = 5, bounds_us=None):
    """Choose count-quantile times; difference cumulative maps without recounting.

    Intervals are (start,end], with start=0 also including initial injections.
    Bounds spanning 0 through the last frame partition the full accumulation.
    Only chosen snapshots are retained in memory, regardless of run length.
    """
    summary = np.array([(d["time_s"], d["weights"].sum()) for d in creation_snapshots(path)])
    if not len(summary) or count < 1:
        raise ValueError("Need at least one snapshot and a positive interval count")
    times, totals = summary.T
    if (times <= 0).any() or (np.diff(times) <= 0).any():
        raise ValueError("Creation snapshot times must be positive and increasing")
    tolerance = 64*np.finfo(float).eps*max(totals.max(),1.)
    if (np.diff(totals) < -tolerance).any():
        raise ValueError("Cumulative first-passage counts decreased")
    if bounds_us is None:
        cuts = np.searchsorted(np.maximum.accumulate(totals),
                               np.arange(1,count)*totals[-1]/count)
        indices = np.unique(np.r_[-1,cuts,len(times)-1])
    else:
        bounds = np.asarray(bounds_us)*1e-6
        if (len(bounds)<2 or (np.diff(bounds)<=0).any() or not np.isfinite(bounds).all()
                or bounds[0]<0 or bounds[-1]>times[-1]+1e-12):
            raise ValueError("Interval bounds must increase within the recorded time range")
        indices = np.array([-1 if t==0 else np.abs(times-t).argmin() for t in bounds])
        if (np.diff(indices)<=0).any():
            raise ValueError("Requested intervals are finer than the saved time resolution")
    selected = {int(i) for i in indices if i>=0} | {len(times)-1}
    maps = {i:d for i,d in enumerate(creation_snapshots(path)) if i in selected}
    full = maps[len(times)-1]
    maps[-1] = dict(weights=np.zeros_like(full["weights"]), time_s=0.)
    panels = []
    if len(times)>1:
        for a,b in zip(indices[:-1],indices[1:]):
            before, after = maps[a], maps[b]
            delta = after["weights"]-before["weights"]
            tolerance = 64*np.finfo(float).eps*np.maximum(after["weights"],1.)
            if (delta < -tolerance).any():
                raise ValueError("A cumulative creation bin decreased")
            panels.append((before["time_s"],after["time_s"],np.maximum(delta,0).sum(axis=0)))
    return full, panels


def plot_creation(data: dict, output: Path, intervals=(), automatic=True) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    weights = data["weights"].sum(axis=0)
    density = weights/data["volume"]
    positive = density[density > 0]
    if not positive.size:
        raise ValueError("No electrons have reached the threshold yet")
    vmax = positive.max()
    vmin = max(positive.min(), vmax * 1e-9)
    if vmin == vmax:
        vmin = vmax / 10
    cmap = plt.colormaps["magma"].copy()
    cmap.set_bad("#eeeeee")
    panels = [(0.,data["time_s"],weights),*intervals]
    cols = min(3,int(np.ceil(np.sqrt(len(panels)))))
    rows = (len(panels)+cols-1)//cols
    fig, axes = plt.subplots(rows,cols,figsize=(4.5*cols+1,4.5*rows+1),
                             sharex=True,sharey=True,squeeze=False,layout="constrained")
    # Show every occupied radial bin, with one bin of margin.
    rlast = min(np.flatnonzero(weights.any(axis=0))[-1]+2,len(data["r"])-1)
    for i,(ax,(start,end,w)) in enumerate(zip(axes.flat,panels)):
        d = w/data["volume"]
        mesh = ax.pcolormesh(data["r"] / 1000, data["z"] / 1000,
            np.ma.masked_where(d == 0,d), shading="flat",
            cmap=cmap, norm=LogNorm(vmin, vmax), rasterized=True)
        ax.set_title(("Full accumulation\n" if i==0 else "Interval accumulation\n")
                     +f"{start*1e6:g}–{end*1e6:g} µs  •  N = {w.sum():.3e}",fontsize=10)
        ax.set(xlabel="Radius (km)",ylabel="Altitude MSL (km)",
               xlim=(data["r"][0]/1000,data["r"][rlast]/1000))
        ax.tick_params(labelbottom=True, labelleft=True)
    for ax in list(axes.flat)[len(panels):]:
        ax.set_visible(False)
    fig.colorbar(mesh, ax=list(axes.flat)[:len(panels)], label="Accumulated first-passage density (electrons m⁻³)",
                 shrink=.85, pad=.025)
    method = "Automatic intervals: count quantiles snapped to saved frames" if automatic else "Specified intervals snapped to saved frames"
    fig.suptitle(f"Where electrons first reach ≥{data['threshold_eV']/1e6:g} MeV\n"
                 + (method if intervals else "Full accumulation from one snapshot")
                 + f"; shared density scale\nIncludes {data['weights'][1].sum():.3e} electrons injected already above threshold",
                 fontsize=12)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("map", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval-count",type=int,default=5)
    parser.add_argument("--bounds-us",type=lambda s:[float(x) for x in s.split(",")])
    args = parser.parse_args()
    data, intervals = accumulation_panels(args.map,args.interval_count,args.bounds_us)
    plot_creation(data,args.output,intervals,automatic=args.bounds_us is None)
    print(f"Through {data['time_s']*1e6:g} µs: "
          f"{data['weights'].sum():.9g} physical electrons; {args.output}")
