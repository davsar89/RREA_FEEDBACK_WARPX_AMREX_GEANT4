#!/usr/bin/env python3
"""Run a compact profiled RREA video-capture simulation."""

from __future__ import annotations

import argparse
from bisect import bisect_left
import csv
import json
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from rrea_profiled_atmosphere import (
    ALTITUDE_AT_Z0_M,
    audit_ambient_field,
    closure_table_max_en_td,
    build_cd_plane_z_m,
    coleman_dwyer_profile_reference,
    conditioned_exponential_seed_energies,
    DEFAULT_DENSITY_PROFILE,
    DEFAULT_FIELD_PROFILE,
    DOMAIN_HEIGHT_M,
    DOMAIN_RADIUS_M,
    ensure_profiles_cover_domain,
    find_lowest_above_threshold_region,
    load_density_ratio_profile,
    load_field_profile,
    load_threshold_profile,
    profile_metadata,
    read_csv_rows,
    region_to_dict,
    SOURCE_RADIUS_M,
    write_json,
    write_seed_schedule,
)
from rrea_profiled_atmosphere import launcher_prefix
from rrea_cd_physics import (
    CD_ESCALATED_SEED_COUNT,
    CD_INITIAL_SEED_COUNT,
    cd_population_ceilings,
    cd_diagnostic_seed_design,
    cd_seed_design,
    cd_transport_rng_seed,
)
from rrea_video_stream import (
    parse_warp_control_points,
    warp_rate_us_per_s_at,
    ENERGY_THRESHOLD_EV,
)
from rrea_parma_source import (
    ensure_parma_source_table,
    write_parma_seed_schedule,
)
from rrea_profiled_atmosphere import SCHEMA6_CHARGED_ENERGY_MAX_EV
from rrea_run_support import (
    CAPTURE_DEFAULTS,
    CELL_SIZE_M,
    TAPER_START_DOMAIN_FRACTION,
    DEFAULT_BASE_INPUT,
    DEFAULT_POPULATION_CONTROL_INTERVAL,
    DEFAULT_POPULATION_CONTROLLER_RNG_SALT,
    DEFAULT_POPULATION_TARGET_ELECTRON_MACROS,
    DEFAULT_SEED_MACRO_COUNT,
    DEFAULT_SEED_SOURCE_MODEL,
    PARMA_CONTINUOUS_SOURCE_DEFAULTS,
    profiled_overrides,
    prepare_fresh_output_directory,
    required_time_steps,
    resolved_driver_arguments,
    run_command,
    validate_fresh_output_paths,
    validate_user_warpx_overrides,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "runs_rrea/profiled_video_capture"

def resolve_max_step(requested: int | None, stop_time_s: float, dt_s: float) -> int:
    required = required_time_steps(stop_time_s, dt_s)
    effective = required if requested is None else requested
    if effective < required:
        raise SystemExit(
            f"--max-step={effective} cannot reach --stop-time-s={stop_time_s:g} "
            f"at dt={dt_s:g}; require at least {required}"
        )
    return effective


def finite_float(text: str) -> float:
    """argparse float type that rejects NaN/Inf at the input boundary."""
    value = float(text)
    if not math.isfinite(value):
        raise argparse.ArgumentTypeError(f"value must be finite (got {text!r})")
    return value


def nonnegative_int(text: str) -> int:
    """argparse integer type that rejects negatives."""
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError(f"value must be non-negative (got {text!r})")
    return value


def resolve_restart_checkpoint(run_dir: Path, spec: str) -> Path:
    """Use an explicit complete checkpoint, or search this run's namespace."""
    run_dir = run_dir.expanduser().resolve()
    if spec != "auto":
        chk = Path(spec).expanduser().resolve()
        if not (chk.name.startswith("chk") and chk.name[3:].isdigit()):
            raise SystemExit(f"--restart-checkpoint {chk}: expected chkNNNNNN")
        if not (chk / "rrea_checkpoint" / "COMPLETE").exists():
            raise SystemExit(
                f"--restart-checkpoint {chk}: missing rrea_checkpoint/COMPLETE "
                "marker (incomplete or not an RREA checkpoint)"
            )
        return chk
    candidates = sorted(
        (
            d
            for d in run_dir.glob("chk*")
            if d.is_dir()
            and not d.is_symlink()
            and not bool(
                getattr(d, "is_junction", lambda: False)()
            )
            and d.name[3:].isdigit()
            and (d / "rrea_checkpoint" / "COMPLETE").exists()
        ),
        key=lambda d: int(d.name[3:]),
    )
    if not candidates:
        raise SystemExit(
            f"--restart-checkpoint auto: no complete chk* directory in {run_dir}"
        )
    return candidates[-1]


def plan_csv_trim(path: Path, step: int) -> tuple[list[str], list[list[str]], int]:
    """READ-ONLY: split a step-stamped CSV into (header, kept_rows, dropped)
    for rows with step <= `step`.  No file is modified here; the caller
    commits the plan only after every consistency check has passed."""
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        raise SystemExit(f"{path}: empty (no header); cannot trim for restart")
    header = rows[0]
    try:
        idx = header.index("step")
    except ValueError as exc:
        raise SystemExit(f"{path}: no 'step' column; cannot trim for restart") from exc
    kept = [r for r in rows[1:] if r and int(float(r[idx])) <= step]
    dropped = len(rows) - 1 - len(kept)
    return header, kept, dropped


def read_csv_step_sequence(path: Path, *, context: str) -> list[int]:
    """Read the ordered ``step`` column from a diagnostic CSV."""
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            header = next(reader)
            step_index = header.index("step")
            return [int(float(row[step_index])) for row in reader if row]
    except (OSError, StopIteration, ValueError, IndexError, OverflowError) as exc:
        raise SystemExit(
            f"{context}: cannot read a valid 'step' sequence from {path}: {exc}"
        ) from exc


def plan_cd_plane_flux_trim(
    path: Path, *, checkpoint_write_count: int, plane_count: int
) -> tuple[list[str], list[list[str]], int]:
    """Keep exactly the plane blocks the checkpoint counted.

    The checkpoint's write count, not ``step <= checkpoint_step``, is
    authoritative because a final and a checkpoint write can share a step;
    blocks after it, complete or torn, are discarded.  The engine owns the
    row format.
    """
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            header, *data = csv.reader(handle)
    except (OSError, ValueError) as exc:
        raise SystemExit(
            f"restart: cannot read the plane-flux file {path}: {exc} -- refusing "
            "to restart (no file modified)") from exc
    required_rows = checkpoint_write_count * plane_count
    if len(data) < required_rows:
        raise SystemExit(
            f"restart: checkpoint records cd_plane_write_count="
            f"{checkpoint_write_count}, but {path} holds only {len(data)} of the "
            f"{required_rows} rows it implies -- refusing to restart (no file modified)")
    return header, data[:required_rows], len(data) - required_rows


def commit_csv_trim(path: Path, header: list[str], kept: list[list[str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(kept)


def frame_stream_end_bytes(
    header: list[str],
    kept_rows: list[list[str]],
    *,
    context: str,
) -> int:
    """Byte length the stream must have to hold exactly ``kept_rows`` frames.

    Frames are compressed and variable length, so this reads the LAST kept
    frame's own recorded offset and length -- exact after any earlier trim, and
    what the restart truncates to.
    """
    if not kept_rows:
        return 0
    try:
        off_i = header.index("byte_offset")
        len_i = header.index("byte_length")
    except ValueError as exc:
        raise SystemExit(
            f"{context}: frame metadata lacks the byte_offset/byte_length "
            "index columns -- cannot locate frames"
        ) from exc
    last = kept_rows[-1]
    try:
        offset, length = int(last[off_i]), int(last[len_i])
    except (ValueError, IndexError) as exc:
        raise SystemExit(
            f"{context}: malformed byte_offset/byte_length in the last kept "
            f"frame row ({exc})"
        ) from exc
    if offset < 0 or length <= 0:
        raise SystemExit(
            f"{context}: frame index has non-positive extent "
            f"(offset={offset}, length={length})"
        )
    return offset + length


def validate_restart_capture(output_dir: Path, chk_step: int) -> None:
    """Non-mutating consistency check for a restart.

    Run this and the schedule compatibility checks before
    ``trim_restart_capture`` so rejection cannot mutate the capture.

    `chk_step` scopes frame identity to the retained prefix; rows after the
    checkpoint are disposable, so a torn append tail remains recoverable.
    """
    video_dir = output_dir / "rrea_video"
    # The two per-frame streams must describe the same frames: identical step
    # sequences, not merely equal row counts.
    per_frame_steps = {
        path.name: [
            s
            for s in read_csv_step_sequence(path, context="restart")
            if s <= chk_step
        ]
        for path in (
            video_dir / "video_frame_metadata.csv",
            video_dir / "particle_moments.csv",
        )
    }
    reference = per_frame_steps["video_frame_metadata.csv"]
    if any(steps != reference for steps in per_frame_steps.values()):
        raise SystemExit(
            "restart: per-frame streams disagree on the captured frame "
            f"sequence at or before checkpoint step {chk_step} (step lists "
            f"differ: { {name: steps[:8] for name, steps in per_frame_steps.items()} }"
            "...); the capture is internally inconsistent -- refusing to "
            "restart"
        )


def require_restart_video_config_match(output_dir: Path, args: Any) -> None:
    """Reject a restart whose video-diagnostic config differs
    from the existing capture's metadata.

    The engine appends frames sized and positioned by the resolved
    rrea.video_diag_* values, so grid or spatial-bound drift silently mixes
    incompatible frames into the existing stream
    (the frame-schedule comparison covers timing only). The energy threshold
    is a fixed diagnostic definition, not a knob."""
    meta = json.loads(
        (output_dir / "rrea_video" / "video_capture_metadata.json").read_text()
    )
    expected = {
        "nr": args.video_nr,
        "nz": args.video_nz,
        "r_min_m": args.video_r_min_m,
        "r_max_m": args.video_r_max_m,
        "z_min_m": args.video_z_min_m,
        "z_max_m": args.video_z_max_m,
    }
    drift = {
        key: {"stored": meta.get(key), "current": want}
        for key, want in expected.items()
        if meta.get(key) is None or float(meta[key]) != float(want)
    }
    if drift:
        raise SystemExit(
            "restart: video-diagnostic config drift against the existing "
            f"capture metadata: {drift}; restarts must use IDENTICAL "
            "video settings -- refusing to restart"
        )


def trim_restart_capture(
    output_dir: Path,
    chk_dir: Path,
    *,
    cd_plane_altitudes_m: list[float] | None = None,
) -> dict[str, Any]:
    """Trim the existing capture to the checkpoint step.

    This mutates files; call it only after restart and schedule validation.

    The engine appends blindly after a restore (rrea_reduced.csv and the
    video stream/CSVs open in append mode; the video restart guard only
    checks that video_frames.bin EXISTS), so rows/frames written after the
    last checkpoint must be removed here or the resumed run would duplicate steps
    and misalign the frame stream.
    """
    name = chk_dir.name
    if not (name.startswith("chk") and name[3:].isdigit()):
        raise SystemExit(f"restart checkpoint dir must be chkNNNNNN, got {name!r}")
    step = int(name[3:])
    video_dir = output_dir / "rrea_video"
    frames_bin = video_dir / "video_frames.bin"
    frame_meta = video_dir / "video_frame_metadata.csv"

    chk_meta_path = chk_dir / "rrea_checkpoint" / "metadata.json"
    try:
        chk_meta = json.loads(chk_meta_path.read_text())
        raw_frames = chk_meta["video_frame_write_count"]
        raw_cd_plane_writes = chk_meta["cd_plane_write_count"]
        if (
            not isinstance(raw_frames, int)
            or isinstance(raw_frames, bool)
            or not isinstance(raw_cd_plane_writes, int)
            or isinstance(raw_cd_plane_writes, bool)
        ):
            raise TypeError("capture counters must be JSON integers")
        chk_frames = raw_frames
        chk_cd_plane_writes = raw_cd_plane_writes
    except (OSError, TypeError, ValueError, KeyError) as exc:
        raise SystemExit(
            "restart: cannot read video_frame_write_count and "
            f"cd_plane_write_count from {chk_meta_path} ({exc}) -- refusing "
            "to restart (no file modified)"
        ) from exc
    if chk_frames < 0 or chk_cd_plane_writes < 0:
        raise SystemExit(
            f"restart: {chk_meta_path} contains a negative capture counter -- "
            "refusing to restart (no file modified)"
        )
    if not cd_plane_altitudes_m and chk_cd_plane_writes != 0:
        raise SystemExit(
            f"restart: checkpoint records cd_plane_write_count="
            f"{chk_cd_plane_writes}, but the current run has plane diagnostics "
            "disabled -- refusing to restart (no file modified)"
        )

    # -------- plan phase: READ-ONLY. Every consistency check runs before any
    # file is touched, so a rejected restart cannot leave a partially mutated
    # capture. Engine-level strict-match
    # aborts after the commit remain possible but harmless: a capture
    # trimmed exactly to the checkpoint step is itself a valid restart
    # state, and this trim is idempotent (re-running drops 0 rows).
    # The current profile is an optional side-car; a changed CSV schema is
    # archived by the engine (ArchiveCsvIfHeaderChanged). Every step-stamped
    # append stream is trimmed here.
    plans = {
        key: (path, plan_csv_trim(path, step))
        for key, path in (
            ("rrea_reduced_rows_dropped", output_dir / "rrea_reduced.csv"),
            ("current_profile_rows_dropped", video_dir / "current_profile.csv"),
        )
        if path.exists()
    }
    plans.update({
        "video_frame_metadata_rows_dropped": (
            frame_meta,
            plan_csv_trim(frame_meta, step),
        ),
        "particle_moments_rows_dropped": (
            video_dir / "particle_moments.csv",
            plan_csv_trim(video_dir / "particle_moments.csv", step),
        ),
    })
    if cd_plane_altitudes_m:
        plane_path = output_dir / "rrea_cd_plane_flux.csv"
        plans["cd_plane_flux_rows_dropped"] = (
            plane_path,
            plan_cd_plane_flux_trim(
                plane_path,
                checkpoint_write_count=chk_cd_plane_writes,
                plane_count=len(cd_plane_altitudes_m),
            ),
        )
    n_frames = len(plans["video_frame_metadata_rows_dropped"][1][1])
    # Reconcile against the engine's own frame counter: the checkpoint records
    # exactly how many frames had been written, so the trimmed metadata must
    # agree.
    if chk_frames != n_frames:
        raise SystemExit(
            f"restart: checkpoint {chk_dir.name} recorded "
            f"video_frame_write_count={chk_frames} but the capture holds "
            f"{n_frames} frames at step <= {step}; capture and checkpoint "
            "disagree -- refusing to restart (no file modified)"
        )
    frame_header, frame_kept, _ = plans["video_frame_metadata_rows_dropped"][1]
    want_bytes = frame_stream_end_bytes(
        frame_header, frame_kept, context="restart"
    )
    if not frames_bin.is_file():
        raise SystemExit(f"restart: {frames_bin} is missing (no file modified)")
    have_bytes = frames_bin.stat().st_size
    if have_bytes < want_bytes:
        raise SystemExit(
            f"restart: video_frames.bin has {have_bytes} bytes but the trimmed "
            f"metadata implies {want_bytes} for {n_frames} "
            "frames; capture is inconsistent -- refusing to restart (no file "
            "modified)"
        )

    # -------- commit phase: all checks passed; apply the plan. run_command
    # invalidates the previous completion markers before the engine starts.
    trim_log: dict[str, Any] = {"checkpoint": str(chk_dir), "checkpoint_step": step}
    for key, (path, (header, kept, dropped)) in plans.items():
        if dropped:
            commit_csv_trim(path, header, kept)
        trim_log[key] = dropped
    trim_log["video_frames_bytes_truncated"] = have_bytes - want_bytes
    if have_bytes > want_bytes:
        with frames_bin.open("r+b") as handle:
            handle.truncate(want_bytes)
    trim_log["frames_kept"] = n_frames
    write_json(output_dir / "restart_trim_log.json", trim_log)
    print(
        f"restart from {chk_dir.name} (step {step}): kept {n_frames} frames, "
        f"dropped rows reduced={trim_log.get('rrea_reduced_rows_dropped', 'ABSENT')} "
        f"frame_meta={trim_log['video_frame_metadata_rows_dropped']} "
        f"moments={trim_log['particle_moments_rows_dropped']}, "
        f"cd_plane_rows={trim_log.get('cd_plane_flux_rows_dropped', 0)}, "
        f"truncated {trim_log['video_frames_bytes_truncated']} frame bytes",
        flush=True,
    )
    return trim_log


def require_compatible_schedule(
    fresh_path: Path, existing_path: Path, what: str, *, allow_append: bool = False
) -> bool:
    """Require equal parsed CSV rows, or an append-only extension.

    ``allow_append`` accepts the existing rows being an exact prefix of the
    regenerated rows, allowing a finished run to extend its stop time.
    A prefix is the whole safety argument: capture rows are emitted in step
    order, so it establishes that every already-captured frame kept both its step and its
    capture_index -- the index the engine uses to position frames inside
    video_frames.bin -- and that the longer schedule only appends later ones.
    A moved retained frame still fails closed.
    Returns True when the regenerated file is strictly longer.

    Numeric cells compare with a 1e-12 relative and absolute tolerance rather
    than by byte: regenerating a schedule on a different machine goes through
    that platform's libm (exp/log/pow), whose last-ulp rounding can differ (and
    sqrt(1 - mu^2) amplifies it near vertical directions), while a changed
    radius, count, realization or table changes the sampled values or row
    ordering materially.
    """
    with fresh_path.open(newline="", encoding="utf-8") as handle:
        fresh = list(csv.reader(handle))
    with existing_path.open(newline="", encoding="utf-8") as handle:
        existing = list(csv.reader(handle))
    if _csv_rows_equivalent(fresh, existing):
        return False
    if allow_append and _csv_rows_equivalent(fresh[:len(existing)], existing):
        return True
    detail = f"row counts {len(fresh)} (regenerated) vs {len(existing)}"
    for index, (fresh_row, existing_row) in enumerate(zip(fresh, existing)):
        if not _csv_rows_equivalent([fresh_row], [existing_row]):
            detail = (
                f"first difference at row {index}:\n"
                f"  regenerated: {','.join(fresh_row)[:400]}\n"
                f"  existing:    {','.join(existing_row)[:400]}"
            )
            break
    raise SystemExit(
        f"restart: regenerated {what} differs from the previous run's "
        f"({existing_path}); the retained schedule prefix must stay equivalent"
        + (", or flags that only APPEND later frames" if allow_append else "")
        + f"\n{detail}"
    )


def _csv_rows_equivalent(
    fresh: list[list[str]], existing: list[list[str]]
) -> bool:
    if len(fresh) != len(existing):
        return False
    for fresh_row, existing_row in zip(fresh, existing):
        if fresh_row == existing_row:
            continue
        if len(fresh_row) != len(existing_row):
            return False
        for fresh_cell, existing_cell in zip(fresh_row, existing_row):
            if fresh_cell == existing_cell:
                continue
            try:
                fresh_value = float(fresh_cell)
                existing_value = float(existing_cell)
            except ValueError:
                return False
            if not math.isclose(
                fresh_value, existing_value, rel_tol=1e-12, abs_tol=1e-12
            ):
                return False
    return True


def build_frame_schedule(
    *,
    dt_s: float,
    frame_interval_s: float,
    final_time_s: float,
    warp_capture_points: list[tuple[float, float]] | None = None,
    warp_capture_fps: float = 12.0,
    retained_rows: list[dict[str, Any]] | None = None,
    append_after_step: int = 0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    # A uniform grid from t=0, with any retained restart prefix restored below.
    by_step: dict[int, dict[str, Any]] = {}
    frames: list[dict[str, Any]] = []
    for frame_index in range(int(math.floor(final_time_s / frame_interval_s + 1e-9)) + 1):
        target_time_s = frame_index * frame_interval_s
        # The RREA video hook is first invoked after WarpX completes step
        # one; a step-zero request is therefore never written.  Preserve
        # the logical t=0 view in the render map, but bind it to the first
        # physical state the engine can produce instead of scheduling an
        # unsupported frame and silently losing it.
        step = max(1, int(round(target_time_s / dt_s)))
        actual_time_s = step * dt_s
        if step not in by_step:
            by_step[step] = {
                "capture_index": len(by_step),
                "step": step,
                "target_time_s": actual_time_s,
            }
        frames.append(
            {
                "frame_index": frame_index,
                "target_time_s": target_time_s,
                "capture_index": by_step[step]["capture_index"],
                "step": step,
                "captured_time_s": actual_time_s,
            }
        )
    if warp_capture_points:
        # Extra capture times paced by the cinematic playback schedule so the
        # time-warped render has real physical states wherever it dwells
        # (instead of long cross-fades between sparse uniform frames). These
        # frames are captured but are not part of the uniform frame grid.
        final_step = int(round(final_time_s / dt_s))
        t_us = 0.0
        final_us = final_time_s * 1.0e6
        while t_us < final_us:
            rate = warp_rate_us_per_s_at(t_us, warp_capture_points)
            t_us += max(rate / warp_capture_fps, dt_s * 1.0e6)
            step = max(
                1,
                min(int(round(t_us * 1.0e-6 / dt_s)), max(1, final_step)),
            )
            if step not in by_step:
                by_step[step] = {
                    "capture_index": len(by_step),
                    "step": step,
                    "target_time_s": step * dt_s,
                }
            if len(by_step) > 6000:
                raise SystemExit(
                    "warp capture schedule exceeds 6000 frames; raise the "
                    "--warp-capture-spec rates or lower --warp-capture-fps"
                )
    capture_rows = sorted(by_step.values(), key=lambda row: row["step"])
    # Renumber via an old->new map in two passes: in-place matching corrupts
    # frame mappings when insertion order differs from step order (old and
    # new index values collide once extra warp frames interleave).
    old_to_new: dict[int, int] = {}
    for index, row in enumerate(capture_rows):
        old_to_new[int(row["capture_index"])] = index
        row["capture_index"] = index
    for frame in frames:
        frame["capture_index"] = old_to_new[int(frame["capture_index"])]
    if retained_rows:
        # A restart may change future snapshot spacing, never stored states
        # or their indices. New times remain on the requested t=0 grid.
        last_step = int(retained_rows[-1]["step"])
        if final_time_s < last_step * dt_s - dt_s * 1e-6:
            raise SystemExit("restart: video schedule cannot end before its retained prefix")
        append_after_step = max(last_step, append_after_step)
        capture_rows = retained_rows + [r for r in capture_rows if r["step"] > append_after_step]
        indices = {int(r["step"]): i for i, r in enumerate(capture_rows)}
        steps = list(indices)
        for i, row in enumerate(capture_rows):
            row["capture_index"] = i
        for frame in frames:
            position = bisect_left(steps, frame["step"])
            step = min(steps[max(0, position - 1):position + 1],
                       key=lambda s: abs(s - frame["step"]))
            frame.update(step=step, captured_time_s=step * dt_s,
                         capture_index=indices[step])
    render_map = {
        "fps": 24,
        "frames": frames,
    }
    return capture_rows, render_map


def write_capture_schedule(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["capture_index", "step", "target_time_s"])
        writer.writeheader()
        writer.writerows(rows)


def validate_completed_video_capture(
    output_dir: Path,
    scheduled_steps: list[int],
) -> int:
    """Require one complete binary frame in all streams per scheduled step."""
    video_dir = output_dir / "rrea_video"
    frames_bin = video_dir / "video_frames.bin"
    stream_paths = (
        video_dir / "video_frame_metadata.csv",
        video_dir / "particle_moments.csv",
    )
    expected_steps = [step for step in scheduled_steps if step > 0]
    per_stream_steps = {
        path.name: read_csv_step_sequence(path, context="video completion")
        for path in stream_paths
    }
    actual_steps = per_stream_steps["video_frame_metadata.csv"]
    if any(steps != actual_steps for steps in per_stream_steps.values()):
        raise SystemExit(
            "video completion: per-frame CSV streams disagree on captured "
            f"steps: {per_stream_steps}"
        )
    if actual_steps != expected_steps:
        raise SystemExit(
            "video completion: written steps do not exactly match scheduled "
            f"positive steps (expected={expected_steps}, actual={actual_steps})"
        )

    with (video_dir / "video_frame_metadata.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        meta_rows = list(csv.reader(handle))
    expected_bytes = frame_stream_end_bytes(
        meta_rows[0], meta_rows[1:], context="video completion"
    )
    actual_bytes = frames_bin.stat().st_size
    if actual_bytes != expected_bytes:
        raise SystemExit(
            f"video completion: {frames_bin} has {actual_bytes} bytes; "
            f"{len(actual_steps)} frames require exactly {expected_bytes}"
        )
    return len(actual_steps)


def capture_feature_overrides(
    args: argparse.Namespace,
    *,
    video_dir: Path,
    frame_schedule: Path,
) -> list[str]:
    """Single owner of capture-specific ParmParse overrides."""
    return [
        "rrea.video_diag_enable=1",
        f"rrea.video_diag_output_dir={video_dir}",
        f"rrea.video_diag_frame_schedule_file={frame_schedule}",
        f"rrea.video_diag_nr={args.video_nr}",
        f"rrea.video_diag_nz={args.video_nz}",
        f"rrea.video_diag_r_min_m={args.video_r_min_m:.17g}",
        f"rrea.video_diag_r_max_m={args.video_r_max_m:.17g}",
        f"rrea.video_diag_z_min_m={args.video_z_min_m:.17g}",
        f"rrea.video_diag_z_max_m={args.video_z_max_m:.17g}",
        f"rrea.video_diag_energy_threshold_eV={args.energy_threshold_eV:.17g}",
        f"rrea.field_taper_r_start_m={args.field_taper_r_start_m:.17g}",
        f"rrea.field_taper_r_end_m={args.field_taper_r_end_m:.17g}",
        "rrea.ion_drift_enable=1",
        f"rrea.ion_mobility_model={args.ion_mobility_model}",
        f"rrea.positive_ion_reduced_mobility_stp_m2_per_vs={args.positive_ion_reduced_mobility_stp_m2_per_vs:.17g}",
        f"rrea.negative_ion_reduced_mobility_stp_m2_per_vs={args.negative_ion_reduced_mobility_stp_m2_per_vs:.17g}",
        f"rrea.ion_mobility_density_ratio_floor={args.ion_mobility_density_ratio_floor:.17g}",
        f"rrea.ion_drift_cfl={args.ion_drift_cfl:.17g}",
        f"rrea.electron_mobility_model={ELECTRON_MOBILITY_MODEL}",
        f"rrea.fluid_attachment_model={FLUID_ATTACHMENT_MODEL}",
        # Chemistry channels (see LowEnergyFluidState.cpp): effective O4+
        # electron-ion loss, separate ion-ion loss, and detachment.
        f"rrea.electron_ion_recombination_coefficient_m3_s={args.electron_ion_recombination_coefficient_m3_s:.17g}",
        f"rrea.ion_ion_recombination_coefficient_m3_s={args.ion_ion_recombination_coefficient_m3_s:.17g}",
        f"rrea.fluid_detachment_frequency_s={args.detachment_frequency_s:.17g}",
        f"rrea.material_conduction_target={args.material_conduction_target:.17g}",
        f"rrea.material_conduction_guard={args.material_conduction_guard:.17g}",
        f"rrea.material_diffusion_target={args.material_diffusion_target:.17g}",
        f"rrea.material_diffusion_guard={args.material_diffusion_guard:.17g}",
        f"rrea.material_reaction_target={args.material_reaction_target:.17g}",
        f"rrea.material_reaction_guard={args.material_reaction_guard:.17g}",
        f"rrea.material_min_substeps={args.material_min_substeps}",
        f"rrea.material_max_substeps={args.material_max_substeps}",
        # One semantic table owns mobility, attachment and diffusion.
        *(
            [
                f"rrea.electron_closure_table={args.electron_closure_table}",
            ]
            if getattr(args, "electron_closure_table", None) is not None
            else []
        ),
        f"amr.max_grid_size={args.max_grid_size}",
        # Newborn native IDs, and so their counter-keyed RNG streams, follow
        # storage order, which periodic sorting permutes (WarpX sorts every four
        # steps on GPU, never on CPU). Pin sorting off on every backend until
        # newborn identity comes from ancestry; see docs/GPU_PORT.md.
        "warpx.sort_intervals=-1",
        *(
            [f"rrea.transport_omp_threads={args.transport_omp_threads}"]
            if args.transport_omp_threads is not None
            else []
        ),
        *(
            [
                "rrea.allow_restart_epoch_change=1",
                f"rrea.restart_epoch_change_reason={args.restart_epoch_reason}",
            ]
            if args.restart_epoch_break
            else []
        ),
        *(
            [
                "diagnostics.diags_names=diag0",
                "diag1",
                "chk",
                "chk.format=checkpoint",
                "chk.diag_type=Full",
                f"chk.intervals={args.checkpoint_interval}",
                f"chk.file_prefix={args.checkpoint_prefix}",
            ]
            if args.checkpoint_interval is not None
            else []
        ),
        *(
            [
                "rrea.cd_plane_z_m="
                + " ".join(f"{value:.17g}" for value in args.cd_plane_z_m),
                f"rrea.cd_plane_output_interval_s={args.cd_plane_output_interval_s:.17g}",
                f"rrea.cd_plane_core_radius_m={args.channel_radius_m:.17g}",
            ]
            if args.cd_plane_diagnostics
            else []
        ),
        *(
            [f"rrea.low_energy_cutoff_eV={args.low_energy_cutoff_eV:.17g}"]
            if args.low_energy_cutoff_eV is not None
            else []
        ),
    ]


def run_capture(args: argparse.Namespace) -> dict[str, Any]:
    has_parma = args.seed_source_model != "coleman_dwyer_7p2mev"
    has_pulse = args.seed_source_model == "parma_plus_1mev_pulse"
    args.warpx_exe = args.warpx_exe.expanduser().resolve()
    args.output_dir = Path(os.path.abspath(args.output_dir.expanduser()))
    args.base_input = args.base_input.expanduser().resolve()
    args.transport_config = transport_config = args.transport_config.expanduser().resolve()
    args.field_profile = args.field_profile.expanduser().resolve()
    args.density_profile = args.density_profile.expanduser().resolve()
    if args.run_root is not None:
        args.run_root = Path(os.path.abspath(args.run_root.expanduser()))
    if args.restart_checkpoint is None:
        validate_fresh_output_paths(args.output_dir, args.run_root)
    else:
        args.output_dir = args.output_dir.resolve()
        if args.run_root is not None:
            args.run_root = args.run_root.resolve()
    if args.checkpoint_prefix is not None:
        args.checkpoint_prefix = args.checkpoint_prefix.expanduser().resolve()
    for label, path in (
        ("WarpX executable", args.warpx_exe),
        ("base input", args.base_input),
        ("transport configuration", transport_config),
        ("field profile", args.field_profile),
        ("density profile", args.density_profile),
    ):
        if not path.is_file():
            raise SystemExit(f"missing {label}: {path}")
    restart_chk: Path | None = None
    if args.restart_checkpoint is not None:
        restart_chk = resolve_restart_checkpoint(
            args.checkpoint_prefix.parent
            if args.checkpoint_prefix is not None
            else args.output_dir,
            args.restart_checkpoint,
        )
        validate_restart_capture(args.output_dir, int(restart_chk.name[3:]))
        require_restart_video_config_match(args.output_dir, args)
    # Defer a fresh-run output wipe until every read-only preflight has passed.
    field = load_field_profile(args.field_profile)
    density = load_density_ratio_profile(args.density_profile)
    threshold = load_threshold_profile(args.density_profile)
    ensure_profiles_cover_domain(field, density, threshold)
    e0_v_per_m = args.e0_peak_kv_per_m * 1000.0
    region = find_lowest_above_threshold_region(
        e0_peak_v_per_m=e0_v_per_m,
        field=field,
        threshold=threshold,
        density_ratio=density,
    )
    # Pulse sources inject at the region's base; only PARMA needs none.
    if region is None and args.seed_source_model != "parma_continuous_column":
        raise SystemExit(f"E0={args.e0_peak_kv_per_m:g} kV/m has no above-threshold region")
    cd_reference = coleman_dwyer_profile_reference(
        field=field,
        density_ratio=density,
        threshold_profile=threshold,
        e0_peak_v_per_m=e0_v_per_m,
    )
    args.cd_plane_z_m = (
        build_cd_plane_z_m(cd_reference, args.n_cell_z)
        if args.cd_plane_diagnostics and cd_reference is not None
        else []
    )
    if args.seed_source_model == "coleman_dwyer_7p2mev":
        raw_m276 = cd_reference["curves"]["276"]["predicted_multiplication"]
        if raw_m276 is None:
            raise SystemExit(
                "C&D 276 multiplication exceeds the finite range needed to "
                "construct deterministic strict population ceilings"
            )
        # Explicit --max-*-macros win; the 276 curve supplies the rest.
        defaults = cd_population_ceilings(args.macro_count, float(raw_m276))
        for species in ("electron", "photon", "positron"):
            if getattr(args, f"max_{species}_macros") is None:
                setattr(args, f"max_{species}_macros", defaults[species])
    population_ceilings = (
        {
            "electron": args.max_electron_macros,
            "photon": args.max_photon_macros,
            "positron": args.max_positron_macros,
        }
        if args.max_electron_macros is not None
        else None
    )

    # Full-domain ambient-field audit: the intended super-threshold avalanche
    # column is the experiment, but the radial taper must not create another
    # super-threshold region. Check above the field top too, where the curl-free
    # fringe stays finite while the threshold falls.
    taper_audit = audit_ambient_field(
        field=field,
        threshold=threshold,
        e0_peak_v_per_m=e0_v_per_m,
        domain_radius_m=args.domain_radius_m,
        taper_r_start_m=args.field_taper_r_start_m,
        taper_r_end_m=args.field_taper_r_end_m,
    )
    fringe_loc = taper_audit["max_fringe_er_over_threshold_location"]
    print(
        f"field taper {args.field_taper_r_start_m:g}->"
        f"{args.field_taper_r_end_m:g} m: column potential span "
        f"{taper_audit['column_potential_span_v'] / 1e6:.1f} MV, max fringe |Er| = "
        f"{taper_audit['max_fringe_er_v_per_m'] / 1e3:.1f} kV/m "
        f"({taper_audit['max_fringe_over_e0']:.2f} x E0), worst "
        f"{taper_audit['max_fringe_er_over_threshold']:.2f}x local threshold at "
        f"r={fringe_loc['r_m']:.0f} m z={fringe_loc['z_m']:.0f} m; "
        f"super-threshold fringe pockets: "
        f"{taper_audit['n_super_threshold_pockets']} "
        "(full-domain field-grid audit)"
    )
    if not taper_audit["passed"] and not args.allow_taper_fringe:
        worst = max(taper_audit["pockets"], key=lambda p: p["er_over_threshold"])
        raise SystemExit(
            f"ambient field audit FAILED: {taper_audit['n_super_threshold_pockets']} "
            "super-threshold radial-fringe pocket(s) (worst "
            f"|Er| {abs(worst['er_v_per_m']) / 1e3:.1f} kV/m = "
            f"{worst['er_over_threshold']:.2f}x "
            f"threshold at r={worst['r_m']:.0f} m z={worst['z_m']:.0f} m); pass "
            "--allow-taper-fringe to accept this audited configuration"
        )
    if not taper_audit["passed"]:
        print(
            "WARNING: --allow-taper-fringe set; accepting "
            f"{taper_audit['n_super_threshold_pockets']} super-threshold pocket(s) "
            "(full-domain field-grid audit)"
        )

    # Closure-range preflight: the en_table closure aborts (by declared
    # policy, not clamp) the first time any cell's E/N exceeds its certified
    # ceiling.  The worst INITIAL E/N is knowable right here from the same
    # field map, so a hot configuration dies at submit time instead of on
    # the first advance of a queued allocation.  Screening only lowers |E|
    # afterwards; a mid-run excursion above this bound would be new physics.
    table_path = Path(args.electron_closure_table)
    max_en_td = closure_table_max_en_td(table_path)
    worst_en_td = taper_audit["worst_en_td"]
    print(
        f"closure preflight: worst initial E/N {worst_en_td:.1f} Td vs "
        f"certified max {max_en_td:g} Td ({table_path.name})"
    )
    if worst_en_td > max_en_td:
        where = taper_audit["max_e_over_threshold_location"]
        raise SystemExit(
            f"closure preflight FAILED: initial E/N reaches "
            f"{worst_en_td:.1f} Td at r={where['r_m']:.0f} m "
            f"z={where['z_m']:.0f} m, above the closure table's certified "
            f"{max_en_td:g} Td; the engine would abort on the first "
            "advance. Lower E0 or certify a wider table."
        )

    case_id = f"profiled_video_E0_{args.e0_peak_kv_per_m:g}kVpm".replace(".", "p")
    if args.seed_source_model == "coleman_dwyer_7p2mev":
        case_id += f"_cd_r{args.seed_realization_id:03d}"
        seed_kwargs: dict[str, Any] = {
            "kinetic_energies_eV": conditioned_exponential_seed_energies(
                args.macro_count,
                realization_id=args.seed_realization_id,
            ),
            "fixed_axial_position": True,
            "source_region": "coleman_dwyer_fixed_z_disk",
            "direction_model": "coleman_dwyer_positive_z",
        }
    else:
        seed_kwargs = {}
    if has_parma:
        case_id += f"_parma{'_pulse' if has_pulse else ''}_r{args.seed_realization_id:03d}"
    schedule = args.output_dir / "profiled_video_seed_schedule.csv"
    parma_table = args.output_dir / "parma_source_table.txt"
    parma_audit: dict | None = None
    sampling_interval_s = args.stop_time_s
    if restart_chk is not None and has_parma:
        if not parma_table.exists():
            raise SystemExit(
                f"restart requires the stored PARMA source table {parma_table}; "
                "it travels with the run directory (olivia.py pull-restart)"
            )
        previous = json.loads((args.output_dir / "command.json").read_text(encoding="utf-8"))
        previous_source = previous["metadata"]["parma_source"]
        sampling_interval_s = previous_source["sampling_interval_s"]

    def write_schedule(target: Path) -> float:
        """Superpose pulse + PARMA sources; the C&D cohort remains isolated."""
        nonlocal parma_audit
        total = 0.0
        if has_pulse or not has_parma:
            total = write_seed_schedule(
                target,
                case_id=case_id,
                injection_z_m=region.injection_z_m,
                macro_count=args.macro_count,
                total_weight_real_electrons=args.injected_real_electrons,
                seed_time_window_s=args.seed_time_window_s,
                source_radius_m=args.channel_radius_m,
                **seed_kwargs,
            )
        if has_parma:
            # Fresh runs query PARMA and the drag once and store the table;
            # restarts regenerate the schedule from the stored table alone.
            table = ensure_parma_source_table(
                parma_table,
                latitude_deg=PARMA_CONTINUOUS_SOURCE_DEFAULTS["latitude_deg"],
                longitude_deg=PARMA_CONTINUOUS_SOURCE_DEFAULTS["longitude_deg"],
                minimum_energy_eV=PARMA_CONTINUOUS_SOURCE_DEFAULTS[
                    "minimum_energy_eV"
                ],
                maximum_energy_eV=SCHEMA6_CHARGED_ENERGY_MAX_EV,
                probe=args.warpx_exe.parent / "rrea_interaction_table_probe",
                transport_config=transport_config,
            )
            background, parma_audit = write_parma_seed_schedule(
                target,
                table=table,
                case_id=case_id,
                macro_count=(int(PARMA_CONTINUOUS_SOURCE_DEFAULTS["macro_count"])
                             if has_pulse else args.macro_count),
                realization_id=args.seed_realization_id,
                stop_time_s=args.stop_time_s,
                source_radius_m=args.parma_source_radius_m,
                sampling_interval_s=sampling_interval_s,
                append_offset=args.macro_count if has_pulse else 0,
            )
            total += background
        return total

    if restart_chk is None:
        # Every input is validated (profiles, above-threshold region, taper
        # fringe and closure range), so it is safe to wipe/create the output dir.
        args.output_dir = prepare_fresh_output_directory(
            args.output_dir,
            run_root=args.run_root,
            overwrite=args.overwrite_output,
        )
    capture_rows, render_map = build_frame_schedule(
        dt_s=args.dt_s,
        frame_interval_s=args.video_frame_interval_s,
        final_time_s=(
            args.video_capture_final_time_s
            if args.video_capture_final_time_s is not None
            else args.stop_time_s
        ),
        warp_capture_points=(
            parse_warp_control_points(args.warp_capture_spec, flag_name="--warp-capture-spec")
            if args.warp_capture_spec
            else None
        ),
        warp_capture_fps=args.warp_capture_fps,
        retained_rows=(read_csv_rows(args.output_dir / "video_capture_steps.csv")
                       if restart_chk is not None else None),
        append_after_step=(int(restart_chk.name[3:]) if restart_chk is not None else 0),
    )
    print(f"video capture schedule: {len(capture_rows)} frames", flush=True)
    frame_schedule = args.output_dir / "video_capture_steps.csv"
    # Generate each schedule once. On restart, check both numeric prefixes
    # before replacing either stored schedule with its validated extension.
    with tempfile.TemporaryDirectory(prefix="rrea_schedules_") as scratch:
        seed_probe = Path(scratch) / schedule.name
        frame_probe = Path(scratch) / frame_schedule.name
        injected = write_schedule(seed_probe)
        write_capture_schedule(frame_probe, capture_rows)
        update_seeds = update_frames = True
        if restart_chk is not None:
            update_seeds = require_compatible_schedule(
                seed_probe, schedule, "seed schedule", allow_append=True)
            update_frames = require_compatible_schedule(
                frame_probe, frame_schedule, "frame schedule", allow_append=True)
        if update_seeds:
            shutil.copyfile(seed_probe, schedule)
        if update_frames:
            shutil.copyfile(frame_probe, frame_schedule)
    # Record the full-domain worst |E|/threshold and initial E/N.
    write_json(args.output_dir / "ambient_field_audit.json", taper_audit)
    video_dir = args.output_dir / "rrea_video"
    # The render map must follow an extended schedule too, or the appended
    # frames are captured into the stream and then never rendered.
    if update_frames:
        write_json(args.output_dir / "video_render_map.json", render_map)

    max_step = resolve_max_step(args.max_step, args.stop_time_s, args.dt_s)
    command = [
        *launcher_prefix(args.launcher, args.ranks, args.cpus_per_rank, args.nice),
        str(args.warpx_exe),
        str(args.base_input),
        # adaptive_resample_v1 population control is selected by a positive
        # target; zero reverts to strict_abort_v1 ceilings.
        *profiled_overrides(
            args=args,
            case_id=case_id,
            run_dir=args.output_dir,
            schedule=schedule,
            e0_v_per_m=e0_v_per_m,
            injected_real_particles=injected,
            region=region,
            stop_time_s=args.stop_time_s,
            max_step=max_step,
            transport_config=transport_config,
            population_target=float(args.population_target_electron_macros),
            population_control_interval=args.population_control_interval,
            population_controller_rng_salt=args.population_controller_rng_salt,
            population_split_min_weight=(
                args.population_split_min_weight
                if args.population_split_min_weight is not None or parma_audit is None
                else (parma_audit["physical_particles_per_s_total"] * sampling_interval_s
                      + parma_audit["bath_injected_real_particles"]) / parma_audit["macro_count"]),
            population_ceilings=population_ceilings,
            rng_seed=args.transport_rng_seed,
        ),
        *capture_feature_overrides(
            args,
            video_dir=video_dir,
            frame_schedule=frame_schedule,
        ),
        *validate_user_warpx_overrides(args.extra_warpx_arg),
        *([f"amr.restart={restart_chk}"] if restart_chk is not None else []),
    ]
    omp_threads = args.cpus_per_rank if args.omp_threads is None else args.omp_threads
    driver_arguments = resolved_driver_arguments(args)
    driver_arguments.update(
        {
            "effective_max_step": max_step,
            "effective_omp_threads": omp_threads,
            "resolved_restart_checkpoint": (
                str(restart_chk.resolve()) if restart_chk is not None else None
            ),
        }
    )
    if restart_chk is not None:
        trim_restart_capture(
            args.output_dir,
            restart_chk,
            cd_plane_altitudes_m=[
                z_m + ALTITUDE_AT_Z0_M for z_m in args.cd_plane_z_m
            ],
        )
    # Build run-identifying fields once for both output records.
    run_identity = {
        "e0_peak_kv_per_m": args.e0_peak_kv_per_m,
        "ranks": args.ranks,
        "cpus_per_rank": args.cpus_per_rank,
        "omp_threads": omp_threads,
        "seed_source_model": args.seed_source_model,
        "seed_realization_id": args.seed_realization_id,
        "transport_rng_seed": args.transport_rng_seed,
        "population_controller_rng_salt": args.population_controller_rng_salt,
        "cd_seed_design": args.cd_seed_design,
        "cd_seed_escalation_reason": args.cd_seed_escalation_reason,
        "cd_population_resampling_reason": args.cd_population_resampling_reason,
        "cd_diagnostic_seed_count": args.cd_diagnostic_seed_count,
        "cd_diagnostic_reason": args.cd_diagnostic_reason,
        "cd_population_resampled": bool(args.cd_allow_population_resampling),
        "restart_epoch_break": bool(args.restart_epoch_break),
        "restart_epoch_reason": args.restart_epoch_reason,
        "cd_seed_permutation": (
            "splitmix64_fisher_yates_v1"
            if args.seed_source_model == "coleman_dwyer_7p2mev"
            else None
        ),
        "parma_source": parma_audit,
        "pulse_real_electrons": args.injected_real_electrons if has_pulse else 0.0,
        "injected_real_particles": injected,
        "coleman_dwyer_reference": cd_reference,
        "frame_schedule": str(frame_schedule),
    }
    run_command(
        command,
        args.output_dir,
        omp_threads,
        provenance={
            **run_identity,
            "warpx_executable": str(args.warpx_exe.resolve()),
            "base_input": str(args.base_input.resolve()),
            "field_profile": str(args.field_profile.resolve()),
            "density_profile": str(args.density_profile.resolve()),
            "transport_config": str(transport_config),
            "seed_schedule": str(schedule),
            "launcher": args.launcher,
            "resolved_driver_arguments": driver_arguments,
        },
        completion_artifacts_to_invalidate=("profiled_video_capture_summary.json",),
        checkpoint_rotation=(
            (args.checkpoint_prefix, args.checkpoint_keep)
            if args.checkpoint_interval is not None
            and args.checkpoint_prefix is not None
            else None
        ),
    )
    video_frame_count = validate_completed_video_capture(
        args.output_dir,
        [int(row["step"]) for row in capture_rows],
    )
    video_logical_view_count = len(render_map["frames"])
    reduced_path = args.output_dir / "rrea_reduced.csv"
    reduced_rows = read_csv_rows(reduced_path) if reduced_path.exists() else []
    summary = {
        **run_identity,
        "status": "completed",
        "interaction_table_config": str(transport_config),
        "restart_checkpoint": str(restart_chk) if restart_chk is not None else None,
        "case_id": case_id,
        "stop_time_s": args.stop_time_s,
        "video_capture_final_time_s": (
            args.video_capture_final_time_s
            if args.video_capture_final_time_s is not None
            else args.stop_time_s
        ),
        "max_step": max_step,
        "dt_s": args.dt_s,
        "macro_count": args.macro_count,
        "injected_real_electrons": args.injected_real_electrons,
        "seed_time_window_s": args.seed_time_window_s,
        "population_target_electron_macros": args.population_target_electron_macros,
        "population_control_interval": args.population_control_interval,
        "strict_population_ceilings": population_ceilings,
        "cd_plane_diagnostics": args.cd_plane_diagnostics,
        "cd_plane_z_m": args.cd_plane_z_m,
        "cd_plane_output_interval_s": args.cd_plane_output_interval_s,
        "cd_plane_output": (
            str(args.output_dir / "rrea_cd_plane_flux.csv")
            if args.cd_plane_diagnostics
            else None
        ),
        "source_radius_m": (
            args.parma_source_radius_m
            if args.seed_source_model == "parma_continuous_column"
            else args.channel_radius_m
        ),
        "above_threshold_region": region_to_dict(region),
        "profiles": profile_metadata(field, density, threshold),
        "base_input_path": str(args.base_input.resolve()),
        "warpx_executable_path": str(args.warpx_exe.resolve()),
        "seed_schedule_path": str(schedule.resolve()),
        "video_frame_count": video_frame_count,
        "video_logical_view_count": video_logical_view_count,
        "video_frame_interval_s": args.video_frame_interval_s,
        "domain_radius_m": args.domain_radius_m,
        "domain_height_m": DOMAIN_HEIGHT_M,
        "n_cell_r": args.n_cell_r,
        "n_cell_z": args.n_cell_z,
        "field_taper_r_start_m": args.field_taper_r_start_m,
        "field_taper_r_end_m": args.field_taper_r_end_m,
        "video_nr": args.video_nr,
        "video_nz": args.video_nz,
        "video_r_min_m": args.video_r_min_m,
        "video_r_max_m": args.video_r_max_m,
        "video_z_min_m": args.video_z_min_m,
        "video_z_max_m": args.video_z_max_m,
        "video_dir": str(video_dir),
        "ion_mobility_model": args.ion_mobility_model,
        "positive_ion_reduced_mobility_stp_m2_per_vs": args.positive_ion_reduced_mobility_stp_m2_per_vs,
        "negative_ion_reduced_mobility_stp_m2_per_vs": args.negative_ion_reduced_mobility_stp_m2_per_vs,
        "ion_mobility_density_ratio_floor": args.ion_mobility_density_ratio_floor,
        "ion_drift_cfl": args.ion_drift_cfl,
        "electron_mobility_model": ELECTRON_MOBILITY_MODEL,
        "fluid_attachment_model": FLUID_ATTACHMENT_MODEL,
        "electron_ion_recombination_coefficient_m3_s": (
            args.electron_ion_recombination_coefficient_m3_s
        ),
        "ion_ion_recombination_coefficient_m3_s": (
            args.ion_ion_recombination_coefficient_m3_s
        ),
        "detachment_frequency_s": args.detachment_frequency_s,
        "material_conduction_target": args.material_conduction_target,
        "material_conduction_guard": args.material_conduction_guard,
        "material_diffusion_target": args.material_diffusion_target,
        "material_diffusion_guard": args.material_diffusion_guard,
        "material_reaction_target": args.material_reaction_target,
        "material_reaction_guard": args.material_reaction_guard,
        "material_min_substeps": args.material_min_substeps,
        "material_max_substeps": args.material_max_substeps,
        "electron_closure_table": (
            str(args.electron_closure_table)
            if args.electron_closure_table is not None
            else None
        ),
        "render_map": str(args.output_dir / "video_render_map.json"),
        "final_reduced_row": reduced_rows[-1] if reduced_rows else None,
        "extra_warpx_arg": args.extra_warpx_arg,
    }
    write_json(args.output_dir / "profiled_video_capture_summary.json", summary)
    return summary


def require_together(args: argparse.Namespace, flag: str, reason: str) -> None:
    """Two options that only mean anything together, checked both directions.

    "Supplied" is an identity test, not truthiness: a legal 0 or 0.0 counts as
    given, which `value in (None, False)` would get wrong because 0 == False.
    """
    def supplied(name: str) -> bool:
        value = getattr(args, name)
        return value is not None and value is not False

    a, b = f"--{flag.replace('_', '-')}", f"--{reason.replace('_', '-')}"
    if supplied(flag) and not supplied(reason):
        raise SystemExit(f"{a} requires {b}")
    if supplied(reason) and not supplied(flag):
        raise SystemExit(f"{b} requires {a}")


# The production low-energy fluid closure.  Both models read the measured
# dry-air swarm bundle: a field-dependent flux mobility mu(E/N) and attachment
# frequency nu(E/N, rho).
ELECTRON_MOBILITY_MODEL = "en_table_flux_v1"
FLUID_ATTACHMENT_MODEL = "en_table_attachment_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warpx-exe", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--run-root",
        type=Path,
        help="Required for a fresh run; --output-dir must resolve strictly below it.",
    )
    parser.add_argument(
        "--overwrite-output",
        action="store_true",
        help="Replace a non-empty output only when it carries the RREA ownership marker.",
    )
    parser.add_argument("--base-input", type=Path, default=DEFAULT_BASE_INPUT)
    parser.add_argument(
        "--transport-config", type=Path,
        default=ROOT / "rrea_transport_tables/schema6/production/transport_physics.json")
    parser.add_argument("--field-profile", type=Path, default=DEFAULT_FIELD_PROFILE)
    parser.add_argument("--density-profile", type=Path, default=DEFAULT_DENSITY_PROFILE)
    parser.add_argument("--launcher", choices=["none", "srun", "mpiexec"], default="none")
    parser.add_argument(
        "--nice", type=int, default=0,
        help="process niceness for a local launcher (default: 0)")
    parser.add_argument("--ranks", type=int, default=1)
    parser.add_argument("--cpus-per-rank", type=int, default=1)
    parser.add_argument(
        "--omp-threads",
        type=int,
        help=(
            "Global OMP_NUM_THREADS value. Defaults to --cpus-per-rank; set to 1 "
            "when allocating multiple CPUs per rank solely for scoped RREA OpenMP."
        ),
    )
    parser.add_argument("--e0-peak-kv-per-m", type=finite_float, required=True)
    parser.add_argument(
        "--population-target-electron-macros",
        type=nonnegative_int,
        default=None,
        help="adaptive_resample_v1 electron macro target (unbiased, CIC-charge-"
             "exact null-space thinning/split band; strict ceilings stay as "
             "backstops). "
             "The default comes from config/rrea_defaults.json through "
             "rrea_run_support.py. 0 disables population control "
             "(strict_abort_v1).",
    )
    parser.add_argument(
        "--population-control-interval",
        type=int,
        default=DEFAULT_POPULATION_CONTROL_INTERVAL,
        help="steps between population-control passes",
    )
    parser.add_argument(
        "--population-controller-rng-salt",
        type=nonnegative_int,
        default=DEFAULT_POPULATION_CONTROLLER_RNG_SALT,
        help="controller-only RNG ensemble salt; transport draws are unchanged",
    )
    parser.add_argument(
        "--population-split-min-weight", type=finite_float, default=None,
        help="never split macros below this weight (default: the seed "
             "macro weight = injected-real-electrons / macro-count)",
    )
    parser.add_argument("--max-electron-macros", type=int)
    parser.add_argument("--max-photon-macros", type=int)
    parser.add_argument("--max-positron-macros", type=int)
    parser.add_argument(
        "--restart-checkpoint",
        default=None,
        help="resume an interrupted run instead of starting fresh: a chkNNNNNN "
             "directory, or 'auto' = the highest complete chkNNNNNN in the "
             "--checkpoint-prefix directory (default: the output dir). The capture dir is NOT wiped; capture files "
             "are trimmed to the checkpoint step and the engine appends. "
             "Use --restart-epoch-break for an intentional engine, MPI-layout "
             "or physics-model transition; seed/frame schedules must remain "
             "compatible with the retained capture prefix.",
    )
    # Seed macro count comes from the single-source config. Population control,
    # when enabled, manages the maximum macro population independently.
    parser.add_argument("--macro-count", type=int,
                        help="Pulse macros in combined mode; PARMA macros per initial interval in PARMA-only mode. Combined background resolution comes from config/rrea_defaults.json.")
    parser.add_argument("--injected-real-electrons", type=finite_float,
                        help="Electron population in the one-off pulse (C&D: fixed cohort count)")
    parser.add_argument("--seed-time-window-s", type=finite_float, default=None)
    parser.add_argument(
        "--seed-source-model",
        choices=[
            "parma_plus_1mev_pulse",
            "parma_continuous_column",
            "coleman_dwyer_7p2mev",
        ],
        default=None,
        help="PARMA background alone or with one 1 MeV pulse; C&D stays isolated. Default from config/rrea_defaults.json.",
    )
    parser.add_argument(
        "--parma-source-radius-m",
        type=finite_float,
        default=None,
        help=(
            "PARMA source cylinder radius; the physical injection rate scales "
            "with its area (default from parma_continuous_source in the JSON)"
        ),
    )
    parser.add_argument("--seed-realization-id", type=nonnegative_int, default=0)
    parser.add_argument(
        "--transport-rng-seed",
        type=nonnegative_int,
        help=(
            "Explicit transport RNG seed for independent production-source "
            "realizations. C&D runs derive and enforce 7000001 + realization id."
        ),
    )
    parser.add_argument(
        "--cd-seed-escalation-200k",
        action="store_true",
        help="Explicitly select the 200k unit-weight C&D seed escalation.",
    )
    parser.add_argument(
        "--cd-seed-escalation-reason",
        help="Required reason for the 200k C&D escalation (for provenance).",
    )
    parser.add_argument(
        "--cd-diagnostic-seed-count",
        type=int,
        help=(
            "Run a reduced-statistics ENGINE DIAGNOSTIC rather than a scientific "
            "realization. Only defensible for paired comparisons such as a "
            "timestep ladder, where seed noise cancels between rungs."
        ),
    )
    parser.add_argument(
        "--cd-diagnostic-reason",
        help="Required reason for a reduced-statistics diagnostic (provenance).",
    )
    parser.add_argument(
        "--cd-allow-population-resampling",
        action="store_true",
        help=(
            "Permit adaptive_resample_v1 on a Coleman-Dwyer run. CIC-conserving "
            "stochastic reweighting keeps the flux estimate unbiased, but the "
            "realization is labelled resampled and must never be pooled with "
            "resampling-off ones."
        ),
    )
    parser.add_argument(
        "--cd-population-resampling-reason",
        help="Required reason for permitting C&D population resampling (provenance).",
    )
    parser.add_argument(
        "--restart-epoch-break",
        action="store_true",
        help=(
            "Record an intentional engine, MPI-layout or physics-model change. "
            "Permits supported rank-count and reduced-schema deviations, but "
            "does not override state-format or mesh checks."
        ),
    )
    parser.add_argument(
        "--restart-epoch-reason",
        help="Required reason for --restart-epoch-break (engine token and provenance).",
    )
    parser.add_argument("--cd-plane-diagnostics", action="store_true")
    parser.add_argument(
        "--cd-plane-output-interval-s", type=finite_float, default=1.0e-8
    )
    parser.add_argument(
        "--channel-radius-m", type=finite_float, default=None
    )
    parser.add_argument("--n-cell-r", type=int, default=None)
    parser.add_argument("--n-cell-z", type=int, default=None)
    parser.add_argument(
        "--allow-taper-fringe",
        action="store_true",
        help=(
            "Override the taper fringe-field guard. Any curl-free field "
            "that terminates the profile laterally must carry the column "
            "potential radially across the fade annulus (fringe |Er| ~= "
            "1.875*(delta_phi/2)/width); by default the driver aborts if that "
            "radial field reaches the local density-scaled RREA threshold."
        ),
    )
    # Defaults for the block below come from config/rrea_defaults.json
    # capture_defaults (late-resolved after parsing).
    parser.add_argument("--dt-s", type=finite_float, default=None)
    parser.add_argument("--stop-time-s", type=finite_float, default=None)
    parser.add_argument(
        "--video-capture-final-time-s",
        type=finite_float,
        help=(
            "Optional fixed end of the compact video schedule. Restart pulse "
            "extensions may retain the original schedule while plane "
            "diagnostics continue to a later --stop-time-s."
        ),
    )
    parser.add_argument("--max-step", type=int)
    parser.add_argument("--diag-interval", type=int, default=None)
    parser.add_argument("--video-nr", type=int, default=None)
    parser.add_argument("--video-nz", type=int, default=None)
    parser.add_argument("--video-frame-interval-s", type=finite_float, default=None,
                        help="capture spacing from t=0; restarts retain prior frames and change only future spacing")
    parser.add_argument(
        "--warp-capture-spec",
        help="Optional cinematic-playback control points "
        "('sim_time_us:playback_rate_us_per_s', comma-separated, same format "
        "as the renderer's --warp-spec). Adds capture frames paced so the "
        "time-warped render has real physical states wherever it dwells.",
    )
    parser.add_argument(
        "--warp-capture-fps",
        type=finite_float,
        default=None,
        help="Capture-frame rate (per second of warped video) for "
        "--warp-capture-spec; 12 halves the data of the 24 fps playback while "
        "keeping interpolation gaps invisible.",
    )
    parser.add_argument(
        "--extra-warpx-arg",
        action="append",
        default=[],
        help="Additional complete, nonphysics WarpX override, e.g. warpx.verbose=1.",
    )
    parser.add_argument("--max-grid-size", type=int, default=None)
    parser.add_argument("--transport-omp-threads", type=int)
    parser.add_argument("--checkpoint-interval", type=int)
    parser.add_argument("--checkpoint-prefix", type=Path,
                        help="checkpoint prefix (default: <output-dir>/chk)")
    parser.add_argument(
        "--checkpoint-keep",
        type=int,
        default=None,
        help="COMPLETE checkpoints to retain while running (config default).",
    )
    parser.add_argument("--low-energy-cutoff-eV", type=finite_float)
    parser.add_argument(
        "--ion-mobility-model",
        choices=["absolute", "reduced_stp_density_scaled"],
        default=None,
    )
    parser.add_argument("--positive-ion-reduced-mobility-stp-m2-per-vs", type=finite_float, default=None)
    parser.add_argument("--negative-ion-reduced-mobility-stp-m2-per-vs", type=finite_float, default=None)
    parser.add_argument("--ion-mobility-density-ratio-floor", type=finite_float, default=None)
    parser.add_argument("--ion-drift-cfl", type=finite_float, default=None)
    # Ion-ion recombination (exact pair-loss update in the engine) and
    # collisional detachment.  Defaults come from config/rrea_defaults.json:
    # Room-temperature lumped dry-air chemistry: O4+ is the effective
    # positive-ion proxy for electron-ion loss; ion-ion loss remains separate.
    parser.add_argument(
        "--electron-ion-recombination-coefficient-m3-s",
        type=finite_float,
        default=None,
    )
    parser.add_argument(
        "--ion-ion-recombination-coefficient-m3-s",
        type=finite_float,
        default=None,
    )
    parser.add_argument("--detachment-frequency-s", type=finite_float, default=None)
    parser.add_argument("--material-conduction-target", type=finite_float, default=None)
    parser.add_argument("--material-conduction-guard", type=finite_float, default=None)
    parser.add_argument("--material-diffusion-target", type=finite_float, default=None)
    parser.add_argument("--material-diffusion-guard", type=finite_float, default=None)
    parser.add_argument("--material-reaction-target", type=finite_float, default=None)
    parser.add_argument("--material-reaction-guard", type=finite_float, default=None)
    parser.add_argument("--material-min-substeps", type=int, default=None)
    parser.add_argument("--material-max-substeps", type=int, default=None)
    parser.add_argument("--electron-closure-table", type=Path, default=None)
    args = parser.parse_args()
    # Late resolution from the single source (config/rrea_defaults.json):
    # these argparse defaults are None on purpose so the json value fills
    # every launch path identically and never bakes into provenance.
    for key in (
        "dt_s", "stop_time_s", "n_cell_z",
        "channel_radius_m", "seed_time_window_s", "diag_interval",
        "video_frame_interval_s", "warp_capture_fps",
        "ion_mobility_model",
        "positive_ion_reduced_mobility_stp_m2_per_vs",
        "negative_ion_reduced_mobility_stp_m2_per_vs",
        "ion_mobility_density_ratio_floor", "ion_drift_cfl",
        "electron_ion_recombination_coefficient_m3_s",
        "ion_ion_recombination_coefficient_m3_s", "detachment_frequency_s",
        "material_conduction_target", "material_conduction_guard",
        "material_diffusion_target", "material_diffusion_guard",
        "material_reaction_target", "material_reaction_guard",
        "material_min_substeps", "material_max_substeps",
        "checkpoint_keep",
    ):
        if getattr(args, key) is None:
            setattr(args, key, CAPTURE_DEFAULTS[key])
    # Resolve the one closure table from the canonical configuration.
    if args.electron_closure_table is None and CAPTURE_DEFAULTS.get(
        "electron_closure_table"
    ):
        args.electron_closure_table = ROOT / CAPTURE_DEFAULTS["electron_closure_table"]
    # One production geometry, derived from config/rrea_defaults.json.
    args.domain_radius_m = DOMAIN_RADIUS_M
    if args.n_cell_r is None:
        args.n_cell_r = round(DOMAIN_RADIUS_M / CELL_SIZE_M)
    args.field_taper_r_start_m = TAPER_START_DOMAIN_FRACTION * DOMAIN_RADIUS_M
    args.field_taper_r_end_m = DOMAIN_RADIUS_M
    # Capture the full domain and crop only while rendering. Pixel counts
    # default to native resolution; the C&D campaign may request thumbnails.
    args.video_r_min_m = 0.0
    args.video_r_max_m = args.domain_radius_m
    args.video_z_min_m = 0.0
    args.video_z_max_m = CAPTURE_DEFAULTS["domain_height_m"]
    args.energy_threshold_eV = ENERGY_THRESHOLD_EV
    if args.video_nr is None:
        args.video_nr = args.n_cell_r
    if args.video_nz is None:
        args.video_nz = args.n_cell_z
    # Epoch escalation: flag and reason are paired in both directions and are
    # meaningful only on a restart.
    require_together(args, "restart_epoch_break", "restart_epoch_reason")
    if args.restart_epoch_break and args.restart_checkpoint is None:
        raise SystemExit(
            "--restart-epoch-break requires "
            "--restart-checkpoint: epoch flags only mean something on a "
            "restart"
        )
    # The reason rides an engine ParmParse token; AMReX re-tokenizes argv on
    # whitespace, so an embedded space would silently truncate the recorded
    # reason and leave a stray garbage token.
    if args.restart_epoch_reason and any(
        ch.isspace() for ch in args.restart_epoch_reason
    ):
        raise SystemExit(
            "--restart-epoch-reason must be a single whitespace-free token "
            "(e.g. engine_or_rank_change)"
        )
    if args.max_grid_size is None:
        # Keep one configuration-owned value for every rank count so the
        # BoxArray remains checkpoint-portable. Scaling must not change mesh
        # layout.
        args.max_grid_size = CAPTURE_DEFAULTS["max_grid_size"]
    if args.seed_source_model is None:
        # The JSON selector is authoritative; the CLI flag exists for explicit
        # per-run and campaign choices only.
        args.seed_source_model = DEFAULT_SEED_SOURCE_MODEL
    if (
        args.parma_source_radius_m is not None
        and args.seed_source_model == "coleman_dwyer_7p2mev"
    ):
        raise SystemExit(
            "--parma-source-radius-m requires the PARMA continuous source"
        )
    if args.seed_source_model == "coleman_dwyer_7p2mev":
        expected_transport_seed = cd_transport_rng_seed(args.seed_realization_id)
        if (
            args.transport_rng_seed is not None
            and args.transport_rng_seed != expected_transport_seed
        ):
            raise SystemExit(
                "Coleman-Dwyer transport RNG seed is fixed by --seed-realization-id"
            )
        args.transport_rng_seed = expected_transport_seed
        require_together(args, "cd_diagnostic_seed_count", "cd_diagnostic_reason")
        if args.cd_diagnostic_seed_count is not None and (
                args.cd_seed_escalation_200k or args.cd_seed_escalation_reason):
            raise SystemExit(
                "a reduced-statistics diagnostic cannot also be a 200k escalation"
            )
        desired_seed_count = (
            args.cd_diagnostic_seed_count
            if args.cd_diagnostic_seed_count is not None
            else CD_ESCALATED_SEED_COUNT
            if args.cd_seed_escalation_200k
            else CD_INITIAL_SEED_COUNT
        )
        try:
            args.cd_seed_design = (
                cd_diagnostic_seed_design(
                    desired_seed_count, args.cd_diagnostic_reason
                )
                if args.cd_diagnostic_seed_count is not None
                else cd_seed_design(
                    desired_seed_count, args.cd_seed_escalation_reason
                )
            )
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        args.macro_count = (
            desired_seed_count if args.macro_count is None else args.macro_count
        )
        args.injected_real_electrons = (
            float(desired_seed_count)
            if args.injected_real_electrons is None
            else args.injected_real_electrons
        )
        args.population_target_electron_macros = (
            0
            if args.population_target_electron_macros is None
            else args.population_target_electron_macros
        )
        if (
            args.macro_count != desired_seed_count
            or args.injected_real_electrons != float(desired_seed_count)
        ):
            raise SystemExit(
                "Coleman-Dwyer source requires exactly 100000 unit-weight seeds, "
                "exactly 200000 with the explicit escalation flag and reason, or "
                "the count named by --cd-diagnostic-seed-count"
            )
        # Resampling is off by default because a C&D realization is a fixed-seed
        # counting experiment. It can be permitted deliberately: at high peak
        # field the population, not physics time, is what exhausts the wall
        # clock, and a run killed before its front clears the top plane yields
        # no endpoint at all.
        require_together(
            args, "cd_allow_population_resampling",
            "cd_population_resampling_reason")
        if (
            args.population_target_electron_macros != 0
            and not args.cd_allow_population_resampling
        ):
            raise SystemExit(
                "Coleman-Dwyer source requires population resampling disabled "
                "unless --cd-allow-population-resampling is given with a reason"
            )
        if args.cd_allow_population_resampling and args.population_target_electron_macros == 0:
            raise SystemExit(
                "--cd-allow-population-resampling requires a nonzero "
                "--population-target-electron-macros"
            )
        if args.seed_time_window_s != 0.0:
            raise SystemExit("Coleman-Dwyer source requires simultaneous injection")
        if not math.isclose(
            args.channel_radius_m,
            SOURCE_RADIUS_M,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise SystemExit("Coleman-Dwyer source requires the fixed 42 m source disk")
    else:
        if args.cd_seed_escalation_200k or args.cd_seed_escalation_reason:
            raise SystemExit("C&D seed escalation options require the Coleman-Dwyer source")
        args.cd_seed_design = None
        if args.seed_source_model == "parma_continuous_column":
            if args.injected_real_electrons is not None:
                raise SystemExit(
                    "the PARMA continuous source derives its physical electron "
                    "total from the flux table; do not pass "
                    "--injected-real-electrons"
                )
            if args.seed_time_window_s not in (None, 0.0):
                raise SystemExit(
                    "the PARMA continuous source owns event timing (stratified "
                    "over the full run); leave --seed-time-window-s unset"
                )
            default_macro_count = int(PARMA_CONTINUOUS_SOURCE_DEFAULTS["macro_count"])
        else:
            default_macro_count = DEFAULT_SEED_MACRO_COUNT
        if args.macro_count is None:
            args.macro_count = default_macro_count
        if args.parma_source_radius_m is None:
            args.parma_source_radius_m = float(PARMA_CONTINUOUS_SOURCE_DEFAULTS["source_radius_m"])
        if not 0.0 < args.parma_source_radius_m <= args.domain_radius_m:
            raise SystemExit(
                f"--parma-source-radius-m {args.parma_source_radius_m} m "
                f"must lie in (0, domain radius {args.domain_radius_m} m]"
            )
        if args.transport_rng_seed is None:
            # Source realizations get independent transport streams (overridable).
            args.transport_rng_seed = 7100001 + args.seed_realization_id
        args.population_target_electron_macros = (
            DEFAULT_POPULATION_TARGET_ELECTRON_MACROS
            if args.population_target_electron_macros is None
            else args.population_target_electron_macros
        )
    if args.cd_plane_diagnostics and args.seed_source_model != "coleman_dwyer_7p2mev":
        raise SystemExit("--cd-plane-diagnostics requires the Coleman-Dwyer source model")
    explicit_ceilings = (
        args.max_electron_macros,
        args.max_photon_macros,
        args.max_positron_macros,
    )
    if any(value is not None for value in explicit_ceilings):
        if not all(value is not None and value > 0 for value in explicit_ceilings):
            raise SystemExit("set all three positive strict macro ceilings together")
        if args.population_target_electron_macros > 0:
            raise SystemExit("explicit strict macro ceilings require population resampling disabled")
    if args.checkpoint_interval is not None and args.checkpoint_prefix is None:
        args.checkpoint_prefix = args.output_dir / "chk"
    if args.checkpoint_keep < 1:
        raise SystemExit("--checkpoint-keep must be >= 1")
    if (
        args.injected_real_electrons is None
        and args.seed_source_model != "parma_continuous_column"
    ):
        # The PARMA source derives its physical total from the flux table, so
        # its value stays None here and provenance records the derived number.
        args.injected_real_electrons = float(
            CAPTURE_DEFAULTS["injected_real_electrons"]
        )
    if args.seed_time_window_s < 0.0:
        raise SystemExit("--seed-time-window-s must be non-negative")
    if args.stop_time_s <= 0.0 or args.dt_s <= 0.0 or args.video_frame_interval_s <= 0.0:
        raise SystemExit("stop time, dt and video frame interval must be positive")
    if args.video_capture_final_time_s is not None and not (
        0.0 < args.video_capture_final_time_s <= args.stop_time_s
    ):
        raise SystemExit(
            "--video-capture-final-time-s must be positive and no later than --stop-time-s"
        )
    if args.electron_closure_table is None:
        raise SystemExit(
            "--electron-closure-table is required for the en_table closure"
        )
    args.electron_closure_table = args.electron_closure_table.resolve()
    if not args.electron_closure_table.is_file():
        raise SystemExit(f"closure table not found: {args.electron_closure_table}")
    if (
        args.population_split_min_weight is not None
        and args.population_split_min_weight < 0.0
    ):
        raise SystemExit("--population-split-min-weight must be non-negative")
    # Every driver knob whose only constraint is "> 0", checked after the
    # CAPTURE_DEFAULTS fill-in and the derivations above.
    REQUIRED_POSITIVE = (
        "cd_plane_output_interval_s", "ranks", "cpus_per_rank", "max_grid_size",
        "macro_count", "channel_radius_m", "n_cell_r", "n_cell_z",
        "video_nr", "video_nz", "population_control_interval",
    )
    # Only the pulse/cohort has a prescribed electron count; PARMA rates are
    # derived from the table, with their combined total recorded as particles.
    OPTIONAL_POSITIVE = (
        "omp_threads", "transport_omp_threads",
        "checkpoint_interval", "injected_real_electrons",
    )
    for name in REQUIRED_POSITIVE + OPTIONAL_POSITIVE:
        value = getattr(args, name)
        if value is None and name in OPTIONAL_POSITIVE:
            continue
        if value is None or value <= 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")
    if not math.isfinite(args.e0_peak_kv_per_m) or args.e0_peak_kv_per_m < 0:
        raise SystemExit("--e0-peak-kv-per-m must be finite and non-negative")
    return args


def main() -> None:
    summary = run_capture(parse_args())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
