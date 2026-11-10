"""Python reader of the engine's compact video frame stream, plus the
warp-schedule and reduced-CSV helpers shared by capture, render and radio."""

from __future__ import annotations

import math
import zlib

from rrea_table_pack import byte_unshuffle


# The engine writes the stream (RreaVideoFrameCodec.H): float32 values, each
# frame byte-shuffled and zlib-deflated independently, so a frame stays
# individually addressable and the stream stays append-only and truncatable at
# a frame boundary.  The per-frame byte_offset/byte_length columns of
# video_frame_metadata.csv are the index; there is no separate index file.
VIDEO_V4_FORMAT = "rrea_video_frames_binary_v4"
VIDEO_V4_ITEMSIZE = 4

# The "threshold" of the engine's *_ge_threshold arrays is a fixed diagnostic
# definition: every downstream label (renderer panels, figure captions) says
# "1 MeV", so this is a constant, not a knob.
ENERGY_THRESHOLD_EV = 1.0e6


def parse_warp_control_points(
    text: str, *, flag_name: str
) -> list[tuple[float, float]]:
    """Parse comma-separated ``sim_time_us:playback_rate_us_per_s`` points.

    Shared by capture and render; ``flag_name`` selects error-message wording.
    """
    points: list[tuple[float, float]] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        pieces = part.split(":")
        if len(pieces) != 2:
            raise SystemExit(
                f"{flag_name} entries must be 'time_us:rate_us_per_s', got: {part!r}"
            )
        t_us, rate = float(pieces[0]), float(pieces[1])
        if not (math.isfinite(t_us) and math.isfinite(rate)) or rate <= 0.0:
            raise SystemExit(
                f"{flag_name} rates must be finite and positive, got: {part!r}"
            )
        points.append((t_us, rate))
    if not points:
        raise SystemExit(
            f"{flag_name} must contain at least one 'time_us:rate_us_per_s' point"
        )
    times = [point[0] for point in points]
    if times != sorted(times):
        raise SystemExit(f"{flag_name} control-point times must be ascending")
    return points


def warp_rate_us_per_s_at(
    t_us: float, points: list[tuple[float, float]]
) -> float:
    """Piecewise-linear playback rate at ``t_us``, clamped outside the points."""
    if t_us <= points[0][0]:
        return points[0][1]
    for (t0, r0), (t1, r1) in zip(points, points[1:], strict=False):
        if t_us <= t1:
            if t1 <= t0:
                return r1
            return r0 + (r1 - r0) * (t_us - t0) / (t1 - t0)
    return points[-1][1]


def decode_v4_frame(blob: bytes, *, expected_bytes: int) -> bytes:
    """Inflate and unshuffle one frame, requiring the exact expected size."""
    raw = zlib.decompress(blob)
    if len(raw) != expected_bytes:
        raise SystemExit(
            f"video-v4 frame inflates to {len(raw)} bytes, expected "
            f"{expected_bytes}; the frame stream and its metadata disagree"
        )
    return byte_unshuffle(raw, VIDEO_V4_ITEMSIZE)


def dedup_reduced_rows_by_step(
    rows: list[dict[str, str]], *, source
) -> list[dict[str, str]]:
    """Order restart-stitched diagnostic rows (rrea_reduced.csv, current_profile.csv).

    A restarted capture can re-append diagnostic rows, so the same step can
    appear more than once: keep the last occurrence per step,
    order by step, and require at least two rows whose time_s values are
    finite and strictly increasing.  Rows come back as read (string dicts);
    column policy stays with the caller.
    """
    if not rows:
        raise SystemExit(f"{source} has no data rows")
    by_step: dict[int, dict[str, str]] = {}
    for row in rows:
        try:
            step_value = float((row.get("step") or "").strip())
        except ValueError as exc:
            raise SystemExit(f"{source} has a malformed step value") from exc
        if not math.isfinite(step_value) or not step_value.is_integer():
            raise SystemExit(f"{source} has a non-integer or non-finite step value")
        by_step[int(step_value)] = row
    ordered = [by_step[step] for step in sorted(by_step)]
    if len(ordered) < 2:
        raise SystemExit(f"{source} has fewer than 2 distinct diagnostic steps")
    try:
        times = [float((row.get("time_s") or "").strip()) for row in ordered]
    except ValueError as exc:
        raise SystemExit(f"{source} has a malformed time_s value") from exc
    if not all(math.isfinite(t) for t in times) or not all(
        later > earlier for earlier, later in zip(times, times[1:])
    ):
        raise SystemExit(
            f"{source} time_s is not finite and strictly increasing after dedup"
        )
    return ordered
