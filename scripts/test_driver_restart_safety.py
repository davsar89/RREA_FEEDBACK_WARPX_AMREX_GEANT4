"""Lean destructive-filesystem, restart, driver, and OLIVIA safety tests."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import math
import os
import time
from pathlib import Path

import pytest

from rrea_profiled_atmosphere import (
    coleman_dwyer_profile_reference,
    conditioned_exponential_seed_energies,
    launcher_prefix,
    load_density_ratio_profile,
    load_field_profile,
    load_threshold_profile,
    write_seed_schedule,
)
from rrea_run_support import (
    prepare_fresh_output_directory,
    rotate_checkpoints,
    validate_user_warpx_overrides,
)
from run_rrea_profiled_video_capture import (
    build_cd_plane_z_m,
    build_frame_schedule,
    capture_feature_overrides,
    parse_args,
    plan_cd_plane_flux_trim,
    require_compatible_schedule,
    resolve_restart_checkpoint,
    resolve_max_step,
    write_capture_schedule,
)


def test_material_accuracy_controls_resolve_and_reach_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "capture", "--warpx-exe", "warpx", "--transport-config", "tables",
            "--e0-peak-kv-per-m", "130", "--material-conduction-target", "0.04",
            "--material-reaction-target", "0.04", "--material-diffusion-target", "0.4",
            "--population-controller-rng-salt", "7",
        ],
    )
    args = parse_args()
    overrides = capture_feature_overrides(
        args, video_dir=Path("video"), frame_schedule=Path("frames.csv"))
    assert args.dt_s == pytest.approx(2.5e-9)
    assert "rrea.material_conduction_target=0.040000000000000001" in overrides
    assert "rrea.material_diffusion_target=0.40000000000000002" in overrides
    assert "rrea.material_reaction_target=0.040000000000000001" in overrides
    assert "rrea.material_min_substeps=2" in overrides
    assert "rrea.material_max_substeps=64" in overrides
    assert args.population_controller_rng_salt == 7
    assert "warpx.sort_intervals=-1" in overrides


@pytest.mark.parametrize("campaign", [False, True])
def test_mesh_refinement_keeps_physical_domain_and_video_cadence(monkeypatch, campaign):
    flags = ["--warpx-exe", "warpx", "--e0-peak-kv-per-m", "125"]
    if campaign:
        from cluster.olivia.submit_rrea_cd_campaign import (
            CampaignExecutionInputs, build_cases, case_driver_args)
        flags = case_driver_args(build_cases("e0-scan", accepted_dt_ns=2.5)[0],
            CampaignExecutionInputs(Path.cwd(), Path("warpx"), Path("tables")))
    monkeypatch.setattr("sys.argv", ["capture", *flags,
        "--n-cell-r", "128", "--n-cell-z", "512"])
    args = parse_args()
    assert (args.n_cell_r, args.n_cell_z) == (128, 512)
    from rrea_run_support import CAPTURE_DEFAULTS
    assert args.domain_radius_m == args.video_r_max_m == CAPTURE_DEFAULTS["domain_radius_m"]
    assert args.video_z_max_m == CAPTURE_DEFAULTS["domain_height_m"]
    assert args.video_frame_interval_s == pytest.approx(1e-6)


def _checkpoint(root: Path, step: int, *, complete: bool = True,
                age_s: float = 3600.0) -> Path:
    path = root / f"chk{step:06d}"
    (path / "rrea_checkpoint/Level_0").mkdir(parents=True)
    (path / "WarpXHeader").write_text("header\n")
    if complete:
        (path / "rrea_checkpoint/COMPLETE").write_text("")
    stamp = time.time() - age_s
    for entry in sorted(path.rglob("*"), reverse=True):
        os.utime(entry, (stamp, stamp))
    os.utime(path, (stamp, stamp))
    return path


@pytest.mark.parametrize("field_kv", [0.0, 1.0])
def test_parma_controls_and_checkpoint_namespaces(monkeypatch, tmp_path, field_kv):
    outputs = [tmp_path / "first", tmp_path / "second"]
    for index, output in enumerate(outputs):
        monkeypatch.setattr("sys.argv", [
            "capture", "--warpx-exe", "warpx", "--transport-config", "tables",
            "--seed-source-model", "parma_continuous_column", "--e0-peak-kv-per-m",
            str(field_kv), "--output-dir", str(output), "--checkpoint-interval", "10"])
        args = parse_args()
        assert args.checkpoint_prefix == output / "chk"
        checkpoint = _checkpoint(output, 10 + index)
        assert resolve_restart_checkpoint(args.checkpoint_prefix.parent, "auto") == checkpoint
        assert coleman_dwyer_profile_reference(
            field=load_field_profile(args.field_profile),
            density_ratio=load_density_ratio_profile(args.density_profile),
            threshold_profile=load_threshold_profile(args.density_profile),
            e0_peak_v_per_m=1000 * field_kv) is None
    assert resolve_restart_checkpoint(tmp_path, str(checkpoint)) == checkpoint


def test_checkpoint_rotation_keeps_newest_complete_only(tmp_path: Path) -> None:
    for step in (998000, 999000, 1000000, 1001000):
        _checkpoint(tmp_path, step)
    assert rotate_checkpoints(tmp_path / "chk", 2) == 2
    assert sorted(path.name for path in tmp_path.glob("chk*")) == [
        "chk1000000", "chk1001000"]


def test_combined_pulse_arguments_and_retired_selector(monkeypatch):
    argv = ["capture", "--warpx-exe", "warpx", "--e0-peak-kv-per-m", "125",
            "--seed-source-model", "parma_plus_1mev_pulse", "--macro-count", "17",
            "--injected-real-electrons", "1e9", "--parma-source-radius-m", "1500"]
    monkeypatch.setattr("sys.argv", argv)
    args = parse_args()
    assert args.macro_count == 17 and args.injected_real_electrons == 1e9
    assert args.parma_source_radius_m == 1500 and args.seed_time_window_s == 0
    argv[6] = "production_mono_1mev"
    with pytest.raises(SystemExit):
        parse_args()


def test_checkpoint_rotation_never_deletes_incomplete_or_live(tmp_path: Path) -> None:
    _checkpoint(tmp_path, 40)
    live = _checkpoint(tmp_path, 80, age_s=0.0)
    partial = _checkpoint(tmp_path, 120, complete=False)
    _checkpoint(tmp_path, 160)
    rotate_checkpoints(tmp_path / "chk", 1, min_age_s=300.0)
    assert live.exists() and partial.exists()


def test_checkpoint_rotation_orders_by_step_not_mtime(tmp_path: Path) -> None:
    newest = _checkpoint(tmp_path, 1000)
    older = _checkpoint(tmp_path, 200)
    os.utime(older, (newest.stat().st_mtime + 100, newest.stat().st_mtime + 100))
    rotate_checkpoints(tmp_path / "chk", 1)
    assert newest.exists() and not older.exists()


def test_safe_output_requires_owned_marker_and_opt_in(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    output = prepare_fresh_output_directory(root / "case", run_root=root, overwrite=False)
    (output / "result").write_text("owned")
    with pytest.raises(SystemExit, match="overwrite-output"):
        prepare_fresh_output_directory(output, run_root=root, overwrite=False)
    prepare_fresh_output_directory(output, run_root=root, overwrite=True)
    unowned = root / "unowned"
    unowned.mkdir()
    (unowned / "user-data").write_text("keep")
    with pytest.raises(SystemExit, match="unowned"):
        prepare_fresh_output_directory(unowned, run_root=root, overwrite=True)


def test_protected_physics_overrides_are_rejected_as_one_family() -> None:
    # Physics rrea.* keys are refused; execution controls have an allowlist.
    # A user override cannot appear
    # to disable or rescale physics; feedback and positron transport are
    # unconditional in the table loader, not in the deck parser.
    for token in ("rrea.field_model=x", "amr.n_cell=1 1", "geometry.prob_hi=1 1",
                  "boundary.particle_hi=periodic periodic", "algo.particle_shape=3",
                  "rrea.debug_disable_transport=1",
                  "rrea.debug_scale_hard_moller_mfp=1",
                  "rrea.debug_transport_sensitivity_enable=1",
                  "rrea.debug_actual_continuity_fixture=1"):
        with pytest.raises(SystemExit, match="protected production key"):
            validate_user_warpx_overrides([token])
    assert validate_user_warpx_overrides(["warpx.verbose=2"]) == ["warpx.verbose=2"]
    diagnostic = ["rrea.electron_creation_bin_m=25"]
    assert validate_user_warpx_overrides(diagnostic) == diagnostic
    with pytest.raises(SystemExit, match="one number"):
        validate_user_warpx_overrides(["rrea.electron_creation_bin_m=25 rrea.low_energy_cutoff_eV=0"])


def test_cd_spectrum_is_bounded_stratified_and_permuted() -> None:
    first = conditioned_exponential_seed_energies(10_000, realization_id=0)
    second = conditioned_exponential_seed_energies(10_000, realization_id=1)
    assert 1.0e3 <= min(first) <= max(first) <= 1.0e10
    assert first != second and sorted(first) == sorted(second)
    assert math.isclose(sum(first) / len(first), 7.2e6, rel_tol=2.0e-3)


def test_cd_schedule_is_fixed_z_unit_weight(tmp_path: Path) -> None:
    path = tmp_path / "seeds.csv"
    energies = conditioned_exponential_seed_energies(8, realization_id=3)
    write_seed_schedule(
        path, case_id="cd", injection_z_m=123.0, macro_count=8,
        total_weight_real_electrons=8.0, source_radius_m=17.0,
        kinetic_energies_eV=energies, fixed_axial_position=True)
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert {float(row["z_m"]) for row in rows} == {123.0}
    assert {float(row["weight_real_electrons"]) for row in rows} == {1.0}
    assert [float(row["kinetic_energy_eV"]) for row in rows] == energies


def test_launcher_runs_every_rank_at_the_requested_priority() -> None:
    """nice goes in front of the launcher, so all ranks inherit one value.

    Renicing afterwards can skew ranks against each other inside a lockstep
    field solve.
    """
    assert launcher_prefix("mpiexec", 12, 1, 19) == [
        "nice", "-n", "19", "mpiexec", "-n", "12"]
    assert launcher_prefix("mpiexec", 12, 1) == ["mpiexec", "-n", "12"]
    assert launcher_prefix("none", 1, 1, 19) == ["nice", "-n", "19"]
    for bad in (-1, 20):
        with pytest.raises(SystemExit, match=r"\[0, 19\]"):
            launcher_prefix("mpiexec", 12, 1, bad)


def test_cd_planes_are_regular_between_inset_endpoints() -> None:
    reference = coleman_dwyer_profile_reference(
        field=load_field_profile(), density_ratio=load_density_ratio_profile(),
        threshold_profile=load_threshold_profile(), e0_peak_v_per_m=110_000.0)
    assert reference is not None
    planes = build_cd_plane_z_m(reference, 1000)
    assert len(planes) >= 3
    interior = [b - a for a, b in zip(planes[1:-2], planes[2:-1])]
    assert all(math.isclose(spacing, 50.0, abs_tol=1e-8) for spacing in interior)


def test_max_step_reaches_stop_time() -> None:
    assert resolve_max_step(None, 20e-6, 2.5e-9) == 8000
    with pytest.raises(SystemExit, match="cannot reach"):
        resolve_max_step(7999, 20e-6, 2.5e-9)


def _plane_row(write: int, step: int, plane: int, altitude: float) -> list[object]:
    return [1, write, step, step * 1e-9, plane, altitude,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 0]


def test_restart_trim_keeps_the_checkpointed_blocks(tmp_path: Path) -> None:
    path = tmp_path / "flux.csv"
    altitudes = [10_000.0, 10_050.0]
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["format_version", "write_index", "step", "time_s",
                         "plane_index", "altitude_m", *range(13)])
        for write, step in enumerate((10, 20)):
            for plane, altitude in enumerate(altitudes):
                writer.writerow(_plane_row(write, step, plane, altitude))
        writer.writerow(_plane_row(2, 30, 0, altitudes[0]))
    _, kept, dropped = plan_cd_plane_flux_trim(
        path, checkpoint_write_count=2, plane_count=len(altitudes))
    assert len(kept) == 4 and dropped == 1
    with pytest.raises(SystemExit):
        plan_cd_plane_flux_trim(path, checkpoint_write_count=3, plane_count=len(altitudes))


def test_frame_schedule_extension_is_append_only(tmp_path: Path) -> None:
    base, render_map = build_frame_schedule(
        dt_s=2.5e-9, frame_interval_s=1.19e-7, final_time_s=200e-6)
    assert len(render_map["frames"]) == 1681 and base[0]["step"] == 1
    previous = base
    for final_time in (250e-6, 275e-6, 500e-6, 500e-6):
        grown, _ = build_frame_schedule(
            dt_s=2.5e-9, frame_interval_s=1.19e-7, final_time_s=final_time)
        assert grown[:len(previous)] == previous
        previous = grown
    assert grown[-1]["step"] == round(4201 * 1.19e-7 / 2.5e-9)
    old, new = tmp_path / "old.csv", tmp_path / "new.csv"
    write_capture_schedule(old, base)
    write_capture_schedule(new, grown)
    assert require_compatible_schedule(new, old, "frame schedule", allow_append=True)


def test_restart_changes_only_future_snapshot_spacing(tmp_path: Path) -> None:
    base, _ = build_frame_schedule(
        dt_s=2.5e-9, frame_interval_s=1.19e-7, final_time_s=500e-6)
    old = tmp_path / "old.csv"
    write_capture_schedule(old, base)
    with old.open() as handle:
        retained = list(csv.DictReader(handle))
    grown, render_map = build_frame_schedule(
        dt_s=2.5e-9, frame_interval_s=1e-6, final_time_s=5e-3,
        retained_rows=retained,
        append_after_step=200000)
    new = tmp_path / "new.csv"
    write_capture_schedule(new, grown)
    assert require_compatible_schedule(new, old, "frame schedule", allow_append=True)
    assert len(render_map["frames"]) == 5001
    assert all(math.isclose(f["target_time_s"], i * 1e-6)
               for i, f in enumerate(render_map["frames"]))
    assert render_map["frames"][500]["step"] == base[-1]["step"]
    assert render_map["frames"][500]["captured_time_s"] < 500e-6
    assert len(grown) == len(base) + 4500
    assert grown[len(base)]["step"] == 200400
    assert grown[-1]["step"] == 2000000
    assert all(b["step"] - a["step"] == 400
               for a, b in zip(grown[len(base):], grown[len(base) + 1:]))
    repeated, repeated_map = build_frame_schedule(
        dt_s=2.5e-9, frame_interval_s=1e-6, final_time_s=5e-3,
        retained_rows=grown,
        append_after_step=300000)
    assert repeated == grown and repeated_map == render_map
    with pytest.raises(SystemExit):
        build_frame_schedule(dt_s=2.5e-9, frame_interval_s=1e-6, final_time_s=400e-6,
                             retained_rows=retained)


def test_olivia_totp_uses_public_rfc_hotp_vector() -> None:
    path = Path(__file__).with_name("olivia_askpass.py")
    spec = importlib.util.spec_from_file_location("olivia_askpass_tested", path)
    assert spec is not None and spec.loader is not None
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    assert helper._hotp(b"12345678901234567890", 1, 8, hashlib.sha1) == "94287082"


def test_video_v4_decoder_inverts_the_engine_codec() -> None:
    """The reader inverts the engine's shuffle-then-deflate bit for bit.

    A frame that does not decode exactly would silently corrupt every panel
    drawn from it, so this is pinned rather than assumed.
    """
    import zlib

    import numpy as np

    from rrea_table_pack import byte_unshuffle
    from rrea_video_stream import decode_v4_frame

    rng = np.random.default_rng(20260824)
    # Spans the real payload's character: exact zeros, a huge dynamic range,
    # and a smooth field-like array.
    values = np.concatenate([
        np.zeros(4096, dtype="<f4"),
        (rng.random(4096) * 1e15).astype("<f4"),
        np.linspace(0.0, 180.0, 4096, dtype="<f4"),
    ])
    payload = values.tobytes()
    # The engine's layout (RreaVideoFrameCodec.H): out[j*N + i] = in[i*S + j].
    assert byte_unshuffle(bytes([0, 4, 1, 5, 2, 6, 3, 7]), 4) == bytes(range(8))
    shuffled = np.frombuffer(payload, dtype=np.uint8).reshape(-1, 4).T.tobytes()
    blob = zlib.compress(shuffled, 1)
    assert decode_v4_frame(blob, expected_bytes=len(payload)) == payload
    with pytest.raises(SystemExit, match="expected"):
        decode_v4_frame(blob, expected_bytes=len(payload) + 4)
    with pytest.raises(zlib.error):
        decode_v4_frame(blob[:-1], expected_bytes=len(payload))


def test_video_v4_trim_truncates_on_a_frame_boundary(tmp_path: Path) -> None:
    """Restart must cut the variable-length frame stream exactly at frame N.

    The trim reads the last kept frame's own recorded extent; getting that
    wrong would leave a partial frame that desynchronizes every byte written
    after the restart.
    """
    from run_rrea_profiled_video_capture import frame_stream_end_bytes

    header = ["write_index", "step", "byte_offset", "byte_length"]
    rows = [
        ["0", "10", "0", "700"],
        ["1", "20", "700", "512"],
        ["2", "30", "1212", "913"],
    ]
    assert frame_stream_end_bytes(header, rows, context="t") == 1212 + 913
    # Trimming to the first two frames lands on frame 1's end -- which is not
    # any multiple of a fixed stride.
    assert frame_stream_end_bytes(header, rows[:2], context="t") == 700 + 512
    assert frame_stream_end_bytes(header, [], context="t") == 0

    with pytest.raises(SystemExit):
        frame_stream_end_bytes(["write_index", "step"], rows, context="t")
