#!/usr/bin/env python3
"""Small shared helpers for the one profiled RREA production driver.

It centralizes defaults, shared profiled-atmosphere overrides, protected user
overrides, fresh-output handling, provenance, and checkpoint rotation for the
production driver.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Iterable

from rrea_profiled_atmosphere import ALTITUDE_AT_Z0_M, DOMAIN_HEIGHT_M, DOMAIN_RADIUS_M


ROOT = Path(__file__).resolve().parents[1]
DEFAULTS_PATH = ROOT / "config/rrea_defaults.json"
DEFAULT_BASE_INPUT = ROOT / "cases/rrea/production/inputs"

RUN_DIRECTORY_MARKER = ".rrea_run_directory.json"

# SI constants (CODATA 2018) for every RREA analysis script; one owner.
C = 299792458.0
MU0 = 1.25663706212e-6
EPS0 = 8.8541878128e-12
ELEMENTARY_CHARGE = 1.602176634e-19
K_BOLTZ = 1.380649e-23
# The video capture grid always spans the full domain; this is the render-time
# default display crop (0.4 x domain covers the strong-field region inside the
# taper).
VIDEO_DISPLAY_DOMAIN_FRACTION = 0.4


def required_time_steps(stop_time_s: float, dt_s: float) -> int:
    """Ceiling division without adding a step for a roundoff-level integer."""
    if (
        not math.isfinite(stop_time_s)
        or not math.isfinite(dt_s)
        or stop_time_s <= 0.0
        or dt_s <= 0.0
    ):
        raise ValueError("stop_time_s and dt_s must be finite and positive")
    ratio = stop_time_s / dt_s
    nearest = round(ratio)
    if math.isclose(ratio, nearest, rel_tol=8.0 * math.ulp(1.0), abs_tol=0.0):
        return int(nearest)
    return int(math.ceil(ratio))


def load_rrea_defaults(path: Path = DEFAULTS_PATH) -> dict[str, Any]:
    try:
        defaults = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SystemExit(f"cannot read single-source RREA defaults {path}: {exc}") from exc
    for key in (
        "population_target_electron_macros",
        "population_control_interval",
        "population_controller_rng_salt",
        "seed_macro_count",
    ):
        value = defaults.get(key)
        if not isinstance(value, int) or value < 0:
            raise SystemExit(f"{path}: {key!r} must be a non-negative integer (got {value!r})")
    if defaults["seed_macro_count"] <= 0:
        raise SystemExit(f"{path}: seed_macro_count must be positive")
    factors = defaults.get("population_ceiling_factors")
    if not isinstance(factors, dict):
        raise SystemExit(f"{path}: population_ceiling_factors must be an object")
    for species in ("electron", "photon", "positron"):
        value = factors.get(species)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value <= 0:
            raise SystemExit(
                f"{path}: population_ceiling_factors.{species} must be positive "
                f"(got {value!r})"
            )
    seed_source_model = defaults.get("seed_source_model")
    if seed_source_model not in ("parma_plus_1mev_pulse", "parma_continuous_column"):
        raise SystemExit(
            f"{path}: seed_source_model must be 'parma_plus_1mev_pulse' or "
            "'parma_continuous_column' (the Coleman-Dwyer campaign model is "
            "selected explicitly per run, never as a standing default)"
        )
    parma = defaults.get("parma_continuous_source")
    if not isinstance(parma, dict):
        raise SystemExit(f"{path}: parma_continuous_source must be an object")
    for key, low, high in (
        ("latitude_deg", -90.0, 90.0),
        ("longitude_deg", -180.0, 180.0),
    ):
        value = parma.get(key)
        if not isinstance(value, (int, float)) or not low <= float(value) <= high:
            raise SystemExit(
                f"{path}: parma_continuous_source.{key} must lie in "
                f"[{low:g}, {high:g}] (got {value!r})"
            )
    minimum_energy = parma.get("minimum_energy_eV")
    if (
        not isinstance(minimum_energy, (int, float))
        or not 1.0e5 <= float(minimum_energy) < 1.0e10
    ):
        raise SystemExit(
            f"{path}: parma_continuous_source.minimum_energy_eV must lie in "
            "[1e5, 1e10) eV; PARMA's electron/positron angular law is only "
            f"parametrized down to 100 keV (got {minimum_energy!r})"
        )
    source_radius = parma.get("source_radius_m")
    if (
        not isinstance(source_radius, (int, float))
        or not math.isfinite(float(source_radius))
        or float(source_radius) <= 0.0
    ):
        raise SystemExit(
            f"{path}: parma_continuous_source.source_radius_m must be finite "
            f"and positive (got {source_radius!r})"
        )
    parma_macro_count = parma.get("macro_count")
    if not isinstance(parma_macro_count, int) or parma_macro_count <= 0:
        raise SystemExit(
            f"{path}: parma_continuous_source.macro_count must be a positive "
            f"integer (got {parma_macro_count!r})"
        )
    capture = defaults.get("capture_defaults")
    if not isinstance(capture, dict):
        raise SystemExit(f"{path}: capture_defaults must be an object")
    for key in ("altitude_at_z0_m", "cell_size_m", "video_frame_interval_s"):
        value = capture.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value <= 0:
            raise SystemExit(f"{path}: capture_defaults.{key} must be finite and positive")
    taper_fraction = capture.get("field_taper_start_fraction")
    if (not isinstance(taper_fraction, (int, float))
            or not math.isfinite(float(taper_fraction))
            or not 0.0 < float(taper_fraction) < 1.0):
        raise SystemExit(
            f"{path}: capture_defaults.field_taper_start_fraction must lie in (0,1)")
    return defaults


RREA_DEFAULTS = load_rrea_defaults()
DEFAULT_POPULATION_TARGET_ELECTRON_MACROS = int(
    RREA_DEFAULTS["population_target_electron_macros"]
)
DEFAULT_POPULATION_CONTROL_INTERVAL = int(RREA_DEFAULTS["population_control_interval"])
DEFAULT_POPULATION_CONTROLLER_RNG_SALT = int(
    RREA_DEFAULTS["population_controller_rng_salt"]
)
DEFAULT_SEED_MACRO_COUNT = int(RREA_DEFAULTS["seed_macro_count"])
DEFAULT_SEED_SOURCE_MODEL = str(RREA_DEFAULTS["seed_source_model"])
PARMA_CONTINUOUS_SOURCE_DEFAULTS = RREA_DEFAULTS["parma_continuous_source"]
POPULATION_CEILING_FACTORS = {
    species: float(RREA_DEFAULTS["population_ceiling_factors"][species])
    for species in ("electron", "photon", "positron")
}
CAPTURE_DEFAULTS = RREA_DEFAULTS["capture_defaults"]
CELL_SIZE_M = float(CAPTURE_DEFAULTS["cell_size_m"])
TAPER_START_DOMAIN_FRACTION = float(
    CAPTURE_DEFAULTS["field_taper_start_fraction"])

_SENSITIVE_ARGUMENT_FRAGMENTS = (
    "password",
    "secret",
    "token",
    "credential",
    "api_key",
    "apikey",
)


def _is_sensitive_name(name: str) -> bool:
    lowered = name.lower()
    return any(fragment in lowered for fragment in _SENSITIVE_ARGUMENT_FRAGMENTS)


def redact_command_for_provenance(command: Iterable[str]) -> list[str]:
    """Redact secret-shaped key=value tokens without changing launch argv."""
    redacted = []
    for raw in command:
        token = str(raw)
        if "=" in token:
            name, _ = token.split("=", maxsplit=1)
            if _is_sensitive_name(name.lstrip("-")):
                token = f"{name}=<redacted>"
        redacted.append(token)
    return redacted


def resolved_driver_arguments(args: Any) -> dict[str, Any]:
    """Return a JSON-safe, path-resolved, secret-filtered argparse snapshot."""

    def convert(value: Any, key_hint: str = "") -> Any:
        if isinstance(value, Path):
            return str(value.expanduser().resolve())
        if value is None or isinstance(value, (bool, int, float)):
            return value
        if isinstance(value, str):
            if "=" in value:
                name, payload = value.split("=", maxsplit=1)
                if _is_sensitive_name(name):
                    return f"{name}=<redacted>"
                return f"{name}={payload}"
            return value
        if isinstance(value, (list, tuple)):
            return [convert(item, key_hint) for item in value]
        if isinstance(value, dict):
            return {
                str(key): convert(item, str(key))
                for key, item in value.items()
                if not _is_sensitive_name(str(key))
            }
        raise TypeError(
            f"driver argument {key_hint!r} has unsupported provenance type "
            f"{type(value).__name__}"
        )

    values = vars(args)
    return {
        key: convert(value, key)
        for key, value in sorted(values.items())
        if not _is_sensitive_name(key)
    }


def population_overrides(
    *,
    population_target: float,
    control_interval: int,
    split_min_weight: float,
    strict_ceilings: dict[str, int] | None,
) -> list[str]:
    if population_target <= 0.0:
        return [
            "rrea.population_ceiling_policy=strict_abort_v1",
            "rrea.population_target_electron_macros=0",
            *(f"rrea.max_{species}_macros={value}"
              for species, value in (strict_ceilings or {}).items()),
        ]
    target = int(round(population_target))
    return [
        "rrea.population_ceiling_policy=adaptive_resample_v1",
        f"rrea.population_target_electron_macros={target}",
        f"rrea.population_control_interval={control_interval}",
        f"rrea.population_split_min_weight={split_min_weight:.17g}",
        f"rrea.max_electron_macros={int(round(POPULATION_CEILING_FACTORS['electron'] * target))}",
        f"rrea.max_photon_macros={int(round(POPULATION_CEILING_FACTORS['photon'] * target))}",
        f"rrea.max_positron_macros={int(round(POPULATION_CEILING_FACTORS['positron'] * target))}",
    ]


def profiled_overrides(
    *,
    args: Any,
    case_id: str,
    run_dir: Path,
    schedule: Path,
    e0_v_per_m: float,
    injected_real_particles: float,
    region: Any,
    stop_time_s: float,
    max_step: int,
    transport_config: Path,
    rng_seed: int | None,
    population_target: float,
    population_control_interval: int,
    population_controller_rng_salt: int,
    population_split_min_weight: float | None,
    population_ceilings: dict[str, int] | None,
) -> list[str]:
    """Build the shared neutral profiled-atmosphere engine configuration.

    Source scheduling, capture diagnostics, and any interpretation of the
    resulting observables stay in their callers.  This function owns only the
    realistic profile/mesh/time/population controls.
    """
    profile_bounds = []
    if region is not None:
        active_min = max(0.0, region.bottom_altitude_m - ALTITUDE_AT_Z0_M)
        active_max = min(DOMAIN_HEIGHT_M, region.top_altitude_m - ALTITUDE_AT_Z0_M)
        profile_bounds = [
            f"rrea.profile_active_z_min_m={active_min:.17g}",
            f"rrea.profile_active_z_max_m={active_max:.17g}",
        ]
    split_min_weight = (
        population_split_min_weight
        if population_split_min_weight is not None
        else injected_real_particles / args.macro_count
    )
    return [
        f"rrea.case_id={case_id}",
        f"rrea.seed_schedule_path={schedule}",
        "rrea.initial_field_mode=altitude_profile",
        f"rrea.field_profile_path={args.field_profile}",
        f"rrea.profile_altitude_msl_at_z0_m={ALTITUDE_AT_Z0_M:.17g}",
        f"rrea.profile_phi_integration_step_m={CELL_SIZE_M:.17g}",
        f"rrea.profile_e0_peak_v_per_m={e0_v_per_m:.17g}",
        f"rrea.acceleration_field_v_per_m={-abs(e0_v_per_m):.17g}",
        f"rrea.acceleration_z_max_m={DOMAIN_HEIGHT_M:.17g}",
        "rrea.air_density_model=altitude_profile",
        f"rrea.density_profile_path={args.density_profile}",
        f"rrea.interaction_table_config={transport_config}",
        *population_overrides(
            population_target=population_target,
            control_interval=population_control_interval,
            split_min_weight=split_min_weight,
            strict_ceilings=population_ceilings,
        ),
        f"rrea.population_controller_rng_salt={population_controller_rng_salt}",
        *([f"rrea.rng_seed={rng_seed}"] if rng_seed is not None else []),
        *profile_bounds,
        f"rrea.diag_interval={args.diag_interval}",
        "rrea.mesh_diag_interval=0",
        f"rrea.output_dir={run_dir}",
        "diag0.fields_to_plot=none",
        "diag0.write_species=0",
        "diag0.dump_last_timestep=0",
        "diag0.intervals=1000000000",
        f"diag0.file_prefix={run_dir}/diags/diag0",
        "diag1.fields_to_plot=none",
        "diag1.write_species=0",
        "diag1.dump_last_timestep=0",
        "diag1.intervals=1000000000",
        f"diag1.file_prefix={run_dir}/diags/diag1",
        f"geometry.prob_hi={DOMAIN_RADIUS_M:.17g} {DOMAIN_HEIGHT_M:.17g}",
        f"amr.n_cell={args.n_cell_r} {args.n_cell_z}",
        f"max_step={max_step}",
        f"stop_time={stop_time_s:.17g}",
        f"warpx.const_dt={args.dt_s:.17g}",
        "my_constants.seed_macro_particle_weight="
        f"{injected_real_particles / args.macro_count:.17g}",
    ]


_PROTECTED_EXACT = {
    "max_step",
    "stop_time",
    "warpx.const_dt",
}
_PROTECTED_PREFIXES = (
    "warpx.",
    "rrea.",
    "amr.",
    "geometry.",
    "boundary.",
    "algo.",
    "particles.",
    "species.",
    "rrea_electrons.",
    "rrea_photons.",
    "rrea_positrons.",
    "diagnostics.",
    "diag",
    "chk.",
    "my_constants.",
)
# Execution-only controls for the experimental CUDA backend (the engine owns
# their ranges). Physics/model, cutoffs, geometry and population policy remain
# protected below.
_GPU_RUNTIME_INTEGERS = {
    "rrea.gpu_kinetic_batch_size",
    "rrea.gpu_kinetic_heap_mib",
    "rrea.gpu_population_batch_size",
    "rrea.gpu_population_cpu_fallback",
    "rrea.gpu_maxwell",
    "rrea.gpu_fluid",
}
_GPU_RUNTIME_ENUMS = {"rrea.kinetic_backend": {"cpu", "gpu"}}
_ALLOWED_USER_OVERRIDE_EXACT = {
    "warpx.verbose", "rrea.electron_creation_bin_m", *_GPU_RUNTIME_INTEGERS, *_GPU_RUNTIME_ENUMS,
}


def _validate_runtime_override(key: str, value: str) -> None:
    """One token only: no bare ParmParse continuation or nested override."""
    if key in _GPU_RUNTIME_ENUMS and value not in _GPU_RUNTIME_ENUMS[key]:
        raise SystemExit(f"{key} must be cpu or gpu; got {value!r}")
    if key in _GPU_RUNTIME_INTEGERS and not (value.isascii() and value.isdecimal()):
        raise SystemExit(f"{key} must be one non-negative integer; got {value!r}")
    if key == "rrea.electron_creation_bin_m":
        try:
            float(value)  # One numeric token; the engine owns its physical range.
        except ValueError:
            raise SystemExit(f"{key} must be one number; got {value!r}") from None


def validate_user_warpx_overrides(values: Iterable[str]) -> list[str]:
    """Validate only user-originated overrides.

    Internal builders may still use bare ParmParse list continuation tokens;
    accepting those from an unstructured user string would make protection of
    the preceding key impossible.
    """
    result: list[str] = []
    for value in values:
        if "=" not in value:
            raise SystemExit(
                f"--extra-warpx-arg must be one complete key=value token; got {value!r}"
            )
        key = value.split("=", 1)[0].strip()
        if not key:
            raise SystemExit(f"--extra-warpx-arg has an empty key: {value!r}")
        if _is_sensitive_name(key):
            raise SystemExit(
                f"--extra-warpx-arg may not contain sensitive key {key!r}; "
                "secrets must never be passed through or recorded by this driver"
            )
        if key not in _ALLOWED_USER_OVERRIDE_EXACT and (
            key in _PROTECTED_EXACT
            or any(key.startswith(prefix) for prefix in _PROTECTED_PREFIXES)
        ):
            raise SystemExit(
                f"--extra-warpx-arg may not override protected production key {key!r}; "
                "use its typed driver option"
            )
        _validate_runtime_override(key, value.split("=", 1)[1].strip())
        result.append(value)
    return result


def validate_fresh_output_paths(
    output_dir: Path, run_root: Path | None
) -> tuple[Path, Path]:
    """A fresh run needs a run root, with its output strictly inside it."""
    if run_root is None:
        raise SystemExit("a fresh run requires an explicit --run-root")
    raw_root = Path(os.path.abspath(run_root.expanduser()))
    raw_output = Path(os.path.abspath(output_dir.expanduser()))
    if raw_root not in raw_output.parents:
        raise SystemExit(
            f"--output-dir {raw_output} must be strictly below --run-root {raw_root}"
        )
    return raw_root, raw_output


def prepare_fresh_output_directory(
    output_dir: Path, *, run_root: Path | None, overwrite: bool
) -> Path:
    raw_root, raw_output = validate_fresh_output_paths(output_dir, run_root)
    root = raw_root.resolve()
    output = raw_output.resolve()
    if output.exists() and any(output.iterdir()):
        marker = output / RUN_DIRECTORY_MARKER
        if not overwrite:
            raise SystemExit(
                f"refusing to replace non-empty output {output}; pass --overwrite-output"
            )
        if not marker.is_file():
            raise SystemExit(
                f"refusing to replace unowned output {output}; missing "
                f"{RUN_DIRECTORY_MARKER}"
            )
        try:
            marker_payload = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SystemExit(f"invalid RREA ownership marker {marker}: {exc}") from exc
        if marker_payload.get("format") != "rrea_run_directory_v1":
            raise SystemExit(f"invalid RREA ownership marker format in {marker}")
        if marker_payload.get("output_dir") != str(output) or marker_payload.get(
            "run_root"
        ) != str(root):
            raise SystemExit(
                f"ownership marker {marker} does not match the resolved output/run root"
            )
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    marker_payload = {
        "format": "rrea_run_directory_v1",
        "output_dir": str(output),
        "run_root": str(root),
    }
    (output / RUN_DIRECTORY_MARKER).write_text(
        json.dumps(marker_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output


def _git_identity() -> dict[str, Any]:
    def query(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", *args], cwd=ROOT, text=True, capture_output=True, check=False
            )
        except OSError:
            return None
        return result.stdout.strip() if result.returncode == 0 else None

    commit = query("rev-parse", "HEAD")
    status = query("status", "--porcelain", "--untracked-files=no")
    identity_source = "git"
    if commit is None:
        marker = ROOT / "CODEX_SOURCE_COMMIT"
        try:
            marked_commit = marker.read_text(encoding="ascii").strip()
        except (OSError, UnicodeError):
            marked_commit = ""
        if re.fullmatch(r"[0-9a-fA-F]{40}", marked_commit):
            commit = marked_commit.lower()
            identity_source = "git_archive_marker"
    return {
        "commit": commit,
        "branch": query("branch", "--show-current"),
        "tracked_worktree_clean": status == "" if status is not None else None,
        "identity_source": identity_source if commit is not None else None,
    }


def write_prelaunch_provenance(
    run_dir: Path,
    command: list[str],
    *,
    metadata: dict[str, Any] | None = None,
    cwd: Path = ROOT,
    launch_environment: dict[str, str] | None = None,
) -> Path:
    """Write an append-only attempt record before the external process starts."""
    attempts = run_dir / "command_attempts"
    attempts.mkdir(parents=True, exist_ok=True)
    ordinal = 1
    while (attempts / f"attempt_{ordinal:04d}.json").exists():
        ordinal += 1
    payload = {
        "format": "rrea_prelaunch_command_v2",
        "attempt": ordinal,
        "command": redact_command_for_provenance(command),
        "cwd": str(cwd.resolve()),
        "git": _git_identity(),
        "environment": {
            key: (launch_environment or os.environ)[key]
            for key in (
                "SLURM_JOB_ID",
                "SLURM_NTASKS",
                "SLURM_CPUS_PER_TASK",
                "OMP_NUM_THREADS",
                "OMP_DYNAMIC",
                "OMP_PLACES",
                "OMP_PROC_BIND",
            )
            if key in (launch_environment or os.environ)
        },
        "metadata": metadata or {},
    }
    attempt_path = attempts / f"attempt_{ordinal:04d}.json"
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    attempt_path.write_text(encoded, encoding="utf-8")
    primary = run_dir / "command.json"
    if not primary.exists():
        primary.write_text(encoded, encoding="utf-8")
    return attempt_path


CHECKPOINT_ROTATE_POLL_S = 60.0
# A checkpoint must be untouched this long before it may be deleted, so that
# rotation cannot race the writer without retaining a whole cadence history.
CHECKPOINT_ROTATE_MIN_AGE_S = 60.0


def rotate_checkpoints(
    prefix: Path, keep: int, min_age_s: float = CHECKPOINT_ROTATE_MIN_AGE_S
) -> int:
    """Keep the newest `keep` COMPLETE, quiet checkpoints; return removals.

    The one rotation owner, for local and cluster runs alike. `min_age_s`
    keeps an active writer out of reach even after the RREA completion marker
    appears.
    """
    if keep < 1:
        raise ValueError("keep must be >= 1: refusing to delete every checkpoint")
    parent, stem = prefix.parent, prefix.name
    if not parent.is_dir():
        return 0
    cutoff = time.time() - min_age_s

    def quiet(path: Path) -> bool:
        """True when nothing inside has been touched since `cutoff`."""
        try:
            return all(e.stat().st_mtime <= cutoff for e in path.rglob("*"))
        except OSError:
            return False  # vanished or churning: leave it alone

    def settled(path: Path) -> bool:
        """The engine's COMPLETE marker plus no recent writes."""
        return (path / "rrea_checkpoint" / "COMPLETE").is_file() and quiet(path)

    removed = 0
    # `.old.*` are abandoned renames, so age alone settles them.
    for stale in sorted(parent.glob(f"{stem}*.old.*")):
        if stale.is_dir() and quiet(stale):
            shutil.rmtree(stale, ignore_errors=True)
            removed += 1
    # Only SETTLED checkpoints are counted or deleted: one still being written
    # neither occupies a keep slot nor becomes a deletion candidate.
    ready = sorted(
        (
            path
            for path in parent.glob(f"{stem}[0-9]*")
            if path.is_dir() and path.name[len(stem):].isdigit() and settled(path)
        ),
        key=lambda path: int(path.name[len(stem):]),
    )
    for old in ready[: max(0, len(ready) - keep)]:
        shutil.rmtree(old, ignore_errors=True)
        removed += 1
    return removed


def run_command(
    command: list[str],
    run_dir: Path,
    cpus_per_rank: int,
    *,
    provenance: dict[str, Any] | None = None,
    completion_artifacts_to_invalidate: Iterable[str] = (),
    checkpoint_rotation: tuple[Path, int] | None = None,
) -> None:
    env = os.environ.copy()
    if cpus_per_rank > 0:
        env["OMP_NUM_THREADS"] = str(cpus_per_rank)
    attempt_path = write_prelaunch_provenance(
        run_dir,
        command,
        metadata=provenance,
        launch_environment=env,
    )
    # A killed continuation must never inherit a previous successful exit or
    # finalization marker.  Invalidate completion state only after the new
    # attempt record exists, and identify the active attempt before
    # launching the external process.
    for relative_name in ("command_exit.json", *completion_artifacts_to_invalidate):
        artifact = run_dir / relative_name
        if artifact.exists():
            artifact.unlink()
    in_progress_path = run_dir / "command_in_progress.json"
    attempt_reference = attempt_path.relative_to(run_dir).as_posix()
    in_progress_path.write_text(
        json.dumps(
            {
                "format": "rrea_command_in_progress_v1",
                "attempt_record": attempt_reference,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    # The engine writes checkpoints itself while this subprocess runs, so the
    # rotation has to poll alongside it rather than run before or after.
    rotate_stop = threading.Event()
    rotate_thread: threading.Thread | None = None
    if checkpoint_rotation is not None:
        rotate_prefix, rotate_keep = checkpoint_rotation

        def _rotate_loop() -> None:
            while not rotate_stop.wait(CHECKPOINT_ROTATE_POLL_S):
                try:
                    rotate_checkpoints(rotate_prefix, rotate_keep)
                except OSError:
                    pass  # a half-written checkpoint is retried next tick

        rotate_thread = threading.Thread(
            target=_rotate_loop, name="checkpoint-rotate", daemon=True
        )
        rotate_thread.start()
    try:
        with (run_dir / "stdout.txt").open("a", encoding="utf-8") as stdout, (
            run_dir / "stderr.txt"
        ).open("a", encoding="utf-8") as stderr:
            result = subprocess.run(
                command,
                cwd=ROOT,
                env=env,
                stdout=stdout,
                stderr=stderr,
                text=True,
                check=False,
            )
    finally:
        rotate_stop.set()
        if rotate_thread is not None:
            rotate_thread.join(timeout=30.0)
        if checkpoint_rotation is not None:
            rotate_checkpoints(*checkpoint_rotation)
    exit_payload = {
        "format": "rrea_command_exit_v1",
        "attempt_record": attempt_reference,
        "returncode": result.returncode,
    }
    encoded_exit = json.dumps(exit_payload, indent=2, sort_keys=True) + "\n"
    exit_path = attempt_path.with_name(
        attempt_path.name.replace("attempt_", "exit_", 1)
    )
    exit_path.write_text(encoded_exit, encoding="utf-8")
    (run_dir / "command_exit.json").write_text(encoded_exit, encoding="utf-8")
    in_progress_path.unlink(missing_ok=True)
    if result.returncode != 0:
        raise SystemExit(f"WarpX failed with exit {result.returncode}; see {run_dir}")
