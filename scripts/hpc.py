"""The one helper for remote HPC work over the persistent mux socket.

Every session task -- running commands, checking Slurm, moving files, building
the engine -- goes through this tool; do not hand-write per-task ssh/scp
scripts. Remote calls use the persistent ControlMaster socket. If it drops,
the helper asks the opener to rebuild it in automated-TOTP mode from private
local state; this module never reads or prints credentials. Small push/pull
streams use gzip; whole-run restart transfers use resumable rsync.

Site identity is deliberately generic: login, remote work root, Slurm account
and module stack live in private local state (credentials.env or HPC_*
environment variables), never in the repository, so this helper works with any
Slurm HPC system.

Profiled captures use `script cluster/hpc/submit_profiled_video.sh
KEY=VALUE...`; C&D cases use their campaign planner. Commands live in
docs/HPC_COMMAND_WORKFLOW_NOTES.md. Run `quota` first.

`build` transfers committed HEAD (never the working tree) and submits the
configured WarpX/AMReX build.
"""

from __future__ import annotations

import argparse
import configparser
import gzip
import io
import os
import shlex
import subprocess
import sys
import tarfile
import time
from pathlib import Path

SOCKET = "/tmp/rrea_hpc_mux"
# One owner for the option set.  ConnectTimeout matches hpc_open_socket.sh's
# own probe: a master that is alive but black-holed must fail, not hang, or
# ensure_socket never reaches its reopen path.
SSH_OPTS = ["-S", SOCKET, "-o", "ProxyCommand=false",
            "-o", "BatchMode=yes", "-o", "ConnectTimeout=10"]
RSH = "ssh " + " ".join(SSH_OPTS)

ROOT = Path(__file__).resolve().parent.parent
TABLE_BUNDLE = "rrea_transport_tables/schema6/production"
WARPX_SUBMODULE = "upstream/WarpX-26.06"
AMREX_SUBMODULE = "upstream/AMReX-26.06"

CONFIG_ERROR = (
    "no HPC site configuration found. Set HPC_LOGIN=user@host and "
    "REMOTE_ROOT=/absolute/work/root either in the environment or in the "
    "private state file\n"
    "  ~/.local/share/rrea_warpx_amrex/hpc/credentials.env\n"
    "(mode 0600; optional HPC_SLURM_ACCOUNT and HPC_SLURM_MODULES configure "
    "the Slurm account and the module stack loaded by cluster jobs). "
    "See docs/HPC_CONNECTION_AND_2FA.md.")


def state_dir() -> Path:
    """The private local state directory holding credentials.env and pins."""
    return Path(os.environ.get(
        "HPC_STATE_DIR",
        str(Path.home() / ".local" / "share" / "rrea_warpx_amrex" / "hpc")))


def _value_from_env_file(path: Path, wanted_key: str) -> str | None:
    if not path.is_file():
        return None
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition("=")
        if separator and key.strip() == wanted_key and value.strip():
            return value.strip().strip("`\"'")
    return None


def site_value(key: str) -> str | None:
    """One non-secret site setting: environment first, then credentials.env.

    Only connection-level settings (HPC_LOGIN, REMOTE_ROOT, HPC_SLURM_ACCOUNT,
    HPC_SLURM_MODULES) are read here; secrets stay in the askpass helper.
    """
    explicit = os.environ.get(key, "").strip()
    if explicit:
        return explicit
    return _value_from_env_file(state_dir() / "credentials.env", key)


def login() -> str:
    """user@host for every remote call; nothing site-specific is hardcoded."""
    login = site_value("HPC_LOGIN")
    if not login:
        raise SystemExit(CONFIG_ERROR)
    return login


def base() -> str:
    """The absolute remote work root (runs/, logs/, repo copy live under it)."""
    root = site_value("REMOTE_ROOT")
    if not root:
        raise SystemExit(CONFIG_ERROR)
    return root.rstrip("/")


def remote_env() -> list[str]:
    """Env assignments that carry the site configuration to remote bash.

    Wrappers and sbatch scripts read REMOTE_ROOT, HPC_SLURM_ACCOUNT and
    HPC_SLURM_MODULES; forwarding them here keeps every site-specific value in
    the one private local config.
    """
    pairs = [f"REMOTE_ROOT={base()}"]
    for key in ("HPC_SLURM_ACCOUNT", "HPC_SLURM_MODULES"):
        value = site_value(key)
        if value:
            pairs.append(f"{key}={value}")
    return ["env", *pairs]

SOCKET_HELP = (
    "HPC control socket is not live and could not be reopened "
    "automatically.  Open it from a Windows-side shell (never git-bash -- it "
    "rewrites the socket path and reports a live socket as dead); the env var "
    "must be set INSIDE WSL because Windows environment does not cross the "
    "WSL boundary:\n"
    "  wsl -e env HPC_TOTP_AUTOMATED=1 bash scripts/hpc_open_socket.sh")


# extractall's `filter` parameter landed in 3.12 and in the 3.10.12 security
# backport.  This checkout's Windows interpreter is 3.10.11 -- one patch short
# -- while its WSL interpreter is 3.12.3, so the identical command succeeded
# from one shell and raised TypeError from the other.  tarfile.data_filter
# exists exactly when the parameter is accepted, so ask about that rather than
# comparing version tuples.
_TAR_EXTRACT_KWARGS = {"filter": "data"} if hasattr(tarfile, "data_filter") else {}


def in_wsl(argv: list[str]) -> list[str]:
    """Run a Linux argv where the mux socket lives: WSL on Windows, here on Linux."""
    return ["wsl", "-e", *argv] if sys.platform == "win32" else argv


def ssh_argv() -> list[str]:
    """The ssh prefix every remote call shares."""
    return in_wsl(["ssh", *SSH_OPTS, login()])


def ssh(args: list[str], *, stdin: bytes | None = None,
        check: bool = True) -> subprocess.CompletedProcess:
    """One remote command over the shared socket, arguments as a list.

    ssh joins its command words with spaces and the remote shell re-splits
    them, so each argument is quoted here to keep list semantics end to end.
    """
    proc = subprocess.run(ssh_argv() + [" ".join(shlex.quote(a) for a in args)],
                          input=stdin, capture_output=True)
    if check and proc.returncode != 0:
        sys.stderr.write(proc.stderr.decode("utf-8", "replace")[-2000:])
        raise SystemExit(
            f"remote command failed ({proc.returncode}): {' '.join(args)[:200]}")
    return proc


def text(args: list[str], *, stdin: bytes | None = None) -> str:
    return ssh(args, stdin=stdin).stdout.decode("utf-8", "replace").strip()


def socket_is_live() -> bool:
    return ssh(["true"], check=False).returncode == 0


def ensure_socket() -> None:
    """Reopen the mux socket automatically when it has dropped.

    Runs the opener in automated-TOTP mode (the opener
    exits 0 immediately when the socket is already usable). Must be invoked
    from a Windows-side process, never git-bash, so the WSL socket path
    survives untranslated.
    """
    if socket_is_live():
        return
    opener = Path(__file__).resolve().with_name("hpc_open_socket.sh")
    # The automated-TOTP flag must be set inside WSL: Windows-side environment
    # does not cross the WSL boundary, and without it the opener blocks
    # forever on an interactive TOTP prompt.
    cmd = in_wsl(["env", "HPC_TOTP_AUTOMATED=1", "bash", opener.name])
    try:
        opened = subprocess.run(cmd, cwd=opener.parent,
                                timeout=240).returncode == 0
    except subprocess.TimeoutExpired:
        opened = False
    if not (opened and socket_is_live()):
        raise SystemExit(SOCKET_HELP)


def cmd_run(args: argparse.Namespace) -> None:
    print(text([*remote_env(), "bash", "-c", args.command]))


def cmd_script(args: argparse.Namespace) -> None:
    forwarded = list(args.args)
    with open(args.file, "rb") as handle:
        # Strip CR before piping: remote bash reads this on stdin, so a CRLF
        # working copy makes `set -euo pipefail` arrive as `pipefail\r` and
        # die with "invalid option name" -- 20 lines before anything runs.
        # .gitattributes pins the tree to eol=lf and git normalises on commit,
        # so the committed script and the synced cluster copy are always fine;
        # only a locally rewritten file on Windows can carry CRs here, and it
        # looks CLEAN to `git status` because text=auto normalises on read.
        script = handle.read().replace(b"\r\n", b"\n")
    print(text([*remote_env(), "bash", "-s", "--", *forwarded], stdin=script))


def cmd_jobs(args: argparse.Namespace) -> None:
    since = time.strftime(
        "%Y-%m-%d", time.localtime(time.time() - args.days * 86400))
    print(text(["bash", "-c",
                'squeue -u "$USER" '
                '--format=%.10i %.25j %.8T %.10M %.6D %R; '
                f'sacct -u "$USER" -S {since} -X '
                '--format=JobID,JobName%35,State%14,Elapsed,End']))


def cmd_push(args: argparse.Namespace) -> None:
    with open(args.local, "rb") as handle:
        payload = gzip.compress(handle.read(), 6)
    remote = shlex.quote(args.remote)
    print(text(["bash", "-c",
                f"mkdir -p $(dirname {remote}) && gzip -dc > {remote}"
                f" && ls -l {remote}"], stdin=payload))


def cmd_pull(args: argparse.Namespace) -> None:
    proc = ssh(["bash", "-c", f"gzip -c < {shlex.quote(args.remote)}"])
    data = gzip.decompress(proc.stdout)
    with open(args.local, "wb") as handle:
        handle.write(data)
    print(f"{args.local}: {len(data)} bytes")


# Exactly what compute_rrea_optical_emissions.optical_series reads.
CAPTURE_ANALYSIS_FILES = (
    "rrea_reduced.csv",
    "rrea_video/particle_moments.csv",
    "rrea_video/video_capture_metadata.json",
)
# Some figure sidecars exist only after their corresponding host-side
# reduction. Probe first so a missing optional file does not fail the remote
# archive.
CAPTURE_OPTIONAL_FILES = (
    "rrea_video/current_profile.csv",
    "rrea_video/photon_band_series.csv",
    "rrea_video/electron_band_series.csv",
    # Huygens/Love observer series: the waveform and spectrum panels. Produced
    # by slurm_rrea_em_observers.sbatch, which reads the multi-GB boundary
    # record that deliberately never leaves the cluster.
    "rrea_video/rrea_em_observer_ground.csv",
    "rrea_video/rrea_em_observer_ground_metadata.json",
    "rrea_video/rrea_em_observer_aircraft.csv",
    "rrea_video/rrea_em_observer_aircraft_metadata.json",
    "rrea_video/rrea_em_observer_nadir.csv",
    "rrea_video/rrea_em_observer_nadir_metadata.json",
)

# Everything scripts/run_rrea_profiled_video_capture.py demands before it will
# resume a capture, in the layout it expects.  The driver trims the appended
# streams back to the checkpoint's write counts, so they must arrive WITH the
# checkpoint or the restart aborts on the first missing one. video_frames.bin
# dominates, and its compressed size is data-dependent. There is no smaller
# correct substitute, because the
# engine appends blindly after a restore and a short or fabricated stream
# silently misaligns every later frame.
RESTART_REQUIRED_FILES = (
    "command.json",
    "rrea_reduced.csv",
    "profiled_video_seed_schedule.csv",
    "video_capture_steps.csv",
    "rrea_video/video_frames.bin",
    "rrea_video/video_frame_metadata.csv",
    "rrea_video/video_capture_metadata.json",
    "rrea_video/particle_moments.csv",
    "rrea_video/field_direction_moments.csv",
)
# Present on most captures, and wanted when they are, but a capture without
# them still restarts.
RESTART_OPTIONAL_FILES = (
    ".rrea_run_directory.json",
    "video_render_map.json",
    "ambient_field_audit.json",
    "restart_trim_log.json",
    # PARMA continuous-source runs cannot restart without their stored flux
    # table (the driver fails closed); pulse/C&D captures don't have one,
    # so it is optional HERE and enforced by its single owner, the driver.
    "parma_source_table.txt",
    # Only a C&D capture has one, but when it does the driver TRIMS it on
    # restart (plan_cd_plane_flux_trim) and it carries the signed plane
    # crossings the whole multiplication measurement is made of.
    "rrea_cd_plane_flux.csv",
    # The Huygens surface record. AppendMaxwellBoundaryRecord trims it to the
    # restart time and appends, so carrying it keeps ONE continuous radio
    # record across a machine move; leaving it behind starts a second segment
    # at the restart time that no observer reduction can join to the first.
    "rrea_em_boundary.bin",
    "rrea_video/current_profile.csv",
    "rrea_video/electron_band_series.csv",
    "rrea_video/photon_band_series.csv",
)


def cmd_pull_capture(args: argparse.Namespace) -> None:
    """Pull captures' compact analysis files so figures redraw without a rerun.

    Only these few small files per capture are read by the plotting
    scripts; the large per-step mesh trees are never touched.  One remote
    tar stream per capture, gzipped on the remote side. Fixed remote
    argument list: capture names cannot be interpolated into a shell command,
    and the optional side-cars are selected by probing, not by letting a
    shell glob decide.
    """
    if args.list:
        print(ssh(["ls", f"{base()}/runs"]).stdout.decode("utf-8", "replace"))
        return
    for capture in args.captures:
        # Accept the two explicit capture layouts; never guess beyond them.
        remote_dir = None
        for candidate in (f"{base()}/runs/{capture}/capture",
                          f"{base()}/runs/{capture}"):
            probe = ssh(["test", "-f", f"{candidate}/rrea_reduced.csv"],
                        check=False)
            if probe.returncode == 0:
                remote_dir = candidate
                break
        if remote_dir is None:
            raise SystemExit(
                f"no rrea_reduced.csv under {base()}/runs/{capture} (or its "
                f"capture/ subdirectory); check the name with --list")
        # `test -f` returns 1 for absent; other nonzero codes are transport
        # failures and must not be treated as missing optional files.
        present = []
        for name in CAPTURE_OPTIONAL_FILES:
            probe = ssh(["test", "-f", f"{remote_dir}/{name}"], check=False)
            if probe.returncode == 0:
                present.append(name)
            elif probe.returncode != 1:
                raise SystemExit(
                    f"{capture}: probing {name} failed with ssh exit "
                    f"{probe.returncode} (not a missing file); refusing to "
                    "pull a capture whose side-cars are unknown")
        for name in CAPTURE_OPTIONAL_FILES:
            if name not in present:
                print(f"{capture}: no {name} (not fetched)")
        result = ssh(
            ["tar", "czf", "-", "-C", remote_dir,
             *CAPTURE_ANALYSIS_FILES, *present],
            check=False)
        if result.returncode != 0:
            raise SystemExit(
                f"{capture}: remote tar failed:\n"
                f"{result.stderr.decode('utf-8', 'replace')[-1500:]}")
        dest = args.dest / capture
        dest.mkdir(parents=True, exist_ok=True)
        archive = dest / "_fetch.tar.gz"
        archive.write_bytes(result.stdout)
        with tarfile.open(archive) as tar:
            tar.extractall(dest, **_TAR_EXTRACT_KWARGS)
        archive.unlink()
        missing = sorted(
            f for f in CAPTURE_ANALYSIS_FILES if not (dest / f).exists())
        if missing:
            raise SystemExit(f"{capture}: incomplete fetch, missing {missing}")
        size = sum((dest / f).stat().st_size for f in CAPTURE_ANALYSIS_FILES)
        print(f"{capture}: {size/1e6:.1f} MB uncompressed "
              f"({len(result.stdout)/1e6:.1f} MB transferred) -> {dest}")


def cmd_pull_checkpoint(args: argparse.Namespace) -> None:
    """Bring one engine checkpoint tree home without capture streams.

    This fetches engine state only; a production capture continuation also
    needs its append-only streams and should use `pull-restart`.
    rrea_em_boundary.bin is omitted because the engine can start a fresh
    diagnostic record; fetch it separately only for a continuous EM record.
    """
    # Probe both supported checkpoint layouts by content.
    for candidate in (f"{base()}/runs/{args.run}/capture",
                      f"{base()}/runs/{args.run}"):
        probe = ssh(["bash", "-c",
                     f"ls -d {shlex.quote(candidate)}/chk[0-9]* >/dev/null 2>&1"],
                    check=False)
        if probe.returncode == 0:
            remote_dir = candidate
            break
    else:
        raise SystemExit(
            f"no chk* checkpoint under {base()}/runs/{args.run} "
            f"(or its capture/ subdirectory)")
    dest = args.dest / args.run
    dest.mkdir(parents=True, exist_ok=True)
    # Stream the growing checkpoint archive instead of buffering it in memory.
    base = ssh_argv()
    # A live run may rotate between probe and fetch. Re-resolve `auto`; a named
    # checkpoint is never silently substituted.
    attempts = 3 if args.checkpoint == "auto" else 1
    for attempt in range(1, attempts + 1):
        # COMPLETE is the engine's atomic publication record for a checkpoint.
        name = _newest_complete_checkpoint(remote_dir, args.checkpoint)
        remote = " ".join(shlex.quote(a) for a in
                          ["tar", "czf", "-", "-C", remote_dir, name])
        proc = subprocess.Popen(base + [remote], stdout=subprocess.PIPE)
        try:
            with tarfile.open(fileobj=proc.stdout, mode="r|gz") as tar:
                tar.extractall(dest, **_TAR_EXTRACT_KWARGS)
            failed = proc.wait() != 0
        except tarfile.TarError:
            proc.wait()
            failed = True
        if not failed:
            break
        if attempt == attempts:
            raise SystemExit(
                f"remote tar failed for {remote_dir}/{name} after "
                f"{attempts} attempts")
        print(f"{name} disappeared mid-pull (the run rotates checkpoints); "
              f"re-resolving, attempt {attempt + 1}/{attempts}")
    local = dest / name
    size = sum(f.stat().st_size for f in local.rglob("*") if f.is_file())
    print(f"{name}: {size/1e6:.1f} MB -> {local}")
    print(f"resume with: --restart-checkpoint {local} --restart-epoch-break "
          f"--restart-epoch-reason cluster_to_local_handoff")


def _newest_complete_checkpoint(remote_dir: str, name: str) -> str:
    """Resolve 'auto' to the newest checkpoint the engine finished writing."""
    if name != "auto":
        return name
    listing = text(["bash", "-c",
                    f"ls -d {remote_dir}/chk[0-9]* 2>/dev/null "
                    "| xargs -r -n1 basename | sort -r"])
    for candidate in listing.split():
        # COMPLETE is written last, so its presence is the only safe signal
        # that the directory is not still being filled in.
        if ssh(["test", "-f",
                f"{remote_dir}/{candidate}/rrea_checkpoint/COMPLETE"],
               check=False).returncode == 0:
            return candidate
    raise SystemExit(f"no COMPLETE chk* under {remote_dir}")


def _tilde(path: str) -> str:
    """Shell-quote a path, leaving a leading ~ expandable.

    shlex.quote("~/x") returns "'~/x'", which bash does not tilde-expand, so
    the staged run would land in a literal "~" directory under the Windows
    cwd -- on /mnt/c inside the repo, where a continuation must never live.
    """
    if path == "~" or path.startswith("~/"):
        return "~" + shlex.quote(path[1:])
    return shlex.quote(path)


def _restart_plan(exists, chk: tuple[str, str],
                  src_capture: str, dst_capture: str, label: str
                  ) -> list[tuple[str, str]]:
    """The artifact set the driver refuses to restart without, either direction.

    `exists` tests one capture-relative file on the source side: ssh one way,
    Path.is_file the other.  Optional files are reported, not demanded.
    """
    missing = [f for f in RESTART_REQUIRED_FILES if not exists(f)]
    if missing:
        raise SystemExit(
            f"{label} is missing restart artifacts the driver requires: "
            + ", ".join(missing))
    present = [f for f in RESTART_OPTIONAL_FILES if exists(f)]
    for f in RESTART_OPTIONAL_FILES:
        if f not in present:
            print(f"{label}: no {f}")
    return [chk] + [(f"{src_capture}/{f}", f"{dst_capture}/{f}")
                    for f in (*RESTART_REQUIRED_FILES, *present)]


def _rsync(source: str, target: str, *, make_parent: str | None = None) -> None:
    """Move one artifact over the mux socket, resumable in place.

    --partial --inplace lets a large stream resume instead of restarting into
    a fresh temporary. Exactly one side
    carries the login (user@host:); the other may start with ~, so both go
    through _tilde.
    """
    prefix = f"mkdir -p {_tilde(make_parent)} && " if make_parent else ""
    print(f"-> {target}", flush=True)
    run_local_shell(
        prefix
        + f"rsync -a --partial --inplace --info=progress2 --no-inc-recursive "
          f"-e {shlex.quote(RSH)} {_tilde(source)} {_tilde(target)}")


def cmd_pull_restart(args: argparse.Namespace) -> None:
    """Stage a cluster capture for continuation on this machine, completely.

    This transfers the checkpoint and every append-only stream the driver
    realigns to it; neither subset is a complete production restart.

    The destination is a LINUX path, not a Windows one, because the run
    continues under a Linux MPI build: staging onto /mnt/c and stepping there
    means every checkpoint write crosses the 9p bridge.  Transfers use rsync
    over the same mux socket, so a dropped link resumes in place.

    Rank count is expected to change -- that is the point of moving a run to a
    laptop -- and the engine treats a rank-count change as a restart-epoch
    deviation, so the printed resume line carries --restart-epoch-break.
    """
    for candidate in (f"{base()}/runs/{args.run}/capture",
                      f"{base()}/runs/{args.run}"):
        if ssh(["test", "-f", f"{candidate}/command.json"],
               check=False).returncode == 0:
            capture_dir = candidate
            break
    else:
        raise SystemExit(
            f"no capture/command.json under {base()}/runs/{args.run}; "
            "a capture that never recorded its command cannot be restarted")
    # chk* sit beside capture/, not inside it (the driver's default prefix is
    # output_dir.parent/"chk").
    chk_root = capture_dir[:-len("/capture")] if capture_dir.endswith("/capture")         else capture_dir
    name = _newest_complete_checkpoint(chk_root, args.checkpoint)

    dest = args.dest.rstrip("/") + "/" + args.run
    plan = _restart_plan(
        lambda f: ssh(["test", "-f", f"{capture_dir}/{f}"],
                      check=False).returncode == 0,
        (f"{chk_root}/{name}/", f"{dest}/{name}/"),
        capture_dir, f"{dest}/capture", args.run)
    for source, target in plan:
        _rsync(login() + ":" + source, target,
               make_parent=target.rsplit("/", 1)[0])

    print("")
    print(f"staged {args.run} at {dest}")
    print("resume from the Linux side; ranks may differ from the cluster's:")
    cont = " \\\n    "
    print("  python3 scripts/run_rrea_profiled_video_capture.py" + cont
          + cont.join([
              "--warpx-exe <local warpx.rz>",
              f"--transport-config <repo>/{TABLE_BUNDLE}/transport_physics.json",
              f"--run-root {args.dest.rstrip('/')}",
              f"--output-dir {dest}/capture",
              "--e0-peak-kv-per-m <E0>",
              f"--restart-checkpoint {dest}/{name}",
              "--restart-epoch-break --restart-epoch-reason cluster_to_local_handoff",
              "--launcher mpiexec --ranks <N> --cpus-per-rank 1",
          ]))


def cmd_push_restart(args: argparse.Namespace) -> None:
    """Send a locally-continued run back to the cluster, completely.

    The same checkpoint-plus-stream artifact set used by pull-restart travels
    in the opposite direction.

    The local copy is left untouched, so a failed or cancelled cluster
    submission costs nothing -- decide afterwards which side is authoritative,
    and do not step both.
    """
    source_run = Path(args.local_run).expanduser()
    capture = source_run / "capture"
    if not (capture / "command.json").is_file():
        raise SystemExit(
            f"{capture}/command.json is missing; a capture that never "
            "recorded its command cannot be restarted")

    complete = sorted(
        d for d in source_run.glob("chk*")
        if (d / "rrea_checkpoint" / "COMPLETE").is_file())
    if args.checkpoint != "auto":
        chosen = source_run / args.checkpoint
        if not (chosen / "rrea_checkpoint" / "COMPLETE").is_file():
            raise SystemExit(f"{chosen} has no rrea_checkpoint/COMPLETE marker")
    elif complete:
        chosen = complete[-1]
    else:
        raise SystemExit(f"no COMPLETE checkpoint under {source_run}")

    dest = f"{base()}/runs/{args.run}"
    # One remote mkdir covers every target: rsync creates the checkpoint
    # directory itself, and rrea_video is the only nested capture parent.
    ssh(["mkdir", "-p", f"{dest}/capture/rrea_video"])
    plan = _restart_plan(
        lambda f: (capture / f).is_file(),
        (f"{chosen.as_posix()}/", f"{dest}/{chosen.name}/"),
        capture.as_posix(), f"{dest}/capture", str(source_run))
    for source, target in plan:
        _rsync(source, login() + ":" + target)

    print("")
    print(f"pushed {source_run} to {dest} (local copy untouched)")
    tag = args.run[len("scratch_"):] if args.run.startswith("scratch_") else None
    if tag is None:
        print("NOTE: submit_profiled_video.sh derives its run dir as "
              f"runs/scratch_<RUN_TAG>, so it will not find {args.run!r}. "
              "Push to a 'scratch_'-prefixed name, or rename it there.")
        tag = args.run
    print("continue it with:")
    print("  hpc.py script cluster/hpc/submit_profiled_video.sh \\")
    print(f"    RUN_TAG={tag} PROFILED_VIDEO_RESTART_FROM=auto \\")
    print("    RREA_WARPX_EXE=<engine> PROFILED_VIDEO_E0_KV_PER_M=<E0> \\")
    print("    NTASKS=<N> WALLTIME=<hh:mm:ss>")
    print("An engine or rank-count change needs the restart-epoch break "
          "(PROFILED_VIDEO_RESTART_EPOCH_REASON), and the stop time must "
          "stay the one the run was continuing from.")


def run_local_shell(script: str) -> None:
    """Run a shell snippet on the Linux side, wherever this tool is running.

    On Windows the mux socket, rsync and the destination filesystem all live
    inside WSL, so the snippet has to execute there; on Linux it is just bash.
    """
    argv = in_wsl(["bash", "-c", script])
    if subprocess.run(argv).returncode != 0:
        raise SystemExit("transfer failed; re-run to resume where it stopped")


def watch(job_id: str, interval: int = 30, tail: int = 60) -> None:
    while True:
        state = text(["sacct", "-j", job_id, "-X", "-n", "-o", "State"])
        if state and "PENDING" not in state and "RUNNING" not in state:
            break
        print(f"  {job_id}: {state or 'submitted'}", flush=True)
        time.sleep(interval)
    print(f"STATE={state}")
    log = text(["bash", "-c",
                f"ls {base()}/logs/*{job_id}*.out {base()}/*{job_id}*.out "
                "2>/dev/null | head -1"])
    if log:
        print(f"LOG={log}")
        print(text(["tail", "-n", str(tail), log]))
    if "COMPLETED" not in state:
        raise SystemExit(1)


def cmd_watch(args: argparse.Namespace) -> None:
    watch(args.job_id, args.interval, args.tail)


def cmd_status(args: argparse.Namespace) -> None:
    """One-shot snapshot of jobs and their capture progress when available.

    Works for any sbatch whose log echoes an OUT=<run dir> line (all capture
    launchers do): shows Slurm state, log/stderr tails, and the tail of the
    run's reduced diagnostics CSV. One remote round trip.
    """
    ids = args.job_ids
    script = f"""set -u
sacct -j {",".join(ids)} -X --format=JobID%12,JobName%22,State%12,Elapsed,End
squeue -j {",".join(ids)} --format='%.10i %.8T %.20S %.10M %R' 2>/dev/null
outs=""
for j in {" ".join(ids)}; do
    log=$(ls {base()}/logs/*"$j"*.out {base()}/*"$j"*.out 2>/dev/null | head -1)
    [ -n "$log" ] || continue
    echo "=== job $j log tail: $log ==="
    tail -n {args.tail} "$log"
    err="${{log%.out}}.err"
    [ -s "$err" ] && {{ echo "=== job $j stderr tail ==="; tail -n 20 "$err"; }}
    outs="$outs $(sed -n 's/^OUT=//p' "$log" | head -1)"
done
for out in $(printf '%s\\n' $outs | sort -u); do
    csv="$out/capture/rrea_reduced.csv"
    if [ -s "$csv" ]; then
        echo "=== capture progress: $csv ==="
        {{ head -1 "$csv"; tail -n 2 "$csv"; }} | cut -d, -f1-6
        echo "rows=$(wc -l < "$csv")"
    fi
done
exit 0
"""
    print(text(["bash", "-s"], stdin=script.encode()))


def cmd_cancel(args: argparse.Namespace) -> None:
    """scancel the jobs, then show the queue so the effect is visible."""
    out = text(["scancel", *args.job_ids])
    print(out or f"cancelled: {' '.join(args.job_ids)}")
    print(text(["bash", "-c", 'squeue -u "$USER" '
                '--format=%.10i %.25j %.8T %.10M %R']))


def cmd_quota(args: argparse.Namespace) -> None:
    """Billing hours + disk/inode quota; the mandatory pre-submission check.

    dusage misbehaves under an inherited Python environment, so it runs with
    the Python/conda variables scrubbed.
    """
    print(text(["cost"]))
    print(text(["env", "-u", "PYTHONPATH", "-u", "PYTHONHOME",
                "-u", "CONDA_PREFIX", "-u", "CONDA_DEFAULT_ENV",
                "-u", "VIRTUAL_ENV", "dusage"]))


def head_archive() -> tuple[str, bytes]:
    """Archive committed HEAD with the embedded source-commit marker."""
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                            text=True, check=True).stdout.strip()
    if subprocess.run(["git", "status", "--porcelain"], capture_output=True,
                      text=True, check=True).stdout.strip():
        print("warning: working tree dirty; archiving committed HEAD only",
              file=sys.stderr)
    tar = subprocess.run(
        ["git", "archive", "--format=tar", "HEAD"],
        capture_output=True, check=True).stdout
    payload = commit.encode()
    buffer = io.BytesIO(tar)
    with tarfile.open(fileobj=buffer, mode="a") as archive:
        marker = tarfile.TarInfo("CODEX_SOURCE_COMMIT")
        marker.size = len(payload)
        marker.mode = 0o644
        archive.addfile(marker, io.BytesIO(payload))
    return commit, buffer.getvalue()


def pinned_submodule(treeish: str, path: str) -> tuple[str, str, str]:
    """Return the fork URL, protected branch and gitlink commit from one tree."""
    modules_text = subprocess.run(
        ["git", "show", f"{treeish}:.gitmodules"],
        capture_output=True, text=True, check=True).stdout
    modules = configparser.ConfigParser()
    modules.read_string(modules_text)
    section = f'submodule "{path}"'
    url = modules[section]["url"]
    branch = modules[section]["branch"]
    fields = subprocess.run(
        ["git", "ls-tree", treeish, "--", path],
        capture_output=True, text=True, check=True).stdout.split()
    if len(fields) < 3 or fields[0] != "160000" or len(fields[2]) != 40:
        raise SystemExit(f"{path} is not a pinned submodule in committed HEAD")
    return url, branch, fields[2]


def cmd_sync(args: argparse.Namespace) -> None:
    """Update the cluster repo copy (repo_amrex_latest) to committed HEAD.

    The tracked production tables travel with the source archive; they have no
    separate transfer step.
    """
    commit, tar = head_archive()
    repo = f"{base()}/repo_amrex_latest"
    script = f"""set -euo pipefail
tmp=$(mktemp /tmp/rrea_sync_XXXXXX.tar)
gzip -dc > "$tmp"
mkdir -p {repo}
tar -xf "$tmp" -C {repo}
rm -f "$tmp"
cat {repo}/CODEX_SOURCE_COMMIT"""
    marker = text(["bash", "-c", script],
                  stdin=gzip.compress(tar, 6)).splitlines()[-1]
    if marker != commit:
        raise SystemExit(f"sync marker mismatch: {marker} != {commit}")
    print(f"repo_amrex_latest -> {commit}")


def cmd_build(args: argparse.Namespace) -> None:
    commit, tar = head_archive()
    warpx_url, warpx_branch, warpx_commit = pinned_submodule(commit, WARPX_SUBMODULE)
    amrex_url, amrex_branch, amrex_commit = pinned_submodule(commit, AMREX_SUBMODULE)
    up = f"{base()}/uploads/{commit}"
    src = f"{base()}/sources/rrea_cd_{commit}"
    warpx_src = f"{base()}/upstream_git/davsar89-WarpX-{warpx_commit[:12]}"
    amrex_src = f"{base()}/upstream_git/davsar89-AMReX-{amrex_commit[:12]}"
    script = f"""set -euo pipefail
ensure_upstream() {{
    local url="$1" branch="$2" commit="$3" path="$4"
    if [[ ! -d "$path/.git" ]]; then
        [[ ! -e "$path" ]] || {{ echo "non-git upstream path exists: $path" >&2; exit 2; }}
        git clone --depth 1 --branch "$branch" "$url" "$path"
    fi
    [[ "$(git -C "$path" remote get-url origin)" == "$url" ]] \
        || {{ echo "upstream origin mismatch: $path" >&2; exit 2; }}
    [[ "$(git -C "$path" rev-parse HEAD)" == "$commit" ]] \
        || {{ echo "upstream commit mismatch: $path" >&2; exit 2; }}
    [[ -z "$(git -C "$path" status --porcelain --untracked-files=all)" ]] \
        || {{ echo "upstream checkout is dirty: $path" >&2; exit 2; }}
}}
ensure_upstream {shlex.quote(warpx_url)} {shlex.quote(warpx_branch)} {warpx_commit} {warpx_src}
ensure_upstream {shlex.quote(amrex_url)} {shlex.quote(amrex_branch)} {amrex_commit} {amrex_src}
mkdir -p {up} {src}
gzip -dc > {up}/rrea_source.tar
tar -xf {up}/rrea_source.tar -C {src}
[ "$(cat {src}/CODEX_SOURCE_COMMIT)" = "{commit}" ] || {{ echo marker mismatch >&2; exit 1; }}
cd {base()}
sbatch --parsable ${HPC_SLURM_ACCOUNT:+--account="$HPC_SLURM_ACCOUNT"} \
  --export=ALL,\
RREA_SOURCE_ARCHIVE={up}/rrea_source.tar,\
WARPX_SRC={warpx_src},\
AMREX_SRC={amrex_src},\
EXPECTED_WARPX_COMMIT={warpx_commit},\
EXPECTED_AMREX_COMMIT={amrex_commit},\
RREA_INSTALL_PARENT={base()}/opt,\
RREA_SMOKE_TABLE_CONFIG={base()}/repo_amrex_latest/rrea_transport_tables/schema6/production/transport_physics.json \
  {src}/cluster/hpc/slurm_build_warpx_lto_only.sbatch"""
    job_id = text([*remote_env(), "bash", "-c", script],
                  stdin=gzip.compress(tar, 6)).splitlines()[-1]
    print(f"commit={commit}")
    print(f"job={job_id}")
    if args.watch:
        watch(job_id)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("run", help="run a remote command")
    p.add_argument("command")
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("script", help="pipe a local script to remote bash -s")
    p.add_argument("file")
    p.add_argument("args", nargs="*",
                   help="forwarded to the remote script as $1..")
    p.set_defaults(fn=cmd_script)

    p = sub.add_parser("jobs", help="queue + recent job history")
    p.add_argument("--days", type=int, default=2)
    p.set_defaults(fn=cmd_jobs)

    p = sub.add_parser("push", help="upload a file, gzip over the pipe")
    p.add_argument("local")
    p.add_argument("remote")
    p.set_defaults(fn=cmd_push)

    p = sub.add_parser("pull", help="download a file, gzip over the pipe")
    p.add_argument("remote")
    p.add_argument("local")
    p.set_defaults(fn=cmd_pull)

    p = sub.add_parser(
        "pull-capture",
        help="fetch a capture's compact analysis CSVs for local figures")
    p.add_argument("captures", nargs="*", help="capture directory names")
    p.add_argument(
        "--dest", type=Path,
        default=ROOT / "tmp" / "hpc_captures")
    p.add_argument("--list", action="store_true",
                   help="list the available captures and exit")
    p.set_defaults(fn=cmd_pull_capture)


    p = sub.add_parser(
        "pull-checkpoint",
        help="fetch engine checkpoint state only")
    p.add_argument("run", help="run name under runs/")
    p.add_argument("checkpoint", nargs="?", default="auto",
                   help="chk000000 name, or 'auto' for the newest complete one")
    p.add_argument("--dest", type=Path, default=ROOT / "tmp/hpc_checkpoints")
    p.set_defaults(fn=cmd_pull_checkpoint)

    p = sub.add_parser(
        "pull-restart",
        help="fetch a checkpoint AND every artifact the driver needs to "
             "resume it here (including the frame stream)")
    p.add_argument("run", help="run name under runs/")
    p.add_argument("checkpoint", nargs="?", default="auto",
                   help="chk000000 name, or 'auto' for the newest COMPLETE one")
    p.add_argument(
        "--dest", default="~/rrea_runs",
        help="LINUX staging root (the run continues under a Linux MPI build, "
             "so keep it off /mnt/c); default ~/rrea_runs")
    p.set_defaults(fn=cmd_pull_restart)

    p = sub.add_parser(
        "push-restart",
        help="send a locally-continued run back to the cluster, complete "
             "enough to restart there")
    p.add_argument(
        "local_run",
        help="run dir holding chk*/ and capture/. It is read directly, so on "
             "Windows run this command itself inside WSL "
             "(wsl -e python3 scripts/hpc.py push-restart ...) -- the run "
             "lives on the Linux filesystem, not a drive letter")
    p.add_argument("run", help="destination run name under runs/")
    p.add_argument("checkpoint", nargs="?", default="auto",
                   help="chk000000 name, or 'auto' for the newest COMPLETE one")
    p.set_defaults(fn=cmd_push_restart)

    p = sub.add_parser("watch", help="poll a Slurm job until done, tail its log")
    p.add_argument("job_id")
    p.add_argument("--interval", type=int, default=30)
    p.add_argument("--tail", type=int, default=60)
    p.set_defaults(fn=cmd_watch)

    p = sub.add_parser(
        "status", help="one-shot job state + log tails + capture progress")
    p.add_argument("job_ids", nargs="+")
    p.add_argument("--tail", type=int, default=15)
    p.set_defaults(fn=cmd_status)

    p = sub.add_parser("cancel", help="scancel jobs, then show the queue")
    p.add_argument("job_ids", nargs="+")
    p.set_defaults(fn=cmd_cancel)

    p = sub.add_parser(
        "quota", help="cost + dusage; the mandatory pre-submission check")
    p.set_defaults(fn=cmd_quota)

    p = sub.add_parser(
        "sync", help="update the cluster repo copy to committed HEAD")
    p.set_defaults(fn=cmd_sync)


    p = sub.add_parser("build", help="archive HEAD, upload, submit the build")
    p.add_argument("--watch", action="store_true")
    p.set_defaults(fn=cmd_build)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    ensure_socket()
    args.fn(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
