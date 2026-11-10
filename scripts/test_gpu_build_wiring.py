"""Build/launcher wiring only; these are not AMReX or CUDA physics tests."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from prepare_rrea_warpx_source import patch_top_cmake, copy_overlay

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "native_amrex_rrea/cmake/RunWarpXSmoke.cmake"


def test_overlay_refresh_rebuilds_changed_code_with_old_source_mtime(tmp_path):
    if not shutil.which("make") or not shutil.which("c++"):
        pytest.skip("incremental build check needs make and a C++ compiler")
    source = tmp_path/"overlay/warpx_overlay/Source/Rrea/probe.cpp"
    source.parent.mkdir(parents=True)
    prepared = tmp_path/"prepared"
    prepared.mkdir()
    (prepared/"Makefile").write_text("probe: Source/Rrea/probe.cpp\n\tc++ $< -o $@\n")
    for result in (1,2):
        source.write_text(f"int main() {{ return {result}; }}\n")
        os.utime(source,(1_000_000_000,1_000_000_000))
        copy_overlay(tmp_path/"overlay",prepared)
        subprocess.run(["make"],cwd=prepared,capture_output=True,check=True)
        assert subprocess.run([str(prepared/"probe")]).returncode == result


@pytest.fixture
def capture_launcher(tmp_path):
    if os.name == "nt":
        pytest.skip("POSIX batch launcher fixture")
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts/run_rrea_profiled_video_capture.py").write_text(
        "import argparse, json, os\nfrom pathlib import Path\n"
        "p=argparse.ArgumentParser()\n"
        "for key in ('output-dir','run-root','transport-omp-threads','omp-threads'): "
        "p.add_argument('--'+key)\n"
        "a,_=p.parse_known_args(); print(json.dumps(vars(a)))\n"
        "if a.output_dir:\n"
        " out=Path(a.output_dir); (out/'frames.bin').write_bytes(b'frames')\n"
        " (out/'diagnostics.csv').write_text('time_s\\n0\\n')\n"
        " for step in (999999,1000000,1000001):\n"
        "  c=Path(a.run_root)/f'chk{step}'/'rrea_checkpoint'; c.mkdir(parents=True)\n"
        "  if step<1000001: (c/'COMPLETE').touch()\n"
        "raise SystemExit(int(os.environ.get('CAPTURE_EXIT','0')))\n")
    shell_env = tmp_path / "shell_env"
    shell_env.write_text('module() { :; }\nnvidia-smi() { :; }\nuname() { echo aarch64; }\n')
    return {**os.environ, "BASH_ENV": str(shell_env), "REPO_ROOT": str(tmp_path),
            "REMOTE_ROOT": str(tmp_path), "GPU_WORK": str(tmp_path / "gpu"),
            "GPUS": "1", "SLURM_NTASKS": "1", "SLURM_CPUS_PER_TASK": "12",
            "SLURM_JOB_ID": "123", "PYTHON_BIN": sys.executable}


@pytest.mark.parametrize("override", [[], ["--transport-omp-threads", "3"]])
def test_gpu_capture_uses_allocated_transport_threads(tmp_path, capture_launcher, override):
    result = subprocess.run(
        ["bash", str(ROOT / "cluster/olivia/slurm_rrea_gpu.sbatch"), "run", *override],
        env=capture_launcher, capture_output=True, text=True, check=True)
    args = json.loads(result.stdout)
    assert int(args["transport_omp_threads"]) == (3 if override else 12)
    assert int(args["omp_threads"]) == 1


@pytest.mark.parametrize("persistent", [False, True])
@pytest.mark.parametrize("status", [0, 7])
def test_capture_stages_only_scratch(tmp_path, capture_launcher, persistent, status):
    if not shutil.which("rsync"):
        pytest.skip("scratch staging needs rsync")
    output = tmp_path / ("runs/case" if persistent else "scratch/case")
    result = subprocess.run(
        ["bash", str(ROOT / "cluster/olivia/slurm_rrea_profiled_video_capture.sbatch")],
        env={**capture_launcher, "PROFILED_VIDEO_SCRATCH_OUT": str(output),
             "CAPTURE_EXIT": str(status)}, capture_output=True, text=True)
    assert result.returncode == status, result.stderr
    final = output if persistent else tmp_path / "runs/profiled_video_schema6_123"
    assert (final / "job_exit_status.txt").read_text().strip() == str(status)
    assert (final / "final_checkpoint_name.txt").read_text().strip() == "chk1000000"
    assert (final / "chk1000000/rrea_checkpoint/COMPLETE").is_file()
    if persistent:
        assert not (tmp_path / "runs/profiled_video_schema6_123").exists()
        assert not list(final.glob("video_capture_*.tar.*"))
    elif status == 0:
        archive, = final.glob("video_capture_*.tar.*")
        recovered = subprocess.check_output(["tar", "-xOf", str(archive), "capture/frames.bin"])
        assert recovered == b"frames"
    if status == 0:
        assert (tmp_path / "runs/profiled_video_schema6_latest").resolve() == final


@pytest.mark.parametrize("gpus", [1, 2, 3, 4])
@pytest.mark.skipif(os.name == "nt", reason="POSIX launcher test; run under WSL/Linux")
def test_olivia_gpu_submission_preserves_driver_arguments(tmp_path, gpus):
    recorder = tmp_path / "sbatch"
    recorder.write_text(
        '#!/usr/bin/env python3\nimport json, sys\nprint(json.dumps(sys.argv[1:]))\n')
    recorder.chmod(0o755)
    result = subprocess.run(
        ["bash", str(ROOT / "cluster/olivia/submit_gpu.sh"),
         f"BASE={tmp_path}", f"GPUS={gpus}", "run", "--extra-warpx-arg", "amr.n_cell=256 128"],
        env={**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}"},
        capture_output=True, text=True, check=True)
    args = json.loads(result.stdout)
    assert "--partition=accel" in args and f"--gpus={gpus}" in args
    assert f"--ntasks={gpus}" in args and "--gpus-per-task=1" in args
    assert "--gpu-bind=single:1" in args
    assert args[-3:] == ["run", "--extra-warpx-arg", "amr.n_cell=256 128"]


def run_smoke_wrapper(tmp_path: Path, cuda: bool, exit_code: int = 0, marker: bool = True):
    if os.name == "nt":
        pytest.skip("POSIX executable fixture; run under WSL/Linux")
    cmake = shutil.which("cmake")
    if cmake is None:
        pytest.skip("cmake is not installed")
    # Deliberate command-line recorder, not a mock numerical implementation.
    executable = tmp_path / "record_arguments.sh"
    executable.write_text(
        "#!/bin/sh\nprintf '%s\\n' \"$@\"\nprintf 'ARGUMENT_RECORD_PASS\\n'\n"
        f"exit {exit_code}\n",
        encoding="utf-8",
    )
    if not marker:
        executable.write_text(executable.read_text().replace("ARGUMENT_RECORD_PASS", "NO_MATCH"))
    executable.chmod(0o755)
    return subprocess.run(
        [cmake, f"-DWARPX_EXE={executable}", "-DINPUT=inputs",
         f"-DOUTPUT_DIR={tmp_path / 'output'}", f"-DWORKING_DIR={tmp_path}",
         "-DPASS_REGEX=ARGUMENT_RECORD_PASS", f"-DRREA_CUDA={'ON' if cuda else 'OFF'}",
         "-P", str(RUNNER)],
        capture_output=True, text=True, check=False,
    )


@pytest.mark.parametrize("cuda", [False, True])
def test_managed_memory_is_requested_before_cuda_initialization(tmp_path, cuda):
    result = run_smoke_wrapper(tmp_path, cuda)
    assert result.returncode == 0, result.stdout + result.stderr
    assert ("amrex.the_arena_is_managed=1" in result.stderr + result.stdout) is cuda


def test_native_failure_is_not_masked_by_a_pass_marker(tmp_path):
    result = run_smoke_wrapper(tmp_path, cuda=True, exit_code=7)
    assert result.returncode != 0
    assert "WarpX smoke exited 7" in result.stderr


def test_success_without_required_output_is_rejected(tmp_path):
    result = run_smoke_wrapper(tmp_path, cuda=True, marker=False)
    assert result.returncode != 0
    assert "ARGUMENT_RECORD_PASS" in result.stderr


def test_prepared_runtime_tests_receive_the_backend(tmp_path):
    cmake = tmp_path / "CMakeLists.txt"
    cmake.write_text(
        "add_subdirectory(Source/ablastr)\n"
        "foreach(D IN LISTS WarpX_DIMS)\n"
        "    add_subdirectory(Source/FieldSolver)\nendforeach()\n",
        encoding="utf-8",
    )
    patch_top_cmake(cmake)
    patch_top_cmake(cmake)  # refresh must not duplicate the production tests
    text = cmake.read_text(encoding="utf-8")
    assert text.count('"-DRREA_CUDA=${RREA_TEST_CUDA}"') == 3
    assert text.count("# BEGIN RREA PRODUCTION EXECUTABLE TEST") == 1
