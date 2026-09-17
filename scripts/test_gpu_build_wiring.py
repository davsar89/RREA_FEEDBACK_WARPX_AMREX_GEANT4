"""Build/launcher wiring only; these are not AMReX or CUDA physics tests."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from prepare_rrea_warpx_source import patch_top_cmake

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "native_amrex_rrea/cmake/RunWarpXSmoke.cmake"


def run_smoke_wrapper(tmp_path: Path, cuda: bool, exit_code: int = 0):
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
