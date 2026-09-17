#!/usr/bin/env python3
"""Prepare a source-built WarpX tree with the RREA coupling overlay.

This script is intentionally deterministic and refuses to modify the pristine
upstream WarpX checkout.  It copies WarpX into a build source directory, copies
native_amrex_rrea/warpx_overlay/Source/Rrea into that tree, and applies
small CMake/source hooks so RREA's Maxwell field evolution owns the WarpX
field callback when rrea.enable=1.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


BEGIN_MARKER = "# BEGIN RREA AMREX OVERLAY"
END_MARKER = "# END RREA AMREX OVERLAY"
CPP_BEGIN_MARKER = "// BEGIN RREA HOOK"
CPP_END_MARKER = "// END RREA HOOK"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--warpx-source",
        type=Path,
        default=ROOT / "upstream/WarpX-26.06",
        help="pristine WarpX source (default: pinned davsar89 fork submodule)",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--overlay-dir",
        type=Path,
        default=ROOT / "native_amrex_rrea",
        help="RREA overlay (default: this repository's native_amrex_rrea)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Remove an existing prepared output directory before copying.",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help=(
            "Re-copy Source/Rrea and re-apply the (idempotent) hook patches "
            "onto an existing prepared tree without re-copying WarpX, so an "
            "incremental cmake --build suffices."
        ),
    )
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path.expanduser().resolve()


def copy_tree(src: Path, dst: Path, force: bool) -> None:
    if src == dst:
        raise SystemExit("Refusing to prepare RREA WarpX in-place")
    if dst.exists():
        if not force:
            raise SystemExit(f"{dst} already exists; pass --force to replace it")
        shutil.rmtree(dst)
    shutil.copytree(
        src,
        dst,
        symlinks=True,
        ignore_dangling_symlinks=True,
        ignore=shutil.ignore_patterns(
            ".git",
            "Docs",
            "Tools",
            "build",
            "build-*",
            "_deps",
            "__pycache__",
            "*.pyc",
        ),
    )


def replace_once(path: Path, needle: str, replacement: str) -> None:
    text = path.read_text(encoding="utf-8")
    if replacement in text:
        return
    if needle not in text:
        raise SystemExit(f"Could not find patch needle in {path}: {needle!r}")
    path.write_text(text.replace(needle, replacement, 1), encoding="utf-8")


def patch_top_cmake(cmake_path: Path) -> None:
    text = cmake_path.read_text(encoding="utf-8")
    overlay_block = f"""{BEGIN_MARKER}
set(RREA_AMREX_OVERLAY "" CACHE PATH "Path to native_amrex_rrea overlay")
if(RREA_AMREX_OVERLAY)
    if(RREA_AMREX_BUILD_TESTS)
        enable_testing()
    endif()
    set(RREA_AMREX_INSTALL OFF CACHE BOOL "Install standalone RREA package" FORCE)
    add_subdirectory(${{RREA_AMREX_OVERLAY}} ${{CMAKE_BINARY_DIR}}/rrea_amrex_overlay)
endif()
{END_MARKER}

"""
    if BEGIN_MARKER not in text:
        needle = "add_subdirectory(Source/ablastr)\n"
        replace_once(cmake_path, needle, overlay_block + needle)

    # enable_testing() in a subdirectory alone leaves root ctest reporting
    # zero tests, so expose native tests at the WarpX top level.
    text = cmake_path.read_text(encoding="utf-8")
    old_overlay_start = """if(RREA_AMREX_OVERLAY)
    set(RREA_AMREX_INSTALL OFF CACHE BOOL "Install standalone RREA package" FORCE)
"""
    new_overlay_start = """if(RREA_AMREX_OVERLAY)
    if(RREA_AMREX_BUILD_TESTS)
        enable_testing()
    endif()
    set(RREA_AMREX_INSTALL OFF CACHE BOOL "Install standalone RREA package" FORCE)
"""
    if old_overlay_start in text:
        replace_once(cmake_path, old_overlay_start, new_overlay_start)

    text = cmake_path.read_text(encoding="utf-8")
    source_block = f"""    {BEGIN_MARKER}
    if(RREA_AMREX_OVERLAY)
        add_subdirectory(Source/Rrea)
    endif()
    {END_MARKER}
"""
    if "add_subdirectory(Source/Rrea)" not in text:
        needle = "    add_subdirectory(Source/FieldSolver)\n"
        replace_once(cmake_path, needle, needle + source_block)

    text = cmake_path.read_text(encoding="utf-8")
    runtime_test_marker = "# BEGIN RREA PRODUCTION EXECUTABLE TEST"
    runtime_test_end_marker = "# END RREA PRODUCTION EXECUTABLE TEST"
    runtime_test_block = f"""
{runtime_test_marker}
if(RREA_AMREX_OVERLAY AND RREA_AMREX_BUILD_TESTS)
    set(RREA_PROJECT_ROOT "${{RREA_AMREX_OVERLAY}}/..")
    set(
        RREA_PRODUCTION_TRANSPORT_CONFIG
        "${{RREA_PROJECT_ROOT}}/rrea_transport_tables/schema6/production/transport_physics.json"
        CACHE FILEPATH "Transport configuration for the production Apply test")
    # AMReX_CUDA is a plain variable in AMReX's own add_subdirectory scope and
    # is empty here; WarpX_COMPUTE is the cache option that is actually in
    # scope. Without this the CUDA fixtures lose amrex.the_arena_is_managed=1
    # and abort on the hybrid startup guard.
    set(RREA_TEST_CUDA 0)
    if(WarpX_COMPUTE STREQUAL "CUDA")
        set(RREA_TEST_CUDA 1)
    endif()
    set(RREA_WARPX_TEST_DIR "${{CMAKE_BINARY_DIR}}/rrea_warpx_test")
    file(MAKE_DIRECTORY
        "${{RREA_WARPX_TEST_DIR}}/actual_continuity"
        "${{RREA_WARPX_TEST_DIR}}/transport_apply")
    add_test(
        NAME rrea_warpx_actual_continuity
        COMMAND ${{CMAKE_COMMAND}}
            "-DWARPX_EXE=$<TARGET_FILE:app_rz>"
            "-DRREA_CUDA=${{RREA_TEST_CUDA}}"
            "-DINPUT=${{RREA_PROJECT_ROOT}}/cases/rrea/amrex_smoke/actual_continuity/inputs"
            "-DOUTPUT_DIR=${{RREA_WARPX_TEST_DIR}}/actual_continuity"
            "-DWORKING_DIR=${{RREA_PROJECT_ROOT}}"
            "-DPASS_REGEX=RREA_ACTUAL_CONTINUITY_FIXTURE_PASS"
            -P "${{RREA_AMREX_OVERLAY}}/cmake/RunWarpXSmoke.cmake"
    )
    set_tests_properties(
        rrea_warpx_actual_continuity
        PROPERTIES
            WORKING_DIRECTORY "${{RREA_PROJECT_ROOT}}"
            ENVIRONMENT "OMP_NUM_THREADS=2"
            TIMEOUT 120
    )
    if(WarpX_MPI AND MPIEXEC_EXECUTABLE)
        add_test(
            NAME rrea_warpx_actual_continuity_mpi4
            COMMAND ${{CMAKE_COMMAND}}
                "-DWARPX_EXE=$<TARGET_FILE:app_rz>"
                "-DRREA_CUDA=${{RREA_TEST_CUDA}}"
                "-DINPUT=${{RREA_PROJECT_ROOT}}/cases/rrea/amrex_smoke/actual_continuity/inputs"
                "-DOUTPUT_DIR=${{RREA_WARPX_TEST_DIR}}/actual_continuity_mpi4"
                "-DWORKING_DIR=${{RREA_PROJECT_ROOT}}"
                "-DPASS_REGEX=RREA_ACTUAL_CONTINUITY_FIXTURE_PASS"
                "-DMPIEXEC=${{MPIEXEC_EXECUTABLE}}"
                "-DMPI_NUMPROC_FLAG=${{MPIEXEC_NUMPROC_FLAG}}"
                "-DMPI_NUMPROCS=4"
                -P "${{RREA_AMREX_OVERLAY}}/cmake/RunWarpXSmoke.cmake"
        )
        set_tests_properties(
            rrea_warpx_actual_continuity_mpi4
            PROPERTIES
                WORKING_DIRECTORY "${{RREA_PROJECT_ROOT}}"
                ENVIRONMENT "OMP_NUM_THREADS=1"
                PROCESSORS 4
                TIMEOUT 120
        )
    endif()
    add_test(
        NAME rrea_warpx_transport_apply
        COMMAND ${{CMAKE_COMMAND}}
            "-DWARPX_EXE=$<TARGET_FILE:app_rz>"
            "-DRREA_CUDA=${{RREA_TEST_CUDA}}"
            "-DINPUT=${{RREA_PROJECT_ROOT}}/cases/rrea/amrex_smoke/transport_apply/inputs"
            "-DOUTPUT_DIR=${{RREA_WARPX_TEST_DIR}}/transport_apply"
            "-DWORKING_DIR=${{RREA_PROJECT_ROOT}}"
            "-DPASS_REGEX=RREA Maxwell field advance"
            "-DTRANSPORT_CONFIG=${{RREA_PRODUCTION_TRANSPORT_CONFIG}}"
            "-DSEED_SCHEDULE=${{RREA_PROJECT_ROOT}}/cases/rrea/amrex_smoke/transport_apply/one_seed.csv"
            -P "${{RREA_AMREX_OVERLAY}}/cmake/RunWarpXSmoke.cmake"
    )
    set_tests_properties(
        rrea_warpx_transport_apply
        PROPERTIES
            WORKING_DIRECTORY "${{RREA_PROJECT_ROOT}}"
            ENVIRONMENT "OMP_NUM_THREADS=2"
            TIMEOUT 120
    )
endif()
{runtime_test_end_marker}
"""
    text = cmake_path.read_text(encoding="utf-8")
    if runtime_test_marker not in text:
        cmake_path.write_text(text + runtime_test_block, encoding="utf-8")
    else:
        begin = text.index(runtime_test_marker)
        end = text.index(runtime_test_end_marker, begin) + len(
            runtime_test_end_marker
        )
        cmake_path.write_text(
            text[:begin] + runtime_test_block.strip() + text[end:],
            encoding="utf-8",
        )


def patch_space_charge_solver(path: Path) -> None:
    include_block = """#ifdef WARPX_RREA
#include "RreaWarpXCoupling.H"
#endif
"""
    replace_once(
        path,
        '#include "WarpX.H"\n',
        '#include "WarpX.H"\n\n' + include_block,
    )

    replace_once(
        path,
        "void WarpX::ComputeSpaceChargeField (bool const reset_fields)\n",
        "void WarpX::ComputeSpaceChargeField (\n"
        "    bool const reset_fields\n"
        "#ifdef WARPX_RREA\n"
        "    , rrea::warpx::SpaceChargeSolvePurpose const purpose\n"
        "#endif\n"
        ")\n",
    )

    hook = f"""    {CPP_BEGIN_MARKER}
#ifdef WARPX_RREA
    if (rrea::warpx::RreaWarpXCoupling::GetInstance().BeforeFieldSolve(
            *this, getdt(0), getistep(0), gett_new(0), purpose)) {{
        return;
    }}
#endif
    {CPP_END_MARKER}

"""
    if CPP_BEGIN_MARKER in path.read_text(encoding="utf-8"):
        replace_once(
            path,
            "            *this, getdt(0), getistep(0), gett_new(0))) {\n",
            "            *this, getdt(0), getistep(0), gett_new(0), purpose)) {\n",
        )
        return
    needle = "    m_electrostatic_solver->ComputeSpaceChargeField(\n"
    replace_once(path, needle, hook + needle)


def patch_evolve(path: Path) -> None:
    include_block = """#ifdef WARPX_RREA
#include "RreaWarpXCoupling.H"
#endif
"""
    replace_once(
        path,
        '#include "WarpX.H"\n',
        '#include "WarpX.H"\n\n' + include_block,
    )

    injection_hook = f"""        {CPP_BEGIN_MARKER}
#ifdef WARPX_RREA
        rrea::warpx::RreaWarpXCoupling::GetInstance().InjectScheduledSeeds(
            *this, step, cur_time, dt[0]);
#endif
        {CPP_END_MARKER}
"""
    text = path.read_text(encoding="utf-8")
    if "InjectScheduledSeeds(" not in text:
        needle = '        ExecutePythonCallback("particleinjection");\n'
        replace_once(path, needle, needle + "\n" + injection_hook)

    transport_hook = """#ifdef WARPX_RREA
        rrea::warpx::RreaWarpXCoupling::GetInstance().CaptureRreaPositionsBeforePush(
            *this, step);
#endif
        OneStep(cur_time, dt[0], step);
#ifdef WARPX_RREA
        rrea::warpx::RreaWarpXCoupling::GetInstance().AfterParticlePushBeforeBoundaries(
            *this, step, cur_time, dt[0]);
#endif
"""
    if "AfterParticlePushBeforeBoundaries(" not in path.read_text(encoding="utf-8"):
        replace_once(
            path,
            "        OneStep(cur_time, dt[0], step);\n",
            transport_hook,
        )

    replace_once(
        path,
        "                ComputeSpaceChargeField( reset_fields );\n",
        "#ifdef WARPX_RREA\n"
        "                ComputeSpaceChargeField(\n"
        "                    reset_fields,\n"
        "                    rrea::warpx::SpaceChargeSolvePurpose::Evolution );\n"
        "#else\n"
        "                ComputeSpaceChargeField( reset_fields );\n"
        "#endif\n",
    )

    # This ordering is a physics contract, not just a convenient insertion
    # point.  RREA must see the pushed chord while the escaping parent still
    # exists, and generic resampling must not run before the atomic secondary
    # groups have been committed.  Fail while preparing the source if a WarpX
    # update moves any of these anchors around the hooks.
    text = path.read_text(encoding="utf-8")
    ordered_markers = (
        "InjectScheduledSeeds(",
        "CaptureRreaPositionsBeforePush(",
        "        OneStep(cur_time, dt[0], step);",
        "AfterParticlePushBeforeBoundaries(",
        "        mypc->doResampling(",
        "        HandleParticlesAtBoundaries(step, cur_time, num_moved);",
        "                    rrea::warpx::SpaceChargeSolvePurpose::Evolution );",
    )
    positions = []
    for marker in ordered_markers:
        position = text.find(marker)
        if position < 0:
            raise SystemExit(
                f"Could not verify RREA evolution ordering in {path}: "
                f"missing {marker!r}"
            )
        positions.append(position)
    if positions != sorted(positions) or len(set(positions)) != len(positions):
        rendered = ", ".join(
            f"{marker!r}@{position}"
            for marker, position in zip(ordered_markers, positions, strict=False)
        )
        raise SystemExit(
            "RREA evolution ordering contract is not satisfied in "
            f"{path}: {rendered}"
        )


def patch_warpx_header(path: Path) -> None:
    declaration = """#ifdef WARPX_RREA
namespace rrea::warpx {
enum class SpaceChargeSolvePurpose;
}
#endif

"""
    replace_once(
        path,
        "class WARPX_EXPORT WarpX\n",
        declaration + "class WARPX_EXPORT WarpX\n",
    )
    replace_once(
        path,
        "    void ComputeSpaceChargeField (bool reset_fields);\n",
        "    void ComputeSpaceChargeField (\n"
        "        bool reset_fields\n"
        "#ifdef WARPX_RREA\n"
        "        , rrea::warpx::SpaceChargeSolvePurpose purpose\n"
        "#endif\n"
        "    );\n",
    )


def patch_initialization(path: Path) -> None:
    include_block = """#ifdef WARPX_RREA
#include "RreaWarpXCoupling.H"
#endif
"""
    replace_once(
        path,
        '#include "WarpX.H"\n',
        '#include "WarpX.H"\n\n' + include_block,
    )
    replace_once(
        path,
        "            ComputeSpaceChargeField(reset_fields);\n",
        "#ifdef WARPX_RREA\n"
        "            ComputeSpaceChargeField(\n"
        "                reset_fields,\n"
        "                rrea::warpx::SpaceChargeSolvePurpose::Initialization);\n"
        "#else\n"
        "            ComputeSpaceChargeField(reset_fields);\n"
        "#endif\n",
    )


def patch_checkpoint_writer(path: Path) -> None:
    include_block = """#ifdef WARPX_RREA
#include "RreaWarpXCoupling.H"
#endif
"""
    replace_once(
        path,
        '#include "WarpX.H"\n',
        '#include "WarpX.H"\n\n' + include_block,
    )

    hook = f"""    {CPP_BEGIN_MARKER}
#ifdef WARPX_RREA
    rrea::warpx::RreaWarpXCoupling::GetInstance().WriteCheckpoint(checkpointname);
#endif
    {CPP_END_MARKER}

"""
    if CPP_BEGIN_MARKER in path.read_text(encoding="utf-8"):
        return
    needle = "    WriteReducedDiagsData(checkpointname);\n"
    replace_once(path, needle, needle + "\n" + hook)


def patch_checkpoint_reader(path: Path) -> None:
    include_block = """#ifdef WARPX_RREA
#include "RreaWarpXCoupling.H"
#endif
"""
    replace_once(
        path,
        '#include "WarpX.H"\n',
        '#include "WarpX.H"\n\n' + include_block,
    )

    hook = f"""    {CPP_BEGIN_MARKER}
#ifdef WARPX_RREA
    rrea::warpx::RreaWarpXCoupling::GetInstance().ReadCheckpoint(restart_chkfile);
#endif
    {CPP_END_MARKER}

"""
    if CPP_BEGIN_MARKER in path.read_text(encoding="utf-8"):
        return
    needle = """    if (m_implicit_solver) {
        m_implicit_solver->Define(this, /*from_restart=*/true);
        m_implicit_solver->CreateParticleAttributes();
    }

}
"""
    replacement = """    if (m_implicit_solver) {
        m_implicit_solver->Define(this, /*from_restart=*/true);
        m_implicit_solver->CreateParticleAttributes();
    }

""" + hook + "}\n"
    replace_once(path, needle, replacement)


# The RZ staggering of E/J is the foundation every Maxwell-side mapping is
# designed against; the
# facts were verified directly in the pinned 26.06 source (WarpX.cpp, the
# WARPX_DIM_XZ/RZ branch) and the fluid faces are built by surroundingNodes,
# so fluid radial flux sits at Jz's staggering and axial at Jr's. If an
# upstream bump ever moves these, source preparation must fail loudly, not
# build a subtly misaligned coupling.
RZ_STAGGERING_MARKER = "defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)"
RZ_STAGGERING_PINS = (
    "Ex_nodal_flag = IntVect(0,1);",
    "Ey_nodal_flag = IntVect(1,1);",
    "Ez_nodal_flag = IntVect(1,0);",
    "jx_nodal_flag = IntVect(0,1);",
    "jy_nodal_flag = IntVect(1,1);",
    "jz_nodal_flag = IntVect(1,0);",
)


def verify_rz_staggering(warpx_source: Path) -> None:
    source = warpx_source / "Source" / "WarpX.cpp"
    text = source.read_text(encoding="utf-8", errors="replace")
    if RZ_STAGGERING_MARKER not in text:
        raise SystemExit(f"RZ staggering branch not found in {source}")
    start = text.index(RZ_STAGGERING_MARKER)
    block = text[start:start + 900]
    for pin in RZ_STAGGERING_PINS:
        if pin not in block:
            raise SystemExit(
                f"upstream RZ staggering changed ({pin!r} not found in "
                f"{source}); every RREA field mapping was designed against "
                "the pinned layout -- re-verify before building"
            )


def copy_overlay(overlay_dir: Path, output_dir: Path) -> None:
    overlay_source = overlay_dir / "warpx_overlay" / "Source" / "Rrea"
    if not overlay_source.is_dir():
        raise SystemExit(f"Missing RREA WarpX overlay source: {overlay_source}")
    target = output_dir / "Source" / "Rrea"
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(overlay_source, target)


def main() -> None:
    args = parse_args()
    warpx_source = resolve(args.warpx_source)
    output_dir = resolve(args.output_dir)
    overlay_dir = resolve(args.overlay_dir)

    if not (warpx_source / "CMakeLists.txt").is_file():
        raise SystemExit(f"Not a WarpX source tree: {warpx_source}")
    if not (overlay_dir / "CMakeLists.txt").is_file():
        raise SystemExit(f"Not a native_amrex_rrea overlay tree: {overlay_dir}")
    verify_rz_staggering(warpx_source)

    # --force removes output_dir recursively; never allow it to contain (or
    # live inside) an input tree, or the pristine sources would be deleted or
    # mutated.
    for label, source_root in (
        ("--warpx-source", warpx_source),
        ("--overlay-dir", overlay_dir),
    ):
        if output_dir == source_root or output_dir in source_root.parents:
            raise SystemExit(
                f"Refusing --output-dir {output_dir}: it contains the {label} tree"
            )
        if source_root in output_dir.parents:
            raise SystemExit(
                f"Refusing --output-dir {output_dir}: it is nested inside the {label} tree"
            )

    if args.refresh:
        if not (output_dir / "CMakeLists.txt").is_file():
            raise SystemExit(
                f"--refresh needs an existing prepared tree at {output_dir}"
            )
    else:
        copy_tree(warpx_source, output_dir, args.force)
    copy_overlay(overlay_dir, output_dir)
    patch_top_cmake(output_dir / "CMakeLists.txt")
    patch_warpx_header(output_dir / "Source" / "WarpX.H")
    patch_space_charge_solver(output_dir / "Source" / "FieldSolver" / "WarpXSolveFieldsES.cpp")
    patch_evolve(output_dir / "Source" / "Evolve" / "WarpXEvolve.cpp")
    patch_initialization(output_dir / "Source" / "Initialization" / "WarpXInitData.cpp")
    patch_checkpoint_writer(
        output_dir / "Source" / "Diagnostics" / "FlushFormats" / "FlushFormatCheckpoint.cpp"
    )
    patch_checkpoint_reader(output_dir / "Source" / "Diagnostics" / "WarpXIO.cpp")

    print(f"Prepared RREA WarpX source: {output_dir}")
    print(f"Overlay source: {overlay_dir}")
    print("Configure with -DRREA_AMREX_OVERLAY=<repo>/native_amrex_rrea")


if __name__ == "__main__":
    main()
