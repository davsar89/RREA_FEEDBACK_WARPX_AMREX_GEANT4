# The video-v4 frame stream deflates each frame as it is written
# (RreaVideoFrameCodec.H). zlib is already present in the WarpX/AMReX
# toolchain, avoiding another codec dependency.
find_package(ZLIB REQUIRED)

# WarpX intentionally keeps AMReX host-side OpenMP off in CUDA builds.
# Retain a threaded CPU reference/fallback independently of CUDA transport.
# Do not redefine AMREX_USE_OMP or change AMReX's GPU launch policy.
option(RREA_CUDA_HOST_OMP "Thread the CUDA build's CPU reference/fallback loops" ON)
if(WarpX_COMPUTE STREQUAL "CUDA" AND RREA_CUDA_HOST_OMP)
    find_package(OpenMP REQUIRED COMPONENTS CXX)
    separate_arguments(rrea_host_omp_flags NATIVE_COMMAND "${OpenMP_CXX_FLAGS}")
endif()

foreach(D IN LISTS WarpX_DIMS)
    warpx_set_suffix_dims(SD ${D})

    target_sources(lib_${SD}
        PRIVATE
            RreaActualContinuityFixture.cpp
            RreaCouplingCheckpoint.cpp
            RreaCouplingConfig.cpp
            RreaCouplingDiagnostics.cpp
            RreaCouplingPlaneWriter.cpp
            RreaCouplingPopulation.cpp
            RreaCouplingSources.cpp
            RreaCouplingVideoFrame.cpp
            RreaElectronTransport.cpp
            RreaNewbornAdvance.cpp
            RreaParticleInteraction.cpp
            RreaPhotonTransport.cpp
            RreaPositronTransport.cpp
            RreaRng.cpp
            RreaSecondaryCommit.cpp
            RreaWarpXCoupling.cpp
    )

    target_include_directories(lib_${SD}
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}
            ${RREA_AMREX_OVERLAY}/include
    )

    target_link_libraries(lib_${SD}
        PRIVATE
            rrea_amrex_field_solver
            ZLIB::ZLIB
    )

    target_compile_definitions(lib_${SD}
        PRIVATE
            WARPX_RREA=1
    )

    if(WarpX_COMPUTE STREQUAL "CUDA" AND RREA_CUDA_HOST_OMP)
        target_link_libraries(lib_${SD} PRIVATE OpenMP::OpenMP_CXX)
        target_compile_definitions(lib_${SD} PRIVATE RREA_USE_HOST_OMP=1)
        foreach(flag IN LISTS rrea_host_omp_flags)
            if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
                target_compile_options(lib_${SD} PRIVATE
                    "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${flag}>")
            else()
                target_compile_options(lib_${SD} PRIVATE
                    "$<$<COMPILE_LANGUAGE:CUDA>:${flag}>")
            endif()
        endforeach()
    endif()

    # Revisit the target after appending the RREA .cpp sources.
    rrea_configure_gpu_target(lib_${SD})

endforeach()
