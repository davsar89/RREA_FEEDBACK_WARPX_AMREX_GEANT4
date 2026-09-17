# The video-v4 frame stream deflates each frame as it is written
# (RreaVideoFrameCodec.H). zlib is already present in the WarpX/AMReX
# toolchain, avoiding another codec dependency.
find_package(ZLIB REQUIRED)

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

endforeach()
