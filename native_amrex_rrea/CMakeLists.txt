cmake_minimum_required(VERSION 3.23)

project(RreaAmrexFieldSolver LANGUAGES C CXX)

include(GNUInstallDirs)

if(NOT TARGET AMReX::amrex)
    find_package(AMReX REQUIRED)
endif()

# CUDA is an execution backend, not just a link dependency. Every translation
# unit that instantiates the shared kernel headers must use the CUDA compiler,
# and every consumer must see the same managed-vector ABI.
if(AMReX_GPU_BACKEND STREQUAL "HIP" OR AMReX_GPU_BACKEND STREQUAL "SYCL")
    message(FATAL_ERROR "The RREA hybrid GPU path currently supports CUDA only")
endif()
if(AMReX_GPU_BACKEND STREQUAL "CUDA" OR AMReX_CUDA)
    enable_language(CUDA)
    if(NOT COMMAND setup_target_for_cuda_compilation)
        include(AMReXTargetHelpers)
    endif()
endif()

function(rrea_configure_gpu_target target_name)
    if(AMReX_GPU_BACKEND STREQUAL "CUDA" OR AMReX_CUDA)
        setup_target_for_cuda_compilation(${target_name})
        target_compile_features(${target_name} PUBLIC cuda_std_20)
        target_compile_definitions(${target_name} PUBLIC RREA_USE_CUDA=1)
        # Keep scientific checks meaningful; the CPU IPO switch does not
        # configure CUDA device link-time optimization.
        set_target_properties(${target_name} PROPERTIES
            CUDA_STANDARD_REQUIRED ON
            INTERPROCEDURAL_OPTIMIZATION OFF)
    endif()
endfunction()

# The production transport tables ship as RREATBL containers (see
# include/rrea/RreaTableContainer.H); inflating them is the only use.  The
# WarpX overlay already links zlib for the video-v4 frame codec.
find_package(ZLIB REQUIRED)

option(
    RREA_AMREX_ENABLE_IPO
    "Build the standalone RREA library/probes with interprocedural optimization"
    OFF
)
if(RREA_AMREX_ENABLE_IPO)
    if(AMReX_GPU_BACKEND STREQUAL "CUDA" OR AMReX_CUDA)
        message(FATAL_ERROR "Set RREA_AMREX_ENABLE_IPO=OFF for the CUDA build")
    endif()
    # CMP0138 makes CheckIPOSupported exercise the caller's actual compiler
    # flags (notably the OLIVIA -pthread contract) when the running CMake knows
    # that policy.  CMake 3.23 remains supported for non-IPO/local builds.
    if(POLICY CMP0138)
        cmake_policy(SET CMP0138 NEW)
    endif()
    include(CheckIPOSupported)
    check_ipo_supported(
        RESULT RREA_AMREX_IPO_SUPPORTED
        OUTPUT RREA_AMREX_IPO_DIAGNOSTIC
        LANGUAGES CXX
    )
    if(NOT RREA_AMREX_IPO_SUPPORTED)
        message(FATAL_ERROR
            "RREA_AMREX_ENABLE_IPO=ON but compiler IPO is unavailable: "
            "${RREA_AMREX_IPO_DIAGNOSTIC}")
    endif()
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
    message(STATUS
        "RREA IPO enabled with ${CMAKE_CXX_COMPILER_ID} "
        "${CMAKE_CXX_COMPILER_VERSION}; compiler-ar=${CMAKE_CXX_COMPILER_AR}; "
        "compiler-ranlib=${CMAKE_CXX_COMPILER_RANLIB}")
endif()

function(rrea_require_target_ipo target_name)
    if(NOT RREA_AMREX_ENABLE_IPO)
        return()
    endif()
    get_target_property(rrea_target_ipo "${target_name}" INTERPROCEDURAL_OPTIMIZATION)
    if(NOT rrea_target_ipo)
        message(FATAL_ERROR
            "RREA_AMREX_ENABLE_IPO=ON but target ${target_name} did not inherit IPO")
    endif()
    message(STATUS "RREA IPO target enabled: ${target_name}")
endfunction()

add_library(rrea_amrex_field_solver STATIC
    src/RreaAltitudeProfile.cpp
    src/RreaEventAccumulator.cpp
    src/RreaFieldInitializer.cpp
    src/RreaAmrexAdvance.cpp
    src/RreaDebugOptions.cpp
    src/RreaInteractionTables.cpp
    src/RreaTransportPath.cpp
    src/RreaSeedSchedule.cpp
    src/LowEnergyFluidState.cpp
    src/LowEnergyFluidFaceTransport.cpp
)

target_include_directories(rrea_amrex_field_solver
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_link_libraries(rrea_amrex_field_solver
    PUBLIC
        AMReX::amrex
        ZLIB::ZLIB
)

target_compile_features(rrea_amrex_field_solver PUBLIC cxx_std_17)
set_target_properties(rrea_amrex_field_solver PROPERTIES POSITION_INDEPENDENT_CODE ON)
rrea_require_target_ipo(rrea_amrex_field_solver)
rrea_configure_gpu_target(rrea_amrex_field_solver)

target_compile_definitions(rrea_amrex_field_solver
    PUBLIC
        RREA_AMREX_FIELD_SOLVER=1
)

option(RREA_AMREX_BUILD_TESTS "Build lightweight RREA AMReX smoke tests" OFF)
option(RREA_AMREX_INSTALL "Install the RREA AMReX overlay library" ON)

if(RREA_AMREX_BUILD_TESTS)
    # Register equation- and component-level native smokes.  The prepared
    # WarpX top level enables CTest and separately registers a real
    # production-executable continuity test.
    enable_testing()

    set(RREA_SMOKE_CONFIG
        "${CMAKE_CURRENT_SOURCE_DIR}/../rrea_transport_tables/schema6/production/transport_physics.json"
        CACHE FILEPATH
        "Schema-6 transport configuration handed to table-loading smokes")

    # These tests read transport_physics.json from argv[1]; every other smoke takes no
    # arguments and treats an unexpected one as a usage error.
    set(RREA_CONFIG_SMOKES
        rrea_interaction_tables_smoke
        rrea_scattering_sampler_smoke
        rrea_kinetic_engine_smoke
    )

    # Every smoke has the same shape: one translation unit, the overlay include
    # path, and the solver library.
    function(rrea_add_smoke name)
        add_executable(${name} ${ARGN})
        target_include_directories(${name}
            PRIVATE
                ${CMAKE_CURRENT_SOURCE_DIR}/warpx_overlay/Source/Rrea
        )
        target_link_libraries(${name}
            PRIVATE
                rrea_amrex_field_solver
        )
        target_compile_features(${name} PRIVATE cxx_std_17)
        rrea_configure_gpu_target(${name})
        if(${name} IN_LIST RREA_CONFIG_SMOKES)
            add_test(NAME ${name} COMMAND ${name} "${RREA_SMOKE_CONFIG}")
        else()
            add_test(NAME ${name} COMMAND ${name})
        endif()
    endfunction()

    rrea_add_smoke(rrea_seed_schedule_smoke src/RreaSeedScheduleSmoke.cpp)
    rrea_add_smoke(rrea_table_container_smoke src/RreaTableContainerSmoke.cpp)
    rrea_add_smoke(rrea_interaction_tables_smoke src/RreaInteractionTablesSmoke.cpp)
    rrea_add_smoke(rrea_transport_path_smoke src/RreaTransportPathSmoke.cpp)
    rrea_add_smoke(rrea_gauss_legendre8_smoke src/RreaGaussLegendre8Smoke.cpp)
    rrea_add_smoke(rrea_altitude_profile_smoke src/RreaAltitudeProfileSmoke.cpp)
    rrea_add_smoke(rrea_field_gather_stencil_smoke src/RreaFieldGatherStencilSmoke.cpp)
    rrea_add_smoke(rrea_secondary_group_smoke src/RreaSecondaryGroupSmoke.cpp)
    rrea_add_smoke(rrea_charged_interleave_smoke
        warpx_overlay/Source/Rrea/RreaChargedInterleaveSmoke.cpp)
    rrea_add_smoke(rrea_low_energy_fluid_smoke src/LowEnergyFluidStateSmoke.cpp)
    rrea_add_smoke(rrea_low_energy_face_transport_smoke
        src/LowEnergyFluidFaceTransportSmoke.cpp)
    rrea_add_smoke(rrea_segment_deposition_smoke src/RreaSegmentDepositionSmoke.cpp)
    rrea_add_smoke(rrea_gpu_primitives_smoke src/RreaGpuPrimitivesSmoke.cpp)
    rrea_add_smoke(rrea_kinetic_storage_smoke src/RreaKineticStorageSmoke.cpp)
    rrea_add_smoke(rrea_kinetic_engine_smoke src/RreaKineticEngineSmoke.cpp)
    rrea_add_smoke(rrea_population_gpu_smoke src/RreaPopulationGpuSmoke.cpp)
    rrea_add_smoke(rrea_maxwell_tm_rz_smoke src/RreaMaxwellTMRZSmoke.cpp)
    rrea_add_smoke(rrea_amrex_advance_smoke src/RreaAmrexAdvanceSmoke.cpp)
    rrea_add_smoke(rrea_scattering_sampler_smoke src/RreaScatteringSamplerSmoke.cpp
        warpx_overlay/Source/Rrea/RreaRng.cpp)
    rrea_add_smoke(rrea_event_accumulator_smoke src/RreaEventAccumulatorSmoke.cpp)
    rrea_add_smoke(rrea_reduced_observables_smoke src/RreaReducedObservablesSmoke.cpp)

    if(AMReX_MPI AND MPIEXEC_EXECUTABLE)
        add_test(
            NAME rrea_low_energy_face_transport_mpi4
            COMMAND
                ${MPIEXEC_EXECUTABLE} --map-by :OVERSUBSCRIBE
                ${MPIEXEC_NUMPROC_FLAG} 4
                ${MPIEXEC_PREFLAGS}
                $<TARGET_FILE:rrea_low_energy_face_transport_smoke>
                ${MPIEXEC_POSTFLAGS}
        )
        set_tests_properties(
            rrea_low_energy_face_transport_mpi4
            PROPERTIES
                ENVIRONMENT "OMP_NUM_THREADS=1"
                PROCESSORS 4
                TIMEOUT 120
        )
        add_test(
            NAME rrea_amrex_advance_mpi4
            COMMAND
                ${MPIEXEC_EXECUTABLE} --map-by :OVERSUBSCRIBE
                ${MPIEXEC_NUMPROC_FLAG} 4
                ${MPIEXEC_PREFLAGS}
                $<TARGET_FILE:rrea_amrex_advance_smoke>
                ${MPIEXEC_POSTFLAGS}
        )
        set_tests_properties(
            rrea_amrex_advance_mpi4
            PROPERTIES
                ENVIRONMENT "OMP_NUM_THREADS=1"
                PROCESSORS 4
                TIMEOUT 120
        )
    endif()

    # A CLI printer rather than a test: it asserts nothing, so it is built and
    # carries the representative IPO check but is never registered with ctest.
    add_executable(rrea_interaction_table_probe src/RreaInteractionTableProbe.cpp)
    target_link_libraries(rrea_interaction_table_probe
        PRIVATE
            rrea_amrex_field_solver
    )
    target_compile_features(rrea_interaction_table_probe PRIVATE cxx_std_17)
    rrea_require_target_ipo(rrea_interaction_table_probe)
    rrea_configure_gpu_target(rrea_interaction_table_probe)

    # Compiles the WarpX-overlay counter RNG directly; deliberately standalone
    # (no AMReX dependency) so the stream pins run anywhere.
    add_executable(rrea_rng_endpoint_smoke
        src/RreaRngEndpointSmoke.cpp
        warpx_overlay/Source/Rrea/RreaRng.cpp
    )
    target_include_directories(rrea_rng_endpoint_smoke
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/warpx_overlay/Source/Rrea
            # RreaUniformStream is header-only, so pinning it here keeps this
            # target free of the AMReX dependency.
            ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    target_compile_features(rrea_rng_endpoint_smoke PRIVATE cxx_std_17)
    add_test(NAME rrea_rng_endpoint_smoke COMMAND rrea_rng_endpoint_smoke)

    # Charge-exact population thinning is another standalone primitive.  Link
    # only the counter RNG used for endpoint selection; no AMReX state is
    # needed to test the four deposited CIC moments and estimator contract.
    add_executable(rrea_cic_resample_smoke
        src/RreaCicResampleSmoke.cpp
        warpx_overlay/Source/Rrea/RreaRng.cpp
    )
    target_include_directories(rrea_cic_resample_smoke
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/warpx_overlay/Source/Rrea
            ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    target_compile_features(rrea_cic_resample_smoke PRIVATE cxx_std_17)
    add_test(NAME rrea_cic_resample_smoke COMMAND rrea_cic_resample_smoke)

endif()

if(RREA_AMREX_INSTALL)
    install(
        TARGETS rrea_amrex_field_solver
        EXPORT RreaAmrexFieldSolverTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )

    install(
        DIRECTORY include/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    )

    install(
        EXPORT RreaAmrexFieldSolverTargets
        NAMESPACE rrea::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/RreaAmrexFieldSolver
    )
endif()
