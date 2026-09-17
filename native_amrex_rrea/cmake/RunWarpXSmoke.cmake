if(NOT DEFINED WARPX_EXE OR NOT DEFINED INPUT OR NOT DEFINED OUTPUT_DIR
   OR NOT DEFINED WORKING_DIR OR NOT DEFINED PASS_REGEX)
    message(FATAL_ERROR "RunWarpXSmoke.cmake is missing a required argument")
endif()

file(REMOVE_RECURSE "${OUTPUT_DIR}")
file(MAKE_DIRECTORY "${OUTPUT_DIR}")
set(command "${WARPX_EXE}" "${INPUT}" "rrea.output_dir=${OUTPUT_DIR}")
if(DEFINED MPIEXEC)
    if(NOT DEFINED MPI_NUMPROC_FLAG OR NOT DEFINED MPI_NUMPROCS)
        message(FATAL_ERROR "MPI WarpX smoke is missing rank-count arguments")
    endif()
    list(PREPEND command
        "${MPIEXEC}" "--map-by" ":OVERSUBSCRIBE"
        "${MPI_NUMPROC_FLAG}" "${MPI_NUMPROCS}")
endif()
if(DEFINED TRANSPORT_CONFIG)
    list(APPEND command
        "rrea.interaction_table_config=${TRANSPORT_CONFIG}")
endif()
if(DEFINED SEED_SCHEDULE)
    list(APPEND command
        "rrea.seed_schedule_path=${SEED_SCHEDULE}"
        "rrea_electrons.source_schedule_file=${SEED_SCHEDULE}")
endif()

execute_process(
    COMMAND ${command}
    WORKING_DIRECTORY "${WORKING_DIR}"
    RESULT_VARIABLE result
    OUTPUT_VARIABLE output
    ERROR_VARIABLE error)
message("${output}")
if(NOT error STREQUAL "")
    message("${error}")
endif()
if(NOT result EQUAL 0)
    message(FATAL_ERROR "WarpX smoke exited ${result}")
endif()
if(NOT output MATCHES "${PASS_REGEX}")
    message(FATAL_ERROR "WarpX smoke did not emit ${PASS_REGEX}")
endif()
