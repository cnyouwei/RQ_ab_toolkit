cmake_minimum_required(VERSION 3.20)

project(wck_solver LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(WCK_BUILD_TESTS "Build wck unit tests" ON)

find_package(Threads REQUIRED)

add_library(wcklib STATIC
    src/common/mini_json.cpp
    src/common/distributions.cpp
    src/common/json_config.cpp
    src/wck_table/solver.cpp
    src/wck_table/tridiag.cpp
    src/rq_calibration/w_table_interpolator.cpp
    src/rq_calibration/b_calibration.cpp
    src/idw_sim/effective_idw_sim_config.cpp
    src/idw_sim/effective_idw_sim.cpp
    src/workload_sim/workload_sim_config.cpp
    src/workload_sim/workload_sim.cpp
    src/workload_sim/tandem_workload_sim_config.cpp
    src/workload_sim/tandem_workload_sim.cpp
)

target_include_directories(wcklib
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(wcklib PUBLIC Threads::Threads)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(wcklib PRIVATE -Wall -Wextra -Wpedantic -Wconversion)
endif()

add_executable(wck
    src/wck_table/main.cpp
)

target_link_libraries(wck PRIVATE wcklib)

add_executable(wck_sweep
    src/wck_table/sweep_main.cpp
)

target_link_libraries(wck_sweep PRIVATE wcklib)

add_executable(idw_sim
    src/idw_sim/effective_idw_sim_main.cpp
)

target_link_libraries(idw_sim PRIVATE wcklib)

add_executable(wck_calibrate_b
    src/rq_calibration/calibrate_b_main.cpp
)

target_link_libraries(wck_calibrate_b PRIVATE wcklib)

add_executable(workload_mc
    src/workload_sim/workload_mc_main.cpp
)

target_link_libraries(workload_mc PRIVATE wcklib)

if(WCK_BUILD_TESTS)
    enable_testing()

    add_executable(wck_tests
        tests/test_solver.cpp
        tests/test_effective_idw_sim.cpp
        tests/test_b_calibration.cpp
        tests/test_workload_sim.cpp
        tests/test_tandem_workload_sim.cpp
    )

    target_link_libraries(wck_tests PRIVATE wcklib)
    target_compile_definitions(wck_tests PRIVATE
        WCK_CLI_PATH="$<TARGET_FILE:wck>"
        WCK_SWEEP_CLI_PATH="$<TARGET_FILE:wck_sweep>"
        WCK_IDW_SIM_CLI_PATH="$<TARGET_FILE:idw_sim>"
        WCK_CALIBRATE_B_CLI_PATH="$<TARGET_FILE:wck_calibrate_b>"
        WCK_WORKLOAD_MC_CLI_PATH="$<TARGET_FILE:workload_mc>"
        WCK_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
    )

    add_test(NAME wck_tests COMMAND wck_tests)
endif()
