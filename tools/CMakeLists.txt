cmake_minimum_required(VERSION 3.18)
# tools/ can also be configured directly, without loading the MD build.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(MetadToyCLI LANGUAGES NONE)
endif()
if(NOT UNIX)
    message(FATAL_ERROR "The CMake CLI installer currently supports Linux/WSL and macOS.")
endif()
find_package(Python3 3.9 REQUIRED COMPONENTS Interpreter)
execute_process(COMMAND "${Python3_EXECUTABLE}" -m pip --version
    RESULT_VARIABLE pip_status OUTPUT_QUIET ERROR_QUIET)
if(NOT pip_status EQUAL 0)
    message(FATAL_ERROR "CLI packaging requires pip. Select a prepared Python with -DPython3_EXECUTABLE=/path/to/python.")
endif()

# These relative directories are resolved against the install-time prefix.
set(ZQC_CLI_BINDIR "bin" CACHE STRING "CLI launcher directory relative to the install prefix")
set(ZQC_CLI_RUNTIME_DIR "libexec/metadtoy" CACHE STRING "Private CLI runtime directory relative to the install prefix")
set(cli_wheel_dir "${CMAKE_CURRENT_BINARY_DIR}/cli-wheel")
set(cli_installer "${CMAKE_CURRENT_SOURCE_DIR}/install_cli.py")
file(GLOB cli_sources CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/metadtoy_cli/*.py")
add_custom_command(OUTPUT "${cli_wheel_dir}/ready.stamp"
    COMMAND "${Python3_EXECUTABLE}" "${cli_installer}" build
            --source "${CMAKE_CURRENT_SOURCE_DIR}" --output "${cli_wheel_dir}"
    COMMAND "${CMAKE_COMMAND}" -E touch "${cli_wheel_dir}/ready.stamp"
    DEPENDS ${cli_sources} "${cli_installer}"
            "${CMAKE_CURRENT_SOURCE_DIR}/pyproject.toml"
            "${CMAKE_CURRENT_SOURCE_DIR}/requirements-analysis.txt"
    BYPRODUCTS "${cli_wheel_dir}/manifest.json"
    COMMENT "Packaging metadtoy CLI and its pip bootstrap"
    VERBATIM)
add_custom_target(metadtoy-cli ALL DEPENDS "${cli_wheel_dir}/ready.stamp")

configure_file("${CMAKE_CURRENT_SOURCE_DIR}/install_cli.cmake.in"
               "${CMAKE_CURRENT_BINARY_DIR}/install_cli.cmake" @ONLY)
install(SCRIPT "${CMAKE_CURRENT_BINARY_DIR}/install_cli.cmake" COMPONENT cli)
message(STATUS "MetadToy CLI: Python=${Python3_EXECUTABLE}; install with --component cli")
