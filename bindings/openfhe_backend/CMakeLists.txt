# ==============================================================================
# CKKS OpenFHE Backend - CMake Build Configuration
# ==============================================================================
# Build pybind11 bindings for OpenFHE CKKS functionality (GPU version)
# ==============================================================================

cmake_minimum_required(VERSION 3.18)
project(ckks_openfhe_backend LANGUAGES CXX CUDA)

# ==============================================================================
# C++ Standard and Global Settings
# ==============================================================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ==============================================================================
# Compiler Warning Configuration
# ==============================================================================
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(
        -Wno-deprecated-declarations
        -Wno-unused-parameter
        -Wno-sign-compare
        -Wno-missing-field-initializers
    )
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unknown-pragmas")
endif()

if(MSVC)
    add_compile_options(
        /wd4267
        /wd4244
        /wd4996
        /wd4100
    )
endif()

# ==============================================================================
# Python and pybind11 Setup
# ==============================================================================
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(OpenMP REQUIRED)

find_package(pybind11 CONFIG QUIET)
if(NOT pybind11_FOUND)
    message(STATUS "pybind11 not found via CMake, trying Python module...")
    execute_process(
        COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
        OUTPUT_VARIABLE _pybind11_dir
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE _pybind11_result
    )
    if(_pybind11_result EQUAL 0 AND EXISTS "${_pybind11_dir}")
        list(APPEND CMAKE_PREFIX_PATH "${_pybind11_dir}")
        find_package(pybind11 CONFIG REQUIRED)
    else()
        message(FATAL_ERROR
            "Could not find pybind11. Install it with:\n"
            "  pip install pybind11\n"
            "Or provide pybind11_DIR to CMake."
        )
    endif()
endif()
message(STATUS "Found pybind11: ${pybind11_VERSION}")

# ==============================================================================
# OpenFHE GPU Configuration
# ==============================================================================
set(OPENFHE_GPU_ROOT "${CMAKE_SOURCE_DIR}/../../openfhe-gpu-public"
    CACHE PATH "OpenFHE GPU source root")
set(OPENFHE_GPU_BUILD_DIR "${OPENFHE_GPU_ROOT}/build"
    CACHE PATH "OpenFHE GPU build directory")

if(NOT EXISTS "${OPENFHE_GPU_ROOT}")
    message(FATAL_ERROR
        "OpenFHE GPU root not found at: ${OPENFHE_GPU_ROOT}\n"
        "Override with: cmake -DOPENFHE_GPU_ROOT=/path/to/openfhe-gpu-public"
    )
endif()

set(OPENFHE_INCLUDE_DIRS
    "${OPENFHE_GPU_ROOT}/src"
    "${OPENFHE_GPU_ROOT}/src/pke/include"
    "${OPENFHE_GPU_ROOT}/src/core/include"
    "${OPENFHE_GPU_ROOT}/src/core/lib"
    "${OPENFHE_GPU_ROOT}/src/binfhe/include"
    "${OPENFHE_GPU_ROOT}/include"
    "${OPENFHE_GPU_ROOT}/third-party/cereal/include"
    "${OPENFHE_GPU_BUILD_DIR}/src/core"
    "${OPENFHE_GPU_BUILD_DIR}/_deps/rmm-src/include"
    "${OPENFHE_GPU_BUILD_DIR}/_deps/spdlog-src/include"
)

set(OPENFHE_LIBRARY_DIR "${OPENFHE_GPU_BUILD_DIR}/lib")

# ==============================================================================
# CUDA Configuration
# ==============================================================================
find_package(CUDAToolkit REQUIRED)

# ==============================================================================
# Main Extension Module (CPU Backend)
# ==============================================================================
add_library(ckks_openfhe_backend MODULE src/ckks_openfhe_backend.cpp)

target_compile_options(ckks_openfhe_backend PRIVATE -ftls-model=global-dynamic)

foreach(_inc ${OPENFHE_INCLUDE_DIRS})
    if(EXISTS "${_inc}")
        target_include_directories(ckks_openfhe_backend SYSTEM PRIVATE "${_inc}")
    endif()
endforeach()

# ==============================================================================
# GPU-Accelerated Extension Module
# ==============================================================================
add_library(ckks_openfhe_gpu_backend MODULE src/ckks_openfhe_gpu_backend.cpp)

target_compile_options(ckks_openfhe_gpu_backend PRIVATE -ftls-model=global-dynamic)

foreach(_inc ${OPENFHE_INCLUDE_DIRS})
    if(EXISTS "${_inc}")
        target_include_directories(ckks_openfhe_gpu_backend SYSTEM PRIVATE "${_inc}")
    endif()
endforeach()

# ==============================================================================
# Library Linking
# ==============================================================================
if(EXISTS "${OPENFHE_LIBRARY_DIR}")
    target_link_directories(ckks_openfhe_backend PRIVATE "${OPENFHE_LIBRARY_DIR}")
    target_link_directories(ckks_openfhe_gpu_backend PRIVATE "${OPENFHE_LIBRARY_DIR}")
endif()

target_link_libraries(ckks_openfhe_backend
    PRIVATE
        pybind11::module
        OPENFHEcore
        OPENFHEpke
        OPENFHEbinfhe
        DeviceFunctions
        CUDA::cudart
        OpenMP::OpenMP_CXX
        pthread
)

target_link_libraries(ckks_openfhe_gpu_backend
    PRIVATE
        pybind11::module
        OPENFHEcore
        OPENFHEpke
        OPENFHEbinfhe
        DeviceFunctions
        CUDA::cudart
        OpenMP::OpenMP_CXX
        pthread
)

# ==============================================================================
# RPATH Configuration
# ==============================================================================
set(_rpath 
    "${OPENFHE_LIBRARY_DIR}"
    "${CUDAToolkit_LIBRARY_DIR}"
)
list(REMOVE_DUPLICATES _rpath)

set_target_properties(ckks_openfhe_backend PROPERTIES
    BUILD_RPATH "${_rpath}"
    INSTALL_RPATH "${_rpath}"
)

set_target_properties(ckks_openfhe_gpu_backend PROPERTIES
    BUILD_RPATH "${_rpath}"
    INSTALL_RPATH "${_rpath}"
)

# ==============================================================================
# pybind11 Module Configuration
# ==============================================================================
pybind11_extension(ckks_openfhe_backend)
pybind11_strip(ckks_openfhe_backend)

pybind11_extension(ckks_openfhe_gpu_backend)
pybind11_strip(ckks_openfhe_gpu_backend)

# ==============================================================================
# Installation
# ==============================================================================
install(TARGETS ckks_openfhe_backend ckks_openfhe_gpu_backend
    LIBRARY DESTINATION ckks/backends
    COMPONENT runtime
)

install(DIRECTORY src/ckks/
    DESTINATION ckks
    COMPONENT python
    FILES_MATCHING PATTERN "*.py"
)

# ==============================================================================
# Build Information Summary
# ==============================================================================
message(STATUS "")
message(STATUS "CKKS OpenFHE GPU Backend Configuration:")
message(STATUS "  Python:           ${Python_EXECUTABLE}")
message(STATUS "  Python Version:   ${Python_VERSION}")
message(STATUS "  pybind11:         ${pybind11_VERSION}")
message(STATUS "  OpenFHE GPU Root: ${OPENFHE_GPU_ROOT}")
message(STATUS "  OpenFHE Build:    ${OPENFHE_GPU_BUILD_DIR}")
message(STATUS "  CUDA Toolkit:     ${CUDAToolkit_VERSION}")
message(STATUS "  Install Prefix:   ${CMAKE_INSTALL_PREFIX}")
message(STATUS "")
