# ==============================================================================
# CuKKS-cu128 - GPU-accelerated CKKS Backend CMake Configuration (CUDA 12.8)
# ==============================================================================
# Build pybind11 bindings for OpenFHE CKKS functionality (GPU version)
# ==============================================================================

cmake_minimum_required(VERSION 3.18)
project(cukks_cu128 LANGUAGES CXX CUDA)

# ==============================================================================
# C++ Standard and Global Settings
# ==============================================================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ==============================================================================
# GPU Support Option (ON for GPU package)
# ==============================================================================
option(CUKKS_ENABLE_GPU "Enable GPU support" ON)
if(CUKKS_ENABLE_GPU)
    add_compile_definitions(CUKKS_ENABLE_GPU=1)
else()
    add_compile_definitions(CUKKS_ENABLE_GPU=0)
endif()

# CUDA architectures: sm_80 (A100), sm_86 (RTX 30xx), sm_89 (RTX 40xx), sm_90 (H100), sm_120 (RTX 5090)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "80;86;89;90;120")
endif()

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
set(OPENFHE_GPU_ROOT "${CMAKE_SOURCE_DIR}/../openfhe-gpu-public"
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

# Source directory (shared with bindings/openfhe_backend)
set(SOURCE_DIR "${CMAKE_SOURCE_DIR}/../bindings/openfhe_backend/src")

# ==============================================================================
# CUDA Configuration
# ==============================================================================
find_package(CUDAToolkit REQUIRED)

# ==============================================================================
# Main Extension Module (CPU Backend with GPU context support)
# ==============================================================================
add_library(ckks_openfhe_backend MODULE "${SOURCE_DIR}/ckks_openfhe_backend.cpp")

target_compile_options(ckks_openfhe_backend PRIVATE -ftls-model=global-dynamic)

foreach(_inc ${OPENFHE_INCLUDE_DIRS})
    if(EXISTS "${_inc}")
        target_include_directories(ckks_openfhe_backend SYSTEM PRIVATE "${_inc}")
    endif()
endforeach()

# ==============================================================================
# GPU-Accelerated Extension Module
# ==============================================================================
add_library(ckks_openfhe_gpu_backend MODULE "${SOURCE_DIR}/ckks_openfhe_gpu_backend.cpp")

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
    INSTALL_RPATH "$ORIGIN/libs"
)

set_target_properties(ckks_openfhe_gpu_backend PROPERTIES
    BUILD_RPATH "${_rpath}"
    INSTALL_RPATH "$ORIGIN/../libs"
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

install(DIRECTORY "${SOURCE_DIR}/ckks/"
    DESTINATION ckks
    COMPONENT runtime
    FILES_MATCHING
        PATTERN "*.py"
        PATTERN "__pycache__" EXCLUDE
)

# Bundle OpenFHE libraries (including GPU)
install(FILES
    "${OPENFHE_LIBRARY_DIR}/libOPENFHEcore.so"
    "${OPENFHE_LIBRARY_DIR}/libOPENFHEcore.so.1"
    "${OPENFHE_LIBRARY_DIR}/libOPENFHEpke.so"
    "${OPENFHE_LIBRARY_DIR}/libOPENFHEpke.so.1"
    "${OPENFHE_LIBRARY_DIR}/libOPENFHEbinfhe.so"
    "${OPENFHE_LIBRARY_DIR}/libOPENFHEbinfhe.so.1"
    "${OPENFHE_LIBRARY_DIR}/libDeviceFunctions.so"
    DESTINATION ckks/libs
    COMPONENT runtime
)

# Bundle RMM library if exists
set(RMM_BUILD_DIR "${OPENFHE_GPU_BUILD_DIR}/_deps/rmm-build")
if(EXISTS "${RMM_BUILD_DIR}/librmm.so")
    install(FILES "${RMM_BUILD_DIR}/librmm.so"
        DESTINATION ckks/libs
        COMPONENT runtime
    )
endif()

message(STATUS "CuKKS-cu128 GPU package configured")
message(STATUS "  CUKKS_ENABLE_GPU: ${CUKKS_ENABLE_GPU}")
message(STATUS "  CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  OpenFHE root: ${OPENFHE_GPU_ROOT}")
message(STATUS "  Source dir: ${SOURCE_DIR}")
