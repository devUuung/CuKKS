# ==============================================================================
# CuKKSBackend.cmake — Shared build logic for all cukks-cu* backend packages
# ==============================================================================
#
# Expected variables (set by the including CMakeLists.txt before include()):
#   CUKKS_PACKAGE_NAME   e.g. "cukks-cu121"  (used in status messages)
#
# Optional overrides (passed via -D or cmake.define in pyproject.toml):
#   OPENFHE_GPU_ROOT      default: <package_dir>/../openfhe-gpu-public
#   OPENFHE_GPU_BUILD_DIR default: ${OPENFHE_GPU_ROOT}/build
#   CUKKS_ENABLE_GPU      default: ON
#   CMAKE_CUDA_ARCHITECTURES  set by caller before include()
# ==============================================================================

# ------------------------------------------------------------------------------
# C++ Standard and Global Settings
# ------------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ------------------------------------------------------------------------------
# Optimization Flags
# ------------------------------------------------------------------------------
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-O3)
endif()

include(CheckIPOSupported)
check_ipo_supported(RESULT _ipo_supported OUTPUT _ipo_output)
if(_ipo_supported)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
    message(STATUS "IPO/LTO enabled")
else()
    message(STATUS "IPO/LTO not supported: ${_ipo_output}")
endif()

# ------------------------------------------------------------------------------
# GPU Support
# ------------------------------------------------------------------------------
option(CUKKS_ENABLE_GPU "Enable GPU support" ON)
if(CUKKS_ENABLE_GPU)
    add_compile_definitions(CUKKS_ENABLE_GPU=1)
else()
    add_compile_definitions(CUKKS_ENABLE_GPU=0)
endif()

# ------------------------------------------------------------------------------
# Compiler Warning Suppression
# ------------------------------------------------------------------------------
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(
        -Wno-deprecated-declarations
        -Wno-unused-parameter
        -Wno-sign-compare
        -Wno-missing-field-initializers
    )
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unknown-pragmas")
endif()

# ------------------------------------------------------------------------------
# Python and pybind11
# ------------------------------------------------------------------------------
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

# ------------------------------------------------------------------------------
# OpenFHE GPU paths
# (CMAKE_SOURCE_DIR here is the cukks-cu* package dir, one level below repo root)
# ------------------------------------------------------------------------------
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

# C++ binding sources — canonical location under cukks/_native/
set(SOURCE_DIR "${CMAKE_SOURCE_DIR}/../cukks/_native")

# ------------------------------------------------------------------------------
# CUDA
# ------------------------------------------------------------------------------
find_package(CUDAToolkit REQUIRED)

# ------------------------------------------------------------------------------
# Extension Modules
# ------------------------------------------------------------------------------
add_library(ckks_openfhe_backend MODULE "${SOURCE_DIR}/ckks_openfhe_backend.cpp")
target_compile_options(ckks_openfhe_backend PRIVATE -ftls-model=global-dynamic)

add_library(ckks_openfhe_gpu_backend MODULE "${SOURCE_DIR}/ckks_openfhe_gpu_backend.cpp")
target_compile_options(ckks_openfhe_gpu_backend PRIVATE -ftls-model=global-dynamic)

foreach(_inc ${OPENFHE_INCLUDE_DIRS})
    if(EXISTS "${_inc}")
        target_include_directories(ckks_openfhe_backend     SYSTEM PRIVATE "${_inc}")
        target_include_directories(ckks_openfhe_gpu_backend SYSTEM PRIVATE "${_inc}")
    endif()
endforeach()

# ------------------------------------------------------------------------------
# Linking
# ------------------------------------------------------------------------------
if(EXISTS "${OPENFHE_LIBRARY_DIR}")
    target_link_directories(ckks_openfhe_backend     PRIVATE "${OPENFHE_LIBRARY_DIR}")
    target_link_directories(ckks_openfhe_gpu_backend PRIVATE "${OPENFHE_LIBRARY_DIR}")
endif()

set(_cukks_link_libs
    pybind11::module
    OPENFHEcore
    OPENFHEpke
    OPENFHEbinfhe
    DeviceFunctions
    CUDA::cudart
    OpenMP::OpenMP_CXX
    pthread
)

target_link_libraries(ckks_openfhe_backend     PRIVATE ${_cukks_link_libs})
target_link_libraries(ckks_openfhe_gpu_backend PRIVATE ${_cukks_link_libs})

# ------------------------------------------------------------------------------
# RPATH
# ------------------------------------------------------------------------------
set(_rpath
    "${OPENFHE_LIBRARY_DIR}"
    "${CUDAToolkit_LIBRARY_DIR}"
)
list(REMOVE_DUPLICATES _rpath)

set_target_properties(ckks_openfhe_backend PROPERTIES
    BUILD_RPATH   "${_rpath}"
    INSTALL_RPATH "$ORIGIN/../libs"
)
set_target_properties(ckks_openfhe_gpu_backend PROPERTIES
    BUILD_RPATH   "${_rpath}"
    INSTALL_RPATH "$ORIGIN/../libs"
)

# ------------------------------------------------------------------------------
# pybind11 post-processing
# ------------------------------------------------------------------------------
pybind11_extension(ckks_openfhe_backend)
pybind11_strip(ckks_openfhe_backend)

pybind11_extension(ckks_openfhe_gpu_backend)
pybind11_strip(ckks_openfhe_gpu_backend)

# ------------------------------------------------------------------------------
# Installation
# ------------------------------------------------------------------------------
install(TARGETS ckks_openfhe_backend ckks_openfhe_gpu_backend
    LIBRARY DESTINATION ckks/backends
    COMPONENT runtime
)

install(FILES "${SOURCE_DIR}/ckks/__init__.py"
    DESTINATION ckks
    COMPONENT runtime
)
install(FILES "${SOURCE_DIR}/ckks/torch_api.py"
    DESTINATION ckks
    COMPONENT runtime
)
install(FILES "${SOURCE_DIR}/ckks/backends/__init__.py"
    DESTINATION ckks/backends
    COMPONENT runtime
)

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

set(RMM_BUILD_DIR "${OPENFHE_GPU_BUILD_DIR}/_deps/rmm-build")
if(EXISTS "${RMM_BUILD_DIR}/librmm.so")
    install(FILES "${RMM_BUILD_DIR}/librmm.so"
        DESTINATION ckks/libs
        COMPONENT runtime
    )
endif()

# ------------------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------------------
if(NOT DEFINED CUKKS_PACKAGE_NAME)
    set(CUKKS_PACKAGE_NAME "cukks-cuXXX")
endif()

message(STATUS "")
message(STATUS "${CUKKS_PACKAGE_NAME} GPU Backend Configuration:")
message(STATUS "  CUKKS_ENABLE_GPU:        ${CUKKS_ENABLE_GPU}")
message(STATUS "  CUDA architectures:      ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  OpenFHE root:            ${OPENFHE_GPU_ROOT}")
message(STATUS "  OpenFHE build:           ${OPENFHE_GPU_BUILD_DIR}")
message(STATUS "  Binding sources:         ${SOURCE_DIR}")
message(STATUS "  CUDA Toolkit:            ${CUDAToolkit_VERSION}")
message(STATUS "")
