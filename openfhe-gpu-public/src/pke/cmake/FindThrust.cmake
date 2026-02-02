# FindThrust
#
# Sets THRUST_INCLUDE_DIR and Thrust

find_path( THRUST_INCLUDE_DIR
  HINTS
    /usr/include/cuda-12.3
    /usr/local/include
    /usr/local/cuda-12.3/include
    ${CUDA_INCLUDE_DIRS}
    ${CUDA_TOOLKIT_ROOT_DIR}
    ${CUDA_SDK_ROOT_DIR}
  NAMES thrust/version.h
)

if (THRUST_INCLUDE_DIR)
  list (REMOVE_DUPLICATES THRUST_INCLUDE_DIR)
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Thrust DEFAULT_MSG
  THRUST_INCLUDE_DIR
)

if(Thrust_FOUND AND NOT TARGET Thrust)
  add_library(Thrust INTERFACE)
  target_include_directories(Thrust INTERFACE "${THRUST_INCLUDE_DIR}")
endif()

mark_as_advanced(THRUST_INCLUDE_DIR)