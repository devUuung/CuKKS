include(FetchContent)

# FetchContent_Declare(googletest
#         GIT_REPOSITORY https://github.com/google/googletest.git
#         GIT_TAG v1.10.x)

# FetchContent_Declare(googlebenchmark
#         GIT_REPOSITORY https://github.com/google/benchmark.git
#         GIT_TAG main) # need main for benchmark::benchmark

# FetchContent_MakeAvailable(
#         googletest
#         googlebenchmark)


FetchContent_Declare(
    rmm
    GIT_REPOSITORY https://github.com/rapidsai/rmm
    GIT_TAG        178c2cd2933fa80b70e69863ba727d19f951a551
)

option(BUILD_TESTS OFF)
# set(BUILD_BENCHMARKS OFF)
# target_link_libraries(rmm benchmark::benchmark)
FetchContent_MakeAvailable(rmm)