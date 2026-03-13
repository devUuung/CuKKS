# cukks-cu128 — CUDA 12.8 backend
# Supported GPUs: V100, T4, RTX 20/30/40xx, A100, H100, RTX 50xx
cmake_minimum_required(VERSION 3.18)
project(cukks_cu128 LANGUAGES CXX CUDA)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    # sm_120 = RTX 5090 (Blackwell)
    set(CMAKE_CUDA_ARCHITECTURES "80;86;89;90;120")
endif()

set(CUKKS_PACKAGE_NAME "cukks-cu128")
include("${CMAKE_SOURCE_DIR}/../cmake/CuKKSBackend.cmake")
