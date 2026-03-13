# cukks-cu124 — CUDA 12.4 backend
# Supported GPUs: V100, T4, RTX 20/30/40xx, A100, H100
cmake_minimum_required(VERSION 3.18)
project(cukks_cu124 LANGUAGES CXX CUDA)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "70;75;80;86;89;90")
endif()

set(CUKKS_PACKAGE_NAME "cukks-cu124")
include("${CMAKE_SOURCE_DIR}/../cmake/CuKKSBackend.cmake")
