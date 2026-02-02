// This program computes the sum of two vectors of length N
// By: Nick from CoffeeBeforeArch

#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>
#include <thrust/device_vector.h>

// CUDA kernel for vector addition
// __global__ means this is called from the CPU, and runs on the GPU
__global__ void vectorAdd(const int *__restrict a, const int *__restrict b, int *__restrict c, int N);

void add_vectors(std::vector<int> &a, std::vector<int> &b, std::vector<int> &c);

// Check vector add result
void verify_result(std::vector<int> &a, std::vector<int> &b, std::vector<int> &c);
