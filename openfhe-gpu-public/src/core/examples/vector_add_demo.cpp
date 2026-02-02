// This program computes the sum of two vectors of length N
// By: Nick from CoffeeBeforeArch

#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>
// #include <thrust/device_vector.h>
#include "gpu/vector_add_kernel.h"

int main() {
  // Array size of 2^16 (65536 elements)
  constexpr int N = 1 << 16;
  // constexpr size_t bytes = sizeof(int) * N;

  // Vectors for holding the host-side (CPU-side) data
  std::vector<int> a;
  a.reserve(N);
  std::vector<int> b;
  b.reserve(N);
  std::vector<int> c;
  c.reserve(N);

  // Initialize random numbers in each array
  for (int i = 0; i < N; i++) {
    a.push_back(rand() % 100);
    b.push_back(rand() % 100);
  }

  add_vectors(a, b, c);

  // Check result for errors
  verify_result(a, b, c);

  std::cout << "COMPLETED SUCCESSFULLY\n";

  return 0;
}
