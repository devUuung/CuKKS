/* Copyright (c) by CryptoLab Inc. and Seoul National University R&DB Foundation.
 * This library is licensed under a
 * Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.
 * You should have received a copy of the license along with this
 * work. If not, see <http://creativecommons.org/licenses/by-nc-sa/4.0/>.
 */
#pragma once

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>

#include "Define.h"

namespace ckks {
using HostVector = thrust::host_vector<word64>;
using DeviceBuffer = rmm::device_buffer;

// A wrapper for device_vector.
class DeviceVector : public rmm::device_uvector<word64> {
 public:
  using Dtype = word64;
  using Base = rmm::device_uvector<Dtype>;

  // A constructor without initilization.
  explicit DeviceVector(int size = 0) : Base(size, cudaStreamLegacy) {}

  // A constructor with initilization.
  explicit DeviceVector(const DeviceVector& ref)
      : Base(ref, cudaStreamLegacy) {}
  DeviceVector(DeviceVector&& other) : Base(std::move(other)) {}

  DeviceVector& operator=(DeviceVector&& other) {
    Base::operator=(std::move(other));
    return *this;
  }

  explicit DeviceVector(const HostVector& ref)
      : Base(ref.size(), cudaStreamLegacy) {
    cudaMemcpyAsync(data(), ref.data(), ref.size() * sizeof(Dtype),
                    cudaMemcpyHostToDevice, stream_);
  }

  operator HostVector() const {
    HostVector host(size());
    cudaMemcpyAsync(host.data(), data(), size() * sizeof(Dtype),
                    cudaMemcpyDeviceToHost, stream_);
    return host;
  }

  void setConstant(const Dtype c) {
    cudaMemsetAsync(data(), c, size() * sizeof(Dtype));
  }

  void resize(int size) { Base::resize(size, stream_); }

  bool operator==(const DeviceVector& other) const{
    return HostVector(*this) == HostVector(other);
  }

  bool operator!=(const DeviceVector& other) const{
    return !operator==(other);
  }

  void append(const DeviceVector& out);

//  private:
  const cudaStream_t stream_ = cudaStreamLegacy;
  };

}  // namespace ckks

// using namespace lbcrypto;
// using namespace ckks;

// ckks::DeviceVector loadIntoDeviceVector(const DCRTPoly& input, const bool verbose = false);
template <typename DCRTPolyType>
inline ckks::DeviceVector loadIntoDeviceVector(const DCRTPolyType& input, const bool verbose = false) {
    if (verbose) {
        std::cout << "DCRTPoly to DeviceVector\n";
    }

    const auto numLimbs = input.m_vectors.size();
    const auto phim = input.GetRingDimension();
    const auto totalSize = numLimbs*phim;

    if (verbose) {
        std::cout << "numLimbs: " << numLimbs << std::endl;
        std::cout << "phim: " << phim << std::endl;
    }

    ckks::HostVector to_load(totalSize);
    for (size_t i = 0; i < numLimbs; i++) {
        for (size_t j = 0; j < phim; j++) {
            to_load[i*phim + j] = (uint64_t)(*(input.m_vectors[i].m_values))[j];
        }
    }

    return ckks::DeviceVector(to_load);
}

template <typename IntType>
inline ckks::DeviceVector LoadIntegerVector(const std::vector<IntType>& vec) {
    ckks::HostVector to_load(vec.size());
    for (size_t i = 0; i < vec.size(); i++) {
        to_load[i] = (uint64_t)(vec[i].ConvertToInt());
    }

    return ckks::DeviceVector(to_load);
}

// inline ckks::DeviceVector LoadConstantVector(const uint64_t size, const uint64_t val = 0) {
//     ckks::HostVector to_load(size);
//     for (size_t i = 0; i < size; i++) {
//         to_load[i] = val;
//     }

//     return ckks::DeviceVector(to_load);
// }