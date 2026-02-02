/* Copyright (c) by CryptoLab Inc.
 * This library is licensed under a
 * Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.
 * You should have received a copy of the license along with this
 * work. If not, see <http://creativecommons.org/licenses/by-nc-sa/4.0/>.
 */
#pragma once

#include <tuple>

#include "Define.h"
#include "DeviceVector.h"
#include "Parameter.h"

namespace ckks {

class Ciphertext {
 public:
  Ciphertext() = default;

  const DeviceVector& getAxDevice() const { return ax__; }
  const DeviceVector& getBxDevice() const { return bx__; }
  DeviceVector& getAxDevice() { return ax__; }
  DeviceVector& getBxDevice() { return bx__; }
  
  bool operator==(const Ciphertext& right) const {
    return (ax__ == right.ax__) && (bx__ == right.bx__);
  }

  bool operator!=(const Ciphertext& right) const {
    return !operator==(right);
  }

//  private:
  DeviceVector ax__;
  DeviceVector bx__;
};

class CtAccurate {
 public:
  CtAccurate() = default;

  const DeviceVector& getAxDevice() const { return ax__; }
  const DeviceVector& getBxDevice() const { return bx__; }
  DeviceVector& getAxDevice() { return ax__; }
  DeviceVector& getBxDevice() { return bx__; }

  bool operator==(const CtAccurate& right) const {
    if (level != right.level) {
      std::cout << "level mismatch\n";
      return false;
    }
    if (noiseScaleDeg != right.noiseScaleDeg) {
      std::cout << "noiseScaleDeg mismatch\n";
      return false;
    }
    if (scalingFactor != right.scalingFactor) {
      std::cout << "scalingFactor mismatch\n";
      return false;
    }
    if (!(ax__ == right.ax__) || !(bx__ == right.bx__)) {
      std::cout << "data mismatch\n";
      return false;
    }
    return (ax__ == right.ax__) && (bx__ == right.bx__) && (level == right.level)
      && (noiseScaleDeg == right.noiseScaleDeg) && (scalingFactor == right.scalingFactor);
  }

  bool operator!=(const CtAccurate& right) const {
    return !operator==(right);
  }

//  private:
  DeviceVector ax__;
  DeviceVector bx__;

  uint32_t level;  // this should always be "max # limbs" - "current # limbs"
  uint32_t noiseScaleDeg;  // basically always 2
  double scalingFactor;  // scaled down by prime for each rescale
  // these two might need to go in params. They're in crypto params in OpenFHE
  // double scalingFactorReal;  
  // double modReduceFactor;  
};

class Plaintext {
 public:
  Plaintext() = default;

  DeviceVector& getMxDevice() { return mx__; }
  const DeviceVector& getMxDevice() const { return mx__; }

//  private:
  DeviceVector mx__;
};

class PtAccurate {
 public:
  PtAccurate() = default;

  const DeviceVector& getMxDevice() const { return mx__; }
  DeviceVector& getMxDevice() { return mx__; }

  bool operator==(const PtAccurate& right) const {
    if (!(mx__ == right.mx__)) {
      std::cout << "data mismatch\n";
      return false;
    }
    if (level != right.level) {
      std::cout << "level mismatch\n";
      return false;
    }
    if (noiseScaleDeg != right.noiseScaleDeg) {
      std::cout << "noiseScaleDeg mismatch\n";
      return false;
    }
    if (scalingFactor != right.scalingFactor) {
      std::cout << "scalingFactor mismatch\n";
      return false;
    }
    return (mx__ == right.mx__) && (level == right.level)
      && (noiseScaleDeg == right.noiseScaleDeg) && (scalingFactor == right.scalingFactor);
  }

  bool operator!=(const PtAccurate& right) const {
    return !operator==(right);
  }

//  private:
  DeviceVector mx__;

  uint32_t level;  // this should always be "max # limbs" - "current # limbs"
  uint32_t noiseScaleDeg;  // basically always 2
  double scalingFactor;  // scaled down by prime for each rescale
  // these two might need to go in params. They're in crypto params in OpenFHE
  // double scalingFactorReal;  
  // double modReduceFactor;  
};

}  // namespace ckks