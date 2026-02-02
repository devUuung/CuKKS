/* Copyright (c) by CryptoLab Inc.
 * This library is licensed under a
 * Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.
 * You should have received a copy of the license along with this
 * work. If not, see <http://creativecommons.org/licenses/by-nc-sa/4.0/>.
 */
#pragma once

#include <cstdint>
#include <cmath>

#include "Define.h"
#include "DeviceVector.h"

namespace ckks {
class Parameter {
 public:
  Parameter(int log_degree, int chain_length, int dnum,
            const std::vector<word64>& primes)
      : log_degree_{log_degree},
        degree_{1 << log_degree_},
        // level_{level},
        dnum_{dnum},
        chain_length_{chain_length},  // length of base modulus chain
        alpha_{(int)std::ceil((float)chain_length_ / (float)dnum_)},
        max_num_moduli_{chain_length_ + alpha_},
        num_special_moduli_{alpha_},
        primes_{primes.begin(), primes.begin() + (size_t)chain_length_ + num_special_moduli_} {
    // if ((level_ + 1) % dnum_ != 0) {
    //   std::cout << (level_ + 1) << " " << dnum_ << " " << (level_ + 1) % dnum_ << std::endl;
    //   throw std::logic_error("wrong dnum value.");
    // }
    // if (primes.size() > (size_t)chain_length_ + num_special_moduli_) {
    //   primes.resize(chain_length_ + num_special_moduli_);
    // }
    // if (primes.size() != (size_t)chain_length_ + num_special_moduli_) {
    //   std::cout << primes.size() << " " << (size_t)chain_length_ + num_special_moduli_ << std::endl;
    //   std::cout << chain_length_ << " " << num_special_moduli_ << std::endl;
    //   throw std::logic_error("the size of the primes passed is wrong");
    // }
  };

  const int log_degree_;
  const int degree_;
  // const int level_;
  const int dnum_;
  const int chain_length_;
  const int alpha_;
  const int max_num_moduli_;
  const int num_special_moduli_;
  HostVector primes_;
  std::vector<double> m_scalingFactorsReal;
  std::vector<double> m_scalingFactorsRealBig;
  std::vector<double> m_dmoduliQ;
};

  }  // namespace ckks