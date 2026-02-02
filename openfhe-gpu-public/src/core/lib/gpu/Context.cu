/* Copyright (c) by CryptoLab Inc.
 * This library is licensed under a
 * Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.
 * You should have received a copy of the license along with this
 * work. If not, see <http://creativecommons.org/licenses/by-nc-sa/4.0/>.
 */
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <tuple>
#include <memory>
#include <iostream>

#include "Basic.h"
#include "Ciphertext.h"
#include "Context.h"
#include "Define.h"
#include "EvaluationKey.h"
#include "MemoryPool.h"
#include "NttImple.h"

using namespace ckks;

#define DEFAULT_BLOCK_DIM 2048

namespace {

// from https://github.com/snucrypto/HEAAN, 131d275
void mulMod(uint64_t &r, uint64_t a, uint64_t b, uint64_t m) {
  unsigned __int128 mul = static_cast<unsigned __int128>(a) * b;
  mul %= static_cast<unsigned __int128>(m);
  r = static_cast<uint64_t>(mul);
}

auto MulMod(const word64 a, const word64 b, const word64 p) {
  word64 r;
  mulMod(r, a, b, p);
  return r;
}

// from https://github.com/snucrypto/HEAAN, 131d275
uint64_t powMod(uint64_t x, uint64_t y, uint64_t modulus) {
  uint64_t res = 1;
  while (y > 0) {
    if (y & 1) {
      mulMod(res, res, x, modulus);
    }
    y = y >> 1;
    mulMod(x, x, x, modulus);
  }
  return res;
}

// from https://github.com/snucrypto/HEAAN, 131d275
void findPrimeFactors(std::vector<uint64_t> &s, uint64_t number) {
  while (number % 2 == 0) {
    s.push_back(2);
    number /= 2;
  }
  for (uint64_t i = 3; i < sqrt(number); i++) {
    while (number % i == 0) {
      s.push_back(i);
      number /= i;
    }
  }
  if (number > 2) {
    s.push_back(number);
  }
}

// from https://github.com/snucrypto/HEAAN, 131d275
uint64_t findPrimitiveRoot(uint64_t modulus) {
  std::vector<uint64_t> s;
  uint64_t phi = modulus - 1;
  findPrimeFactors(s, phi);
  for (uint64_t r = 2; r <= phi; r++) {
    bool flag = false;
    for (auto it = s.begin(); it != s.end(); it++) {
      if (powMod(r, phi / (*it), modulus) == 1) {
        flag = true;
        break;
      }
    }
    if (flag == false) {
      return r;
    }
  }
  throw "Cannot find the primitive root of unity";
}

// from https://github.com/snucrypto/HEAAN, 131d275
auto BitReverse(std::vector<word64> &vals) {
  const int n = vals.size();
  for (long i = 1, j = 0; i < n; ++i) {
    long bit = n >> 1;
    for (; j >= bit; bit >>= 1) {
      j -= bit;
    }
    j += bit;
    if (i < j) {
      std::swap(vals[i], vals[j]);
    }
  }
}

word64 Inverse(const word64 op, const word64 prime) {
  word64 tmp = op > prime ? (op % prime) : op;
  return powMod(tmp, prime - 2, prime);
}

auto genBitReversedTwiddleFacotrs(const word64 root, const word64 p,
                                  const int degree) {
  std::vector<word64> pow_roots(degree), inv_pow_roots(degree);
  const auto root_inverse = Inverse(root, p);
  pow_roots[0] = 1;
  inv_pow_roots[0] = 1;
  for (int i = 1; i < degree; i++) {
    pow_roots[i] = MulMod(pow_roots[i - 1], root, p);
    inv_pow_roots[i] = MulMod(inv_pow_roots[i - 1], root_inverse, p);
  }
  BitReverse(pow_roots);
  BitReverse(inv_pow_roots);
  return std::pair{pow_roots, inv_pow_roots};
}

auto Shoup(const word64 in, const word64 prime) {
  word128 temp = static_cast<word128>(in) << 64;
  return static_cast<word64>(temp / prime);
}

auto ShoupEach(const std::vector<word64> &in, const word64 prime) {
  std::vector<word64> shoup;
  shoup.reserve(in.size());
  for (size_t i = 0; i < in.size(); i++) {
    shoup.push_back(Shoup(in[i], prime));
  }
  return shoup;
}

auto DivTwo(const std::vector<word64> &in, const word64 prime) {
  const auto two_inv = Inverse(2, prime);
  std::vector<word64> out(in.size());
  for (size_t i = 0; i < in.size(); i++) {
    out[i] = MulMod(in[i], two_inv, prime);
  }
  return out;
}

auto Flatten = [](const auto &device_vec) {
  size_t flat_size = 0;
  for (size_t i = 0; i < device_vec.size(); i++) {
    flat_size += device_vec[i].size();
  }
  DeviceVector flat;
  flat.resize(flat_size);
  word64 *out = flat.data();
  for (size_t i = 0; i < device_vec.size(); i++) {
    const auto &src = device_vec[i];
    cudaMemcpyAsync(out, src.data(), src.size() * sizeof(word64),
                    cudaMemcpyDefault);
    out += device_vec[i].size();
  }
  return flat;
};

auto Append = [](auto &vec, const auto &vec_to_append) {
  vec.insert(vec.end(), vec_to_append.begin(), vec_to_append.end());
};

auto ProductExcept = [](auto beg, auto end, word64 except, word64 modulus) {
  return std::accumulate(
      beg, end, (word64)1, [modulus, except](word64 accum, word64 prime) {
        return prime == except ? accum
                               : MulMod(accum, prime % modulus, modulus);
      });
};

auto ComputeQiHats(const std::vector<word64> &primes) {
  HostVector q_i_hats;
  std::transform(primes.begin(), primes.end(), back_inserter(q_i_hats),
                 [&primes](const auto modulus) {
                   return ProductExcept(primes.begin(), primes.end(), modulus,
                                        modulus);
                 });
  return q_i_hats;
}

auto ComputeQiModQj(const std::vector<word64> &primes) {
  std::vector<word64> hat_inv_vec;
  std::vector<word64> hat_inv_shoup_vec;
  const auto q_i_hats = ComputeQiHats(primes);  // q_i_hat is Q/q_i
  std::transform(q_i_hats.begin(), q_i_hats.end(), primes.begin(),
                 back_inserter(hat_inv_vec), Inverse);
  std::transform(hat_inv_vec.begin(), hat_inv_vec.end(), primes.begin(),
                 back_inserter(hat_inv_shoup_vec), Shoup);
  return std::pair{hat_inv_vec, hat_inv_shoup_vec};
}

auto SetDifference = [](auto begin, auto end, auto remove_begin,
                        auto remove_end) {
  std::vector<word64> slimed;
  std::remove_copy_if(begin, end, std::back_inserter(slimed), [&](auto p) {
    return std::find(remove_begin, remove_end, p) != remove_end;
  });
  return slimed;
};

auto ComputeProdQiModQj = [](const auto &end_primes, auto start_begin,
                             auto start_end) {
  HostVector prod_q_i_mod_q_j;
  for (auto modulus : end_primes) {
    std::for_each(start_begin, start_end, [&](auto p) {
      prod_q_i_mod_q_j.push_back(
          ProductExcept(start_begin, start_end, p, modulus));
    });
  }
  return prod_q_i_mod_q_j;
};

}  // namespace

void Context::setup(const Parameter &param, const std::vector<word64>& primitiveRoots) {
  HostVector barret_k, barret_ratio, power_of_roots_vec,
      power_of_roots_shoup_vec, inv_power_of_roots_vec,
      inv_power_of_roots_shoup_vec;
  size_t i = 0; 
  for (auto p : param.primes_) {
    long barret = floor(log2(p)) + 63;
    barret_k.push_back(barret);
    word128 temp = ((word64)1 << (barret - 64));
    temp <<= 64;
    barret_ratio.push_back((word64)(temp / p));
    auto root = primitiveRoots[i++];
    // std::cout << "gpu setup " << i << "th root: " << root << std::endl;
    auto [power_of_roots, inverse_power_of_roots] =
        genBitReversedTwiddleFacotrs(root, p, degree__);
    auto power_of_roots_shoup = ShoupEach(power_of_roots, p);
    auto inv_power_of_roots_div_two = DivTwo(inverse_power_of_roots, p);
    auto inv_power_of_roots_shoup = ShoupEach(inv_power_of_roots_div_two, p);
    Append(power_of_roots_vec, power_of_roots);
    Append(power_of_roots_shoup_vec, power_of_roots_shoup);
    Append(inv_power_of_roots_vec, inv_power_of_roots_div_two);
    Append(inv_power_of_roots_shoup_vec, inv_power_of_roots_shoup);
  }
  barret_ratio__ = DeviceVector(barret_ratio);
  barret_k__ = DeviceVector(barret_k);
  power_of_roots__ = DeviceVector(power_of_roots_vec);
  power_of_roots_shoup__ = DeviceVector(power_of_roots_shoup_vec);
  inverse_power_of_roots_div_two__ = DeviceVector(inv_power_of_roots_vec);
  inverse_scaled_power_of_roots_div_two__ =
      DeviceVector(inv_power_of_roots_shoup_vec);
  // make base-conversion-related parameters
  GenModUpParams();
  GenModDownParams();
  GenRescaleParams();
}

Context::Context(const Parameter &param, const std::vector<word64>& primitiveRoots)
  : param__{param},
    degree__{param.degree_},
    alpha__{param.alpha_},
    primes__{param.primes_} {

    setup(param, primitiveRoots);
}

Context::Context(const Parameter &param)
  : param__{param},
    degree__{param.degree_},
    alpha__{param.alpha_},
    primes__{param.primes_} {

  std::vector<word64> roots; roots.reserve(param.primes_.size());
  for (auto p : param.primes_) {
    auto root = findPrimitiveRoot(p);
    root = powMod(root, (p - 1) / (2 * degree__), p);
    roots.push_back(root);
  }

  setup(param, roots);

}

void Context::GenModUpParams() {
  prod_q_i_mod_q_j__.reserve(param__.chain_length_);
  hat_inverse_vec__.reserve(param__.chain_length_);
  hat_inverse_vec_shoup__.reserve(param__.chain_length_);
  hat_inverse_vec_batched__.reserve(param__.chain_length_);
  hat_inverse_vec_shoup_batched__.reserve(param__.chain_length_);

  for (int num_limbs = 1; num_limbs <= param__.chain_length_ ; num_limbs++) {

      // std::cout << "computing ModUp parameters for digit " << num_limbs << " limbs\n"; 

    // std::vector<word64> curr_primes(param__.primes_.begin(), param__.primes_.begin() + num_limbs);
    std::vector<word64> curr_primes(num_limbs+alpha__);
    for (int i = 0; i < num_limbs; i++) curr_primes[i] = param__.primes_[i];
    for (int i = 0; i < alpha__; i++) curr_primes[num_limbs+i] = param__.primes_[param__.chain_length_  + i];
    // assert(curr_primes.size() == num_limbs + alpha__);

    const int beta = ceil((float)num_limbs / (float)alpha__);
    prod_q_i_mod_q_j__.push_back(std::vector<DeviceVector>());
    prod_q_i_mod_q_j__[num_limbs - 1].reserve(beta);

    hat_inverse_vec__.push_back(std::vector<DeviceVector>());
    hat_inverse_vec__[num_limbs - 1].reserve(beta);

    hat_inverse_vec_shoup__.push_back(std::vector<DeviceVector>());
    hat_inverse_vec_shoup__[num_limbs - 1].reserve(beta);

    for (int dnum_idx = 0; dnum_idx < beta; dnum_idx++) {
      auto prime_begin = curr_primes.begin();
      auto prime_end = curr_primes.end();
      auto start_begin = prime_begin + dnum_idx * alpha__;

      auto num_raised_input_limbs = alpha__;
      if (dnum_idx  == beta-1) {
        num_raised_input_limbs = num_limbs - (dnum_idx)*alpha__;
      }

      // std::cout << "\tcomputing parameters for digit " << dnum_idx << " with " << num_raised_input_limbs << " limbs\n"; 

      auto start_end = start_begin + num_raised_input_limbs;

      auto [hat_inv, hat_inv_shoup] =
          ComputeQiModQj(std::vector<word64>(start_begin, start_end));
      hat_inverse_vec__[num_limbs - 1].push_back(DeviceVector(hat_inv));
      hat_inverse_vec_shoup__[num_limbs - 1].push_back(DeviceVector(hat_inv_shoup));
      auto end_primes =
          SetDifference(prime_begin, prime_end, start_begin, start_end);
      prod_q_i_mod_q_j__[num_limbs - 1].push_back(
          DeviceVector(ComputeProdQiModQj(end_primes, start_begin, start_end)));
    }

    hat_inverse_vec_batched__.push_back(Flatten(hat_inverse_vec__[num_limbs-1]));
    hat_inverse_vec_shoup_batched__.push_back(Flatten(hat_inverse_vec_shoup__[num_limbs-1]));
  }
}

void Context::GenModDownParams() {
  auto prime_begin = param__.primes_.begin();
  // auto prime_end = param__.primes_.end();
  // the prime basis is just the output primes plus the extension limbs
  // the input limbs to the basis change is always just the extension limbs
  for (int gap = 0; gap < param__.chain_length_; gap++) {
    // int start_length = param__.num_special_moduli_ + gap;
    int start_length = param__.num_special_moduli_;
    int end_length = param__.chain_length_ - gap;
    auto prime_end = param__.primes_.begin() + end_length;
    // auto start_begin = param__.primes_.begin() + end_length;
    auto start_begin = param__.primes_.begin() + param__.chain_length_;  // start of extension limbs
    auto start_end = start_begin + start_length;

    // these are always the same...
    auto [hat_inv, hat_inv_shoup] =
        ComputeQiModQj(std::vector<word64>(start_begin, start_end));
    hat_inverse_vec_moddown__.push_back(DeviceVector(hat_inv));
    hat_inverse_vec_shoup_moddown__.push_back(DeviceVector(hat_inv_shoup));

    // auto end_primes =
        // SetDifference(prime_begin, prime_end, start_begin, start_end);
    auto end_primes = std::vector<word64>(prime_begin, prime_end);
    prod_q_i_mod_q_j_moddown__.push_back(
        DeviceVector(ComputeProdQiModQj(end_primes, start_begin, start_end)));
    std::vector<word64> prod_inv, prod_shoup;
    for (auto p : end_primes) {
      auto prod = ProductExcept(start_begin, start_end, 0, p);
      auto inv = Inverse(prod, p);
      prod_inv.push_back(inv);
      prod_shoup.push_back(Shoup(inv, p));
    }
    prod_inv_moddown__.push_back(DeviceVector(prod_inv));
    prod_inv_shoup_moddown__.push_back(DeviceVector(prod_shoup));
  }
}

void Context::GenRescaleParams() {
  auto prime_begin = param__.primes_.begin();
  for (int gap = 0; gap < param__.chain_length_- 2; gap++) {
    int end_length = param__.chain_length_ - gap - 1;
    auto prime_end = prime_begin + end_length+1;
    auto start_begin = prime_end - 1;
    auto start_end = start_begin+1;
    auto end_primes =
        SetDifference(prime_begin, prime_begin + end_length+1, start_begin, start_end);
    std::vector<word64> prod_inv, prod_shoup;
    for (auto p : end_primes) {
      auto prod = ProductExcept(start_begin, start_end, 0, p);
      auto inv = Inverse(prod, p);
      prod_inv.push_back(inv);
      prod_shoup.push_back(Shoup(inv, p));
    }
    prod_inv_rescale__.push_back(DeviceVector(prod_inv));
    prod_inv_shoup_rescale__.push_back(DeviceVector(prod_shoup));
  }
}


//
// Top-level functions
//

Ciphertext Context::EvalMultAndRelin(const Ciphertext& ct1, const Ciphertext& ct2, const EvaluationKey& evk, const bool verbose) const {
  Ciphertext multInput1, multInput2;
  Rescale(ct1, multInput1);
  Rescale(ct2, multInput2);

  // if (verbose) std::cout << "\tFinished rescale\n";

  DeviceVector gpu_to_relin_2;
  Ciphertext orig_elems;
  EvalMult(multInput1, multInput2, orig_elems.bx__, orig_elems.ax__, gpu_to_relin_2);

  // if (verbose) std::cout << "\tFinished eval mult\n";

  DeviceVector raisedDigits = ModUp(gpu_to_relin_2);

  // if (verbose) std::cout << "\tFinished modup\n";

  DeviceVector ks_a, ks_b;

  KeySwitch(raisedDigits, evk, ks_a, ks_b);

  // if (verbose) std::cout << "\tFinished keyswitch\n";

  Ciphertext ks_output;
  ModDown(ks_a, ks_output.ax__);
  ModDown(ks_b, ks_output.bx__);

  // if (verbose) std::cout << "\tFinished mod down\n";

  Ciphertext toLoad; 
  Add(ks_output, orig_elems, toLoad);

  return toLoad;
}

CtAccurate Context::EvalMultAndRelin(const CtAccurate& ct1, const CtAccurate& ct2, const EvaluationKey& evk, const bool verbose) const {
  
  CtAccurate multInput1, multInput2;
  Rescale(ct1, multInput1);
  Rescale(ct2, multInput2);

  if (multInput1.level != multInput2.level) {
    if (multInput1.level > multInput2.level) { // scale down multInput2
      multInput2 = DropLimbs(multInput2, multInput1.level - multInput2.level);
    } else {
      multInput1 = DropLimbs(multInput1, multInput2.level - multInput1.level);
    }
  }

  // if (verbose) std::cout << "\tFinished rescale\n";
  assert(multInput1.level == multInput2.level);

  DeviceVector gpu_to_relin_2;
  CtAccurate orig_elems;
  // std::cout << "input levels: " << multInput1.level << " " << multInput2.level << std::endl;
  EvalMult(multInput1, multInput2, orig_elems.bx__, orig_elems.ax__, gpu_to_relin_2);
  orig_elems.level = multInput1.level;
  orig_elems.noiseScaleDeg = multInput1.noiseScaleDeg + multInput2.noiseScaleDeg;
  orig_elems.scalingFactor = multInput1.scalingFactor * multInput2.scalingFactor;

  // if (verbose) std::cout << "\tFinished eval mult\n";

  DeviceVector raisedDigits = ModUp(gpu_to_relin_2);

  // if (verbose) std::cout << "\tFinished modup\n";

  DeviceVector ks_a, ks_b;

  KeySwitch(raisedDigits, evk, ks_a, ks_b);

  // if (verbose) std::cout << "\tFinished keyswitch\n";

  Ciphertext ks_output;
  ModDown(ks_a, ks_output.ax__);
  ModDown(ks_b, ks_output.bx__);
  // ks_output.level = orig_elems.level;
  // ks_output.noiseScaleDeg = orig_elems.noiseScaleDeg;
  // ks_output.scalingFactor = orig_elems.scalingFactor;

  // if (verbose) std::cout << "\tFinished mod down\n";

  CtAccurate toLoad; 
  // Add(ks_output, orig_elems, toLoad);
  AddCore(ks_output.ax__, ks_output.bx__, orig_elems.ax__, orig_elems.bx__, toLoad.ax__, toLoad.bx__);

  toLoad.level = orig_elems.level;
  toLoad.noiseScaleDeg = orig_elems.noiseScaleDeg;
  toLoad.scalingFactor = orig_elems.scalingFactor;

  return toLoad;
}

Ciphertext Context::EvalMultAndRelinNoRescale(const Ciphertext& ct1, const Ciphertext& ct2, const EvaluationKey& evk) const {
  DeviceVector gpu_to_relin_2;
  Ciphertext orig_elems;
  EvalMult(ct1, ct2, orig_elems.bx__, orig_elems.ax__, gpu_to_relin_2);

  DeviceVector raisedDigits = ModUp(gpu_to_relin_2);

  DeviceVector ks_a, ks_b;

  KeySwitch(raisedDigits, evk, ks_a, ks_b);

  Ciphertext ks_output;
  ModDown(ks_a, ks_output.ax__);
  ModDown(ks_b, ks_output.bx__);

  Ciphertext toLoad; 
  Add(ks_output, orig_elems, toLoad);

  return toLoad;
}

Ciphertext Context::EvalSquareAndRelin(const Ciphertext& ct, const EvaluationKey& evk) const {
  Ciphertext multInput;
  Rescale(ct, multInput);

  DeviceVector gpu_to_relin_2;
  Ciphertext orig_elems;
  EvalSquare(multInput, orig_elems.bx__, orig_elems.ax__, gpu_to_relin_2);

  DeviceVector raisedDigits = ModUp(gpu_to_relin_2);

  DeviceVector ks_a, ks_b;

  KeySwitch(raisedDigits, evk, ks_a, ks_b);

  Ciphertext ks_output;
  ModDown(ks_a, ks_output.ax__);
  ModDown(ks_b, ks_output.bx__);

  Ciphertext toLoad; 
  Add(ks_output, orig_elems, toLoad);

  return toLoad;
}

void Context::EvalSquareAndRelin(const Ciphertext& ct, const EvaluationKey& evk, Ciphertext& out) const {
  out = EvalSquareAndRelin(ct, evk);
}

CtAccurate Context::EvalSquareAndRelin(const CtAccurate& ct, const EvaluationKey& evk) const {
  CtAccurate multInput;
  Rescale(ct, multInput);

  DeviceVector gpu_to_relin_2;
  CtAccurate orig_elems;
  EvalSquare(multInput, orig_elems.bx__, orig_elems.ax__, gpu_to_relin_2);
  orig_elems.level = multInput.level;
  orig_elems.noiseScaleDeg = multInput.noiseScaleDeg + multInput.noiseScaleDeg;
  orig_elems.scalingFactor = multInput.scalingFactor * multInput.scalingFactor;

  DeviceVector raisedDigits = ModUp(gpu_to_relin_2);

  DeviceVector ks_a, ks_b;

  KeySwitch(raisedDigits, evk, ks_a, ks_b);

  Ciphertext ks_output;
  ModDown(ks_a, ks_output.ax__);
  ModDown(ks_b, ks_output.bx__);

  CtAccurate toLoad; 
  // Add(ks_output, orig_elems, toLoad);
  AddCore(ks_output.ax__, ks_output.bx__, orig_elems.ax__, orig_elems.bx__, toLoad.ax__, toLoad.bx__);

  toLoad.level = orig_elems.level;
  toLoad.noiseScaleDeg = orig_elems.noiseScaleDeg;
  toLoad.scalingFactor = orig_elems.scalingFactor;

  return toLoad;
}

Ciphertext Context::EvalSquareAndRelinNoRescale(const Ciphertext& ct, const EvaluationKey& evk) const {
  DeviceVector gpu_to_relin_2;
  Ciphertext orig_elems;
  EvalSquare(ct, orig_elems.bx__, orig_elems.ax__, gpu_to_relin_2);

  DeviceVector raisedDigits = ModUp(gpu_to_relin_2);

  DeviceVector ks_a, ks_b;

  KeySwitch(raisedDigits, evk, ks_a, ks_b);

  Ciphertext ks_output;
  ModDown(ks_a, ks_output.ax__);
  ModDown(ks_b, ks_output.bx__);

  Ciphertext toLoad; 
  Add(ks_output, orig_elems, toLoad);

  return toLoad;
}

CtAccurate Context::EvalMultAndRelinNoRescale(const CtAccurate& ct1, const CtAccurate& ct2, const EvaluationKey& evk) const {
  DeviceVector gpu_to_relin_2;
  CtAccurate orig_elems;
  EvalMult(ct1, ct2, orig_elems.bx__, orig_elems.ax__, gpu_to_relin_2);
  orig_elems.level = ct1.level;
  orig_elems.noiseScaleDeg = ct1.noiseScaleDeg + ct2.noiseScaleDeg;
  orig_elems.scalingFactor = ct1.scalingFactor * ct2.scalingFactor;



  DeviceVector raisedDigits = ModUp(gpu_to_relin_2);



  DeviceVector ks_a, ks_b;
  KeySwitch(raisedDigits, evk, ks_a, ks_b);



  Ciphertext ks_output;
  ModDown(ks_a, ks_output.ax__);
  ModDown(ks_b, ks_output.bx__);



  CtAccurate toLoad; 
  AddCore(ks_output.ax__, ks_output.bx__, orig_elems.ax__, orig_elems.bx__, toLoad.ax__, toLoad.bx__);

  toLoad.level = orig_elems.level;
  toLoad.noiseScaleDeg = orig_elems.noiseScaleDeg;
  toLoad.scalingFactor = orig_elems.scalingFactor;



  return toLoad;
}

CtAccurate Context::EvalSquareAndRelinNoRescale(const CtAccurate& ct, const EvaluationKey& evk) const {
  DeviceVector gpu_to_relin_2;
  CtAccurate orig_elems;
  EvalSquare(ct, orig_elems.bx__, orig_elems.ax__, gpu_to_relin_2);
  orig_elems.level = ct.level;
  orig_elems.noiseScaleDeg = ct.noiseScaleDeg + ct.noiseScaleDeg;
  orig_elems.scalingFactor = ct.scalingFactor * ct.scalingFactor;

  DeviceVector raisedDigits = ModUp(gpu_to_relin_2);

  DeviceVector ks_a, ks_b;

  KeySwitch(raisedDigits, evk, ks_a, ks_b);

  Ciphertext ks_output;
  ModDown(ks_a, ks_output.ax__);
  ModDown(ks_b, ks_output.bx__);

  CtAccurate toLoad; 
  // Add(ks_output, orig_elems, toLoad);
  AddCore(ks_output.ax__, ks_output.bx__, orig_elems.ax__, orig_elems.bx__, toLoad.ax__, toLoad.bx__);

  toLoad.level = orig_elems.level;
  toLoad.noiseScaleDeg = orig_elems.noiseScaleDeg;
  toLoad.scalingFactor = orig_elems.scalingFactor;

  return toLoad;
}


__global__ void permute_(word64* out, const word64* in, const word64* indices, const uint32_t log_degree, const uint32_t degree, const uint32_t degree_mask) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;  // output index
  const uint32_t limbInd = i >> log_degree;
  const uint32_t dataInd = i & degree_mask;
  out[i] = in[limbInd*degree + indices[dataInd]];
}

__global__ void permute_two_(word64* out_a, word64* out_b, 
    const word64* in_a, const word64* in_b, const word64* indices, const uint32_t log_degree, const uint32_t degree, const uint32_t degree_mask) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;  // output index
  const uint32_t limbInd = i >> log_degree;
  const uint32_t dataInd = i & degree_mask;
  out_a[i] = in_a[limbInd*degree + indices[dataInd]];
  out_b[i] = in_b[limbInd*degree + indices[dataInd]];
}

inline __device__ static unsigned char reverse_byte(unsigned char x) {
    static const unsigned char table[] = {
        0x00, 0x80, 0x40, 0xc0, 0x20, 0xa0, 0x60, 0xe0, 0x10, 0x90, 0x50, 0xd0, 0x30, 0xb0, 0x70, 0xf0, 0x08, 0x88,
        0x48, 0xc8, 0x28, 0xa8, 0x68, 0xe8, 0x18, 0x98, 0x58, 0xd8, 0x38, 0xb8, 0x78, 0xf8, 0x04, 0x84, 0x44, 0xc4,
        0x24, 0xa4, 0x64, 0xe4, 0x14, 0x94, 0x54, 0xd4, 0x34, 0xb4, 0x74, 0xf4, 0x0c, 0x8c, 0x4c, 0xcc, 0x2c, 0xac,
        0x6c, 0xec, 0x1c, 0x9c, 0x5c, 0xdc, 0x3c, 0xbc, 0x7c, 0xfc, 0x02, 0x82, 0x42, 0xc2, 0x22, 0xa2, 0x62, 0xe2,
        0x12, 0x92, 0x52, 0xd2, 0x32, 0xb2, 0x72, 0xf2, 0x0a, 0x8a, 0x4a, 0xca, 0x2a, 0xaa, 0x6a, 0xea, 0x1a, 0x9a,
        0x5a, 0xda, 0x3a, 0xba, 0x7a, 0xfa, 0x06, 0x86, 0x46, 0xc6, 0x26, 0xa6, 0x66, 0xe6, 0x16, 0x96, 0x56, 0xd6,
        0x36, 0xb6, 0x76, 0xf6, 0x0e, 0x8e, 0x4e, 0xce, 0x2e, 0xae, 0x6e, 0xee, 0x1e, 0x9e, 0x5e, 0xde, 0x3e, 0xbe,
        0x7e, 0xfe, 0x01, 0x81, 0x41, 0xc1, 0x21, 0xa1, 0x61, 0xe1, 0x11, 0x91, 0x51, 0xd1, 0x31, 0xb1, 0x71, 0xf1,
        0x09, 0x89, 0x49, 0xc9, 0x29, 0xa9, 0x69, 0xe9, 0x19, 0x99, 0x59, 0xd9, 0x39, 0xb9, 0x79, 0xf9, 0x05, 0x85,
        0x45, 0xc5, 0x25, 0xa5, 0x65, 0xe5, 0x15, 0x95, 0x55, 0xd5, 0x35, 0xb5, 0x75, 0xf5, 0x0d, 0x8d, 0x4d, 0xcd,
        0x2d, 0xad, 0x6d, 0xed, 0x1d, 0x9d, 0x5d, 0xdd, 0x3d, 0xbd, 0x7d, 0xfd, 0x03, 0x83, 0x43, 0xc3, 0x23, 0xa3,
        0x63, 0xe3, 0x13, 0x93, 0x53, 0xd3, 0x33, 0xb3, 0x73, 0xf3, 0x0b, 0x8b, 0x4b, 0xcb, 0x2b, 0xab, 0x6b, 0xeb,
        0x1b, 0x9b, 0x5b, 0xdb, 0x3b, 0xbb, 0x7b, 0xfb, 0x07, 0x87, 0x47, 0xc7, 0x27, 0xa7, 0x67, 0xe7, 0x17, 0x97,
        0x57, 0xd7, 0x37, 0xb7, 0x77, 0xf7, 0x0f, 0x8f, 0x4f, 0xcf, 0x2f, 0xaf, 0x6f, 0xef, 0x1f, 0x9f, 0x5f, 0xdf,
        0x3f, 0xbf, 0x7f, 0xff,
    };
    return table[x];
}

// static int shift_trick[] = {0, 7, 6, 5, 4, 3, 2, 1};
// static DeviceVector shift_trick

/* Function to reverse bits of num */
inline __device__ uint32_t ReverseBits(uint32_t num, uint32_t msb) {
    constexpr int shift_trick[] = {0, 7, 6, 5, 4, 3, 2, 1};
    uint32_t msbb = (msb >> 3) + (msb & 0x7 ? 1 : 0);
    switch (msbb) {
        case 1:
            return (reverse_byte((num)&0xff) >> shift_trick[msb & 0x7]);

        case 2:
            return (reverse_byte((num)&0xff) << 8 | reverse_byte((num >> 8) & 0xff)) >> shift_trick[msb & 0x7];

        case 3:
            return (reverse_byte((num)&0xff) << 16 | reverse_byte((num >> 8) & 0xff) << 8 |
                    reverse_byte((num >> 16) & 0xff)) >>
                   shift_trick[msb & 0x7];
        case 4:
            return (reverse_byte((num)&0xff) << 24 | reverse_byte((num >> 8) & 0xff) << 16 |
                    reverse_byte((num >> 16) & 0xff) << 8 | reverse_byte((num >> 24) & 0xff)) >>
                   shift_trick[msb & 0x7];
        default:
            // throw std::logic_error("msbb value not handled:" + std::to_string(msbb));
            assert(false);
            return 0;  // error result
            // OPENFHE_THROW(math_error, "msbb value not handled:" +
            // std::to_string(msbb));
    }
}


__global__ void automorph_(word64* out,
  const word64* in, const word64 auto_index, const uint32_t log_degree, const uint32_t degree, const uint32_t degree_mask) {

  const int i = blockIdx.x * blockDim.x + threadIdx.x;  // output index
  const uint32_t limbInd = i >> log_degree;
  const uint32_t dataInd = i & degree_mask;

  // compute output index
  // uint32_t jTmp    = ((j << 1) + 1);
  // usint idx        = ((jTmp * k) - (((jTmp * k) >> logm) << logm)) >> 1;
  // usint jrev       = ReverseBits(j, logn);
  // usint idxrev     = ReverseBits(idx, logn);
  // (*precomp)[jrev] = idxrev;
  const uint32_t jTmp    = ((dataInd << 1) + 1);
  const uint32_t idx        = ((jTmp * auto_index) - (((jTmp * auto_index) >> (log_degree+1)) << (log_degree+1))) >> 1;
  const uint32_t jrev       = ReverseBits(dataInd, log_degree);
  const uint32_t idxrev     = ReverseBits(idx, log_degree);

  out[limbInd*degree + jrev] = in[limbInd*degree + idxrev];
}

__global__ void automorph_pair_(word64* out_a, word64* out_b,
  const word64* in_a, const word64* in_b, const word64 auto_index, const uint32_t log_degree, const uint32_t degree, const uint32_t degree_mask) {

  const int i = blockIdx.x * blockDim.x + threadIdx.x;  // output index
  const uint32_t limbInd = i >> log_degree;
  const uint32_t dataInd = i & degree_mask;

  // compute output index
  // uint32_t jTmp    = ((j << 1) + 1);
  // usint idx        = ((jTmp * k) - (((jTmp * k) >> logm) << logm)) >> 1;
  // usint jrev       = ReverseBits(j, logn);
  // usint idxrev     = ReverseBits(idx, logn);
  // (*precomp)[jrev] = idxrev;
  const uint32_t jTmp    = ((dataInd << 1) + 1);
  const uint32_t idx        = ((jTmp * auto_index) - (((jTmp * auto_index) >> (log_degree+1)) << (log_degree+1))) >> 1;
  const uint32_t jrev       = ReverseBits(dataInd, log_degree);
  const uint32_t idxrev     = ReverseBits(idx, log_degree);

  out_a[limbInd*degree + jrev] = in_a[limbInd*degree + idxrev];
  out_b[limbInd*degree + jrev] = in_b[limbInd*degree + idxrev];
}

DeviceVector Context::AutomorphismTransform(const DeviceVector& in, const uint32_t auto_index) const {
  // This function is untested.
  
  DeviceVector out(in.size());
  const uint32_t numLimbs = in.size() / degree__;

  const int block_dim = 256;
  const int grid_dim = degree__ * numLimbs / block_dim;
  const uint32_t degree_mask = degree__-1;
  automorph_<<<grid_dim,block_dim>>>(out.data(), in.data(), auto_index, param__.log_degree_, degree__, degree_mask);

  return out;
}

DeviceVector Context::AutomorphismTransform(const DeviceVector& in, const std::vector<uint32_t>& inds) const {
  // execute automorphism on each limb
  assert(inds.size() == degree__);

  DeviceVector out(in.size());
  const uint32_t numLimbs = in.size() / degree__;
  assert(in.size() == numLimbs*degree__);

  // std::cout << "TODO: This automorphism function is also a complete mess....\n";

  DeviceVector indices_gpu(inds);

  const int block_dim = 256;
  const int grid_dim = degree__ * numLimbs / block_dim;
  const uint32_t degree_mask = degree__-1;
  permute_<<<grid_dim,block_dim>>>(out.data(), in.data(), indices_gpu.data(), param__.log_degree_, degree__, degree_mask);

  return out;
}

void Context::AutomorphismTransformInPlace(CtAccurate& in, const std::vector<uint32_t>& inds) const {
  // execute automorphism on each limb
  assert(inds.size() == degree__);

  DeviceVector out_a(in.ax__.size());
  DeviceVector out_b(in.bx__.size());
  const uint32_t numLimbs = in.ax__.size() / degree__;

  // std::cout << "TODO: This automorphism function is also a complete mess....\n";

  DeviceVector indices_gpu(inds);

  const int block_dim = 256;
  const int grid_dim = degree__ * numLimbs / block_dim;
  const uint32_t degree_mask = degree__-1;
  permute_two_<<<grid_dim,block_dim>>>(out_a.data(), out_b.data(), in.ax__.data(), in.bx__.data(), indices_gpu.data(), param__.log_degree_, degree__, degree_mask);

  in.ax__ = DeviceVector(out_a);
  in.bx__ = DeviceVector(out_b);
}

void Context::AutomorphismTransformInPlace(CtAccurate& in, const uint32_t auto_index) const {
  // This function seems to work fine but should be tested further.

  DeviceVector out_a(in.ax__.size());
  DeviceVector out_b(in.bx__.size());
  const uint32_t numLimbs = in.ax__.size() / degree__;

  const int block_dim = 256;
  const int grid_dim = degree__ * numLimbs / block_dim;
  const uint32_t degree_mask = degree__-1;
  automorph_pair_<<<grid_dim,block_dim>>>(out_a.data(), out_b.data(), in.ax__.data(), in.bx__.data(), auto_index, param__.log_degree_, degree__, degree_mask);

  in.ax__ = DeviceVector(out_a);
  in.bx__ = DeviceVector(out_b);
}

CtAccurate Context::EvalAtIndex(const CtAccurate& ct, const EvaluationKey& evk, const uint32_t auto_index) const {
  CtAccurate rot_ct(ct);
  AutomorphismTransformInPlace(rot_ct, auto_index);

  DeviceVector raisedDigits = ModUp(rot_ct.ax__);

  // if (verbose) std::cout << "\tFinished modup\n";

  DeviceVector ks_a, ks_b;

  KeySwitch(raisedDigits, evk, ks_a, ks_b);

  // if (verbose) std::cout << "\tFinished keyswitch\n";

  CtAccurate toLoad;
  ModDown(ks_a, toLoad.ax__);
  ModDown(ks_b, toLoad.bx__);

  AddCoreInPlace(toLoad.bx__, rot_ct.bx__);

  toLoad.level = ct.level;
  toLoad.noiseScaleDeg = ct.noiseScaleDeg;
  toLoad.scalingFactor = ct.scalingFactor;

  return toLoad;
}

// CtAccurate Context::AutomorphismTransform(const CtAccurate& ct, const std::vector<uint32_t> inds) const {
//   // execute automorphism on each limb
// }

//
// Utility functions
//

__global__ void mulAndAddScaled_(word64 *op1, const word64* op2, const word64* scalar, 
  // const word64* scalar_shoup,
  const word64* primes, 
  const word64 *barrett_ratios, const word64 *barrett_Ks,
  const uint32_t op2_limbs, const uint32_t degree) {

  STRIDED_LOOP_START(degree * op2_limbs, i);
  // const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int prime_idx = i / degree;
  const auto prime = primes[prime_idx];
  // const word64 toAdd = mul_and_reduce_shoup(op2[i], scalar[prime_idx], scalar_shoup[prime_idx], prime);
  const auto toReduce = mult_64_64_128(op2[i], scalar[prime_idx]);
  const auto toAdd = barret_reduction_128_64(toReduce, prime, barrett_ratios[prime_idx], barrett_Ks[prime_idx]);
  op1[i] += toAdd;
  if (op1[i] >= prime) op1[i] -= prime;
  STRIDED_LOOP_END;
}

// __global__ void mulShoup_(word64 *op1, const word64* op2, 
//   const word64* scalar, const word64* scalar_shoup, const word64* primes, 
//   const uint32_t op2_limbs, const uint32_t degree) {

//   STRIDED_LOOP_START(degree * op2_limbs, i);
//   const int prime_idx = i / degree;
//   const auto prime = primes[prime_idx];
//   op1[i] = mul_and_reduce_shoup(op2[i], scalar[prime_idx], scalar_shoup[prime_idx], prime);
//   if (op1[i] >= prime) op1[i] -= prime;
//   STRIDED_LOOP_END;
// }

void Context::AddScaledMessageTerm(DeviceVector& inner_prod_b, const DeviceVector& message_term) const {
  // first argument is raised key-switch innter product term.
  // second argument is unraised ciphertext message term.
  // after P multiplication, all raised limbs are 0
  // this is essentially eval

  const uint32_t message_limbs = message_term.size() / degree__;

  const int block_dim = 256;
  const int grid_dim = degree__ * message_limbs / block_dim;
  mulAndAddScaled_<<<grid_dim, block_dim>>>(
    inner_prod_b.data(), message_term.data(), Pmodq.data(), 
    // Pmodq_shoup.data(), 
    primes__.data(), 
    barret_ratio__.data(), barret_k__.data(),
    message_limbs, degree__);
}

// NTT is made up of two NTT stages. Here the 'first' means the first NTT
// stage in NTT (so it becomes second in view of iNTT).
DeviceVector Context::FromNTT(const DeviceVector &in) const {
  DeviceVector out;
  out.resize(in.size());
  const int batch = in.size() / degree__;
  const int start_prime_idx = 0;
  dim3 grid(2048);
  dim3 block(256);
  const int per_thread_ntt_size = 8;
  const int first_stage_radix_size = 256;
  const int second_radix_size = degree__ / first_stage_radix_size;
  const int pad = 4;
  const int per_thread_storage = block.x * per_thread_ntt_size * sizeof(word64);
  Intt8PointPerThreadPhase2OoP<<<grid, block, per_thread_storage>>>(
      in.data(), first_stage_radix_size, batch, degree__, start_prime_idx,
      second_radix_size / per_thread_ntt_size,
      inverse_power_of_roots_div_two__.data(),
      inverse_scaled_power_of_roots_div_two__.data(), primes__.data(),
      out.data());
  Intt8PointPerThreadPhase1OoP<<<grid, (first_stage_radix_size / 8) * pad,
                                 (first_stage_radix_size + pad + 1) * pad *
                                     sizeof(uint64_t)>>>(
      out.data(), 1, batch, degree__, start_prime_idx, pad,
      first_stage_radix_size / 8, inverse_power_of_roots_div_two__.data(),
      inverse_scaled_power_of_roots_div_two__.data(), primes__.data(),
      out.data());
  CudaCheckError();
  return out;
}

void Context::FromNTTInplace(word64 *op1, int start_prime_idx, int batch, const bool verbose) const {
  dim3 gridDim(2048);
  dim3 blockDim(256);
  // NTT is made up of two NTT stages. Here the 'first' means the first NTT
  // stage in NTT (so it becomes second in view of iNTT).
  const int per_thread_ntt_size = 8;
  const int first_stage_radix_size = 256;
  const int second_radix_size = degree__ / first_stage_radix_size;
  const int pad = 4;
  const int per_thread_storage =
      blockDim.x * per_thread_ntt_size * sizeof(word64);
  Intt8PointPerThreadPhase2OoP<<<gridDim, blockDim, per_thread_storage>>>(
      op1, first_stage_radix_size, batch, degree__, start_prime_idx,
      second_radix_size / per_thread_ntt_size,
      inverse_power_of_roots_div_two__.data(),
      inverse_scaled_power_of_roots_div_two__.data(), primes__.data(), op1);
  Intt8PointPerThreadPhase1OoP<<<gridDim, (first_stage_radix_size / 8) * pad,
                                 (first_stage_radix_size + pad + 1) * pad *
                                     sizeof(uint64_t)>>>(
      op1, 1, batch, degree__, start_prime_idx, pad, first_stage_radix_size / 8,
      inverse_power_of_roots_div_two__.data(),
      inverse_scaled_power_of_roots_div_two__.data(), primes__.data(), op1);
  CudaCheckError();
}

DeviceVector Context::FromNTT(const DeviceVector &in,
                              const DeviceVector &scale_constants,
                              const DeviceVector &scale_constants_shoup) const {
  DeviceVector out;
  out.resize(in.size());
  const int batch = in.size() / degree__;
  const int start_prime_idx = 0;
  dim3 grid(2048);
  dim3 block(256);
  const int per_thread_ntt_size = 8;
  const int first_stage_radix_size = 256;
  const int second_radix_size = degree__ / first_stage_radix_size;
  const int pad = 4;
  const int per_thread_storage = block.x * per_thread_ntt_size * sizeof(word64);
  Intt8PointPerThreadPhase2OoP<<<grid, block, per_thread_storage>>>(
      in.data(), first_stage_radix_size, batch, degree__, start_prime_idx,
      second_radix_size / per_thread_ntt_size,
      inverse_power_of_roots_div_two__.data(),
      inverse_scaled_power_of_roots_div_two__.data(), primes__.data(),
      out.data());
  Intt8PointPerThreadPhase1OoPWithEpilogue<<<
      grid, (first_stage_radix_size / 8) * pad,
      (first_stage_radix_size + pad + 1) * pad * sizeof(uint64_t)>>>(
      out.data(), 1, batch, degree__, start_prime_idx, pad,
      first_stage_radix_size / 8, inverse_power_of_roots_div_two__.data(),
      inverse_scaled_power_of_roots_div_two__.data(), primes__.data(),
      out.data(), scale_constants.data(), scale_constants_shoup.data());
  CudaCheckError();
  return out;
}

DeviceVector Context::ModUp(const DeviceVector &in) const {
  const int num_moduli = in.size() / degree__;
  const int beta = ceil(float(num_moduli) / float(alpha__));
  const int num_moduli_after_modup = num_moduli + alpha__;
  // std::cout << "Context ModUp params: " << alpha__ << " " << beta << std::endl;
  // std::cout <<"\t" << num_moduli << " " << alpha__ << std::endl;
  // if ((num_moduli+1) % alpha__ != 0) {
  //   std::cout << num_moduli << " " << alpha__ << std::endl;
  //   throw std::logic_error("Size does not match; apply RNS-decompose first.");
  // }
  DeviceVector raised;
  raised.resize(num_moduli_after_modup * degree__ * beta);
  // std::cout << "TODO: reimplement batched\n";
  if (is_modup_batched)
    ModUpBatchImpl(in, raised, beta);
  else {
    for (int i = 0; i < beta; ++i) {
      ModUpImpl(in.data() + (alpha__ * degree__ * i),
                raised.data() + (num_moduli_after_modup * degree__) * i, i, num_moduli);
    }
  }
  return raised;
}

void Context::ModUpImpl(const word64 *from, word64 *to, int idx, const int num_original_input_limbs) const {
  const int num_moduli_after_modup = num_original_input_limbs + alpha__;
  // if (alpha__ == 1) {
  //   DeviceVector temp(param__.max_num_moduli_ * degree__);
  //   cudaMemcpyAsync(temp.data() + idx * degree__, from, 8 * degree__,
  //                   cudaMemcpyDeviceToDevice, cudaStreamLegacy);
  //   FromNTTInplace(temp.data(), idx, alpha__);
  //   word64 *target = to;
  //   ModUpLengthIsOne(temp.data() + idx * degree__, from, idx,
  //                    num_moduli_after_modup, target);
  //   ToNTTInplaceExceptSomeRange(target, 0, num_moduli_after_modup, idx, alpha__);
  // } else {
    size_t begin_idx = idx * alpha__;
    int num_raised_input_limbs = alpha__;
    const int beta = ceil(float(num_original_input_limbs)/(float)alpha__);
    if (idx == beta-1) 
      num_raised_input_limbs = num_original_input_limbs - idx*alpha__;
    cudaMemcpyAsync(to + (degree__ * begin_idx), from, 8 * num_raised_input_limbs * degree__, cudaMemcpyDeviceToDevice, cudaStreamLegacy);
    // if (idx == 0) {
    //   DeviceVector from_vec(degree__*num_raised_input_limbs);
    //   cudaMemcpyAsync(from_vec.data(), to, 8*num_raised_input_limbs*degree__, cudaMemcpyDeviceToDevice, cudaStreamLegacy);
    //   HostVector from_host(from_vec);
    //   for (uint32_t limbInd = 0; limbInd < alpha__; limbInd++)
    //     std::cout << "ModUpImpl (gpu) input limb (eval) " << limbInd << " : " << from_host[limbInd*degree__] << std::endl; 
    // }
    FromNTTInplace(to, begin_idx, num_raised_input_limbs);  // these are always over standard limbs
    // if (idx == 0) {
    //   DeviceVector from_vec(degree__*num_raised_input_limbs);
    //   cudaMemcpyAsync(from_vec.data(), to, 8*num_raised_input_limbs*degree__, cudaMemcpyDeviceToDevice, cudaStreamLegacy);
    //   HostVector from_host(from_vec);
    //   for (uint32_t limbInd = 0; limbInd < alpha__; limbInd++)
    //     std::cout << "input limb (coeff) " << limbInd << " : " << from_host[limbInd*degree__] << std::endl; 
    // }
    const DeviceVector &hat_inverse_vec = hat_inverse_vec__[num_original_input_limbs-1].at(idx);
    const DeviceVector &hat_inverse_vec_psinv = hat_inverse_vec_shoup__[num_original_input_limbs-1].at(idx);
    ConstMultBatch(to, hat_inverse_vec, hat_inverse_vec_psinv, begin_idx, num_raised_input_limbs, to);  // these are always over standard limbs

    ModUpMatMul(to + degree__ * begin_idx, idx, to, num_original_input_limbs);  // this handles the shifted limbs
    
    HostVector prime_inds(num_original_input_limbs + alpha__);
    assert(prime_inds.size() == num_original_input_limbs + alpha__);
    for (size_t i = 0; i < num_original_input_limbs; i++) prime_inds[i] = i;
    for (size_t i = 0; i < alpha__; i++) prime_inds[num_original_input_limbs+i] = param__.chain_length_ + i;

    DeviceVector prime_inds_device(prime_inds);

    ToNTTInplaceExceptSomeRange(to, 0, num_moduli_after_modup, begin_idx, num_raised_input_limbs, prime_inds_device);
    // This NTT has three pieces. The first is from 0 to the start of the input limbs
    // ToNTTInplace(to, 0, begin_idx);
    // The second is from the end of the input limbs to the end of the end of the standard basis
    // ToNTTInplace(to, begin_idx+num_raised_input_limbs, num_original_input_limbs - (begin_idx+num_raised_input_limbs));
    // The third piece this is the extension limbs
    // ToNTTInplace(to - num_original_input_limbs, )
    cudaMemcpyAsync(to + (degree__ * begin_idx), from, 8 * num_raised_input_limbs * degree__, cudaMemcpyDeviceToDevice, cudaStreamLegacy);
  // }
}

__global__ void constMultBatch_(size_t degree, const word64 *primes,
                                const word64 *op1, const word64 *op2,
                                const word64 *op2_psinv,
                                const int start_prime_idx, const int batch,
                                word64 *to) {
  STRIDED_LOOP_START(degree * batch, i);
  const int op2_idx = i / degree;
  const int prime_idx = op2_idx + start_prime_idx;
  const auto prime = primes[prime_idx];
  word64 out = mul_and_reduce_shoup(op1[start_prime_idx * degree + i],
                                    op2[op2_idx], op2_psinv[op2_idx], prime);
  if (out >= prime) out -= prime;
  to[start_prime_idx * degree + i] = out;
  STRIDED_LOOP_END;
}

void Context::ConstMultBatch(const word64 *op1, const DeviceVector &op2,
                             const DeviceVector &op2_psinv, int start_prime_idx,
                             int batch, word64 *res) const {
  assert(op2.size() == op2_psinv.size());
  assert(op2.size() == batch);
  const int block_dim = 256;
  const int grid_dim = degree__ * batch / block_dim;
  constMultBatch_<<<grid_dim, block_dim>>>(degree__, primes__.data(), op1,
                                           op2.data(), op2_psinv.data(),
                                           start_prime_idx, batch, res);
}

__global__ void constMultBatchModDown_(
  size_t degree, const word64 *primes,
  const word64 *op1, const int start_limb_idx,
  const word64 *op2, const word64 *op2_psinv,
  const int start_prime_idx, const int batch, word64 *to) {
  STRIDED_LOOP_START(degree * batch, i);
  const int op2_idx = i / degree;
  const int prime_idx = op2_idx + start_prime_idx;
  const auto prime = primes[prime_idx];
  word64 out = mul_and_reduce_shoup(op1[start_limb_idx * degree + i],
                                    op2[op2_idx], op2_psinv[op2_idx], prime);
  if (out >= prime) out -= prime;
  to[start_limb_idx * degree + i] = out;
  STRIDED_LOOP_END;
}

void Context::ConstMultBatchModDown(const word64 *op1, const int start_limb_idx,
  const DeviceVector &op2, const DeviceVector &op2_psinv, 
  int start_prime_idx, int batch, word64 *res) const {
  assert(op2.size() == op2_psinv.size());
  assert(op2.size() == batch);
  const int block_dim = 256;
  const int grid_dim = degree__ * batch / block_dim;
  constMultBatchModDown_<<<grid_dim, block_dim>>>(
    degree__, primes__.data(), op1, start_limb_idx, op2.data(), op2_psinv.data(), 
    start_prime_idx, batch, res);
}

__device__ uint128_t4 AccumulateInModUp(const word64 *ptr, const int degree,
                                        const word64 *hat_mod_end,
                                        const int start_length,
                                        const int degree_idx,
                                        const int hat_mod_end_idx) {
  uint128_t4 accum{0};
  for (int i = 0; i < start_length; i++) {
    const uint64_t op2 = hat_mod_end[hat_mod_end_idx * start_length + i];
    uint128_t4 out;
    // cache streaming?
    // or, texture?
    uint64_t op1_x, op1_y, op1_z, op1_w;
    asm("{\n\t"
        "ld.global.v2.u64 {%0, %1}, [%2];\n\t"
        "}"
        : "=l"(op1_x), "=l"(op1_y)
        : "l"(ptr + i * degree + degree_idx));

    out.x = mult_64_64_128(op1_x, op2);
    inplace_add_128_128(out.x, accum.x);
    out.y = mult_64_64_128(op1_y, op2);
    inplace_add_128_128(out.y, accum.y);
    asm("{\n\t"
        "ld.global.v2.u64 {%0, %1}, [%2];\n\t"
        "}"
        : "=l"(op1_z), "=l"(op1_w)
        : "l"(ptr + i * degree + degree_idx + 2));
    out.z = mult_64_64_128(op1_z, op2);
    inplace_add_128_128(out.z, accum.z);
    out.w = mult_64_64_128(op1_w, op2);
    inplace_add_128_128(out.w, accum.w);
  }
  return accum;
}

// Applied loop unroll.
// Mod-up `ptr` to `to`.
// hat_mod_end[:end_length], ptr[:start_length][:degree],
// to[:start_length+end_length][:degree] ptr` is entirely overlapped with `to`.
__global__ void modUpStepTwoKernel(
  const word64 *ptr, const int begin_idx, const int degree, const word64 *primes,
  const word64 *barrett_ratios, const word64 *barrett_Ks,
  const word64 *hat_mod_end, const int hat_mod_end_size,
  const word64 start_length, const word64 end_length, word64 *to) {
  constexpr const int unroll_number = 4;
  extern __shared__ word64 s_hat_mod_end[];
  for (int i = threadIdx.x; i < hat_mod_end_size; i += blockDim.x) {
    s_hat_mod_end[i] = hat_mod_end[i];
  }
  __syncthreads();
  STRIDED_LOOP_START((degree * end_length + unroll_number - 1) / unroll_number, i);
  const int degree_idx = unroll_number * (i / end_length);
  const int hat_mod_end_idx = i % end_length;
  // Leap over the overlapped region.
  const int out_prime_idx =
      hat_mod_end_idx + ((hat_mod_end_idx >= begin_idx) ? start_length : 0);
  assert(degree_idx < degree);
  uint128_t4 accum = AccumulateInModUp(ptr, degree, s_hat_mod_end, start_length, degree_idx, hat_mod_end_idx);
  const auto prime = primes[out_prime_idx];
  const auto barret_ratio = barrett_ratios[out_prime_idx];
  const auto barret_k = barrett_Ks[out_prime_idx];
  {
    uint64_t out =
        barret_reduction_128_64(accum.x, prime, barret_ratio, barret_k);
    uint64_t out2 =
        barret_reduction_128_64(accum.y, prime, barret_ratio, barret_k);
    asm("st.cs.global.v2.u64 [%0],{%1, %2};" ::"l"(to + out_prime_idx * degree +
                                                   degree_idx),
        "l"(out), "l"(out2));
  }
  {
    uint64_t out =
        barret_reduction_128_64(accum.z, prime, barret_ratio, barret_k);
    uint64_t out2 =
        barret_reduction_128_64(accum.w, prime, barret_ratio, barret_k);
    asm("st.cs.global.v2.u64 [%0],{%1, %2};" ::"l"(to + out_prime_idx * degree +
                                                   degree_idx + 2),
        "l"(out), "l"(out2));
  }
  STRIDED_LOOP_END;
}

__global__ void ModDownLengthOne(const uint64_t *poly, const int current_level,
                                 const KernelParams ring, uint64_t *out) {
  const int degree = ring.degree;
  STRIDED_LOOP_START(degree * current_level, i);
  const int prime_idx = i / degree;
  const uint64_t start_prime = ring.primes[current_level];
  const uint64_t end_prime = ring.primes[prime_idx];
  const int degree_idx = i % degree;
  if (end_prime < start_prime) {
    barret_reduction_64_64(poly[degree_idx], out[i], ring.primes[prime_idx],
                           ring.barret_ratio[prime_idx],
                           ring.barret_k[prime_idx]);
  } else {
    out[i] = poly[degree_idx];
  }
  STRIDED_LOOP_END;
}

__global__ void ModDownKernel(
  KernelParams ring, const word64 *ptr,
  const word64 *hat_mod_end, const int hat_mod_end_size,
  const word64 start_length, const word64 end_length, word64 *to) {
  constexpr const int unroll_number = 4;
  extern __shared__ word64 s_hat_mod_end[];
  for (int i = threadIdx.x; i < hat_mod_end_size; i += blockDim.x) {
    s_hat_mod_end[i] = hat_mod_end[i];
  }
  __syncthreads();
  STRIDED_LOOP_START(
      (ring.degree * end_length + unroll_number - 1) / unroll_number, i);
  const int degree_idx = unroll_number * (i / end_length);
  const int out_prime_idx = i % end_length;
  uint128_t4 accum = AccumulateInModUp(ptr, ring.degree, hat_mod_end,
                                       start_length, degree_idx, out_prime_idx);
  const auto prime = ring.primes[out_prime_idx];
  const auto barret_ratio = ring.barret_ratio[out_prime_idx];
  const auto barret_k = ring.barret_k[out_prime_idx];
  {
    uint64_t out =
        barret_reduction_128_64(accum.x, prime, barret_ratio, barret_k);
    uint64_t out2 =
        barret_reduction_128_64(accum.y, prime, barret_ratio, barret_k);
    asm("st.cs.global.v2.u64 [%0],{%1, %2};" ::"l"(
            to + out_prime_idx * ring.degree + degree_idx),
        "l"(out), "l"(out2));
  }
  {
    uint64_t out =
        barret_reduction_128_64(accum.z, prime, barret_ratio, barret_k);
    uint64_t out2 =
        barret_reduction_128_64(accum.w, prime, barret_ratio, barret_k);
    asm("st.cs.global.v2.u64 [%0],{%1, %2};" ::"l"(
            to + out_prime_idx * ring.degree + degree_idx + 2),
        "l"(out), "l"(out2));
  }
  STRIDED_LOOP_END;
}

__global__ void negateInplace_(size_t degree, size_t log_degree, size_t batch,
                               const uint64_t *primes, uint64_t *op) {
  STRIDED_LOOP_START(batch * degree, i);
  const int prime_idx = i >> log_degree;
  const uint64_t prime = primes[prime_idx];
  if (op[i] != 0) op[i] = prime - op[i];
  STRIDED_LOOP_END;
}

void Context::NegateInplace(word64 *op1, const int batch) const {
  const int block_dim = 256;
  const int grid_dim = degree__ * batch / block_dim;
  negateInplace_<<<grid_dim, block_dim>>>(degree__, log2(degree__), batch,
                                          primes__.data(), op1);
}

__global__ void subInplace_(size_t degree, size_t log_degree, size_t batch,
                            const uint64_t *primes, uint64_t *op1,
                            const uint64_t *op2) {
  STRIDED_LOOP_START(batch * degree, i);
  const int prime_idx = i >> log_degree;
  const uint64_t prime = primes[prime_idx];
  if (op1[i] >= op2[i]) {
    op1[i] -= op2[i];
  } else {
    op1[i] = prime - (op2[i] - op1[i]);
  }
  STRIDED_LOOP_END;
}

void Context::SubInplace(word64 *op1, const word64 *op2,
                         const int batch) const {
  const int block_dim = 256;
  const int grid_dim = degree__ * batch / block_dim;
  subInplace_<<<grid_dim, block_dim>>>(degree__, log2(degree__), batch,
                                       primes__.data(), op1, op2);
}

// Input should be base limbs + extension limbs
// output should just be base limbs
// void Context::ModDown(DeviceVector &from_v, DeviceVector &to_v, long target_chain_idx) const {
void Context::ModDown(DeviceVector &from_v, DeviceVector &to_v) const {
  // if (target_chain_idx > param__.chain_length_)
    // throw std::logic_error("Target chain index is too big.");
  
  const int input_num_limbs = from_v.size()/degree__;
  const int output_num_limbs = input_num_limbs - param__.alpha_;

  to_v.resize(output_num_limbs * degree__);

  const int gap = param__.chain_length_ - output_num_limbs;
  // const int start_length = param__.max_num_moduli_ - output_num_limbs;
  const int start_length = param__.alpha_;
  const int end_length = output_num_limbs;
  const int block_dim = 256;
  const int grid_dim = degree__ * end_length / block_dim;
  // if (start_length == 1) {
  //   // std::cout << "ModDown start_length 1\n";
  //   FromNTTInplace(from_v, end_length, start_length);
  //   ModDownLengthOne<<<grid_dim, block_dim>>>(
  //       from_v.data() + end_length * degree__, end_length, GetKernelParams(),
  //       to_v.data());
  // } else {
    // std::cout << "ModDown start length: " << start_length << std::endl;
    // std::cout << "ModDown end length: " << end_length << std::endl;

    // shift primes to always be over the final primes
    // FromNTTInplace(from_v.data(), end_length, start_length);
    // when we shift by param__.chain_length_, we want to land on from_v.data() + end_length
    FromNTTInplace((from_v.data()) - degree__*(param__.chain_length_ - end_length), param__.chain_length_, start_length);

    // these are always the same
    const DeviceVector &hat_inverse_vec = hat_inverse_vec_moddown__.at(gap);
    const DeviceVector &hat_inverse_vec_psinv = hat_inverse_vec_shoup_moddown__.at(gap);

    // shift the primes again
    ConstMultBatchModDown(from_v.data(), end_length, hat_inverse_vec, hat_inverse_vec_psinv,
                   param__.chain_length_, start_length, from_v.data());

    auto ptr = from_v.data() + degree__ * end_length;
    const auto &prod_q_i_mod_q_j = prod_q_i_mod_q_j_moddown__[gap];
    ModDownKernel<<<grid_dim, block_dim, prod_q_i_mod_q_j.size() * sizeof(word64)>>>(
        GetKernelParams(), ptr, prod_q_i_mod_q_j.data(),
        start_length * end_length, start_length, end_length, to_v.data());
  // }
  const auto &prod_inv = prod_inv_moddown__.at(gap);
  const auto &prod_inv_psinv = prod_inv_shoup_moddown__.at(gap);
  // fuse four functions: NTT, sub, negate, and const-mult.
  if (is_moddown_fused)
    ToNTTInplaceFused(to_v, from_v, prod_inv, prod_inv_psinv);
  else {
    ToNTTInplace(to_v.data(), 0, end_length);
    SubInplace(to_v.data(), from_v.data(), end_length);
    NegateInplace(to_v.data(), end_length);
    ConstMultBatch(to_v.data(), prod_inv, prod_inv_psinv, 0, end_length, to_v.data());
  }
}

// __global__ void switchModulus_(word64* op, const size_t size, const word64 oldModulus, const word64 newModulus) {
//     const int j = blockIdx.x * blockDim.x + threadIdx.x;

//     const auto halfQ{oldModulus >> 1};
//     if (newModulus > oldModulus) {
//         const auto diff{newModulus - oldModulus};
//         // for (size_t j = 0; j < size; ++j) {
//             if (op[j] > halfQ)
//                 op[j] += diff;
//         // }
//     }
//     else {
//         const auto diff{newModulus - (oldModulus % newModulus)};
//         // for (size_t j = 0; j < size; ++j) {
//             if (op[j] > halfQ)
//                 op[j] += diff;
//             if (op[j] >= newModulus)
//                 op[j] %= newModulus;
//         // }
//     }
// }

__global__ void switchModulus_BigNewModulus_(word64* op, const word64 halfQ, const word64 diff) {
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (op[j] > halfQ)
    op[j] += diff;
}

__global__ void switchModulus_SmallNewModulus_(word64* op, const uint32_t limbInd, const word64 halfQ, const word64 diff, word64 newModulus, const word64* barret_ratio__, const word64* barret_k__) {
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (op[j] > halfQ)
      op[j] += diff;
  if (op[j] >= newModulus)
      // op[j] %= newModulus;
      barret_reduction_64_64(op[j], op[j], newModulus, barret_ratio__[limbInd], barret_k__[limbInd]);
}

// all limbs
__global__ void switchModulusFused_(
  word64* to, const word64* from, const uint32_t degree, 
  const word64 oldModulus, 
  const word64* primes, const word64* barret_ratio__, const word64* barret_k__) {

  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t limbInd = j / degree;
  const uint32_t dataInd = j & (degree - 1);  // degree is a power of 2

  const word64 newModulus = primes[limbInd];

  const word64 halfQ = oldModulus >> 1;

  if (newModulus > oldModulus) {
      const auto diff{newModulus - oldModulus};
      if (from[dataInd] > halfQ)
        to[j] = from[dataInd] + diff;
      else
        to[j] = from[dataInd];
    } else {
      const auto diff{newModulus - (oldModulus % newModulus)};
      if (from[dataInd] > halfQ)
        to[j] = from[dataInd] + diff;
      else
        to[j] = from[dataInd];
      if (to[j] >= newModulus)
          barret_reduction_64_64(to[j], to[j], newModulus, barret_ratio__[limbInd], barret_k__[limbInd]);
    }
}


// void Context::SwitchModulus(word64* op, const size_t size, const word64 oldModulus, const word64 newModulus) const {
//   // const int block_dim = 256;
//   // const int grid_dim = size / block_dim;
//   // std::cout << "TODO: get a good block and grid size for switchModulus\n";
//   switchModulus_<<<1, 1>>>(op, size, oldModulus, newModulus);
// }

// Just drop the last limb
void Context::Rescale(const DeviceVector &from_v, DeviceVector &to_v) const {
  
  const int inputLimbs = from_v.size()/degree__;
  assert(inputLimbs <= param__.chain_length_);
  assert(inputLimbs > 1);
    // throw std::logic_error("Target chain index is too big.");

  // const int start_length = inputLimbs;
  const int end_length = inputLimbs-1;
  
  // std::cout << "Rescale start length: " << start_length << std::endl;
  // std::cout << "Rescale end length: " << end_length << std::endl;
  DeviceVector last_from_limb(degree__);
  cudaMemcpyAsync(last_from_limb.data(), from_v.data() + end_length*degree__, degree__ * sizeof(DeviceVector::Dtype),
                  cudaMemcpyDefault, last_from_limb.stream_);
  FromNTTInplaceShiftedPointer(last_from_limb, end_length, 1);

  // FromNTTInplace(from_v.data(), end_length, 1);

  const int gap = param__.chain_length_ - inputLimbs;
  const auto &prod_inv = prod_inv_rescale__.at(gap);
  const auto &prod_inv_psinv = prod_inv_shoup_rescale__.at(gap);

  to_v.resize(end_length * degree__);

  // const int block_dim = 256;
  // const int grid_dim = degree__ * end_length / block_dim;

  // std::cout << "TODO: get a good block and grid size for switchModulus\n";

  const auto oldModulus = param__.primes_[end_length];
  // const auto halfQ{oldModulus >> 1};
  // for (size_t i = 0; i < end_length; i++) {
  //   cudaMemcpyAsync(to_v.data() + i*degree__, last_from_limb.data(), degree__ * sizeof(DeviceVector::Dtype), cudaMemcpyDefault, last_from_limb.stream_);
  //   // cudaMemcpy(to_v.data() + i*degree__, from_v.data() + (end_length)*degree__, degree__ * sizeof(DeviceVector::Dtype), cudaMemcpyDefault);
    
  //   // switchModulus_<<<degree__ / 256, 256>>>(to_v.data()+i*degree__, degree__, param__.primes_[end_length], param__.primes_[i]);
  //   const auto newModulus = param__.primes_[i];
  //   if (newModulus > oldModulus) {
  //     const auto diff{newModulus - oldModulus};
  //     switchModulus_BigNewModulus_<<<degree__ / 256, 256>>>(to_v.data()+i*degree__, halfQ, diff);
  //   } else {
  //     const auto diff{newModulus - (oldModulus % newModulus)};
  //     switchModulus_SmallNewModulus_<<<degree__ / 256, 256>>>(to_v.data()+i*degree__, i, halfQ, diff, newModulus, barret_ratio__.data(), barret_k__.data());
  //   }
  // }

  switchModulusFused_<<<degree__ * end_length / 256, 256>>>(to_v.data(), last_from_limb.data(), degree__, oldModulus, primes__.data(), barret_ratio__.data(), barret_k__.data());

  // fuse four functions: NTT, sub, negate, and const-mult.
  if (is_rescale_fused)
    ToNTTInplaceFused(to_v, from_v, prod_inv, prod_inv_psinv);
  else {
    ToNTTInplace(to_v.data(), 0, end_length);
    SubInplace(to_v.data(), from_v.data(), end_length);
    NegateInplace(to_v.data(), end_length);
    ConstMultBatch(to_v.data(), prod_inv, prod_inv_psinv, 0, end_length, to_v.data());
  }
}

void Context::Rescale(const Ciphertext &from_ct, Ciphertext &to_ct) const {
  Rescale(from_ct.bx__, to_ct.bx__);
  Rescale(from_ct.ax__, to_ct.ax__);
}

void Context::Rescale(const CtAccurate &from_ct, CtAccurate &to_ct) const {
  Rescale(from_ct.bx__, to_ct.bx__);
  Rescale(from_ct.ax__, to_ct.ax__);

  to_ct.level = from_ct.level + 1;
  to_ct.noiseScaleDeg = from_ct.noiseScaleDeg - 1;
  const uint32_t from_ct_limbs = from_ct.ax__.size() / degree__;
  to_ct.scalingFactor = from_ct.scalingFactor / param__.primes_[from_ct_limbs-1];
}

Ciphertext Context::Rescale(const Ciphertext& from) const {
  Ciphertext to; Rescale(from, to); return to;
}

CtAccurate Context::Rescale(const CtAccurate& from) const {
  CtAccurate to; Rescale(from, to); return to;
}

Ciphertext Context::DropLimbs(const Ciphertext& ct, const uint32_t numDropLimbs) const {
  Ciphertext out;
  const uint32_t inputLimbs = ct.ax__.size()/param__.degree_;
  const uint32_t outputLimbs = inputLimbs-numDropLimbs;
  out.ax__.resize(degree__*outputLimbs);
  out.bx__.resize(degree__*outputLimbs);

  cudaMemcpy(out.ax__.data(), ct.ax__.data(), outputLimbs*degree__*sizeof(word64), cudaMemcpyHostToDevice);
  cudaMemcpy(out.bx__.data(), ct.bx__.data(), outputLimbs*degree__*sizeof(word64), cudaMemcpyHostToDevice);

  return out;
}

CtAccurate Context::DropLimbs(const CtAccurate& ct, const uint32_t numDropLimbs) const {
  CtAccurate out;
  const uint32_t inputLimbs = ct.ax__.size()/param__.degree_;
  const uint32_t outputLimbs = inputLimbs-numDropLimbs;
  out.ax__.resize(degree__*outputLimbs);
  out.bx__.resize(degree__*outputLimbs);

  cudaMemcpy(out.ax__.data(), ct.ax__.data(), outputLimbs*degree__*sizeof(word64), cudaMemcpyHostToDevice);
  cudaMemcpy(out.bx__.data(), ct.bx__.data(), outputLimbs*degree__*sizeof(word64), cudaMemcpyHostToDevice);

  out.level = ct.level + numDropLimbs;
  out.scalingFactor = ct.scalingFactor;
  out.noiseScaleDeg = ct.noiseScaleDeg;

  return out;
}

DeviceVector Context::ToNTT(const DeviceVector& in) const {
  DeviceVector out(in);
  ToNTTInplace(out.data(), 0, in.size()/degree__);
  return out;
}

void Context::ToNTTInplace(word64 *op, int start_prime_idx, int batch) const {
  dim3 gridDim(2048);
  dim3 blockDim(256);
  const int per_thread_ntt_size = 8;
  const int first_stage_radix_size = 256;
  const int second_radix_size = degree__ / first_stage_radix_size;
  const int pad = 4;
  const int per_thread_storage =
      blockDim.x * per_thread_ntt_size * sizeof(word64);
  Ntt8PointPerThreadPhase1<<<gridDim, (first_stage_radix_size / 8) * pad,
                             (first_stage_radix_size + pad + 1) * pad *
                                 sizeof(uint64_t)>>>(
      op, 1, batch, degree__, start_prime_idx, pad,
      first_stage_radix_size / per_thread_ntt_size, power_of_roots__.data(),
      power_of_roots_shoup__.data(), primes__.data());
  Ntt8PointPerThreadPhase2<<<gridDim, blockDim.x, per_thread_storage>>>(
      op, first_stage_radix_size, batch, degree__, start_prime_idx,
      second_radix_size / per_thread_ntt_size, power_of_roots__.data(),
      power_of_roots_shoup__.data(), primes__.data());
  CudaCheckError();
}

DeviceVector Context::copyLimbData(const DeviceVector& from, const int num_orig_limbs) const {
  DeviceVector to; to.resize(num_orig_limbs+alpha__);
  cudaMemcpy(to.data(), from.data(), num_orig_limbs * sizeof(word64), cudaMemcpyHostToDevice);
  cudaMemcpy(to.data() + num_orig_limbs, from.data() + param__.chain_length_, alpha__ * sizeof(word64), 
      cudaMemcpyHostToDevice);
  // cudaMemcpyAsync(to.data(), from.data(), num_orig_limbs * sizeof(word64), cudaMemcpyHostToDevice, to.stream_);
  // cudaMemcpyAsync(to.data() + num_orig_limbs, from.data() + param__.chain_length_, alpha__ * sizeof(word64), 
  //     cudaMemcpyHostToDevice, to.stream_);
  return to;
}

void Context::ModUpMatMul(const word64 *ptr, int beta_idx, word64 *to, const int num_orig_limbs) const {
  // std::cout << "In ModUpMatMul\n";
  const int num_moduli_after_modup = num_orig_limbs + alpha__;
  const int unroll_factor = 4;

  int num_raised_input_limbs = alpha__;
  int beta = ceil((float)num_orig_limbs/(float)alpha__);
  if (beta_idx == beta-1)
    num_raised_input_limbs = num_orig_limbs - beta_idx*alpha__;

  const int start_length = num_raised_input_limbs;
  const int begin_idx = beta_idx * alpha__;
 
  const int end_length = num_moduli_after_modup - start_length;

  DeviceVector curr_primes = copyLimbData(primes__, num_orig_limbs);
  // assert(curr_primes == primes__);
  DeviceVector curr_barret_ratio = copyLimbData(barret_ratio__, num_orig_limbs);
  // assert(curr_barret_ratio == barret_ratio__);
  DeviceVector curr_barret_k = copyLimbData(barret_k__, num_orig_limbs);
  // assert(curr_barret_k == barret_k__);

  // std::cout << "finished copying limb data\n";

  long grid_dim{degree__ * end_length / 256 / unroll_factor};
  int block_dim{256};
  const auto &prod_q_i_mod_q_j = prod_q_i_mod_q_j__[num_orig_limbs-1].at(beta_idx);
  // std::cout << "launching modUpStepTwoKernel\n";
  modUpStepTwoKernel<<<grid_dim, block_dim, prod_q_i_mod_q_j.size() * sizeof(word64)>>>(
      ptr, begin_idx, degree__, 
      // primes__.data(), barret_ratio__.data(), barret_k__.data(), 
      curr_primes.data(), curr_barret_ratio.data(), curr_barret_k.data(),
      prod_q_i_mod_q_j.data(), prod_q_i_mod_q_j.size(),
      start_length, end_length, to);
}

void Context::ModUpBatchImpl(const DeviceVector &from, DeviceVector &to, int beta) const {
  // std::cout << "In ModUpBatchImpl\n";

  const int num_input_limbs = from.size()/param__.degree_;

  const int num_moduli_after_modup = num_input_limbs + alpha__;

  HostVector prime_inds(num_input_limbs + alpha__);
  assert(prime_inds.size() == num_input_limbs + alpha__);
  for (size_t i = 0; i < num_input_limbs; i++) prime_inds[i] = i;
  for (size_t i = 0; i < alpha__; i++) prime_inds[num_input_limbs+i] = param__.chain_length_ + i;

  DeviceVector prime_inds_device(prime_inds);
  // std::cout << "ModUp Primes\n";
  // for (auto prime_ind : prime_inds) std::cout << param__.primes_[prime_ind] << std::endl;

  // {
  //   HostVector from_host(from);
  //   for (uint32_t limbInd = 0; limbInd < alpha__; limbInd++)
  //     std::cout << "ModUpImpl (gpu) input limb (eval) " << limbInd << ": " << from_host[limbInd*degree__] << std::endl; 
  // }

  // if (alpha__ == 1) {
  //   DeviceVector from_after_intt = FromNTT(from);  // no scaling when alpha = 1
  //   for (int idx = 0; idx < beta; idx++) {
  //     const int begin_idx = idx * alpha__;
  //     word64 *target = to.data() + idx * num_moduli_after_modup * degree__;
  //     int end_length = num_moduli_after_modup - alpha__;
  //     ModUpLengthIsOne(from_after_intt.data() + degree__ * idx,
  //                      from.data() + degree__ * idx, begin_idx, end_length + 1,
  //                      target);
  //     ToNTTInplaceExceptSomeRange(target, 0, num_moduli_after_modup, begin_idx, alpha__, prime_inds_device);
  //   }
  // } else {
    // Apply iNTT and multiplies \hat{q}_i (fast base conversion)
    DeviceVector temp = FromNTT(from, hat_inverse_vec_batched__[num_input_limbs-1],
                                hat_inverse_vec_shoup_batched__[num_input_limbs-1]);
    
    // std::cout << "mod down inverse ntt finished\n";

    for (int idx = 0; idx < beta; idx++) {
      const int begin_idx = idx * alpha__;
      word64 *target = to.data() + idx * num_moduli_after_modup * degree__;
      
      int num_raised_input_limbs = alpha__;
      if (idx == beta-1) num_raised_input_limbs = num_input_limbs - idx*alpha__;

      // if (idx == 0) {
      //   std::cout << "ModUp index " << idx << std::endl;
      //   assert(begin_idx == 0);
      //   std::cout << "number of input limbs: " << num_raised_input_limbs << std::endl;
      //   HostVector temp_host(temp);
      //   for (uint32_t limbInd = 0; limbInd < num_raised_input_limbs; limbInd++)
      //     std::cout << "input limb (coeff)" << limbInd << " : " << temp_host[limbInd*degree__] << std::endl; 
      // }

      ModUpMatMul(temp.data() + degree__ * begin_idx, idx, target, num_input_limbs);
      // std::cout << "ModUpMatMul finished\n";
      ToNTTInplaceExceptSomeRange(target, 0, num_moduli_after_modup, begin_idx, num_raised_input_limbs, prime_inds_device);
      // std::cout << "mod down forward ntt finished\n";
      cudaMemcpy(
          target + begin_idx * degree__, from.data() + idx * alpha__ * degree__,
          8 * num_raised_input_limbs * degree__, cudaMemcpyDeviceToDevice);
    }
  // }
}

void Context::ModUpLengthIsOne(const word64 *ptr_after_intt,
                               const word64 *ptr_before_intt, int begin_idx,
                               int end_length, word64 *to) const {
  int block_dim{256};
  long grid_dim{degree__ * end_length / block_dim};
  modUpStepTwoSimple<<<grid_dim, block_dim>>>(
      ptr_after_intt, ptr_before_intt, begin_idx, degree__, primes__.data(),
      barret_ratio__.data(), barret_k__.data(), end_length, to);
}

void Context::ToNTTInplaceExceptSomeRange(
  word64 *op, int start_prime_idx, int batch, 
  int excluded_range_start, int excluded_range_size,
  const DeviceVector& prime_inds) const {

  const int excluded_range_end = excluded_range_start + excluded_range_size;
  if (excluded_range_start < start_prime_idx ||
      excluded_range_end > (start_prime_idx + batch)) {
    throw "Excluded range in NTT is invalid.";
  }
  dim3 grid(2048);
  dim3 block(256);
  const int per_thread_ntt_size = 8;
  const int first_stage_radix_size = 256;
  const int second_radix_size = degree__ / first_stage_radix_size;
  const int pad = 4;
  const int per_thread_storage = block.x * per_thread_ntt_size * sizeof(word64);
  Ntt8PointPerThreadPhase1ExcludeSomeRange<<<
      grid, (first_stage_radix_size / 8) * pad,
      (first_stage_radix_size + pad + 1) * pad * sizeof(uint64_t)>>>(
      op, 1, batch, degree__, start_prime_idx, excluded_range_start,
      excluded_range_end, pad, first_stage_radix_size / per_thread_ntt_size,
      prime_inds.data(),
      power_of_roots__.data(), power_of_roots_shoup__.data(), primes__.data());
  Ntt8PointPerThreadPhase2ExcludeSomeRange<<<grid, block.x,
                                             per_thread_storage>>>(
      op, first_stage_radix_size, batch, degree__, start_prime_idx,
      excluded_range_start, excluded_range_end,
      second_radix_size / per_thread_ntt_size, 
      prime_inds.data(),
      power_of_roots__.data(), power_of_roots_shoup__.data(), primes__.data());
  CudaCheckError();
}

void Context::ToNTTInplaceFused(DeviceVector &op1, const DeviceVector &op2,
                                const DeviceVector &epilogue,
                                const DeviceVector &epilogue_) const {
  dim3 gridDim(2048);
  // dim3 gridDim(8192);
  // dim3 gridDim(32768);
  dim3 blockDim(256);
  // dim3 blockDim(64);
  const int per_thread_ntt_size = 8;
  const int first_stage_radix_size = 256;
  // const int first_stage_radix_size = 512;
  const int second_radix_size = degree__ / first_stage_radix_size;
  const int pad = 4;
  const int per_thread_storage =
      blockDim.x * per_thread_ntt_size * sizeof(word64);  // this might be too small. this might be limiting the # of blocks
  const int start_prime_idx = 0;
  const int batch = op1.size() / degree__;
  Ntt8PointPerThreadPhase1<<<gridDim, (first_stage_radix_size / 8) * pad,
                             (first_stage_radix_size + pad + 1) * pad *
                                 sizeof(uint64_t)>>>(
      op1.data(), 1, batch, degree__, start_prime_idx, pad,
      first_stage_radix_size / per_thread_ntt_size, power_of_roots__.data(),
      power_of_roots_shoup__.data(), primes__.data());
  Ntt8PointPerThreadPhase2FusedWithSubNegateConstMult<<<gridDim, blockDim.x,
                                                        per_thread_storage>>>(
      op1.data(), first_stage_radix_size, batch, degree__, start_prime_idx,
      second_radix_size / per_thread_ntt_size, power_of_roots__.data(),
      power_of_roots_shoup__.data(), primes__.data(), op2.data(),
      epilogue.data(), epilogue_.data());
  CudaCheckError();
}

__global__ void sumAndReduceFused(
  const word64 *modup_out, const int degree, const int length, const int batch,
  const int eval_length,
  const word64 *eval_ax, const word64 *eval_bx,
  const word64 * prime_inds,
  const word64 *primes, const word64 *barret_ks, const word64 *barret_ratios, 
  word64 *dst_ax, word64 *dst_bx) {

  STRIDED_LOOP_START(degree * length, i);
  const int stride_between_batch = degree * length;
  const int stride_between_eval_batch = degree * eval_length;
  uint128_t accum_ax{0, 0};
  uint128_t accum_bx{0, 0};
  for (int batch_idx = 0; batch_idx < batch; batch_idx++) {
    const int idx = i + stride_between_batch * batch_idx;
    const int idx_extended = prime_inds[i/degree]*degree + (i % degree) + stride_between_eval_batch*batch_idx;
    // assert(idx == idx_extended);
    const word64 op1 = modup_out[idx];
    const word64 op2_ax = eval_ax[idx_extended];
    const auto mul_ax = mult_64_64_128(op1, op2_ax);
    accum_ax += mul_ax;
    const word64 op2_bx = eval_bx[idx_extended];
    const auto mul_bx = mult_64_64_128(op1, op2_bx);
    accum_bx += mul_bx;
  }
  const int prime_idx = prime_inds[i / degree];
  const auto prime = primes[prime_idx];
  const auto barret_ratio = barret_ratios[prime_idx];
  const auto barret_k = barret_ks[prime_idx];
  const auto res_ax =
      barret_reduction_128_64(accum_ax, prime, barret_ratio, barret_k);
  const auto res_bx =
      barret_reduction_128_64(accum_bx, prime, barret_ratio, barret_k);
  dst_ax[i] = res_ax;
  dst_bx[i] = res_bx;
  STRIDED_LOOP_END;
}

template <bool Accum>
__global__ void mult_(const word64 *modup_out, 
  const word64 *eval_poly_ax, const word64 *eval_poly_bx, const int degree,
  const int length, uint128_t *accum_ptr_ax, uint128_t *accum_ptr_bx) {

  STRIDED_LOOP_START(degree * length, i);
  const word64 op1 = modup_out[i];
  const word64 op2_ax = eval_poly_ax[i];
  const word64 op2_bx = eval_poly_bx[i];
  const auto mul_ax = mult_64_64_128(op1, op2_ax);
  const auto mul_bx = mult_64_64_128(op1, op2_bx);
  if (Accum) {
    accum_ptr_ax[i] += mul_ax;
    accum_ptr_bx[i] += mul_bx;
  } else {
    accum_ptr_ax[i] = mul_ax;
    accum_ptr_bx[i] = mul_bx;
  }
  STRIDED_LOOP_END;
}

__global__ void Reduce(
  const uint128_t *accum, const int degree, const int length, 
  const word64 *primes, const word64 *barret_ks, const word64 *barret_ratios,
  word64 *res) {

  STRIDED_LOOP_START(degree * length, i);
  const int prime_idx = i / degree;
  const auto prime = primes[prime_idx];
  const auto barret_ratio = barret_ratios[prime_idx];
  const auto barret_k = barret_ks[prime_idx];
  const auto res_ax =
      barret_reduction_128_64(accum[i], prime, barret_ratio, barret_k);
  res[i] = res_ax;
  STRIDED_LOOP_END;
}

void Context::KeySwitch(const DeviceVector &modup_out, const EvaluationKey &evk, DeviceVector &sum_ax, DeviceVector &sum_bx) const {

  assert(modup_out.size() > 0 && modup_out.size() % degree__ == 0);
  const int total_length = modup_out.size() / degree__;
  // assert(total_length % param__.max_num_moduli_ == 0);
  // const int beta = ceil((float)total_length / (float)param__.max_num_moduli_);
  // const int length = param__.max_num_moduli_;

  int beta = param__.dnum_;
  int length = 0;
  while (beta > 0) {
    if (total_length % beta != 0) {
      beta -= 1;
      continue;
    }
    // possible correct beta
    int cand_num_modup_limbs = total_length / beta;
    // check that these input limbs correspond to alpha__
    if (cand_num_modup_limbs > param__.max_num_moduli_ || beta != ceil((float)(cand_num_modup_limbs-alpha__) / (float)alpha__)) {
      beta -= 1;
      continue;
    }

    length = cand_num_modup_limbs;
    break;
  }

  assert(beta > 0 && length > 0);
  const int num_original_input_limbs = length - alpha__;

  // std::cout << "KeySwitch params\n";
  // std::cout << "\t" << total_length << " " << beta << " " << length << std::endl;

  const int gridDim = 1024;
  const int blockDim = 256;
  const int size_after_reduced = length * degree__;
  sum_ax.resize(size_after_reduced);
  sum_bx.resize(size_after_reduced);

  const auto &eval_ax = evk.getAxDevice();
  const auto &eval_bx = evk.getBxDevice();
  assert(param__.max_num_moduli_ == eval_ax.size()/(param__.dnum_ * degree__));

  HostVector prime_inds(num_original_input_limbs + alpha__);
  for (size_t i = 0; i < num_original_input_limbs; i++) prime_inds[i] = i;
  for (size_t i = 0; i < alpha__; i++) prime_inds[num_original_input_limbs+i] = param__.chain_length_ + i;

  DeviceVector prime_inds_device(prime_inds);

  if (is_keyswitch_fused) {
    sumAndReduceFused<<<gridDim, blockDim>>>(
        modup_out.data(), degree__, length, beta, param__.max_num_moduli_, eval_ax.data(), eval_bx.data(), 
        prime_inds_device.data(),
        primes__.data(), barret_k__.data(), barret_ratio__.data(), sum_ax.data(), sum_bx.data());
  } else {
    const int quad_word_size_byte = sizeof(uint128_t);
    DeviceBuffer accum_ax(modup_out.size() * quad_word_size_byte);
    DeviceBuffer accum_bx(modup_out.size() * quad_word_size_byte);
    auto accum_ax_ptr = (uint128_t *)accum_ax.data();
    auto accum_bx_ptr = (uint128_t *)accum_bx.data();
    mult_<false><<<gridDim, blockDim>>>(modup_out.data(), eval_ax.data(),
                                        eval_bx.data(), degree__, length,
                                        accum_ax_ptr, accum_bx_ptr);
    for (int i = 1; i < beta; i++) {
      const auto d2_ptr = modup_out.data() + i * degree__ * length;
      const auto ax_ptr = eval_ax.data() + i * degree__ * param__.max_num_moduli_;
      const auto bx_ptr = eval_bx.data() + i * degree__ * param__.max_num_moduli_;
      mult_<true><<<gridDim, blockDim>>>(d2_ptr, ax_ptr, bx_ptr, degree__, length, accum_ax_ptr, accum_bx_ptr);
    }
    Reduce<<<gridDim, blockDim>>>(accum_ax_ptr, degree__, length, primes__.data(), 
                                  barret_k__.data(), barret_ratio__.data(), sum_ax.data());
    Reduce<<<gridDim, blockDim>>>(accum_bx_ptr, degree__, length, primes__.data(), 
                                  barret_k__.data(), barret_ratio__.data(), sum_bx.data());
  }
}

__global__ void hadamardMultAndAddBatch_(
    const KernelParams ring, const word64 **ax_addr, const word64 **bx_addr,
    const word64 **mx_addr, const int fold_size, const size_t size,
    const int log_degree, word64 *out_ax, word64 *out_bx) {
  STRIDED_LOOP_START(size, idx);
  const int prime_idx = idx >> log_degree;
  const uint64_t prime = ring.primes[prime_idx];
  uint128_t sum_ax = {0};
  uint128_t sum_bx = {0};
  for (int fold_idx = 0; fold_idx < fold_size; fold_idx++) {
    const word64 *ax = ax_addr[fold_idx];
    const word64 *bx = bx_addr[fold_idx];
    const word64 *mx = mx_addr[fold_idx];
    const word64 mx_element = mx[idx];
    sum_ax += mult_64_64_128(ax[idx], mx_element);
    sum_bx += mult_64_64_128(bx[idx], mx_element);
  }
  out_ax[idx] = barret_reduction_128_64(
      sum_ax, prime, ring.barret_ratio[prime_idx], ring.barret_k[prime_idx]);
  out_bx[idx] = barret_reduction_128_64(
      sum_bx, prime, ring.barret_ratio[prime_idx], ring.barret_k[prime_idx]);
  if (out_ax[idx] > prime) out_ax[idx] -= prime;
  if (out_bx[idx] > prime) out_bx[idx] -= prime;
  STRIDED_LOOP_END;
}

void Context::hadamardMultAndAddBatch(const std::vector<const word64 *> ax_addr,
                                      const std::vector<const word64 *> bx_addr,
                                      const std::vector<const word64 *> mx_addr,
                                      const int num_primes,
                                      DeviceVector &out_ax,
                                      DeviceVector &out_bx) const {
  assert(ax_addr.size() == bx_addr.size() && ax_addr.size() == mx_addr.size());
  if (out_ax.size() != (size_t)num_primes * degree__ ||
      out_bx.size() != (size_t)num_primes * degree__)
    throw std::logic_error("Output has no proper size");
  const int fold_size = ax_addr.size();
  const int each_operand_size = num_primes * degree__;
  size_t addr_buffer_size = fold_size * sizeof(word64 *);
  const DeviceBuffer d_ax_addr(ax_addr.data(), addr_buffer_size, cudaStreamLegacy);
  const DeviceBuffer d_bx_addr(bx_addr.data(), addr_buffer_size, cudaStreamLegacy);
  const DeviceBuffer d_mx_addr(mx_addr.data(), addr_buffer_size, cudaStreamLegacy);
  const int block_dim = 256;
  const int grid_dim = each_operand_size / block_dim;
  hadamardMultAndAddBatch_<<<grid_dim, block_dim>>>(
      GetKernelParams(), (const word64 **)d_ax_addr.data(),
      (const word64 **)d_bx_addr.data(), (const word64 **)d_mx_addr.data(),
      fold_size, each_operand_size, param__.log_degree_, out_ax.data(),
      out_bx.data());
}

__global__ void hadamardMultFused_(size_t degree, size_t log_degree,
                              size_t num_primes, const uint64_t* primes,
                              const uint64_t* barret_ratio,
                              const uint64_t* barret_k, const uint64_t* op1,
                              const uint64_t* op2, const uint64_t* mx,
                              uint64_t* op1_out, uint64_t* op2_out) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int prime_idx = i >> log_degree;
  const uint64_t prime = primes[prime_idx];
  uint64_t mx_element = mx[i];
  uint128_t out_op1 = mult_64_64_128(op1[i], mx_element);
  uint128_t out_op2 = mult_64_64_128(op2[i], mx_element);
  op1_out[i] = barret_reduction_128_64(out_op1, prime, barret_ratio[prime_idx],
                                   barret_k[prime_idx]);
  op2_out[i] = barret_reduction_128_64(out_op2, prime, barret_ratio[prime_idx],
                                   barret_k[prime_idx]);
}

__global__ void constantMultFusedInPlace_(size_t degree, size_t log_degree,
                              size_t num_primes, const uint64_t* primes,
                              const uint64_t* barret_ratio, const uint64_t* barret_k, 
                              uint64_t* op1, uint64_t* op2, const uint64_t* mx) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int prime_idx = i >> log_degree;
  const uint64_t prime = primes[prime_idx];
  uint64_t mx_element = mx[prime_idx];
  uint128_t out_op1 = mult_64_64_128(op1[i], mx_element);
  uint128_t out_op2 = mult_64_64_128(op2[i], mx_element);
  op1[i] = barret_reduction_128_64(out_op1, prime, barret_ratio[prime_idx], barret_k[prime_idx]);
  op2[i] = barret_reduction_128_64(out_op2, prime, barret_ratio[prime_idx], barret_k[prime_idx]);
}

__global__ void integerMultFusedInPlace_(size_t degree, size_t log_degree,
                              size_t num_primes, const uint64_t* primes,
                              const uint64_t* barret_ratio, const uint64_t* barret_k, 
                              uint64_t* op1, uint64_t* op2, const uint64_t m) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int prime_idx = i >> log_degree;
  const uint64_t prime = primes[prime_idx];
  uint128_t out_op1 = mult_64_64_128(op1[i], m);
  uint128_t out_op2 = mult_64_64_128(op2[i], m);
  op1[i] = barret_reduction_128_64(out_op1, prime, barret_ratio[prime_idx], barret_k[prime_idx]);
  op2[i] = barret_reduction_128_64(out_op2, prime, barret_ratio[prime_idx], barret_k[prime_idx]);
}

__global__ void evalMultFused_(
  size_t degree, size_t log_degree, size_t num_primes, const uint64_t* primes,
  const uint64_t* barret_ratio, const uint64_t* barret_k, 
  const uint64_t* op1_0, const uint64_t* op1_1, const uint64_t* op2_0, const uint64_t* op2_1, 
  uint64_t* op0_out, uint64_t* op1_out, uint64_t* op2_out) {

  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int prime_idx = i >> log_degree;
  const uint64_t prime = primes[prime_idx];

  uint128_t out_op0 = mult_64_64_128(op1_0[i], op2_0[i]);
  uint128_t out_op1 = mult_64_64_128(op1_0[i], op2_1[i]); 
  out_op1 += mult_64_64_128(op1_1[i], op2_0[i]);
  uint128_t out_op2 = mult_64_64_128(op1_1[i], op2_1[i]);

  op0_out[i] = barret_reduction_128_64(out_op0, prime, barret_ratio[prime_idx], barret_k[prime_idx]);
  op1_out[i] = barret_reduction_128_64(out_op1, prime, barret_ratio[prime_idx], barret_k[prime_idx]);
  op2_out[i] = barret_reduction_128_64(out_op2, prime, barret_ratio[prime_idx], barret_k[prime_idx]);
}

__global__ void evalMultPlainFused_(
  size_t log_degree, 
  const uint64_t* primes, const uint64_t* barret_ratio, const uint64_t* barret_k, 
  const uint64_t* c_0, const uint64_t* c_1, const uint64_t* m_op, 
  uint64_t* out_0, uint64_t* out_1) {

  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int prime_idx = i >> log_degree;
  const uint64_t prime = primes[prime_idx];

  uint128_t out_op0 = mult_64_64_128(c_0[i], m_op[i]);
  uint128_t out_op1 = mult_64_64_128(c_1[i], m_op[i]); 

  out_0[i] = barret_reduction_128_64(out_op0, prime, barret_ratio[prime_idx], barret_k[prime_idx]);
  out_1[i] = barret_reduction_128_64(out_op1, prime, barret_ratio[prime_idx], barret_k[prime_idx]);
}

__global__ void evalSquareFused_(
  size_t degree, size_t log_degree, size_t num_primes, const uint64_t* primes,
  const uint64_t* barret_ratio, const uint64_t* barret_k, 
  const uint64_t* op_0, const uint64_t* op_1,
  uint64_t* op0_out, uint64_t* op1_out, uint64_t* op2_out) {

  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int prime_idx = i >> log_degree;
  const uint64_t prime = primes[prime_idx];

  // constexpr uint128_t two_const{0,2};

  uint128_t out_op0 = mult_64_64_128(op_0[i], op_0[i]);
  uint128_t out_op1 = mult_64_64_128(op_0[i], op_1[i]);
  out_op1 += out_op1; 
  // out_op1 += mult_64_64_128(op_1[i], op_0[i]);
  uint128_t out_op2 = mult_64_64_128(op_1[i], op_1[i]);

  op0_out[i] = barret_reduction_128_64(out_op0, prime, barret_ratio[prime_idx], barret_k[prime_idx]);
  op1_out[i] = barret_reduction_128_64(out_op1, prime, barret_ratio[prime_idx], barret_k[prime_idx]);
  op2_out[i] = barret_reduction_128_64(out_op2, prime, barret_ratio[prime_idx], barret_k[prime_idx]);
}

void Context::EvalMult(
  const Ciphertext& ct1, const Ciphertext& ct2, 
  DeviceVector& res0, DeviceVector& res1, DeviceVector& res2) const {

  const auto &op1_0 = ct1.getBxDevice();
  const auto &op1_1 = ct1.getAxDevice();
  const auto &op2_0 = ct2.getBxDevice();
  const auto &op2_1 = ct2.getAxDevice();

  // std::cout << "op1 numLimbs: " << op1_0.size() / degree__ << std::endl;
  // std::cout << "op2 numLimbs: " << op2_0.size() / degree__ << std::endl;

  // assert(op1_0.size() == op1_1.size());
  // assert(op2_0.size() == op2_1.size());
  // assert(op1_0.size() == op2_1.size());

  res0.resize(op1_0.size());
  res1.resize(op1_0.size());
  res2.resize(op1_0.size());

  int num_primes = op1_0.size() / degree__;

  // size_t i = 0;
  // for (uint32_t prime_idx = 0; prime_idx < num_primes; prime_idx++) {
  //   for (uint32_t dataInd = 0; dataInd < degree__; dataInd++) {

  //     const uint64_t prime = primes__.data()[prime_idx];

  //     uint128_t out_op0 = mult_64_64_128(op1_0.data()[i], op2_0.data()[i]);
  //     uint128_t out_op1 = mult_64_64_128(op1_0.data()[i], op2_1.data()[i]); 
  //     out_op1 += mult_64_64_128(op1_1.data()[i], op2_0.data()[i]);
  //     uint128_t out_op2 = mult_64_64_128(op1_1.data()[i], op2_1.data()[i]);

  //     res0.data()[i] = barret_reduction_128_64(out_op0, prime, barret_ratio__.data()[prime_idx], barret_k__.data()[prime_idx]);
  //     res1.data()[i] = barret_reduction_128_64(out_op1, prime, barret_ratio__.data()[prime_idx], barret_k__.data()[prime_idx]);
  //     res2.data()[i] = barret_reduction_128_64(out_op2, prime, barret_ratio__.data()[prime_idx], barret_k__.data()[prime_idx]);

  //     i++;
  //   }
  // }

  // std::cout << "max num primes: " << num_primes << std::endl;
  // std::cout << "max index: " << degree__ * num_primes << std::endl;

  evalMultFused_<<<degree__ * num_primes / 256, 256>>>(
      degree__, param__.log_degree_, num_primes, primes__.data(), 
      barret_ratio__.data(), barret_k__.data(), 
      op1_0.data(), op1_1.data(), op2_0.data(), op2_1.data(), 
      res0.data(), res1.data(), res2.data());
  CudaCheckError();
}

// Last num
CtAccurate Context::EvalMultPlainExt(const CtAccurate& ct, const PtAccurate& pt) const {

  const uint32_t numLimbs = ct.bx__.size()/degree__;
  const uint32_t numRegularLimbs = numLimbs - param__.alpha_;

  CtAccurate out;
  out.ax__.resize(numLimbs*degree__);
  out.bx__.resize(numLimbs*degree__);

  const int block_dim = 256;
  const int regular_grid_dim = degree__ * numRegularLimbs / block_dim;
  evalMultPlainFused_<<<regular_grid_dim, block_dim>>>(
    param__.log_degree_, primes__.data(), barret_ratio__.data(), barret_k__.data(), 
    ct.ax__.data(), ct.bx__.data(), pt.mx__.data(), out.ax__.data(), out.bx__.data());

  const auto data_shift = numRegularLimbs*degree__;
  const int ext_grid_dim = degree__ * param__.alpha_ / block_dim;
  evalMultPlainFused_<<<ext_grid_dim, block_dim>>>(
    param__.log_degree_, 
    primes__.data() + param__.chain_length_, barret_ratio__.data() + param__.chain_length_, barret_k__.data() + param__.chain_length_, 
    ct.ax__.data() + data_shift, ct.bx__.data() + data_shift, pt.mx__.data() + data_shift, 
    out.ax__.data() + data_shift, out.bx__.data() + data_shift);

  out.level = ct.level;
  out.noiseScaleDeg = ct.noiseScaleDeg + pt.noiseScaleDeg;
  out.scalingFactor = ct.scalingFactor * pt.scalingFactor;

  return out;
}

void Context::EvalMult(
  const CtAccurate& ct1, const CtAccurate& ct2, 
  DeviceVector& res0, DeviceVector& res1, DeviceVector& res2) const {

  // assert(ct1.level == 1);
  // assert(ct2.level == 1);

  const auto &op1_0 = ct1.getBxDevice();
  const auto &op1_1 = ct1.getAxDevice();
  const auto &op2_0 = ct2.getBxDevice();
  const auto &op2_1 = ct2.getAxDevice();

  // std::cout << "op1 numLimbs: " << op1_0.size() / degree__ << std::endl;
  // std::cout << "op2 numLimbs: " << op2_0.size() / degree__ << std::endl;

  // assert(op1_0.size() == op1_1.size());
  // assert(op2_0.size() == op2_1.size());
  // assert(op1_0.size() == op2_1.size());

  res0.resize(op1_0.size());
  res1.resize(op1_0.size());
  res2.resize(op1_0.size());

  int num_primes = op1_0.size() / degree__;

  // std::cout << "max num primes: " << num_primes << std::endl;
  // std::cout << "max index: " << degree__ * num_primes << std::endl;

  evalMultFused_<<<degree__ * num_primes / 256, 256>>>(
      degree__, param__.log_degree_, num_primes, primes__.data(), 
      barret_ratio__.data(), barret_k__.data(), 
      op1_0.data(), op1_1.data(), op2_0.data(), op2_1.data(), 
      res0.data(), res1.data(), res2.data());
  // CudaCheckError();
}

void Context::EvalSquare(
  const Ciphertext& ct, 
  DeviceVector& res0, DeviceVector& res1, DeviceVector& res2) const {

  const auto &op_0 = ct.getBxDevice();
  const auto &op_1 = ct.getAxDevice();

  res0.resize(op_0.size());
  res1.resize(op_0.size());
  res2.resize(op_0.size());

  int num_primes = op_0.size() / degree__;

  evalSquareFused_<<<degree__ * num_primes / 256, 256>>>(
      degree__, param__.log_degree_, num_primes, primes__.data(), 
      barret_ratio__.data(), barret_k__.data(), 
      op_0.data(), op_1.data(),
      res0.data(), res1.data(), res2.data());
}

// duplicate function from above...
void Context::EvalSquare(
  const CtAccurate& ct, 
  DeviceVector& res0, DeviceVector& res1, DeviceVector& res2) const {

  const auto &op_0 = ct.getBxDevice();
  const auto &op_1 = ct.getAxDevice();

  res0.resize(op_0.size());
  res1.resize(op_0.size());
  res2.resize(op_0.size());

  int num_primes = op_0.size() / degree__;

  evalSquareFused_<<<degree__ * num_primes / 256, 256>>>(
      degree__, param__.log_degree_, num_primes, primes__.data(), 
      barret_ratio__.data(), barret_k__.data(), 
      op_0.data(), op_1.data(),
      res0.data(), res1.data(), res2.data());
}


__global__ void setMonomialSmallPower_(word64* op, const word64 index, const word64 degree) {
  const int limbInd = blockIdx.x * blockDim.x + threadIdx.x;
  op[limbInd*degree + index] = 1;
}

__global__ void setMonomialLargePower_(word64* op, const word64 index, const word64 degree, const word64* primes) {
  const int limbInd = blockIdx.x * blockDim.x + threadIdx.x;
  op[limbInd*degree + index] = primes[limbInd] - 1;
}

void Context::MultByMonomialInPlace(CtAccurate& ct1, const uint32_t power) const{
  // create one-hot coefficient vector
  // negative if power >= degree
  // take NTT and multiply
  // NO adjustment to the scaling factor or level

  const uint32_t numLimbs = ct1.ax__.size()/degree__;

  DeviceVector coeffs(degree__*numLimbs); coeffs.setConstant(0);
  const uint32_t index = power % degree__; 
  if (power >= degree__) {
    // for (uint32_t i = 0; i < numLimbs; i++)
    //   coeffs.data()[i*degree__ + index] = param__.primes_[i] - 1;
    setMonomialLargePower_<<<1, numLimbs>>>(coeffs.data(), index, degree__, primes__.data());
  } else {
    setMonomialSmallPower_<<<1, numLimbs>>>(coeffs.data(), index, degree__);
    // for (uint32_t i = 0; i < numLimbs; i++)
    //   coeffs.data()[i*degree__ + index] = 1;
  }

  ToNTTInplace(coeffs.data(), 0, numLimbs);

  hadamardMultFused_<<<degree__ * numLimbs / 256, 256>>>(degree__, param__.log_degree_, numLimbs, primes__.data(), barret_ratio__.data(), barret_k__.data(), 
    ct1.ax__.data(), ct1.bx__.data(), coeffs.data(), ct1.ax__.data(), ct1.bx__.data());
}

void Context::PMult(const Ciphertext &ct, const Plaintext &pt, Ciphertext &out) const {
  const auto &op1 = ct.getAxDevice();
  const auto &op2 = ct.getBxDevice();
  const auto &mx = pt.getMxDevice();
  auto &op1_out = out.getAxDevice();
  auto &op2_out = out.getBxDevice();
  op1_out.resize(op1.size());
  op2_out.resize(op1.size());
  int num_primes = op1.size() / degree__;
  hadamardMultFused_<<<degree__ * num_primes / 256, 256>>>(
      degree__, param__.log_degree_, num_primes, primes__.data(), barret_ratio__.data(),
      barret_k__.data(), op1.data(), op2.data(), mx.data(), op1_out.data(),
      op2_out.data());
}

Ciphertext Context::EvalMultConst(const Ciphertext& ct, const DeviceVector& op) const {
  Ciphertext out(ct);
  EvalMultConstInPlace(out, op);
  return out;
}

CtAccurate Context::EvalMultConst(const CtAccurate& ct, const DeviceVector& op) const {
  CtAccurate out(ct);
  EvalMultConstInPlace(out, op);
  return out;
}

// assumes only regular limbs for now
void Context::EvalMultConstInPlace(Ciphertext& ct, const DeviceVector& op) const {
  auto &op1 = ct.getAxDevice();
  auto &op2 = ct.getBxDevice();
  int num_primes = op1.size() / degree__;
  constantMultFusedInPlace_<<<degree__ * num_primes / 256, 256>>>(
      degree__, param__.log_degree_, num_primes, primes__.data(), barret_ratio__.data(),
      barret_k__.data(), op1.data(), op2.data(), op.data());
}

void Context::EvalMultConstInPlace(CtAccurate& ct, const DeviceVector& op) const {
  auto &op1 = ct.getAxDevice();
  auto &op2 = ct.getBxDevice();
  int num_primes = op1.size() / degree__;
  constantMultFusedInPlace_<<<degree__ * num_primes / 256, 256>>>(
      degree__, param__.log_degree_, num_primes, primes__.data(), barret_ratio__.data(),
      barret_k__.data(), op1.data(), op2.data(), op.data());

  // ciphertext->SetNoiseScaleDeg(ciphertext->GetNoiseScaleDeg() + 1);
  ct.noiseScaleDeg += 1;

  // double scFactor = cryptoParams->GetScalingFactorReal(ciphertext->GetLevel());
  double scFactor = param__.m_scalingFactorsReal[ct.level];

  // ciphertext->SetScalingFactor(ciphertext->GetScalingFactor() * scFactor);
  ct.scalingFactor *= scFactor;
}

void Context::EvalMultIntegerInPlace(CtAccurate& ct, const uint64_t c) const {
  const uint32_t blockDim = 256;
  const uint32_t gridDim = ct.ax__.size() / blockDim;
  const uint32_t numLimbs = ct.ax__.size() / degree__;
  integerMultFusedInPlace_<<<gridDim, blockDim>>>(degree__, param__.log_degree_, numLimbs, primes__.data(), barret_ratio__.data(), barret_k__.data(), ct.ax__.data(), ct.bx__.data(), c);
}

__global__ void add_(const KernelParams params,
                     const int batch, const word64* op1, const word64* op2,
                     word64* op3) {
  STRIDED_LOOP_START(batch * params.degree, i);
  const int prime_idx = i >> params.log_degree;
  const uint64_t prime = params.primes[prime_idx];
  op3[i] = op1[i] + op2[i];
  if (prime - op3[i] >> 63) op3[i] -= prime;
  STRIDED_LOOP_END;
}

__global__ void addInPlace_(const KernelParams params,
                     const int batch, word64* op1, const word64* op2, const word64* primes) {
  STRIDED_LOOP_START(batch * params.degree, i);
  const int prime_idx = i >> params.log_degree;
  const uint64_t prime = primes[prime_idx];
  op1[i] += op2[i];
  if (prime - op1[i] >> 63) op1[i] -= prime;
  STRIDED_LOOP_END;
}

__global__ void sub_(const KernelParams params,
                     const int batch, const word64* op1, const word64* op2,
                     word64* op3) {
  STRIDED_LOOP_START(batch * params.degree, i);
  const int prime_idx = i >> params.log_degree;
  const uint64_t prime = params.primes[prime_idx];
  op3[i] = op1[i] + prime - op2[i];
  if (prime - op3[i] >> 63) op3[i] -= prime;
  STRIDED_LOOP_END;
}

double Context::GetAdjustScalar(const CtAccurate& ct1, const CtAccurate& ct2) const {
  
  if (ct1.level < ct2.level) {
    if (ct1.noiseScaleDeg == 2) {
      if (ct2.noiseScaleDeg == 2) {
        // rescale ct1
        const double scf1 = ct1.scalingFactor;
        const double scf2 = ct2.scalingFactor;
        const double scf  = param__.m_scalingFactorsReal[ct1.level];
        const uint64_t sizeQl1 = param__.chain_length_ - ct1.level;
        const double q1 = param__.m_dmoduliQ[sizeQl1 - 1];
        return scf2 / scf1 * q1 / scf;
      } else {
        const double scf1 = ct1.scalingFactor;
        const double scf2 = param__.m_scalingFactorsRealBig[ct2.level - 1];
        const double scf  = param__.m_scalingFactorsReal[ct1.level];
        const uint64_t sizeQl1 = param__.chain_length_ - ct1.level;
        const double q1   = param__.m_dmoduliQ[sizeQl1 - 1];
        return scf2 / scf1 * q1 / scf;
      } 
    } else {
      if (ct2.noiseScaleDeg == 2) {
        const double scf1 = ct1.scalingFactor;
        const double scf2 = ct2.scalingFactor;
        const double scf  = param__.m_scalingFactorsReal[ct1.level];
        return scf2 / scf1 / scf;
      } else {
        const double scf1 = ct1.scalingFactor;
        const double scf2 = param__.m_scalingFactorsRealBig[ct2.level - 1];
        const double scf  = param__.m_scalingFactorsReal[ct1.level];
        return scf2 / scf1 / scf;
      }
    } 
  } else if (ct1.level > ct2.level) {
    // rescale ct2
    return GetAdjustScalar(ct2, ct1);
    // if (ct2.noiseScaleDeg == 2) {
    //   if (ct1.noiseScaleDeg == 2) {
    //     const double scf1 = ct1.scalingFactor;
    //     const double scf2 = ct2.scalingFactor;
    //     const double scf  = param__.m_scalingFactorsReal[ct2.level];
    //     const uint64_t sizeQl2 = param__.chain_length_ - ct2.level;
    //     const double q2 = param__.m_dmoduliQ[sizeQl2 - 1];
    //     return scf1 / scf2 * q2 / scf;
    //   } else {

    //   }
    // } 
    
  } else 
    throw std::logic_error("levels are already balanced\n");
}

void Context::AddCore(
  const DeviceVector& op1_ax, const DeviceVector& op1_bx,
  const DeviceVector& op2_ax, const DeviceVector& op2_bx,
  DeviceVector& out_ax, DeviceVector& out_bx) const {
  out_ax.resize(op1_ax.size());
  out_bx.resize(op1_ax.size());
  const int length = op1_ax.size() / degree__;
  if (op1_ax.size() > op2_ax.size() || op1_ax.size() != out_ax.size())
    throw std::logic_error("op2 size too small");
  int blockDim_ = 256;
  int gridDim_ = op1_ax.size()/blockDim_;
  add_<<<gridDim_, blockDim_>>>(GetKernelParams(), length, op1_ax.data(), op2_ax.data(), out_ax.data());
  add_<<<gridDim_, blockDim_>>>(GetKernelParams(), length, op1_bx.data(), op2_bx.data(), out_bx.data());
}

void Context::AddCoreInPlace(DeviceVector& x1, const DeviceVector& x2) const {
  const int numLimbs = x1.size() / degree__;
  int blockDim_ = 256;
  int gridDim_ = x1.size()/blockDim_;
  addInPlace_<<<gridDim_, blockDim_>>>(GetKernelParams(), numLimbs, x1.data(), x2.data(), primes__.data());
}


// Assume ct1 and ct2 has the same size
// This high-level function only operates over regular limbs
void Context::Add(const Ciphertext &ct1, const Ciphertext &ct2, Ciphertext &out) const {
  const auto &op1_ax = ct1.getAxDevice();
  const auto &op2_ax = ct2.getAxDevice();
  const auto &op1_bx = ct1.getBxDevice();
  const auto &op2_bx = ct2.getBxDevice();
  auto &out_ax = out.getAxDevice();
  auto &out_bx = out.getBxDevice();
  AddCore(op1_ax, op1_bx, op2_ax, op2_bx, out_ax, out_bx);  
}

void Context::Add(const CtAccurate &ct1, const CtAccurate &ct2, CtAccurate &out) const {

  // balance scaling factor
  uint32_t maxLevel = ct1.level;
  double scalingFactor = ct1.scalingFactor;

  // std::cout << "noise scale deg: " << ct1.noiseScaleDeg << " " << ct2.noiseScaleDeg << std::endl;
  // assert(ct1.level == ct2.level);
  assert(ct1.level >= ct2.level);
  assert(ct1.noiseScaleDeg == ct2.noiseScaleDeg);
  // std::cout << "scalingFactor: " << ct1.scalingFactor << " " << ct2.scalingFactor << std::endl;
  // assert(ct1.scalingFactor == ct2.scalingFactor);
  
  const auto &op1_ax = ct1.getAxDevice();
  const auto &op2_ax = ct2.getAxDevice();
  const auto &op1_bx = ct1.getBxDevice();
  const auto &op2_bx = ct2.getBxDevice();
  auto &out_ax = out.getAxDevice();
  auto &out_bx = out.getBxDevice();

  AddCore(op1_ax, op1_bx, op2_ax, op2_bx, out_ax, out_bx);  

  out.level = maxLevel;
  out.scalingFactor = scalingFactor;
  out.noiseScaleDeg = ct1.noiseScaleDeg;
}

CtAccurate Context::Add(const CtAccurate &ct1, const CtAccurate &ct2) const {
  CtAccurate res;
  if (ct1.level >= ct2.level) Add(ct1, ct2, res);
  else Add(ct2, ct1, res);
  return res;
}

void Context::EvalAddInPlace(CtAccurate &ct1, const CtAccurate &ct2) const {

  // assert(ct1.level == ct2.level);
  assert(ct1.noiseScaleDeg == ct2.noiseScaleDeg);
  if (ct1.level != ct2.level) {
    if (ct1.level > ct2.level) {
      // this is handled naturally 
    } else {
      // add the other direction
      ct1 = Add(ct2, ct1);
      return;
    }
  }

  auto &op1_ax = ct1.getAxDevice();
  auto &op1_bx = ct1.getBxDevice();

  const auto &op2_ax = ct2.getAxDevice();
  const auto &op2_bx = ct2.getBxDevice();

  int length = op1_ax.size() / degree__;
  if (op1_ax.size() > op2_ax.size())
    // int length = op2_ax.size() / degree__;
    throw std::logic_error("op2 size too small");
  int gridDim_ = 2048;
  int blockDim_ = 256;
  addInPlace_<<<gridDim_, blockDim_>>>(GetKernelParams(), length, op1_ax.data(), op2_ax.data(), primes__.data());
  addInPlace_<<<gridDim_, blockDim_>>>(GetKernelParams(), length, op1_bx.data(), op2_bx.data(), primes__.data());
  // ct1.level = min(ct1.level, ct2.level);
}


void Context::EvalAddInPlaceExt(CtAccurate& ct1, const CtAccurate& ct2) const {
  const uint32_t numLimbs = ct1.bx__.size()/degree__;
  const uint32_t numRegularLimbs = numLimbs - param__.alpha_;

  const int block_dim = 256;
  const int regular_grid_dim = degree__ * numRegularLimbs / block_dim;

  const auto kernelParams = GetKernelParams();

  addInPlace_<<<regular_grid_dim, block_dim>>>(kernelParams, numRegularLimbs, ct1.ax__.data(), ct2.ax__.data(), primes__.data());
  addInPlace_<<<regular_grid_dim, block_dim>>>(kernelParams, numRegularLimbs, ct1.bx__.data(), ct2.bx__.data(), primes__.data());

  const auto data_shift = numRegularLimbs*degree__;
  const int ext_grid_dim = degree__ * param__.alpha_ / block_dim;
  addInPlace_<<<ext_grid_dim, block_dim>>>(kernelParams, param__.alpha_, 
    ct1.ax__.data() + data_shift, ct2.ax__.data() + data_shift, primes__.data() + param__.chain_length_);
  addInPlace_<<<ext_grid_dim, block_dim>>>(kernelParams, param__.alpha_, 
    ct1.bx__.data() + data_shift, ct2.bx__.data() + data_shift, primes__.data() + param__.chain_length_);
}

void Context::Sub(const Ciphertext &ct1, const Ciphertext &ct2, Ciphertext &out) const {
  const auto &op1_ax = ct1.getAxDevice();
  const auto &op2_ax = ct2.getAxDevice();
  const auto &op1_bx = ct1.getBxDevice();
  const auto &op2_bx = ct2.getBxDevice();
  auto &out_ax = out.getAxDevice();
  auto &out_bx = out.getBxDevice();
  out_ax.resize(op1_ax.size());
  out_bx.resize(op1_ax.size());
  const int length = op1_ax.size() / degree__;
  if (op1_ax.size() < op2_ax.size() || op1_ax.size() != out_ax.size()) {
    std::cout << op1_ax.size() << " " << op2_ax.size() << " " << out_ax.size() << std::endl;
    throw std::logic_error("Size does not match");
  }
  int gridDim_ = 2048;
  int blockDim_ = 256;
  sub_<<<gridDim_, blockDim_>>>(GetKernelParams(), length, op1_ax.data(),
                                op2_ax.data(), out_ax.data());
  sub_<<<gridDim_, blockDim_>>>(GetKernelParams(), length, op1_bx.data(),
                                op2_bx.data(), out_bx.data());
}

CtAccurate Context::Sub(const CtAccurate &ct1, const CtAccurate &ct2) const {
  CtAccurate out;
  out.level = ct1.level;
  out.scalingFactor = ct1.scalingFactor;
  out.noiseScaleDeg = ct1.noiseScaleDeg;

  const auto &op1_ax = ct1.getAxDevice();
  const auto &op2_ax = ct2.getAxDevice();
  const auto &op1_bx = ct1.getBxDevice();
  const auto &op2_bx = ct2.getBxDevice();
  auto &out_ax = out.getAxDevice();
  auto &out_bx = out.getBxDevice();
  out_ax.resize(op1_ax.size());
  out_bx.resize(op1_ax.size());
  const int length = op1_ax.size() / degree__;
  if (op1_ax.size() < op2_ax.size() || op1_ax.size() != out_ax.size()) {
    std::cout << op1_ax.size() << " " << op2_ax.size() << " " << out_ax.size() << std::endl;
    throw std::logic_error("Size does not match");
  }
  int gridDim_ = 2048;
  int blockDim_ = 256;
  sub_<<<gridDim_, blockDim_>>>(GetKernelParams(), length, op1_ax.data(),
                                op2_ax.data(), out_ax.data());
  sub_<<<gridDim_, blockDim_>>>(GetKernelParams(), length, op1_bx.data(),
                                op2_bx.data(), out_bx.data());

  return out;
}

void Context::SubInPlace(Ciphertext &ct1, const Ciphertext &ct2) const {
  auto &op1_ax = ct1.getAxDevice();
  const auto &op2_ax = ct2.getAxDevice();
  auto &op1_bx = ct1.getBxDevice();
  const auto &op2_bx = ct2.getBxDevice();
  const int length = op1_ax.size() / degree__;
  int gridDim_ = 2048;
  int blockDim_ = 256;
  sub_<<<gridDim_, blockDim_>>>(GetKernelParams(), length, op1_ax.data(),
                                op2_ax.data(), op1_ax.data());
  sub_<<<gridDim_, blockDim_>>>(GetKernelParams(), length, op1_bx.data(),
                                op2_bx.data(), op1_bx.data());
}

// duplicate as above
void Context::SubInPlace(CtAccurate &ct1, const CtAccurate &ct2) const {
  auto &op1_ax = ct1.getAxDevice();
  const auto &op2_ax = ct2.getAxDevice();
  auto &op1_bx = ct1.getBxDevice();
  const auto &op2_bx = ct2.getBxDevice();
  const int length = op1_ax.size() / degree__;
  int gridDim_ = 2048;
  int blockDim_ = 256;
  sub_<<<gridDim_, blockDim_>>>(GetKernelParams(), length, op1_ax.data(),
                                op2_ax.data(), op1_ax.data());
  sub_<<<gridDim_, blockDim_>>>(GetKernelParams(), length, op1_bx.data(),
                                op2_bx.data(), op1_bx.data());
}

__global__ void add_scalar_(const KernelParams params,
                     const int batch, const word64* op1, const word64* op2,
                     word64* op3) {
  STRIDED_LOOP_START(batch * params.degree, i);
  const int prime_idx = i >> params.log_degree;
  const uint64_t prime = params.primes[prime_idx];
  op3[i] = op1[i] + op2[prime_idx];
  if (prime - op3[i] >> 63) op3[i] -= prime;
  STRIDED_LOOP_END;
}

__global__ void sub_scalar_(const KernelParams params,
                     const int batch, const word64* op1, const word64* op2,
                     word64* op3) {
  STRIDED_LOOP_START(batch * params.degree, i);
  const int prime_idx = i >> params.log_degree;
  const uint64_t prime = params.primes[prime_idx];
  op3[i] = op1[i] + prime - op2[prime_idx];
  if (prime - op3[i] >> 63) op3[i] -= prime;
  STRIDED_LOOP_END;
}

__global__ void sub_from_scalar_(const KernelParams params,
                     const int batch, const word64* op1, const word64* op2,
                     word64* op3) {
  STRIDED_LOOP_START(batch * params.degree, i);
  const int prime_idx = i >> params.log_degree;
  const uint64_t prime = params.primes[prime_idx];
  op3[i] = op1[prime_idx] + prime - op2[i];
  if (prime - op3[i] >> 63) op3[i] -= prime;
  STRIDED_LOOP_END;
}

void Context::AddScalarInPlace(Ciphertext &ct, const word64* op) const {
  auto &ct_bx = ct.getBxDevice();
  const int length = ct_bx.size() / degree__;
  int gridDim_ = 2048;
  int blockDim_ = 256;
  add_scalar_<<<gridDim_, blockDim_>>>(GetKernelParams(), length, ct_bx.data(), op, ct_bx.data());
}

void Context::AddScalarInPlace(CtAccurate &ct, const word64* op) const {
  auto &ct_bx = ct.getBxDevice();
  const int length = ct_bx.size() / degree__;
  int gridDim_ = 2048;
  int blockDim_ = 256;
  add_scalar_<<<gridDim_, blockDim_>>>(GetKernelParams(), length, ct_bx.data(), op, ct_bx.data());
}

void Context::SubScalarInPlace(Ciphertext &ct, const word64* op) const {
  auto &ct_bx = ct.getBxDevice();
  const int length = ct_bx.size() / degree__;
  int gridDim_ = 2048;
  int blockDim_ = 256;
  sub_scalar_<<<gridDim_, blockDim_>>>(GetKernelParams(), length, ct_bx.data(), op, ct_bx.data());
}

// assume scaling and loading has already happened
// duplicate function as above
void Context::SubScalarInPlace(CtAccurate &ct, const word64* op) const {
  auto &ct_bx = ct.getBxDevice();
  const int length = ct_bx.size() / degree__;
  int gridDim_ = 2048;
  int blockDim_ = 256;
  sub_scalar_<<<gridDim_, blockDim_>>>(GetKernelParams(), length, ct_bx.data(), op, ct_bx.data());
}

void Context::SubScalarInPlace(const word64* op, CtAccurate &ct) const {
  auto &ct_bx = ct.getBxDevice();
  const int length = ct_bx.size() / degree__;
  int gridDim_ = 2048;
  int blockDim_ = 256;
  // sub_scalar_<<<gridDim_, blockDim_>>>(GetKernelParams(), length, ct_bx.data(), op, ct_bx.data());
  sub_from_scalar_<<<gridDim_, blockDim_>>>(GetKernelParams(), length, op, ct_bx.data(), ct_bx.data());
}

void Context::EnableMemoryPool() {
  if (pool__ == nullptr) {
    pool__ = std::make_shared<MemoryPool>(param__);
    pool__->UseMemoryPool(true);
  } else {
    throw std::logic_error("Enable memory pool twice?");
  }
}