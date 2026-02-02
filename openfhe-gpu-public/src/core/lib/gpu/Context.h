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
#include "Ciphertext.h"

namespace ckks {

struct KernelParams {
  const int degree;
  const word64* primes;
  const word64* barret_ratio;
  const word64* barret_k;
  const int log_degree;
};

// class Ciphertext;
// class CtAccurate;
// class Plaintext;
class EvaluationKey;
class MemoryPool;

class Context {
  friend class MultPtxtBatch;

 public:
  Context(const Parameter& param);
  Context(const Parameter& param, const std::vector<word64>& primitiveRoots);
  void setup(const Parameter &param, const std::vector<word64>& primitiveRoots);

  Ciphertext EvalMultAndRelin(const Ciphertext& ct1, const Ciphertext& ct2, const EvaluationKey& evk, const bool verbose = false) const;
  CtAccurate EvalMultAndRelin(const CtAccurate& ct1, const CtAccurate& ct2, const EvaluationKey& evk, const bool verbose = false) const;
  Ciphertext EvalMultAndRelinNoRescale(const Ciphertext& ct1, const Ciphertext& ct2, const EvaluationKey& evk) const;
  CtAccurate EvalMultAndRelinNoRescale(const CtAccurate& ct1, const CtAccurate& ct2, const EvaluationKey& evk) const;

  void EvalMult(const Ciphertext& ct1, const Ciphertext& ct2, DeviceVector& res0, DeviceVector& res1, DeviceVector& res2) const;
  void EvalMult(const CtAccurate& ct1, const CtAccurate& ct2, DeviceVector& res0, DeviceVector& res1, DeviceVector& res2) const;

  Ciphertext EvalSquareAndRelin(const Ciphertext& ct, const EvaluationKey& evk) const;
  void EvalSquareAndRelin(const Ciphertext& ct, const EvaluationKey& evk, Ciphertext& out) const;
  CtAccurate EvalSquareAndRelin(const CtAccurate& ct, const EvaluationKey& evk) const;
  Ciphertext EvalSquareAndRelinNoRescale(const Ciphertext& ct, const EvaluationKey& evk) const;
  CtAccurate EvalSquareAndRelinNoRescale(const CtAccurate& ct, const EvaluationKey& evk) const;

  void EvalSquare(const Ciphertext& ct, DeviceVector& res0, DeviceVector& res1, DeviceVector& res2) const;
  void EvalSquare(const CtAccurate& ct, DeviceVector& res0, DeviceVector& res1, DeviceVector& res2) const;

  void AutomorphismTransformInPlace(CtAccurate& in, const std::vector<uint32_t>& inds) const;
  void AutomorphismTransformInPlace(CtAccurate& in, const uint32_t auto_index) const;
  DeviceVector AutomorphismTransform(const DeviceVector& in, const uint32_t auto_index) const;
  DeviceVector AutomorphismTransform(const DeviceVector& in, const std::vector<uint32_t>& inds) const;

  CtAccurate EvalAtIndex(const CtAccurate& ct, const EvaluationKey& evk, const uint32_t auto_index) const;

  void AddScaledMessageTerm(DeviceVector& inner_prod_b, const DeviceVector& message_term) const;

  CtAccurate EvalMultPlainExt(const CtAccurate& ct, const PtAccurate& pt) const;

  void EvalAddInPlaceExt(CtAccurate& ct1, const CtAccurate& ct2) const;

  void MultByMonomialInPlace(CtAccurate& ct1, const uint32_t power) const;

  Ciphertext EvalMultConst(const Ciphertext& ct, const DeviceVector& op) const;
  void EvalMultConstInPlace(Ciphertext& ct, const DeviceVector& op) const;

  CtAccurate EvalMultConst(const CtAccurate& ct, const DeviceVector& op) const;
  void EvalMultConstInPlace(CtAccurate& ct, const DeviceVector& op) const;

  template <typename SchemeType, typename CryptoParamsType>
  inline CtAccurate EvalMultConstWithLoad(const CtAccurate& ct, const double op, 
    const SchemeType& scheme, const CryptoParamsType& cryptoParams) const {

    const uint32_t ct_num_elems = ct.ax__.size()/degree__;
    const auto op_scaled = scheme->GetElementForEvalMult(cryptoParams, ct.level, ct_num_elems, ct.noiseScaleDeg, op);
    const auto op_scaled_gpu = LoadIntegerVector(op_scaled);

    return EvalMultConst(ct, op_scaled_gpu);
  }

  template <typename SchemeType, typename CryptoParamsType>
  inline CtAccurate EvalMultConstWithLoadAndRescale(const CtAccurate& ct, const double op, 
    const SchemeType& scheme, const CryptoParamsType& cryptoParams) const {

    const auto res = EvalMultConstWithLoad(ct, op, scheme, cryptoParams);
    return Rescale(res);
  }

  void EvalMultIntegerInPlace(CtAccurate& ct, const uint64_t c) const; 


  template <typename SchemeType, typename CryptoParamsType>
  inline void EvalAddConstInPlaceWithLoad(CtAccurate& ct, const double op, 
    const SchemeType& scheme, const CryptoParamsType& cryptoParams) const {

    const uint32_t ct_num_elems = ct.ax__.size()/degree__;
    const auto op_scaled = scheme->GetElementForEvalAddOrSub(cryptoParams, ct.level, ct_num_elems, ct.noiseScaleDeg, op);
    const auto op_scaled_gpu = LoadIntegerVector(op_scaled);

    AddScalarInPlace(ct, op_scaled_gpu.data());
  }

  template <typename SchemeType, typename CryptoParamsType>
  inline CtAccurate EvalAddConstWithLoad(const CtAccurate& ct, const double op, 
    const SchemeType& scheme, const CryptoParamsType& cryptoParams) const {
    CtAccurate res(ct);
    EvalAddConstInPlaceWithLoad(res, op, scheme, cryptoParams);
    return res;
  }

  template <typename SchemeType, typename CryptoParamsType>
  inline void EvalSubConstInPlaceWithLoad(CtAccurate& ct, const double op, 
    const SchemeType& scheme, const CryptoParamsType& cryptoParams) const {

    const uint32_t ct_num_elems = ct.ax__.size()/degree__;
    const auto op_scaled = scheme->GetElementForEvalAddOrSub(cryptoParams, ct.level, ct_num_elems, ct.noiseScaleDeg, op);
    const auto op_scaled_gpu = LoadIntegerVector(op_scaled);

    SubScalarInPlace(ct, op_scaled_gpu.data());
  }

  template <typename SchemeType, typename CryptoParamsType>
  inline void EvalSubConstInPlaceWithLoad(const double op, CtAccurate& ct, 
    const SchemeType& scheme, const CryptoParamsType& cryptoParams) const {

    const uint32_t ct_num_elems = ct.ax__.size()/degree__;
    const auto op_scaled = scheme->GetElementForEvalAddOrSub(cryptoParams, ct.level, ct_num_elems, ct.noiseScaleDeg, op);
    const auto op_scaled_gpu = LoadIntegerVector(op_scaled);

    SubScalarInPlace(op_scaled_gpu.data(), ct);
  }

  template <typename SchemeType, typename CryptoParamsType>
  inline CtAccurate EvalSubConstWithLoad(const CtAccurate& ct, const double op, 
    const SchemeType& scheme, const CryptoParamsType& cryptoParams) const {
    CtAccurate res(ct);
    EvalSubConstInPlaceWithLoad(res, op, scheme, cryptoParams);
    return res; 
  }

  template <typename SchemeType, typename CryptoParamsType>
  inline CtAccurate EvalSubConstWithLoad(const double op, const CtAccurate& ct, 
    const SchemeType& scheme, const CryptoParamsType& cryptoParams) const {
    CtAccurate res(ct);
    EvalSubConstInPlaceWithLoad(op, res, scheme, cryptoParams);
    return res; 
  }

  void KeySwitch(const DeviceVector& modup_out, const EvaluationKey& evk, DeviceVector& sum_ax, DeviceVector& sum_bx) const;

  void PMult(const Ciphertext&, const Plaintext&, Ciphertext&) const;
  
  void Add(const Ciphertext&, const Ciphertext&, Ciphertext&) const;
  void Add(const CtAccurate&, const CtAccurate&, CtAccurate&) const;
  CtAccurate Add(const CtAccurate &ct1, const CtAccurate &ct2) const;
  void EvalAddInPlace(CtAccurate &ct1, const CtAccurate &ct2) const;

  void AddCore(const DeviceVector& op1_ax, const DeviceVector& op1_bx, 
      const DeviceVector& op2_ax, const DeviceVector& op2_bx, DeviceVector& out_ax, DeviceVector& out_bx) const;
  void AddCoreInPlace(DeviceVector& x1, const DeviceVector& x2) const;

  void Sub(const Ciphertext&, const Ciphertext&, Ciphertext&) const;
  CtAccurate Sub(const CtAccurate&, const CtAccurate&) const;
  void SubInPlace(Ciphertext &ct1, const Ciphertext &ct2) const;
  void SubInPlace(CtAccurate &ct1, const CtAccurate &ct2) const;

  void AddScalarInPlace(Ciphertext&, const word64*) const;
  void AddScalarInPlace(CtAccurate &ct, const word64* op) const;

  void SubScalarInPlace(Ciphertext&, const word64*) const;
  void SubScalarInPlace(CtAccurate &ct, const word64* op) const;
  void SubScalarInPlace(const word64* op, CtAccurate &ct) const;
  
  double GetAdjustScalar(const CtAccurate& ct1, const CtAccurate& ct2) const;

  template <typename SchemeType, typename CryptoParamsType>
  void AdjustLevelsAndDepthInPlace(CtAccurate& ct1, CtAccurate& ct2, const SchemeType& scheme, const CryptoParamsType& cryptoParams) const {
    const uint32_t ct1lvl = ct1.level;
    const uint32_t ct2lvl = ct2.level;

    if (ct1.level < ct2.level) {
      // rescale ct1
      if (ct1.noiseScaleDeg == 2) {
          if (ct2.noiseScaleDeg == 2) {
              // std::cout << "normal rescale case\n";
              const double scf1 = ct1.scalingFactor;
              const double scf2 = ct2.scalingFactor;
              const double scf  = param__.m_scalingFactorsReal[ct1.level];
              const uint64_t sizeQl1 = param__.chain_length_ - ct1.level;
              const double q1 = param__.m_dmoduliQ[sizeQl1 - 1];
              const double scale_fac = scf2 / scf1 * q1 / scf;

              ct1 = EvalMultConstWithLoad(ct1, scale_fac, scheme, cryptoParams);
              ct1 = Rescale(ct1);
              if (ct1lvl + 1 < ct2lvl) ct1 = DropLimbs(ct1, ct2lvl - ct1lvl - 1);
              ct1.scalingFactor = ct2.scalingFactor;
          } else {
              const double scf1 = ct1.scalingFactor;
              const double scf2 = param__.m_scalingFactorsRealBig[ct2.level - 1];
              const double scf  = param__.m_scalingFactorsReal[ct1.level];
              const uint64_t sizeQl1 = param__.chain_length_ - ct1.level;
              const double q1   = param__.m_dmoduliQ[sizeQl1 - 1];
              const double scale_fac = scf2 / scf1 * q1 / scf;

              ct1 = EvalMultConstWithLoad(ct1, scale_fac, scheme, cryptoParams);
              if (ct1.noiseScaleDeg > 2) ct1 = Rescale(ct1);
              ct1.scalingFactor = ct2.scalingFactor;
          } 
      } else {
          if (ct2.noiseScaleDeg == 2) {
              // std::cout << "adjusting ct1 with normal ct2\n";
              const double scf1 = ct1.scalingFactor;
              const double scf2 = ct2.scalingFactor;
              const double scf  = param__.m_scalingFactorsReal[ct1.level];
              const double scale_fac =  scf2 / scf1 / scf;

              ct1 = EvalMultConstWithLoad(ct1, scale_fac, scheme, cryptoParams);
              ct1 = DropLimbs(ct1, ct2.level - ct1.level);
              // if (ct1.noiseScaleDeg > 2) ct1 = Rescale(ct1);
              ct1.scalingFactor = ct2.scalingFactor;

          } else {
              const double scf1 = ct1.scalingFactor;
              const double scf2 = param__.m_scalingFactorsRealBig[ct2.level - 1];
              const double scf  = param__.m_scalingFactorsReal[ct1.level];
              const double scale_fac = scf2 / scf1 / scf;

              ct1 = EvalMultConstWithLoad(ct1, scale_fac, scheme, cryptoParams);
              if (ct1.noiseScaleDeg > 2) ct1 = Rescale(ct1);
              ct1.scalingFactor = ct2.scalingFactor;
          }
      } 
    } else if (ct1.level > ct2.level) {
        // rescale ct2
        // throw std::logic_error("swap levels\n");
        AdjustLevelsAndDepthInPlace(ct2, ct1, scheme, cryptoParams);
    } else 
      return;
  }

  DeviceVector ModUp(const DeviceVector& in) const;
  void EnableMemoryPool();
  auto GetDegree() const { return degree__; }

  void Rescale(const DeviceVector &from_v, DeviceVector &to_v) const;
  void Rescale(const Ciphertext &from_ct, Ciphertext &to_ct) const;
  void Rescale(const CtAccurate &from_ct, CtAccurate &to_ct) const;
  Ciphertext Rescale(const Ciphertext& from) const;
  CtAccurate Rescale(const CtAccurate& from) const;

  Ciphertext DropLimbs(const Ciphertext& ct, const uint32_t numDropLimbs = 1) const;
  CtAccurate DropLimbs(const CtAccurate& ct, const uint32_t numDropLimbs = 1) const;

  // void ModDown(DeviceVector& from, DeviceVector& to,
  //              long target_chain_idx) const;
  void ModDown(DeviceVector& from, DeviceVector& to) const;
  bool is_modup_batched = true;
  bool is_moddown_fused = true;
  bool is_rescale_fused = true;
  bool is_keyswitch_fused = true;

//  private:
  DeviceVector FromNTT(const DeviceVector& from) const;
  DeviceVector ToNTT(const DeviceVector& input) const;
  // scales with scaling_constants after iNTT
  DeviceVector FromNTT(const DeviceVector& from,
                       const DeviceVector& scaling_constants,
                       const DeviceVector& scaling_constants_shoup) const;
  void ModUpImpl(const word64* from, word64* to, int idx, const int num_original_input_limbs) const;
  void ModUpBatchImpl(const DeviceVector& from, DeviceVector& to, int beta) const;
  void ModUpLengthIsOne(const word64* ptr_after_intt,
                        const word64* ptr_before_intt, int begin_idx,
                        int end_length, word64* to) const;

  void ToNTTInplaceExceptSomeRange(
    word64* base_addr, int start_prime_idx, int batch, 
    int excluded_range_start, int excluded_range_size, const DeviceVector& prime_inds) const;

  void FromNTTInplace(DeviceVector& op1, int start_prime_idx, int batch) const {
    FromNTTInplace(op1.data(), start_prime_idx, batch);
  }
  void FromNTTInplace(word64* op1, int start_prime_idx, int batch, const bool verbose = false) const;
  void FromNTTInplaceShiftedPointer(DeviceVector& op1, int start_prime_idx, int batch) const {
    FromNTTInplace((op1.data()) - start_prime_idx*degree__, start_prime_idx, batch);
  }
  void ToNTTInplace(word64* op1, int start_prime_idx, int batch) const;
  void ToNTTInplaceFused(DeviceVector& op1, const DeviceVector& op2,
                         const DeviceVector& epilogue, const DeviceVector& epilogue_) const;
  void SubInplace(word64* op1, const word64* op2, const int batch) const;
  void NegateInplace(word64* op1, const int batch) const;
  void ConstMultBatch(const word64* op1, const DeviceVector& op2, const DeviceVector& op2_psinv, 
                      int start_prime_idx, int batch, word64* res) const;
  void ConstMultBatchModDown(const word64 *op1, const int start_limb_idx,
    const DeviceVector &op2, const DeviceVector &op2_psinv, int start_prime_idx, int batch, word64 *res) const;
  void ModUpMatMul(const word64* ptr, int beta_idx, word64* to, const int num_original_input_limbs) const;
  void hadamardMultAndAddBatch(const std::vector<const word64*> ax_addr,
                               const std::vector<const word64*> bx_addr,
                               const std::vector<const word64*> mx_addr,
                               const int num_primes, DeviceVector& out_ax,
                               DeviceVector& out_bx) const;
  auto GetKernelParams() const {
    return KernelParams{degree__, primes__.data(), barret_ratio__.data(),
                        barret_k__.data(), param__.log_degree_};
  }
  void GenModUpParams();
  void GenModDownParams();
  void GenRescaleParams();

  DeviceVector copyLimbData(const DeviceVector& from, const int num_orig_limbs) const;

  std::shared_ptr<MemoryPool> pool__;
  int degree__;
  // int num_moduli_after_modup__;
  int alpha__;
  Parameter param__;
  DeviceVector __align__(8) primes__;
  DeviceVector __align__(8) barret_ratio__;
  DeviceVector __align__(8) barret_k__;
  DeviceVector __align__(8) power_of_roots__;
  DeviceVector __align__(8) power_of_roots_shoup__;
  DeviceVector __align__(8) inverse_power_of_roots_div_two__;
  DeviceVector __align__(8) inverse_scaled_power_of_roots_div_two__;

  // for modup
  // {prod q_i}_{n * alpha <= i < (n+1) * alpha)} mod q_j
  // for j not in [n * alpha, n * alpha + alpha) for n in [0, dnum)
  std::vector<std::vector<DeviceVector>> prod_q_i_mod_q_j__;
  // prod q_i mod q_j for i in [n * alpha, (n+1) * alpha) && i != j
  std::vector<std::vector<DeviceVector>> hat_inverse_vec__;
  std::vector<std::vector<DeviceVector>> hat_inverse_vec_shoup__;
  std::vector<DeviceVector> hat_inverse_vec_batched__;
  std::vector<DeviceVector> hat_inverse_vec_shoup_batched__;

  // for moddown
  std::vector<DeviceVector> hat_inverse_vec_moddown__;
  std::vector<DeviceVector> hat_inverse_vec_shoup_moddown__;
  std::vector<DeviceVector> prod_q_i_mod_q_j_moddown__;
  std::vector<DeviceVector> prod_inv_moddown__;
  std::vector<DeviceVector> prod_inv_shoup_moddown__;

  // for rescale
  std::vector<DeviceVector> prod_inv_rescale__;
  std::vector<DeviceVector> prod_inv_shoup_rescale__;

  // KeySwitching addition
  DeviceVector Pmodq;  // P mod each regular prime
  // DeviceVector Pmodq_shoup;  // P mod each regular prime

  // loaded EvalMultKey
  EvaluationKey * preloaded_evaluation_key;
  std::map<uint32_t, EvaluationKey> * preloaded_rotation_key_map;
};


}  // namespace ckks