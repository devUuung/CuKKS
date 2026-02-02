#pragma once

#include "Ciphertext.h"
#include "Context.h"
#include "Define.h"
#include "EvaluationKey.h"
#include "MultPtxtBatch.h"
#include "Parameter.h"
#include "Test.h"

using namespace lbcrypto;
// using namespace ckks;

// inline ckks::DeviceVector loadIntoDeviceVector(const DCRTPoly& input, const bool verbose = false) {
//     if (verbose) {
//         std::cout << "DCRTPoly to DeviceVector\n";
//     }

//     const auto numLimbs = input.m_vectors.size();
//     const auto phim = input.GetRingDimension();
//     const auto totalSize = numLimbs*phim;

//     if (verbose) {
//         std::cout << "numLimbs: " << numLimbs << std::endl;
//         std::cout << "phim: " << phim << std::endl;
//     }

//     ckks::HostVector to_load(totalSize);
//     for (size_t i = 0; i < numLimbs; i++) {
//         for (size_t j = 0; j < phim; j++) {
//             to_load[i*phim + j] = (uint64_t)(*(input.m_vectors[i].m_values))[j];
//         }
//     }

//     return ckks::DeviceVector(to_load);
// }

inline ckks::DeviceVector loadIntoDeviceVector(const std::vector<DCRTPoly>& input, const bool verbose = false) {
    ckks::DeviceVector result;
    for (const auto& in : input) {
        result.append(loadIntoDeviceVector(in, verbose));
    }
    return result;
}

inline DCRTPoly loadIntoDCRTPoly(
    const ckks::DeviceVector& input, 
    const std::shared_ptr<lbcrypto::M4DCRTParams> elementParams, const Format format = Format::EVALUATION, 
    const bool verbose = false) {

    ckks::HostVector host_vec(input);

    const size_t phim = elementParams->GetRingDimension();
    const size_t numLimbs = host_vec.size()/phim;
    if (numLimbs > elementParams->m_params.size()) {
        std::cout << numLimbs << " " << elementParams->m_params.size() << std::endl;
        throw std::logic_error("loadIntoDCRTPoly: Not enough limbs provided");
    }
    // const size_t numLimbs = elementParams->m_params.size();

    std::vector<std::shared_ptr<ILNativeParams>> new_param_vec(numLimbs);
    // if (verbose) {
    //     for (uint32_t i = 0; i < numLimbs; i++) {
    //         std::cout << "Limb " << i << " modulus " << elementParams->m_params[i]->GetModulus() << std::endl;
    //         new_param_vec[i] = elementParams->m_params[i];
    //     }
    // } else {
        for (uint32_t i = 0; i < numLimbs; i++) 
            new_param_vec[i] = elementParams->m_params[i];
    // }
    const lbcrypto::M4DCRTParams new_params = lbcrypto::M4DCRTParams(elementParams->GetCyclotomicOrder(), new_param_vec);
    const std::shared_ptr<lbcrypto::M4DCRTParams> new_params_ptr = std::make_shared<lbcrypto::M4DCRTParams>(new_params);

    if (verbose) {
        std::cout << numLimbs << " " << phim << std::endl;
    }

    DCRTPoly result(new_params_ptr, format, true);  // initialize element to zero

    // if (host_vec.size() != numLimbs*phim) {
    //     std::cout << host_vec.size() << " " << numLimbs*phim << std::endl;
    //     std::cout << numLimbs << " " << phim << " " << host_vec.size()/phim << std::endl;
    //     throw std::logic_error("length mismatch");
    // }

    for (size_t i = 0; i < numLimbs; i++) {
        for (size_t j = 0; j < phim; j++) {
            (*(result.m_vectors[i].m_values))[j] = host_vec[i*phim + j];
        }
    }

    return result;
}

template <typename CryptoParamsType>
inline ckks::Context GenGPUContext(const std::shared_ptr<CryptoParamsType>& cryptoParams) {
    const auto elementParams = cryptoParams->GetElementParams();
    const auto limb_params = elementParams->GetParams();
    const auto ext_params = cryptoParams->GetParamsP()->GetParams();

    // std::cout << "alpha: " << cryptoParams->GetNumPerPartQ() << std::endl;
    // std::cout << "real dnum: " << cryptoParams->GetNumberOfQPartitions() << std::endl;

    std::cout << "log(n) = " << log2(limb_params[0]->GetRingDimension()) << std::endl;
    const size_t n = limb_params[0]->GetRingDimension();
    const size_t logn = log2(n);
    // const size_t numModuli = limb_params.size();
    // std::cout << "num moduli = " << numModuli << std::endl;        

    std::vector<uint64_t> limb_moduli;
    std::vector<uint64_t> limb_rous;
    for (const auto& limb : limb_params) {
        limb_moduli.push_back((uint64_t)limb->GetModulus());
        limb_rous.push_back((uint64_t)limb->GetRootOfUnity());
        // std::cout << "Standard mod " << limb_moduli.size() << " " << limb->GetModulus() << " " << limb->GetRootOfUnity() << std::endl;
    }

    // const size_t chain_length = limb_moduli.size();
    const int dnum = ceil((float)limb_moduli.size() / (float)ext_params.size()); 

    for (const auto& limb : ext_params) {
        limb_moduli.push_back((uint64_t)limb->GetModulus());
        limb_rous.push_back((uint64_t)limb->GetRootOfUnity());
        // std::cout << "Extension mod " << limb_moduli.size() - limb_params.size() << " " << limb->GetModulus() << " " << limb->GetRootOfUnity() << std::endl;
    }


    ckks::Parameter gpu_params(logn, limb_params.size(), dnum, limb_moduli);

    gpu_params.m_scalingFactorsReal.reserve(cryptoParams->m_scalingFactorsReal.size());
    for (const auto& sF : cryptoParams->m_scalingFactorsReal) gpu_params.m_scalingFactorsReal.push_back(sF);

    gpu_params.m_scalingFactorsRealBig.reserve(cryptoParams->m_scalingFactorsReal.size());
    for (const auto& sF : cryptoParams->m_scalingFactorsRealBig) gpu_params.m_scalingFactorsRealBig.push_back(sF);

    gpu_params.m_dmoduliQ.reserve(cryptoParams->m_dmoduliQ.size());
    for (const auto& sF : cryptoParams->m_dmoduliQ) gpu_params.m_dmoduliQ.push_back(sF);
    
    ckks::Context gpu_context(gpu_params, limb_rous);

    gpu_context.Pmodq = LoadIntegerVector(cryptoParams->GetPModq());

    // assert(gpu_context.Pmodq.size() == size_t(gpu_context.param__.chain_length_));

    // ckks::HostVector Pmodq_shoup_host(gpu_context.Pmodq.size());
    // for (uint32_t i = 0; i < gpu_context.Pmodq.size(); i++) {
    //     // copied from the Shoup function
    //     const ckks::word64 in = cryptoParams->GetPModq()[i]; 
    //     const ckks::word128 temp = static_cast<ckks::word128>(in) << 64;
    //     Pmodq_shoup_host[i] = static_cast<ckks::word64>(temp / gpu_context.param__.primes_[i]);
    // }
    // gpu_context.Pmodq_shoup = ckks::DeviceVector(Pmodq_shoup_host);

    return gpu_context;
}

// ckks::EvaluationKey LoadRelinKey(const EvalKey<DCRTPoly>& evk) {
template <typename KeyType>
inline ckks::EvaluationKey LoadRelinKey(const KeyType& evk) {
    ckks::EvaluationKey gpu_evk;
    const auto evk_a = evk->GetAVector();
    const auto evk_b = evk->GetBVector();
    gpu_evk.ax__ = loadIntoDeviceVector(evk_a);
    gpu_evk.bx__ = loadIntoDeviceVector(evk_b);

    return gpu_evk;
}

// ckks::EvaluationKey LoadEvalMultRelinKey(const Ciphertext<DCRTPoly>& ct) {
// ckks::EvaluationKey LoadEvalMultRelinKey(const CryptoContext<DCRTPoly>& cc);
// ckks::EvaluationKey LoadEvalMultRelinKey(const CryptoContext<DCRTPoly>& cc) {
template <typename CryptoContextType>
inline ckks::EvaluationKey LoadEvalMultRelinKey(const CryptoContextType& cc, const std::string keyTag = "") {
    // const auto cryptoParams = std::dynamic_pointer_cast<CryptoParametersRNS>(cc->GetCryptoParameters());
    auto evks = cc->GetAllEvalMultKeys();
    if (keyTag == "") {
        assert(evks.size() == 1);
        assert(evks.begin()->second.size() == 1);
        // auto evk_tags = evks.
        // EvalKey<DCRTPoly> evk = evks.begin()->second[0];
        const auto evk = evks.begin()->second[0];

        return LoadRelinKey(evk);
    } else {
        // assert(evks.find(keyTag) != evks.end());
        assert(evks.at(keyTag).size() == 1);
        const auto evk = evks.at(keyTag)[0];

        return LoadRelinKey(evk);
    }
}

template <typename CiphertextType>
inline ckks::Ciphertext LoadCiphertext(const CiphertextType& ct, const bool verbose = false) {
    ckks::Ciphertext res;
    assert(ct->GetElements().size() == 2);

    res.bx__ = loadIntoDeviceVector(ct->GetElements()[0], verbose);
    res.ax__ = loadIntoDeviceVector(ct->GetElements()[1], verbose);

    return res;
}

// // duplicate of above
// inline ckks::Ciphertext LoadCiphertext(ConstCiphertext<DCRTPoly>& ct, const bool verbose = false) {
//     ckks::Ciphertext res;
//     assert(ct->GetElements().size() == 2);

//     res.bx__ = loadIntoDeviceVector(ct->GetElements()[0], verbose);
//     res.ax__ = loadIntoDeviceVector(ct->GetElements()[1], verbose);

//     return res;
// }

template <typename CiphertextType>
inline ckks::CtAccurate LoadAccurateCiphertext(const CiphertextType& ct, const bool verbose = false) {
    ckks::CtAccurate res;
    assert(ct->GetElements().size() == 2);

    res.bx__ = loadIntoDeviceVector(ct->GetElements()[0], verbose);
    res.ax__ = loadIntoDeviceVector(ct->GetElements()[1], verbose);

    res.level = ct->GetLevel();
    res.noiseScaleDeg = ct->GetNoiseScaleDeg();
    res.scalingFactor = ct->GetScalingFactor();

    return res;
}

template <typename PlaintextType, typename DCRTPolyType>
inline ckks::PtAccurate LoadAccuratePlaintext(const PlaintextType& pt, const DCRTPolyType& data, const bool verbose = false) {
    ckks::PtAccurate res;

    res.mx__ = loadIntoDeviceVector(data, verbose);

    res.level = pt->GetLevel();
    res.noiseScaleDeg = pt->GetNoiseScaleDeg();
    res.scalingFactor = pt->GetScalingFactor();

    return res;
}

// duplicate of above
// inline ckks::CtAccurate LoadAccurateCiphertext(ConstCiphertext<DCRTPoly>& ct, const bool verbose = false) {
//     ckks::CtAccurate res;
//     assert(ct->GetElements().size() == 2);

//     res.bx__ = loadIntoDeviceVector(ct->GetElements()[0], verbose);
//     res.ax__ = loadIntoDeviceVector(ct->GetElements()[1], verbose);

//     res.level = ct->GetLevel();
//     res.noiseScaleDeg = ct->GetNoiseScaleDeg();
//     res.scalingFactor = ct->GetScalingFactor();

//     return res;
// }

template <typename CiphertextType>
inline void LoadCtAccurateFromGPU(CiphertextType& to, const ckks::CtAccurate& from, const std::shared_ptr<lbcrypto::M4DCRTParams> elementParams) {
    std::vector<DCRTPoly> gpu_res_elems(2);
    gpu_res_elems[0] = loadIntoDCRTPoly(from.bx__, elementParams);
    gpu_res_elems[1] = loadIntoDCRTPoly(from.ax__, elementParams);

    to->SetElements(gpu_res_elems);
    to->SetLevel(from.level);
    to->SetScalingFactor(from.scalingFactor);
    to->SetNoiseScaleDeg(from.noiseScaleDeg);
}

// template <typename PlaintextType>
// inline void LoadPtAccurateFromGPU(PlaintextType& to, const ckks::PtAccurate& from, const std::shared_ptr<lbcrypto::M4DCRTParams> elementParams) {
//     to->encodedVectorDCRT = loadIntoDCRTPoly(from.mx__, elementParams);
//     to->SetLevel(from.level);
//     to->SetScalingFactor(from.scalingFactor);
//     to->SetNoiseScaleDeg(from.noiseScaleDeg);
//     to->SetSlots(elementParams->GetRingDimension()/2);
//     // double scalingFactor           = 1;
//     // NativeInteger scalingFactorInt = 1;
//     // size_t level                   = 0;
//     // size_t noiseScaleDeg           = 1;
//     // usint slots                    = 0;
//     // SCHEME schemeID;
// }
