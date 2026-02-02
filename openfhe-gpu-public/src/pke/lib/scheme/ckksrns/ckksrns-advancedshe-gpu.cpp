#define PROFILE

#include "cryptocontext.h"
#include "scheme/ckksrns/ckksrns-cryptoparameters.h"
#include "scheme/ckksrns/ckksrns-leveledshe.h"
#include "scheme/ckksrns/ckksrns-advancedshe.h"
#include "scheme/ckksrns/ckksrns-utils.h"

#include "schemebase/base-scheme.h"


namespace lbcrypto {


// Ciphertext<DCRTPoly> 
ckks::CtAccurate
AdvancedSHECKKSRNS::EvalLinearWSumMutableGPU(
    // std::vector<Ciphertext<DCRTPoly>>& ciphertexts, 
    std::vector<ckks::CtAccurate>& gpu_ciphertexts, 
    const CryptoContext<DCRTPoly>& cc,
    const std::vector<double>& constants, const ckks::Context& gpu_context) const {

    // return EvalLinearWSumMutable(ciphertexts, constants);

    // std::cout << "Running EvalLinearWSumMutableGPU\n";

    // const auto cc   = ciphertexts[0]->GetCryptoContext();
    const auto cryptoParams = std::dynamic_pointer_cast<CryptoParametersCKKSRNS>(cc->GetCryptoParameters());
    const auto elementParams = cc->GetElementParams();
    const auto algo = cc->GetScheme();
    const auto scheme = std::dynamic_pointer_cast<LeveledSHECKKSRNS>(cc->GetScheme());

    // Load Ciphertexts into GPU
    // std::vector<ckks::CtAccurate> gpu_ciphertexts(input_gpu_ciphertexts.size());
    // for (uint32_t i = 0; i < ciphertexts.size(); i++) {
    //     gpu_ciphertexts[i] = LoadAccurateCiphertext(ciphertexts[i]);
    //     // std::cout << gpu_ciphertexts[i].level << " " << input_gpu_ciphertexts[i].level << std::endl;
    //     // std::cout << gpu_ciphertexts[i].noiseScaleDeg << " " << input_gpu_ciphertexts[i].noiseScaleDeg << std::endl;
    //     // std::cout << gpu_ciphertexts[i].scalingFactor << " " << input_gpu_ciphertexts[i].scalingFactor << std::endl;
    //     assert(gpu_ciphertexts[i] == input_gpu_ciphertexts[i]);
    // }

    if (cryptoParams->GetScalingTechnique() != FIXEDMANUAL) {
        // std::cout << "running EvalLinearWSumMutableGPU inner rescaling\n";
        // Check to see if input ciphertexts are of same level
        // and adjust if needed to the max level among them
        uint32_t maxLevel = gpu_ciphertexts[0].level;
        uint32_t maxIdx   = 0;
        for (uint32_t i = 1; i < gpu_ciphertexts.size(); i++) {
            if ((gpu_ciphertexts[i].level > maxLevel) ||
                ((gpu_ciphertexts[i].level == maxLevel) && (gpu_ciphertexts[i].noiseScaleDeg == 2))) {
                maxLevel = gpu_ciphertexts[i].level;
                maxIdx   = i;
            }
        }

        for (uint32_t i = 0; i < maxIdx; i++) {
            // algo->AdjustLevelsAndDepthInPlace(ciphertexts[i], ciphertexts[maxIdx]);

            if (gpu_ciphertexts[i].level < gpu_ciphertexts[maxIdx].level) {
                const double adjustScalar = gpu_context.GetAdjustScalar(gpu_ciphertexts[i], gpu_ciphertexts[maxIdx]);
                gpu_ciphertexts[i] = gpu_context.EvalMultConstWithLoad(gpu_ciphertexts[i], adjustScalar, scheme, cryptoParams);
                gpu_ciphertexts[i] = gpu_context.Rescale(gpu_ciphertexts[i]);
            }

            // {
            //     // load correct ciphertext
            //     const auto correct_ciphertext = LoadAccurateCiphertext(ciphertexts[i]);
            //     // std::cout << correct_ciphertext.level << " " << gpu_ciphertexts[i].level << std::endl;
            //     // std::cout << correct_ciphertext.scalingFactor << " " << gpu_ciphertexts[i].scalingFactor << std::endl;
            //     // std::cout << correct_ciphertext.noiseScaleDeg << " " << gpu_ciphertexts[i].noiseScaleDeg << std::endl;
            //     assert(correct_ciphertext == gpu_ciphertexts[i]);
            // }
        }

        for (uint32_t i = maxIdx + 1; i < gpu_ciphertexts.size(); i++) {
            // algo->AdjustLevelsAndDepthInPlace(ciphertexts[i], ciphertexts[maxIdx]);

            if (gpu_ciphertexts[i].level < gpu_ciphertexts[maxIdx].level) {
                const double adjustScalar = gpu_context.GetAdjustScalar(gpu_ciphertexts[i], gpu_ciphertexts[maxIdx]);
                gpu_ciphertexts[i] = gpu_context.EvalMultConstWithLoad(gpu_ciphertexts[i], adjustScalar, scheme, cryptoParams);
                gpu_ciphertexts[i] = gpu_context.Rescale(gpu_ciphertexts[i]);
            }

            // {
            //     // load correct ciphertext
            //     const auto correct_ciphertext = LoadAccurateCiphertext(ciphertexts[i]);
            //     // std::cout << correct_ciphertext.level << " " << gpu_ciphertexts[i].level << std::endl;
            //     // std::cout << correct_ciphertext.scalingFactor << " " << gpu_ciphertexts[i].scalingFactor << std::endl;
            //     // std::cout << correct_ciphertext.noiseScaleDeg << " " << gpu_ciphertexts[i].noiseScaleDeg << std::endl;
            //     assert(correct_ciphertext == gpu_ciphertexts[i]);
            // }
        }

        if (gpu_ciphertexts[maxIdx].noiseScaleDeg == 2) {
            for (uint32_t i = 0; i < gpu_ciphertexts.size(); i++) {
                // algo->ModReduceInternalInPlace(ciphertexts[i], BASE_NUM_LEVELS_TO_DROP);
                gpu_ciphertexts[i] = gpu_context.Rescale(gpu_ciphertexts[i]);
                // {
                //     // load correct ciphertext
                //     const auto correct_ciphertext = LoadAccurateCiphertext(ciphertexts[i]);
                //     // std::cout << "checking ciphertext " << i << std::endl;
                //     // std::cout << correct_ciphertext.level << " " << gpu_ciphertexts[i].level << std::endl;
                //     // std::cout << correct_ciphertext.scalingFactor << " " << gpu_ciphertexts[i].scalingFactor << std::endl;
                //     // std::cout << correct_ciphertext.noiseScaleDeg << " " << gpu_ciphertexts[i].noiseScaleDeg << std::endl;
                //     assert(correct_ciphertext == gpu_ciphertexts[i]);
                // }
            }
        }
    }

    // std::cout << "Beginning GPU section\n";

    // Load Ciphertexts into GPU
    // std::vector<ckks::CtAccurate> gpu_ciphertexts(ciphertexts.size());
    // for (uint32_t i = 0; i < ciphertexts.size(); i++) {
    //     gpu_ciphertexts[i] = LoadAccurateCiphertext(ciphertexts[i]);
    // }

    // std::cout << "loaded ciphertexts\n";

    auto weightedSum_gpu = gpu_context.EvalMultConstWithLoad(gpu_ciphertexts[0], constants[0], scheme, cryptoParams);

    // Ciphertext<DCRTPoly> weightedSum = cc->EvalMult(ciphertexts[0], constants[0]);
    // {
    //     // load correct ciphertext
    //     const auto correct_weightedsum = LoadAccurateCiphertext(weightedSum);
    //     assert(correct_weightedsum == weightedSum_gpu);
    // }

    // Ciphertext<DCRTPoly> tmp;
    ckks::CtAccurate tmp_gpu;
    for (uint32_t i = 1; i < gpu_ciphertexts.size(); i++) {
        // tmp = cc->EvalMult(ciphertexts[i], constants[i]);
        tmp_gpu = gpu_context.EvalMultConstWithLoad(gpu_ciphertexts[i], constants[i], scheme, cryptoParams);
        // {
        //     // load correct ciphertext
        //     const auto correct_tmp = LoadAccurateCiphertext(tmp);
        //     assert(correct_tmp == tmp_gpu);
        // }
        // cc->EvalAddInPlace(weightedSum, tmp);
        gpu_context.EvalAddInPlace(weightedSum_gpu, tmp_gpu);
        // {
        //     // load correct ciphertext
        //     const auto correct_weightedsum = LoadAccurateCiphertext(weightedSum);
        //     assert(correct_weightedsum == weightedSum_gpu);
        // }
    }

    // cc->ModReduceInPlace(weightedSum);

    // {
    //     // load correct ciphertext
    //     const auto correct_weightedsum = LoadAccurateCiphertext(weightedSum);
    //     assert(correct_weightedsum == weightedSum_gpu);
    // }

    // const auto resParams = ciphertexts[0]->GetElements()[0].GetParams();

    return weightedSum_gpu;

    // Ciphertext<DCRTPoly> result = ciphertexts[0]->CloneZero();
    // LoadCtAccurateFromGPU(result, weightedSum_gpu, elementParams);
    // return result;

    // assert(*result == *weightedSum);

    // return weightedSum;
}


ckks::CtAccurate AdvancedSHECKKSRNS::EvalChebyshevSeriesGPU(const ckks::CtAccurate& x, 
    const std::vector<double>& coefficients, double a, double b, const ckks::Context& gpu_context, const CryptoContext<DCRTPoly>& cc) const {
    uint32_t n = Degree(coefficients);

    if (n < 5) {
        throw std::logic_error("EvalChebyshevSeriesLinearGPU is not implemented\n");
        // return EvalChebyshevSeriesLinear(x, coefficients, a, b);
    }

    return EvalChebyshevSeriesPSGPU(x, coefficients, a, b, gpu_context, cc);
}


ckks::CtAccurate 
AdvancedSHECKKSRNS::InnerEvalChebyshevPSGPU(const ckks::CtAccurate& x, // ConstCiphertext<DCRTPoly> x_cpu,
        const std::vector<double>& coefficients, uint32_t k, uint32_t m, 
        // std::vector<Ciphertext<DCRTPoly>>& T, std::vector<Ciphertext<DCRTPoly>>& T2, 
        std::vector<ckks::CtAccurate>& T_gpu, std::vector<ckks::CtAccurate>& T2_gpu, 
        const ckks::Context& gpu_context, const CryptoContext<DCRTPoly>& cc) const {

    // std::cout << "Running InnerEvalChebyshevPSGPU (todo: return GPU ciphertext)\n";                                                            
    
    // auto cc = x->GetCryptoContext();

    // Compute k*2^{m-1}-k because we use it a lot
    uint32_t k2m2k = k * (1 << (m - 1)) - k;

    // Divide coefficients by T^{k*2^{m-1}}
    std::vector<double> Tkm(int32_t(k2m2k + k) + 1, 0.0);
    Tkm.back() = 1;
    auto divqr = LongDivisionChebyshev(coefficients, Tkm);

    // Subtract x^{k(2^{m-1} - 1)} from r
    std::vector<double> r2 = divqr->r;
    if (int32_t(k2m2k - Degree(divqr->r)) <= 0) {
        r2[int32_t(k2m2k)] -= 1;
        r2.resize(Degree(r2) + 1);
    }
    else {
        r2.resize(int32_t(k2m2k + 1), 0.0);
        r2.back() = -1;
    }

    // Divide r2 by q
    auto divcs = LongDivisionChebyshev(r2, divqr->q);

    // Add x^{k(2^{m-1} - 1)} to s
    std::vector<double> s2 = divcs->r;
    s2.resize(int32_t(k2m2k + 1), 0.0);
    s2.back() = 1;

    // Evaluate c at u
    // Ciphertext<DCRTPoly> cu;
    ckks::CtAccurate cu_gpu;
    const auto scheme = std::dynamic_pointer_cast<LeveledSHECKKSRNS>(cc->GetScheme());
    const auto cryptoParams = std::dynamic_pointer_cast<CryptoParametersCKKSRNS>(cc->GetCryptoParameters());
    const auto elementParams = cc->GetElementParams();
    uint32_t dc = Degree(divcs->q);
    bool flag_c = false;
    if (dc >= 1) {
        if (dc == 1) {
            if (divcs->q[1] != 1) {
                // cu = cc->EvalMult(T.front(), divcs->q[1]);
                // cc->ModReduceInPlace(cu);
                cu_gpu = gpu_context.EvalMultConstWithLoad(T_gpu.front(), divcs->q[1], scheme, cryptoParams);
            }
            else {
                // cu = T.front();
                cu_gpu = ckks::CtAccurate(T_gpu.front());
            }
        }
        else {
            // std::vector<Ciphertext<DCRTPoly>> ctxs(dc);
            std::vector<ckks::CtAccurate> ctxs_gpu(dc);
            std::vector<double> weights(dc);

            for (uint32_t i = 0; i < dc; i++) {
                // ctxs[i]    = T[i];
                ctxs_gpu[i]    = ckks::CtAccurate(T_gpu[i]);
                weights[i] = divcs->q[i + 1];
            }

            // cu = cc->EvalLinearWSumMutable(ctxs, weights);
            cu_gpu = cc->EvalLinearWSumMutableGPU(ctxs_gpu, cc, weights, gpu_context);
        }

        // LoadCtAccurateFromGPU(cu, cu_gpu, elementParams);

        // adds the free term (at x^0)
        // cc->EvalAddInPlace(cu, divcs->q.front() / 2);
        if (divcs->q.front() < 0)
            gpu_context.EvalSubConstInPlaceWithLoad(cu_gpu, std::fabs(divcs->q.front() / 2), scheme, cryptoParams);
        else 
            gpu_context.EvalAddConstInPlaceWithLoad(cu_gpu, divcs->q.front() / 2, scheme, cryptoParams);

        // Need to reduce levels up to the level of T2[m-1].
        // usint levelDiff = T2[m - 1]->GetLevel() - cu->GetLevel();
        // cc->LevelReduceInPlace(cu, nullptr, levelDiff);
        flag_c = true;
    }

    // Ciphertext<DCRTPoly> cu = T.front()->CloneZero();
    // LoadCtAccurateFromGPU(cu, cu_gpu, elementParams);
    // {
    //     Ciphertext<DCRTPoly> should_be_cu = cu->CloneZero();
    //     LoadCtAccurateFromGPU(should_be_cu, cu_gpu, elementParams);
    //     if (*should_be_cu != *cu) {
    //         throw std::logic_error("InnerCheby cu mismatch");
    //     }
    // }


    // Evaluate q and s2 at u. If their degrees are larger than k, then recursively apply the Paterson-Stockmeyer algorithm.
    // Ciphertext<DCRTPoly> qu;
    ckks::CtAccurate qu_gpu;

    if (Degree(divqr->q) > k) {
        // qu = InnerEvalChebyshevPS(x_cpu, divqr->q, k, m - 1, T, T2);
        // const auto temp = InnerEvalChebyshevPSGPU(x, divqr->q, k, m - 1, T, T2, T_gpu, T2_gpu, gpu_context);
        qu_gpu = InnerEvalChebyshevPSGPU(x, divqr->q, k, m - 1, T_gpu, T2_gpu, gpu_context, cc);
        // qu_gpu = LoadAccurateCiphertext(temp);
    }
    else {
        // dq = k from construction
        // perform scalar multiplication for all other terms and sum them up if there are non-zero coefficients
        auto qcopy = divqr->q;
        qcopy.resize(k);
        if (Degree(qcopy) > 0) {
            // std::vector<Ciphertext<DCRTPoly>> ctxs(Degree(qcopy));
            std::vector<ckks::CtAccurate> ctxs_gpu(Degree(qcopy));
            std::vector<double> weights(Degree(qcopy));

            for (uint32_t i = 0; i < Degree(qcopy); i++) {
                // ctxs[i]    = T[i];
                ctxs_gpu[i]    = ckks::CtAccurate(T_gpu[i]);
                weights[i] = divqr->q[i + 1];
            }

            // qu = cc->EvalLinearWSumMutable(ctxs, weights);
            qu_gpu = cc->EvalLinearWSumMutableGPU(ctxs_gpu, cc, weights, gpu_context);

            // the highest order coefficient will always be a power of two up to 2^{m-1} because q is "monic" but the Chebyshev rule adds a factor of 2
            // we don't need to increase the depth by multiplying the highest order coefficient, but instead checking and summing, since we work with m <= 4.
            // Ciphertext<DCRTPoly> sum = T[k - 1];
            ckks::CtAccurate sum_gpu = T_gpu[k - 1];
            for (uint32_t i = 0; i < log2(divqr->q.back()); i++) {
                // sum = cc->EvalAdd(sum, sum);
                gpu_context.EvalAddInPlace(sum_gpu, sum_gpu);
            }
            // cc->EvalAddInPlace(qu, sum);
            const double adjustScalar = gpu_context.GetAdjustScalar(qu_gpu, sum_gpu);
            auto toAdd = gpu_context.EvalMultConstWithLoad(sum_gpu, adjustScalar, scheme, cryptoParams);
            toAdd = gpu_context.Rescale(toAdd);
            gpu_context.EvalAddInPlace(qu_gpu, toAdd);
        }
        else {
            // Ciphertext<DCRTPoly> sum = T[k - 1];
            ckks::CtAccurate sum_gpu = T_gpu[k - 1];
            for (uint32_t i = 0; i < log2(divqr->q.back()); i++) {
                // sum = cc->EvalAdd(sum, sum);
                gpu_context.EvalAddInPlace(sum_gpu, sum_gpu);
            }
            // qu = sum;
            qu_gpu = ckks::CtAccurate(sum_gpu);
        }


        // adds the free term (at x^0)
        // cc->EvalAddInPlace(qu, divqr->q.front() / 2);
        if (divqr->q.front() < 0) 
            gpu_context.EvalSubConstInPlaceWithLoad(qu_gpu, std::fabs(divqr->q.front() / 2), scheme, cryptoParams);
        else 
            gpu_context.EvalAddConstInPlaceWithLoad(qu_gpu, divqr->q.front() / 2, scheme, cryptoParams);
        // The number of levels of qu is the same as the number of levels of T[k-1] or T[k-1] + 1.
        // No need to reduce it to T2[m-1] because it only reaches here when m = 2.

        // {
        //     Ciphertext<DCRTPoly> should_be_qu = qu->CloneZero();
        //     LoadCtAccurateFromGPU(should_be_qu, qu_gpu, elementParams);
        //     if (*should_be_qu != *qu) {
        //         throw std::logic_error("InnerCheby qu branch mismatch");
        //     }
        // }
    }

    // Ciphertext<DCRTPoly> qu = T.front()->CloneZero();
    // LoadCtAccurateFromGPU(qu, qu_gpu, elementParams);


    // Ciphertext<DCRTPoly> su; 
    ckks::CtAccurate su_gpu;

    if (Degree(s2) > k) {
        // su = InnerEvalChebyshevPS(x_cpu, s2, k, m - 1, T, T2);
        // const auto temp = InnerEvalChebyshevPSGPU(x, s2, k, m - 1, T, T2, T_gpu, T2_gpu, gpu_context);
        // su_gpu = LoadAccurateCiphertext(temp);
        su_gpu = InnerEvalChebyshevPSGPU(x, s2, k, m - 1, T_gpu, T2_gpu, gpu_context, cc);
    }
    else {
        // ds = k from construction
        // perform scalar multiplication for all other terms and sum them up if there are non-zero coefficients
        auto scopy = s2;
        scopy.resize(k);
        if (Degree(scopy) > 0) {
            // std::vector<Ciphertext<DCRTPoly>> ctxs(Degree(scopy));
            std::vector<ckks::CtAccurate> ctxs_gpu(Degree(scopy));
            std::vector<double> weights(Degree(scopy));

            for (uint32_t i = 0; i < Degree(scopy); i++) {
                // ctxs[i]    = T[i];
                ctxs_gpu[i]    = ckks::CtAccurate(T_gpu[i]);
                weights[i] = s2[i + 1];
            }

            // su = cc->EvalLinearWSumMutable(ctxs, weights);
            su_gpu = cc->EvalLinearWSumMutableGPU(ctxs_gpu, cc, weights, gpu_context);
            // {
            //     Ciphertext<DCRTPoly> should_be_su = su->CloneZero();
            //     LoadCtAccurateFromGPU(should_be_su, su_gpu, elementParams);
            //     if (*should_be_su != *su) {
            //         throw std::logic_error("InnerCheby su branch EvalLinearWSumMutable output mismatch");
            //     }
            // }
            // the highest order coefficient will always be 1 because s2 is monic.
            // std::cout << "beginning add to replicate\n";
            // cc->EvalAddInPlace(su, T[k - 1]);

            const double adjustScalar = gpu_context.GetAdjustScalar(su_gpu, T_gpu[k-1]);
            auto toAdd = gpu_context.EvalMultConstWithLoad(T_gpu[k-1], adjustScalar, scheme, cryptoParams);
            toAdd = gpu_context.Rescale(toAdd);
            toAdd.scalingFactor = su_gpu.scalingFactor;
            gpu_context.EvalAddInPlace(su_gpu, toAdd);
            // {
            //     Ciphertext<DCRTPoly> should_be_su = su->CloneZero();
            //     LoadCtAccurateFromGPU(should_be_su, su_gpu, elementParams);
            //     if (*should_be_su != *su) {
            //         throw std::logic_error("InnerCheby su branch sum and add output mismatch");
            //     }
            // }
        }
        else {
            // su = T[k - 1];
            su_gpu = ckks::CtAccurate(T_gpu[k - 1]);
        }

        // adds the free term (at x^0)
        // cc->EvalAddInPlace(su, s2.front() / 2);
        // The number of levels of su is the same as the number of levels of T[k-1] or T[k-1] + 1. Need to reduce it to T2[m-1] + 1.
        // su = cc->LevelReduce(su, nullptr, su->GetElements()[0].GetNumOfElements() - Lm + 1) ;
        // cc->LevelReduceInPlace(su, nullptr);

        if (s2.front() < 0)
            gpu_context.EvalSubConstInPlaceWithLoad(su_gpu, std::fabs(s2.front() / 2), scheme, cryptoParams);
        else 
            gpu_context.EvalAddConstInPlaceWithLoad(su_gpu, s2.front() / 2, scheme, cryptoParams);

        // {
        //     Ciphertext<DCRTPoly> should_be_su = su->CloneZero();
        //     LoadCtAccurateFromGPU(should_be_su, su_gpu, elementParams);
        //     if (*should_be_su != *su) {
        //         throw std::logic_error("InnerCheby su branch mismatch");
        //     }
        // }
    }

    // Ciphertext<DCRTPoly> su = T.front()->CloneZero();
    // LoadCtAccurateFromGPU(su, su_gpu, elementParams);

    // Ciphertext<DCRTPoly> result;
    ckks::CtAccurate result_gpu;

    if (flag_c) {
        // result = cc->EvalAdd(T2[m - 1], cu);

        if (T2_gpu[m-1].level != cu_gpu.level) {
            const double adjust_scale = gpu_context.GetAdjustScalar(T2_gpu[m-1], cu_gpu);
            auto toAdd = gpu_context.EvalMultConstWithLoad(cu_gpu, adjust_scale, scheme, cryptoParams);
            toAdd = gpu_context.Rescale(toAdd);
            toAdd.scalingFactor = T2_gpu[m-1].scalingFactor;
            result_gpu = gpu_context.Add(T2_gpu[m-1], toAdd);
        } else {
            result_gpu = gpu_context.Add(T2_gpu[m-1], cu_gpu);
        }
    }
    else {
        // result = cc->EvalAdd(T2[m - 1], divcs->q.front() / 2);
        if (divcs->q.front() < 0) 
            result_gpu = gpu_context.EvalSubConstWithLoad(T2_gpu[m-1], std::fabs(divcs->q.front() / 2), scheme, cryptoParams);
        else 
            result_gpu = gpu_context.EvalAddConstWithLoad(T2_gpu[m-1], divcs->q.front() / 2, scheme, cryptoParams);
    }

    // {
    //     Ciphertext<DCRTPoly> should_be_result = result->CloneZero();
    //     LoadCtAccurateFromGPU(should_be_result, result_gpu, elementParams);
    //     if (*should_be_result != *result) {
    //         throw std::logic_error("result branch output mismatch");
    //     }
    // }

    // result = cc->EvalMult(result, qu);
    result_gpu = gpu_context.EvalMultAndRelin(result_gpu, qu_gpu, *gpu_context.preloaded_evaluation_key);

    // cc->ModReduceInPlace(result);

    // {
    //     Ciphertext<DCRTPoly> should_be_result = result->CloneZero();
    //     LoadCtAccurateFromGPU(should_be_result, result_gpu, elementParams);
    //     if (*should_be_result != *result) {
    //         throw std::logic_error("result pre add mismatch");
    //     }
    // }

    // std::cout << "add to replicate\n";
    // cc->EvalAddInPlace(result, su);

    const double adjust_scale = gpu_context.GetAdjustScalar(result_gpu, su_gpu);
    auto toAdd = gpu_context.EvalMultConstWithLoad(su_gpu, adjust_scale, scheme, cryptoParams);
    toAdd = gpu_context.Rescale(toAdd);
    toAdd.scalingFactor = result_gpu.scalingFactor;
    gpu_context.EvalAddInPlace(result_gpu, toAdd);

    // {
    //     Ciphertext<DCRTPoly> should_be_result = result->CloneZero();
    //     LoadCtAccurateFromGPU(should_be_result, result_gpu, elementParams);
    //     if (*should_be_result != *result) {
    //         throw std::logic_error("result mismatch");
    //     }
    // }

    return result_gpu;

    // LoadCtAccurateFromGPU(result, result_gpu, elementParams);

    // return result;
}


// Ciphertext<DCRTPoly> 
ckks::CtAccurate
AdvancedSHECKKSRNS::EvalChebyshevSeriesPSGPU(const ckks::CtAccurate& x, 
    // ConstCiphertext<DCRTPoly> x_cpu,
    const std::vector<double>& coefficients, double a, double b, const ckks::Context& gpu_context, const CryptoContext<DCRTPoly>& cc) const {

    // std::cout << "Running EvalChebyshev PS GPU\n";

    uint32_t n = Degree(coefficients);

    std::vector<double> f2 = coefficients;

    // Make sure the coefficients do not have the zero dominant terms
    if (coefficients[coefficients.size() - 1] == 0)
        f2.resize(n + 1);

    std::vector<uint32_t> degs = ComputeDegreesPS(n);
    uint32_t k                 = degs[0];
    uint32_t m                 = degs[1];

    //  std::cerr << "\n Degree: n = " << n << ", k = " << k << ", m = " << m << endl;


    // computes linear transformation y = -1 + 2 (x-a)/(b-a)
    // consumes one level when a <> -1 && b <> 1
    // const auto cc = x->GetCryptoContext();
    const auto cryptoParams = std::dynamic_pointer_cast<CryptoParametersCKKSRNS>(cc->GetCryptoParameters());
    const auto scheme = std::dynamic_pointer_cast<LeveledSHECKKSRNS>(cc->GetScheme());
    const auto elementParams = cc->GetElementParams();

    // std::vector<Ciphertext<DCRTPoly>> T(k);
    std::vector<ckks::CtAccurate> T_gpu(k);
    if ((a - std::round(a) < 1e-10) && (b - std::round(b) < 1e-10) && (std::round(a) == -1) && (std::round(b) == 1)) {
        // no linear transformation is needed if a = -1, b = 1
        // T_1(y) = y
        // T[0] = x_cpu->Clone();
        // T_gpu[0] = LoadAccurateCiphertext(x);
        T_gpu[0] = ckks::CtAccurate(x);
    }
    else {
        std::cout << "Linear Transformation is needed\n";
        throw std::logic_error("linear transform not implemented for GPU");
        // original code
        // // linear transformation is needed
        // double alpha = 2 / (b - a);
        // double beta  = 2 * a / (b - a);

        // T[0] = cc->EvalMult(x, alpha);
        // const auto toMult = scheme->GetElementForEvalMult(x, alpha);
        // T_gpu[0] = LoadAccurateCiphertext(x);
        // const auto toMult_gpu = LoadIntegerVector(toMult);
        // gpu_context.EvalMultConstInPlace(T_gpu[0], toMult_gpu);
        // // {
        // //     // load correct ciphertext
        // //     const auto should_be_t_gpu = LoadCiphertext(T[0]);
        // //     assert(should_be_t_gpu.ax__ == T_gpu[0].ax__);
        // //     assert(should_be_t_gpu.bx__ == T_gpu[0].bx__);
        // // }
        // cc->ModReduceInPlace(T[0]);
        // cc->EvalAddInPlace(T[0], -1.0 - beta);
    }

    // const Ciphertext<DCRTPoly> y = T[0]->Clone();
    // const ckks::CtAccurate y_gpu_const = LoadAccurateCiphertext(y);
    const ckks::CtAccurate y_gpu_const(T_gpu[0]);
    // const auto y_gpu = LoadCiphertext(y);

    // Original loop
    // Computes Chebyshev polynomials up to degree k
    // for y: T_1(y) = y, T_2(y), ... , T_k(y)
    // uses binary tree multiplication
    // original loop
    // for (uint32_t i = 2; i <= k; i++) {
    //     // if i is a power of two
    //     if (!(i & (i - 1))) {
    //         // compute T_{2i}(y) = 2*T_i(y)^2 - 1
    //         auto square = cc->EvalSquare(T[i / 2 - 1]);
    //         T[i - 1]    = cc->EvalAdd(square, square);
    //         cc->ModReduceInPlace(T[i - 1]);
    //         cc->EvalAddInPlace(T[i - 1], -1.0);
    //     }
    //     else {
    //         // non-power of 2
    //         if (i % 2 == 1) {
    //             // if i is odd
    //             // compute T_{2i+1}(y) = 2*T_i(y)*T_{i+1}(y) - y
    //             auto prod = cc->EvalMult(T[i / 2 - 1], T[i / 2]);
    //             T[i - 1]  = cc->EvalAdd(prod, prod);

    //             cc->ModReduceInPlace(T[i - 1]);
    //             cc->EvalSubInPlace(T[i - 1], y);
    //         }
    //         else {
    //             // i is even but not power of 2
    //             // compute T_{2i}(y) = 2*T_i(y)^2 - 1
    //             auto square = cc->EvalSquare(T[i / 2 - 1]);
    //             T[i - 1]    = cc->EvalAdd(square, square);
    //             cc->ModReduceInPlace(T[i - 1]);
    //             cc->EvalAddInPlace(T[i - 1], -1.0);
    //         }
    //     }
    // }

    ckks::EvaluationKey& evk = *(gpu_context.preloaded_evaluation_key);

    // NOTE: Only tested for k = 6

    for (uint32_t i = 2; i <= k; i++) {
        // if i is a power of two
        // std::cout << "Running iteration index " << i << std::endl;
        if (!(i & (i - 1))) {
            // compute T_{2i}(y) = 2*T_i(y)^2 - 1
            // auto square = cc->EvalSquare(T[i / 2 - 1]);
            const auto square_gpu = (i == 2) ? gpu_context.EvalSquareAndRelinNoRescale(T_gpu[i/2 - 1], evk) : 
               gpu_context.EvalSquareAndRelin(T_gpu[i/2 - 1], evk);

            // T[i - 1]    = cc->EvalAdd(square, square);
            gpu_context.Add(square_gpu, square_gpu, T_gpu[i-1]);

            // cc->ModReduceInPlace(T[i - 1]);
            // // gpu_context.Rescale(T_gpu[i-1], T_gpu[i-1]);

            // const auto to_load_one = scheme->GetElementForEvalAddOrSub(T[i-1], 1.0);
            const uint64_t numElems = T_gpu[i-1].ax__.size()/gpu_context.degree__;
            const auto to_load_one = scheme->GetElementForEvalAddOrSub(cryptoParams, T_gpu[i-1].level, numElems, T_gpu[i-1].noiseScaleDeg, 1.0);
            // cc->EvalAddInPlace(T[i - 1], -1.0);
            const auto one_gpu = LoadIntegerVector(to_load_one);
            gpu_context.SubScalarInPlace(T_gpu[i-1], one_gpu.data());
        }
        else {
            // non-power of 2
            if (i % 2 == 1) {
                // if i is odd
                // compute T_{2i+1}(y) = 2*T_i(y)*T_{i+1}(y) - y

                ckks::CtAccurate left;
                if (i == 3) {
                    // std::vector<DCRTPoly::Integer> scale = scheme->GetElementForEvalMult(T[i/2 - 1], 1);
                    // const auto scale_gpu = LoadIntegerVector(scale);
                    // // std::cout << "loaded integer vector\n";
                    // left = gpu_context.EvalMultConst(T_gpu[i/2-1], scale_gpu);
                    left = gpu_context.EvalMultConstWithLoad(T_gpu[i/2-1], 1, scheme, cryptoParams);
                } else if (i > 3) {
                    // usint c1lvl             = T[i/2 - 1]->GetLevel();
                    usint c1lvl             = T_gpu[i/2-1].level;
                    // assert(c1lvl == T_gpu[i/2-1].level);
                    // const auto sizeQl1            = T[i/2 - 1]->GetElements()[0].GetNumOfElements();
                    const auto sizeQl1 = T_gpu[i/2-1].ax__.size()/gpu_context.degree__;
                    // assert(sizeQl1 == sizeQl1_gpu);
                    // const double scf1 = T[i/2 - 1]->GetScalingFactor();
                    const double scf1 = T_gpu[i/2 - 1].scalingFactor;
                    // assert(scf1 == scf1_gpu);
                    // const double scf2 = T[i/2]->GetScalingFactor();
                    const double scf2 = T_gpu[i/2].scalingFactor;
                    // assert(scf2 == scf2_gpu);
                    const double scf  = cryptoParams->GetScalingFactorReal(c1lvl);
                    const double q1   = cryptoParams->GetModReduceFactor(sizeQl1 - 1);

                    // const double adjustFactor = gpu_context.GetAdjustScalar(T_gpu[i], T_gpu[i/2-1]);

                    // std::cout << std::setprecision(15) << adjustFactor << " " << scf2 / scf1 * q1 / scf << std::endl;

                    // left = gpu_context.EvalMultConstWithLoad(T_gpu[i/2-1], adjustFactor, scheme, cryptoParams);

                    // scale left operand by multiplying by 1
                    // std::vector<DCRTPoly::Integer> scale = scheme->GetElementForEvalMult(T[i/2 - 1], scf2 / scf1 * q1 / scf);
                    const uint32_t numElems = T_gpu[i/2 - 1].ax__.size()/gpu_context.degree__;
                    std::vector<DCRTPoly::Integer> scale = scheme->GetElementForEvalMult(cryptoParams, T_gpu[i/2 - 1].level, numElems, T_gpu[i/2-1].noiseScaleDeg, scf2 / scf1 * q1 / scf);
                    const auto scale_gpu = LoadIntegerVector(scale);
                    // std::cout << "loaded integer vector\n";
                    left = gpu_context.EvalMultConst(T_gpu[i/2-1], scale_gpu);
                    // left = gpu_context.DropLimbs(left);
                    left = gpu_context.Rescale(left);
                    left.scalingFactor = T_gpu[i/2].scalingFactor;
                }
                // std::cout << "Finished constant mult\n";
                // auto prod_gpu = gpu_context.EvalMultAndRelinNoRescale(T_gpu[i / 2 - 1], T_gpu[i / 2], evk);
                // auto prod_gpu = gpu_context.EvalMultAndRelin(T_gpu[i / 2 - 1], T_gpu[i / 2], evk);
                auto prod_gpu = gpu_context.EvalMultAndRelin(left, T_gpu[i / 2], evk);

                // std::cout << "eval mult completed\n";
                // std::cout << "running EvalMult to replicated\n";
                // auto prod = cc->EvalMult(T[i / 2 - 1], T[i / 2]);

                // {
                //     // Check first element
                //     DCRTPoly gpu_b = loadIntoDCRTPoly(prod_gpu.bx__, prod->GetElements()[0].GetParams());
                //     // ckks::HostVector prod_gpu_b_host(prod_gpu.bx__);
                //     for (uint32_t limbInd = 0; limbInd < prod->GetElements()[0].GetParams()->GetParams().size(); limbInd++) {
                //         // std::cout << "checking limb " << limbInd << std::endl;
                //         for (uint32_t dataInd = 0; dataInd < cc->GetRingDimension(); dataInd++) {
                //             if (gpu_b.m_vectors[limbInd].m_values->at(dataInd) != prod->GetElements()[0].m_vectors[limbInd].m_values->at(dataInd)) {
                //                 throw std::logic_error("prod mismatch\n");
                //             }
                //         }
                //     }
                // }
                // { // load correct ciphertext
                //     // std::cout << "prod output limbs: " << prod->GetElements()[0].GetParams()->GetParams().size() << std::endl;
                //     // ckks::HostVector prod_gpu_a(prod_gpu.ax__);
                //     // std::cout << "prod gpu output limbs: " << prod_gpu_a.size() / cc->GetRingDimension() << std::endl;
                //     const auto should_be_prod = LoadAccurateCiphertext(prod);
                //     if (should_be_prod.ax__ != prod_gpu.ax__) throw std::logic_error("prod 1");
                //     if (should_be_prod.bx__ != prod_gpu.bx__) throw std::logic_error("prod 2");
                //     if (prod_gpu.level != should_be_prod.level) throw std::logic_error("prod 3");
                //     if (prod_gpu.noiseScaleDeg != should_be_prod.noiseScaleDeg) throw std::logic_error("prod 4");
                //     // std::cout << prod_gpu.scalingFactor << " " << should_be_prod.scalingFactor << " " << abs(prod_gpu.scalingFactor - should_be_prod.scalingFactor) << std::endl;
                //     // assert(prod_gpu.scalingFactor == should_be_prod.scalingFactor);
                //     if (prod_gpu.scalingFactor != should_be_prod.scalingFactor) {
                //         // std::cout << prod_gpu.scalingFactor << " " << should_be_prod.scalingFactor << std::endl;
                //         std::cout << prod_gpu.scalingFactor << " " << should_be_prod.scalingFactor << " " << abs(prod_gpu.scalingFactor - should_be_prod.scalingFactor) << std::endl;
                //         throw std::logic_error("prod scaling factor");
                //     }
                // }

                // T[i - 1]  = cc->EvalAdd(prod, prod);
                gpu_context.Add(prod_gpu, prod_gpu, T_gpu[i-1]);
                // {
                //     std::cout << "level " << T[i-1]->GetLevel() << " " << T_gpu[i-1].level << std::endl;
                //     std::cout << "noiseScaleDeg " << T[i-1]->GetNoiseScaleDeg() << " " << T_gpu[i-1].noiseScaleDeg << std::endl;
                // }
                // { // load correct ciphertext
                //     const auto should_be_t_gpu = LoadAccurateCiphertext(T[i - 1]);
                //     if (should_be_t_gpu.ax__ != T_gpu[i-1].ax__) throw std::logic_error("sum 1");
                //     if (should_be_t_gpu.bx__ != T_gpu[i-1].bx__) throw std::logic_error("sum 2");
                //     if (T_gpu[i-1].level != should_be_t_gpu.level) throw std::logic_error("sum 3");
                //     if (T_gpu[i-1].noiseScaleDeg != should_be_t_gpu.noiseScaleDeg) throw std::logic_error("sum 4");
                //     // std::cout << T_gpu[i-1].scalingFactor << " " << should_be_t_gpu.scalingFactor << " " << abs(T_gpu[i-1].scalingFactor - should_be_t_gpu.scalingFactor) << std::endl;
                //     if (should_be_t_gpu != T_gpu[i-1]) {
                //         throw std::logic_error("prod sum mismatch\n");
                //     }
                //     // assert(should_be_t_gpu.bx__ == T_gpu[i-1].bx__);
                // }

                // cc->ModReduceInPlace(T[i - 1]);

                // std::cout << "Running EvalSubInPlace\n";

                // const usint c1lvl = T[i-1]->GetLevel();
                auto y_gpu = ckks::CtAccurate(y_gpu_const);
                {
                    // const usint c2lvl = y->GetLevel();
                    // std::cout << c2lvl << " " << y_gpu.level << std::endl;
                    // double scf2 = y->GetScalingFactor();
                    // std::cout << scf2 << " " << y_gpu.scalingFactor << std::endl;
                    // double scf1 = T[i-1]->GetScalingFactor();
                    // std::cout << scf1 << " " << T_gpu[i-1].scalingFactor << std::endl;
                    const usint c2lvl = y_gpu.level;
                    double scf2 = y_gpu.scalingFactor;
                    double scf1 = T_gpu[i-1].scalingFactor;
                    double scf  = cryptoParams->GetScalingFactorReal(c2lvl);
                    const auto y_num_elems = y_gpu.ax__.size()/gpu_context.degree__;
                    std::vector<DCRTPoly::Integer> y_adjust = scheme->GetElementForEvalMult(cryptoParams, y_gpu.level, y_num_elems, y_gpu.noiseScaleDeg, scf1 / scf2 / scf);
                    const auto y_adjust_gpu = LoadIntegerVector(y_adjust);
                    gpu_context.EvalMultConstInPlace(y_gpu, y_adjust_gpu);
                }

                // cc->EvalSubInPlace(T[i - 1], y);

                gpu_context.SubInPlace(T_gpu[i-1], y_gpu);
            }
            else {
                // i is even but not power of 2
                // compute T_{2i}(y) = 2*T_i(y)^2 - 1
                // auto square = cc->EvalSquare(T[i / 2 - 1]);
                auto square_gpu = gpu_context.EvalSquareAndRelin(T_gpu[i/2 - 1], evk);

                // T[i - 1]    = cc->EvalAdd(square, square);
                gpu_context.Add(square_gpu, square_gpu, T_gpu[i-1]);

                // cc->ModReduceInPlace(T[i - 1]);

                // const auto to_load_one = scheme->GetElementForEvalAddOrSub(T[i-1], 1.0);
                const uint32_t numElems = T_gpu[i-1].ax__.size()/gpu_context.degree__;
                const auto to_load_one = scheme->GetElementForEvalAddOrSub(cryptoParams, T_gpu[i-1].level, numElems, T_gpu[i-1].noiseScaleDeg, 1.0);
                // cc->EvalAddInPlace(T[i - 1], -1.0);
                const auto one_gpu = LoadIntegerVector(to_load_one);
                gpu_context.SubScalarInPlace(T_gpu[i-1], one_gpu.data());
            }
        }

        // {
        //     auto T_loaded = x_cpu->CloneZero();
        //     LoadCtAccurateFromGPU(T_loaded, T_gpu[i-1], elementParams);
        //     if (*T_loaded != *T[i-1]) {
        //         throw std::logic_error("T value cpu mismatch\n");
        //     }
        // }
        // {
        //     auto should_be_T_gpu = LoadAccurateCiphertext(T[i-1]);
        //     if (should_be_T_gpu != T_gpu[i-1]) {
        //         throw std::logic_error("T value gpu mismatch\n");
        //     }
        // }
    }

    // std::cout << "T loop complete\n";

    // const auto cryptoParams = std::dynamic_pointer_cast<CryptoParametersCKKSRNS>(T[k - 1]->GetCryptoParameters());

    // auto algo = cc->GetScheme();

    if (cryptoParams->GetScalingTechnique() == FIXEDMANUAL) {
        throw std::logic_error("Not implemented for GPU");
        // original loop
        // // brings all powers of x to the same level
        // for (size_t i = 1; i < k; i++) {
        //     usint levelDiff = T[k - 1]->GetLevel() - T[i - 1]->GetLevel();
        //     cc->LevelReduceInPlace(T[i - 1], nullptr, levelDiff);
        // }
    }
    else {
        for (size_t i = 1; i < k; i++) {
            // std::cout << "Running rescaling iteration " << i << std::endl;

            // algo->AdjustLevelsAndDepthInPlace(T[i - 1], T[k - 1]);

            gpu_context.AdjustLevelsAndDepthInPlace(T_gpu[i-1], T_gpu[k-1], scheme, cryptoParams);

            // {
            //     auto T_loaded = x_cpu->CloneZero();
            //     LoadCtAccurateFromGPU(T_loaded, T_gpu[i-1], elementParams);
            //     if (*T_loaded != *T[i-1]) {
            //         throw std::logic_error("T value cpu mismatch\n");
            //     }
            // }
            // {
            //     auto should_be_T_gpu = LoadAccurateCiphertext(T[i-1]);
            //     if (should_be_T_gpu != T_gpu[i-1]) {
            //         throw std::logic_error("T value gpu mismatch\n");
            //     }
            // }
        }
    }

    // std::vector<Ciphertext<DCRTPoly>> T2(m);
    std::vector<ckks::CtAccurate> T2_gpu(m);
    // Compute the Chebyshev polynomials T_{2k}(y), T_{4k}(y), ... , T_{2^{m-1}k}(y)
    // T2.front() = T.back();  // keep this since T ciphertexts should already be loaded
    T2_gpu[0] = ckks::CtAccurate(T_gpu.back());
    for (uint32_t i = 1; i < m; i++) {
        // std::cout << "running t2 iteration " << i << std::endl;
        // auto square = cc->EvalSquare(T2[i - 1]);
        auto square_gpu = gpu_context.EvalSquareAndRelin(T2_gpu[i-1], evk);
        // T2[i]       = cc->EvalAdd(square, square);
        gpu_context.Add(square_gpu, square_gpu, T2_gpu[i]);
        // cc->ModReduceInPlace(T2[i]);

        // const auto one_scale = scheme->GetElementForEvalAddOrSub(T2[i], 1);
        const uint64_t numElems = T2_gpu[i].ax__.size() / gpu_context.degree__;
        const auto one_scale = scheme->GetElementForEvalAddOrSub(cryptoParams, T2_gpu[i].level, numElems, T2_gpu[i].noiseScaleDeg, 1.0);
        const auto one_scale_gpu = LoadIntegerVector(one_scale);

        // cc->EvalAddInPlace(T2[i], -1.0);

        gpu_context.SubScalarInPlace(T2_gpu[i], one_scale_gpu.data());

        // T2[i] = T.back()->CloneZero();
        // {
        //     auto T2_loaded = T.back()->CloneZero();
        //     LoadCtAccurateFromGPU(T2[i], T2_gpu[i], elementParams);
        //     assert(*T2_loaded == *T2[i]);
        // }
    }

    // std::cout << "T2 loop passed\n";

    // std::cout << "Doublings ("<<m<<") loop passed\n";

    // computes T_{k(2*m - 1)}(y)
    // auto T2km1 = T2.front();
    ckks::CtAccurate T2km1_gpu(T2_gpu[0]);
    for (uint32_t i = 1; i < m; i++) {
        // std::cout << "Running 2km1 iteration " << i << std::endl;
        // compute T_{k(2*m - 1)} = 2*T_{k(2^{m-1}-1)}(y)*T_{k*2^{m-1}}(y) - T_k(y)
        if (i == 1) {
            // usint c1lvl             = T2km1->GetLevel();
            // auto sizeQl1            = T2km1->GetElements()[0].GetNumOfElements();
            // double scf1 = T2km1->GetScalingFactor();
            // double scf2 = T2[i]->GetScalingFactor();
            // double scf  = cryptoParams->GetScalingFactorReal(c1lvl);
            // double q1   = cryptoParams->GetModReduceFactor(sizeQl1 - 1);
            // const auto scale = scheme->GetElementForEvalMult(T2km1, scf2 / scf1 * q1 / scf);
            // const auto scale_gpu = LoadIntegerVector(scale);
            // gpu_context.EvalMultConstInPlace(T2km1_gpu, scale_gpu);
            // T2km1_gpu = gpu_context.Rescale(T2km1_gpu);

            const double adjustScale = gpu_context.GetAdjustScalar(T2km1_gpu, T2_gpu[i]);
            T2km1_gpu = gpu_context.EvalMultConstWithLoad(T2km1_gpu, adjustScale, scheme, cryptoParams);
            T2km1_gpu = gpu_context.Rescale(T2km1_gpu);
            T2km1_gpu.scalingFactor = T2_gpu[i].scalingFactor;
        }
        auto prod_gpu = gpu_context.EvalMultAndRelin(T2km1_gpu, T2_gpu[i], evk);

        // std::cout << "beginning mult to replicate\n";

        // auto prod = cc->EvalMult(T2km1, T2[i]);

        // {
        //     auto should_be_prod = LoadAccurateCiphertext(prod);
        //     if (should_be_prod != prod_gpu) {
        //         std::cout << should_be_prod.level << " " << prod_gpu.level << std::endl;
        //         std::cout << should_be_prod.noiseScaleDeg << " " << prod_gpu.noiseScaleDeg << std::endl;
        //         std::cout << should_be_prod.scalingFactor << " " << prod_gpu.scalingFactor << " " << abs(should_be_prod.scalingFactor - prod_gpu.scalingFactor) << std::endl;
        //         throw std::logic_error("T2km1 first mult mismatch");
        //     }
        // }

        // T2km1     = cc->EvalAdd(prod, prod);
        gpu_context.Add(prod_gpu, prod_gpu, T2km1_gpu);
        // cc->ModReduceInPlace(T2km1);

        ckks::CtAccurate toSub;
        {
            // toSub = gpu_context.EvalMultConst(T2_gpu[0], scale_gpu);
            const double adjustScale = gpu_context.GetAdjustScalar(T2km1_gpu, T2_gpu[0]);

            // {
            //     const uint32_t ct_num_elems = T2_gpu[0].ax__.size()/gpu_context.degree__;
            //     const auto op_scaled = scheme->GetElementForEvalMult(cryptoParams, T2_gpu[0].level, ct_num_elems, T2_gpu[0].noiseScaleDeg, 1.0);
            //     assert(op_scaled == scale);
            // }

            // std::cout << std::setprecision (15) << " " << adjustScale << " " << scf1 / scf2 * q2 / scf << std::endl;
            // assert(adjustScale == scf1 / scf2 * q2 / scf);
            toSub = gpu_context.EvalMultConstWithLoad(T2_gpu[0], adjustScale, scheme, cryptoParams);
            toSub = gpu_context.Rescale(toSub);
            // toSub = gpu_context.Rescale(T2_gpu[0]);
        }        
        gpu_context.SubInPlace(T2km1_gpu, toSub);
        // gpu_context.SubInPlace(T2km1_gpu, T2_gpu[0]);

        // cc->EvalSubInPlace(T2km1, T2.front());

        // {
        //     auto should_be_T2km1 = LoadAccurateCiphertext(T2km1);
        //     if (should_be_T2km1 != T2km1_gpu) {
        //         throw std::logic_error("T2km1 mismatch");
        //     }
        // }
    }

    // std::cout << "T2km1 loop passed\n";

    // // TODO Load gpu result into CPU
    // auto T2km1 = T2[0]->CloneZero();
    // LoadCtAccurateFromGPU(T2km1, T2km1_gpu, elementParams);
    // std::cout << T2km1_loaded->m_scalingFactor << " " << T2km1->m_scalingFactor << std::endl;
    // assert(*T2km1_loaded == *T2km1);


    // We also need to reduce the number of levels of T[k-1] and of T2[0] by another level.
    //  cc->LevelReduceInPlace(T[k-1], nullptr);
    //  cc->LevelReduceInPlace(T2.front(), nullptr);

    // Compute k*2^{m-1}-k because we use it a lot
    uint32_t k2m2k = k * (1 << (m - 1)) - k;

    // Add T^{k(2^m - 1)}(y) to the polynomial that has to be evaluated
    f2.resize(2 * k2m2k + k + 1, 0.0);
    f2.back() = 1;

    // Divide f2 by T^{k*2^{m-1}}
    std::vector<double> Tkm(int32_t(k2m2k + k) + 1, 0.0);
    Tkm.back() = 1;
    auto divqr = LongDivisionChebyshev(f2, Tkm);

    // Subtract x^{k(2^{m-1} - 1)} from r
    std::vector<double> r2 = divqr->r;
    if (int32_t(k2m2k - Degree(divqr->r)) <= 0) {
        r2[int32_t(k2m2k)] -= 1;
        r2.resize(Degree(r2) + 1);
    }
    else {
        r2.resize(int32_t(k2m2k + 1), 0.0);
        r2.back() = -1;
    }

    // Divide r2 by q
    auto divcs = LongDivisionChebyshev(r2, divqr->q);

    // Add x^{k(2^{m-1} - 1)} to s
    std::vector<double> s2 = divcs->r;
    s2.resize(int32_t(k2m2k + 1), 0.0);
    s2.back() = 1;

    // Evaluate c at u
    // Ciphertext<DCRTPoly> cu;
    ckks::CtAccurate cu_gpu;

    uint32_t dc = Degree(divcs->q);
    bool flag_c = false;
    if (dc >= 1) {
        if (dc == 1) {
            if (divcs->q[1] != 1) {
                // cu = cc->EvalMult(T.front(), divcs->q[1]);
                // cc->ModReduceInPlace(cu);
                cu_gpu = gpu_context.EvalMultConstWithLoad(T_gpu.front(), divcs->q[1], scheme, cryptoParams);
            }
            else {
                // cu = T.front();
                cu_gpu = ckks::CtAccurate(T_gpu.front());
            }
        }
        else {
            // std::vector<Ciphertext<DCRTPoly>> ctxs(dc);
            std::vector<ckks::CtAccurate> ctxs_gpu(dc);
            std::vector<double> weights(dc);

            for (uint32_t i = 0; i < dc; i++) {
                // ctxs[i]    = T[i];
                ctxs_gpu[i]    = ckks::CtAccurate(T_gpu[i]);
                weights[i] = divcs->q[i + 1];
            }

            cu_gpu = cc->EvalLinearWSumMutableGPU(ctxs_gpu, cc, weights, gpu_context);
            // cu = cc->EvalLinearWSumMutable(ctxs, weights);
            // {
            //     Ciphertext<DCRTPoly> should_be_cu = cu->CloneZero();
            //     LoadCtAccurateFromGPU(should_be_cu, cu_gpu, elementParams);
            //     if (*should_be_cu != *cu) {
            //         throw std::logic_error("linear w sum output mismatch");
            //     }
            // }
        }

        // cc->EvalAddInPlace(cu, divcs->q.front() / 2);

        gpu_context.EvalSubConstInPlaceWithLoad(cu_gpu, std::fabs(divcs->q.front() / 2), scheme, cryptoParams);

        // TODO : Andrey why not T2[m-1]->GetLevel() instead?
        // Need to reduce levels to the level of T2[m-1].
        //    usint levelDiff = y->GetLevel() - cu->GetLevel() + ceil(log2(k)) + m - 1;
        //    cc->LevelReduceInPlace(cu, nullptr, levelDiff);

        flag_c = true;
    }

    // {
    //     Ciphertext<DCRTPoly> should_be_cu = cu->CloneZero();
    //     LoadCtAccurateFromGPU(should_be_cu, cu_gpu, elementParams);
    //     if (*should_be_cu != *cu) {
    //         throw std::logic_error("cu mismatch");
    //     }
    // }

    // Ciphertext<DCRTPoly> cu = T.front()->CloneZero();
    // LoadCtAccurateFromGPU(cu, cu_gpu, elementParams);

    // Evaluate q and s2 at u. If their degrees are larger than k, then recursively apply the Paterson-Stockmeyer algorithm.
    // Ciphertext<DCRTPoly> qu = T.front()->CloneZero();
    ckks::CtAccurate qu_gpu;

    if (Degree(divqr->q) > k) {
        // {   // check that inputs are the same
        //     Ciphertext<DCRTPoly> should_be_x = x_cpu->CloneZero();
        //     LoadCtAccurateFromGPU(should_be_x, x, elementParams);
        //     if (*should_be_x != *x_cpu) {
        //         throw std::logic_error("x InnerEvalCheby input mismatch");
        //     }
        // }
        // qu = InnerEvalChebyshevPS(x_cpu, divqr->q, k, m - 1, T, T2);
        // qu = InnerEvalChebyshevPSGPU(x, divqr->q, k, m - 1, T, T2, gpu_context);
        // qu_gpu = InnerEvalChebyshevPSGPU(x, x_cpu, divqr->q, k, m - 1, T, T2, T_gpu, T2_gpu, gpu_context, cc);
        qu_gpu = InnerEvalChebyshevPSGPU(x, divqr->q, k, m - 1, T_gpu, T2_gpu, gpu_context, cc);

        // {
        //     Ciphertext<DCRTPoly> should_be_qu = qu->CloneZero();
        //     LoadCtAccurateFromGPU(should_be_qu, qu_gpu, elementParams);
        //     if (*should_be_qu != *qu) {
        //         throw std::logic_error("qu top level InnerEvalCheby mismatch");
        //     }
        // }
    }
    else {
        // dq = k from construction
        // perform scalar multiplication for all other terms and sum them up if there are non-zero coefficients
        auto qcopy = divqr->q;
        qcopy.resize(k);
        if (Degree(qcopy) > 0) {
            // std::vector<Ciphertext<DCRTPoly>> ctxs(Degree(qcopy));
            std::vector<ckks::CtAccurate> ctxs_gpu(Degree(qcopy));
            std::vector<double> weights(Degree(qcopy));

            for (uint32_t i = 0; i < Degree(qcopy); i++) {
                // ctxs[i]    = T[i];
                ctxs_gpu[i]    = ckks::CtAccurate(T_gpu[i]);
                weights[i] = divqr->q[i + 1];
            }

            qu_gpu = cc->EvalLinearWSumMutableGPU(ctxs_gpu, cc, weights, gpu_context);

            // // the highest order coefficient will always be 2 after one division because of the Chebyshev division rule
            // Ciphertext<DCRTPoly> sum = cc->EvalAdd(T[k - 1], T[k - 1]);
            // cc->EvalAddInPlace(qu, sum);

            const auto sum_gpu = gpu_context.Add(T_gpu[k-1], T_gpu[k-1]);
            gpu_context.EvalAddInPlace(qu_gpu, sum_gpu);

        }
        else {
            // qu = T[k - 1];
            qu_gpu = ckks::CtAccurate(T_gpu[k - 1]);

            for (uint32_t i = 1; i < divqr->q.back(); i++) {
                // cc->EvalAddInPlace(qu, T[k - 1]);
                gpu_context.EvalAddInPlace(qu_gpu, T_gpu[k - 1]);
            }
        }

        // {
        //     Ciphertext<DCRTPoly> should_be_qu = qu->CloneZero();
        //     LoadCtAccurateFromGPU(should_be_qu, qu_gpu, elementParams);
        //     if (*should_be_qu != *qu) {
        //         throw std::logic_error("qu branch mismatch");
        //     }
        // }

        // LoadCtAccurateFromGPU(qu, qu_gpu, elementParams);

        // adds the free term (at x^0)
        // cc->EvalAddInPlace(qu, divqr->q.front() / 2);
        if (divqr->q.front() < 0)
            gpu_context.EvalSubConstInPlaceWithLoad(qu_gpu, std::fabs(divqr->q.front() / 2), scheme, cryptoParams);
        else 
            gpu_context.EvalAddConstInPlaceWithLoad(qu_gpu, std::fabs(divqr->q.front() / 2), scheme, cryptoParams);
        // The number of levels of qu is the same as the number of levels of T[k-1] + 1.
        // Will only get here when m = 2, so the number of levels of qu and T2[m-1] will be the same.
    }

    // {
    //     Ciphertext<DCRTPoly> should_be_qu = qu->CloneZero();
    //     LoadCtAccurateFromGPU(should_be_qu, qu_gpu, elementParams);
    //     if (*should_be_qu != *qu) {
    //         throw std::logic_error("qu mismatch");
    //     }
    // }

    // LoadCtAccurateFromGPU(qu, qu_gpu, elementParams);


    // Ciphertext<DCRTPoly> su = T.front()->CloneZero();
    ckks::CtAccurate su_gpu;

    if (Degree(s2) > k) {
        // su = InnerEvalChebyshevPS(x, s2, k, m - 1, T, T2);
        su_gpu = InnerEvalChebyshevPSGPU(x, s2, k, m - 1, T_gpu, T2_gpu, gpu_context, cc);
    }
    else {
        // ds = k from construction
        // perform scalar multiplication for all other terms and sum them up if there are non-zero coefficients
        auto scopy = s2;
        scopy.resize(k);
        if (Degree(scopy) > 0) {
            // std::vector<Ciphertext<DCRTPoly>> ctxs(Degree(scopy));
            std::vector<ckks::CtAccurate> ctxs_gpu(Degree(scopy));
            std::vector<double> weights(Degree(scopy));

            for (uint32_t i = 0; i < Degree(scopy); i++) {
                // ctxs[i]    = T[i];
                ctxs_gpu[i]    = ckks::CtAccurate(T_gpu[i]);
                weights[i] = s2[i + 1];
            }

            su_gpu = cc->EvalLinearWSumMutableGPU(ctxs_gpu, cc, weights, gpu_context);
            // LoadCtAccurateFromGPU(su, su_gpu, elementParams); 

            // the highest order coefficient will always be 1 because s2 is monic.
            // cc->EvalAddInPlace(su, T[k - 1]);
            gpu_context.EvalAddInPlace(su_gpu, T_gpu[k-1]);
        }
        else {
            // su = T[k - 1];
            su_gpu = ckks::CtAccurate(T_gpu[k - 1]);
        }


        // adds the free term (at x^0)
        // cc->EvalAddInPlace(su, s2.front() / 2);
        gpu_context.EvalAddConstInPlaceWithLoad(su_gpu, s2.front() / 2, scheme, cryptoParams);
        // The number of levels of su is the same as the number of levels of T[k-1] + 1.
        // Will only get here when m = 2, so need to reduce the number of levels by 1.
    }

    // LoadCtAccurateFromGPU(su, su_gpu, elementParams); 

    // TODO : Andrey : here is different from 895 line
    // Reduce number of levels of su to number of levels of T2km1.
    //  cc->LevelReduceInPlace(su, nullptr);

    ckks::CtAccurate result_gpu;

    if (flag_c) {
        // result = cc->EvalAdd(T2[m - 1], cu);

        // result_gpu = gpu_context.Add(T2_gpu[m-1], cu_gpu);
        if (T2_gpu[m-1].level != cu_gpu.level) {
            // std::cout << "In level adjust\n";
            // std::cout << T2_gpu[m-1].level << " " << cu_gpu.level << std::endl;
            const double adjust_scale = gpu_context.GetAdjustScalar(T2_gpu[m-1], cu_gpu);
            auto toAdd = gpu_context.EvalMultConstWithLoad(cu_gpu, adjust_scale, scheme, cryptoParams);
            toAdd = gpu_context.Rescale(toAdd);
            toAdd = gpu_context.DropLimbs(toAdd);
            result_gpu = gpu_context.Add(T2_gpu[m-1], toAdd);
        } else {
            result_gpu = gpu_context.Add(T2_gpu[m-1], cu_gpu);
        }
    }
    else {
        // result = cc->EvalAdd(T2[m - 1], divcs->q.front() / 2);

        if (divcs->q.front() < 0) result_gpu = gpu_context.EvalSubConstWithLoad(T2_gpu[m-1], std::fabs(divcs->q.front() / 2), scheme, cryptoParams);
        else result_gpu = gpu_context.EvalAddConstWithLoad(T2_gpu[m-1], divcs->q.front() / 2, scheme, cryptoParams);
    }

    // result = cc->EvalMult(result, qu);
    result_gpu = gpu_context.EvalMultAndRelin(result_gpu, qu_gpu, *gpu_context.preloaded_evaluation_key);

    // cc->ModReduceInPlace(result);

    // cc->EvalAddInPlace(result, su);

    const double adjust_scale = gpu_context.GetAdjustScalar(result_gpu, su_gpu);
    auto toAdd = gpu_context.EvalMultConstWithLoad(su_gpu, adjust_scale, scheme, cryptoParams);
    toAdd = gpu_context.Rescale(toAdd);
    // gpu_context.EvalAddInPlace(result_gpu, su_gpu);
    gpu_context.EvalAddInPlace(result_gpu, toAdd);

    // cc->EvalSubInPlace(result, T2km1);
    gpu_context.SubInPlace(result_gpu, T2km1_gpu);

    return result_gpu;

    // Ciphertext<DCRTPoly> result = x->CloneZero();
    // LoadCtAccurateFromGPU(result, result_gpu, elementParams);

    // return result;
}



};