#define PROFILE
#define BOOTSTRAPTIMING

#include "scheme/ckksrns/ckksrns-leveledshe.h"
#include "scheme/ckksrns/ckksrns-advancedshe.h"
#include "scheme/ckksrns/ckksrns-fhe.h"

#include "key/privatekey.h"
#include "scheme/ckksrns/ckksrns-cryptoparameters.h"
#include "schemebase/base-scheme.h"
#include "cryptocontext.h"
#include "ciphertext.h"

#include "lattice/lat-hal.h"

#include "math/hal/basicint.h"
#include "math/dftransform.h"

#include "utils/exception.h"
#include "utils/parallel.h"
#include "utils/utilities.h"
#include "scheme/ckksrns/ckksrns-utils.h"

#include "gpu/Utils.h"

#include <cmath>
#include <memory>
#include <vector>
#include <assert.h>

// #define assert(x, s) if (!x) throw std::logic_error(s);
// #define assert(x) assert(x, "assert failure\n")

namespace lbcrypto {

// void ApplyDoubleAngleIterationsGPU(Ciphertext<DCRTPoly>& ciphertext, ckks::CtAccurate& gpu_ct, uint32_t numIter, const ckks::Context& gpu_context) {
void ApplyDoubleAngleIterationsGPU(ckks::CtAccurate& gpu_ct, uint32_t numIter, const ckks::Context& gpu_context, const CryptoContext<DCRTPoly>& cc) {
    // auto cc = ciphertext->GetCryptoContext();

    const auto elementParams = cc->GetElementParams();

    const int32_t r = numIter;

    // ckks::CtAccurate gpu_ct = LoadAccurateCiphertext(ciphertext);
    // assert(gpu_ct == gpu_ciphertext);
    // ckks::EvaluationKey evk = LoadEvalMultRelinKey(cc);
    ckks::EvaluationKey& evk = *gpu_context.preloaded_evaluation_key;

    const auto cryptoParams = std::dynamic_pointer_cast<CryptoParametersCKKSRNS>(cc->GetCryptoParameters());
    const auto scheme = std::dynamic_pointer_cast<LeveledSHECKKSRNS>(cc->GetScheme());

    uint32_t level = gpu_ct.level;
    uint32_t numElems = gpu_ct.ax__.size()/gpu_context.degree__;
    const uint32_t noiseScaleDeg = 2;
    for (int32_t j = 1; j < r + 1; j++) {
        if (j == 1) {
            gpu_ct = gpu_context.EvalSquareAndRelinNoRescale(gpu_ct, evk);
        } else {
            gpu_ct = gpu_context.EvalSquareAndRelin(gpu_ct, evk);
            level += 1; 
        }
        gpu_context.Add(gpu_ct, gpu_ct, gpu_ct);
        // TODO: Preload these scalars in the bootstrapping parameters
        double scalar = -1.0 / std::pow((2.0 * M_PI), std::pow(2.0, j - r));
        double scalar_abs = std::fabs(scalar);
        const auto toSub = scheme->GetElementForEvalAddOrSub(cryptoParams, level, numElems, noiseScaleDeg, scalar_abs);
        // {  // TODO: this test fails but the end-to-end bootstrapping still works....
        //     const auto toSub_orig = scheme->GetElementForEvalAddOrSub(ciphertext, scalar_abs);
        //     std::cout << toSub_orig << " " << toSub << std::endl;
        //     assert(toSub_orig == toSub);
        // }
        const auto toSub_gpu = LoadIntegerVector(toSub);
        gpu_context.SubScalarInPlace(gpu_ct, toSub_gpu.data());

        numElems -= 1;
    }

    // const auto resParams = ciphertext->GetElements()[0].GetParams();

    // std::vector<DCRTPoly> gpu_res_elems(2);
    // gpu_res_elems[0] = loadIntoDCRTPoly(gpu_ct.bx__, elementParams);
    // gpu_res_elems[1] = loadIntoDCRTPoly(gpu_ct.ax__, elementParams);

    // // Original loop
    // // for (int32_t j = 1; j < r + 1; j++) {
    // //     cc->EvalSquareInPlace(ciphertext);
    // //     ciphertext    = cc->EvalAdd(ciphertext, ciphertext);
    // //     double scalar = -1.0 / std::pow((2.0 * M_PI), std::pow(2.0, j - r));
    // //     cc->EvalAddInPlace(ciphertext, scalar);
    // //     cc->ModReduceInPlace(ciphertext);
    // // }

    // ciphertext->SetElements(gpu_res_elems);
    // ciphertext->SetLevel(level);
    // // ciphertext->SetScalingFactor(ciphertext->m_scalingFactor * ciphertext->m_scalingFactor);
    // ciphertext->SetScalingFactor(gpu_ct.scalingFactor);

    // LoadCtAccurateFromGPU(ciphertext, gpu_ct, elementParams);
}

// Ciphertext<DCRTPoly> 
ckks::CtAccurate
FHECKKSRNS::EvalCoeffsToSlotsGPU(
    // const std::vector<std::vector<ConstPlaintext>>& A,
    const std::vector<std::vector<ckks::PtAccurate>>& A_gpu, 
    ConstCiphertext<DCRTPoly> ctxt, 
    const ckks::CtAccurate& ctxt_gpu,
    const ckks::Context& gpu_context) const {
    
    // std::cout << "Running EvalCoeffToSlots GPU\n";
    
    // const ckks::CtAccurate ctxt_gpu = LoadAccurateCiphertext(ctxt); 

    uint32_t slots = ctxt->GetSlots();

    auto pair = m_bootPrecomMap.find(slots);
    if (pair == m_bootPrecomMap.end()) {
        std::string errorMsg(std::string("Precomputations for ") + std::to_string(slots) +
                             std::string(" slots were not generated") +
                             std::string(" Need to call EvalBootstrapSetup and EvalBootstrapKeyGen to proceed"));
        OPENFHE_THROW(type_error, errorMsg);
    }
    const std::shared_ptr<CKKSBootstrapPrecom> precom = pair->second;

    const auto cc    = ctxt->GetCryptoContext();
    const auto evalKeys = cc->GetEvalAutomorphismKeyMap(ctxt->GetKeyTag());
    const auto elementParams = cc->GetElementParams();
    uint32_t M = cc->GetCyclotomicOrder();
    uint32_t N = cc->GetRingDimension();

    int32_t levelBudget     = precom->m_paramsEnc[CKKS_BOOT_PARAMS::LEVEL_BUDGET];
    int32_t layersCollapse  = precom->m_paramsEnc[CKKS_BOOT_PARAMS::LAYERS_COLL];
    int32_t remCollapse     = precom->m_paramsEnc[CKKS_BOOT_PARAMS::LAYERS_REM];
    int32_t numRotations    = precom->m_paramsEnc[CKKS_BOOT_PARAMS::NUM_ROTATIONS];
    int32_t b               = precom->m_paramsEnc[CKKS_BOOT_PARAMS::BABY_STEP];
    int32_t g               = precom->m_paramsEnc[CKKS_BOOT_PARAMS::GIANT_STEP];
    int32_t numRotationsRem = precom->m_paramsEnc[CKKS_BOOT_PARAMS::NUM_ROTATIONS_REM];
    int32_t bRem            = precom->m_paramsEnc[CKKS_BOOT_PARAMS::BABY_STEP_REM];
    int32_t gRem            = precom->m_paramsEnc[CKKS_BOOT_PARAMS::GIANT_STEP_REM];

    int32_t stop    = -1;
    int32_t flagRem = 0;

    auto algo = cc->GetScheme();

    if (remCollapse != 0) {
        stop    = 0;
        flagRem = 1;
    }

    // precompute the inner and outer rotations
    std::vector<std::vector<int32_t>> rot_in(levelBudget);
    for (uint32_t i = 0; i < uint32_t(levelBudget); i++) {
        if (flagRem == 1 && i == 0) {
            // remainder corresponds to index 0 in encoding and to last index in decoding
            rot_in[i] = std::vector<int32_t>(numRotationsRem + 1);
        }
        else {
            rot_in[i] = std::vector<int32_t>(numRotations + 1);
        }
    }

    std::vector<std::vector<int32_t>> rot_out(levelBudget);
    for (uint32_t i = 0; i < uint32_t(levelBudget); i++) {
        rot_out[i] = std::vector<int32_t>(b + bRem);
    }

    for (int32_t s = levelBudget - 1; s > stop; s--) {
        for (int32_t j = 0; j < g; j++) {
            rot_in[s][j] = ReduceRotation(
                (j - int32_t((numRotations + 1) / 2) + 1) * (1 << ((s - flagRem) * layersCollapse + remCollapse)),
                slots);
        }

        for (int32_t i = 0; i < b; i++) {
            rot_out[s][i] = ReduceRotation((g * i) * (1 << ((s - flagRem) * layersCollapse + remCollapse)), M / 4);
        }
    }

    if (flagRem) {
        for (int32_t j = 0; j < gRem; j++) {
            rot_in[stop][j] = ReduceRotation((j - int32_t((numRotationsRem + 1) / 2) + 1), slots);
        }

        for (int32_t i = 0; i < bRem; i++) {
            rot_out[stop][i] = ReduceRotation((gRem * i), M / 4);
        }
    }

    // Ciphertext<DCRTPoly> result = ctxt->Clone();
    ckks::CtAccurate result_gpu(ctxt_gpu);

    // hoisted automorphisms
    for (int32_t s = levelBudget - 1; s > stop; s--) {
        // std::cout << "Running hoisting iteration " << levelBudget - s << std::endl;
        if (s != levelBudget - 1) {
            // algo->ModReduceInternalInPlace(result, BASE_NUM_LEVELS_TO_DROP);
            result_gpu = gpu_context.Rescale(result_gpu);
        }

        // computes the NTTs for each CRT limb (for the hoisted automorphisms used later on)
        // auto digits = cc->EvalFastRotationPrecompute(result);
        auto digits_gpu = gpu_context.ModUp(result_gpu.ax__);

        // {
        //     ckks::HostVector digits_gpu_host(digits_gpu);
        //
        //     // check digits. // gpu digits are concatenated
        //     const uint32_t numDigits = digits->size();
        //     const uint32_t numLimbs = digits->at(0).GetNumOfElements();
        //     const uint32_t phim = digits->at(0).GetRingDimension();
        //     // std::cout << numDigits << " " << numLimbs << " " << phim << std::endl;
        //     // std::cout << digits_gpu_host.size() << " " << digits_gpu_host.size()/(phim*numLimbs) << std::endl;
        //     assert(digits_gpu_host.size() == phim * numLimbs * numDigits);
        //
        //     for (uint32_t digit_ind = 0; digit_ind < numDigits; digit_ind++) {
        //         for (uint32_t limbInd = 0; limbInd < numLimbs; limbInd++) {
        //             for (uint32_t dataInd = 0; dataInd < phim; dataInd++) {
        //                 const uint32_t gpu_index = digit_ind*numLimbs*phim + limbInd*phim  + dataInd;
        //                 const uint64_t gpu_val = digits_gpu_host[gpu_index];
        //                 assert(gpu_val == digits->at(digit_ind).m_vectors[limbInd].m_values->at(dataInd));
        //             }
        //         }
        //     }
        // }

        // std::vector<Ciphertext<DCRTPoly>> fastRotation(g);
        std::vector<ckks::CtAccurate> fastRotation_gpu(g);
// #pragma omp parallel for
        for (int32_t j = 0; j < g; j++) {
            // std::cout << "Running inner loop " << j << std::endl;
            if (rot_in[s][j] != 0) {
                // fastRotation[j] = cc->EvalFastRotationExt(result, rot_in[s][j], digits, true);
                // std::cout << "TODO: write gpu fast rotation function\n";
                fastRotation_gpu[j] = EvalFastRotateExtGPU(result_gpu, digits_gpu, rot_in[s][j], evalKeys, cc, gpu_context, true);
                // {
                //     const auto shouldBeFastRotGPU = LoadAccurateCiphertext(fastRotation[j]);
                //     assert(fastRotation_gpu[j] == shouldBeFastRotGPU);
                // }
            }
            else {
                // fastRotation[j] = cc->KeySwitchExt(result, true);
                // this seems to be just a null raise. the boolean indicates if multiplication by P should be performed
                const uint32_t totalLimbs = result_gpu.ax__.size()/gpu_context.degree__ + gpu_context.param__.num_special_moduli_;
                // assert(fastRotation[j]->GetElements()[0].GetNumOfElements() == totalLimbs);
                // fastRotation_gpu[j] = ckks::CtAccurate(result_gpu);
                fastRotation_gpu[j].level = result_gpu.level;
                fastRotation_gpu[j].noiseScaleDeg = result_gpu.noiseScaleDeg;
                fastRotation_gpu[j].scalingFactor = result_gpu.scalingFactor;
                // fastRotation_gpu[j].bx__ = LoadConstantVector(gpu_context.degree__ * totalLimbs, 0);
                fastRotation_gpu[j].bx__.resize(gpu_context.degree__ * totalLimbs);
                fastRotation_gpu[j].bx__.setConstant(0);
                gpu_context.AddScaledMessageTerm(fastRotation_gpu[j].bx__ , result_gpu.bx__);
                // fastRotation_gpu[j].ax__ = LoadConstantVector(gpu_context.degree__ * totalLimbs, 0);
                fastRotation_gpu[j].ax__.resize(gpu_context.degree__ * totalLimbs);
                fastRotation_gpu[j].ax__.setConstant(0);
                gpu_context.AddScaledMessageTerm(fastRotation_gpu[j].ax__ , result_gpu.ax__);

                // {
                //     const auto shouldBeFastRotGPU = LoadAccurateCiphertext(fastRotation[j]);
                //     assert(fastRotation_gpu[j] == shouldBeFastRotGPU);
                // }
            }
        }

        // Ciphertext<DCRTPoly> outer;
        ckks::CtAccurate outer_gpu;
        // DCRTPoly first;
        ckks::DeviceVector first_gpu;
        for (int32_t i = 0; i < b; i++) {
            // std::cout << "running outer & first loop iteration " << i << " " << rot_out[s][i] << std::endl;
            // for the first iteration with j=0:
            int32_t G                  = g * i;
            // Ciphertext<DCRTPoly> inner = EvalMultExt(fastRotation[0], A[s][G]);
            // assert(A[s][G]->GetElement<DCRTPoly>().GetFormat() == Format::EVALUATION);
            // const ckks::PtAccurate A_gpu = LoadAccuratePlaintext(A[s][G], A[s][G]->GetElement<DCRTPoly>());
            // ckks::CtAccurate inner_gpu = gpu_context.EvalMultPlainExt(fastRotation_gpu[0], A_gpu);
            ckks::CtAccurate inner_gpu = gpu_context.EvalMultPlainExt(fastRotation_gpu[0], A_gpu[s][G]);
            // {
            //     ckks::HostVector inner_gpu_b(inner_gpu.bx__);
            //     const uint32_t numLimbs = inner->GetElements()[0].GetNumOfElements();
            //     const uint32_t phim = inner->GetElements()[0].GetRingDimension();
            //     for (uint32_t limbInd = 0; limbInd < numLimbs; limbInd++) {
            //         for (uint32_t dataInd = 0; dataInd < phim; dataInd++) {
            //             const uint32_t gpu_index = limbInd*phim  + dataInd;
            //             const uint64_t gpu_val_b = inner_gpu_b[gpu_index];
            //             // std::cout << "gpu_index: " << gpu_index << std::endl;
            //             // std::cout << gpu_val_b << " " << inner->GetElements()[0].m_vectors[limbInd].m_values->at(dataInd) << std::endl;
            //             assert(gpu_val_b == inner->GetElements()[0].m_vectors[limbInd].m_values->at(dataInd));
            //         }
            //     }
            //     const auto should_be_inner_gpu = LoadAccurateCiphertext(inner);
            //     assert(should_be_inner_gpu == inner_gpu);
            // }

            // continue the loop
            for (int32_t j = 1; j < g; j++) {
                if ((G + j) != int32_t(numRotations)) {
                    // EvalAddExtInPlace(inner, EvalMultExt(fastRotation[j], A[s][G + j]));
                    // const ckks::PtAccurate A_gpu = LoadAccuratePlaintext(A[s][G + j], A[s][G + j]->GetElement<DCRTPoly>());
                    // const ckks::CtAccurate toAdd = gpu_context.EvalMultPlainExt(fastRotation_gpu[j], A_gpu);
                    const ckks::CtAccurate toAdd = gpu_context.EvalMultPlainExt(fastRotation_gpu[j], A_gpu[s][G + j]);
                    gpu_context.EvalAddInPlaceExt(inner_gpu, toAdd);
                    // {
                    //     const auto should_be_inner_gpu = LoadAccurateCiphertext(inner);
                    //     assert(should_be_inner_gpu == inner_gpu);
                    // }
                }
            }

            if (i == 0) {
                // first         = cc->KeySwitchDownFirstElement(inner);
                gpu_context.ModDown(inner_gpu.bx__, first_gpu);
                // {
                //     ckks::DeviceVector should_be_first = loadIntoDeviceVector(first);
                //     assert(first_gpu == should_be_first);
                // }
                // auto elements = inner->GetElements();
                // elements[0].SetValuesToZero();
                // inner->SetElements(elements);
                // inner_gpu.bx__ = LoadConstantVector(inner_gpu.bx__.size(), 0);
                inner_gpu.bx__.setConstant(0);
                // outer = inner;
                outer_gpu = ckks::CtAccurate(inner_gpu);
                // {
                //     ckks::CtAccurate should_be_inner_gpu = LoadAccurateCiphertext(inner);
                //     assert(inner_gpu == should_be_inner_gpu);
                //     ckks::CtAccurate should_be_outer_gpu = LoadAccurateCiphertext(outer);
                //     assert(outer_gpu == should_be_outer_gpu);
                // }
            } else {
                // {
                //     ckks::DeviceVector should_be_first = loadIntoDeviceVector(first);
                //     assert(first_gpu == should_be_first);
                // }
                if (rot_out[s][i] != 0) {
                    // inner = cc->KeySwitchDown(inner);
                    ckks::DeviceVector temp(inner_gpu.ax__);
                    gpu_context.ModDown(temp, inner_gpu.ax__);
                    temp = ckks::DeviceVector(inner_gpu.bx__);
                    gpu_context.ModDown(temp, inner_gpu.bx__);
                    // {
                    //     ckks::CtAccurate should_be_inner = LoadAccurateCiphertext(inner);
                    //     assert(inner_gpu == should_be_inner);
                    // }
                    // Find the automorphism index that corresponds to rotation index index.
                    usint autoIndex = FindAutomorphismIndex2nComplex(rot_out[s][i], M);
                    std::vector<usint> map(N);
                    PrecomputeAutoMap(N, autoIndex, &map);
                    // first += inner->GetElements()[0].AutomorphismTransform(autoIndex, map);
                    ckks::DeviceVector inner_b_rot = gpu_context.AutomorphismTransform(inner_gpu.bx__, map);
                    gpu_context.AddCoreInPlace(first_gpu, inner_b_rot);
                    // {
                    //     ckks::DeviceVector should_be_first = loadIntoDeviceVector(first);
                    //     assert(first_gpu == should_be_first);
                    // }
                    // auto innerDigits = cc->EvalFastRotationPrecompute(inner);
                    auto innerDigits_gpu = gpu_context.ModUp(inner_gpu.ax__);
                    // EvalAddExtInPlace(outer, cc->EvalFastRotationExt(inner, rot_out[s][i], innerDigits, false));
                    auto inner_rot_gpu = EvalFastRotateExtGPU(inner_gpu, innerDigits_gpu, rot_out[s][i], evalKeys, cc, gpu_context, false);
                    gpu_context.EvalAddInPlaceExt(outer_gpu, inner_rot_gpu);
                } else {
                    // first += cc->KeySwitchDownFirstElement(inner);
                    ckks::DeviceVector inner_mod_down; gpu_context.ModDown(inner_gpu.bx__, inner_mod_down);
                    gpu_context.AddCoreInPlace(first_gpu, inner_mod_down);
                    // {
                    //     ckks::DeviceVector should_be_first = loadIntoDeviceVector(first);
                    //     assert(first_gpu == should_be_first);
                    // }
                    // auto elements = inner->GetElements();
                    // elements[0].SetValuesToZero();
                    // inner->SetElements(elements);
                    // inner_gpu.bx__ = LoadConstantVector(inner_gpu.bx__.size(), 0);
                    inner_gpu.bx__.setConstant(0);
                    // EvalAddExtInPlace(outer, inner);
                    gpu_context.EvalAddInPlaceExt(outer_gpu, inner_gpu);
                }

                // {
                //     ckks::CtAccurate should_be_outer = LoadAccurateCiphertext(outer);
                //     assert(outer_gpu == should_be_outer);
                // }
            }
        }
        // result                          = cc->KeySwitchDown(outer);
        gpu_context.ModDown(outer_gpu.ax__, result_gpu.ax__);
        gpu_context.ModDown(outer_gpu.bx__, result_gpu.bx__);
        result_gpu.level = outer_gpu.level;
        result_gpu.noiseScaleDeg = outer_gpu.noiseScaleDeg;
        result_gpu.scalingFactor = outer_gpu.scalingFactor;
        // {
        //     const auto should_be_result = LoadAccurateCiphertext(result);
        //     assert(result_gpu == should_be_result);
        // }
        // std::vector<DCRTPoly>& elements = result->GetElements();
        // elements[0] += first;
        
        gpu_context.AddCoreInPlace(result_gpu.bx__, first_gpu);
        // {
        //     const auto should_be_result = LoadAccurateCiphertext(result);
        //     assert(result_gpu == should_be_result);
        // }
    }

    if (flagRem) {
        // std::cout << "running flagRem branch\n";
        // throw std::logic_error("flagRem branch is not implemented for the GPU\n");

        // algo->ModReduceInternalInPlace(result, BASE_NUM_LEVELS_TO_DROP);
        result_gpu = gpu_context.Rescale(result_gpu);

        // computes the NTTs for each CRT limb (for the hoisted automorphisms used later on)
        // auto digits = cc->EvalFastRotationPrecompute(result);
        auto digits_gpu = gpu_context.ModUp(result_gpu.ax__);

        // std::vector<Ciphertext<DCRTPoly>> fastRotation(gRem);
        std::vector<ckks::CtAccurate> fastRotation_gpu(gRem);

// #pragma omp parallel for
        for (int32_t j = 0; j < gRem; j++) {
            if (rot_in[stop][j] != 0) {
                // fastRotation[j] = cc->EvalFastRotationExt(result, rot_in[stop][j], digits, true);
                fastRotation_gpu[j] = EvalFastRotateExtGPU(result_gpu, digits_gpu, rot_in[stop][j], evalKeys, cc, gpu_context, true);
            }
            else {
                // fastRotation[j] = cc->KeySwitchExt(result, true);
                const uint32_t totalLimbs = result_gpu.ax__.size()/gpu_context.degree__ + gpu_context.param__.num_special_moduli_;
                // assert(fastRotation[j]->GetElements()[0].GetNumOfElements() == totalLimbs);
                fastRotation_gpu[j] = ckks::CtAccurate(result_gpu);
                // fastRotation_gpu[j].bx__ = LoadConstantVector(gpu_context.degree__ * totalLimbs, 0);
                fastRotation_gpu[j].bx__.resize(gpu_context.degree__ * totalLimbs);
                fastRotation_gpu[j].bx__.setConstant(0);
                gpu_context.AddScaledMessageTerm(fastRotation_gpu[j].bx__ , result_gpu.bx__);
                // fastRotation_gpu[j].ax__ = LoadConstantVector(gpu_context.degree__ * totalLimbs, 0);
                fastRotation_gpu[j].ax__.resize(gpu_context.degree__ * totalLimbs);
                fastRotation_gpu[j].ax__.setConstant(0);
                gpu_context.AddScaledMessageTerm(fastRotation_gpu[j].ax__ , result_gpu.ax__);
            }

            // {
            //     const auto should_be_fast_rotation = LoadAccurateCiphertext(fastRotation[j]);
            //     assert(should_be_fast_rotation == fastRotation_gpu[j]);
            // }
        }

        // Ciphertext<DCRTPoly> outer;
        ckks::CtAccurate outer_gpu;
        // DCRTPoly first;
        ckks::DeviceVector first_gpu;
        for (int32_t i = 0; i < bRem; i++) {
            // Ciphertext<DCRTPoly> inner;
            ckks::CtAccurate inner_gpu;
            // for the first iteration with j=0:
            int32_t GRem = gRem * i;
            // inner        = EvalMultExt(fastRotation[0], A[stop][GRem]);
            // const ckks::PtAccurate A_gpu = LoadAccuratePlaintext(A[stop][GRem], A[stop][GRem]->GetElement<DCRTPoly>());
            // inner_gpu = gpu_context.EvalMultPlainExt(fastRotation_gpu[0], A_gpu);
            inner_gpu = gpu_context.EvalMultPlainExt(fastRotation_gpu[0], A_gpu[stop][GRem]);

            // continue the loop
            for (int32_t j = 1; j < gRem; j++) {
                if ((GRem + j) != int32_t(numRotationsRem)) {
                    // EvalAddExtInPlace(inner, EvalMultExt(fastRotation[j], A[stop][GRem + j]));
                    // const ckks::PtAccurate A_gpu = LoadAccuratePlaintext(A[stop][GRem + j], A[stop][GRem + j]->GetElement<DCRTPoly>());
                    // const ckks::CtAccurate toAdd = gpu_context.EvalMultPlainExt(fastRotation_gpu[j], A_gpu);
                    const ckks::CtAccurate toAdd = gpu_context.EvalMultPlainExt(fastRotation_gpu[j], A_gpu[stop][GRem + j]);
                    gpu_context.EvalAddInPlaceExt(inner_gpu, toAdd);
                }
            }

            if (i == 0) {
                // first         = cc->KeySwitchDownFirstElement(inner);
                gpu_context.ModDown(inner_gpu.bx__, first_gpu);

                // auto elements = inner->GetElements();
                // elements[0].SetValuesToZero();
                // inner->SetElements(elements);
                // outer = inner;

                // inner_gpu.bx__ = LoadConstantVector(inner_gpu.bx__.size(), 0);
                inner_gpu.bx__.setConstant(0);
                outer_gpu = ckks::CtAccurate(inner_gpu);
            }
            else {
                if (rot_out[stop][i] != 0) {
                    // inner = cc->KeySwitchDown(inner);
                    ckks::DeviceVector temp(inner_gpu.ax__);
                    gpu_context.ModDown(temp, inner_gpu.ax__);
                    temp = ckks::DeviceVector(inner_gpu.bx__);
                    gpu_context.ModDown(temp, inner_gpu.bx__);

                    // Find the automorphism index that corresponds to rotation index index.
                    usint autoIndex = FindAutomorphismIndex2nComplex(rot_out[stop][i], M);
                    std::vector<usint> map(N);
                    PrecomputeAutoMap(N, autoIndex, &map);

                    // first += inner->GetElements()[0].AutomorphismTransform(autoIndex, map);
                    ckks::DeviceVector inner_b_rot = gpu_context.AutomorphismTransform(inner_gpu.bx__, map);
                    gpu_context.AddCoreInPlace(first_gpu, inner_b_rot);

                    // auto innerDigits = cc->EvalFastRotationPrecompute(inner);
                    auto innerDigits_gpu = gpu_context.ModUp(inner_gpu.ax__);
                    // EvalAddExtInPlace(outer, cc->EvalFastRotationExt(inner, rot_out[stop][i], innerDigits, false));
                    auto inner_rot_gpu = EvalFastRotateExtGPU(inner_gpu, innerDigits_gpu, rot_out[stop][i], evalKeys, cc, gpu_context, false);
                    gpu_context.EvalAddInPlaceExt(outer_gpu, inner_rot_gpu);
                }
                else {
                    // first += cc->KeySwitchDownFirstElement(inner);
                    ckks::DeviceVector inner_mod_down; gpu_context.ModDown(inner_gpu.bx__, inner_mod_down);
                    gpu_context.AddCoreInPlace(first_gpu, inner_mod_down);

                    // auto elements = inner->GetElements();
                    // elements[0].SetValuesToZero();
                    // inner->SetElements(elements);
                    // inner_gpu.bx__ = LoadConstantVector(inner_gpu.bx__.size(), 0);
                    inner_gpu.bx__.setConstant(0);

                    // EvalAddExtInPlace(outer, inner);
                    gpu_context.EvalAddInPlaceExt(outer_gpu, inner_gpu);
                }

                // {
                //     ckks::CtAccurate should_be_outer = LoadAccurateCiphertext(outer);
                //     assert(outer_gpu == should_be_outer);
                // }
            }
        }

        // result                          = cc->KeySwitchDown(outer);
        gpu_context.ModDown(outer_gpu.ax__, result_gpu.ax__);
        gpu_context.ModDown(outer_gpu.bx__, result_gpu.bx__);
        result_gpu.level = outer_gpu.level;
        result_gpu.noiseScaleDeg = outer_gpu.noiseScaleDeg;
        result_gpu.scalingFactor = outer_gpu.scalingFactor;
        // std::vector<DCRTPoly>& elements = result->GetElements();
        // elements[0] += first;
        gpu_context.AddCoreInPlace(result_gpu.bx__, first_gpu);
    }

    // {
    //     const auto should_be_result = LoadAccurateCiphertext(result);
    //     assert(result_gpu == should_be_result);
    // }

    return result_gpu;
    // Ciphertext<DCRTPoly> result = ctxt->Clone();
    // LoadCtAccurateFromGPU(result, result_gpu, elementParams);

    // return result;
}

ckks::CtAccurate  // Ciphertext<DCRTPoly> 
FHECKKSRNS::EvalSlotsToCoeffsGPU(
    // const std::vector<std::vector<ConstPlaintext>>& A,
    const std::vector<std::vector<ckks::PtAccurate>>& A,
    // ConstCiphertext<DCRTPoly> ctxt, 
    const ckks::CtAccurate& ctxt_gpu, const uint32_t slots, const string keyTag,
    const ckks::Context& gpu_context, const CryptoContext<DCRTPoly>& cc) const {
    
    // std::cout << "Running EvalSlotsToCoeffs GPU\n";

    // const ckks::CtAccurate ctxt_gpu = LoadAccurateCiphertext(ctxt); 
    
    // uint32_t slots = ctxt->GetSlots();

    auto pair = m_bootPrecomMap.find(slots);
    if (pair == m_bootPrecomMap.end()) {
        std::string errorMsg(std::string("Precomputations for ") + std::to_string(slots) +
                             std::string(" slots were not generated") +
                             std::string(" Need to call EvalBootstrapSetup and EvalBootstrapKeyGen to proceed"));
        OPENFHE_THROW(type_error, errorMsg);
    }

    const std::shared_ptr<CKKSBootstrapPrecom> precom = pair->second;

    // const auto cc    = ctxt->GetCryptoContext();
    
    // const auto evalKeys = cc->GetEvalAutomorphismKeyMap(ctxt->GetKeyTag());
    const auto evalKeys = cc->GetEvalAutomorphismKeyMap(keyTag);
    const auto elementParams = cc->GetElementParams();
    uint32_t M = cc->GetCyclotomicOrder();
    uint32_t N = cc->GetRingDimension();

    int32_t levelBudget     = precom->m_paramsDec[CKKS_BOOT_PARAMS::LEVEL_BUDGET];
    int32_t layersCollapse  = precom->m_paramsDec[CKKS_BOOT_PARAMS::LAYERS_COLL];
    int32_t remCollapse     = precom->m_paramsDec[CKKS_BOOT_PARAMS::LAYERS_REM];
    int32_t numRotations    = precom->m_paramsDec[CKKS_BOOT_PARAMS::NUM_ROTATIONS];
    int32_t b               = precom->m_paramsDec[CKKS_BOOT_PARAMS::BABY_STEP];
    int32_t g               = precom->m_paramsDec[CKKS_BOOT_PARAMS::GIANT_STEP];
    int32_t numRotationsRem = precom->m_paramsDec[CKKS_BOOT_PARAMS::NUM_ROTATIONS_REM];
    int32_t bRem            = precom->m_paramsDec[CKKS_BOOT_PARAMS::BABY_STEP_REM];
    int32_t gRem            = precom->m_paramsDec[CKKS_BOOT_PARAMS::GIANT_STEP_REM];

    auto algo = cc->GetScheme();

    int32_t flagRem = 0;

    if (remCollapse != 0) {
        flagRem = 1;
    }

    // precompute the inner and outer rotations

    std::vector<std::vector<int32_t>> rot_in(levelBudget);
    for (uint32_t i = 0; i < uint32_t(levelBudget); i++) {
        if (flagRem == 1 && i == uint32_t(levelBudget - 1)) {
            // remainder corresponds to index 0 in encoding and to last index in decoding
            rot_in[i] = std::vector<int32_t>(numRotationsRem + 1);
        }
        else {
            rot_in[i] = std::vector<int32_t>(numRotations + 1);
        }
    }

    std::vector<std::vector<int32_t>> rot_out(levelBudget);
    for (uint32_t i = 0; i < uint32_t(levelBudget); i++) {
        rot_out[i] = std::vector<int32_t>(b + bRem);
    }

    for (int32_t s = 0; s < levelBudget - flagRem; s++) {
        for (int32_t j = 0; j < g; j++) {
            rot_in[s][j] =
                ReduceRotation((j - int32_t((numRotations + 1) / 2) + 1) * (1 << (s * layersCollapse)), M / 4);
        }

        for (int32_t i = 0; i < b; i++) {
            rot_out[s][i] = ReduceRotation((g * i) * (1 << (s * layersCollapse)), M / 4);
        }
    }

    if (flagRem) {
        int32_t s = levelBudget - flagRem;
        for (int32_t j = 0; j < gRem; j++) {
            rot_in[s][j] =
                ReduceRotation((j - int32_t((numRotationsRem + 1) / 2) + 1) * (1 << (s * layersCollapse)), M / 4);
        }

        for (int32_t i = 0; i < bRem; i++) {
            rot_out[s][i] = ReduceRotation((gRem * i) * (1 << (s * layersCollapse)), M / 4);
        }
    }

    //  No need for Encrypted Bit Reverse
    // Ciphertext<DCRTPoly> result = ctxt->Clone();
    ckks::CtAccurate result_gpu(ctxt_gpu);

    // hoisted automorphisms
    for (int32_t s = 0; s < levelBudget - flagRem; s++) {
        if (s != 0) {
            // algo->ModReduceInternalInPlace(result, BASE_NUM_LEVELS_TO_DROP);
            result_gpu = gpu_context.Rescale(result_gpu);
        }
        // computes the NTTs for each CRT limb (for the hoisted automorphisms used later on)
        // auto digits = cc->EvalFastRotationPrecompute(result);
        auto digits_gpu = gpu_context.ModUp(result_gpu.ax__);

        // std::vector<Ciphertext<DCRTPoly>> fastRotation(g);
        std::vector<ckks::CtAccurate> fastRotation_gpu(g);
// #pragma omp parallel for
        for (int32_t j = 0; j < g; j++) {
            if (rot_in[s][j] != 0) {
                // fastRotation[j] = cc->EvalFastRotationExt(result, rot_in[s][j], digits, true);
                fastRotation_gpu[j] = EvalFastRotateExtGPU(result_gpu, digits_gpu, rot_in[s][j], evalKeys, cc, gpu_context, true);
            }
            else {
                // fastRotation[j] = cc->KeySwitchExt(result, true);

                const uint32_t totalLimbs = result_gpu.ax__.size()/gpu_context.degree__ + gpu_context.param__.num_special_moduli_;
                fastRotation_gpu[j].level = result_gpu.level;
                fastRotation_gpu[j].noiseScaleDeg = result_gpu.noiseScaleDeg;
                fastRotation_gpu[j].scalingFactor = result_gpu.scalingFactor;
                fastRotation_gpu[j].bx__ .resize(gpu_context.degree__ * totalLimbs);
                fastRotation_gpu[j].bx__.setConstant(0);
                gpu_context.AddScaledMessageTerm(fastRotation_gpu[j].bx__ , result_gpu.bx__);
                fastRotation_gpu[j].ax__ .resize(gpu_context.degree__ * totalLimbs);
                fastRotation_gpu[j].ax__.setConstant(0);
                gpu_context.AddScaledMessageTerm(fastRotation_gpu[j].ax__ , result_gpu.ax__);
            }
        }

        // Ciphertext<DCRTPoly> outer;
        ckks::CtAccurate outer_gpu;
        // DCRTPoly first;
        ckks::DeviceVector first_gpu;

        for (int32_t i = 0; i < b; i++) {
            // Ciphertext<DCRTPoly> inner;
            ckks::CtAccurate inner_gpu;
            // for the first iteration with j=0:
            int32_t G = g * i;
            // inner     = EvalMultExt(fastRotation[0], A[s][G]);
            // const ckks::PtAccurate A_gpu = LoadAccuratePlaintext(A[s][G], A[s][G]->GetElement<DCRTPoly>());
            inner_gpu = gpu_context.EvalMultPlainExt(fastRotation_gpu[0], A[s][G]);

            // continue the loop
            for (int32_t j = 1; j < g; j++) {
                if ((G + j) != int32_t(numRotations)) {
                    // EvalAddExtInPlace(inner, EvalMultExt(fastRotation[j], A[s][G + j]));
                    // const ckks::PtAccurate A_gpu = LoadAccuratePlaintext(A[s][G + j], A[s][G + j]->GetElement<DCRTPoly>());
                    const ckks::CtAccurate toAdd = gpu_context.EvalMultPlainExt(fastRotation_gpu[j], A[s][G + j]);
                    gpu_context.EvalAddInPlaceExt(inner_gpu, toAdd);
                }
            }

            if (i == 0) {
                // first         = cc->KeySwitchDownFirstElement(inner);
                gpu_context.ModDown(inner_gpu.bx__, first_gpu);

                // auto elements = inner->GetElements();
                // elements[0].SetValuesToZero();
                // inner->SetElements(elements);
                // inner_gpu.bx__ = LoadConstantVector(inner_gpu.bx__.size(), 0);
                inner_gpu.bx__.setConstant(0);

                // outer = inner;
                outer_gpu = ckks::CtAccurate(inner_gpu);
            }
            else {
                if (rot_out[s][i] != 0) {
                    // inner = cc->KeySwitchDown(inner);
                    ckks::DeviceVector temp(inner_gpu.ax__);
                    gpu_context.ModDown(temp, inner_gpu.ax__);
                    temp = ckks::DeviceVector(inner_gpu.bx__);
                    gpu_context.ModDown(temp, inner_gpu.bx__);

                    // Find the automorphism index that corresponds to rotation index index.
                    usint autoIndex = FindAutomorphismIndex2nComplex(rot_out[s][i], M);
                    std::vector<usint> map(N);
                    PrecomputeAutoMap(N, autoIndex, &map);

                    // first += inner->GetElements()[0].AutomorphismTransform(autoIndex, map);
                    ckks::DeviceVector inner_b_rot = gpu_context.AutomorphismTransform(inner_gpu.bx__, map);
                    gpu_context.AddCoreInPlace(first_gpu, inner_b_rot);

                    // auto innerDigits = cc->EvalFastRotationPrecompute(inner);
                    auto innerDigits_gpu = gpu_context.ModUp(inner_gpu.ax__);

                    auto inner_rot_gpu = EvalFastRotateExtGPU(inner_gpu, innerDigits_gpu, rot_out[s][i], evalKeys, cc, gpu_context, false);

                    // EvalAddExtInPlace(outer, cc->EvalFastRotationExt(inner, rot_out[s][i], innerDigits, false));
                    gpu_context.EvalAddInPlaceExt(outer_gpu, inner_rot_gpu);

                }
                else {
                    // first += cc->KeySwitchDownFirstElement(inner);
                    ckks::DeviceVector inner_mod_down; gpu_context.ModDown(inner_gpu.bx__, inner_mod_down);
                    gpu_context.AddCoreInPlace(first_gpu, inner_mod_down);

                    // auto elements = inner->GetElements();
                    // elements[0].SetValuesToZero();
                    // inner->SetElements(elements);
                    // inner_gpu.bx__ = LoadConstantVector(inner_gpu.bx__.size(), 0);
                    inner_gpu.bx__.setConstant(0);

                    // EvalAddExtInPlace(outer, inner);
                    gpu_context.EvalAddInPlaceExt(outer_gpu, inner_gpu);
                }
            }
        }

        // result                          = cc->KeySwitchDown(outer);
        gpu_context.ModDown(outer_gpu.ax__, result_gpu.ax__);
        gpu_context.ModDown(outer_gpu.bx__, result_gpu.bx__);
        result_gpu.level = outer_gpu.level;
        result_gpu.noiseScaleDeg = outer_gpu.noiseScaleDeg;
        result_gpu.scalingFactor = outer_gpu.scalingFactor;

        // std::vector<DCRTPoly>& elements = result->GetElements();
        // elements[0] += first;
        gpu_context.AddCoreInPlace(result_gpu.bx__, first_gpu);
    }

    if (flagRem) {
        // algo->ModReduceInternalInPlace(result, BASE_NUM_LEVELS_TO_DROP);
        result_gpu = gpu_context.Rescale(result_gpu);

        // computes the NTTs for each CRT limb (for the hoisted automorphisms used later on)
        // auto digits = cc->EvalFastRotationPrecompute(result);
        auto digits_gpu = gpu_context.ModUp(result_gpu.ax__);

        // std::vector<Ciphertext<DCRTPoly>> fastRotation(gRem);
        std::vector<ckks::CtAccurate> fastRotation_gpu(gRem);

        int32_t s = levelBudget - flagRem;
// #pragma omp parallel for
        for (int32_t j = 0; j < gRem; j++) {
            if (rot_in[s][j] != 0) {
                // fastRotation[j] = cc->EvalFastRotationExt(result, rot_in[s][j], digits, true);
                fastRotation_gpu[j] = EvalFastRotateExtGPU(result_gpu, digits_gpu, rot_in[s][j], evalKeys, cc, gpu_context, true);
            }
            else {
                // fastRotation[j] = cc->KeySwitchExt(result, true);

                const uint32_t totalLimbs = result_gpu.ax__.size()/gpu_context.degree__ + gpu_context.param__.num_special_moduli_;
                fastRotation_gpu[j].level = result_gpu.level;
                fastRotation_gpu[j].noiseScaleDeg = result_gpu.noiseScaleDeg;
                fastRotation_gpu[j].scalingFactor = result_gpu.scalingFactor;
                // fastRotation_gpu[j].bx__ = LoadConstantVector(gpu_context.degree__ * totalLimbs, 0);
                fastRotation_gpu[j].bx__.resize(gpu_context.degree__ * totalLimbs);
                fastRotation_gpu[j].bx__.setConstant(0);
                // cudaMemset(fastRotation_gpu[j].bx__.data(), 0, gpu_context.degree__ * totalLimbs * sizeof(ckks::word64));
                gpu_context.AddScaledMessageTerm(fastRotation_gpu[j].bx__ , result_gpu.bx__);
                // fastRotation_gpu[j].ax__ = LoadConstantVector(gpu_context.degree__ * totalLimbs, 0);
                fastRotation_gpu[j].ax__.resize(gpu_context.degree__ * totalLimbs);
                fastRotation_gpu[j].ax__.setConstant(0);
                gpu_context.AddScaledMessageTerm(fastRotation_gpu[j].ax__ , result_gpu.ax__);
            }
        }

        // Ciphertext<DCRTPoly> outer;
        ckks::CtAccurate outer_gpu;
        // DCRTPoly first;
        ckks::DeviceVector first_gpu;
        for (int32_t i = 0; i < bRem; i++) {
            // Ciphertext<DCRTPoly> inner;
            ckks::CtAccurate inner_gpu;
            // for the first iteration with j=0:
            int32_t GRem = gRem * i;
            // inner        = EvalMultExt(fastRotation[0], A[s][GRem]);
            // const ckks::PtAccurate A_gpu = LoadAccuratePlaintext(A[s][GRem], A[s][GRem]->GetElement<DCRTPoly>());
            inner_gpu = gpu_context.EvalMultPlainExt(fastRotation_gpu[0], A[s][GRem]);
            // continue the loop
            for (int32_t j = 1; j < gRem; j++) {
                if ((GRem + j) != int32_t(numRotationsRem)) {
                    // EvalAddExtInPlace(inner, EvalMultExt(fastRotation[j], A[s][GRem + j]));
                    // const ckks::PtAccurate A_gpu = LoadAccuratePlaintext(A[s][GRem + j], A[s][GRem + j]->GetElement<DCRTPoly>());
                    const ckks::CtAccurate toAdd = gpu_context.EvalMultPlainExt(fastRotation_gpu[j], A[s][GRem + j]);
                    gpu_context.EvalAddInPlaceExt(inner_gpu, toAdd);
                }
            }

            if (i == 0) {
                // first         = cc->KeySwitchDownFirstElement(inner);
                gpu_context.ModDown(inner_gpu.bx__, first_gpu);

                // auto elements = inner->GetElements();
                // elements[0].SetValuesToZero();
                // inner->SetElements(elements);
                // outer = inner;

                // inner_gpu.bx__ = LoadConstantVector(inner_gpu.bx__.size(), 0);
                inner_gpu.bx__.setConstant(0);
                outer_gpu = ckks::CtAccurate(inner_gpu);
            } else {
                if (rot_out[s][i] != 0) {
                    // inner = cc->KeySwitchDown(inner);
                    ckks::DeviceVector temp(inner_gpu.ax__);
                    gpu_context.ModDown(temp, inner_gpu.ax__);
                    temp = ckks::DeviceVector(inner_gpu.bx__);
                    gpu_context.ModDown(temp, inner_gpu.bx__);

                    // Find the automorphism index that corresponds to rotation index index.
                    usint autoIndex = FindAutomorphismIndex2nComplex(rot_out[s][i], M);
                    std::vector<usint> map(N);
                    PrecomputeAutoMap(N, autoIndex, &map);

                    // first += inner->GetElements()[0].AutomorphismTransform(autoIndex, map);
                    ckks::DeviceVector inner_b_rot = gpu_context.AutomorphismTransform(inner_gpu.bx__, map);
                    gpu_context.AddCoreInPlace(first_gpu, inner_b_rot);

                    // auto innerDigits = cc->EvalFastRotationPrecompute(inner);
                    auto innerDigits_gpu = gpu_context.ModUp(inner_gpu.ax__);
                    // EvalAddExtInPlace(outer, cc->EvalFastRotationExt(inner, rot_out[s][i], innerDigits, false));
                    auto inner_rot_gpu = EvalFastRotateExtGPU(inner_gpu, innerDigits_gpu, rot_out[s][i], evalKeys, cc, gpu_context, false);
                    gpu_context.EvalAddInPlaceExt(outer_gpu, inner_rot_gpu);
                }
                else {
                    // first += cc->KeySwitchDownFirstElement(inner);
                    ckks::DeviceVector inner_mod_down; gpu_context.ModDown(inner_gpu.bx__, inner_mod_down);
                    gpu_context.AddCoreInPlace(first_gpu, inner_mod_down);

                    // auto elements = inner->GetElements();
                    // elements[0].SetValuesToZero();
                    // inner->SetElements(elements);
                    // inner_gpu.bx__ = LoadConstantVector(inner_gpu.bx__.size(), 0);
                    inner_gpu.bx__.setConstant(0);

                    // EvalAddExtInPlace(outer, inner);
                    gpu_context.EvalAddInPlaceExt(outer_gpu, inner_gpu);
                }
            }
        }

        // result                          = cc->KeySwitchDown(outer);
        gpu_context.ModDown(outer_gpu.ax__, result_gpu.ax__);
        gpu_context.ModDown(outer_gpu.bx__, result_gpu.bx__);
        result_gpu.level = outer_gpu.level;
        result_gpu.noiseScaleDeg = outer_gpu.noiseScaleDeg;
        result_gpu.scalingFactor = outer_gpu.scalingFactor;
        // std::vector<DCRTPoly>& elements = result->GetElements();
        // elements[0] += first;
        gpu_context.AddCoreInPlace(result_gpu.bx__, first_gpu);
    }

    // {
    //     const auto should_be_result = LoadAccurateCiphertext(result);
    //     assert(result_gpu == should_be_result);
    // }


    return result_gpu;

    // Ciphertext<DCRTPoly> result = ctxt->Clone();
    // LoadCtAccurateFromGPU(result, result_gpu, elementParams);

    // return result;
}

ckks::CtAccurate FHECKKSRNS::ConjugateGPU(const ckks::CtAccurate& ciphertext, const ckks::EvaluationKey& evk, const ckks::Context& gpu_context) const {
    const usint N = gpu_context.degree__;
    std::vector<usint> vec(N);
    PrecomputeAutoMap(N, 2 * N - 1, &vec);

    // ckks::CtAccurate result = gpu_context.KeySwitch()

    ckks::DeviceVector raisedDigits = gpu_context.ModUp(ciphertext.ax__);

    // if (verbose) std::cout << "\tFinished modup\n";

    ckks::DeviceVector ks_a, ks_b;

    gpu_context.KeySwitch(raisedDigits, evk, ks_a, ks_b);

    // if (verbose) std::cout << "\tFinished keyswitch\n";

    // ckks::Ciphertext ks_output;
    ckks::CtAccurate toLoad; 
    gpu_context.ModDown(ks_a, toLoad.ax__);
    gpu_context.ModDown(ks_b, toLoad.bx__);
    // ks_output.level = orig_elems.level;
    // ks_output.noiseScaleDeg = orig_elems.noiseScaleDeg;
    // ks_output.scalingFactor = orig_elems.scalingFactor;

    // if (verbose) std::cout << "\tFinished mod down\n";

    // Add(ks_output, orig_elems, toLoad);
    // gpu_context.AddCore(ks_output.ax__, ks_output.bx__, orig_elems.ax__, orig_elems.bx__, toLoad.ax__, toLoad.bx__);
    gpu_context.AddCoreInPlace(toLoad.bx__, ciphertext.bx__);

    toLoad.level = ciphertext.level;
    toLoad.noiseScaleDeg = ciphertext.noiseScaleDeg;
    toLoad.scalingFactor = ciphertext.scalingFactor;

    gpu_context.AutomorphismTransformInPlace(toLoad, vec);

    return toLoad;
}


Ciphertext<DCRTPoly> FHECKKSRNS::EvalBootstrapGPU(ConstCiphertext<DCRTPoly> ciphertext, const ckks::Context& gpu_context, uint32_t numIterations, uint32_t precision) const {
    const auto cryptoParams = std::dynamic_pointer_cast<CryptoParametersCKKSRNS>(ciphertext->GetCryptoParameters());

    if (cryptoParams->GetKeySwitchTechnique() != HYBRID)
        OPENFHE_THROW(config_error, "CKKS Bootstrapping is only supported for the Hybrid key switching method.");
#if NATIVEINT == 128 && !defined(__EMSCRIPTEN__)
    if (cryptoParams->GetScalingTechnique() == FLEXIBLEAUTO || cryptoParams->GetScalingTechnique() == FLEXIBLEAUTOEXT)
        OPENFHE_THROW(config_error,
                      "128-bit CKKS Bootstrapping is supported for FIXEDMANUAL and FIXEDAUTO methods only.");
#endif
    if (numIterations != 1 && numIterations != 2) {
        OPENFHE_THROW(config_error, "CKKS Iterative Bootstrapping is only supported for 1 or 2 iterations.");
    }

#ifdef BOOTSTRAPTIMING
    TimeVar t;
    double timeEncode(0.0);
    double timeModReduce(0.0);
    double timeDecode(0.0);

    // const uint32_t num_iters = 5;
#endif

    auto cc        = ciphertext->GetCryptoContext();
    uint32_t M     = cc->GetCyclotomicOrder();
    uint32_t L0    = cryptoParams->GetElementParams()->GetParams().size();
    auto initSizeQ = ciphertext->GetElements()[0].GetNumOfElements();

    std::cout << "Beginning CKKS bootstrapping\n";
    std::cout << "numIterations: " << numIterations << std::endl;

    const auto elemParamsPointer = cc->GetElementParams();

    if (numIterations > 1) {
        // Step 1: Get the input.
        uint32_t powerOfTwoModulus = 1 << precision;

        // Step 2: Scale up by powerOfTwoModulus, and extend the modulus to powerOfTwoModulus * q.
        // Note that we extend the modulus implicitly without any code calls because the value always stays 0.
        Ciphertext<DCRTPoly> ctScaledUp = ciphertext->Clone();
        // We multiply by powerOfTwoModulus, and leave the last CRT value to be 0 (mod powerOfTwoModulus).
        cc->GetScheme()->MultByIntegerInPlace(ctScaledUp, powerOfTwoModulus);
        ctScaledUp->SetLevel(L0 - ctScaledUp->GetElements()[0].GetNumOfElements());

        // Step 3: Bootstrap the initial ciphertext.
        auto ctInitialBootstrap = cc->EvalBootstrap(ciphertext, numIterations - 1, precision);
        cc->GetScheme()->ModReduceInternalInPlace(ctInitialBootstrap, BASE_NUM_LEVELS_TO_DROP);

        // Step 4: Scale up by powerOfTwoModulus.
        cc->GetScheme()->MultByIntegerInPlace(ctInitialBootstrap, powerOfTwoModulus);

        // Step 5: Mod-down to powerOfTwoModulus * q
        // We mod down, and leave the last CRT value to be 0 because it's divisible by powerOfTwoModulus.
        auto ctBootstrappedScaledDown = ctInitialBootstrap->Clone();
        auto bootstrappingSizeQ       = ctBootstrappedScaledDown->GetElements()[0].GetNumOfElements();

        // If we start with more towers, than we obtain from bootstrapping, return the original ciphertext.
        if (bootstrappingSizeQ <= initSizeQ) {
            return ciphertext->Clone();
        }
        for (auto& cv : ctBootstrappedScaledDown->GetElements()) {
            cv.DropLastElements(bootstrappingSizeQ - initSizeQ);
        }
        ctBootstrappedScaledDown->SetLevel(L0 - ctBootstrappedScaledDown->GetElements()[0].GetNumOfElements());

        // Step 6 and 7: Calculate the bootstrapping error by subtracting the original ciphertext from the bootstrapped ciphertext. Mod down to q is done implicitly.
        auto ctBootstrappingError = cc->EvalSub(ctBootstrappedScaledDown, ctScaledUp);

        // Step 8: Bootstrap the error.
        auto ctBootstrappedError = cc->EvalBootstrap(ctBootstrappingError, 1, 0);
        cc->GetScheme()->ModReduceInternalInPlace(ctBootstrappedError, BASE_NUM_LEVELS_TO_DROP);

        // Step 9: Subtract the bootstrapped error from the initial bootstrap to get even lower error.
        auto finalCiphertext = cc->EvalSub(ctInitialBootstrap, ctBootstrappedError);

        // Step 10: Scale back down by powerOfTwoModulus to get the original message.
        cc->EvalMultInPlace(finalCiphertext, static_cast<double>(1) / powerOfTwoModulus);
        return finalCiphertext;
    }

    uint32_t slots = ciphertext->GetSlots();

    auto pair = m_bootPrecomMap.find(slots);
    if (pair == m_bootPrecomMap.end()) {
        std::string errorMsg(std::string("Precomputations for ") + std::to_string(slots) +
                             std::string(" slots were not generated") +
                             std::string(" Need to call EvalBootstrapSetup and then EvalBootstrapKeyGen to proceed"));
        OPENFHE_THROW(type_error, errorMsg);
    }
    const std::shared_ptr<CKKSBootstrapPrecom> precom = pair->second;
    size_t N                                          = cc->GetRingDimension();

    auto elementParamsRaised = *(cryptoParams->GetElementParams());

    // For FLEXIBLEAUTOEXT we raised ciphertext does not include extra modulus
    // as it is multiplied by auxiliary plaintext
    if (cryptoParams->GetScalingTechnique() == FLEXIBLEAUTOEXT) {
        elementParamsRaised.PopLastParam();
    }

    auto paramsQ = elementParamsRaised.GetParams();
    usint sizeQ  = paramsQ.size();

    std::vector<NativeInteger> moduli(sizeQ);
    std::vector<NativeInteger> roots(sizeQ);
    for (size_t i = 0; i < sizeQ; i++) {
        moduli[i] = paramsQ[i]->GetModulus();
        roots[i]  = paramsQ[i]->GetRootOfUnity();
    }
    auto elementParamsRaisedPtr = std::make_shared<ILDCRTParams<DCRTPoly::Integer>>(M, moduli, roots);

    NativeInteger q = elementParamsRaisedPtr->GetParams()[0]->GetModulus().ConvertToInt();
    double qDouble  = q.ConvertToDouble();

    const auto p = cryptoParams->GetPlaintextModulus();
    double powP  = pow(2, p);

    int32_t deg = std::round(std::log2(qDouble / powP));
#if NATIVEINT != 128
    if (deg > static_cast<int32_t>(m_correctionFactor)) {
        OPENFHE_THROW(math_error, "Degree [" + std::to_string(deg) +
                                      "] must be less than or equal to the correction factor [" +
                                      std::to_string(m_correctionFactor) + "].");
    }
#endif
    uint32_t correction = m_correctionFactor - deg;
    double post         = std::pow(2, static_cast<double>(deg));

    double pre      = 1. / post;
    uint64_t scalar = std::llround(post);

    //------------------------------------------------------------------------------
    // GPU Preprocessing
    //------------------------------------------------------------------------------

    std::vector<std::vector<ckks::PtAccurate>> A_gpu(precom->m_U0hatTPreFFT.size());
    for (uint32_t i = 0; i < A_gpu.size(); i++) {
        A_gpu[i] = std::vector<ckks::PtAccurate>(precom->m_U0hatTPreFFT[i].size());
        for (uint32_t j = 0; j < A_gpu[i].size(); j++) {
            A_gpu[i][j] = LoadAccuratePlaintext(precom->m_U0hatTPreFFT[i][j], precom->m_U0hatTPreFFT[i][j]->GetElement<DCRTPoly>());
        }
    }

    std::vector<std::vector<ckks::PtAccurate>> A_gpu_stoc(precom->m_U0PreFFT.size());
    for (uint32_t i = 0; i < A_gpu_stoc.size(); i++) {
        A_gpu_stoc[i] = std::vector<ckks::PtAccurate>(precom->m_U0PreFFT[i].size());
        for (uint32_t j = 0; j < A_gpu_stoc[i].size(); j++) {
            A_gpu_stoc[i][j] = LoadAccuratePlaintext(precom->m_U0PreFFT[i][j], precom->m_U0PreFFT[i][j]->GetElement<DCRTPoly>());
        }
    }

    const auto& loaded_rot_keys = gpu_context.preloaded_rotation_key_map;

    std::cout << "num giant steps: " << precom->m_paramsEnc[CKKS_BOOT_PARAMS::GIANT_STEP] << std::endl;
    std::cout << "num baby steps: " << precom->m_paramsEnc[CKKS_BOOT_PARAMS::BABY_STEP] << std::endl;
    std::cout << "level budget: " << precom->m_paramsEnc[CKKS_BOOT_PARAMS::LEVEL_BUDGET] << std::endl;

    //------------------------------------------------------------------------------
    // RAISING THE MODULUS
    //------------------------------------------------------------------------------

    // In FLEXIBLEAUTO, raising the ciphertext to a larger number
    // of towers is a bit more complex, because we need to adjust
    // it's scaling factor to the one that corresponds to the level
    // it's being raised to.
    // Increasing the modulus

    Ciphertext<DCRTPoly> raised = ciphertext->Clone();
    auto algo                   = cc->GetScheme();
    algo->ModReduceInternalInPlace(raised, raised->GetNoiseScaleDeg() - 1);

    AdjustCiphertext(raised, correction);
    auto ctxtDCRT = raised->GetElements();

    // We only use the level 0 ciphertext here. All other towers are automatically ignored to make
    // CKKS bootstrapping faster.
    for (size_t i = 0; i < ctxtDCRT.size(); i++) {
        DCRTPoly temp(elementParamsRaisedPtr, COEFFICIENT);
        ctxtDCRT[i].SetFormat(COEFFICIENT);
        temp = ctxtDCRT[i].GetElementAtIndex(0);
        temp.SetFormat(EVALUATION);
        ctxtDCRT[i] = temp;
    }

    raised->SetElements(ctxtDCRT);
    raised->SetLevel(L0 - ctxtDCRT[0].GetNumOfElements());

#ifdef BOOTSTRAPTIMING
    std::cerr << "\nNumber of levels at the beginning of bootstrapping: "
              << raised->GetElements()[0].GetNumOfElements() - 1 << std::endl;
#endif

    //------------------------------------------------------------------------------
    // SETTING PARAMETERS FOR APPROXIMATE MODULAR REDUCTION
    //------------------------------------------------------------------------------

    // Coefficients of the Chebyshev series interpolating 1/(2 Pi) Sin(2 Pi K x)
    std::vector<double> coefficients;
    double k = 0;

    if (cryptoParams->GetSecretKeyDist() == SPARSE_TERNARY) {
        coefficients = g_coefficientsSparse;
        // k = K_SPARSE;
        k = 1.0;  // do not divide by k as we already did it during precomputation
    }
    else {
        coefficients = g_coefficientsUniform;
        k            = K_UNIFORM;
    }

    double constantEvalMult = pre * (1.0 / (k * N));

    cc->EvalMultInPlace(raised, constantEvalMult);

    // no linear transformations are needed for Chebyshev series as the range has been normalized to [-1,1]
    double coeffLowerBound = -1;
    double coeffUpperBound = 1;

    Ciphertext<DCRTPoly> ctxtDec;
    ckks::CtAccurate ctxtDec_gpu; 

    // bool isLTBootstrap = (precom->m_paramsEnc[CKKS_BOOT_PARAMS::LEVEL_BUDGET] == 1) &&
    //                      (precom->m_paramsDec[CKKS_BOOT_PARAMS::LEVEL_BUDGET] == 1);
    // assert(!isLTBootstrap);

    if (slots == M / 4) {
        //------------------------------------------------------------------------------
        // FULLY PACKED CASE
        //------------------------------------------------------------------------------

        std::cout << "Running fully packed case\n";

        // ckks::CtAccurate ctxtEnc_gpu, ctxtEncI_gpu, empty;

        ckks::CtAccurate raised_gpu = LoadAccurateCiphertext(raised);

#ifdef BOOTSTRAPTIMING
        TIC(t);
        // for (uint32_t iter = 0; iter < num_iters; iter++) {
            // ctxtEnc_gpu = ckks::CtAccurate(empty);
            // ctxtEncI_gpu = ckks::CtAccurate(empty);
#endif

        //------------------------------------------------------------------------------
        // Running CoeffToSlot
        //------------------------------------------------------------------------------

        // need to call internal modular reduction so it also works for FLEXIBLEAUTO
        // algo->ModReduceInternalInPlace(raised, BASE_NUM_LEVELS_TO_DROP);
        raised_gpu = gpu_context.Rescale(raised_gpu);

        // only one linear transform is needed as the other one can be derived
        // auto ctxtEnc_gpu = (isLTBootstrap) ? EvalLinearTransform(precom->m_U0hatTPre, raised) :
            // auto ctxtEnc = EvalCoeffsToSlots(precom->m_U0hatTPreFFT, raised);
                                        //  EvalCoeffsToSlotsGPU(precom->m_U0hatTPreFFT, raised, gpu_context);
                                        // EvalCoeffsToSlotsGPU(A_gpu, raised, gpu_context);

        auto ctxtEnc_gpu = EvalCoeffsToSlotsGPU(A_gpu, raised, raised_gpu, gpu_context);

        // Ciphertext<DCRTPoly> ctxtEnc;
        // LoadCtAccurateFromGPU(ctxtEnc, ctxtEnc_gpu, elemParamsPointer);

        // auto evalKeyMap = cc->GetEvalAutomorphismKeyMap(ctxtEnc->GetKeyTag());
        // auto conj       = Conjugate(ctxtEnc, evalKeyMap);
        // auto ctxtEncI   = cc->EvalSub(ctxtEnc, conj);
        // cc->EvalAddInPlace(ctxtEnc, conj);

        const usint conj_ind = 2 * N - 1;
        const auto conjKeyPair = loaded_rot_keys->find(conj_ind);
        ckks::CtAccurate conj_gpu;
        if (conjKeyPair == loaded_rot_keys->end()) {
            // load key
            auto evalKeyMap = cc->GetEvalAutomorphismKeyMap(ciphertext->GetKeyTag());
            const auto cpu_conj_key = evalKeyMap.find(conj_ind);
            assert(cpu_conj_key != evalKeyMap.end());
            const ckks::EvaluationKey conj_key = LoadRelinKey(cpu_conj_key->second);

            conj_gpu = ConjugateGPU(ctxtEnc_gpu, conj_key, gpu_context);
        } else {
            // use preloaded key
            conj_gpu = ConjugateGPU(ctxtEnc_gpu, conjKeyPair->second, gpu_context);
            // gpu_context.EvalAddInPlace(ctxtEnc_gpu, conj_gpu);
        }
        ckks::CtAccurate ctxtEncI_gpu = gpu_context.Sub(ctxtEnc_gpu, conj_gpu);
        gpu_context.EvalAddInPlace(ctxtEnc_gpu, conj_gpu);

        // algo->MultByMonomialInPlace(ctxtEncI, 3 * M / 4);
        gpu_context.MultByMonomialInPlace(ctxtEncI_gpu, 3 * M / 4);

        if (cryptoParams->GetScalingTechnique() == FIXEDMANUAL) {
            // while (ctxtEnc->GetNoiseScaleDeg() > 1) {
            //     cc->ModReduceInPlace(ctxtEnc);
            //     cc->ModReduceInPlace(ctxtEncI);
            // }
            throw std::logic_error("not implemented for GPU");
        }
        else {
            // if (ctxtEnc->GetNoiseScaleDeg() == 2) {
            //     algo->ModReduceInternalInPlace(ctxtEnc, BASE_NUM_LEVELS_TO_DROP);
            //     algo->ModReduceInternalInPlace(ctxtEncI, BASE_NUM_LEVELS_TO_DROP);
            // }
            if (ctxtEnc_gpu.noiseScaleDeg == 2) {
                ctxtEnc_gpu = gpu_context.Rescale(ctxtEnc_gpu);
                ctxtEncI_gpu = gpu_context.Rescale(ctxtEncI_gpu);
            }
        }

#ifdef BOOTSTRAPTIMING
        // }
        timeEncode = TOC(t);

        // std::cerr << "\nEncoding time: " << timeEncode / 1000.0 / num_iters << " s" << std::endl;
        std::cerr << "\nEncoding time: " << timeEncode / 1000.0 << " s" << std::endl;

        // Running Approximate Mod Reduction

        TIC(t);
        // ckks::CtAccurate c1_tmp(ctxtEnc_gpu), c2_tmp(ctxtEncI_gpu);
        // for (uint32_t iter = 0; iter < num_iters; iter++) {
        //     ctxtEnc_gpu = ckks::CtAccurate(c1_tmp); 
        //     ctxtEncI_gpu = ckks::CtAccurate(c2_tmp);
#endif

        //------------------------------------------------------------------------------
        // Running Approximate Mod Reduction
        //------------------------------------------------------------------------------

        // Evaluate Chebyshev series for the sine wave
        // ctxtEnc  = cc->EvalChebyshevSeries(ctxtEnc, coefficients, coeffLowerBound, coeffUpperBound);
        // ctxtEncI = cc->EvalChebyshevSeries(ctxtEncI, coefficients, coeffLowerBound, coeffUpperBound);
       
        ctxtEnc_gpu = cc->EvalChebyshevSeriesGPU(ctxtEnc_gpu, coefficients, coeffLowerBound, coeffUpperBound, gpu_context, cc);
        ctxtEncI_gpu = cc->EvalChebyshevSeriesGPU(ctxtEncI_gpu, coefficients, coeffLowerBound, coeffUpperBound, gpu_context, cc);

        // auto cheby_caller = std::dynamic_pointer_cast<AdvancedSHECKKSRNS>(cc->GetScheme()->m_AdvancedSHE);
        // ctxtEnc_gpu = cheby_caller->EvalChebyshevSeriesPSGPU(ctxtEnc_gpu, ctxtEnc, coefficients, coeffLowerBound, coeffUpperBound, gpu_context, cc);
        // ctxtEncI_gpu = cheby_caller->EvalChebyshevSeriesPSGPU(ctxtEncI_gpu, ctxtEncI, coefficients, coeffLowerBound, coeffUpperBound, gpu_context, cc);

        // Double-angle iterations
        if ((cryptoParams->GetSecretKeyDist() == UNIFORM_TERNARY) ||
            (cryptoParams->GetSecretKeyDist() == SPARSE_TERNARY)) {
            if (cryptoParams->GetScalingTechnique() != FIXEDMANUAL) {
                // algo->ModReduceInternalInPlace(ctxtEnc, BASE_NUM_LEVELS_TO_DROP);
                // algo->ModReduceInternalInPlace(ctxtEncI, BASE_NUM_LEVELS_TO_DROP);
                ctxtEnc_gpu = gpu_context.Rescale(ctxtEnc_gpu);
                ctxtEncI_gpu = gpu_context.Rescale(ctxtEncI_gpu);
            }
            uint32_t numIter;
            if (cryptoParams->GetSecretKeyDist() == UNIFORM_TERNARY)
                numIter = R_UNIFORM;
            else
                numIter = R_SPARSE;
            // ApplyDoubleAngleIterations(ctxtEnc, numIter);
            // ApplyDoubleAngleIterations(ctxtEncI, numIter);
            ApplyDoubleAngleIterationsGPU(ctxtEnc_gpu, numIter, gpu_context, cc);
            ApplyDoubleAngleIterationsGPU(ctxtEncI_gpu, numIter, gpu_context, cc);
        }

        // algo->MultByMonomialInPlace(ctxtEncI, M / 4);
        gpu_context.MultByMonomialInPlace(ctxtEncI_gpu, M / 4);
        // cc->EvalAddInPlace(ctxtEnc, ctxtEncI);
        gpu_context.EvalAddInPlace(ctxtEnc_gpu, ctxtEncI_gpu);

        // scale the message back up after Chebyshev interpolation
        // algo->MultByIntegerInPlace(ctxtEnc, scalar);
        gpu_context.EvalMultIntegerInPlace(ctxtEnc_gpu, scalar);

#ifdef BOOTSTRAPTIMING
        // }
        timeModReduce = TOC(t);

        // std::cerr << "Approximate modular reduction time: " << timeModReduce / 1000.0 / num_iters << " s" << std::endl;
        std::cerr << "Approximate modular reduction time: " << timeModReduce / 1000.0 << " s" << std::endl;

        // Running SlotToCoeff

        TIC(t);
        // ckks::CtAccurate c_tmp(ctxtEnc_gpu);
        // for (uint32_t iter = 0; iter < num_iters; iter++) {
        //     ctxtEnc_gpu = ckks::CtAccurate(c_tmp);
#endif

        // Ciphertext<DCRTPoly> ctxtEnc = ciphertext->CloneZero();
        // LoadCtAccurateFromGPU(ctxtEnc, ctxtEnc_gpu, elemParamsPointer);

        //------------------------------------------------------------------------------
        // Running SlotToCoeff
        //------------------------------------------------------------------------------

        // In the case of FLEXIBLEAUTO, we need one extra tower
        // TODO: See if we can remove the extra level in FLEXIBLEAUTO
        if (cryptoParams->GetScalingTechnique() != FIXEDMANUAL) {
            // algo->ModReduceInternalInPlace(ctxtEnc, BASE_NUM_LEVELS_TO_DROP);
            ctxtEnc_gpu = gpu_context.Rescale(ctxtEnc_gpu);
        }

        // Only one linear transform is needed
        // ctxtDec = (isLTBootstrap) ? EvalLinearTransform(precom->m_U0Pre, ctxtEnc) :
                                    // EvalSlotsToCoeffs(precom->m_U0PreFFT, ctxtEnc);
                                    // EvalSlotsToCoeffsGPU(A_gpu_stoc, ctxtEnc, gpu_context);
        ctxtDec_gpu = EvalSlotsToCoeffsGPU(A_gpu_stoc, ctxtEnc_gpu, ciphertext->GetSlots(), ciphertext->GetKeyTag(), gpu_context, cc);

#ifdef BOOTSTRAPTIMING
        // }
#endif
    }
    else {
        //------------------------------------------------------------------------------
        // SPARSELY PACKED CASE
        //------------------------------------------------------------------------------

        std::cout << "Running partially packed case\n";

        //------------------------------------------------------------------------------
        // Running PartialSum
        //------------------------------------------------------------------------------

        for (uint32_t j = 1; j < N / (2 * slots); j <<= 1) {
            auto temp = cc->EvalRotate(raised, j * slots);
            cc->EvalAddInPlace(raised, temp);
        }

        ckks::CtAccurate raised_gpu = LoadAccurateCiphertext(raised);

#ifdef BOOTSTRAPTIMING
        TIC(t);
#endif

        //------------------------------------------------------------------------------
        // Running CoeffsToSlots
        //------------------------------------------------------------------------------

        // algo->ModReduceInternalInPlace(raised, BASE_NUM_LEVELS_TO_DROP);  // TODO: comment out once tests pass
        raised_gpu = gpu_context.Rescale(raised_gpu);

        // assert(!isLTBootstrap);
        // auto ctxtEnc = (isLTBootstrap) ? EvalLinearTransform(precom->m_U0hatTPre, raised) :
        // auto ctxtEnc = EvalCoeffsToSlots(precom->m_U0hatTPreFFT, raised);    // TODO: comment out once tests pass
        //                                 //  EvalCoeffsToSlotsGPU(precom->m_U0hatTPreFFT, raised, gpu_context);
        //                                  EvalCoeffsToSlotsGPU(A_gpu, raised, gpu_context);

        auto ctxtEnc_gpu = EvalCoeffsToSlotsGPU(A_gpu, raised, raised_gpu, gpu_context);

        // {
        //     // Ciphertext<DCRTPoly> ctxtEnc = ctxtEnc_correct->CloneZero();
        //     auto should_be_gpu = LoadAccurateCiphertext(ctxtEnc);
        //     // std::cout << "checking accurate ct...\n";
        //     if (ctxtEnc_gpu != should_be_gpu) {
        //         throw std::logic_error("EvalCoeffsToSlots gpu mismatch\n");
        //     }
        // }

        // {
        //     Ciphertext<DCRTPoly> ctxtEnc_from_gpu = ctxtEnc->CloneZero();
        //     // Ciphertext<DCRTPoly> ctxtEnc_from_gpu;
        //     LoadCtAccurateFromGPU(ctxtEnc_from_gpu, ctxtEnc_gpu, elemParamsPointer);
        //     // std::cout << "checking ct...\n";
        //     if (*ctxtEnc_from_gpu != *ctxtEnc) {
        //         throw std::logic_error("EvalCoeffsToSlots cpu mismatch\n");
        //     }
        // }

        const usint conj_ind = 2 * N - 1;
        const auto conjKeyPair = loaded_rot_keys->find(conj_ind);
        if (conjKeyPair == loaded_rot_keys->end()) {
            // load key
            auto evalKeyMap = cc->GetEvalAutomorphismKeyMap(ciphertext->GetKeyTag());
            const auto cpu_conj_key = evalKeyMap.find(conj_ind);
            assert(cpu_conj_key != evalKeyMap.end());
            const ckks::EvaluationKey conj_key = LoadRelinKey(cpu_conj_key->second);

            auto conj_gpu = ConjugateGPU(ctxtEnc_gpu, conj_key, gpu_context);
            gpu_context.EvalAddInPlace(ctxtEnc_gpu, conj_gpu);
        } else {
            // use preloaded key
            auto conj_gpu = ConjugateGPU(ctxtEnc_gpu, conjKeyPair->second, gpu_context);
            gpu_context.EvalAddInPlace(ctxtEnc_gpu, conj_gpu);
        }
        // TODO: uncomment this block when tests pass
        // auto evalKeyMap = cc->GetEvalAutomorphismKeyMap(ctxtEnc->GetKeyTag());
        // auto conj       = Conjugate(ctxtEnc, evalKeyMap);
        // cc->EvalAddInPlace(ctxtEnc, conj);

        if (cryptoParams->GetScalingTechnique() == FIXEDMANUAL) {
            // while (ctxtEnc->GetNoiseScaleDeg() > 1) {
            //     cc->ModReduceInPlace(ctxtEnc);

            // }
            throw std::logic_error("not implemented for GPU");
        }
        else {
            // if (ctxtEnc->GetNoiseScaleDeg() == 2) {
            //     algo->ModReduceInternalInPlace(ctxtEnc, BASE_NUM_LEVELS_TO_DROP);
            // }
            if (ctxtEnc_gpu.noiseScaleDeg == 2) ctxtEnc_gpu = gpu_context.Rescale(ctxtEnc_gpu);
        }

#ifdef BOOTSTRAPTIMING
        timeEncode = TOC(t);

        std::cerr << "\nEncoding time: " << timeEncode / 1000.0 << " s" << std::endl;

        // Running Approximate Mod Reduction

        TIC(t);
#endif

        //------------------------------------------------------------------------------
        // Running Approximate Mod Reduction
        //------------------------------------------------------------------------------

        // Ciphertext<DCRTPoly> ctxtEnc = ciphertext->CloneZero();
        // LoadCtAccurateFromGPU(ctxtEnc, ctxtEnc_gpu, elemParamsPointer);

        // {   // check chebyshev inputs
        //     // Ciphertext<DCRTPoly> ctxtEnc = ctxtEnc_correct->CloneZero();
        //     auto should_be_gpu = LoadAccurateCiphertext(ctxtEnc);
        //     // std::cout << "checking accurate ct...\n";
        //     if (ctxtEnc_gpu != should_be_gpu) {
        //         throw std::logic_error("EvalChebyshev input gpu mismatch\n");
        //     }
        // }

        // {
        //     Ciphertext<DCRTPoly> ctxtEnc_from_gpu = ctxtEnc->CloneZero();
        //     LoadCtAccurateFromGPU(ctxtEnc_from_gpu, ctxtEnc_gpu, elemParamsPointer);
        //     // std::cout << "checking ct...\n";
        //     if (*ctxtEnc_from_gpu != *ctxtEnc) {
        //         throw std::logic_error("EvalChebyshev input cpu mismatch\n");
        //     }
        // }

        // Evaluate Chebyshev series for the sine wave
        // Essentially all of the time is here...

        // auto cheby_caller = std::dynamic_pointer_cast<AdvancedSHECKKSRNS>(cc->GetScheme()->m_AdvancedSHE);
        // ctxtEnc_gpu = cheby_caller->EvalChebyshevSeriesPSGPU(ctxtEnc_gpu, ctxtEnc, coefficients, coeffLowerBound, coeffUpperBound, gpu_context, cc);
        ctxtEnc_gpu = cc->EvalChebyshevSeriesGPU(ctxtEnc_gpu, coefficients, coeffLowerBound, coeffUpperBound, gpu_context, cc);

        // ctxtEnc = cc->EvalChebyshevSeries(ctxtEnc, coefficients, coeffLowerBound, coeffUpperBound);  // TODO: uncomment once tests pass
        // ctxtEnc = cc->EvalChebyshevSeriesGPU(ctxtEnc, coefficients, coeffLowerBound, coeffUpperBound, gpu_context);
        // ckks::CtAccurate ctxtEnc_gpu = cc->EvalChebyshevSeriesGPU(ctxtEnc, coefficients, coeffLowerBound, coeffUpperBound, gpu_context);
        // LoadCtAccurateFromGPU(ctxtEnc, ctxtEnc_gpu, elemParamsPointer);

        // {   // check chebyshev outputs
        //     // Ciphertext<DCRTPoly> ctxtEnc = ctxtEnc_correct->CloneZero();
        //     auto should_be_gpu = LoadAccurateCiphertext(ctxtEnc);
        //     // std::cout << "checking accurate ct...\n";
        //     if (ctxtEnc_gpu != should_be_gpu) {
        //         throw std::logic_error("EvalChebyshev output gpu mismatch\n");
        //     }
        // }

        // {
        //     Ciphertext<DCRTPoly> ctxtEnc_from_gpu = ctxtEnc->CloneZero();
        //     LoadCtAccurateFromGPU(ctxtEnc_from_gpu, ctxtEnc_gpu, elemParamsPointer);
        //     // std::cout << "checking ct...\n";
        //     if (*ctxtEnc_from_gpu != *ctxtEnc) {
        //         throw std::logic_error("EvalChebyshev output cpu mismatch\n");
        //     }
        // }

        // Double-angle iterations
        if ((cryptoParams->GetSecretKeyDist() == UNIFORM_TERNARY) ||
            (cryptoParams->GetSecretKeyDist() == SPARSE_TERNARY)) {
            if (cryptoParams->GetScalingTechnique() != FIXEDMANUAL) {
                // algo->ModReduceInternalInPlace(ctxtEnc, BASE_NUM_LEVELS_TO_DROP);
                ctxtEnc_gpu = gpu_context.Rescale(ctxtEnc_gpu);
            }
            uint32_t numIter;
            if (cryptoParams->GetSecretKeyDist() == UNIFORM_TERNARY)
                numIter = R_UNIFORM;
            else
                numIter = R_SPARSE;
            // ApplyDoubleAngleIterations(ctxtEnc, numIter);
            // ApplyDoubleAngleIterationsGPU(ctxtEnc, numIter, gpu_context);
            ApplyDoubleAngleIterationsGPU(ctxtEnc_gpu, numIter, gpu_context, cc);
        }

        // scale the message back up after Chebyshev interpolation
        // algo->MultByIntegerInPlace(ctxtEnc, scalar);
        gpu_context.EvalMultIntegerInPlace(ctxtEnc_gpu, scalar);

#ifdef BOOTSTRAPTIMING
        timeModReduce = TOC(t);

        std::cerr << "Approximate modular reduction time: " << timeModReduce / 1000.0 << " s" << std::endl;

        // Running SlotToCoeff

        TIC(t);
#endif

        // Ciphertext<DCRTPoly> ctxtEnc = ciphertext->CloneZero();
        // LoadCtAccurateFromGPU(ctxtEnc, ctxtEnc_gpu, elemParamsPointer);

        //------------------------------------------------------------------------------
        // Running SlotsToCoeffs
        //------------------------------------------------------------------------------

        // In the case of FLEXIBLEAUTO, we need one extra tower
        // TODO: See if we can remove the extra level in FLEXIBLEAUTO
        if (cryptoParams->GetScalingTechnique() != FIXEDMANUAL) {
            // algo->ModReduceInternalInPlace(ctxtEnc, BASE_NUM_LEVELS_TO_DROP);
            ctxtEnc_gpu = gpu_context.Rescale(ctxtEnc_gpu);
        }

        // {   // check StoC inputs
        //     // Ciphertext<DCRTPoly> ctxtEnc = ctxtEnc_correct->CloneZero();
        //     auto should_be_gpu = LoadAccurateCiphertext(ctxtEnc);
        //     // std::cout << "checking accurate ct...\n";
        //     if (ctxtEnc_gpu != should_be_gpu) {
        //         throw std::logic_error("EvalSlotstoCoeffs input gpu mismatch\n");
        //     }
        // }

        // {
        //     Ciphertext<DCRTPoly> ctxtEnc_from_gpu = ctxtEnc->CloneZero();
        //     LoadCtAccurateFromGPU(ctxtEnc_from_gpu, ctxtEnc_gpu, elemParamsPointer);
        //     // std::cout << "checking ct...\n";
        //     if (*ctxtEnc_from_gpu != *ctxtEnc) {
        //         throw std::logic_error("EvalSlotstoCoeffs input cpu mismatch\n");
        //     }
        // }

        // linear transform for decoding
        // ctxtDec = (isLTBootstrap) ? EvalLinearTransform(precom->m_U0Pre, ctxtEnc) :
        // ctxtDec = EvalSlotsToCoeffs(precom->m_U0PreFFT, ctxtEnc);
                                    // EvalSlotsToCoeffsGPU(precom->m_U0PreFFT, ctxtEnc, gpu_context);
                                    // EvalSlotsToCoeffsGPU(A_gpu_stoc, ctxtEnc, gpu_context);
        ctxtDec_gpu = EvalSlotsToCoeffsGPU(A_gpu_stoc, ctxtEnc_gpu, ciphertext->GetSlots(), ciphertext->GetKeyTag(), gpu_context, cc);

        // {   // check StoC outputs
        //     auto should_be_gpu = LoadAccurateCiphertext(ctxtDec);
        //     // std::cout << "checking accurate ct...\n";
        //     if (ctxtDec_gpu != should_be_gpu) {
        //         throw std::logic_error("EvalSlotstoCoeffs input gpu mismatch\n");
        //     }
        // }

        // {
        //     Ciphertext<DCRTPoly> ctxtDec_from_gpu = ctxtDec->CloneZero();
        //     LoadCtAccurateFromGPU(ctxtDec_from_gpu, ctxtDec_gpu, elemParamsPointer);
        //     // std::cout << "checking ct...\n";
        //     if (*ctxtDec_from_gpu != *ctxtDec) {
        //         throw std::logic_error("EvalSlotstoCoeffs input cpu mismatch\n");
        //     }
        // }

        // ctxtDec = ciphertext->CloneZero();
        // LoadCtAccurateFromGPU(ctxtDec, ctxtDec_gpu, elemParamsPointer);

        // std::cout << "\nBeginning difficult rotate\n";
        // Ciphertext<DCRTPoly> ctxtDec_rot = cc->EvalRotate(ctxtDec, slots);

        // const std::map<usint, lbcrypto::EvalKey<lbcrypto::DCRTPoly>>& evalKeys = cc->GetEvalAutomorphismKeyMap(ciphertext->GetKeyTag());
        const uint32_t autoIndex = algo->FindAutomorphismIndex(slots, M);
        // std::cout << "\tGPU autoindex: " << autoIndex << " " << slots << " " << M << std::endl; 

        auto gpu_eval_key_iterator = gpu_context.preloaded_rotation_key_map->find(autoIndex);

        const auto ctxtDec_digits = gpu_context.ModUp(ctxtDec_gpu.ax__);
        // ckks::CtAccurate ctxtDec_gpu_rot_ext = EvalFastRotateGPUCore(ctxtDec_gpu, ctxtDec_digits, autoIndex, evalKeys, cc, gpu_context, true);
        ckks::CtAccurate ctxtDec_gpu_rot_ext;
        gpu_context.KeySwitch(ctxtDec_digits, gpu_eval_key_iterator->second, ctxtDec_gpu_rot_ext.ax__, ctxtDec_gpu_rot_ext.bx__);
        ckks::CtAccurate ctxtDec_gpu_rot;
        ctxtDec_gpu_rot.level = ctxtDec_gpu.level; 
        ctxtDec_gpu_rot.noiseScaleDeg = ctxtDec_gpu.noiseScaleDeg;
        ctxtDec_gpu_rot.scalingFactor = ctxtDec_gpu.scalingFactor; 
        gpu_context.ModDown(ctxtDec_gpu_rot_ext.ax__, ctxtDec_gpu_rot.ax__);
        gpu_context.ModDown(ctxtDec_gpu_rot_ext.bx__, ctxtDec_gpu_rot.bx__);
        // ckks::DeviceVector ctxtDec_gpu_rot_msg = gpu_context.AutomorphismTransform()

        gpu_context.AddCoreInPlace(ctxtDec_gpu_rot.bx__, ctxtDec_gpu.bx__);

        gpu_context.AutomorphismTransformInPlace(ctxtDec_gpu_rot, autoIndex);

        // auto ctxtDec_gpu_rot_msg = gpu_context.AutomorphismTransform(ctxtDec_gpu.bx__, autoIndex);

        // {
        //     // check equality
        //     Ciphertext<DCRTPoly> ctxtDec_rot_from_gpu = ctxtDec_rot->CloneZero();
        //     LoadCtAccurateFromGPU(ctxtDec_rot_from_gpu, ctxtDec_gpu_rot, elemParamsPointer);
        //     // std::cout << "checking ct...\n";
        //     if (*ctxtDec_rot_from_gpu != *ctxtDec_rot) {
        //         throw std::logic_error("Final rotate mismatch\n");
        //     }
        // }
        gpu_context.EvalAddInPlace(ctxtDec_gpu, ctxtDec_gpu_rot);

        // cc->EvalAddInPlace(ctxtDec, cc->EvalRotate(ctxtDec, slots));
        // {
        //     // check equality
        //     Ciphertext<DCRTPoly> ctxtDec_from_gpu = ctxtDec->CloneZero();
        //     LoadCtAccurateFromGPU(ctxtDec_from_gpu, ctxtDec_gpu, elemParamsPointer);
        //     // std::cout << "checking ct...\n";
        //     if (*ctxtDec_from_gpu != *ctxtDec) {
        //         throw std::logic_error("Final rotate and add mismatch\n");
        //     }
        // }
    }

    // ctxtDec = ciphertext->CloneZero();
    // LoadCtAccurateFromGPU(ctxtDec, ctxtDec_gpu, elemParamsPointer);

#if NATIVEINT != 128
    // 64-bit only: scale back the message to its original scale.
    uint64_t corFactor = (uint64_t)1 << std::llround(correction);
    // algo->MultByIntegerInPlace(ctxtDec, corFactor);
    gpu_context.EvalMultIntegerInPlace(ctxtDec_gpu, corFactor);
    // {
    //     // check equality
    //     Ciphertext<DCRTPoly> ctxtDec_from_gpu = ctxtDec->CloneZero();
    //     LoadCtAccurateFromGPU(ctxtDec_from_gpu, ctxtDec_gpu, elemParamsPointer);
    //     // std::cout << "checking ct...\n";
    //     if (*ctxtDec_from_gpu != *ctxtDec) {
    //         throw std::logic_error("Final pt mult mismatch\n");
    //     }
    // }
#endif

#ifdef BOOTSTRAPTIMING
    timeDecode = TOC(t);

    std::cout << "Decoding time: " << timeDecode / 1000.0 << " s" << std::endl;
#endif

    ctxtDec = ciphertext->CloneZero();
    LoadCtAccurateFromGPU(ctxtDec, ctxtDec_gpu, elemParamsPointer);

    auto bootstrappingNumTowers = ctxtDec->GetElements()[0].GetNumOfElements();

    // If we start with more towers, than we obtain from bootstrapping, return the original ciphertext.
    if (bootstrappingNumTowers <= initSizeQ) {
        return ciphertext->Clone();
    }

    return ctxtDec;
}


void FHECKKSRNS::EvalModReduce(ConstCiphertext<DCRTPoly> ciphertext, uint32_t numIterations, uint32_t precision) const {
    const auto cryptoParams = std::dynamic_pointer_cast<CryptoParametersCKKSRNS>(ciphertext->GetCryptoParameters());

    if (cryptoParams->GetKeySwitchTechnique() != HYBRID)
        OPENFHE_THROW(config_error, "CKKS Bootstrapping is only supported for the Hybrid key switching method.");


    if (numIterations != 1 && numIterations != 2) {
        OPENFHE_THROW(config_error, "CKKS Iterative Bootstrapping is only supported for 1 or 2 iterations.");
    }

#ifdef BOOTSTRAPTIMING
    TimeVar t;
    double timeModReduce(0.0);
#endif

    auto cc        = ciphertext->GetCryptoContext();
    uint32_t M     = cc->GetCyclotomicOrder();
    uint32_t L0    = cryptoParams->GetElementParams()->GetParams().size();
    // auto initSizeQ = ciphertext->GetElements()[0].GetNumOfElements();

    std::cout << "Beginning CKKS mod reduction (ONLY FOR BENCHMARKING PURPOSES)\n";

    ckks::Context gpu_context = GenGPUContext(cryptoParams);
    gpu_context.EnableMemoryPool();
    std::cout << "Generated gpu context\n";
    auto evk = LoadEvalMultRelinKey(cc);
    gpu_context.preloaded_evaluation_key = &evk;

    const auto elemParamsPointer = cc->GetElementParams();

    uint32_t slots = ciphertext->GetSlots();

    auto pair = m_bootPrecomMap.find(slots);
    if (pair == m_bootPrecomMap.end()) {
        std::string errorMsg(std::string("Precomputations for ") + std::to_string(slots) +
                             std::string(" slots were not generated") +
                             std::string(" Need to call EvalBootstrapSetup and then EvalBootstrapKeyGen to proceed"));
        OPENFHE_THROW(type_error, errorMsg);
    }
    const std::shared_ptr<CKKSBootstrapPrecom> precom = pair->second;
    size_t N                                          = cc->GetRingDimension();

    auto elementParamsRaised = *(cryptoParams->GetElementParams());

    // For FLEXIBLEAUTOEXT we raised ciphertext does not include extra modulus
    // as it is multiplied by auxiliary plaintext
    if (cryptoParams->GetScalingTechnique() == FLEXIBLEAUTOEXT) {
        elementParamsRaised.PopLastParam();
    }

    auto paramsQ = elementParamsRaised.GetParams();
    usint sizeQ  = paramsQ.size();

    std::vector<NativeInteger> moduli(sizeQ);
    std::vector<NativeInteger> roots(sizeQ);
    for (size_t i = 0; i < sizeQ; i++) {
        moduli[i] = paramsQ[i]->GetModulus();
        roots[i]  = paramsQ[i]->GetRootOfUnity();
    }
    auto elementParamsRaisedPtr = std::make_shared<ILDCRTParams<DCRTPoly::Integer>>(M, moduli, roots);

    NativeInteger q = elementParamsRaisedPtr->GetParams()[0]->GetModulus().ConvertToInt();
    double qDouble  = q.ConvertToDouble();

    const auto p = cryptoParams->GetPlaintextModulus();
    double powP  = pow(2, p);

    int32_t deg = std::round(std::log2(qDouble / powP));
#if NATIVEINT != 128
    if (deg > static_cast<int32_t>(m_correctionFactor)) {
        OPENFHE_THROW(math_error, "Degree [" + std::to_string(deg) +
                                      "] must be less than or equal to the correction factor [" +
                                      std::to_string(m_correctionFactor) + "].");
    }
#endif
    uint32_t correction = m_correctionFactor - deg;
    double post         = std::pow(2, static_cast<double>(deg));

    double pre      = 1. / post;
    uint64_t scalar = std::llround(post);

    //------------------------------------------------------------------------------
    // RAISING THE MODULUS
    //------------------------------------------------------------------------------

    // In FLEXIBLEAUTO, raising the ciphertext to a larger number
    // of towers is a bit more complex, because we need to adjust
    // it's scaling factor to the one that corresponds to the level
    // it's being raised to.
    // Increasing the modulus

    Ciphertext<DCRTPoly> raised = ciphertext->Clone();
    auto algo                   = cc->GetScheme();
    algo->ModReduceInternalInPlace(raised, raised->GetNoiseScaleDeg() - 1);

    AdjustCiphertext(raised, correction);
    auto ctxtDCRT = raised->GetElements();

    // We only use the level 0 ciphertext here. All other towers are automatically ignored to make
    // CKKS bootstrapping faster.
    for (size_t i = 0; i < ctxtDCRT.size(); i++) {
        DCRTPoly temp(elementParamsRaisedPtr, COEFFICIENT);
        ctxtDCRT[i].SetFormat(COEFFICIENT);
        temp = ctxtDCRT[i].GetElementAtIndex(0);
        temp.SetFormat(EVALUATION);
        ctxtDCRT[i] = temp;
    }

    raised->SetElements(ctxtDCRT);
    raised->SetLevel(L0 - ctxtDCRT[0].GetNumOfElements());

    ckks::CtAccurate ctxtEnc_input = LoadAccurateCiphertext(raised);

#ifdef BOOTSTRAPTIMING
    std::cerr << "\nNumber of levels at the beginning of bootstrapping: "
              << raised->GetElements()[0].GetNumOfElements() - 1 << std::endl;
#endif

    //------------------------------------------------------------------------------
    // SETTING PARAMETERS FOR APPROXIMATE MODULAR REDUCTION
    //------------------------------------------------------------------------------

    // Coefficients of the Chebyshev series interpolating 1/(2 Pi) Sin(2 Pi K x)
    std::vector<double> coefficients;
    double k = 0;

    if (cryptoParams->GetSecretKeyDist() == SPARSE_TERNARY) {
        coefficients = g_coefficientsSparse;
        // k = K_SPARSE;
        k = 1.0;  // do not divide by k as we already did it during precomputation
    }
    else {
        coefficients = g_coefficientsUniform;
        k            = K_UNIFORM;
    }

    double constantEvalMult = pre * (1.0 / (k * N));

    cc->EvalMultInPlace(raised, constantEvalMult);

    // no linear transformations are needed for Chebyshev series as the range has been normalized to [-1,1]
    double coeffLowerBound = -1;
    double coeffUpperBound = 1;

#ifdef BOOTSTRAPTIMING
        // const uint32_t num_iters = 10;
        const uint32_t num_iters = 1;
        std::cout << "running " << num_iters << " iterations...\n";
        TIC(t);
        for (uint32_t iter = 0; iter < num_iters; iter++) {
#endif
        //------------------------------------------------------------------------------
        // Running Approximate Mod Reduction
        //------------------------------------------------------------------------------

        // Ciphertext<DCRTPoly> ctxtEnc = ciphertext->CloneZero();
        // LoadCtAccurateFromGPU(ctxtEnc, ctxtEnc_gpu, elemParamsPointer);

        // Evaluate Chebyshev series for the sine wave
        // Essentially all of the time is here...
        // ctxtEnc = cc->EvalChebyshevSeries(ctxtEnc, coefficients, coeffLowerBound, coeffUpperBound);
        // ctxtEnc = cc->EvalChebyshevSeriesGPU(ctxtEnc, coefficients, coeffLowerBound, coeffUpperBound, gpu_context);
        // ckks::CtAccurate ctxtEnc_gpu = cc->EvalChebyshevSeriesGPU(ctxtEnc, coefficients, coeffLowerBound, coeffUpperBound, gpu_context);
        ckks::CtAccurate ctxtEnc_gpu = cc->EvalChebyshevSeriesGPU(ctxtEnc_input, coefficients, coeffLowerBound, coeffUpperBound, gpu_context, cc);
        // LoadCtAccurateFromGPU(ctxtEnc, ctxtEnc_gpu, elemParamsPointer);

        // Double-angle iterations
        if ((cryptoParams->GetSecretKeyDist() == UNIFORM_TERNARY) ||
            (cryptoParams->GetSecretKeyDist() == SPARSE_TERNARY)) {
            if (cryptoParams->GetScalingTechnique() != FIXEDMANUAL) {
                // algo->ModReduceInternalInPlace(ctxtEnc, BASE_NUM_LEVELS_TO_DROP);
                ctxtEnc_gpu = gpu_context.Rescale(ctxtEnc_gpu);
            }
            uint32_t numIter;
            if (cryptoParams->GetSecretKeyDist() == UNIFORM_TERNARY)
                numIter = R_UNIFORM;
            else
                numIter = R_SPARSE;
            // ApplyDoubleAngleIterations(ctxtEnc, numIter);
            // ApplyDoubleAngleIterationsGPU(ctxtEnc, numIter, gpu_context);
            ApplyDoubleAngleIterationsGPU(ctxtEnc_gpu, numIter, gpu_context, cc);
        }

        // scale the message back up after Chebyshev interpolation
        // algo->MultByIntegerInPlace(ctxtEnc, scalar);
        gpu_context.EvalMultIntegerInPlace(ctxtEnc_gpu, scalar);

#ifdef BOOTSTRAPTIMING
        }
        timeModReduce = TOC(t);

        std::cerr << "Approximate modular reduction time: " << timeModReduce / 1000.0 / num_iters << " s" << std::endl;
#endif
    
}


ckks::CtAccurate FHECKKSRNS::EvalFastRotateGPUCore(
    const ckks::CtAccurate& ciphertext,
    const ckks::DeviceVector& digits,
    const uint32_t autoIndex,  // sometimes the mapping is different
    const std::map<usint, lbcrypto::EvalKey<lbcrypto::DCRTPoly>>& evalKeys,
    const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc,
    const ckks::Context& gpu_context,
    const bool addFirst
) const {
    ckks::CtAccurate inner_prod_gpu;

    auto gpu_eval_key_iterator = gpu_context.preloaded_rotation_key_map->find(autoIndex);

    if (gpu_eval_key_iterator == gpu_context.preloaded_rotation_key_map->end()) {

        // std::cout << "Loading rotation key from CPU\n";

        // Retrieve the automorphism key that corresponds to the auto index.
        auto evalKeyIterator = evalKeys.find(autoIndex);
        if (evalKeyIterator == evalKeys.end()) {
            OPENFHE_THROW(openfhe_error, "EvalKey for index [" + std::to_string(autoIndex) + "] is not found.");
        }
        auto evalKey = evalKeyIterator->second;

        const ckks::EvaluationKey evalKey_gpu = LoadRelinKey(evalKey);

        // This is just the inner product with the rotation key
        gpu_context.KeySwitch(digits, evalKey_gpu, inner_prod_gpu.ax__, inner_prod_gpu.bx__);
    } else {
        gpu_context.KeySwitch(digits, gpu_eval_key_iterator->second, inner_prod_gpu.ax__, inner_prod_gpu.bx__);
    }

    // This is the message multiplication by P to avoid the rescale after the ModDown
    if (addFirst) {
        gpu_context.AddScaledMessageTerm(inner_prod_gpu.bx__, ciphertext.bx__);
    }

    gpu_context.AutomorphismTransformInPlace(inner_prod_gpu, autoIndex);

    inner_prod_gpu.level = ciphertext.level;
    inner_prod_gpu.scalingFactor = ciphertext.scalingFactor;
    inner_prod_gpu.noiseScaleDeg = ciphertext.noiseScaleDeg;

    return inner_prod_gpu;
}

ckks::CtAccurate FHECKKSRNS::EvalFastRotateExtGPU(
    const ckks::CtAccurate& ciphertext,
    const ckks::DeviceVector& digits,
    const uint32_t rot_ind,
    const std::map<usint, lbcrypto::EvalKey<lbcrypto::DCRTPoly>>& evalKeys,
    const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc,
    const ckks::Context& gpu_context,
    const bool addFirst
) const {
    const auto cryptoParams = std::dynamic_pointer_cast<CryptoParametersCKKSRNS>(cc->GetCryptoParameters());

    usint M = cryptoParams->GetElementParams()->GetCyclotomicOrder();

    // Find the automorphism index that corresponds to rotation index index.
    usint autoIndex = FindAutomorphismIndex2nComplex(rot_ind, M);
    // std::cout << "autoIndex: " << autoIndex << std::endl;

    return EvalFastRotateGPUCore(ciphertext, digits, autoIndex, evalKeys, cc, gpu_context, addFirst);

    // ckks::CtAccurate inner_prod_gpu;

    // auto gpu_eval_key_iterator = gpu_context.preloaded_rotation_key_map->find(autoIndex);

    // if (gpu_eval_key_iterator == gpu_context.preloaded_rotation_key_map->end()) {

    //     // Retrieve the automorphism key that corresponds to the auto index.
    //     auto evalKeyIterator = evalKeys.find(autoIndex);
    //     if (evalKeyIterator == evalKeys.end()) {
    //         OPENFHE_THROW(openfhe_error, "EvalKey for index [" + std::to_string(autoIndex) + "] is not found.");
    //     }
    //     auto evalKey = evalKeyIterator->second;

    //     const ckks::EvaluationKey evalKey_gpu = LoadRelinKey(evalKey);

    //     // This is just the inner product with the rotation key
    //     gpu_context.KeySwitch(digits, evalKey_gpu, inner_prod_gpu.ax__, inner_prod_gpu.bx__);
    // } else {
    //     gpu_context.KeySwitch(digits, gpu_eval_key_iterator->second, inner_prod_gpu.ax__, inner_prod_gpu.bx__);
    // }

    // // This is the message multiplication by P to avoid the rescale after the ModDown
    // if (addFirst) {
    //     gpu_context.AddScaledMessageTerm(inner_prod_gpu.bx__, ciphertext.bx__);
    // }

    // // usint N = cryptoParams->GetElementParams()->GetRingDimension();
    // // std::vector<usint> vec(N);
    // // PrecomputeAutoMap(N, autoIndex, &vec);

    // // // inner_prod_gpu.bx__ = gpu_context.AutomorphismTransform(inner_prod_gpu.bx__, vec);
    // // // inner_prod_gpu.ax__ = gpu_context.AutomorphismTransform(inner_prod_gpu.ax__, vec);
    // // gpu_context.AutomorphismTransformInPlace(inner_prod_gpu, vec);

    // gpu_context.AutomorphismTransformInPlace(inner_prod_gpu, autoIndex);

    // inner_prod_gpu.level = ciphertext.level;
    // inner_prod_gpu.scalingFactor = ciphertext.scalingFactor;
    // inner_prod_gpu.noiseScaleDeg = ciphertext.noiseScaleDeg;

    // return inner_prod_gpu;
}




};