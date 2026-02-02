/**
 * CKKS OpenFHE GPU Backend for ckks-torch
 * 
 * Hybrid CPU/GPU implementation:
 * - CPU (OpenFHE): Encryption, Decryption, Key Generation
 * - GPU (ckks::Context): Heavy compute operations (mul, add, rotate, rescale)
 * 
 * Uses Utils.h bridge functions for CPU <-> GPU data transfer.
 */
#include <openfhe.h>
#include "scheme/ckksrns/ckksrns-cryptoparameters.h"
#include "scheme/ckksrns/ckksrns-fhe.h"
#include "gpu/Utils.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <limits>
#include <stdexcept>
#include <vector>
#include <map>
#include <memory>
#include <chrono>

namespace py = pybind11;
using namespace lbcrypto;

namespace {

struct GPUContextHandle {
    CryptoContext<DCRTPoly> context;
    std::vector<uint32_t> level_budget;
    bool bootstrap_enabled;
    uint32_t bootstrap_num_slots = 0;
    
    // GPU context and keys
    std::unique_ptr<ckks::Context> gpu_context;
    std::unique_ptr<ckks::EvaluationKey> gpu_evk;
    std::unique_ptr<std::map<int, ckks::EvaluationKey>> gpu_rot_keys;
    
    // Track if GPU is initialized
    bool gpu_initialized = false;
    
    // Crypto parameters for GPU operations
    std::shared_ptr<CryptoParametersCKKSRNS> crypto_params;
};

struct GPUKeySetHandle {
    std::shared_ptr<GPUContextHandle> context;
    PublicKey<DCRTPoly> public_key;
    PrivateKey<DCRTPoly> secret_key;
};

struct GPUCiphertextHandle {
    std::shared_ptr<GPUContextHandle> context;
    Ciphertext<DCRTPoly> ciphertext;
    
    // GPU ciphertext (lazy loaded)
    mutable std::unique_ptr<ckks::CtAccurate> gpu_ct;
    mutable bool gpu_loaded = false;
    
    // Load to GPU if not already loaded
    void ensureGPU() const {
        if (!gpu_loaded && context->gpu_initialized) {
            gpu_ct = std::make_unique<ckks::CtAccurate>(
                LoadAccurateCiphertext(ciphertext)
            );
            gpu_loaded = true;
        }
    }
    
    void syncFromGPU() {
        if (gpu_loaded && gpu_ct && context->crypto_params) {
            auto params = context->crypto_params;
            auto allParams = params->GetElementParams();
            
            auto paramsVec = allParams->GetParams();
            size_t numTowers = paramsVec.size() - gpu_ct->level;
            
            std::vector<std::shared_ptr<lbcrypto::ILNativeParams>> nativeParams;
            for (size_t i = 0; i < numTowers; i++) {
                nativeParams.push_back(paramsVec[i]);
            }
            auto levelParams = std::make_shared<ILDCRTParams<BigInteger>>(allParams->GetCyclotomicOrder(), nativeParams);
            
            DCRTPoly gpu_res_0 = loadIntoDCRTPoly(gpu_ct->bx__, levelParams);
            DCRTPoly gpu_res_1 = loadIntoDCRTPoly(gpu_ct->ax__, levelParams);
            
            std::vector<DCRTPoly> elements = {gpu_res_0, gpu_res_1};
            ciphertext->SetElements(elements);
            ciphertext->SetLevel(gpu_ct->level);
            ciphertext->SetScalingFactor(gpu_ct->scalingFactor);
            ciphertext->SetNoiseScaleDeg(gpu_ct->noiseScaleDeg);
        }
    }
};

std::shared_ptr<GPUCiphertextHandle> make_cipher(
    const std::shared_ptr<GPUContextHandle>& ctx,
    const Ciphertext<DCRTPoly>& cipher
) {
    auto handle = std::make_shared<GPUCiphertextHandle>();
    handle->context = ctx;
    handle->ciphertext = cipher;
    return handle;
}

std::shared_ptr<GPUCiphertextHandle> make_cipher_from_gpu(
    const std::shared_ptr<GPUContextHandle>& ctx,
    const ckks::CtAccurate& gpu_ct
) {
    auto handle = std::make_shared<GPUCiphertextHandle>();
    handle->context = ctx;
    handle->gpu_ct = std::make_unique<ckks::CtAccurate>(gpu_ct);
    handle->gpu_loaded = true;
    
    // Create CPU ciphertext from GPU result
    handle->ciphertext = ctx->context->Encrypt(
        ctx->context->KeyGen().publicKey,
        ctx->context->MakeCKKSPackedPlaintext(std::vector<double>{0.0})
    );
    handle->syncFromGPU();
    
    return handle;
}

std::shared_ptr<GPUCiphertextHandle> make_cipher_from_gpu_lazy(
    const std::shared_ptr<GPUContextHandle>& ctx,
    ckks::CtAccurate&& gpu_ct,
    const Ciphertext<DCRTPoly>& template_ct
) {
    auto handle = std::make_shared<GPUCiphertextHandle>();
    handle->context = ctx;
    handle->gpu_ct = std::make_unique<ckks::CtAccurate>(std::move(gpu_ct));
    handle->gpu_loaded = true;
    
    handle->ciphertext = template_ct->Clone();
    
    return handle;
}

void syncGPUtoCPU(GPUCiphertextHandle* handle) {
    if (!handle->gpu_loaded || !handle->gpu_ct) return;
    
    auto ctx = handle->context;
    auto params = ctx->crypto_params;
    auto allParams = params->GetElementParams();
    
    auto paramsVec = allParams->GetParams();
    size_t numTowers = paramsVec.size() - handle->gpu_ct->level;
    std::vector<std::shared_ptr<lbcrypto::ILNativeParams>> nativeParams;
    for (size_t i = 0; i < numTowers; i++) {
        nativeParams.push_back(paramsVec[i]);
    }
    auto levelParams = std::make_shared<ILDCRTParams<BigInteger>>(allParams->GetCyclotomicOrder(), nativeParams);
    
    DCRTPoly gpu_res_0 = loadIntoDCRTPoly(handle->gpu_ct->bx__, levelParams);
    DCRTPoly gpu_res_1 = loadIntoDCRTPoly(handle->gpu_ct->ax__, levelParams);
    
    std::vector<DCRTPoly> elements = {gpu_res_0, gpu_res_1};
    handle->ciphertext->SetElements(elements);
    handle->ciphertext->SetLevel(handle->gpu_ct->level);
    handle->ciphertext->SetScalingFactor(handle->gpu_ct->scalingFactor);
    handle->ciphertext->SetNoiseScaleDeg(handle->gpu_ct->noiseScaleDeg);
}

std::shared_ptr<GPUCiphertextHandle> make_cipher_from_gpu_fast(
    const std::shared_ptr<GPUContextHandle>& ctx,
    const ckks::CtAccurate& gpu_ct,
    const Ciphertext<DCRTPoly>& template_ct
) {
    auto handle = std::make_shared<GPUCiphertextHandle>();
    handle->context = ctx;
    handle->gpu_ct = std::make_unique<ckks::CtAccurate>(gpu_ct);
    handle->gpu_loaded = true;
    
    handle->ciphertext = template_ct->CloneZero();
    
    syncGPUtoCPU(handle.get());
    
    return handle;
}

Plaintext make_plaintext(
    const std::shared_ptr<GPUContextHandle>& ctx,
    const std::vector<double>& values
) {
    return ctx->context->MakeCKKSPackedPlaintext(values);
}

py::dict cipher_metadata(const std::shared_ptr<GPUCiphertextHandle>& handle) {
    py::dict meta;
    auto cc = handle->context->context;
    meta["scale"] = handle->ciphertext->GetScalingFactor();
    meta["level"] = handle->ciphertext->GetLevel();
    meta["noise_scale"] = handle->ciphertext->GetNoiseScaleDeg();
    meta["slots"] = static_cast<std::uint32_t>(cc->GetRingDimension() / 2);
    meta["gpu_loaded"] = handle->gpu_loaded;
    meta["gpu_enabled"] = handle->context->gpu_initialized;
    auto params = std::dynamic_pointer_cast<CryptoParametersCKKSRNS>(cc->GetCryptoParameters());
    if (params) {
        meta["base_scale"] = params->GetScalingFactorReal();
        meta["max_level"] = static_cast<std::uint32_t>(params->GetElementParams()->GetParams().size() - 1);
    }
    return meta;
}

}  // namespace

// Initialize GPU context from existing OpenFHE context
void init_gpu(const std::shared_ptr<GPUContextHandle>& ctx) {
    if (ctx->gpu_initialized) return;
    
    try {
        auto params = std::dynamic_pointer_cast<CryptoParametersCKKSRNS>(
            ctx->context->GetCryptoParameters()
        );
        if (!params) {
            throw std::runtime_error("Failed to get CKKS crypto parameters");
        }
        
        ctx->crypto_params = params;
        
        // Generate GPU context
        ctx->gpu_context = std::make_unique<ckks::Context>(
            GenGPUContext(params)
        );
        
        // Load evaluation keys to GPU
        ctx->gpu_evk = std::make_unique<ckks::EvaluationKey>(
            LoadEvalMultRelinKey(ctx->context)
        );
        
        ctx->gpu_rot_keys = std::make_unique<std::map<int, ckks::EvaluationKey>>();
        
        ctx->gpu_initialized = true;
        std::cout << "GPU context initialized successfully" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "GPU initialization failed: " << e.what() << std::endl;
        ctx->gpu_initialized = false;
    }
}

std::shared_ptr<GPUContextHandle> create_context(
    std::uint32_t poly_mod_degree,
    const std::vector<std::uint32_t>& coeff_mod_bits,
    std::uint32_t scale_bits,
    std::uint32_t security_level_code,
    bool enable_bootstrap,
    const std::vector<std::uint32_t>& level_budget,
    std::uint32_t batch_size,
    bool enable_gpu
) {
    CCParams<CryptoContextCKKSRNS> parameters;
    auto sec_level = security_level_code == std::numeric_limits<uint32_t>::max()
                         ? HEStd_NotSet
                         : convertToSecurityLevel(security_level_code);
    parameters.SetSecurityLevel(sec_level);
    parameters.SetRingDim(poly_mod_degree);
    parameters.SetBatchSize(batch_size);
    parameters.SetScalingModSize(scale_bits);
    parameters.SetScalingTechnique(FLEXIBLEAUTO);
    parameters.SetSecretKeyDist(UNIFORM_TERNARY);
    parameters.SetFirstModSize(60);
    
    if (enable_bootstrap && !level_budget.empty()) {
        uint32_t levelsAfterBootstrap = 10;
        uint32_t bootstrapDepth = FHECKKSRNS::GetBootstrapDepth(level_budget, UNIFORM_TERNARY);
        parameters.SetMultiplicativeDepth(levelsAfterBootstrap + bootstrapDepth);
    } else if (!coeff_mod_bits.empty()) {
        parameters.SetMultiplicativeDepth(static_cast<uint32_t>(coeff_mod_bits.size() - 1));
    }
    parameters.SetExecutionMode(EXEC_EVALUATION);

    CryptoContext<DCRTPoly> context = GenCryptoContext(parameters);
    context->Enable(PKE);
    context->Enable(KEYSWITCH);
    context->Enable(LEVELEDSHE);
    context->Enable(ADVANCEDSHE);
    uint32_t bootstrap_slots = 0;
    if (enable_bootstrap) {
        context->Enable(FHE);
        if (!level_budget.empty()) {
            std::vector<uint32_t> bsgsDim = {0, 0};
            bootstrap_slots = poly_mod_degree / 2;
            uint32_t correctionFactor = 21;
            context->EvalBootstrapSetup(level_budget, bsgsDim, bootstrap_slots, correctionFactor);
        }
    }

    auto handle = std::make_shared<GPUContextHandle>();
    handle->context = context;
    handle->bootstrap_enabled = enable_bootstrap;
    handle->level_budget = level_budget;
    handle->bootstrap_num_slots = bootstrap_slots;
    
    return handle;
}

std::shared_ptr<GPUKeySetHandle> keygen(
    const std::shared_ptr<GPUContextHandle>& ctx,
    const std::vector<int>& rotations,
    bool relin,
    bool conj,
    bool enable_gpu
) {
    auto context = ctx->context;
    auto key_pair = context->KeyGen();
    if (!key_pair.good()) {
        throw std::runtime_error("Key generation failed");
    }

    if (relin) {
        context->EvalMultKeyGen(key_pair.secretKey);
    }

    if (!rotations.empty()) {
        context->EvalAtIndexKeyGen(key_pair.secretKey, rotations);
    }

    context->EvalSumKeyGen(key_pair.secretKey);
    if (conj) {
        std::vector<usint> conj_index{static_cast<usint>(2 * context->GetRingDimension() - 1)};
        context->EvalAutomorphismKeyGen(key_pair.secretKey, conj_index);
    }

    if (ctx->bootstrap_enabled) {
        std::uint32_t num_slots = ctx->bootstrap_num_slots;
        if (num_slots == 0) {
            num_slots = context->GetRingDimension() / 4;
        }
        context->EvalBootstrapKeyGen(key_pair.secretKey, num_slots);
    }
    
    if (enable_gpu) {
        init_gpu(ctx);
        
        if (ctx->gpu_initialized && !rotations.empty()) {
            try {
                auto key_tag = key_pair.secretKey->GetKeyTag();
                auto automorph_keys = context->GetEvalAutomorphismKeyMap(key_tag);
                
                for (int rot : rotations) {
                    uint32_t auto_idx = context->FindAutomorphismIndex(rot);
                    auto it = automorph_keys.find(auto_idx);
                    if (it != automorph_keys.end()) {
                        ctx->gpu_rot_keys->emplace(rot, LoadRelinKey(it->second));
                    }
                }
            } catch (const std::exception& e) {
                std::cerr << "GPU rotation key loading failed: " << e.what() << std::endl;
            }
        }
    }

    auto handle = std::make_shared<GPUKeySetHandle>();
    handle->context = ctx;
    handle->public_key = key_pair.publicKey;
    handle->secret_key = key_pair.secretKey;
    return handle;
}

std::shared_ptr<GPUCiphertextHandle> encrypt(
    const std::shared_ptr<GPUContextHandle>& ctx,
    const std::shared_ptr<GPUKeySetHandle>& keys,
    const std::vector<double>& values
) {
    auto plaintext = make_plaintext(ctx, values);
    auto cipher = ctx->context->Encrypt(keys->public_key, plaintext);
    return make_cipher(ctx, cipher);
}

std::vector<double> decrypt(
    const std::shared_ptr<GPUContextHandle>& ctx,
    const std::shared_ptr<GPUKeySetHandle>& keys,
    const std::shared_ptr<GPUCiphertextHandle>& cipher
) {
    // Sync from GPU if needed
    if (cipher->gpu_loaded) {
        const_cast<GPUCiphertextHandle*>(cipher.get())->syncFromGPU();
    }
    
    Plaintext result;
    ctx->context->Decrypt(keys->secret_key, cipher->ciphertext, &result);
    result->SetLength(result->GetCKKSPackedValue().size());
    return result->GetRealPackedValue();
}

// ============== GPU-Accelerated Operations ==============

std::shared_ptr<GPUCiphertextHandle> add_cipher(
    const std::shared_ptr<GPUCiphertextHandle>& lhs,
    const std::shared_ptr<GPUCiphertextHandle>& rhs
) {
    auto ctx = lhs->context;
    
    if (ctx->gpu_initialized && lhs->gpu_loaded && rhs->gpu_loaded) {
        lhs->ensureGPU();
        rhs->ensureGPU();
        
        ckks::CtAccurate result = ctx->gpu_context->Add(*lhs->gpu_ct, *rhs->gpu_ct);
        return make_cipher_from_gpu_lazy(ctx, std::move(result), lhs->ciphertext);
    } else {
        auto result = ctx->context->EvalAdd(lhs->ciphertext, rhs->ciphertext);
        return make_cipher(ctx, result);
    }
}

std::shared_ptr<GPUCiphertextHandle> add_plain(
    const std::shared_ptr<GPUCiphertextHandle>& lhs,
    const std::vector<double>& plain
) {
    auto ctx = lhs->context;
    auto plaintext = make_plaintext(ctx, plain);
    auto result = ctx->context->EvalAdd(lhs->ciphertext, plaintext);
    return make_cipher(ctx, result);
}

std::shared_ptr<GPUCiphertextHandle> sub_cipher(
    const std::shared_ptr<GPUCiphertextHandle>& lhs,
    const std::shared_ptr<GPUCiphertextHandle>& rhs
) {
    auto ctx = lhs->context;
    
    if (ctx->gpu_initialized && lhs->gpu_loaded && rhs->gpu_loaded) {
        lhs->ensureGPU();
        rhs->ensureGPU();
        
        ckks::CtAccurate result = ctx->gpu_context->Sub(*lhs->gpu_ct, *rhs->gpu_ct);
        return make_cipher_from_gpu_lazy(ctx, std::move(result), lhs->ciphertext);
    } else {
        auto result = ctx->context->EvalSub(lhs->ciphertext, rhs->ciphertext);
        return make_cipher(ctx, result);
    }
}

std::shared_ptr<GPUCiphertextHandle> sub_plain(
    const std::shared_ptr<GPUCiphertextHandle>& lhs,
    const std::vector<double>& plain
) {
    auto ctx = lhs->context;
    auto plaintext = make_plaintext(ctx, plain);
    auto result = ctx->context->EvalSub(lhs->ciphertext, plaintext);
    return make_cipher(ctx, result);
}

std::shared_ptr<GPUCiphertextHandle> mul_cipher(
    const std::shared_ptr<GPUCiphertextHandle>& lhs,
    const std::shared_ptr<GPUCiphertextHandle>& rhs
) {
    auto ctx = lhs->context;
    
    if (ctx->gpu_initialized && ctx->gpu_evk) {
        lhs->ensureGPU();
        rhs->ensureGPU();
        ckks::CtAccurate result = ctx->gpu_context->EvalMultAndRelinNoRescale(
            *lhs->gpu_ct, *rhs->gpu_ct, *ctx->gpu_evk);
        return make_cipher_from_gpu_lazy(ctx, std::move(result), lhs->ciphertext);
    }
    
    lhs->syncFromGPU();
    rhs->syncFromGPU();
    
    auto result = ctx->context->EvalMultAndRelinearize(lhs->ciphertext, rhs->ciphertext);
    return make_cipher(ctx, result);
}

std::shared_ptr<GPUCiphertextHandle> mul_plain(
    const std::shared_ptr<GPUCiphertextHandle>& lhs,
    const std::vector<double>& plain
) {
    auto ctx = lhs->context;
    
    if (ctx->gpu_initialized) {
        lhs->ensureGPU();
        auto plaintext = make_plaintext(ctx, plain);
        plaintext->Encode();
        ckks::PtAccurate gpu_pt = LoadAccuratePlaintext(
            plaintext, plaintext->GetElement<DCRTPoly>());
        ckks::CtAccurate result = ctx->gpu_context->EvalMultPlainExt(
            *lhs->gpu_ct, gpu_pt);
        return make_cipher_from_gpu_lazy(ctx, std::move(result), lhs->ciphertext);
    }
    
    auto plaintext = make_plaintext(ctx, plain);
    auto result = ctx->context->EvalMult(lhs->ciphertext, plaintext);
    return make_cipher(ctx, result);
}

std::shared_ptr<GPUCiphertextHandle> square_cipher(
    const std::shared_ptr<GPUCiphertextHandle>& cipher
) {
    auto ctx = cipher->context;
    
    if (ctx->gpu_initialized && ctx->gpu_evk) {
        // GPU path: square + relin (no rescale) - matches mul_cipher behavior
        // Python API calls .rescale() separately after square()
        cipher->ensureGPU();
        ckks::CtAccurate result = ctx->gpu_context->EvalSquareAndRelinNoRescale(
            *cipher->gpu_ct, *ctx->gpu_evk);
        return make_cipher_from_gpu_lazy(ctx, std::move(result), cipher->ciphertext);
    }
    
    // CPU fallback: square + relin (no rescale) - matches GPU path
    auto result = ctx->context->EvalMultAndRelinearize(cipher->ciphertext, cipher->ciphertext);
    return make_cipher(ctx, result);
}

std::shared_ptr<GPUCiphertextHandle> rescale_cipher(
    const std::shared_ptr<GPUCiphertextHandle>& tensor
) {
    auto ctx = tensor->context;
    
    if (ctx->gpu_initialized && tensor->gpu_loaded) {
        // GPU path
        ckks::CtAccurate result = ctx->gpu_context->Rescale(*tensor->gpu_ct);
        return make_cipher_from_gpu_fast(ctx, result, tensor->ciphertext);
    } else {
        // CPU fallback
        auto result = ctx->context->ModReduce(tensor->ciphertext);
        return make_cipher(ctx, result);
    }
}

std::shared_ptr<GPUCiphertextHandle> rotate_cipher(
    const std::shared_ptr<GPUCiphertextHandle>& tensor,
    int index
) {
    auto ctx = tensor->context;

    if (ctx->gpu_initialized && ctx->gpu_rot_keys) {
        auto it = ctx->gpu_rot_keys->find(index);
        if (it != ctx->gpu_rot_keys->end()) {
            // GPU path: use existing GPU rotation kernel
            tensor->ensureGPU();
            uint32_t auto_index = ctx->context->FindAutomorphismIndex(index);
            ckks::CtAccurate result = ctx->gpu_context->EvalAtIndex(
                *tensor->gpu_ct, it->second, auto_index);
            return make_cipher_from_gpu_lazy(ctx, std::move(result), tensor->ciphertext);
        }
        // Key not found for this rotation index - fall through to CPU with warning
    }

    // CPU fallback (existing behavior)
    if (tensor->gpu_loaded) {
        tensor->syncFromGPU();
    }
    auto result = ctx->context->EvalAtIndex(tensor->ciphertext, index);
    return make_cipher(ctx, result);
}

std::shared_ptr<GPUCiphertextHandle> sum_slots_cipher(
    const std::shared_ptr<GPUCiphertextHandle>& tensor
) {
    auto ctx = tensor->context;
    std::uint32_t slots = ctx->context->GetRingDimension() / 2;
    auto result = ctx->context->EvalSum(tensor->ciphertext, slots);
    return make_cipher(ctx, result);
}

std::shared_ptr<GPUCiphertextHandle> matvec_diag_cipher(
    const std::shared_ptr<GPUCiphertextHandle>& tensor,
    const std::vector<std::vector<double>>& diagonals
) {
    auto ctx = tensor->context;
    auto cc = ctx->context;
    auto accumulator = cc->EvalMult(tensor->ciphertext, make_plaintext(ctx, diagonals.front()));
    for (std::size_t idx = 1; idx < diagonals.size(); ++idx) {
        auto rotated = cc->EvalAtIndex(tensor->ciphertext, static_cast<int>(idx));
        auto term = cc->EvalMult(rotated, make_plaintext(ctx, diagonals[idx]));
        accumulator = cc->EvalAdd(accumulator, term);
    }
    return make_cipher(ctx, accumulator);
}

std::shared_ptr<GPUCiphertextHandle> poly_eval_cipher(
    const std::shared_ptr<GPUCiphertextHandle>& tensor,
    const std::vector<double>& coeffs
) {
    auto ctx = tensor->context;
    auto result = ctx->context->EvalPoly(tensor->ciphertext, coeffs);
    return make_cipher(ctx, result);
}

std::shared_ptr<GPUCiphertextHandle> matmul_dense_cipher(
    const std::shared_ptr<GPUCiphertextHandle>& tensor,
    const std::vector<std::vector<double>>& matrix
) {
    auto ctx = tensor->context;
    auto cc = ctx->context;
    const std::size_t m = matrix.size();
    if (m == 0) throw std::invalid_argument("matrix must not be empty");
    const std::size_t n = matrix.front().size();
    if (n == 0) throw std::invalid_argument("matrix rows must not be empty");
    const std::size_t slots = static_cast<std::size_t>(cc->GetRingDimension() / 2);
    if (n > slots) throw std::invalid_argument("matrix column dimension exceeds available CKKS slots");

    Ciphertext<DCRTPoly> accumulator;
    for (std::size_t k = 0; k < n; ++k) {
        std::vector<double> diag(slots, 0.0);
        for (std::size_t i = 0; i < m; ++i) {
            diag[i] = matrix[i][(i + k) % n];
        }
        auto rotated = (k == 0) ? tensor->ciphertext : cc->EvalAtIndex(tensor->ciphertext, static_cast<int>(k));
        auto term = cc->EvalMult(rotated, make_plaintext(ctx, diag));
        if (!accumulator) {
            accumulator = term;
        } else {
            accumulator = cc->EvalAdd(accumulator, term);
        }
    }
    return make_cipher(ctx, accumulator);
}

std::shared_ptr<GPUCiphertextHandle> conjugate_cipher(
    const std::shared_ptr<GPUCiphertextHandle>& tensor
) {
    auto ctx = tensor->context;
    auto cc = ctx->context;
    const auto& key_map = cc->GetEvalAutomorphismKeyMap(tensor->ciphertext->GetKeyTag());
    usint conj_index = static_cast<usint>(2 * cc->GetRingDimension() - 1);
    auto result = cc->EvalAutomorphism(tensor->ciphertext, conj_index, key_map);
    return make_cipher(ctx, result);
}

std::shared_ptr<GPUCiphertextHandle> matmul_bsgs_cipher(
    const std::shared_ptr<GPUCiphertextHandle>& tensor,
    const std::vector<std::vector<double>>& matrix,
    std::uint32_t bsgs_n1,
    std::uint32_t bsgs_n2
) {
    auto ctx = tensor->context;
    auto cc = ctx->context;
    const std::size_t out_features = matrix.size();
    if (out_features == 0) throw std::invalid_argument("matrix must not be empty");
    const std::size_t in_features = matrix.front().size();
    if (in_features == 0) throw std::invalid_argument("matrix rows must not be empty");
    const std::size_t slots = static_cast<std::size_t>(cc->GetRingDimension() / 2);
    
    if (bsgs_n1 == 0) {
        bsgs_n1 = static_cast<std::uint32_t>(std::ceil(std::sqrt(static_cast<double>(in_features))));
    }
    if (bsgs_n2 == 0) {
        bsgs_n2 = (static_cast<std::uint32_t>(in_features) + bsgs_n1 - 1) / bsgs_n1;
    }
    
    if (tensor->gpu_loaded) {
        const_cast<GPUCiphertextHandle*>(tensor.get())->syncFromGPU();
    }
    
    std::vector<Ciphertext<DCRTPoly>> baby_ciphers;
    baby_ciphers.reserve(bsgs_n1);
    
    auto precomp = cc->EvalFastRotationPrecompute(tensor->ciphertext);
    
    for (std::uint32_t j = 0; j < bsgs_n1 && j < in_features; ++j) {
        if (j == 0) {
            baby_ciphers.push_back(tensor->ciphertext);
        } else {
            auto rotated = cc->EvalFastRotation(tensor->ciphertext, j, cc->GetCyclotomicOrder(), precomp);
            baby_ciphers.push_back(rotated);
        }
    }
    
    Ciphertext<DCRTPoly> accumulator;
    
    for (std::uint32_t k = 0; k < bsgs_n2; ++k) {
        std::uint32_t giant_step = k * bsgs_n1;
        Ciphertext<DCRTPoly> block;
        
        for (std::uint32_t j = 0; j < baby_ciphers.size(); ++j) {
            std::uint32_t d = giant_step + j;
            if (d >= in_features) break;
            
            std::vector<double> diag(slots, 0.0);
            for (std::size_t row = 0; row < out_features; ++row) {
                std::size_t col = d;
                if (col < in_features) {
                    diag[row] = matrix[row][col];
                }
            }
            
            auto term = cc->EvalMult(baby_ciphers[j], make_plaintext(ctx, diag));
            
            if (!block) {
                block = term;
            } else {
                block = cc->EvalAdd(block, term);
            }
        }
        
        if (!block) continue;
        
        if (!accumulator) {
            accumulator = block;
        } else {
            accumulator = cc->EvalAdd(accumulator, block);
        }
    }
    
    auto result = make_cipher(ctx, accumulator);
    if (ctx->gpu_initialized) {
        result->ensureGPU();
    }
    return result;
}

std::shared_ptr<GPUCiphertextHandle> bootstrap_cipher(
    const std::shared_ptr<GPUCiphertextHandle>& tensor
) {
    auto ctx = tensor->context;
    if (tensor->gpu_loaded) {
        const_cast<GPUCiphertextHandle*>(tensor.get())->syncFromGPU();
    }
    auto result = ctx->context->EvalBootstrap(tensor->ciphertext);
    return make_cipher(ctx, result);
}

// GPU status check
bool is_gpu_available() {
    // Check if CUDA is available
    try {
        // Simple check - if we can create a DeviceVector, GPU is available
        ckks::HostVector test_vec(16, 0);
        ckks::DeviceVector gpu_vec(test_vec);
        return true;
    } catch (...) {
        return false;
    }
}

py::dict get_gpu_info() {
    py::dict info;
    info["gpu_available"] = is_gpu_available();
    return info;
}

PYBIND11_MODULE(ckks_openfhe_gpu_backend, m) {
    m.doc() = "CKKS OpenFHE GPU Backend - Hybrid CPU/GPU acceleration for HE operations";
    
    py::class_<GPUContextHandle, std::shared_ptr<GPUContextHandle>>(m, "ContextHandle")
        .def_property_readonly("ring_dim", [](const GPUContextHandle& handle) {
            return handle.context->GetRingDimension();
        })
        .def_property_readonly("gpu_enabled", [](const GPUContextHandle& handle) {
            return handle.gpu_initialized;
        });

    py::class_<GPUKeySetHandle, std::shared_ptr<GPUKeySetHandle>>(m, "KeySetHandle");

    py::class_<GPUCiphertextHandle, std::shared_ptr<GPUCiphertextHandle>>(m, "CiphertextHandle")
        .def_property_readonly("gpu_loaded", [](const GPUCiphertextHandle& handle) {
            return handle.gpu_loaded;
        });

    m.def("create_context", &create_context, 
          py::arg("poly_mod_degree"), py::arg("coeff_mod_bits"),
          py::arg("scale_bits"), py::arg("security_level_code"), 
          py::arg("enable_bootstrap"), py::arg("level_budget"),
          py::arg("batch_size"), py::arg("enable_gpu") = true);
    m.def("keygen", &keygen, 
          py::arg("context"), py::arg("rotations"), 
          py::arg("relin"), py::arg("conj"), 
          py::arg("enable_gpu") = true);
    m.def("encrypt", &encrypt, py::arg("context"), py::arg("keys"), py::arg("values"));
    m.def("decrypt", &decrypt, py::arg("context"), py::arg("keys"), py::arg("cipher"));
    m.def("add_cipher", &add_cipher, py::arg("lhs"), py::arg("rhs"));
    m.def("add_plain", &add_plain, py::arg("lhs"), py::arg("plain"));
    m.def("sub_cipher", &sub_cipher, py::arg("lhs"), py::arg("rhs"));
    m.def("sub_plain", &sub_plain, py::arg("lhs"), py::arg("plain"));
    m.def("mul_cipher", &mul_cipher, py::arg("lhs"), py::arg("rhs"));
    m.def("mul_plain", &mul_plain, py::arg("lhs"), py::arg("plain"));
    m.def("square", &square_cipher, py::arg("cipher"));
    m.def("rescale", &rescale_cipher, py::arg("tensor"));
    m.def("rotate", &rotate_cipher, py::arg("tensor"), py::arg("index"));
    m.def("sum_slots", &sum_slots_cipher, py::arg("tensor"));
    m.def("matvec_diag", &matvec_diag_cipher, py::arg("tensor"), py::arg("diagonals"));
    m.def("matmul_dense", &matmul_dense_cipher, py::arg("tensor"), py::arg("matrix"));
    m.def("matmul_bsgs", &matmul_bsgs_cipher, py::arg("tensor"), py::arg("matrix"), 
          py::arg("bsgs_n1") = 0, py::arg("bsgs_n2") = 0);
    m.def("poly_eval", &poly_eval_cipher, py::arg("tensor"), py::arg("coeffs"));
    m.def("conjugate", &conjugate_cipher, py::arg("tensor"));
    m.def("bootstrap", &bootstrap_cipher, py::arg("tensor"));
    m.def("cipher_metadata", &cipher_metadata, py::arg("cipher"));
    m.def("is_gpu_available", &is_gpu_available);
    m.def("get_gpu_info", &get_gpu_info);
}
