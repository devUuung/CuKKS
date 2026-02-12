/**
 * CKKS OpenFHE GPU Backend for CuKKS
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
#include <cmath>
#include <unordered_set>
#include <unordered_map>
#include <cstdlib>
#include <cstring>
#include <string>

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

    std::string key_tag;

    std::unordered_map<std::uint64_t, ckks::PtAccurate> gpu_plain_cache;
    std::unordered_map<std::uint64_t, ckks::PtAccurate> gpu_plain_cache_pinned;
    std::size_t gpu_plain_cache_limit = 2048;
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
    
    handle->ciphertext = template_ct->CloneZero();
    
    return handle;
}

void syncGPUtoCPU(GPUCiphertextHandle* handle) {
    if (!handle->gpu_loaded || !handle->gpu_ct) return;
    
    auto ctx = handle->context;
    auto params = ctx->crypto_params;
    auto allParams = params->GetElementParams();
    
    auto paramsVec = allParams->GetParams();
    
    if (handle->gpu_ct->level > paramsVec.size()) {
        throw std::runtime_error("syncGPUtoCPU: GPU level exceeds available towers");
    }
    
    size_t numTowers = paramsVec.size() - handle->gpu_ct->level;
    
    std::vector<std::shared_ptr<lbcrypto::ILNativeParams>> nativeParams;
    for (size_t i = 0; i < numTowers; i++) {
        nativeParams.push_back(paramsVec[i]);
    }
    auto levelParams = std::make_shared<ILDCRTParams<BigInteger>>(
        allParams->GetCyclotomicOrder(), nativeParams);
    
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

bool has_gpu_rotation(const std::shared_ptr<GPUContextHandle>& ctx, int rotation) {
    if (!ctx->gpu_rot_keys) return false;
    return ctx->gpu_rot_keys->find(rotation) != ctx->gpu_rot_keys->end();
}

bool ensure_gpu_rotation_key(const std::shared_ptr<GPUContextHandle>& ctx, int rotation) {
    if (!ctx->gpu_initialized || !ctx->gpu_rot_keys) return false;
    if (has_gpu_rotation(ctx, rotation)) return true;
    if (ctx->key_tag.empty()) return false;

    try {
        auto automorph_keys = ctx->context->GetEvalAutomorphismKeyMap(ctx->key_tag);
        uint32_t auto_idx = ctx->context->FindAutomorphismIndex(rotation);
        auto it = automorph_keys.find(auto_idx);
        if (it == automorph_keys.end()) {
            return false;
        }
        ctx->gpu_rot_keys->emplace(rotation, LoadRelinKey(it->second));
        return true;
    } catch (...) {
        return false;
    }
}

bool env_flag(const char* name, bool default_value) {
    const char* v = std::getenv(name);
    if (v == nullptr) return default_value;
    if (v[0] == '\0') return default_value;
    return v[0] != '0';
}

bool use_gpu_plaintext_cache() {
    static const bool enabled = env_flag("CUKKS_GPU_PTXT_CACHE", true);
    return enabled;
}

bool use_tree_reduction() {
    static const bool enabled = env_flag("CUKKS_GPU_TREE_REDUCTION", false);
    return enabled;
}

bool use_batch_muladd() {
    static const bool enabled = env_flag("CUKKS_GPU_BATCH_MULADD", true);
    return enabled;
}

// ============== Profiling infrastructure ==============
bool profiling_enabled() {
    static const bool enabled = env_flag("CUKKS_PROFILE", false);
    return enabled;
}

struct OpTimer {
    const char* name;
    std::chrono::high_resolution_clock::time_point start;
    bool active;

    explicit OpTimer(const char* op_name) : name(op_name), active(profiling_enabled()) {
        if (active) {
            start = std::chrono::high_resolution_clock::now();
        }
    }
    ~OpTimer() {
        if (active) {
            auto end = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(end - start).count();
            std::cerr << "[CUKKS_PROFILE] " << name << ": " << ms << " ms" << std::endl;
        }
    }
};

std::uint64_t hash_plain_vector(const std::vector<double>& values, uint32_t level) {
    std::uint64_t h = 1469598103934665603ULL;
    auto mix = [&](std::uint64_t x) {
        h ^= x;
        h *= 1099511628211ULL;
    };

    mix(static_cast<std::uint64_t>(level));
    mix(static_cast<std::uint64_t>(values.size()));
    for (double v : values) {
        std::uint64_t bits = 0;
        std::memcpy(&bits, &v, sizeof(bits));
        mix(bits);
    }
    return h;
}

std::uint64_t hash_weight_diag(std::uint64_t weight_hash, std::uint32_t d, std::uint32_t bsgs_n1, std::uint32_t level) {
    std::uint64_t h = 0x517cc1b727220a95ULL;
    auto mix = [&](std::uint64_t x) {
        h ^= x;
        h *= 1099511628211ULL;
    };
    mix(weight_hash);
    mix(static_cast<std::uint64_t>(d));
    mix(static_cast<std::uint64_t>(bsgs_n1));
    mix(static_cast<std::uint64_t>(level));
    return h;
}

ckks::PtAccurate make_gpu_plaintext(
    const std::shared_ptr<GPUContextHandle>& ctx,
    const std::vector<double>& values,
    uint32_t level
) {
    auto plaintext = ctx->context->MakeCKKSPackedPlaintext(values, 1, level);
    plaintext->Encode();
    return LoadAccuratePlaintext(plaintext, plaintext->GetElement<DCRTPoly>());
}

const ckks::PtAccurate& get_cached_gpu_plaintext(
    const std::shared_ptr<GPUContextHandle>& ctx,
    const std::vector<double>& values,
    uint32_t level
) {
    const std::uint64_t key = hash_plain_vector(values, level);

    auto pin_it = ctx->gpu_plain_cache_pinned.find(key);
    if (pin_it != ctx->gpu_plain_cache_pinned.end()) {
        return pin_it->second;
    }

    auto it = ctx->gpu_plain_cache.find(key);
    if (it != ctx->gpu_plain_cache.end()) {
        return it->second;
    }

    if (ctx->gpu_plain_cache_limit > 0 &&
        ctx->gpu_plain_cache.size() >= ctx->gpu_plain_cache_limit) {
        ctx->gpu_plain_cache.clear();
    }

    ckks::PtAccurate gpu_pt = make_gpu_plaintext(ctx, values, level);
    auto inserted = ctx->gpu_plain_cache.emplace(key, std::move(gpu_pt));
    return inserted.first->second;
}

std::vector<ckks::PtAccurate> make_gpu_plaintext_cache(
    const std::shared_ptr<GPUContextHandle>& ctx,
    const std::vector<std::vector<double>>& diagonals,
    uint32_t level
) {
    std::vector<ckks::PtAccurate> cache;
    cache.reserve(diagonals.size());
    for (const auto& diag : diagonals) {
        cache.push_back(make_gpu_plaintext(ctx, diag, level));
    }
    return cache;
}

ckks::CtAccurate reduce_add_gpu(
    const std::shared_ptr<GPUContextHandle>& ctx,
    std::vector<ckks::CtAccurate>& terms
) {
    if (terms.empty()) {
        throw std::runtime_error("reduce_add_gpu received empty terms");
    }
    if (!use_tree_reduction() || terms.size() <= 128) {
        ckks::CtAccurate acc = std::move(terms.front());
        for (std::size_t i = 1; i < terms.size(); ++i) {
            acc = ctx->gpu_context->Add(acc, terms[i]);
        }
        return acc;
    }
    while (terms.size() > 1) {
        std::vector<ckks::CtAccurate> next;
        next.reserve((terms.size() + 1) / 2);
        for (std::size_t i = 0; i + 1 < terms.size(); i += 2) {
            next.push_back(ctx->gpu_context->Add(terms[i], terms[i + 1]));
        }
        if (terms.size() % 2 == 1) {
            next.push_back(std::move(terms.back()));
        }
        terms = std::move(next);
    }
    return std::move(terms.front());
}

ckks::CtAccurate batch_mult_plain_add_gpu(
    const std::shared_ptr<GPUContextHandle>& ctx,
    const std::vector<const ckks::CtAccurate*>& cts,
    const std::vector<const ckks::PtAccurate*>& pts
) {
    if (cts.empty() || pts.empty() || cts.size() != pts.size()) {
        throw std::runtime_error("batch_mult_plain_add_gpu received invalid inputs");
    }

    const auto degree = ctx->gpu_context->GetDegree();
    const int num_primes = static_cast<int>(cts.front()->ax__.size() / degree);

    std::vector<const std::uint64_t*> ax_addr;
    std::vector<const std::uint64_t*> bx_addr;
    std::vector<const std::uint64_t*> mx_addr;
    ax_addr.reserve(cts.size());
    bx_addr.reserve(cts.size());
    mx_addr.reserve(cts.size());

    for (std::size_t i = 0; i < cts.size(); ++i) {
        ax_addr.push_back(cts[i]->ax__.data());
        bx_addr.push_back(cts[i]->bx__.data());
        mx_addr.push_back(pts[i]->mx__.data());
    }

    ckks::CtAccurate out;
    out.ax__.resize(static_cast<std::size_t>(num_primes) * degree);
    out.bx__.resize(static_cast<std::size_t>(num_primes) * degree);
    ctx->gpu_context->hadamardMultAndAddBatch(ax_addr, bx_addr, mx_addr, num_primes, out.ax__, out.bx__);

    out.level = cts.front()->level;
    out.noiseScaleDeg = cts.front()->noiseScaleDeg + pts.front()->noiseScaleDeg;
    out.scalingFactor = cts.front()->scalingFactor * pts.front()->scalingFactor;
    return out;
}

Plaintext make_plaintext(
    const std::shared_ptr<GPUContextHandle>& ctx,
    const std::vector<double>& values,
    uint32_t level = 0
) {
    return ctx->context->MakeCKKSPackedPlaintext(values, 1, level);
}

py::dict cipher_metadata(const std::shared_ptr<GPUCiphertextHandle>& handle) {
    py::dict meta;
    auto cc = handle->context->context;
    
    if (handle->gpu_loaded && handle->gpu_ct) {
        meta["scale"] = handle->gpu_ct->scalingFactor;
        meta["level"] = handle->gpu_ct->level;
        meta["noise_scale"] = handle->gpu_ct->noiseScaleDeg;
    } else {
        meta["scale"] = handle->ciphertext->GetScalingFactor();
        meta["level"] = handle->ciphertext->GetLevel();
        meta["noise_scale"] = handle->ciphertext->GetNoiseScaleDeg();
    }
    
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
            LoadEvalMultRelinKey(ctx->context, ctx->key_tag)
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

    ctx->key_tag = key_pair.secretKey->GetKeyTag();
    
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
    OpTimer timer("encrypt");
    auto plaintext = make_plaintext(ctx, values);
    auto cipher = ctx->context->Encrypt(keys->public_key, plaintext);
    return make_cipher(ctx, cipher);
}

std::vector<double> decrypt(
    const std::shared_ptr<GPUContextHandle>& ctx,
    const std::shared_ptr<GPUKeySetHandle>& keys,
    const std::shared_ptr<GPUCiphertextHandle>& cipher
) {
    OpTimer timer("decrypt");
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
    OpTimer timer("add_plain");
    auto ctx = lhs->context;
    if (ctx->gpu_initialized) {
        lhs->ensureGPU();
        uint32_t ct_level = lhs->gpu_ct->level;
        const ckks::PtAccurate& gpu_pt = get_cached_gpu_plaintext(ctx, plain, ct_level);
        ckks::CtAccurate result = *lhs->gpu_ct;
        ctx->gpu_context->AddCoreInPlace(result.bx__, gpu_pt.mx__);
        return make_cipher_from_gpu_lazy(ctx, std::move(result), lhs->ciphertext);
    }
    if (lhs->gpu_loaded) {
        const_cast<GPUCiphertextHandle*>(lhs.get())->syncFromGPU();
    }
    uint32_t ct_level = lhs->ciphertext->GetLevel();
    auto plaintext = make_plaintext(ctx, plain, ct_level);
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
    if (ctx->gpu_initialized) {
        lhs->ensureGPU();
        uint32_t ct_level = lhs->gpu_ct->level;
        const ckks::PtAccurate& gpu_pt = get_cached_gpu_plaintext(ctx, plain, ct_level);
        ckks::CtAccurate result = *lhs->gpu_ct;
        const auto degree = ctx->gpu_context->GetDegree();
        const int num_primes = static_cast<int>(result.bx__.size() / degree);
        ctx->gpu_context->SubInplace(result.bx__.data(), gpu_pt.mx__.data(), num_primes);
        return make_cipher_from_gpu_lazy(ctx, std::move(result), lhs->ciphertext);
    }
    if (lhs->gpu_loaded) {
        const_cast<GPUCiphertextHandle*>(lhs.get())->syncFromGPU();
    }
    uint32_t ct_level = lhs->ciphertext->GetLevel();
    auto plaintext = make_plaintext(ctx, plain, ct_level);
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
        uint32_t ct_level = lhs->gpu_ct->level;
        auto plaintext = ctx->context->MakeCKKSPackedPlaintext(plain, 1, ct_level);
        plaintext->Encode();
        ckks::PtAccurate gpu_pt = LoadAccuratePlaintext(
            plaintext, plaintext->GetElement<DCRTPoly>());
        ckks::CtAccurate result = ctx->gpu_context->EvalMultPlain(
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
    OpTimer timer("square");
    auto ctx = cipher->context;
    
    if (ctx->gpu_initialized && ctx->gpu_evk) {
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
    OpTimer timer("rescale");
    auto ctx = tensor->context;
    
    if (ctx->gpu_initialized && tensor->gpu_loaded) {
        ckks::CtAccurate result = ctx->gpu_context->Rescale(*tensor->gpu_ct);
        return make_cipher_from_gpu_lazy(ctx, std::move(result), tensor->ciphertext);
    } else {
        auto result = ctx->context->ModReduce(tensor->ciphertext);
        return make_cipher(ctx, result);
    }
}

std::shared_ptr<GPUCiphertextHandle> rotate_cipher(
    const std::shared_ptr<GPUCiphertextHandle>& tensor,
    int index
) {
    OpTimer timer("rotate");
    auto ctx = tensor->context;

    if (ctx->gpu_initialized && ctx->gpu_rot_keys) {
        auto it = ctx->gpu_rot_keys->find(index);
        if (it != ctx->gpu_rot_keys->end()) {
            tensor->ensureGPU();
            uint32_t auto_index = ctx->context->FindAutomorphismIndex(index);
            ckks::CtAccurate result = ctx->gpu_context->EvalAtIndex(
                *tensor->gpu_ct, it->second, auto_index);
            return make_cipher_from_gpu_lazy(ctx, std::move(result), tensor->ciphertext);
        }
    }

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
    if (ctx->gpu_initialized && ctx->gpu_rot_keys) {
        std::uint32_t slots = ctx->context->GetRingDimension() / 2;
        bool rotations_available = true;
        for (std::uint32_t step = 1; step < slots; step <<= 1) {
            if (!ensure_gpu_rotation_key(ctx, static_cast<int>(step))) {
                rotations_available = false;
                break;
            }
        }
        if (rotations_available) {
            tensor->ensureGPU();
            ckks::CtAccurate result = *tensor->gpu_ct;
            for (std::uint32_t step = 1; step < slots; step <<= 1) {
                auto it = ctx->gpu_rot_keys->find(static_cast<int>(step));
                uint32_t auto_index = ctx->context->FindAutomorphismIndex(static_cast<int>(step));
                ckks::CtAccurate rotated = ctx->gpu_context->EvalAtIndex(result, it->second, auto_index);
                result = ctx->gpu_context->Add(result, rotated);
            }
            return make_cipher_from_gpu_lazy(ctx, std::move(result), tensor->ciphertext);
        }
    }

    if (tensor->gpu_loaded) {
        const_cast<GPUCiphertextHandle*>(tensor.get())->syncFromGPU();
    }
    std::uint32_t slots = ctx->context->GetRingDimension() / 2;
    auto result = ctx->context->EvalSum(tensor->ciphertext, slots);
    return make_cipher(ctx, result);
}

std::shared_ptr<GPUCiphertextHandle> matvec_diag_cipher(
    const std::shared_ptr<GPUCiphertextHandle>& tensor,
    const std::vector<std::vector<double>>& diagonals
) {
    auto ctx = tensor->context;
    if (ctx->gpu_initialized && ctx->gpu_rot_keys) {
        std::vector<bool> diag_nonzero(diagonals.size(), false);
        for (std::size_t idx = 0; idx < diagonals.size(); ++idx) {
            for (double v : diagonals[idx]) {
                if (v != 0.0) {
                    diag_nonzero[idx] = true;
                    break;
                }
            }
        }

        std::unordered_set<int> required_rotations;
        for (std::size_t idx = 1; idx < diagonals.size(); ++idx) {
            if (diag_nonzero[idx]) {
                required_rotations.insert(static_cast<int>(idx));
            }
        }

        bool rotations_available = true;
        for (int rot : required_rotations) {
            if (!ensure_gpu_rotation_key(ctx, rot)) {
                rotations_available = false;
                break;
            }
        }

        if (rotations_available) {
            tensor->ensureGPU();
            std::vector<std::size_t> active_indices;
            active_indices.reserve(diagonals.size());
            for (std::size_t idx = 0; idx < diagonals.size(); ++idx) {
                if (idx == 0 || diag_nonzero[idx]) {
                    active_indices.push_back(idx);
                }
            }
            std::vector<ckks::PtAccurate> pt_cache;
            if (use_gpu_plaintext_cache()) {
                std::vector<std::vector<double>> active_diagonals;
                active_diagonals.reserve(active_indices.size());
                for (std::size_t idx : active_indices) {
                    active_diagonals.push_back(diagonals[idx]);
                }
                pt_cache = make_gpu_plaintext_cache(ctx, active_diagonals, tensor->gpu_ct->level);
            }
            std::vector<ckks::CtAccurate> rotated_terms;
            rotated_terms.reserve(active_indices.size());
            std::vector<ckks::PtAccurate> pt_terms;
            if (!use_gpu_plaintext_cache()) {
                pt_terms.reserve(active_indices.size());
            }

            for (std::size_t t = 0; t < active_indices.size(); ++t) {
                std::size_t idx = active_indices[t];

                ckks::CtAccurate rotated = *tensor->gpu_ct;
                if (idx > 0) {
                    auto it = ctx->gpu_rot_keys->find(static_cast<int>(idx));
                    uint32_t auto_index = ctx->context->FindAutomorphismIndex(static_cast<int>(idx));
                    rotated = ctx->gpu_context->EvalAtIndex(*tensor->gpu_ct, it->second, auto_index);
                }

                rotated_terms.push_back(std::move(rotated));

                if (!use_gpu_plaintext_cache()) {
                    auto gpu_pt = make_gpu_plaintext(ctx, diagonals[idx], tensor->gpu_ct->level);
                    pt_terms.push_back(std::move(gpu_pt));
                }
            }

            if (!rotated_terms.empty()) {
                if (use_batch_muladd()) {
                    std::vector<const ckks::CtAccurate*> ct_ptrs;
                    std::vector<const ckks::PtAccurate*> pt_ptrs;
                    ct_ptrs.reserve(rotated_terms.size());
                    pt_ptrs.reserve(rotated_terms.size());
                    for (std::size_t i = 0; i < rotated_terms.size(); ++i) {
                        ct_ptrs.push_back(&rotated_terms[i]);
                        pt_ptrs.push_back(use_gpu_plaintext_cache() ? &pt_cache[i] : &pt_terms[i]);
                    }
                    ckks::CtAccurate accumulator = batch_mult_plain_add_gpu(ctx, ct_ptrs, pt_ptrs);
                    return make_cipher_from_gpu_lazy(ctx, std::move(accumulator), tensor->ciphertext);
                }

                std::vector<ckks::CtAccurate> terms;
                terms.reserve(rotated_terms.size());
                for (std::size_t i = 0; i < rotated_terms.size(); ++i) {
                    const auto& pt = use_gpu_plaintext_cache() ? pt_cache[i] : pt_terms[i];
                    terms.push_back(ctx->gpu_context->EvalMultPlain(rotated_terms[i], pt));
                }
                ckks::CtAccurate accumulator = reduce_add_gpu(ctx, terms);
                return make_cipher_from_gpu_lazy(ctx, std::move(accumulator), tensor->ciphertext);
            }
        }
    }

    if (tensor->gpu_loaded) {
        const_cast<GPUCiphertextHandle*>(tensor.get())->syncFromGPU();
    }
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
    if (ctx->gpu_initialized && !coeffs.empty() && ctx->gpu_evk) {
        tensor->ensureGPU();

        if (coeffs.size() == 3 && coeffs[0] == 0.0 && coeffs[1] == 0.0 && coeffs[2] == 1.0) {
            ckks::CtAccurate squared = ctx->gpu_context->EvalSquareAndRelinNoRescale(*tensor->gpu_ct, *ctx->gpu_evk);
            return make_cipher_from_gpu_lazy(ctx, std::move(squared), tensor->ciphertext);
        }

        if (coeffs.size() == 5 && std::abs(coeffs[3]) <= 1e-12) {
            const auto slots = static_cast<std::size_t>(ctx->context->GetRingDimension() / 2);
            const std::vector<double> c0v(slots, coeffs[0]);
            const std::vector<double> c1v(slots, coeffs[1]);
            const std::vector<double> c2v(slots, coeffs[2]);
            const std::vector<double> c4v(slots, coeffs[4]);

            const ckks::CtAccurate& x0 = *tensor->gpu_ct;
            ckks::CtAccurate x2 = ctx->gpu_context->EvalSquareAndRelinNoRescale(x0, *ctx->gpu_evk);
            x2 = ctx->gpu_context->Rescale(x2);
            ckks::CtAccurate x4 = ctx->gpu_context->EvalSquareAndRelinNoRescale(x2, *ctx->gpu_evk);
            x4 = ctx->gpu_context->Rescale(x4);

            const ckks::PtAccurate& pt_c1 = get_cached_gpu_plaintext(ctx, c1v, x0.level);
            const ckks::PtAccurate& pt_c2 = get_cached_gpu_plaintext(ctx, c2v, x2.level);
            const ckks::PtAccurate& pt_c4 = get_cached_gpu_plaintext(ctx, c4v, x4.level);

            ckks::CtAccurate t1 = ctx->gpu_context->EvalMultPlain(x0, pt_c1);
            t1 = ctx->gpu_context->Rescale(t1);
            ckks::CtAccurate t2 = ctx->gpu_context->EvalMultPlain(x2, pt_c2);
            t2 = ctx->gpu_context->Rescale(t2);
            ckks::CtAccurate t4 = ctx->gpu_context->EvalMultPlain(x4, pt_c4);
            t4 = ctx->gpu_context->Rescale(t4);

            ckks::CtAccurate acc = ctx->gpu_context->Add(t1, t2);
            acc = ctx->gpu_context->Add(acc, t4);
            const ckks::PtAccurate& pt_c0 = get_cached_gpu_plaintext(ctx, c0v, acc.level);
            ctx->gpu_context->AddCoreInPlace(acc.bx__, pt_c0.mx__);

            return make_cipher_from_gpu_lazy(ctx, std::move(acc), tensor->ciphertext);
        }
    }

    if (tensor->gpu_loaded) {
        const_cast<GPUCiphertextHandle*>(tensor.get())->syncFromGPU();
    }
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

    // Pre-scan: identify non-zero diagonals to skip zero diagonals
    // (block-diagonal weight matrices have exact structural zeros)
    std::vector<bool> diag_nonzero(n, false);
    for (std::size_t k = 0; k < n; ++k) {
        for (std::size_t i = 0; i < m; ++i) {
            if (matrix[i][(i + k) % n] != 0.0) {
                diag_nonzero[k] = true;
                break;
            }
        }
    }

    if (ctx->gpu_initialized && ctx->gpu_rot_keys) {
        std::unordered_set<int> required_rotations;
        for (std::size_t k = 1; k < n; ++k) {
            if (diag_nonzero[k]) {
                required_rotations.insert(static_cast<int>(k));
            }
        }
        bool rotations_available = true;
        for (int rot : required_rotations) {
            if (!ensure_gpu_rotation_key(ctx, rot)) {
                rotations_available = false;
                break;
            }
        }

        if (rotations_available) {
            tensor->ensureGPU();
            std::vector<std::size_t> active_indices;
            active_indices.reserve(n);
            std::vector<std::vector<double>> active_diagonals;
            if (use_gpu_plaintext_cache()) {
                active_diagonals.reserve(n);
            }
            for (std::size_t k = 0; k < n; ++k) {
                if (!diag_nonzero[k]) continue;
                if (use_gpu_plaintext_cache()) {
                    std::vector<double> diag(slots, 0.0);
                    for (std::size_t i = 0; i < m; ++i) {
                        diag[i] = matrix[i][(i + k) % n];
                    }
                    active_diagonals.push_back(std::move(diag));
                }
                active_indices.push_back(k);
            }
            std::vector<ckks::PtAccurate> pt_cache;
            if (use_gpu_plaintext_cache()) {
                pt_cache = make_gpu_plaintext_cache(ctx, active_diagonals, tensor->gpu_ct->level);
            }
            std::vector<ckks::CtAccurate> rotated_terms;
            rotated_terms.reserve(active_indices.size());
            std::vector<ckks::PtAccurate> pt_terms;
            if (!use_gpu_plaintext_cache()) {
                pt_terms.reserve(active_indices.size());
            }

            for (std::size_t t = 0; t < active_indices.size(); ++t) {
                std::size_t k = active_indices[t];

                ckks::CtAccurate rotated = *tensor->gpu_ct;
                if (k > 0) {
                    auto it = ctx->gpu_rot_keys->find(static_cast<int>(k));
                    uint32_t auto_index = ctx->context->FindAutomorphismIndex(static_cast<int>(k));
                    rotated = ctx->gpu_context->EvalAtIndex(*tensor->gpu_ct, it->second, auto_index);
                }

                rotated_terms.push_back(std::move(rotated));

                if (!use_gpu_plaintext_cache()) {
                    std::vector<double> diag(slots, 0.0);
                    for (std::size_t i = 0; i < m; ++i) {
                        diag[i] = matrix[i][(i + k) % n];
                    }
                    auto gpu_pt = make_gpu_plaintext(ctx, diag, tensor->gpu_ct->level);
                    pt_terms.push_back(std::move(gpu_pt));
                }
            }

            if (rotated_terms.empty()) {
                throw std::runtime_error("failed to build accumulator in dense matmul (all-zero matrix)");
            }
            if (use_batch_muladd()) {
                std::vector<const ckks::CtAccurate*> ct_ptrs;
                std::vector<const ckks::PtAccurate*> pt_ptrs;
                ct_ptrs.reserve(rotated_terms.size());
                pt_ptrs.reserve(rotated_terms.size());
                for (std::size_t i = 0; i < rotated_terms.size(); ++i) {
                    ct_ptrs.push_back(&rotated_terms[i]);
                    pt_ptrs.push_back(use_gpu_plaintext_cache() ? &pt_cache[i] : &pt_terms[i]);
                }
                ckks::CtAccurate accumulator = batch_mult_plain_add_gpu(ctx, ct_ptrs, pt_ptrs);
                return make_cipher_from_gpu_lazy(ctx, std::move(accumulator), tensor->ciphertext);
            }

            std::vector<ckks::CtAccurate> terms;
            terms.reserve(rotated_terms.size());
            for (std::size_t i = 0; i < rotated_terms.size(); ++i) {
                const auto& pt = use_gpu_plaintext_cache() ? pt_cache[i] : pt_terms[i];
                terms.push_back(ctx->gpu_context->EvalMultPlain(rotated_terms[i], pt));
            }
            ckks::CtAccurate accumulator = reduce_add_gpu(ctx, terms);
            return make_cipher_from_gpu_lazy(ctx, std::move(accumulator), tensor->ciphertext);
        }
    }

    if (tensor->gpu_loaded) {
        const_cast<GPUCiphertextHandle*>(tensor.get())->syncFromGPU();
    }

    Ciphertext<DCRTPoly> accumulator;
    for (std::size_t k = 0; k < n; ++k) {
        if (!diag_nonzero[k]) continue;  // skip zero diagonal
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
    if (!accumulator) {
        throw std::runtime_error("failed to build accumulator in dense matmul (all-zero matrix)");
    }
    return make_cipher(ctx, accumulator);
}

std::shared_ptr<GPUCiphertextHandle> conjugate_cipher(
    const std::shared_ptr<GPUCiphertextHandle>& tensor
) {
    auto ctx = tensor->context;
    if (tensor->gpu_loaded) {
        const_cast<GPUCiphertextHandle*>(tensor.get())->syncFromGPU();
    }
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
    std::uint32_t bsgs_n2,
    std::uint64_t weight_hash = 0,
    const std::vector<bool>& precomputed_nonzero = {}
) {
    OpTimer timer("matmul_bsgs");
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
    
    // Pre-scan: identify non-zero diagonals to skip zero diagonals
    // (block-diagonal weight matrices have exact structural zeros)
    std::vector<bool> diag_nonzero_vec;
    if (!precomputed_nonzero.empty() && precomputed_nonzero.size() == in_features) {
        diag_nonzero_vec = precomputed_nonzero;
    } else {
        diag_nonzero_vec.resize(in_features, false);
        for (std::size_t d = 0; d < in_features; ++d) {
            for (std::size_t i = 0; i < out_features; ++i) {
                if (matrix[i][(i + d) % in_features] != 0.0) {
                    diag_nonzero_vec[d] = true;
                    break;
                }
            }
        }
    }

    if (ctx->gpu_initialized && ctx->gpu_rot_keys) {
        std::unordered_set<int> required_rotations;
        for (std::size_t d = 1; d < in_features; ++d) {
            if (!diag_nonzero_vec[d]) continue;
            int baby_step = static_cast<int>(d % bsgs_n1);
            int giant_step = static_cast<int>((d / bsgs_n1) * bsgs_n1);
            if (baby_step != 0) required_rotations.insert(baby_step);
            if (giant_step != 0) required_rotations.insert(giant_step);
        }

        bool rotations_available = true;
        for (int rot : required_rotations) {
            if (!ensure_gpu_rotation_key(ctx, rot)) {
                rotations_available = false;
                break;
            }
        }

        if (rotations_available) {
            tensor->ensureGPU();
            // Map diagonal index → pointer into the global gpu_plain_cache
            std::vector<const ckks::PtAccurate*> pt_cached_ptrs(in_features, nullptr);
            {
                OpTimer pt_timer("matmul_bsgs.pt_encode");
                if (use_gpu_plaintext_cache()) {
                    for (std::size_t d = 0; d < in_features; ++d) {
                        if (!diag_nonzero_vec[d]) continue;

                        if (weight_hash != 0) {
                            std::uint64_t fast_key = hash_weight_diag(weight_hash, static_cast<std::uint32_t>(d), bsgs_n1, tensor->gpu_ct->level);
                            auto pin_it = ctx->gpu_plain_cache_pinned.find(fast_key);
                            if (pin_it != ctx->gpu_plain_cache_pinned.end()) {
                                pt_cached_ptrs[d] = &pin_it->second;
                                continue;
                            }
                            auto cache_it = ctx->gpu_plain_cache.find(fast_key);
                            if (cache_it != ctx->gpu_plain_cache.end()) {
                                pt_cached_ptrs[d] = &cache_it->second;
                                continue;
                            }
                        }

                        std::uint32_t j = static_cast<std::uint32_t>(d % bsgs_n1);
                        std::uint32_t giant_step = static_cast<std::uint32_t>(d) - j;
                        std::vector<double> diag(slots, 0.0);
                        for (std::size_t i = 0; i < slots; ++i) {
                            std::int64_t row_signed = static_cast<std::int64_t>(i) - static_cast<std::int64_t>(giant_step);
                            std::size_t row = static_cast<std::size_t>((row_signed % static_cast<std::int64_t>(out_features) + static_cast<std::int64_t>(out_features)) % static_cast<std::int64_t>(out_features));
                            std::size_t col = (i + j) % in_features;
                            diag[i] = matrix[row][col];
                        }

                        if (weight_hash != 0) {
                            std::uint64_t fast_key = hash_weight_diag(weight_hash, static_cast<std::uint32_t>(d), bsgs_n1, tensor->gpu_ct->level);
                            if (ctx->gpu_plain_cache_limit > 0 && ctx->gpu_plain_cache.size() >= ctx->gpu_plain_cache_limit)
                                ctx->gpu_plain_cache.clear();
                            auto gpu_pt = make_gpu_plaintext(ctx, diag, tensor->gpu_ct->level);
                            auto [it, _] = ctx->gpu_plain_cache.emplace(fast_key, std::move(gpu_pt));
                            pt_cached_ptrs[d] = &it->second;
                        } else {
                            pt_cached_ptrs[d] = &get_cached_gpu_plaintext(ctx, diag, tensor->gpu_ct->level);
                        }
                    }
                }
            }

            std::vector<ckks::CtAccurate> baby_ciphers;
            baby_ciphers.reserve(bsgs_n1);

            {
                OpTimer baby_timer("matmul_bsgs.baby_steps");
                for (std::uint32_t j = 0; j < bsgs_n1 && j < in_features; ++j) {
                    if (j == 0) {
                        baby_ciphers.push_back(*tensor->gpu_ct);
                    } else {
                        auto it = ctx->gpu_rot_keys->find(static_cast<int>(j));
                        uint32_t auto_index = ctx->context->FindAutomorphismIndex(static_cast<int>(j));
                        baby_ciphers.push_back(ctx->gpu_context->EvalAtIndex(*tensor->gpu_ct, it->second, auto_index));
                    }
                }
            }

            std::vector<ckks::CtAccurate> block_terms;
            block_terms.reserve(bsgs_n2);

            {
                OpTimer gs_timer("matmul_bsgs.giant_steps");
                for (std::uint32_t k = 0; k < bsgs_n2; ++k) {
                    std::uint32_t giant_step = k * bsgs_n1;
                    std::vector<std::uint32_t> active_js;
                    active_js.reserve(baby_ciphers.size());
                    std::vector<ckks::PtAccurate> pt_terms;
                    if (!use_gpu_plaintext_cache()) {
                        pt_terms.reserve(baby_ciphers.size());
                    }

                    for (std::uint32_t j = 0; j < baby_ciphers.size(); ++j) {
                        std::uint32_t d = giant_step + j;
                        if (d >= in_features) break;
                        if (!diag_nonzero_vec[d]) continue;
                        active_js.push_back(j);
                        if (use_gpu_plaintext_cache()) {
                        } else {
                            std::vector<double> diag(slots, 0.0);
                            for (std::size_t i = 0; i < slots; ++i) {
                                std::int64_t row_signed = static_cast<std::int64_t>(i) - static_cast<std::int64_t>(giant_step);
                                std::size_t row = static_cast<std::size_t>((row_signed % static_cast<std::int64_t>(out_features) + static_cast<std::int64_t>(out_features)) % static_cast<std::int64_t>(out_features));
                                std::size_t col = (i + j) % in_features;
                                diag[i] = matrix[row][col];
                            }
                            auto gpu_pt = make_gpu_plaintext(ctx, diag, tensor->gpu_ct->level);
                            pt_terms.push_back(std::move(gpu_pt));
                        }
                    }

                    if (active_js.empty()) continue;
                    ckks::CtAccurate block;
                    if (use_batch_muladd()) {
                        std::vector<const ckks::CtAccurate*> ct_ptrs;
                        std::vector<const ckks::PtAccurate*> pt_ptrs;
                        ct_ptrs.reserve(active_js.size());
                        pt_ptrs.reserve(active_js.size());
                        for (std::size_t idx = 0; idx < active_js.size(); ++idx) {
                            std::uint32_t j = active_js[idx];
                            std::uint32_t d = giant_step + j;
                            ct_ptrs.push_back(&baby_ciphers[j]);
                            if (use_gpu_plaintext_cache()) {
                                pt_ptrs.push_back(pt_cached_ptrs[d]);
                            } else {
                                pt_ptrs.push_back(&pt_terms[idx]);
                            }
                        }
                        block = batch_mult_plain_add_gpu(ctx, ct_ptrs, pt_ptrs);
                    } else {
                        std::vector<ckks::CtAccurate> mult_terms;
                        mult_terms.reserve(active_js.size());
                        for (std::size_t idx = 0; idx < active_js.size(); ++idx) {
                            std::uint32_t j = active_js[idx];
                            std::uint32_t d = giant_step + j;
                            if (use_gpu_plaintext_cache()) {
                                mult_terms.push_back(ctx->gpu_context->EvalMultPlain(baby_ciphers[j], *pt_cached_ptrs[d]));
                            } else {
                                mult_terms.push_back(ctx->gpu_context->EvalMultPlain(baby_ciphers[j], pt_terms[idx]));
                            }
                        }
                        block = reduce_add_gpu(ctx, mult_terms);
                    }

                    if (k > 0) {
                        auto it = ctx->gpu_rot_keys->find(static_cast<int>(giant_step));
                        uint32_t auto_index = ctx->context->FindAutomorphismIndex(static_cast<int>(giant_step));
                        block = ctx->gpu_context->EvalAtIndex(block, it->second, auto_index);
                    }

                    block_terms.push_back(std::move(block));
                }
            }

            if (block_terms.empty()) {
                throw std::runtime_error("failed to build accumulator in BSGS matmul (all-zero matrix)");
            }
            ckks::CtAccurate accumulator = reduce_add_gpu(ctx, block_terms);

            return make_cipher_from_gpu_lazy(ctx, std::move(accumulator), tensor->ciphertext);
        }
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
            if (!diag_nonzero_vec[d]) continue;  // skip zero diagonal

            std::vector<double> diag(slots, 0.0);
            for (std::size_t i = 0; i < slots; ++i) {
                std::int64_t row_signed = static_cast<std::int64_t>(i) - static_cast<std::int64_t>(giant_step);
                std::size_t row = static_cast<std::size_t>((row_signed % static_cast<std::int64_t>(out_features) + static_cast<std::int64_t>(out_features)) % static_cast<std::int64_t>(out_features));
                std::size_t col = (i + j) % in_features;
                diag[i] = matrix[row][col];
            }

            auto term = cc->EvalMult(baby_ciphers[j], make_plaintext(ctx, diag));

            if (!block) {
                block = term;
            } else {
                block = cc->EvalAdd(block, term);
            }
        }

        if (!block) continue;

        if (k > 0) {
            block = cc->EvalAtIndex(block, static_cast<int>(giant_step));
        }

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

void set_plain_cache_limit(
    const std::shared_ptr<GPUContextHandle>& ctx,
    std::size_t limit
) {
    ctx->gpu_plain_cache_limit = limit;
}

void pin_plain_cache(const std::shared_ptr<GPUContextHandle>& ctx) {
    for (auto& kv : ctx->gpu_plain_cache) {
        ctx->gpu_plain_cache_pinned.emplace(kv.first, std::move(kv.second));
    }
    ctx->gpu_plain_cache.clear();
}

void clear_pinned_plain_cache(const std::shared_ptr<GPUContextHandle>& ctx) {
    ctx->gpu_plain_cache_pinned.clear();
}

std::size_t plain_cache_stats_count(const std::shared_ptr<GPUContextHandle>& ctx) {
    return ctx->gpu_plain_cache.size() + ctx->gpu_plain_cache_pinned.size();
}

std::size_t plain_cache_stats_pinned(const std::shared_ptr<GPUContextHandle>& ctx) {
    return ctx->gpu_plain_cache_pinned.size();
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
          py::arg("bsgs_n1") = 0, py::arg("bsgs_n2") = 0,
          py::arg("weight_hash") = 0, py::arg("diag_nonzero") = std::vector<bool>{});
    m.def("poly_eval", &poly_eval_cipher, py::arg("tensor"), py::arg("coeffs"));
    m.def("conjugate", &conjugate_cipher, py::arg("tensor"));
    m.def("bootstrap", &bootstrap_cipher, py::arg("tensor"));
    m.def("cipher_metadata", &cipher_metadata, py::arg("cipher"));
    m.def("is_gpu_available", &is_gpu_available);
    m.def("get_gpu_info", &get_gpu_info);
    m.def("set_plain_cache_limit", &set_plain_cache_limit,
          py::arg("context"), py::arg("limit"));
    m.def("pin_plain_cache", &pin_plain_cache, py::arg("context"));
    m.def("clear_pinned_plain_cache", &clear_pinned_plain_cache, py::arg("context"));
    m.def("plain_cache_stats_count", &plain_cache_stats_count, py::arg("context"));
    m.def("plain_cache_stats_pinned", &plain_cache_stats_pinned, py::arg("context"));
}
