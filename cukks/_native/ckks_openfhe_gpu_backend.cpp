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
#include "math/dftransform.h"
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
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <sstream>
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
    std::unique_ptr<std::map<uint32_t, ckks::EvaluationKey>> gpu_rot_keys_auto;
    
    // Track if GPU is initialized
    bool gpu_initialized = false;
    
    // Crypto parameters for GPU operations
    std::shared_ptr<CryptoParametersCKKSRNS> crypto_params;

    std::string key_tag;

    std::unordered_map<std::uint64_t, ckks::PtAccurate> gpu_plain_cache;
    std::unordered_map<std::uint64_t, ckks::PtAccurate> gpu_plain_cache_pinned;
    std::size_t gpu_plain_cache_limit = 2048;
};

void sync_large_ring_gpu(const std::shared_ptr<GPUContextHandle>& ctx, const char* stage) {
    if (!ctx || !ctx->gpu_context || ctx->gpu_context->GetDegree() <= 32768) {
        return;
    }
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(stage) + ": " + cudaGetErrorString(err));
    }
}

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
    bool force_cpu = false;
    
    // Load to GPU if not already loaded
    void ensureGPU() const {
        if (!gpu_loaded && context->gpu_initialized) {
            gpu_ct = std::make_unique<ckks::CtAccurate>(
                LoadAccurateCiphertext(ciphertext)
            );
            if (context->gpu_context && context->gpu_context->GetDegree() > 32768) {
                cudaError_t err = cudaDeviceSynchronize();
                if (err != cudaSuccess) {
                    throw std::runtime_error(std::string("failed to synchronize fresh ciphertext GPU load: ") + cudaGetErrorString(err));
                }
            }
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

NativeInteger mul_scaling_factor_int(
    const Ciphertext<DCRTPoly>& lhs,
    const Ciphertext<DCRTPoly>& rhs
) {
    const auto plain_mod = lhs->GetCryptoParameters()->GetPlaintextModulus();
    return lhs->GetScalingFactorInt().ModMul(rhs->GetScalingFactorInt(), plain_mod);
}

NativeInteger mul_plain_scaling_factor_int(
    const Ciphertext<DCRTPoly>& lhs,
    ConstPlaintext plaintext
) {
    const auto plain_mod = lhs->GetCryptoParameters()->GetPlaintextModulus();
    return lhs->GetScalingFactorInt().ModMul(plaintext->GetScalingFactorInt(), plain_mod);
}

NativeInteger representative_plain_scaling_factor_int(
    const std::shared_ptr<GPUContextHandle>& ctx,
    uint32_t level
) {
    auto plaintext = ctx->context->MakeCKKSPackedPlaintext(std::vector<double>{0.0}, 1, level);
    return plaintext->GetScalingFactorInt();
}

static ckks::CtAccurate make_fast_rotation_identity_ext(
    const ckks::CtAccurate& ciphertext,
    ckks::Context& gpu_context
) {
    const uint32_t total_limbs =
        ciphertext.ax__.size() / gpu_context.degree__ + gpu_context.param__.num_special_moduli_;

    ckks::CtAccurate ext;
    ext.bx__.resize(gpu_context.degree__ * total_limbs);
    ext.bx__.setConstant(0);
    gpu_context.AddScaledMessageTerm(ext.bx__, ciphertext.bx__);

    ext.ax__.resize(gpu_context.degree__ * total_limbs);
    ext.ax__.setConstant(0);
    gpu_context.AddScaledMessageTerm(ext.ax__, ciphertext.ax__);

    ext.level = ciphertext.level;
    ext.noiseScaleDeg = ciphertext.noiseScaleDeg;
    ext.scalingFactor = ciphertext.scalingFactor;
    return ext;
}

void set_result_scaling_factor_int(
    const std::shared_ptr<GPUCiphertextHandle>& handle,
    const NativeInteger& scaling_factor_int
) {
    handle->ciphertext->SetScalingFactorInt(scaling_factor_int);
}

std::vector<double> build_attention_plain(
    std::size_t slot_count,
    std::uint32_t batch_size,
    std::uint32_t seq_len,
    std::uint32_t embed_dim,
    std::uint32_t query_index,
    double inactive_value,
    double active_value
) {
    const std::size_t slots_per_sample = static_cast<std::size_t>(seq_len) * embed_dim;
    const std::size_t active_size = static_cast<std::size_t>(batch_size) * slots_per_sample;
    const std::size_t query_start = static_cast<std::size_t>(query_index) * embed_dim;
    std::vector<double> plain(slot_count, inactive_value);
    for (std::uint32_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        const std::size_t block_start = static_cast<std::size_t>(batch_idx) * slots_per_sample + query_start;
        for (std::uint32_t offset = 0; offset < embed_dim; ++offset) {
            const std::size_t pos = block_start + offset;
            if (pos < slot_count && pos < active_size) {
                plain[pos] = active_value;
            }
        }
    }
    return plain;
}

std::vector<std::vector<double>> build_repeated_block_diagonals(
    const std::vector<std::vector<double>>& block_weight,
    std::uint32_t num_blocks,
    std::uint32_t total_in,
    std::size_t slot_count
) {
    const std::size_t out_features = block_weight.size();
    const std::size_t in_features = block_weight.front().size();
    std::vector<std::vector<double>> diagonals(total_in, std::vector<double>(slot_count, 0.0));

    for (std::uint32_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
        const std::size_t base_row = static_cast<std::size_t>(block_idx) * out_features;
        const std::size_t base_col = static_cast<std::size_t>(block_idx) * in_features;
        for (std::size_t row_idx = 0; row_idx < out_features; ++row_idx) {
            const std::size_t global_row = base_row + row_idx;
            for (std::size_t col_idx = 0; col_idx < in_features; ++col_idx) {
                const double value = block_weight[row_idx][col_idx];
                if (std::abs(value) <= 1e-30) {
                    continue;
                }
                const std::size_t global_col = base_col + col_idx;
                const std::size_t diagonal_idx = (global_col + total_in - global_row) % total_in;
                diagonals[diagonal_idx][global_row] = value;
            }
        }
    }
    return diagonals;
}

std::shared_ptr<GPUCiphertextHandle> make_cipher(
    const std::shared_ptr<GPUContextHandle>& ctx,
    const Ciphertext<DCRTPoly>& cipher
) {
    auto handle = std::make_shared<GPUCiphertextHandle>();
    handle->context = ctx;
    handle->ciphertext = cipher;
    return handle;
}

void mark_force_cpu(const std::shared_ptr<GPUCiphertextHandle>& handle) {
    if (handle) {
        handle->force_cpu = true;
        handle->gpu_ct.reset();
        handle->gpu_loaded = false;
    }
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

void mod_down_in_place(
    const std::shared_ptr<GPUContextHandle>& ctx,
    ckks::CtAccurate& ciphertext
) {
    ckks::DeviceVector temp_ax(ciphertext.ax__);
    ctx->gpu_context->ModDown(temp_ax, ciphertext.ax__);
    ckks::DeviceVector temp_bx(ciphertext.bx__);
    ctx->gpu_context->ModDown(temp_bx, ciphertext.bx__);
}

ckks::CtAccurate eval_fast_rotate_ext(
    const std::shared_ptr<GPUContextHandle>& ctx,
    const ckks::CtAccurate& ciphertext,
    const ckks::DeviceVector& digits,
    int rotation,
    bool add_first
) {
    const uint32_t auto_index = FindAutomorphismIndex2nComplex(
        rotation,
        ctx->context->GetCyclotomicOrder());

    ckks::CtAccurate inner_prod;
    auto gpu_eval_key_iterator =
        ctx->gpu_rot_keys_auto ? ctx->gpu_rot_keys_auto->find(auto_index) : ctx->gpu_rot_keys_auto->end();

    if (!ctx->gpu_rot_keys_auto || gpu_eval_key_iterator == ctx->gpu_rot_keys_auto->end()) {
        auto eval_keys = ctx->context->GetEvalAutomorphismKeyMap(ctx->key_tag);
        auto eval_key_iterator = eval_keys.find(auto_index);
        if (eval_key_iterator == eval_keys.end()) {
            OPENFHE_THROW(openfhe_error, "EvalKey for index [" + std::to_string(auto_index) + "] is not found.");
        }
        ckks::EvaluationKey gpu_key = LoadRelinKey(eval_key_iterator->second);
        if (ctx->gpu_rot_keys_auto) {
            ctx->gpu_rot_keys_auto->emplace(auto_index, ckks::EvaluationKey(gpu_key));
        }
        ctx->gpu_context->KeySwitch(digits, gpu_key, inner_prod.ax__, inner_prod.bx__);
    } else {
        ctx->gpu_context->KeySwitch(digits, gpu_eval_key_iterator->second, inner_prod.ax__, inner_prod.bx__);
    }

    // GPU CtAccurate stores c0 in bx__ and c1 in ax__.
    // EvalFastRotationExt(addFirst=true) must inject P*c0 only into the first
    // component, matching LeveledSHECKKSRNS::EvalFastRotationExt.
    if (add_first) {
        ctx->gpu_context->AddScaledMessageTerm(inner_prod.bx__, ciphertext.bx__);
    }

    ctx->gpu_context->AutomorphismTransformInPlace(inner_prod, auto_index);
    inner_prod.level = ciphertext.level;
    inner_prod.noiseScaleDeg = ciphertext.noiseScaleDeg;
    inner_prod.scalingFactor = ciphertext.scalingFactor;
    return inner_prod;
}

ckks::CtAccurate eval_rotate_standard_gpu(
    const std::shared_ptr<GPUContextHandle>& ctx,
    const ckks::CtAccurate& ciphertext,
    int rotation
) {
    if (rotation == 0) {
        return ciphertext;
    }

    const uint32_t auto_index = FindAutomorphismIndex2nComplex(
        rotation,
        ctx->context->GetCyclotomicOrder());

    auto gpu_eval_key_iterator =
        ctx->gpu_rot_keys_auto ? ctx->gpu_rot_keys_auto->find(auto_index) : ctx->gpu_rot_keys_auto->end();
    if (!ctx->gpu_rot_keys_auto || gpu_eval_key_iterator == ctx->gpu_rot_keys_auto->end()) {
        auto eval_keys = ctx->context->GetEvalAutomorphismKeyMap(ctx->key_tag);
        auto eval_key_iterator = eval_keys.find(auto_index);
        if (eval_key_iterator == eval_keys.end()) {
            OPENFHE_THROW(openfhe_error, "EvalKey for index [" + std::to_string(auto_index) + "] is not found.");
        }
        ckks::EvaluationKey gpu_key = LoadRelinKey(eval_key_iterator->second);
        if (ctx->gpu_rot_keys_auto) {
            ctx->gpu_rot_keys_auto->emplace(auto_index, ckks::EvaluationKey(gpu_key));
        }
        return ctx->gpu_context->EvalAtIndex(ciphertext, gpu_key, auto_index);
    }

    return ctx->gpu_context->EvalAtIndex(ciphertext, gpu_eval_key_iterator->second, auto_index);
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
        ckks::EvaluationKey gpu_key = LoadRelinKey(it->second);
        ctx->gpu_rot_keys->emplace(rotation, ckks::EvaluationKey(gpu_key));
        if (ctx->gpu_rot_keys_auto) {
            ctx->gpu_rot_keys_auto->emplace(auto_idx, std::move(gpu_key));
        }
        if (ctx->gpu_context && ctx->gpu_context->GetDegree() > 32768) {
            cudaError_t err = cudaDeviceSynchronize();
            if (err != cudaSuccess) {
                throw std::runtime_error(std::string("failed to synchronize GPU rotation key load: ") + cudaGetErrorString(err));
            }
        }
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

bool use_batch_muladd_for_context(const std::shared_ptr<GPUContextHandle>& ctx) {
    return use_batch_muladd() && ctx && ctx->gpu_context && ctx->gpu_context->GetDegree() <= 32768;
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
    auto plaintext = ctx->context->MakeCKKSPackedPlaintext(values, 1, 0);
    plaintext->SetLevel(level);
    plaintext->Encode();
    return LoadAccuratePlaintext(plaintext, plaintext->GetElement<DCRTPoly>());
}

Plaintext make_aux_plaintext_ext_local(
    const std::shared_ptr<GPUContextHandle>& ctx,
    const std::vector<double>& values,
    uint32_t level,
    size_t noiseScaleDeg = 1);

ckks::PtAccurate make_gpu_plaintext_ext(
    const std::shared_ptr<GPUContextHandle>& ctx,
    const std::vector<double>& values,
    uint32_t level
) {
    Plaintext plaintext = make_aux_plaintext_ext_local(ctx, values, level, 1);
    return LoadAccuratePlaintext(plaintext, plaintext->GetElement<DCRTPoly>());
}

Plaintext make_plaintext_ext_cpu(
    const std::shared_ptr<GPUContextHandle>& ctx,
    const std::vector<double>& values,
    uint32_t level
) {
    return make_aux_plaintext_ext_local(ctx, values, level, 1);
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

void fit_to_native_vector_local(uint32_t ringDim, const std::vector<int64_t>& vec, int64_t bigBound,
                                NativeVector* nativeVec) {
    if (nativeVec == nullptr) {
        throw std::runtime_error("fit_to_native_vector_local: native vector is null");
    }
    NativeInteger bigValueHf(static_cast<uint64_t>(bigBound) >> 1);
    NativeInteger modulus(nativeVec->GetModulus());
    NativeInteger diff = NativeInteger(static_cast<uint64_t>(bigBound)) - modulus;
    uint32_t dslots = vec.size();
    uint32_t gap = ringDim / dslots;
    for (usint i = 0; i < vec.size(); i++) {
        NativeInteger n(static_cast<uint64_t>(vec[i]));
        if (n > bigValueHf) {
            (*nativeVec)[gap * i] = n.ModSub(diff, modulus);
        } else {
            (*nativeVec)[gap * i] = n.Mod(modulus);
        }
    }
}

Plaintext make_aux_plaintext_ext_local(
    const std::shared_ptr<GPUContextHandle>& ctx,
    const std::vector<double>& values,
    uint32_t level,
    size_t noiseScaleDeg
) {
    const auto crypto_params = ctx->crypto_params;
    if (!crypto_params) {
        throw std::runtime_error("make_aux_plaintext_ext_local: missing CKKS crypto parameters");
    }

    const auto& q_params_all = crypto_params->GetElementParams()->GetParams();
    const std::size_t active_q = q_params_all.size() - level;
    if (active_q == 0) {
        throw std::runtime_error("make_aux_plaintext_ext_local: no active Q limbs at requested level");
    }

    std::vector<std::shared_ptr<ILNativeParams>> p_params;
    if (auto params_p = crypto_params->GetParamsP()) {
        p_params = params_p->GetParams();
    }

    std::vector<NativeInteger> moduli(active_q + p_params.size());
    std::vector<NativeInteger> roots(active_q + p_params.size());
    for (std::size_t i = 0; i < active_q; ++i) {
        moduli[i] = q_params_all[i]->GetModulus();
        roots[i] = q_params_all[i]->GetRootOfUnity();
    }
    for (std::size_t i = 0; i < p_params.size(); ++i) {
        moduli[active_q + i] = p_params[i]->GetModulus();
        roots[active_q + i] = p_params[i]->GetRootOfUnity();
    }

    auto params_qp = std::make_shared<ILDCRTParams<BigInteger>>(
        ctx->context->GetCyclotomicOrder(), moduli, roots);

    std::vector<std::complex<double>> complex_values(values.begin(), values.end());
    const usint slots = static_cast<usint>(values.size());
    double scFact = crypto_params->GetScalingFactorReal(level);

    Plaintext p = Plaintext(std::make_shared<CKKSPackedEncoding>(
        params_qp, ctx->context->GetEncodingParams(), complex_values, noiseScaleDeg, level, scFact, slots));

    DCRTPoly& plainElement = p->GetElement<DCRTPoly>();
    usint N = ctx->context->GetRingDimension();

    std::vector<std::complex<double>> inverse = complex_values;
    inverse.resize(slots);
    DiscreteFourierTransform::FFTSpecialInv(inverse, N * 2);
    double powP = scFact;

    constexpr int32_t MAX_BITS_IN_WORD = 61;
    int32_t logc = 0;
    for (size_t i = 0; i < slots; ++i) {
        inverse[i] *= powP;
        if (inverse[i].real() != 0) {
            int32_t logci = static_cast<int32_t>(ceil(log2(std::abs(inverse[i].real()))));
            if (logc < logci)
                logc = logci;
        }
        if (inverse[i].imag() != 0) {
            int32_t logci = static_cast<int32_t>(ceil(log2(std::abs(inverse[i].imag()))));
            if (logc < logci)
                logc = logci;
        }
    }
    if (logc < 0) {
        throw std::runtime_error("make_aux_plaintext_ext_local: too small scaling factor");
    }
    int32_t logValid = (logc <= MAX_BITS_IN_WORD) ? logc : MAX_BITS_IN_WORD;
    int32_t logApprox = logc - logValid;
    double approxFactor = pow(2, logApprox);

    std::vector<int64_t> temp(2 * slots);
    for (size_t i = 0; i < slots; ++i) {
        double dre = inverse[i].real() / approxFactor;
        double dim = inverse[i].imag() / approxFactor;
        if (is64BitOverflow(dre) || is64BitOverflow(dim)) {
            throw std::runtime_error("make_aux_plaintext_ext_local: overflow, decrease scaling factor");
        }
        int64_t re = std::llround(dre);
        int64_t im = std::llround(dim);
        temp[i] = (re < 0) ? Max64BitValue() + re : re;
        temp[i + slots] = (im < 0) ? Max64BitValue() + im : im;
    }

    const auto bigParams = plainElement.GetParams();
    const auto& nativeParams = bigParams->GetParams();
    for (size_t i = 0; i < nativeParams.size(); i++) {
        NativeVector nativeVec(N, nativeParams[i]->GetModulus());
        fit_to_native_vector_local(N, temp, Max64BitValue(), &nativeVec);
        NativePoly element = plainElement.GetElementAtIndex(i);
        element.SetValues(nativeVec, Format::COEFFICIENT);
        plainElement.SetElementAtIndex(i, element);
    }

    usint numTowers = nativeParams.size();
    std::vector<DCRTPoly::Integer> crtModuli(numTowers);
    for (usint i = 0; i < numTowers; i++) {
        crtModuli[i] = nativeParams[i]->GetModulus();
    }

    DCRTPoly::Integer intPowP{static_cast<uint64_t>(std::llround(powP))};
    std::vector<DCRTPoly::Integer> crtPowP(numTowers, intPowP);
    auto currPowP = crtPowP;
    for (size_t i = 2; i < noiseScaleDeg; i++) {
        currPowP = CKKSPackedEncoding::CRTMult(currPowP, crtPowP, crtModuli);
    }
    if (noiseScaleDeg > 1) {
        plainElement = plainElement.Times(currPowP);
    }

    if (logApprox > 0) {
        int32_t logStep = (logApprox <= MAX_LOG_STEP) ? logApprox : MAX_LOG_STEP;
        auto intStep = DCRTPoly::Integer(uint64_t(1) << logStep);
        std::vector<DCRTPoly::Integer> crtApprox(numTowers, intStep);
        logApprox -= logStep;
        while (logApprox > 0) {
            logStep = (logApprox <= MAX_LOG_STEP) ? logApprox : MAX_LOG_STEP;
            intStep = DCRTPoly::Integer(uint64_t(1) << logStep);
            std::vector<DCRTPoly::Integer> crtSF(numTowers, intStep);
            crtApprox = CKKSPackedEncoding::CRTMult(crtApprox, crtSF, crtModuli);
            logApprox -= logStep;
        }
        plainElement = plainElement.Times(crtApprox);
    }

    p->SetFormat(Format::EVALUATION);
    p->SetScalingFactor(pow(p->GetScalingFactor(), noiseScaleDeg));
    return p;
}

Plaintext make_aux_plaintext_local_q(
    const std::shared_ptr<GPUContextHandle>& ctx,
    const std::vector<double>& values,
    uint32_t level,
    size_t noiseScaleDeg = 1
) {
    const auto crypto_params = ctx->crypto_params;
    if (!crypto_params) {
        throw std::runtime_error("make_aux_plaintext_local_q: missing CKKS crypto parameters");
    }

    const auto& q_params_all = crypto_params->GetElementParams()->GetParams();
    const std::size_t active_q = q_params_all.size() - level;
    if (active_q == 0) {
        throw std::runtime_error("make_aux_plaintext_local_q: no active Q limbs at requested level");
    }

    std::vector<NativeInteger> moduli(active_q);
    std::vector<NativeInteger> roots(active_q);
    for (std::size_t i = 0; i < active_q; ++i) {
        moduli[i] = q_params_all[i]->GetModulus();
        roots[i] = q_params_all[i]->GetRootOfUnity();
    }

    auto params_q = std::make_shared<ILDCRTParams<BigInteger>>(
        ctx->context->GetCyclotomicOrder(), moduli, roots);

    std::vector<std::complex<double>> complex_values(values.begin(), values.end());
    const usint slots = static_cast<usint>(values.size());
    double scFact = ctx->crypto_params->GetScalingFactorReal(level);
    Plaintext p = Plaintext(std::make_shared<CKKSPackedEncoding>(
        params_q, ctx->context->GetEncodingParams(), complex_values, noiseScaleDeg, level, scFact, slots));

    DCRTPoly& plainElement = p->GetElement<DCRTPoly>();
    usint N = ctx->context->GetRingDimension();
    std::vector<std::complex<double>> inverse = complex_values;
    inverse.resize(slots);
    DiscreteFourierTransform::FFTSpecialInv(inverse, N * 2);
    double powP = scFact;

    constexpr int32_t MAX_BITS_IN_WORD = 61;
    int32_t logc = 0;
    for (size_t i = 0; i < slots; ++i) {
        inverse[i] *= powP;
        if (inverse[i].real() != 0) {
            int32_t logci = static_cast<int32_t>(ceil(log2(std::abs(inverse[i].real()))));
            if (logc < logci)
                logc = logci;
        }
        if (inverse[i].imag() != 0) {
            int32_t logci = static_cast<int32_t>(ceil(log2(std::abs(inverse[i].imag()))));
            if (logc < logci)
                logc = logci;
        }
    }
    if (logc < 0) {
        throw std::runtime_error("make_aux_plaintext_local_q: too small scaling factor");
    }
    int32_t logValid = (logc <= MAX_BITS_IN_WORD) ? logc : MAX_BITS_IN_WORD;
    int32_t logApprox = logc - logValid;
    double approxFactor = pow(2, logApprox);

    std::vector<int64_t> temp(2 * slots);
    for (size_t i = 0; i < slots; ++i) {
        double dre = inverse[i].real() / approxFactor;
        double dim = inverse[i].imag() / approxFactor;
        if (is64BitOverflow(dre) || is64BitOverflow(dim)) {
            throw std::runtime_error("make_aux_plaintext_local_q: overflow, decrease scaling factor");
        }
        int64_t re = std::llround(dre);
        int64_t im = std::llround(dim);
        temp[i] = (re < 0) ? Max64BitValue() + re : re;
        temp[i + slots] = (im < 0) ? Max64BitValue() + im : im;
    }

    const auto bigParams = plainElement.GetParams();
    const auto& nativeParams = bigParams->GetParams();
    for (size_t i = 0; i < nativeParams.size(); i++) {
        NativeVector nativeVec(N, nativeParams[i]->GetModulus());
        fit_to_native_vector_local(N, temp, Max64BitValue(), &nativeVec);
        NativePoly element = plainElement.GetElementAtIndex(i);
        element.SetValues(nativeVec, Format::COEFFICIENT);
        plainElement.SetElementAtIndex(i, element);
    }

    usint numTowers = nativeParams.size();
    std::vector<DCRTPoly::Integer> crtModuli(numTowers);
    for (usint i = 0; i < numTowers; i++) {
        crtModuli[i] = nativeParams[i]->GetModulus();
    }

    DCRTPoly::Integer intPowP{static_cast<uint64_t>(std::llround(powP))};
    std::vector<DCRTPoly::Integer> crtPowP(numTowers, intPowP);
    auto currPowP = crtPowP;
    for (size_t i = 2; i < noiseScaleDeg; i++) {
        currPowP = CKKSPackedEncoding::CRTMult(currPowP, crtPowP, crtModuli);
    }
    if (noiseScaleDeg > 1) {
        plainElement = plainElement.Times(currPowP);
    }

    if (logApprox > 0) {
        int32_t logStep = (logApprox <= MAX_LOG_STEP) ? logApprox : MAX_LOG_STEP;
        auto intStep = DCRTPoly::Integer(uint64_t(1) << logStep);
        std::vector<DCRTPoly::Integer> crtApprox(numTowers, intStep);
        logApprox -= logStep;
        while (logApprox > 0) {
            logStep = (logApprox <= MAX_LOG_STEP) ? logApprox : MAX_LOG_STEP;
            intStep = DCRTPoly::Integer(uint64_t(1) << logStep);
            std::vector<DCRTPoly::Integer> crtSF(numTowers, intStep);
            crtApprox = CKKSPackedEncoding::CRTMult(crtApprox, crtSF, crtModuli);
            logApprox -= logStep;
        }
        plainElement = plainElement.Times(crtApprox);
    }

    p->SetFormat(Format::EVALUATION);
    p->SetScalingFactor(pow(p->GetScalingFactor(), noiseScaleDeg));
    return p;
}

ckks::PtAccurate make_gpu_aux_plaintext(
    const std::shared_ptr<GPUContextHandle>& ctx,
    const std::vector<double>& values,
    uint32_t level
) {
    Plaintext plaintext = make_aux_plaintext_local_q(ctx, values, level, 1);
    return LoadAccuratePlaintext(plaintext, plaintext->GetElement<DCRTPoly>());
}

std::vector<double> build_bsgs_aux_diag(
    const std::vector<std::vector<double>>& matrix,
    std::size_t slots,
    std::size_t diagonal_index,
    std::size_t giant_step
) {
    const std::size_t rows = matrix.size();
    const std::size_t cols = matrix.front().size();
    std::vector<double> diag(slots);
    for (std::size_t k = 0; k < slots; ++k) {
        diag[k] = matrix[k % rows][(k + diagonal_index) % cols];
    }
    if (slots == 0) {
        return diag;
    }
    std::int64_t rot = -static_cast<std::int64_t>(giant_step);
    std::int64_t mod = static_cast<std::int64_t>(slots);
    rot %= mod;
    if (rot < 0) {
        rot += mod;
    }
    if (rot == 0) {
        return diag;
    }
    std::vector<double> out(slots);
    for (std::size_t i = 0; i < slots - static_cast<std::size_t>(rot); ++i) {
        out[i] = diag[i + static_cast<std::size_t>(rot)];
    }
    for (std::size_t i = slots - static_cast<std::size_t>(rot); i < slots; ++i) {
        out[i] = diag[i + static_cast<std::size_t>(rot) - slots];
    }
    return out;
}

py::dict cipher_metadata(const std::shared_ptr<GPUCiphertextHandle>& handle);

std::shared_ptr<ILDCRTParams<BigInteger>> level_params_for_gpu_ct(
    const std::shared_ptr<GPUContextHandle>& ctx,
    const ckks::CtAccurate& gpu_ct
) {
    auto params = ctx->crypto_params;
    auto allParams = params->GetElementParams();
    auto paramsVec = allParams->GetParams();

    if (gpu_ct.level > paramsVec.size()) {
        throw std::runtime_error("level_params_for_gpu_ct: GPU level exceeds available towers");
    }

    size_t numTowers = paramsVec.size() - gpu_ct.level;
    std::vector<std::shared_ptr<lbcrypto::ILNativeParams>> nativeParams;
    nativeParams.reserve(numTowers);
    for (size_t i = 0; i < numTowers; i++) {
        nativeParams.push_back(paramsVec[i]);
    }

    return std::make_shared<ILDCRTParams<BigInteger>>(allParams->GetCyclotomicOrder(), nativeParams);
}

py::dict serialize_poly_head(const DCRTPoly& poly, std::size_t coeff_limit) {
    py::dict out;
    const auto ring_dim = static_cast<std::size_t>(poly.GetRingDimension());
    const auto num_limbs = poly.m_vectors.size();
    out["ring_dim"] = static_cast<std::uint32_t>(ring_dim);
    out["num_limbs"] = static_cast<std::uint32_t>(num_limbs);
    out["format"] = static_cast<int>(poly.GetFormat());

    py::list limbs;
    const auto take = coeff_limit == 0 ? ring_dim : std::min(coeff_limit, ring_dim);
    for (std::size_t limb_idx = 0; limb_idx < num_limbs; ++limb_idx) {
        py::dict limb;
        limb["limb_index"] = static_cast<std::uint32_t>(limb_idx);
        limb["modulus"] = poly.GetParams()->GetParams()[limb_idx]->GetModulus().ToString();
        limb["coeff_count"] = static_cast<std::uint32_t>(ring_dim);

        py::list coeffs;
        for (std::size_t coeff_idx = 0; coeff_idx < take; ++coeff_idx) {
            coeffs.append(poly.m_vectors[limb_idx].m_values->at(coeff_idx).ToString());
        }
        limb["coeffs_head"] = coeffs;
        limbs.append(limb);
    }

    out["limbs"] = limbs;
    return out;
}

py::dict serialize_device_vector_head(
    const ckks::DeviceVector& vec,
    std::size_t ring_dim,
    std::size_t coeff_limit
) {
    py::dict out;
    ckks::HostVector host(vec);
    const auto num_limbs = ring_dim == 0 ? std::size_t{0} : host.size() / ring_dim;
    out["ring_dim"] = static_cast<std::uint32_t>(ring_dim);
    out["num_limbs"] = static_cast<std::uint32_t>(num_limbs);

    py::list limbs;
    const auto take = coeff_limit == 0 ? ring_dim : std::min(coeff_limit, ring_dim);
    for (std::size_t limb_idx = 0; limb_idx < num_limbs; ++limb_idx) {
        py::dict limb;
        limb["limb_index"] = static_cast<std::uint32_t>(limb_idx);
        py::list coeffs;
        for (std::size_t coeff_idx = 0; coeff_idx < take; ++coeff_idx) {
            coeffs.append(std::to_string(host[limb_idx * ring_dim + coeff_idx]));
        }
        limb["coeffs_head"] = coeffs;
        limbs.append(limb);
    }
    out["limbs"] = limbs;
    return out;
}

py::dict serialize_device_vector_flat_head(const ckks::DeviceVector& vec, std::size_t value_limit) {
    py::dict out;
    ckks::HostVector host(vec);
    const auto take = value_limit == 0 ? host.size() : std::min(value_limit, host.size());
    out["size"] = static_cast<std::uint64_t>(host.size());
    py::list values;
    for (std::size_t i = 0; i < take; ++i) {
        values.append(std::to_string(host[i]));
    }
    out["values_head"] = values;
    return out;
}

py::dict diff_device_vector_exact(const ckks::DeviceVector& lhs, const ckks::DeviceVector& rhs, std::size_t value_limit) {
    py::dict out;
    ckks::HostVector lhs_host(lhs);
    ckks::HostVector rhs_host(rhs);
    const auto lhs_size = lhs_host.size();
    const auto rhs_size = rhs_host.size();
    const auto common = std::min(lhs_size, rhs_size);
    bool exact_equal = lhs_size == rhs_size;
    std::size_t mismatches = 0;
    std::size_t first_idx = std::numeric_limits<std::size_t>::max();
    std::string first_lhs;
    std::string first_rhs;

    for (std::size_t i = 0; i < common; ++i) {
        if (lhs_host[i] != rhs_host[i]) {
            exact_equal = false;
            ++mismatches;
            if (first_idx == std::numeric_limits<std::size_t>::max()) {
                first_idx = i;
                first_lhs = std::to_string(lhs_host[i]);
                first_rhs = std::to_string(rhs_host[i]);
            }
        }
    }
    mismatches += (lhs_size > common ? lhs_size - common : 0) + (rhs_size > common ? rhs_size - common : 0);

    out["exact_equal"] = exact_equal;
    out["lhs_size"] = static_cast<std::uint64_t>(lhs_size);
    out["rhs_size"] = static_cast<std::uint64_t>(rhs_size);
    out["mismatches"] = static_cast<std::uint64_t>(mismatches);
    if (first_idx != std::numeric_limits<std::size_t>::max()) {
        py::dict first;
        first["index"] = static_cast<std::uint64_t>(first_idx);
        first["lhs_value"] = first_lhs;
        first["rhs_value"] = first_rhs;
        out["first_mismatch"] = first;
    }
    out["lhs_head"] = serialize_device_vector_flat_head(lhs, value_limit)["values_head"];
    out["rhs_head"] = serialize_device_vector_flat_head(rhs, value_limit)["values_head"];
    return out;
}

DCRTPoly approx_mod_down_like_keyswitch(
    const DCRTPoly& ext_poly,
    const std::shared_ptr<ILDCRTParams<BigInteger>>& paramsQl,
    const std::shared_ptr<CryptoParametersCKKSRNS>& cryptoParams
) {
    PlaintextModulus t = (cryptoParams->GetNoiseScale() == 1) ? 0 : cryptoParams->GetPlaintextModulus();
    return ext_poly.ApproxModDown(
        paramsQl,
        cryptoParams->GetParamsP(),
        cryptoParams->GetPInvModq(),
        cryptoParams->GetPInvModqPrecon(),
        cryptoParams->GetPHatInvModp(),
        cryptoParams->GetPHatInvModpPrecon(),
        cryptoParams->GetPHatModq(),
        cryptoParams->GetModqBarrettMu(),
        cryptoParams->GettInvModp(),
        cryptoParams->GettInvModpPrecon(),
        t,
        cryptoParams->GettModqPrecon());
}

py::dict diff_poly_exact(const DCRTPoly& cpu_poly, const DCRTPoly& gpu_poly, std::size_t coeff_limit) {
    py::dict out;
    const auto cpu_ring_dim = static_cast<std::size_t>(cpu_poly.GetRingDimension());
    const auto gpu_ring_dim = static_cast<std::size_t>(gpu_poly.GetRingDimension());
    const auto cpu_num_limbs = cpu_poly.m_vectors.size();
    const auto gpu_num_limbs = gpu_poly.m_vectors.size();
    const auto common_ring_dim = std::min(cpu_ring_dim, gpu_ring_dim);
    const auto common_limbs = std::min(cpu_num_limbs, gpu_num_limbs);

    bool exact_equal = (cpu_ring_dim == gpu_ring_dim) && (cpu_num_limbs == gpu_num_limbs);
    std::size_t mismatched_limbs = 0;
    std::size_t mismatched_coeffs = 0;
    std::size_t first_limb = std::numeric_limits<std::size_t>::max();
    std::size_t first_coeff = std::numeric_limits<std::size_t>::max();
    std::string first_cpu;
    std::string first_gpu;
    std::string first_cpu_mod;
    std::string first_gpu_mod;

    py::list limb_summaries;
    for (std::size_t limb_idx = 0; limb_idx < common_limbs; ++limb_idx) {
        const auto cpu_mod = cpu_poly.GetParams()->GetParams()[limb_idx]->GetModulus().ToString();
        const auto gpu_mod = gpu_poly.GetParams()->GetParams()[limb_idx]->GetModulus().ToString();
        bool limb_equal = (cpu_mod == gpu_mod);
        std::size_t limb_mismatch_count = 0;
        std::size_t limb_first_coeff = std::numeric_limits<std::size_t>::max();

        for (std::size_t coeff_idx = 0; coeff_idx < common_ring_dim; ++coeff_idx) {
            const auto cpu_val = cpu_poly.m_vectors[limb_idx].m_values->at(coeff_idx).ToString();
            const auto gpu_val = gpu_poly.m_vectors[limb_idx].m_values->at(coeff_idx).ToString();
            if (cpu_val != gpu_val) {
                limb_equal = false;
                ++limb_mismatch_count;
                ++mismatched_coeffs;
                if (limb_first_coeff == std::numeric_limits<std::size_t>::max()) {
                    limb_first_coeff = coeff_idx;
                }
                if (first_limb == std::numeric_limits<std::size_t>::max()) {
                    first_limb = limb_idx;
                    first_coeff = coeff_idx;
                    first_cpu = cpu_val;
                    first_gpu = gpu_val;
                    first_cpu_mod = cpu_mod;
                    first_gpu_mod = gpu_mod;
                }
            }
        }

        if (!limb_equal) {
            exact_equal = false;
            ++mismatched_limbs;
        }

        py::dict limb_summary;
        limb_summary["limb_index"] = static_cast<std::uint32_t>(limb_idx);
        limb_summary["cpu_modulus"] = cpu_mod;
        limb_summary["gpu_modulus"] = gpu_mod;
        limb_summary["equal"] = limb_equal;
        limb_summary["mismatch_count"] = static_cast<std::uint32_t>(limb_mismatch_count);
        if (limb_first_coeff != std::numeric_limits<std::size_t>::max()) {
            limb_summary["first_mismatch_coeff"] = static_cast<std::uint32_t>(limb_first_coeff);
        }

        const auto take = coeff_limit == 0 ? common_ring_dim : std::min(coeff_limit, common_ring_dim);
        py::list cpu_head;
        py::list gpu_head;
        for (std::size_t coeff_idx = 0; coeff_idx < take; ++coeff_idx) {
            cpu_head.append(cpu_poly.m_vectors[limb_idx].m_values->at(coeff_idx).ToString());
            gpu_head.append(gpu_poly.m_vectors[limb_idx].m_values->at(coeff_idx).ToString());
        }
        limb_summary["cpu_coeffs_head"] = cpu_head;
        limb_summary["gpu_coeffs_head"] = gpu_head;
        limb_summaries.append(limb_summary);
    }

    out["exact_equal"] = exact_equal;
    out["cpu_ring_dim"] = static_cast<std::uint32_t>(cpu_ring_dim);
    out["gpu_ring_dim"] = static_cast<std::uint32_t>(gpu_ring_dim);
    out["cpu_num_limbs"] = static_cast<std::uint32_t>(cpu_num_limbs);
    out["gpu_num_limbs"] = static_cast<std::uint32_t>(gpu_num_limbs);
    out["mismatched_limbs"] = static_cast<std::uint32_t>(mismatched_limbs);
    out["mismatched_coeffs"] = static_cast<std::uint64_t>(mismatched_coeffs);
    out["limbs"] = limb_summaries;
    if (first_limb != std::numeric_limits<std::size_t>::max()) {
        py::dict first;
        first["limb_index"] = static_cast<std::uint32_t>(first_limb);
        first["coeff_index"] = static_cast<std::uint32_t>(first_coeff);
        first["cpu_value"] = first_cpu;
        first["gpu_value"] = first_gpu;
        first["cpu_modulus"] = first_cpu_mod;
        first["gpu_modulus"] = first_gpu_mod;
        out["first_mismatch"] = first;
    }
    return out;
}

py::dict rotate_debug_compare(
    const std::shared_ptr<GPUCiphertextHandle>& tensor,
    int rotation,
    std::size_t coeff_limit = 16
) {
    auto ctx = tensor->context;

    if (tensor->gpu_loaded) {
        const_cast<GPUCiphertextHandle*>(tensor.get())->syncFromGPU();
    }
    tensor->ensureGPU();

    auto input_level_params = level_params_for_gpu_ct(ctx, *tensor->gpu_ct);
    auto input_gpu_cpu = tensor->ciphertext->CloneZero();
    LoadCtAccurateFromGPU(input_gpu_cpu, *tensor->gpu_ct, input_level_params);

    if (!ensure_gpu_rotation_key(ctx, rotation)) {
        throw std::runtime_error("rotate_debug_compare: missing GPU rotation key for rotation " + std::to_string(rotation));
    }

    const uint32_t auto_index = FindAutomorphismIndex2nComplex(rotation, ctx->context->GetCyclotomicOrder());
    auto eval_keys = ctx->context->GetEvalAutomorphismKeyMap(ctx->key_tag);
    auto eval_key_it = eval_keys.find(auto_index);
    if (eval_key_it == eval_keys.end()) {
        throw std::runtime_error("rotate_debug_compare: CPU eval key not found for auto index " + std::to_string(auto_index));
    }
    auto eval_key = eval_key_it->second;
    auto algo = ctx->context->GetScheme();
    auto crypto_params_ckks = std::dynamic_pointer_cast<CryptoParametersCKKSRNS>(ctx->context->GetCryptoParameters());
    const auto& cpu_in_elems = tensor->ciphertext->GetElements();

    auto cpu_digits = algo->EvalKeySwitchPrecomputeCore(cpu_in_elems[1], tensor->ciphertext->GetCryptoParameters());
    auto cpu_ks_ext = algo->EvalFastKeySwitchCoreExt(cpu_digits, eval_key, cpu_in_elems[0].GetParams());
    auto cpu_ks = algo->EvalFastKeySwitchCore(cpu_digits, eval_key, cpu_in_elems[0].GetParams());

    DCRTPoly cpu_pre_auto_c0 = (*cpu_ks)[0];
    cpu_pre_auto_c0 += cpu_in_elems[0];
    DCRTPoly cpu_pre_auto_c1 = (*cpu_ks)[1];

    std::vector<usint> auto_map(cpu_in_elems[0].GetRingDimension());
    PrecomputeAutoMap(cpu_in_elems[0].GetRingDimension(), auto_index, &auto_map);
    DCRTPoly cpu_post_auto_c0 = cpu_pre_auto_c0.AutomorphismTransform(auto_index, auto_map);
    DCRTPoly cpu_post_auto_c1 = cpu_pre_auto_c1.AutomorphismTransform(auto_index, auto_map);

    std::unique_ptr<ckks::EvaluationKey> owned_gpu_eval_key;
    const ckks::EvaluationKey* gpu_eval_key = nullptr;
    if (ctx->gpu_rot_keys_auto && ctx->gpu_rot_keys_auto->count(auto_index)) {
        gpu_eval_key = &ctx->gpu_rot_keys_auto->at(auto_index);
    } else {
        owned_gpu_eval_key = std::make_unique<ckks::EvaluationKey>(LoadRelinKey(eval_key));
        gpu_eval_key = owned_gpu_eval_key.get();
    }

    ckks::DeviceVector gpu_digits = ctx->gpu_context->ModUp(tensor->gpu_ct->ax__);
    ckks::DeviceVector gpu_ks_a, gpu_ks_b;
    ctx->gpu_context->KeySwitch(gpu_digits, *gpu_eval_key, gpu_ks_a, gpu_ks_b);

    ckks::DeviceVector gpu_moddown_c1;
    ckks::DeviceVector gpu_moddown_c0;
    ctx->gpu_context->ModDown(gpu_ks_a, gpu_moddown_c1);
    ctx->gpu_context->ModDown(gpu_ks_b, gpu_moddown_c0);

    auto gpu_ks_ext_params = (*cpu_ks_ext)[0].GetParams();
    DCRTPoly gpu_ks_ext_c0 = loadIntoDCRTPoly(gpu_ks_b, gpu_ks_ext_params);
    DCRTPoly gpu_ks_ext_c1 = loadIntoDCRTPoly(gpu_ks_a, gpu_ks_ext_params);

    auto gpu_q_params = (*cpu_ks)[0].GetParams();
    DCRTPoly gpu_moddown_poly_c0 = loadIntoDCRTPoly(gpu_moddown_c0, gpu_q_params);
    DCRTPoly gpu_moddown_poly_c1 = loadIntoDCRTPoly(gpu_moddown_c1, gpu_q_params);
    DCRTPoly cpu_moddown_from_gpu_ext_c0 = approx_mod_down_like_keyswitch(gpu_ks_ext_c0, cpu_in_elems[0].GetParams(), crypto_params_ckks);
    DCRTPoly cpu_moddown_from_gpu_ext_c1 = approx_mod_down_like_keyswitch(gpu_ks_ext_c1, cpu_in_elems[0].GetParams(), crypto_params_ckks);

    ckks::CtAccurate gpu_pre_auto;
    gpu_pre_auto.ax__ = std::move(gpu_moddown_c1);
    gpu_pre_auto.bx__ = std::move(gpu_moddown_c0);
    gpu_pre_auto.level = tensor->gpu_ct->level;
    gpu_pre_auto.noiseScaleDeg = tensor->gpu_ct->noiseScaleDeg;
    gpu_pre_auto.scalingFactor = tensor->gpu_ct->scalingFactor;
    ctx->gpu_context->AddCoreInPlace(gpu_pre_auto.bx__, tensor->gpu_ct->bx__);

    ckks::CtAccurate gpu_post_auto = gpu_pre_auto;
    ctx->gpu_context->AutomorphismTransformInPlace(gpu_post_auto, auto_index);
    sync_large_ring_gpu(ctx, "rotate_debug_compare staged gpu ops");

    auto gpu_pre_auto_params = level_params_for_gpu_ct(ctx, gpu_pre_auto);
    auto gpu_pre_auto_cpu = tensor->ciphertext->CloneZero();
    LoadCtAccurateFromGPU(gpu_pre_auto_cpu, gpu_pre_auto, gpu_pre_auto_params);

    auto gpu_post_auto_params = level_params_for_gpu_ct(ctx, gpu_post_auto);
    auto gpu_post_auto_cpu = tensor->ciphertext->CloneZero();
    LoadCtAccurateFromGPU(gpu_post_auto_cpu, gpu_post_auto, gpu_post_auto_params);

    auto cpu_result = ctx->context->EvalAtIndex(tensor->ciphertext, rotation);
    auto gpu_result = eval_rotate_standard_gpu(ctx, *tensor->gpu_ct, rotation);
    sync_large_ring_gpu(ctx, "rotate_debug_compare gpu rotate");

    auto gpu_level_params = level_params_for_gpu_ct(ctx, gpu_result);
    auto gpu_result_cpu = tensor->ciphertext->CloneZero();
    LoadCtAccurateFromGPU(gpu_result_cpu, gpu_result, gpu_level_params);

    py::dict out;
    out["rotation"] = rotation;
    out["auto_index"] = auto_index;
    out["input_metadata"] = cipher_metadata(tensor);
    out["input_cpu_c0"] = serialize_poly_head(tensor->ciphertext->GetElements()[0], coeff_limit);
    out["input_cpu_c1"] = serialize_poly_head(tensor->ciphertext->GetElements()[1], coeff_limit);
    out["input_gpu_synced_c0"] = serialize_poly_head(input_gpu_cpu->GetElements()[0], coeff_limit);
    out["input_gpu_synced_c1"] = serialize_poly_head(input_gpu_cpu->GetElements()[1], coeff_limit);
    out["input_c0_diff"] = diff_poly_exact(tensor->ciphertext->GetElements()[0], input_gpu_cpu->GetElements()[0], coeff_limit);
    out["input_c1_diff"] = diff_poly_exact(tensor->ciphertext->GetElements()[1], input_gpu_cpu->GetElements()[1], coeff_limit);
    out["gpu_raw_bx"] = serialize_device_vector_head(gpu_result.bx__, ctx->context->GetRingDimension(), coeff_limit);
    out["gpu_raw_ax"] = serialize_device_vector_head(gpu_result.ax__, ctx->context->GetRingDimension(), coeff_limit);
    out["gpu_raw_input_bx"] = serialize_device_vector_head(tensor->gpu_ct->bx__, ctx->context->GetRingDimension(), coeff_limit);
    out["gpu_raw_input_ax"] = serialize_device_vector_head(tensor->gpu_ct->ax__, ctx->context->GetRingDimension(), coeff_limit);

    ckks::DeviceVector cpu_digits_gpu = loadIntoDeviceVector(*cpu_digits);
    out["stage_modup_flat_diff"] = diff_device_vector_exact(cpu_digits_gpu, gpu_digits, coeff_limit * 8);

    out["stage_ks_ext_c0_diff"] = diff_poly_exact((*cpu_ks_ext)[0], gpu_ks_ext_c0, coeff_limit);
    out["stage_ks_ext_c1_diff"] = diff_poly_exact((*cpu_ks_ext)[1], gpu_ks_ext_c1, coeff_limit);
    out["stage_cpu_moddown_replay_c0_diff"] = diff_poly_exact((*cpu_ks)[0], cpu_moddown_from_gpu_ext_c0, coeff_limit);
    out["stage_cpu_moddown_replay_c1_diff"] = diff_poly_exact((*cpu_ks)[1], cpu_moddown_from_gpu_ext_c1, coeff_limit);
    out["stage_moddown_c0_diff"] = diff_poly_exact((*cpu_ks)[0], gpu_moddown_poly_c0, coeff_limit);
    out["stage_moddown_c1_diff"] = diff_poly_exact((*cpu_ks)[1], gpu_moddown_poly_c1, coeff_limit);
    out["stage_moddown_cross_c0_to_gpu_c1_diff"] = diff_poly_exact((*cpu_ks)[0], gpu_moddown_poly_c1, coeff_limit);
    out["stage_moddown_cross_c1_to_gpu_c0_diff"] = diff_poly_exact((*cpu_ks)[1], gpu_moddown_poly_c0, coeff_limit);
    out["stage_pre_auto_c0_diff"] = diff_poly_exact(cpu_pre_auto_c0, gpu_pre_auto_cpu->GetElements()[0], coeff_limit);
    out["stage_pre_auto_c1_diff"] = diff_poly_exact(cpu_pre_auto_c1, gpu_pre_auto_cpu->GetElements()[1], coeff_limit);
    out["stage_post_auto_c0_diff"] = diff_poly_exact(cpu_post_auto_c0, gpu_post_auto_cpu->GetElements()[0], coeff_limit);
    out["stage_post_auto_c1_diff"] = diff_poly_exact(cpu_post_auto_c1, gpu_post_auto_cpu->GetElements()[1], coeff_limit);

    py::dict cpu_meta;
    cpu_meta["level"] = cpu_result->GetLevel();
    cpu_meta["scale"] = cpu_result->GetScalingFactor();
    cpu_meta["noise_scale"] = cpu_result->GetNoiseScaleDeg();
    cpu_meta["ring_dim"] = static_cast<std::uint32_t>(cpu_result->GetElements()[0].GetRingDimension());
    cpu_meta["num_limbs"] = static_cast<std::uint32_t>(cpu_result->GetElements()[0].m_vectors.size());
    out["cpu_metadata"] = cpu_meta;

    py::dict gpu_meta;
    gpu_meta["level"] = gpu_result.level;
    gpu_meta["scale"] = gpu_result.scalingFactor;
    gpu_meta["noise_scale"] = gpu_result.noiseScaleDeg;
    gpu_meta["gpu_ax_size"] = static_cast<std::uint32_t>(gpu_result.ax__.size());
    gpu_meta["gpu_bx_size"] = static_cast<std::uint32_t>(gpu_result.bx__.size());
    gpu_meta["gpu_num_limbs"] = static_cast<std::uint32_t>(gpu_result.ax__.size() / ctx->context->GetRingDimension());
    gpu_meta["synced_num_limbs"] = static_cast<std::uint32_t>(gpu_result_cpu->GetElements()[0].m_vectors.size());
    out["gpu_metadata"] = gpu_meta;

    out["cpu_c0"] = serialize_poly_head(cpu_result->GetElements()[0], coeff_limit);
    out["cpu_c1"] = serialize_poly_head(cpu_result->GetElements()[1], coeff_limit);
    out["gpu_c0"] = serialize_poly_head(gpu_result_cpu->GetElements()[0], coeff_limit);
    out["gpu_c1"] = serialize_poly_head(gpu_result_cpu->GetElements()[1], coeff_limit);
    out["c0_diff"] = diff_poly_exact(cpu_result->GetElements()[0], gpu_result_cpu->GetElements()[0], coeff_limit);
    out["c1_diff"] = diff_poly_exact(cpu_result->GetElements()[1], gpu_result_cpu->GetElements()[1], coeff_limit);
    return out;
}

py::dict fast_rotate_ext_debug_compare(
    const std::shared_ptr<GPUCiphertextHandle>& tensor,
    int rotation,
    bool add_first,
    std::size_t coeff_limit = 16
) {
    auto ctx = tensor->context;
    if (tensor->gpu_loaded) {
        const_cast<GPUCiphertextHandle*>(tensor.get())->syncFromGPU();
    }
    tensor->ensureGPU();

    if (!ensure_gpu_rotation_key(ctx, rotation)) {
        throw std::runtime_error("fast_rotate_ext_debug_compare: missing GPU rotation key for rotation " + std::to_string(rotation));
    }

    auto cpu_digits = ctx->context->EvalFastRotationPrecompute(tensor->ciphertext);
    auto cpu_result = ctx->context->EvalFastRotationExt(tensor->ciphertext, rotation, cpu_digits, add_first);

    ckks::DeviceVector gpu_digits = ctx->gpu_context->ModUp(tensor->gpu_ct->ax__);
    auto gpu_result = eval_fast_rotate_ext(ctx, *tensor->gpu_ct, gpu_digits, rotation, add_first);
    sync_large_ring_gpu(ctx, "fast_rotate_ext_debug_compare gpu");

    auto cpu_params_c0 = cpu_result->GetElements()[0].GetParams();
    auto cpu_params_c1 = cpu_result->GetElements()[1].GetParams();
    DCRTPoly gpu_c0 = loadIntoDCRTPoly(gpu_result.bx__, cpu_params_c0);
    DCRTPoly gpu_c1 = loadIntoDCRTPoly(gpu_result.ax__, cpu_params_c1);

    py::dict out;
    out["rotation"] = rotation;
    out["add_first"] = add_first;
    py::dict cpu_meta;
    cpu_meta["level"] = cpu_result->GetLevel();
    cpu_meta["scale"] = cpu_result->GetScalingFactor();
    cpu_meta["noise_scale"] = cpu_result->GetNoiseScaleDeg();
    cpu_meta["num_limbs"] = static_cast<std::uint32_t>(cpu_result->GetElements()[0].m_vectors.size());
    out["cpu_metadata"] = cpu_meta;
    py::dict gpu_meta;
    gpu_meta["level"] = gpu_result.level;
    gpu_meta["scale"] = gpu_result.scalingFactor;
    gpu_meta["noise_scale"] = gpu_result.noiseScaleDeg;
    gpu_meta["num_limbs"] = static_cast<std::uint32_t>(gpu_result.bx__.size() / ctx->context->GetRingDimension());
    out["gpu_metadata"] = gpu_meta;
    out["c0_diff"] = diff_poly_exact(cpu_result->GetElements()[0], gpu_c0, coeff_limit);
    out["c1_diff"] = diff_poly_exact(cpu_result->GetElements()[1], gpu_c1, coeff_limit);
    return out;
}

py::dict bsgs_block_debug_compare(
    const std::shared_ptr<GPUCiphertextHandle>& tensor,
    const std::vector<std::vector<double>>& matrix,
    std::uint32_t bsgs_n1 = 0,
    std::uint32_t bsgs_n2 = 0,
    std::size_t coeff_limit = 8
) {
    auto ctx = tensor->context;
    if (tensor->gpu_loaded) {
        const_cast<GPUCiphertextHandle*>(tensor.get())->syncFromGPU();
    }
    tensor->ensureGPU();

    const auto cc = ctx->context;
    const std::size_t out_features = matrix.size();
    const std::size_t in_features = matrix.empty() ? 0 : matrix.front().size();
    const std::size_t slots = static_cast<std::size_t>(cc->GetRingDimension() / 2);
    if (in_features == 0 || out_features == 0) {
        throw std::runtime_error("bsgs_block_debug_compare: empty matrix");
    }
    if (bsgs_n1 == 0) {
        bsgs_n1 = static_cast<std::uint32_t>(std::ceil(std::sqrt(static_cast<double>(in_features))));
    }
    if (bsgs_n2 == 0) {
        bsgs_n2 = (static_cast<std::uint32_t>(in_features) + bsgs_n1 - 1) / bsgs_n1;
    }

    std::vector<bool> diag_nonzero_vec(in_features, false);
    for (std::size_t d = 0; d < in_features; ++d) {
        int baby_step = static_cast<int>(d % bsgs_n1);
        int giant_step = static_cast<int>((d / bsgs_n1) * bsgs_n1);
        if (baby_step != 0 && !ensure_gpu_rotation_key(ctx, baby_step)) {
            throw std::runtime_error("bsgs_block_debug_compare: missing baby-step rotation key " + std::to_string(baby_step));
        }
        if (giant_step != 0 && !ensure_gpu_rotation_key(ctx, giant_step)) {
            throw std::runtime_error("bsgs_block_debug_compare: missing giant-step rotation key " + std::to_string(giant_step));
        }
        for (std::size_t i = 0; i < slots; ++i) {
            std::int64_t row_signed = static_cast<std::int64_t>(i) - static_cast<std::int64_t>(giant_step);
            std::size_t row = static_cast<std::size_t>((row_signed % static_cast<std::int64_t>(out_features) + static_cast<std::int64_t>(out_features)) % static_cast<std::int64_t>(out_features));
            std::size_t col = (i + static_cast<std::size_t>(baby_step)) % in_features;
            if (matrix[row][col] != 0.0) {
                diag_nonzero_vec[d] = true;
                break;
            }
        }
    }

    auto cpu_precomp = cc->EvalFastRotationPrecompute(tensor->ciphertext);
    auto eval_keys = cc->GetEvalAutomorphismKeyMap(ctx->key_tag);
    std::vector<Ciphertext<DCRTPoly>> cpu_babies;
    cpu_babies.reserve(bsgs_n1);
    for (std::uint32_t j = 0; j < bsgs_n1 && j < in_features; ++j) {
        if (j == 0) {
            cpu_babies.push_back(cc->KeySwitchExt(tensor->ciphertext, true));
        } else {
            cpu_babies.push_back(cc->GetScheme()->EvalFastRotationExt(tensor->ciphertext, j, cpu_precomp, true, eval_keys));
        }
    }

    std::vector<bool> active_baby_steps(bsgs_n1, false);
    active_baby_steps[0] = true;
    for (std::size_t d = 1; d < in_features; ++d) {
        if (diag_nonzero_vec[d]) {
            active_baby_steps[d % bsgs_n1] = true;
        }
    }
    ckks::DeviceVector base_digits = ctx->gpu_context->ModUp(tensor->gpu_ct->ax__);
    std::vector<std::unique_ptr<ckks::CtAccurate>> fast_rotations(bsgs_n1);
    for (std::uint32_t j = 0; j < bsgs_n1 && j < in_features; ++j) {
        if (!active_baby_steps[j]) {
            continue;
        }
        if (j == 0) {
            fast_rotations[j] = std::make_unique<ckks::CtAccurate>(make_fast_rotation_identity_ext(*tensor->gpu_ct, *ctx->gpu_context));
        } else {
            fast_rotations[j] = std::make_unique<ckks::CtAccurate>(eval_fast_rotate_ext(ctx, *tensor->gpu_ct, base_digits, static_cast<int>(j), true));
        }
    }
    sync_large_ring_gpu(ctx, "bsgs_block_debug_compare setup");

    py::list blocks;
    for (std::uint32_t k = 0; k < bsgs_n2; ++k) {
        std::uint32_t giant_step = k * bsgs_n1;
        std::vector<std::uint32_t> active_js;
            std::vector<std::vector<double>> diags;
            std::vector<Plaintext> cpu_pts_ext;
            std::vector<ckks::PtAccurate> pt_terms_ext;
            active_js.reserve(bsgs_n1);
            diags.reserve(bsgs_n1);
            cpu_pts_ext.reserve(bsgs_n1);
            pt_terms_ext.reserve(bsgs_n1);

        for (std::uint32_t j = 0; j < bsgs_n1; ++j) {
            std::uint32_t d = giant_step + j;
            if (d >= in_features) {
                break;
            }
            if (!diag_nonzero_vec[d]) {
                continue;
            }
            std::vector<double> diag(slots, 0.0);
            for (std::size_t i = 0; i < slots; ++i) {
                std::int64_t row_signed = static_cast<std::int64_t>(i) - static_cast<std::int64_t>(giant_step);
                std::size_t row = static_cast<std::size_t>((row_signed % static_cast<std::int64_t>(out_features) + static_cast<std::int64_t>(out_features)) % static_cast<std::int64_t>(out_features));
                std::size_t col = (i + j) % in_features;
                diag[i] = matrix[row][col];
            }
            active_js.push_back(j);
            diags.push_back(diag);
            cpu_pts_ext.push_back(make_plaintext_ext_cpu(ctx, diag, tensor->ciphertext->GetLevel()));
            pt_terms_ext.push_back(make_gpu_plaintext_ext(ctx, diag, tensor->gpu_ct->level));
        }

        if (active_js.empty()) {
            continue;
        }

        Ciphertext<DCRTPoly> cpu_inner_ext;
        for (std::size_t idx = 0; idx < active_js.size(); ++idx) {
            auto term = cpu_babies[active_js[idx]]->Clone();
            auto pt_poly = cpu_pts_ext[idx]->GetElement<DCRTPoly>();
            pt_poly.SetFormat(Format::EVALUATION);
            auto& elems = term->GetElements();
            for (auto& c : elems) {
                c *= pt_poly;
            }
            term->SetNoiseScaleDeg(term->GetNoiseScaleDeg() + cpu_pts_ext[idx]->GetNoiseScaleDeg());
            term->SetScalingFactor(term->GetScalingFactor() * cpu_pts_ext[idx]->GetScalingFactor());
            if (cpu_inner_ext) {
                auto accum = cpu_inner_ext;
                auto& accum_elems = accum->GetElements();
                const auto& term_elems = term->GetElements();
                for (std::size_t el = 0; el < accum_elems.size(); ++el) {
                    accum_elems[el] += term_elems[el];
                }
                cpu_inner_ext = accum;
            } else {
                cpu_inner_ext = term;
            }
        }

        ckks::CtAccurate inner = ctx->gpu_context->EvalMultPlainExt(*fast_rotations[active_js[0]], pt_terms_ext[0]);
        for (std::size_t idx = 1; idx < active_js.size(); ++idx) {
            ckks::CtAccurate to_add = ctx->gpu_context->EvalMultPlainExt(*fast_rotations[active_js[idx]], pt_terms_ext[idx]);
            ctx->gpu_context->EvalAddInPlaceExt(inner, to_add);
        }

        DCRTPoly gpu_inner_c0 = loadIntoDCRTPoly(inner.bx__, cpu_inner_ext->GetElements()[0].GetParams());
        DCRTPoly gpu_inner_c1 = loadIntoDCRTPoly(inner.ax__, cpu_inner_ext->GetElements()[1].GetParams());

        ckks::CtAccurate gpu_block;
        gpu_block.level = tensor->gpu_ct->level;
        gpu_block.noiseScaleDeg = tensor->gpu_ct->noiseScaleDeg;
        gpu_block.scalingFactor = tensor->gpu_ct->scalingFactor;

        if (k == 0) {
            ckks::DeviceVector first_term;
            ctx->gpu_context->ModDown(inner.bx__, first_term);
            ctx->gpu_context->ModDown(inner.ax__, gpu_block.ax__);
            gpu_block.bx__.resize(first_term.size());
            gpu_block.bx__.setConstant(0);
            ctx->gpu_context->AddCoreInPlace(gpu_block.bx__, first_term);
        } else {
            mod_down_in_place(ctx, inner);
            const uint32_t auto_index = FindAutomorphismIndex2nComplex(static_cast<int>(giant_step), ctx->context->GetCyclotomicOrder());
            ckks::DeviceVector inner_b_rot = ctx->gpu_context->AutomorphismTransform(inner.bx__, auto_index);
            ckks::DeviceVector inner_digits = ctx->gpu_context->ModUp(inner.ax__);
            ckks::CtAccurate inner_rot = eval_fast_rotate_ext(ctx, inner, inner_digits, static_cast<int>(giant_step), false);
            ctx->gpu_context->ModDown(inner_rot.ax__, gpu_block.ax__);
            ctx->gpu_context->ModDown(inner_rot.bx__, gpu_block.bx__);
            ctx->gpu_context->AddCoreInPlace(gpu_block.bx__, inner_b_rot);
        }
        Ciphertext<DCRTPoly> cpu_block;
        if (k == 0) {
            auto first = cc->KeySwitchDownFirstElement(cpu_inner_ext);
            auto elems = cpu_inner_ext->GetElements();
            elems[0].SetValuesToZero();
            cpu_inner_ext->SetElements(elems);
            cpu_block = cc->KeySwitchDown(cpu_inner_ext);
            auto final_elems = cpu_block->GetElements();
            final_elems[0] += first;
            cpu_block->SetElements(final_elems);
        } else {
            auto inner_std = cc->KeySwitchDown(cpu_inner_ext);
            usint autoIndex = FindAutomorphismIndex2nComplex(giant_step, cc->GetCyclotomicOrder());
            std::vector<usint> map(cc->GetRingDimension());
            PrecomputeAutoMap(cc->GetRingDimension(), autoIndex, &map);
            auto firstCurrent = inner_std->GetElements()[0].AutomorphismTransform(autoIndex, map);
            auto innerDigits = cc->EvalFastRotationPrecompute(inner_std);
            auto outer_piece = cc->GetScheme()->EvalFastRotationExt(inner_std, giant_step, innerDigits, false, eval_keys);
            auto outer_std = cc->KeySwitchDown(outer_piece);
            auto final_elems = outer_std->GetElements();
            final_elems[0] += firstCurrent;
            outer_std->SetElements(final_elems);
            cpu_block = outer_std;
        }

        sync_large_ring_gpu(ctx, "bsgs_block_debug_compare block");

        auto gpu_params = level_params_for_gpu_ct(ctx, gpu_block);
        auto gpu_block_cpu = tensor->ciphertext->CloneZero();
        LoadCtAccurateFromGPU(gpu_block_cpu, gpu_block, gpu_params);

        py::dict block_info;
        block_info["k"] = k;
        block_info["giant_step"] = giant_step;
        block_info["active_js"] = py::cast(active_js);
        py::dict cpu_meta;
        cpu_meta["level"] = cpu_block->GetLevel();
        cpu_meta["scale"] = cpu_block->GetScalingFactor();
        cpu_meta["noise_scale"] = cpu_block->GetNoiseScaleDeg();
        cpu_meta["scaling_factor_int"] = cpu_block->GetScalingFactorInt().ToString();
        block_info["cpu_metadata"] = cpu_meta;
        py::dict gpu_meta;
        gpu_meta["level"] = gpu_block.level;
        gpu_meta["scale"] = gpu_block.scalingFactor;
        gpu_meta["noise_scale"] = gpu_block.noiseScaleDeg;
        block_info["inner_c0_diff"] = diff_poly_exact(cpu_inner_ext->GetElements()[0], gpu_inner_c0, coeff_limit);
        block_info["inner_c1_diff"] = diff_poly_exact(cpu_inner_ext->GetElements()[1], gpu_inner_c1, coeff_limit);
        block_info["c0_diff"] = diff_poly_exact(cpu_block->GetElements()[0], gpu_block_cpu->GetElements()[0], coeff_limit);
        block_info["c1_diff"] = diff_poly_exact(cpu_block->GetElements()[1], gpu_block_cpu->GetElements()[1], coeff_limit);
        gpu_block_cpu->SetScalingFactorInt(cpu_block->GetScalingFactorInt());
        gpu_meta["synced_scaling_factor_int"] = gpu_block_cpu->GetScalingFactorInt().ToString();
        block_info["gpu_metadata"] = gpu_meta;
        blocks.append(block_info);
    }

    py::dict out;
    out["blocks"] = blocks;
    out["bsgs_n1"] = bsgs_n1;
    out["bsgs_n2"] = bsgs_n2;
    return out;
}

py::dict ext_mult_term_debug_compare(
    const std::shared_ptr<GPUCiphertextHandle>& tensor,
    const std::vector<double>& diag,
    int rotation,
    bool add_first,
    std::size_t coeff_limit = 8
) {
    auto ctx = tensor->context;
    if (tensor->gpu_loaded) {
        const_cast<GPUCiphertextHandle*>(tensor.get())->syncFromGPU();
    }
    tensor->ensureGPU();

    if (rotation != 0 && !ensure_gpu_rotation_key(ctx, rotation)) {
        throw std::runtime_error("ext_mult_term_debug_compare: missing rotation key " + std::to_string(rotation));
    }

    auto cpu_digits = ctx->context->EvalFastRotationPrecompute(tensor->ciphertext);
    auto eval_keys = ctx->context->GetEvalAutomorphismKeyMap(ctx->key_tag);
    auto cpu_fast = ctx->context->GetScheme()->EvalFastRotationExt(tensor->ciphertext, rotation, cpu_digits, add_first, eval_keys);
    auto cpu_pt = make_plaintext_ext_cpu(ctx, diag, tensor->ciphertext->GetLevel());
    auto cpu_term = cpu_fast->Clone();
    auto pt_poly = cpu_pt->GetElement<DCRTPoly>();
    pt_poly.SetFormat(Format::EVALUATION);
    auto& cpu_elems = cpu_term->GetElements();
    for (auto& c : cpu_elems) {
        c *= pt_poly;
    }
    cpu_term->SetNoiseScaleDeg(cpu_term->GetNoiseScaleDeg() + cpu_pt->GetNoiseScaleDeg());
    cpu_term->SetScalingFactor(cpu_term->GetScalingFactor() * cpu_pt->GetScalingFactor());

    ckks::DeviceVector gpu_digits = ctx->gpu_context->ModUp(tensor->gpu_ct->ax__);
    auto gpu_fast = eval_fast_rotate_ext(ctx, *tensor->gpu_ct, gpu_digits, rotation, add_first);
    auto gpu_pt = make_gpu_plaintext_ext(ctx, diag, tensor->gpu_ct->level);
    auto gpu_term = ctx->gpu_context->EvalMultPlainExt(gpu_fast, gpu_pt);
    sync_large_ring_gpu(ctx, "ext_mult_term_debug_compare");

    DCRTPoly gpu_c0 = loadIntoDCRTPoly(gpu_term.bx__, cpu_term->GetElements()[0].GetParams());
    DCRTPoly gpu_c1 = loadIntoDCRTPoly(gpu_term.ax__, cpu_term->GetElements()[1].GetParams());

    py::dict out;
    out["rotation"] = rotation;
    out["add_first"] = add_first;
    out["c0_diff"] = diff_poly_exact(cpu_term->GetElements()[0], gpu_c0, coeff_limit);
    out["c1_diff"] = diff_poly_exact(cpu_term->GetElements()[1], gpu_c1, coeff_limit);
    return out;
}

py::dict cpu_ext_term_vs_standard_debug(
    const std::shared_ptr<GPUCiphertextHandle>& tensor,
    const std::vector<double>& diag,
    int rotation,
    std::size_t coeff_limit = 8
) {
    auto ctx = tensor->context;
    if (tensor->gpu_loaded) {
        const_cast<GPUCiphertextHandle*>(tensor.get())->syncFromGPU();
    }

    auto cpu_std_precomp = ctx->context->EvalFastRotationPrecompute(tensor->ciphertext);
    auto cpu_std_rot = ctx->context->EvalFastRotation(tensor->ciphertext, rotation, ctx->context->GetCyclotomicOrder(), cpu_std_precomp);
    auto cpu_std_pt = make_plaintext(ctx, diag, tensor->ciphertext->GetLevel());
    auto cpu_std_term = ctx->context->EvalMult(cpu_std_rot, cpu_std_pt);

    auto cpu_ext_digits = ctx->context->EvalFastRotationPrecompute(tensor->ciphertext);
    auto eval_keys = ctx->context->GetEvalAutomorphismKeyMap(ctx->key_tag);
    auto cpu_ext_rot = ctx->context->GetScheme()->EvalFastRotationExt(tensor->ciphertext, rotation, cpu_ext_digits, true, eval_keys);
    auto cpu_ext_pt = make_plaintext_ext_cpu(ctx, diag, tensor->ciphertext->GetLevel());
    auto cpu_ext_term = cpu_ext_rot->Clone();
    auto pt_poly = cpu_ext_pt->GetElement<DCRTPoly>();
    pt_poly.SetFormat(Format::EVALUATION);
    auto& ext_elems = cpu_ext_term->GetElements();
    for (auto& c : ext_elems) {
        c *= pt_poly;
    }
    cpu_ext_term->SetNoiseScaleDeg(cpu_ext_term->GetNoiseScaleDeg() + cpu_ext_pt->GetNoiseScaleDeg());
    cpu_ext_term->SetScalingFactor(cpu_ext_term->GetScalingFactor() * cpu_ext_pt->GetScalingFactor());

    auto first = ctx->context->KeySwitchDownFirstElement(cpu_ext_term);
    auto ext_term_elems = cpu_ext_term->GetElements();
    ext_term_elems[0].SetValuesToZero();
    cpu_ext_term->SetElements(ext_term_elems);
    auto cpu_ext_std = ctx->context->KeySwitchDown(cpu_ext_term);
    auto final_elems = cpu_ext_std->GetElements();
    final_elems[0] += first;
    cpu_ext_std->SetElements(final_elems);

    py::dict out;
    out["rotation"] = rotation;
    out["c0_diff"] = diff_poly_exact(cpu_std_term->GetElements()[0], cpu_ext_std->GetElements()[0], coeff_limit);
    out["c1_diff"] = diff_poly_exact(cpu_std_term->GetElements()[1], cpu_ext_std->GetElements()[1], coeff_limit);
    return out;
}

py::dict rescale_debug_compare(
    const std::shared_ptr<GPUCiphertextHandle>& tensor,
    std::size_t coeff_limit = 8
) {
    auto ctx = tensor->context;
    if (tensor->gpu_loaded) {
        const_cast<GPUCiphertextHandle*>(tensor.get())->syncFromGPU();
    }
    tensor->ensureGPU();

    auto cpu_rescaled = ctx->context->ModReduce(tensor->ciphertext);
    ckks::CtAccurate gpu_rescaled = ctx->gpu_context->Rescale(*tensor->gpu_ct);
    sync_large_ring_gpu(ctx, "rescale_debug_compare after full rescale");
    ckks::DeviceVector gpu_c0_switched, gpu_c1_switched;
    ckks::DeviceVector gpu_c0_scaled, gpu_c1_scaled;
    ctx->gpu_context->RescaleSwitchOnly(tensor->gpu_ct->bx__, gpu_c0_switched);
    ctx->gpu_context->RescaleSwitchOnly(tensor->gpu_ct->ax__, gpu_c1_switched);
    ctx->gpu_context->RescaleLowerScaleOnly(tensor->gpu_ct->bx__, gpu_c0_scaled);
    ctx->gpu_context->RescaleLowerScaleOnly(tensor->gpu_ct->ax__, gpu_c1_scaled);
    sync_large_ring_gpu(ctx, "rescale_debug_compare");

    auto cryptoParams = std::dynamic_pointer_cast<CryptoParametersCKKSRNS>(tensor->ciphertext->GetCryptoParameters());
    const auto& cv = tensor->ciphertext->GetElements();
    size_t sizeQ = cryptoParams->GetElementParams()->GetParams().size();
    size_t sizeQl = cv[0].GetNumOfElements();
    size_t diffQl = sizeQ - sizeQl;
    const auto& qlQlInv = cryptoParams->GetQlQlInvModqlDivqlModq(diffQl);
    const auto& qlInv = cryptoParams->GetqlInvModq(diffQl);

    auto build_cpu_stage = [&](const DCRTPoly& poly) {
        auto lastPoly = poly.GetElementAtIndex(poly.GetNumOfElements() - 1);
        lastPoly.SetFormat(Format::COEFFICIENT);
        DCRTPoly lower = poly.Clone();
        lower.DropLastElement();
        DCRTPoly switched(lower.GetParams(), Format::COEFFICIENT, true);
        for (size_t i = 0; i < lower.GetNumOfElements(); ++i) {
            auto tmp = lastPoly;
            tmp.SwitchModulus(lower.GetElementAtIndex(i).GetModulus(), lower.GetElementAtIndex(i).GetRootOfUnity(), 0, 0);
            tmp *= qlQlInv[i];
            switched.SetElementAtIndex(i, std::move(tmp));
            auto cur = lower.GetElementAtIndex(i);
            cur *= qlInv[i];
            lower.SetElementAtIndex(i, std::move(cur));
        }
        return std::pair<DCRTPoly, DCRTPoly>(std::move(switched), std::move(lower));
    };

    auto [cpu_c0_switched, cpu_c0_scaled] = build_cpu_stage(cv[0]);
    auto [cpu_c1_switched, cpu_c1_scaled] = build_cpu_stage(cv[1]);

    auto lowerParamsC0 = cpu_rescaled->GetElements()[0].GetParams();
    auto lowerParamsC1 = cpu_rescaled->GetElements()[1].GetParams();
    DCRTPoly gpu_c0_switched_poly = loadIntoDCRTPoly(gpu_c0_switched, lowerParamsC0, Format::COEFFICIENT);
    DCRTPoly gpu_c1_switched_poly = loadIntoDCRTPoly(gpu_c1_switched, lowerParamsC1, Format::COEFFICIENT);
    DCRTPoly gpu_c0_scaled_poly = loadIntoDCRTPoly(gpu_c0_scaled, lowerParamsC0, Format::EVALUATION);
    DCRTPoly gpu_c1_scaled_poly = loadIntoDCRTPoly(gpu_c1_scaled, lowerParamsC1, Format::EVALUATION);

    auto cpu_template = cpu_rescaled->CloneZero();
    auto gpu_params = level_params_for_gpu_ct(ctx, gpu_rescaled);
    LoadCtAccurateFromGPU(cpu_template, gpu_rescaled, gpu_params);

    py::dict out;
    py::dict cpu_meta;
    cpu_meta["level"] = cpu_rescaled->GetLevel();
    cpu_meta["scale"] = cpu_rescaled->GetScalingFactor();
    cpu_meta["noise_scale"] = cpu_rescaled->GetNoiseScaleDeg();
    cpu_meta["scaling_factor_int"] = cpu_rescaled->GetScalingFactorInt().ToString();
    out["cpu_metadata"] = cpu_meta;
    py::dict gpu_meta;
    gpu_meta["level"] = gpu_rescaled.level;
    gpu_meta["scale"] = gpu_rescaled.scalingFactor;
    gpu_meta["noise_scale"] = gpu_rescaled.noiseScaleDeg;
    gpu_meta["gpu_num_limbs"] = static_cast<std::uint32_t>(gpu_rescaled.ax__.size() / ctx->context->GetRingDimension());
    gpu_meta["synced_scaling_factor_int"] = cpu_template->GetScalingFactorInt().ToString();
    out["gpu_metadata"] = gpu_meta;
    out["c0_switched_diff"] = diff_poly_exact(cpu_c0_switched, gpu_c0_switched_poly, coeff_limit);
    out["c1_switched_diff"] = diff_poly_exact(cpu_c1_switched, gpu_c1_switched_poly, coeff_limit);
    out["c0_scaled_diff"] = diff_poly_exact(cpu_c0_scaled, gpu_c0_scaled_poly, coeff_limit);
    out["c1_scaled_diff"] = diff_poly_exact(cpu_c1_scaled, gpu_c1_scaled_poly, coeff_limit);
    out["c0_diff"] = diff_poly_exact(cpu_rescaled->GetElements()[0], cpu_template->GetElements()[0], coeff_limit);
    out["c1_diff"] = diff_poly_exact(cpu_rescaled->GetElements()[1], cpu_template->GetElements()[1], coeff_limit);
    return out;
}

py::dict cipher_metadata(const std::shared_ptr<GPUCiphertextHandle>& handle) {
    py::dict meta;
    auto cc = handle->context->context;
    const auto ring_dim = static_cast<std::uint32_t>(cc->GetRingDimension());
    
    if (handle->gpu_loaded && handle->gpu_ct) {
        meta["scale"] = handle->gpu_ct->scalingFactor;
        meta["level"] = handle->gpu_ct->level;
        meta["noise_scale"] = handle->gpu_ct->noiseScaleDeg;
        meta["gpu_ax_size"] = static_cast<std::uint32_t>(handle->gpu_ct->ax__.size());
        meta["gpu_bx_size"] = static_cast<std::uint32_t>(handle->gpu_ct->bx__.size());
        meta["gpu_num_limbs"] = ring_dim == 0 ? 0u : static_cast<std::uint32_t>(handle->gpu_ct->ax__.size() / ring_dim);
    } else {
        meta["scale"] = handle->ciphertext->GetScalingFactor();
        meta["level"] = handle->ciphertext->GetLevel();
        meta["noise_scale"] = handle->ciphertext->GetNoiseScaleDeg();
    }
    
    meta["slots"] = ring_dim / 2;
    meta["ring_dim"] = ring_dim;
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
        ctx->gpu_context->EnableMemoryPool();
        if (ctx->gpu_context->GetDegree() > 32768) {
            // Large-ring rescale is unstable with the fused epilogue path.
            // Keep the arithmetic on GPU, but use the unfused sequence.
            ctx->gpu_context->is_rescale_fused = false;
        }
        
        // Load evaluation keys to GPU
        ctx->gpu_evk = std::make_unique<ckks::EvaluationKey>(
            LoadEvalMultRelinKey(ctx->context, ctx->key_tag)
        );
        
        ctx->gpu_rot_keys = std::make_unique<std::map<int, ckks::EvaluationKey>>();
        ctx->gpu_rot_keys_auto = std::make_unique<std::map<uint32_t, ckks::EvaluationKey>>();
        ctx->gpu_context->preloaded_evaluation_key = ctx->gpu_evk.get();
        ctx->gpu_context->preloaded_rotation_key_map = ctx->gpu_rot_keys_auto.get();
        
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
            num_slots = context->GetRingDimension() / 2;
        }
        context->EvalBootstrapKeyGen(key_pair.secretKey, num_slots);
    }

    ctx->key_tag = key_pair.secretKey->GetKeyTag();
    
    if (enable_gpu) {
        init_gpu(ctx);
        
        const bool eager_gpu_rotation_load = ctx->gpu_initialized && ctx->gpu_context && ctx->gpu_context->GetDegree() <= 32768;
        if (eager_gpu_rotation_load && !rotations.empty()) {
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
    
    if (ctx->gpu_initialized && !lhs->force_cpu && !rhs->force_cpu) {
        lhs->ensureGPU();
        rhs->ensureGPU();
        
        ckks::CtAccurate result = ctx->gpu_context->Add(*lhs->gpu_ct, *rhs->gpu_ct);
        return make_cipher_from_gpu_lazy(ctx, std::move(result), lhs->ciphertext);
    } else {
        auto result = ctx->context->EvalAdd(lhs->ciphertext, rhs->ciphertext);
        auto handle = make_cipher(ctx, result);
        if (lhs->force_cpu || rhs->force_cpu) {
            mark_force_cpu(handle);
        }
        return handle;
    }
}

std::shared_ptr<GPUCiphertextHandle> add_plain(
    const std::shared_ptr<GPUCiphertextHandle>& lhs,
    const std::vector<double>& plain
) {
    OpTimer timer("add_plain");
    auto ctx = lhs->context;
    if (ctx->gpu_initialized && !lhs->force_cpu) {
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
    auto handle = make_cipher(ctx, result);
    if (lhs->force_cpu) {
        mark_force_cpu(handle);
    }
    return handle;
}

std::shared_ptr<GPUCiphertextHandle> sub_cipher(
    const std::shared_ptr<GPUCiphertextHandle>& lhs,
    const std::shared_ptr<GPUCiphertextHandle>& rhs
) {
    auto ctx = lhs->context;
    
    if (ctx->gpu_initialized && !lhs->force_cpu && !rhs->force_cpu) {
        lhs->ensureGPU();
        rhs->ensureGPU();
        
        ckks::CtAccurate result = ctx->gpu_context->Sub(*lhs->gpu_ct, *rhs->gpu_ct);
        sync_large_ring_gpu(ctx, "sub_cipher GPU Sub");
        return make_cipher_from_gpu_lazy(ctx, std::move(result), lhs->ciphertext);
    } else {
        auto result = ctx->context->EvalSub(lhs->ciphertext, rhs->ciphertext);
        auto handle = make_cipher(ctx, result);
        if (lhs->force_cpu || rhs->force_cpu) {
            mark_force_cpu(handle);
        }
        return handle;
    }
}

std::shared_ptr<GPUCiphertextHandle> sub_plain(
    const std::shared_ptr<GPUCiphertextHandle>& lhs,
    const std::vector<double>& plain
) {
    auto ctx = lhs->context;
    if (ctx->gpu_initialized && !lhs->force_cpu) {
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
    auto handle = make_cipher(ctx, result);
    if (lhs->force_cpu) {
        mark_force_cpu(handle);
    }
    return handle;
}

std::shared_ptr<GPUCiphertextHandle> mul_cipher(
    const std::shared_ptr<GPUCiphertextHandle>& lhs,
    const std::shared_ptr<GPUCiphertextHandle>& rhs
) {
    auto ctx = lhs->context;
    
    if (ctx->gpu_initialized && ctx->gpu_evk && !lhs->force_cpu && !rhs->force_cpu) {
        lhs->ensureGPU();
        rhs->ensureGPU();
        ckks::CtAccurate result;
        const bool use_high_level_mult = ctx->gpu_context && ctx->gpu_context->GetDegree() > 32768;
        const bool use_local_evk = use_high_level_mult;
        if (use_local_evk) {
            ckks::EvaluationKey local_gpu_evk = LoadEvalMultRelinKey(ctx->context, ctx->key_tag);
            cudaError_t err = cudaDeviceSynchronize();
            if (err != cudaSuccess) {
                throw std::runtime_error(std::string("failed to synchronize local GPU eval key load: ") + cudaGetErrorString(err));
            }
            auto* previous_evk = ctx->gpu_context->preloaded_evaluation_key;
            ctx->gpu_context->preloaded_evaluation_key = &local_gpu_evk;
            result = use_high_level_mult
                ? ctx->gpu_context->EvalMultAndRelin(*lhs->gpu_ct, *rhs->gpu_ct, local_gpu_evk)
                : ctx->gpu_context->EvalMultAndRelinNoRescale(*lhs->gpu_ct, *rhs->gpu_ct, local_gpu_evk);
            ctx->gpu_context->preloaded_evaluation_key = previous_evk;
        } else {
            result = ctx->gpu_context->EvalMultAndRelinNoRescale(
                *lhs->gpu_ct, *rhs->gpu_ct, *ctx->gpu_evk);
        }
        sync_large_ring_gpu(ctx, use_high_level_mult ? "mul_cipher EvalMultAndRelin" : "mul_cipher EvalMultAndRelinNoRescale");
        auto handle = make_cipher_from_gpu_lazy(ctx, std::move(result), lhs->ciphertext);
        set_result_scaling_factor_int(handle, mul_scaling_factor_int(lhs->ciphertext, rhs->ciphertext));
        return handle;
    }
    
    lhs->syncFromGPU();
    rhs->syncFromGPU();
    
    auto result = ctx->context->EvalMultAndRelinearize(lhs->ciphertext, rhs->ciphertext);
    auto handle = make_cipher(ctx, result);
    if (lhs->force_cpu || rhs->force_cpu) {
        mark_force_cpu(handle);
    }
    return handle;
}

std::shared_ptr<GPUCiphertextHandle> mul_plain(
    const std::shared_ptr<GPUCiphertextHandle>& lhs,
    const std::vector<double>& plain
) {
    auto ctx = lhs->context;
    
    if (ctx->gpu_initialized && !lhs->force_cpu) {
        lhs->ensureGPU();
        uint32_t ct_level = lhs->gpu_ct->level;
        auto plaintext = ctx->context->MakeCKKSPackedPlaintext(plain, 1, ct_level);
        plaintext->Encode();
        ckks::PtAccurate gpu_pt = LoadAccuratePlaintext(
            plaintext, plaintext->GetElement<DCRTPoly>());
        ckks::CtAccurate result = ctx->gpu_context->EvalMultPlain(
            *lhs->gpu_ct, gpu_pt);
        auto handle = make_cipher_from_gpu_lazy(ctx, std::move(result), lhs->ciphertext);
        set_result_scaling_factor_int(handle, mul_plain_scaling_factor_int(lhs->ciphertext, plaintext));
        return handle;
    }
    
    const uint32_t ct_level = lhs->ciphertext->GetLevel();
    auto plaintext = make_plaintext(ctx, plain, ct_level);
    auto result = ctx->context->EvalMult(lhs->ciphertext, plaintext);
    auto handle = make_cipher(ctx, result);
    if (lhs->force_cpu) {
        mark_force_cpu(handle);
    }
    return handle;
}

std::shared_ptr<GPUCiphertextHandle> square_cipher(
    const std::shared_ptr<GPUCiphertextHandle>& cipher
) {
    OpTimer timer("square");
    auto ctx = cipher->context;
    
    if (ctx->gpu_initialized && ctx->gpu_evk && !cipher->force_cpu) {
        cipher->ensureGPU();
        ckks::CtAccurate result;
        const bool use_high_level_square =
            ctx->gpu_context && ctx->gpu_context->GetDegree() > 32768;
        const bool use_local_evk = use_high_level_square;
        std::optional<ckks::EvaluationKey> local_gpu_evk;
        if (use_local_evk) {
            local_gpu_evk.emplace(LoadEvalMultRelinKey(ctx->context, ctx->key_tag));
            cudaError_t err = cudaDeviceSynchronize();
            if (err != cudaSuccess) {
                throw std::runtime_error(std::string("failed to synchronize local GPU eval key load: ") + cudaGetErrorString(err));
            }
        }
        if (use_high_level_square) {
            auto* previous_evk = ctx->gpu_context->preloaded_evaluation_key;
            if (use_local_evk) {
                ctx->gpu_context->preloaded_evaluation_key = &*local_gpu_evk;
            }
            result = ctx->gpu_context->EvalSquareAndRelin(*cipher->gpu_ct, use_local_evk ? *local_gpu_evk : *ctx->gpu_evk);
            ctx->gpu_context->preloaded_evaluation_key = previous_evk;
            sync_large_ring_gpu(ctx, "square_cipher EvalSquareAndRelin");
        } else {
            result = ctx->gpu_context->EvalMultAndRelinNoRescale(
                *cipher->gpu_ct, *cipher->gpu_ct, use_local_evk ? *local_gpu_evk : *ctx->gpu_evk);
            sync_large_ring_gpu(ctx, "square_cipher EvalMultAndRelinNoRescale");
        }
        auto handle = make_cipher_from_gpu_lazy(ctx, std::move(result), cipher->ciphertext);
        set_result_scaling_factor_int(handle, mul_scaling_factor_int(cipher->ciphertext, cipher->ciphertext));
        return handle;
    }
    
    // CPU fallback: square + relin (no rescale) - matches GPU path
    auto result = ctx->context->EvalMultAndRelinearize(cipher->ciphertext, cipher->ciphertext);
    auto handle = make_cipher(ctx, result);
    if (cipher->force_cpu) {
        mark_force_cpu(handle);
    }
    return handle;
}

std::shared_ptr<GPUCiphertextHandle> rescale_cipher(
    const std::shared_ptr<GPUCiphertextHandle>& tensor
) {
    OpTimer timer("rescale");
    auto ctx = tensor->context;
    
    if (ctx->gpu_initialized && !tensor->force_cpu) {
        tensor->ensureGPU();
        const bool large_ring = ctx->gpu_context && ctx->gpu_context->GetDegree() > 32768;
        if (large_ring && tensor->gpu_loaded) {
            const_cast<GPUCiphertextHandle*>(tensor.get())->syncFromGPU();
        }
        ckks::CtAccurate result = ctx->gpu_context->Rescale(*tensor->gpu_ct);
        sync_large_ring_gpu(ctx, "rescale_cipher GPU Rescale");
        if (large_ring) {
            auto cpu_template = ctx->context->ModReduce(tensor->ciphertext);
            auto handle = make_cipher_from_gpu_lazy(ctx, std::move(result), cpu_template);
            handle->ciphertext = cpu_template;
            return handle;
        }
        return make_cipher_from_gpu_lazy(ctx, std::move(result), tensor->ciphertext);
    } else {
        auto result = ctx->context->ModReduce(tensor->ciphertext);
        auto handle = make_cipher(ctx, result);
        if (tensor->force_cpu) {
            mark_force_cpu(handle);
        }
        return handle;
    }
}

std::shared_ptr<GPUCiphertextHandle> rotate_cipher(
    const std::shared_ptr<GPUCiphertextHandle>& tensor,
    int index
) {
    OpTimer timer("rotate");
    auto ctx = tensor->context;

    if (ctx->gpu_initialized && ctx->gpu_rot_keys) {
        tensor->ensureGPU();

        if (ensure_gpu_rotation_key(ctx, index)) {
            auto it = ctx->gpu_rot_keys->find(index);
            if (it != ctx->gpu_rot_keys->end()) {
                ckks::CtAccurate result = eval_rotate_standard_gpu(ctx, *tensor->gpu_ct, index);
                sync_large_ring_gpu(ctx, "rotate_cipher cached EvalAtIndex");
                return make_cipher_from_gpu_lazy(ctx, std::move(result), tensor->ciphertext);
            }
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

        auto canonical_rotation = [&diagonals](std::size_t diagonal_index) -> int {
            if (diagonal_index == 0) return 0;
            const std::size_t dimension = diagonals.size();
            const std::size_t half = dimension / 2;
            if (diagonal_index > half) {
                return static_cast<int>(diagonal_index) - static_cast<int>(dimension);
            }
            return static_cast<int>(diagonal_index);
        };

        std::unordered_set<int> required_rotations;
        for (std::size_t idx = 1; idx < diagonals.size(); ++idx) {
            if (diag_nonzero[idx]) {
                required_rotations.insert(canonical_rotation(idx));
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
            std::vector<std::pair<std::size_t, int>> active_terms;
            active_terms.reserve(diagonals.size());
            for (std::size_t idx = 0; idx < diagonals.size(); ++idx) {
                if (idx == 0 || diag_nonzero[idx]) {
                    active_terms.emplace_back(idx, canonical_rotation(idx));
                }
            }
            std::vector<ckks::CtAccurate> rotated_terms;
            rotated_terms.reserve(active_terms.size());
            std::vector<ckks::PtAccurate> pt_terms;
            pt_terms.reserve(active_terms.size());
            std::vector<const ckks::PtAccurate*> pt_ptrs;
            pt_ptrs.reserve(active_terms.size());

            for (std::size_t t = 0; t < active_terms.size(); ++t) {
                const auto [idx, rotation] = active_terms[t];

                ckks::CtAccurate rotated = *tensor->gpu_ct;
                if (rotation != 0) {
                    auto it = ctx->gpu_rot_keys->find(rotation);
                    uint32_t auto_index = ctx->context->FindAutomorphismIndex(rotation);
                    rotated = ctx->gpu_context->EvalAtIndex(*tensor->gpu_ct, it->second, auto_index);
                }

                rotated_terms.push_back(std::move(rotated));
                pt_terms.push_back(make_gpu_plaintext(ctx, diagonals[idx], tensor->gpu_ct->level));
            }

            for (std::size_t i = 0; i < pt_terms.size(); ++i) {
                pt_ptrs.push_back(&pt_terms[i]);
            }

            if (!rotated_terms.empty()) {
                if (use_batch_muladd_for_context(ctx)) {
                    std::vector<const ckks::CtAccurate*> ct_ptrs;
                    ct_ptrs.reserve(rotated_terms.size());
                    for (std::size_t i = 0; i < rotated_terms.size(); ++i) {
                        ct_ptrs.push_back(&rotated_terms[i]);
                    }
                    ckks::CtAccurate accumulator = batch_mult_plain_add_gpu(ctx, ct_ptrs, pt_ptrs);
                    auto handle = make_cipher_from_gpu_lazy(ctx, std::move(accumulator), tensor->ciphertext);
                    auto representative_pt = make_aux_plaintext_local_q(ctx, std::vector<double>{0.0}, tensor->gpu_ct->level, 1);
                    set_result_scaling_factor_int(handle, mul_plain_scaling_factor_int(tensor->ciphertext, representative_pt));
                    return handle;
                }

                std::vector<ckks::CtAccurate> terms;
                terms.reserve(rotated_terms.size());
                for (std::size_t i = 0; i < rotated_terms.size(); ++i) {
                    terms.push_back(ctx->gpu_context->EvalMultPlain(rotated_terms[i], *pt_ptrs[i]));
                }
                ckks::CtAccurate accumulator = reduce_add_gpu(ctx, terms);
                auto handle = make_cipher_from_gpu_lazy(ctx, std::move(accumulator), tensor->ciphertext);
                auto representative_pt = make_aux_plaintext_local_q(ctx, std::vector<double>{0.0}, tensor->gpu_ct->level, 1);
                set_result_scaling_factor_int(handle, mul_plain_scaling_factor_int(tensor->ciphertext, representative_pt));
                return handle;
            }
        }
    }

    if (tensor->gpu_loaded) {
        const_cast<GPUCiphertextHandle*>(tensor.get())->syncFromGPU();
    }
    auto cc = ctx->context;
    const uint32_t ct_level = tensor->ciphertext->GetLevel();
    auto accumulator = cc->EvalMult(tensor->ciphertext, make_plaintext(ctx, diagonals.front(), ct_level));
    for (std::size_t idx = 1; idx < diagonals.size(); ++idx) {
        auto rotated = cc->EvalAtIndex(tensor->ciphertext, static_cast<int>(idx));
        auto term = cc->EvalMult(rotated, make_plaintext(ctx, diagonals[idx], ct_level));
        accumulator = cc->EvalAdd(accumulator, term);
    }
    return make_cipher(ctx, accumulator);
}

std::shared_ptr<GPUCiphertextHandle> repeated_block_matmul_cipher(
    const std::shared_ptr<GPUCiphertextHandle>& tensor,
    const std::vector<std::vector<double>>& block_weight,
    std::uint32_t num_blocks,
    std::uint32_t total_in
) {
    const auto slots = static_cast<std::size_t>(tensor->context->context->GetRingDimension() / 2);
    auto diagonals = build_repeated_block_diagonals(block_weight, num_blocks, total_in, slots);
    return matvec_diag_cipher(tensor, diagonals);
}

std::shared_ptr<GPUCiphertextHandle> poly_eval_cipher(
    const std::shared_ptr<GPUCiphertextHandle>& tensor,
    const std::vector<double>& coeffs
) {
    auto ctx = tensor->context;
    if (ctx->gpu_initialized && !coeffs.empty() && ctx->gpu_evk) {
        tensor->ensureGPU();

        if (coeffs.size() == 3 && coeffs[0] == 0.0 && coeffs[1] == 0.0 && coeffs[2] == 1.0) {
            ckks::CtAccurate squared = ctx->gpu_context->EvalMultAndRelinNoRescale(*tensor->gpu_ct, *tensor->gpu_ct, *ctx->gpu_evk);
            auto handle = make_cipher_from_gpu_lazy(ctx, std::move(squared), tensor->ciphertext);
            set_result_scaling_factor_int(handle, mul_scaling_factor_int(tensor->ciphertext, tensor->ciphertext));
            return handle;
        }

        if (coeffs.size() == 5 && std::abs(coeffs[3]) <= 1e-12) {
            const auto slots = static_cast<std::size_t>(ctx->context->GetRingDimension() / 2);
            const std::vector<double> c0v(slots, coeffs[0]);
            const std::vector<double> c1v(slots, coeffs[1]);
            const std::vector<double> c2v(slots, coeffs[2]);
            const std::vector<double> c4v(slots, coeffs[4]);

            const ckks::CtAccurate& x0 = *tensor->gpu_ct;
            ckks::CtAccurate x2 = ctx->gpu_context->EvalMultAndRelinNoRescale(x0, x0, *ctx->gpu_evk);
            x2 = ctx->gpu_context->Rescale(x2);
            ckks::CtAccurate x4 = ctx->gpu_context->EvalMultAndRelinNoRescale(x2, x2, *ctx->gpu_evk);
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
        const bool large_ring_gpu = ctx->gpu_context && ctx->gpu_context->GetDegree() > 32768;
        auto canonical_rotation = [n, large_ring_gpu](std::size_t diagonal_index) -> int {
            if (diagonal_index == 0) return 0;
            if (large_ring_gpu) {
                // Keep the same positive rotation schedule as the CPU path.
                // Large-ring key generation already includes these indices,
                // while the signed canonical form can miss required keys.
                return static_cast<int>(diagonal_index);
            }
            const std::size_t half = n / 2;
            if (diagonal_index > half) {
                return static_cast<int>(diagonal_index) - static_cast<int>(n);
            }
            return static_cast<int>(diagonal_index);
        };

        for (std::size_t k = 1; k < n; ++k) {
            if (diag_nonzero[k]) {
                required_rotations.insert(canonical_rotation(k));
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
            std::vector<std::pair<std::size_t, int>> active_terms;
            active_terms.reserve(n);
            for (std::size_t k = 0; k < n; ++k) {
                if (!diag_nonzero[k]) continue;
                active_terms.emplace_back(k, canonical_rotation(k));
            }
            std::vector<ckks::CtAccurate> rotated_terms;
            rotated_terms.reserve(active_terms.size());
            std::vector<ckks::PtAccurate> pt_terms;
            pt_terms.reserve(active_terms.size());
            std::vector<const ckks::PtAccurate*> pt_ptrs;
            pt_ptrs.reserve(active_terms.size());

            for (std::size_t t = 0; t < active_terms.size(); ++t) {
                const auto [k, rotation] = active_terms[t];

                ckks::CtAccurate rotated = *tensor->gpu_ct;
                if (rotation != 0) {
                    auto it = ctx->gpu_rot_keys->find(rotation);
                    uint32_t auto_index = ctx->context->FindAutomorphismIndex(rotation);
                    rotated = ctx->gpu_context->EvalAtIndex(*tensor->gpu_ct, it->second, auto_index);
                }

                rotated_terms.push_back(std::move(rotated));

                std::vector<double> diag(slots, 0.0);
                for (std::size_t i = 0; i < m; ++i) {
                    diag[i] = matrix[i][(i + k) % n];
                }
                pt_terms.push_back(make_gpu_plaintext(ctx, diag, tensor->gpu_ct->level));
            }

            for (std::size_t i = 0; i < pt_terms.size(); ++i) {
                pt_ptrs.push_back(&pt_terms[i]);
            }

            if (rotated_terms.empty()) {
                throw std::runtime_error("failed to build accumulator in dense matmul (all-zero matrix)");
            }
            if (use_batch_muladd_for_context(ctx)) {
                std::vector<const ckks::CtAccurate*> ct_ptrs;
                ct_ptrs.reserve(rotated_terms.size());
                for (std::size_t i = 0; i < rotated_terms.size(); ++i) {
                    ct_ptrs.push_back(&rotated_terms[i]);
                }
                ckks::CtAccurate accumulator = batch_mult_plain_add_gpu(ctx, ct_ptrs, pt_ptrs);
                auto handle = make_cipher_from_gpu_lazy(ctx, std::move(accumulator), tensor->ciphertext);
                auto representative_pt = make_plaintext(ctx, std::vector<double>{0.0}, tensor->gpu_ct->level);
                set_result_scaling_factor_int(handle, mul_plain_scaling_factor_int(tensor->ciphertext, representative_pt));
                return handle;
            }

            std::vector<ckks::CtAccurate> terms;
            terms.reserve(rotated_terms.size());
            for (std::size_t i = 0; i < rotated_terms.size(); ++i) {
                terms.push_back(ctx->gpu_context->EvalMultPlain(rotated_terms[i], *pt_ptrs[i]));
            }
            ckks::CtAccurate accumulator = reduce_add_gpu(ctx, terms);
            auto handle = make_cipher_from_gpu_lazy(ctx, std::move(accumulator), tensor->ciphertext);
            auto representative_pt = make_plaintext(ctx, std::vector<double>{0.0}, tensor->gpu_ct->level);
            set_result_scaling_factor_int(handle, mul_plain_scaling_factor_int(tensor->ciphertext, representative_pt));
            return handle;
        }
    }

    if (tensor->gpu_loaded) {
        const_cast<GPUCiphertextHandle*>(tensor.get())->syncFromGPU();
    }

    const uint32_t ct_level = tensor->ciphertext->GetLevel();
    Ciphertext<DCRTPoly> accumulator;
    for (std::size_t k = 0; k < n; ++k) {
        if (!diag_nonzero[k]) continue;  // skip zero diagonal
        std::vector<double> diag(slots, 0.0);
        for (std::size_t i = 0; i < m; ++i) {
            diag[i] = matrix[i][(i + k) % n];
        }
        auto rotated = (k == 0) ? tensor->ciphertext : cc->EvalAtIndex(tensor->ciphertext, static_cast<int>(k));
        auto term = cc->EvalMult(rotated, make_plaintext(ctx, diag, ct_level));
        if (!accumulator) {
            accumulator = term;
        } else {
            accumulator = cc->EvalAdd(accumulator, term);
        }
    }
    if (!accumulator) {
        throw std::runtime_error("failed to build accumulator in dense matmul (all-zero matrix)");
    }
    auto result = make_cipher(ctx, accumulator);
    if (ctx->gpu_initialized && cc->GetRingDimension() > 32768) {
        mark_force_cpu(result);
    }
    return result;
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
    const std::vector<bool>& precomputed_nonzero = std::vector<bool>()
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
                            std::vector<double> diag = build_bsgs_aux_diag(matrix, slots, d, giant_step);

                            if (weight_hash != 0) {
                                std::uint64_t fast_key = hash_weight_diag(weight_hash, static_cast<std::uint32_t>(d), bsgs_n1, tensor->gpu_ct->level);
                                if (ctx->gpu_plain_cache_limit > 0 && ctx->gpu_plain_cache.size() >= ctx->gpu_plain_cache_limit)
                                    ctx->gpu_plain_cache.clear();
                                auto gpu_pt = make_gpu_aux_plaintext(ctx, diag, tensor->gpu_ct->level);
                                auto [it, _] = ctx->gpu_plain_cache.emplace(fast_key, std::move(gpu_pt));
                                pt_cached_ptrs[d] = &it->second;
                            } else {
                                auto gpu_pt = make_gpu_aux_plaintext(ctx, diag, tensor->gpu_ct->level);
                                auto [it, _] = ctx->gpu_plain_cache.emplace(hash_plain_vector(diag, tensor->gpu_ct->level), std::move(gpu_pt));
                                pt_cached_ptrs[d] = &it->second;
                            }
                        }
                    }
                }

                const bool use_large_ring_fast_rotation =
                    ctx->gpu_context && ctx->gpu_context->GetDegree() > 32768;

                if (use_large_ring_fast_rotation) {
                    OpTimer fast_timer("matmul_bsgs.large_ring_rotations");

                    std::vector<bool> active_baby_steps(bsgs_n1, false);
                    active_baby_steps[0] = true;
                    for (std::size_t d = 1; d < in_features; ++d) {
                        if (diag_nonzero_vec[d]) {
                            active_baby_steps[d % bsgs_n1] = true;
                        }
                    }

                    ckks::DeviceVector base_digits = ctx->gpu_context->ModUp(tensor->gpu_ct->ax__);
                    std::vector<std::unique_ptr<ckks::CtAccurate>> fast_rotations(bsgs_n1);
                    for (std::uint32_t j = 0; j < bsgs_n1 && j < in_features; ++j) {
                        if (!active_baby_steps[j]) {
                            continue;
                        }
                        if (j == 0) {
                            fast_rotations[j] = std::make_unique<ckks::CtAccurate>(
                                make_fast_rotation_identity_ext(*tensor->gpu_ct, *ctx->gpu_context));
                        } else {
                            fast_rotations[j] = std::make_unique<ckks::CtAccurate>(
                                eval_fast_rotate_ext(ctx, *tensor->gpu_ct, base_digits, static_cast<int>(j), true));
                        }
                    }
                    sync_large_ring_gpu(ctx, "matmul_bsgs large_ring fast rotations");

                    ckks::CtAccurate accum_std;
                    accum_std.level = tensor->gpu_ct->level;
                    accum_std.noiseScaleDeg = tensor->gpu_ct->noiseScaleDeg;
                    accum_std.scalingFactor = tensor->gpu_ct->scalingFactor;
                    bool have_accum = false;
                    bool have_any_block = false;

                    for (std::uint32_t k = 0; k < bsgs_n2; ++k) {
                        std::uint32_t giant_step = k * bsgs_n1;
                        std::vector<std::uint32_t> active_js;
                        active_js.reserve(fast_rotations.size());
                        std::vector<ckks::PtAccurate> pt_terms;
                        pt_terms.reserve(fast_rotations.size());

                        for (std::uint32_t j = 0; j < fast_rotations.size(); ++j) {
                            std::uint32_t d = giant_step + j;
                            if (d >= in_features) {
                                break;
                            }
                            if (!diag_nonzero_vec[d]) {
                                continue;
                            }
                            active_js.push_back(j);
                            std::vector<double> diag = build_bsgs_aux_diag(matrix, slots, d, giant_step);
                            pt_terms.push_back(make_gpu_plaintext_ext(ctx, diag, tensor->gpu_ct->level));
                        }

                        if (active_js.empty()) {
                            continue;
                        }

                        have_any_block = true;

                        ckks::CtAccurate inner = ctx->gpu_context->EvalMultPlainExt(*fast_rotations[active_js[0]], pt_terms[0]);

                        for (std::size_t idx = 1; idx < active_js.size(); ++idx) {
                            std::uint32_t j = active_js[idx];
                            ckks::CtAccurate to_add = ctx->gpu_context->EvalMultPlainExt(*fast_rotations[j], pt_terms[idx]);
                            ctx->gpu_context->EvalAddInPlaceExt(inner, to_add);
                        }

                        ckks::CtAccurate block_std;

                        if (k == 0) {
                            ckks::DeviceVector first_term;
                            ctx->gpu_context->ModDown(inner.bx__, first_term);
                            ctx->gpu_context->ModDown(inner.ax__, block_std.ax__);
                            block_std.bx__.resize(first_term.size());
                            block_std.bx__.setConstant(0);
                            ctx->gpu_context->AddCoreInPlace(block_std.bx__, first_term);
                            block_std.level = inner.level;
                            block_std.noiseScaleDeg = inner.noiseScaleDeg;
                            block_std.scalingFactor = inner.scalingFactor;
                        } else if (giant_step != 0) {
                            mod_down_in_place(ctx, inner);
                            const uint32_t auto_index = FindAutomorphismIndex2nComplex(
                                static_cast<int>(giant_step),
                                ctx->context->GetCyclotomicOrder());
                            ckks::DeviceVector inner_b_rot = ctx->gpu_context->AutomorphismTransform(inner.bx__, auto_index);

                            ckks::DeviceVector inner_digits = ctx->gpu_context->ModUp(inner.ax__);
                            ckks::CtAccurate inner_rot = eval_fast_rotate_ext(
                                ctx,
                                inner,
                                inner_digits,
                                static_cast<int>(giant_step),
                                false);
                            ctx->gpu_context->ModDown(inner_rot.ax__, block_std.ax__);
                            ctx->gpu_context->ModDown(inner_rot.bx__, block_std.bx__);
                            ctx->gpu_context->AddCoreInPlace(block_std.bx__, inner_b_rot);
                            block_std.level = inner_rot.level;
                            block_std.noiseScaleDeg = inner_rot.noiseScaleDeg;
                            block_std.scalingFactor = inner_rot.scalingFactor;
                        } else {
                            ckks::DeviceVector inner_mod_down;
                            ctx->gpu_context->ModDown(inner.bx__, inner_mod_down);
                            ctx->gpu_context->ModDown(inner.ax__, block_std.ax__);
                            block_std.bx__.resize(inner_mod_down.size());
                            block_std.bx__.setConstant(0);
                            ctx->gpu_context->AddCoreInPlace(block_std.bx__, inner_mod_down);
                            block_std.level = inner.level;
                            block_std.noiseScaleDeg = inner.noiseScaleDeg;
                            block_std.scalingFactor = inner.scalingFactor;
                        }

                        if (!have_accum) {
                            accum_std = std::move(block_std);
                            have_accum = true;
                        } else {
                            accum_std = ctx->gpu_context->Add(accum_std, block_std);
                        }
                    }

                    if (!have_any_block || !have_accum) {
                        throw std::runtime_error("failed to build accumulator in BSGS matmul (all-zero matrix)");
                    }

                    auto handle = make_cipher_from_gpu_lazy(ctx, std::move(accum_std), tensor->ciphertext);
                    auto representative_pt = make_plaintext(ctx, std::vector<double>{0.0}, tensor->gpu_ct->level);
                    set_result_scaling_factor_int(handle, mul_plain_scaling_factor_int(tensor->ciphertext, representative_pt));
                    return handle;
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
                            baby_ciphers.push_back(eval_rotate_standard_gpu(ctx, *tensor->gpu_ct, static_cast<int>(j)));
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
                                std::vector<double> diag = build_bsgs_aux_diag(matrix, slots, d, giant_step);
                                auto gpu_pt = make_gpu_aux_plaintext(ctx, diag, tensor->gpu_ct->level);
                                pt_terms.push_back(std::move(gpu_pt));
                            }
                        }

                        if (active_js.empty()) continue;
                        ckks::CtAccurate block;
                        if (use_batch_muladd_for_context(ctx)) {
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

                auto handle = make_cipher_from_gpu_lazy(ctx, std::move(accumulator), tensor->ciphertext);
                auto representative_pt = make_plaintext(ctx, std::vector<double>{0.0}, tensor->gpu_ct->level);
                set_result_scaling_factor_int(handle, mul_plain_scaling_factor_int(tensor->ciphertext, representative_pt));
                return handle;
            }
    }

    if (tensor->gpu_loaded) {
        const_cast<GPUCiphertextHandle*>(tensor.get())->syncFromGPU();
    }

    const uint32_t ct_level = tensor->ciphertext->GetLevel();
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

            auto term = cc->EvalMult(baby_ciphers[j], make_plaintext(ctx, diag, ct_level));

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
    if (ctx->gpu_initialized && cc->GetRingDimension() > 32768) {
        mark_force_cpu(result);
    }
    return result;
}

std::shared_ptr<GPUCiphertextHandle> packed_self_attention_power_cipher(
    const std::shared_ptr<GPUCiphertextHandle>& query,
    const std::shared_ptr<GPUCiphertextHandle>& key,
    const std::shared_ptr<GPUCiphertextHandle>& value,
    std::uint32_t batch_size,
    std::uint32_t seq_len,
    std::uint32_t embed_dim,
    double scale,
    double shift,
    const std::vector<double>& reciprocal_coeffs,
    const std::vector<double>& renorm_reciprocal_coeffs
) {
    auto ctx = query->context;
    const std::size_t slot_count = static_cast<std::size_t>(ctx->context->GetRingDimension() / 2);
    const std::uint32_t slots_per_sample = seq_len * embed_dim;
    const std::uint32_t total_in = batch_size * slots_per_sample;
    std::vector<std::vector<double>> ones_block(embed_dim, std::vector<double>(embed_dim, 1.0));

    std::map<int, std::shared_ptr<GPUCiphertextHandle>> rotated_k{{0, key}};
    std::map<int, std::shared_ptr<GPUCiphertextHandle>> rotated_v{{0, value}};
    std::shared_ptr<GPUCiphertextHandle> combined;

    for (std::uint32_t query_index = 0; query_index < seq_len; ++query_index) {
        auto query_mask = build_attention_plain(slot_count, batch_size, seq_len, embed_dim, query_index, 0.0, 1.0);
        auto reciprocal_fill = build_attention_plain(slot_count, batch_size, seq_len, embed_dim, query_index, 1.0, 0.0);
        auto outside_fill = build_attention_plain(slot_count, batch_size, seq_len, embed_dim, query_index, -2.0, 0.0);
        std::vector<std::shared_ptr<GPUCiphertextHandle>> scores;
        scores.reserve(seq_len);

        for (std::uint32_t key_index = 0; key_index < seq_len; ++key_index) {
            const int offset = static_cast<int>((static_cast<int>(key_index) - static_cast<int>(query_index)) * static_cast<int>(embed_dim));
            if (!rotated_k.count(offset)) {
                rotated_k[offset] = rotate_cipher(key, offset);
                rotated_v[offset] = rotate_cipher(value, offset);
            }
            auto aligned_product = rescale_cipher(mul_cipher(query, rotated_k[offset]));
            auto masked_scores = rescale_cipher(mul_plain(aligned_product, query_mask));
            auto score = repeated_block_matmul_cipher(masked_scores, ones_block, batch_size * seq_len, total_in);
            score = rescale_cipher(mul_plain(score, std::vector<double>(slot_count, scale)));
            score = add_plain(score, outside_fill);
            scores.push_back(score);
        }

        std::vector<std::shared_ptr<GPUCiphertextHandle>> squared;
        squared.reserve(scores.size());
        std::shared_ptr<GPUCiphertextHandle> z_sum;
        for (const auto& score : scores) {
            auto shifted = add_plain(score, std::vector<double>(slot_count, shift));
            auto sq = rescale_cipher(mul_cipher(shifted, shifted));
            squared.push_back(sq);
            z_sum = z_sum ? add_cipher(z_sum, sq) : sq;
        }

        auto z_sum_safe = add_plain(z_sum, std::vector<double>(slot_count, 0.01));
        z_sum_safe = add_plain(z_sum_safe, reciprocal_fill);
        auto inv_z_sum = poly_eval_cipher(z_sum_safe, reciprocal_coeffs);

        std::vector<std::shared_ptr<GPUCiphertextHandle>> weights;
        weights.reserve(squared.size());
        for (const auto& sq : squared) {
            auto weight_ct = rescale_cipher(mul_cipher(sq, inv_z_sum));
            weights.push_back(weight_ct);
        }
        std::shared_ptr<GPUCiphertextHandle> token_output;
        for (std::uint32_t key_index = 0; key_index < seq_len; ++key_index) {
            auto normalized_weight = rescale_cipher(mul_plain(weights[key_index], query_mask));
            const int offset = static_cast<int>((static_cast<int>(key_index) - static_cast<int>(query_index)) * static_cast<int>(embed_dim));
            auto term = rescale_cipher(mul_cipher(normalized_weight, rotated_v[offset]));
            token_output = token_output ? add_cipher(token_output, term) : term;
        }

        token_output = rescale_cipher(mul_plain(token_output, query_mask));
        combined = combined ? add_cipher(combined, token_output) : token_output;
    }

    return combined;
}

std::vector<std::shared_ptr<GPUCiphertextHandle>> halved_ccmm_fused(
    const std::vector<std::shared_ptr<GPUCiphertextHandle>>& queries,
    const std::vector<std::shared_ptr<GPUCiphertextHandle>>& keys_hybrid,
    std::uint32_t half_seq_len
) {
    if (queries.empty() || keys_hybrid.empty()) {
        return {};
    }
    if (queries.size() != keys_hybrid.size()) {
        throw std::runtime_error("halved_ccmm_fused requires queries and keys_hybrid to have the same size");
    }

    auto ctx = queries.front()->context;
    std::vector<std::shared_ptr<GPUCiphertextHandle>> out;
    out.reserve(half_seq_len);

    for (std::uint32_t r = 0; r < half_seq_len; ++r) {
        std::vector<std::shared_ptr<GPUCiphertextHandle>> terms;
        terms.reserve(queries.size());

        for (std::size_t c = 0; c < queries.size(); ++c) {
            auto kr_c = rotate_cipher(keys_hybrid[c], static_cast<int>(r));
            auto term = rescale_cipher(mul_cipher(queries[c], kr_c));
            terms.push_back(std::move(term));
        }

        if (terms.size() == 1) {
            out.push_back(std::move(terms.front()));
            continue;
        }

        if (ctx->gpu_initialized && ctx->gpu_context) {
            std::vector<ckks::CtAccurate> gpu_terms;
            gpu_terms.reserve(terms.size());
            for (const auto& term : terms) {
                term->ensureGPU();
                gpu_terms.push_back(*term->gpu_ct);
            }
            ckks::CtAccurate reduced = reduce_add_gpu(ctx, gpu_terms);
            out.push_back(make_cipher_from_gpu_lazy(ctx, std::move(reduced), terms.front()->ciphertext));
            continue;
        }

        auto acc = std::move(terms.front());
        for (std::size_t i = 1; i < terms.size(); ++i) {
            acc = add_cipher(acc, terms[i]);
        }
        out.push_back(std::move(acc));
    }

    return out;
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
          py::arg("batch_size"), py::arg("enable_gpu") = true,
          py::call_guard<py::gil_scoped_release>());
    m.def("keygen", &keygen,
          py::arg("context"), py::arg("rotations"),
          py::arg("relin"), py::arg("conj"),
          py::arg("enable_gpu") = true,
          py::call_guard<py::gil_scoped_release>());
    m.def("encrypt", &encrypt, py::arg("context"), py::arg("keys"), py::arg("values"),
          py::call_guard<py::gil_scoped_release>());
    m.def("decrypt", &decrypt, py::arg("context"), py::arg("keys"), py::arg("cipher"),
          py::call_guard<py::gil_scoped_release>());
    m.def("add_cipher", &add_cipher, py::arg("lhs"), py::arg("rhs"),
          py::call_guard<py::gil_scoped_release>());
    m.def("add_plain", &add_plain, py::arg("lhs"), py::arg("plain"),
          py::call_guard<py::gil_scoped_release>());
    m.def("sub_cipher", &sub_cipher, py::arg("lhs"), py::arg("rhs"),
          py::call_guard<py::gil_scoped_release>());
    m.def("sub_plain", &sub_plain, py::arg("lhs"), py::arg("plain"),
          py::call_guard<py::gil_scoped_release>());
    m.def("mul_cipher", &mul_cipher, py::arg("lhs"), py::arg("rhs"),
          py::call_guard<py::gil_scoped_release>());
    m.def("mul_plain", &mul_plain, py::arg("lhs"), py::arg("plain"),
          py::call_guard<py::gil_scoped_release>());
    m.def("square", &square_cipher, py::arg("cipher"),
          py::call_guard<py::gil_scoped_release>());
    m.def("rescale", &rescale_cipher, py::arg("tensor"),
          py::call_guard<py::gil_scoped_release>());
    m.def("rotate", &rotate_cipher, py::arg("tensor"), py::arg("index"),
          py::call_guard<py::gil_scoped_release>());
    m.def("sum_slots", &sum_slots_cipher, py::arg("tensor"),
          py::call_guard<py::gil_scoped_release>());
    m.def("matvec_diag", &matvec_diag_cipher, py::arg("tensor"), py::arg("diagonals"),
          py::call_guard<py::gil_scoped_release>());
    m.def("matmul_dense", &matmul_dense_cipher, py::arg("tensor"), py::arg("matrix"),
          py::call_guard<py::gil_scoped_release>());
    m.def("matmul_bsgs", &matmul_bsgs_cipher, py::arg("tensor"), py::arg("matrix"),
          py::arg("bsgs_n1") = 0, py::arg("bsgs_n2") = 0,
          py::arg("weight_hash") = 0, py::arg("diag_nonzero") = std::vector<bool>{},
          py::call_guard<py::gil_scoped_release>());
    m.def("packed_self_attention_power", &packed_self_attention_power_cipher,
          py::arg("query"), py::arg("key"), py::arg("value"),
          py::arg("batch_size"), py::arg("seq_len"), py::arg("embed_dim"),
          py::arg("scale"), py::arg("shift"),
          py::arg("reciprocal_coeffs"), py::arg("renorm_reciprocal_coeffs"),
          py::call_guard<py::gil_scoped_release>());
    m.def("halved_ccmm_fused", &halved_ccmm_fused,
          py::arg("queries"), py::arg("keys_hybrid"), py::arg("half_seq_len"),
          py::call_guard<py::gil_scoped_release>());
    m.def("poly_eval", &poly_eval_cipher, py::arg("tensor"), py::arg("coeffs"),
          py::call_guard<py::gil_scoped_release>());
    m.def("conjugate", &conjugate_cipher, py::arg("tensor"),
          py::call_guard<py::gil_scoped_release>());
    m.def("bootstrap", &bootstrap_cipher, py::arg("tensor"),
          py::call_guard<py::gil_scoped_release>());
    m.def("cipher_metadata", &cipher_metadata, py::arg("cipher"));
    m.def("rotate_debug_compare", &rotate_debug_compare,
          py::arg("tensor"), py::arg("rotation"), py::arg("coeff_limit") = 16);
    m.def("fast_rotate_ext_debug_compare", &fast_rotate_ext_debug_compare,
          py::arg("tensor"), py::arg("rotation"), py::arg("add_first"), py::arg("coeff_limit") = 16);
    m.def("bsgs_block_debug_compare", &bsgs_block_debug_compare,
          py::arg("tensor"), py::arg("matrix"), py::arg("bsgs_n1") = 0, py::arg("bsgs_n2") = 0,
          py::arg("coeff_limit") = 8);
    m.def("ext_mult_term_debug_compare", &ext_mult_term_debug_compare,
          py::arg("tensor"), py::arg("diag"), py::arg("rotation"), py::arg("add_first"), py::arg("coeff_limit") = 8);
    m.def("cpu_ext_term_vs_standard_debug", &cpu_ext_term_vs_standard_debug,
          py::arg("tensor"), py::arg("diag"), py::arg("rotation"), py::arg("coeff_limit") = 8);
    m.def("rescale_debug_compare", &rescale_debug_compare,
          py::arg("tensor"), py::arg("coeff_limit") = 8);
    m.def("is_gpu_available", &is_gpu_available);
    m.def("get_gpu_info", &get_gpu_info);
    m.def("set_plain_cache_limit", &set_plain_cache_limit,
          py::arg("context"), py::arg("limit"),
          py::call_guard<py::gil_scoped_release>());
    m.def("pin_plain_cache", &pin_plain_cache, py::arg("context"),
          py::call_guard<py::gil_scoped_release>());
    m.def("clear_pinned_plain_cache", &clear_pinned_plain_cache, py::arg("context"),
          py::call_guard<py::gil_scoped_release>());
    m.def("plain_cache_stats_count", &plain_cache_stats_count, py::arg("context"),
          py::call_guard<py::gil_scoped_release>());
    m.def("plain_cache_stats_pinned", &plain_cache_stats_pinned, py::arg("context"),
          py::call_guard<py::gil_scoped_release>());
}
