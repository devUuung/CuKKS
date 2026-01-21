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

namespace py = pybind11;
using namespace lbcrypto;

namespace {

struct ContextHandle {
    CryptoContext<DCRTPoly> context;
    std::vector<uint32_t> level_budget;
    bool bootstrap_enabled;
    
    // GPU bootstrap support
    std::unique_ptr<ckks::Context> gpu_context;
    std::unique_ptr<ckks::EvaluationKey> gpu_evk;
    std::unique_ptr<std::map<uint32_t, ckks::EvaluationKey>> gpu_rot_keys;
};

struct KeySetHandle {
    std::shared_ptr<ContextHandle> context;
    PublicKey<DCRTPoly> public_key;
    PrivateKey<DCRTPoly> secret_key;
};

struct CiphertextHandle {
    std::shared_ptr<ContextHandle> context;
    Ciphertext<DCRTPoly> ciphertext;
};

std::shared_ptr<CiphertextHandle> make_cipher(const std::shared_ptr<ContextHandle>& ctx,
                                              const Ciphertext<DCRTPoly>& cipher) {
    auto handle = std::make_shared<CiphertextHandle>();
    handle->context     = ctx;
    handle->ciphertext  = cipher;
    return handle;
}

Plaintext make_plaintext(const std::shared_ptr<ContextHandle>& ctx, const std::vector<double>& values) {
    auto plaintext = ctx->context->MakeCKKSPackedPlaintext(values);
    return plaintext;
}

py::dict cipher_metadata(const std::shared_ptr<CiphertextHandle>& handle) {
    py::dict meta;
    auto cc = handle->context->context;
    meta["scale"] = handle->ciphertext->GetScalingFactor();
    meta["level"] = handle->ciphertext->GetLevel();
    meta["noise_scale"] = handle->ciphertext->GetNoiseScaleDeg();
    meta["slots"] = static_cast<std::uint32_t>(cc->GetRingDimension() / 2);
    auto params = std::dynamic_pointer_cast<CryptoParametersCKKSRNS>(cc->GetCryptoParameters());
    if (params) {
        meta["base_scale"] = params->GetScalingFactorReal();
        meta["max_level"] = static_cast<std::uint32_t>(params->GetElementParams()->GetParams().size() - 1);
    } else {
        meta["base_scale"] = handle->ciphertext->GetScalingFactor();
        meta["max_level"] = handle->ciphertext->GetLevel();
    }
    return meta;
}

}  // namespace

std::shared_ptr<ContextHandle> create_context(std::uint32_t poly_mod_degree,
                                              const std::vector<std::uint32_t>& coeff_mod_bits,
                                              std::uint32_t scale_bits,
                                              std::uint32_t security_level_code,
                                              bool enable_bootstrap,
                                              const std::vector<std::uint32_t>& level_budget,
                                              std::uint32_t batch_size) {
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
    if (enable_bootstrap) {
        context->Enable(FHE);
        if (!level_budget.empty()) {
            context->EvalBootstrapSetup(level_budget);
        }
    }

    auto handle = std::make_shared<ContextHandle>();
    handle->context = context;
    handle->bootstrap_enabled = enable_bootstrap;
    handle->level_budget = level_budget;
    return handle;
}

std::shared_ptr<KeySetHandle> keygen(const std::shared_ptr<ContextHandle>& ctx,
                                     const std::vector<int>& rotations,
                                     bool relin,
                                     bool conj) {
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
        std::uint32_t num_slots = context->GetRingDimension() / 2;
        context->EvalBootstrapKeyGen(key_pair.secretKey, num_slots);
    }

    auto handle = std::make_shared<KeySetHandle>();
    handle->context = ctx;
    handle->public_key = key_pair.publicKey;
    handle->secret_key = key_pair.secretKey;
    return handle;
}

std::shared_ptr<CiphertextHandle> encrypt(const std::shared_ptr<ContextHandle>& ctx,
                                          const std::shared_ptr<KeySetHandle>& keys,
                                          const std::vector<double>& values) {
    auto plaintext = make_plaintext(ctx, values);
    auto cipher = ctx->context->Encrypt(keys->public_key, plaintext);
    return make_cipher(ctx, cipher);
}

std::vector<double> decrypt(const std::shared_ptr<ContextHandle>& ctx,
                            const std::shared_ptr<KeySetHandle>& keys,
                            const std::shared_ptr<CiphertextHandle>& cipher) {
    Plaintext result;
    ctx->context->Decrypt(keys->secret_key, cipher->ciphertext, &result);
    result->SetLength(result->GetCKKSPackedValue().size());
    return result->GetRealPackedValue();
}

std::shared_ptr<CiphertextHandle> add_cipher(const std::shared_ptr<CiphertextHandle>& lhs,
                                             const std::shared_ptr<CiphertextHandle>& rhs) {
    auto result = lhs->context->context->EvalAdd(lhs->ciphertext, rhs->ciphertext);
    return make_cipher(lhs->context, result);
}

std::shared_ptr<CiphertextHandle> add_plain(const std::shared_ptr<CiphertextHandle>& lhs,
                                            const std::vector<double>& plain) {
    auto plaintext = make_plaintext(lhs->context, plain);
    auto result = lhs->context->context->EvalAdd(lhs->ciphertext, plaintext);
    return make_cipher(lhs->context, result);
}

std::shared_ptr<CiphertextHandle> sub_cipher(const std::shared_ptr<CiphertextHandle>& lhs,
                                             const std::shared_ptr<CiphertextHandle>& rhs) {
    auto result = lhs->context->context->EvalSub(lhs->ciphertext, rhs->ciphertext);
    return make_cipher(lhs->context, result);
}

std::shared_ptr<CiphertextHandle> sub_plain(const std::shared_ptr<CiphertextHandle>& lhs,
                                            const std::vector<double>& plain) {
    auto plaintext = make_plaintext(lhs->context, plain);
    auto result = lhs->context->context->EvalSub(lhs->ciphertext, plaintext);
    return make_cipher(lhs->context, result);
}

std::shared_ptr<CiphertextHandle> mul_cipher(const std::shared_ptr<CiphertextHandle>& lhs,
                                             const std::shared_ptr<CiphertextHandle>& rhs) {
    auto result = lhs->context->context->EvalMultAndRelinearize(lhs->ciphertext, rhs->ciphertext);
    return make_cipher(lhs->context, result);
}

std::shared_ptr<CiphertextHandle> mul_plain(const std::shared_ptr<CiphertextHandle>& lhs,
                                            const std::vector<double>& plain) {
    auto plaintext = make_plaintext(lhs->context, plain);
    auto result = lhs->context->context->EvalMult(lhs->ciphertext, plaintext);
    return make_cipher(lhs->context, result);
}

std::shared_ptr<CiphertextHandle> rescale_cipher(const std::shared_ptr<CiphertextHandle>& tensor) {
    auto result = tensor->context->context->ModReduce(tensor->ciphertext);
    return make_cipher(tensor->context, result);
}

std::shared_ptr<CiphertextHandle> rotate_cipher(const std::shared_ptr<CiphertextHandle>& tensor, int index) {
    auto result = tensor->context->context->EvalAtIndex(tensor->ciphertext, index);
    return make_cipher(tensor->context, result);
}

std::shared_ptr<CiphertextHandle> sum_slots_cipher(const std::shared_ptr<CiphertextHandle>& tensor) {
    std::uint32_t slots = tensor->context->context->GetRingDimension() / 2;
    auto result = tensor->context->context->EvalSum(tensor->ciphertext, slots);
    return make_cipher(tensor->context, result);
}

std::shared_ptr<CiphertextHandle> matvec_diag_cipher(const std::shared_ptr<CiphertextHandle>& tensor,
                                                     const std::vector<std::vector<double>>& diagonals) {
    auto cc = tensor->context->context;
    auto accumulator = cc->EvalMult(tensor->ciphertext, make_plaintext(tensor->context, diagonals.front()));
    for (std::size_t idx = 1; idx < diagonals.size(); ++idx) {
        auto rotated = cc->EvalAtIndex(tensor->ciphertext, static_cast<int>(idx));
        auto term = cc->EvalMult(rotated, make_plaintext(tensor->context, diagonals[idx]));
        accumulator = cc->EvalAdd(accumulator, term);
    }
    return make_cipher(tensor->context, accumulator);
}

std::shared_ptr<CiphertextHandle> poly_eval_cipher(const std::shared_ptr<CiphertextHandle>& tensor,
                                                   const std::vector<double>& coeffs) {
    auto result = tensor->context->context->EvalPoly(tensor->ciphertext, coeffs);
    return make_cipher(tensor->context, result);
}

std::shared_ptr<CiphertextHandle> matmul_dense_cipher(const std::shared_ptr<CiphertextHandle>& tensor,
                                                      const std::vector<std::vector<double>>& matrix) {
    auto cc = tensor->context->context;
    const std::size_t m = matrix.size();
    if (m == 0) {
        throw std::invalid_argument("matrix must not be empty");
    }
    const std::size_t n = matrix.front().size();
    if (n == 0) {
        throw std::invalid_argument("matrix rows must not be empty");
    }
    for (const auto& row : matrix) {
        if (row.size() != n) {
            throw std::invalid_argument("matrix must be rectangular with consistent row sizes");
        }
    }
    const std::size_t slots = static_cast<std::size_t>(cc->GetRingDimension() / 2);
    if (n > slots) {
        throw std::invalid_argument("matrix column dimension exceeds available CKKS slots");
    }

    // Diagonal method for dense (possibly rectangular) matrix-vector multiply.
    Ciphertext<DCRTPoly> accumulator;
    for (std::size_t k = 0; k < n; ++k) {
        std::vector<double> diag(slots, 0.0);
        for (std::size_t i = 0; i < m; ++i) {
            diag[i] = matrix[i][(i + k) % n];
        }
        auto rotated = (k == 0) ? tensor->ciphertext
                                : cc->EvalAtIndex(tensor->ciphertext, static_cast<int>(k));
        auto term = cc->EvalMult(rotated, make_plaintext(tensor->context, diag));
        if (!accumulator) {
            accumulator = term;
        } else {
            accumulator = cc->EvalAdd(accumulator, term);
        }
    }
    if (!accumulator) {
        throw std::runtime_error("failed to build accumulator");
    }
    return make_cipher(tensor->context, accumulator);
}

std::shared_ptr<CiphertextHandle> conjugate_cipher(const std::shared_ptr<CiphertextHandle>& tensor) {
    auto cc = tensor->context->context;
    const auto& key_map = cc->GetEvalAutomorphismKeyMap(tensor->ciphertext->GetKeyTag());
    usint conj_index = static_cast<usint>(2 * cc->GetRingDimension() - 1);
    auto result = cc->EvalAutomorphism(tensor->ciphertext, conj_index, key_map);
    return make_cipher(tensor->context, result);
}

std::shared_ptr<CiphertextHandle> matmul_bsgs_cipher(const std::shared_ptr<CiphertextHandle>& tensor,
                                                      const std::vector<std::vector<double>>& matrix,
                                                      std::uint32_t bsgs_n1,
                                                      std::uint32_t bsgs_n2) {
    auto cc = tensor->context->context;
    const std::size_t out_features = matrix.size();
    if (out_features == 0) {
        throw std::invalid_argument("matrix must not be empty");
    }
    const std::size_t in_features = matrix.front().size();
    if (in_features == 0) {
        throw std::invalid_argument("matrix rows must not be empty");
    }
    const std::size_t slots = static_cast<std::size_t>(cc->GetRingDimension() / 2);
    
    if (bsgs_n1 == 0) {
        bsgs_n1 = static_cast<std::uint32_t>(std::ceil(std::sqrt(static_cast<double>(in_features))));
    }
    if (bsgs_n2 == 0) {
        bsgs_n2 = (static_cast<std::uint32_t>(in_features) + bsgs_n1 - 1) / bsgs_n1;
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
            for (std::size_t i = 0; i < slots; ++i) {
                std::int64_t row_signed = static_cast<std::int64_t>(i) - static_cast<std::int64_t>(giant_step);
                std::size_t row = static_cast<std::size_t>((row_signed % static_cast<std::int64_t>(out_features) + static_cast<std::int64_t>(out_features)) % static_cast<std::int64_t>(out_features));
                std::size_t col = (i + j) % in_features;
                diag[i] = matrix[row][col];
            }
            
            auto term = cc->EvalMult(baby_ciphers[j], make_plaintext(tensor->context, diag));
            
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
    
    if (!accumulator) {
        throw std::runtime_error("failed to build accumulator in BSGS matmul");
    }
    return make_cipher(tensor->context, accumulator);
}

std::shared_ptr<CiphertextHandle> bootstrap_cipher(const std::shared_ptr<CiphertextHandle>& tensor) {
    auto result = tensor->context->context->EvalBootstrap(tensor->ciphertext);
    return make_cipher(tensor->context, result);
}

PYBIND11_MODULE(ckks_openfhe_backend, m) {
    py::class_<ContextHandle, std::shared_ptr<ContextHandle>>(m, "ContextHandle")
        .def_property_readonly("ring_dim", [](const ContextHandle& handle) {
            return handle.context->GetRingDimension();
        });

    py::class_<KeySetHandle, std::shared_ptr<KeySetHandle>>(m, "KeySetHandle");

    py::class_<CiphertextHandle, std::shared_ptr<CiphertextHandle>>(m, "CiphertextHandle");

    m.def("create_context", &create_context, py::arg("poly_mod_degree"), py::arg("coeff_mod_bits"),
          py::arg("scale_bits"), py::arg("security_level_code"), py::arg("enable_bootstrap"), py::arg("level_budget"),
          py::arg("batch_size"));
    m.def("keygen", &keygen, py::arg("context"), py::arg("rotations"), py::arg("relin"), py::arg("conj"));
    m.def("encrypt", &encrypt, py::arg("context"), py::arg("keys"), py::arg("values"));
    m.def("decrypt", &decrypt, py::arg("context"), py::arg("keys"), py::arg("cipher"));
    m.def("add_cipher", &add_cipher, py::arg("lhs"), py::arg("rhs"));
    m.def("add_plain", &add_plain, py::arg("lhs"), py::arg("plain"));
    m.def("sub_cipher", &sub_cipher, py::arg("lhs"), py::arg("rhs"));
    m.def("sub_plain", &sub_plain, py::arg("lhs"), py::arg("plain"));
    m.def("mul_cipher", &mul_cipher, py::arg("lhs"), py::arg("rhs"));
    m.def("mul_plain", &mul_plain, py::arg("lhs"), py::arg("plain"));
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
}
