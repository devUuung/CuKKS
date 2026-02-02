#include "openfhe.h"

#include "gpu/Utils.h"

using namespace lbcrypto;

/*
    This should be essentially the same as the basic multiplication test, 
    but now we're also keeping track of the scaling factor and other accuracy-preserving data.

    The following data should be tracked and checked:
        - Level
        - NoiseScaleDeg
        - # of elements (should basically always be 2)
        - ScalingFactor
        - ScalingFactorReal
        - ModReduceFactor
*/

void basic_sub_test() {
    CCParams<CryptoContextCKKSRNS> parameters;
    const size_t levels = 16;
    parameters.SetMultiplicativeDepth(levels);
    parameters.SetScalingModSize(59);
    parameters.SetFirstModSize(60);

    CryptoContext<DCRTPoly> cc = GenCryptoContext(parameters);
    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);
    cc->Enable(ADVANCEDSHE);

    // CPU Computation

    std::vector<std::complex<double>> input({0.5, 0.7, 0.9, 0.95, 0.93});
    std::vector<double> coefficients1({0.15, 0.75, 0, 1.25, 0, 0, 1, 0, 1, 2, 0, 1, 0, 0, 0, 0, 1});

    // size_t encodedLength = input.size();
    Plaintext plaintext1 = cc->MakeCKKSPackedPlaintext(input);
    Plaintext plaintext2 = cc->MakeCKKSPackedPlaintext(coefficients1);

    auto keyPair = cc->KeyGen();

    // cc->EvalMultKeyGen(keyPair.secretKey);

    const auto ciphertext1 = cc->Encrypt(keyPair.publicKey, plaintext1);
    const auto ciphertext2 = cc->Encrypt(keyPair.publicKey, plaintext2);

    const auto result = cc->EvalSub(ciphertext1, ciphertext2);

    // Plaintext res = cc->Decrypt(keyPair.secretKey, res_ct);

    // GPU Boiler

    const auto cryptoParams = std::dynamic_pointer_cast<CryptoParametersRNS>(cc->GetCryptoParameters());
    const auto gpu_context = GenGPUContext(cryptoParams);
    // const std::string keyTag = keyPair.publicKey->GetKeyTag();
    // const auto gpu_evk = LoadEvalMultRelinKey(cc, keyTag);

    // GPU Computation

    // const auto ct1 = LoadCiphertext(ciphertext1);
    // const auto ct2 = LoadCiphertext(ciphertext2);
    const auto ct1 = LoadAccurateCiphertext(ciphertext1);
    const auto ct2 = LoadAccurateCiphertext(ciphertext2);

    const auto gpu_result = gpu_context.Sub(ct1, ct2);

    // Check result
    const auto resParams = result->GetElements()[0].GetParams();
    // const auto resParams = depth1->GetElements()[0].GetParams();
    DCRTPoly gpu_res_0 = loadIntoDCRTPoly(gpu_result.bx__, resParams);
    DCRTPoly gpu_res_1 = loadIntoDCRTPoly(gpu_result.ax__, resParams);

    assert(gpu_res_0 == result->GetElements()[0]);
    assert(gpu_res_1 == result->GetElements()[1]);
    // assert(gpu_res_0 == depth1->GetElements()[0]);
    // assert(gpu_res_1 == depth1->GetElements()[1]);
    assert(gpu_result.level == result->GetLevel());
    assert(gpu_result.noiseScaleDeg == result->GetNoiseScaleDeg());
    assert(gpu_result.scalingFactor == result->GetScalingFactor());

    std::cout << "accurate sub test passed\n";
}

void basic_mult_test() {
    CCParams<CryptoContextCKKSRNS> parameters;
    const size_t levels = 16;
    parameters.SetMultiplicativeDepth(levels);
    parameters.SetScalingModSize(59);
    parameters.SetFirstModSize(60);

    CryptoContext<DCRTPoly> cc = GenCryptoContext(parameters);
    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);
    cc->Enable(ADVANCEDSHE);

    // CPU Computation

    std::vector<std::complex<double>> input({0.5, 0.7, 0.9, 0.95, 0.93});
    std::vector<double> coefficients1({0.15, 0.75, 0, 1.25, 0, 0, 1, 0, 1, 2, 0, 1, 0, 0, 0, 0, 1});

    // size_t encodedLength = input.size();
    Plaintext plaintext1 = cc->MakeCKKSPackedPlaintext(input);
    Plaintext plaintext2 = cc->MakeCKKSPackedPlaintext(coefficients1);

    auto keyPair = cc->KeyGen();

    cc->EvalMultKeyGen(keyPair.secretKey);

    const auto ciphertext1 = cc->Encrypt(keyPair.publicKey, plaintext1);
    const auto ciphertext2 = cc->Encrypt(keyPair.publicKey, plaintext2);

    auto depth1 = cc->EvalMult(ciphertext1, ciphertext2);
    // auto sqr1 = cc->EvalMult(ciphertext1, ciphertext1);
    // Ciphertext<DCRTPoly> sqr1(ciphertext1);
    // assert(sqr1 == ciphertext1);
    auto sqr1 = cc->EvalSquare(ciphertext1);
    // cc->EvalSquareInPlace(sqr1);
    auto result = cc->EvalMult(depth1, sqr1);

    // GPU Boiler

    const auto cryptoParams = std::dynamic_pointer_cast<CryptoParametersRNS>(cc->GetCryptoParameters());
    const auto gpu_context = GenGPUContext(cryptoParams);
    const std::string keyTag = keyPair.publicKey->GetKeyTag();
    const auto gpu_evk = LoadEvalMultRelinKey(cc, keyTag);

    // GPU Computation

    // const auto ct1 = LoadCiphertext(ciphertext1);
    // const auto ct2 = LoadCiphertext(ciphertext2);
    const auto ct1 = LoadAccurateCiphertext(ciphertext1);
    const auto ct2 = LoadAccurateCiphertext(ciphertext2);

    // std::cout << "ciphertext1 " << ciphertext1->GetLevel() << " " << ciphertext1->GetNoiseScaleDeg() << " " << ciphertext1->GetScalingFactor() << std::endl;
    // std::cout << "ciphertext2 " << ciphertext2->GetLevel() << " " << ciphertext2->GetNoiseScaleDeg() << " " << ciphertext2->GetScalingFactor() << std::endl;
    // std::cout << "depth1 " << depth1->GetLevel() << " " << depth1->GetNoiseScaleDeg() << " " << depth1->GetScalingFactor() << std::endl;

    const auto gpu_depth1 = gpu_context.EvalMultAndRelin(ct1, ct2, gpu_evk);

    // std::cout << "gpu_depth1 " << gpu_depth1.level << " " << gpu_depth1.noiseScaleDeg << " " << gpu_depth1.scalingFactor << std::endl;

    {
        assert(gpu_depth1.level == depth1->GetLevel());
        assert(gpu_depth1.noiseScaleDeg == depth1->GetNoiseScaleDeg());
        assert(gpu_depth1.scalingFactor == depth1->GetScalingFactor());
    }

    auto gpu_result = gpu_context.EvalSquareAndRelin(ct1, gpu_evk);

    gpu_result = gpu_context.EvalMultAndRelin(gpu_depth1, gpu_result, gpu_evk);

    // Check result
    const auto resParams = result->GetElements()[0].GetParams();
    // const auto resParams = depth1->GetElements()[0].GetParams();
    DCRTPoly gpu_res_0 = loadIntoDCRTPoly(gpu_result.bx__, resParams);
    DCRTPoly gpu_res_1 = loadIntoDCRTPoly(gpu_result.ax__, resParams);

    assert(gpu_res_0 == result->GetElements()[0]);
    assert(gpu_res_1 == result->GetElements()[1]);
    // assert(gpu_res_0 == depth1->GetElements()[0]);
    // assert(gpu_res_1 == depth1->GetElements()[1]);
    assert(gpu_result.level == result->GetLevel());
    assert(gpu_result.noiseScaleDeg == result->GetNoiseScaleDeg());
    assert(gpu_result.scalingFactor == result->GetScalingFactor());

    std::cout << "accurate mult test passed\n";
    // std::cout << "depth two mult test passed\n";
}

void mult_and_add_test() {
    CCParams<CryptoContextCKKSRNS> parameters;
    const size_t levels = 16;
    parameters.SetMultiplicativeDepth(levels);
    parameters.SetScalingModSize(59);
    parameters.SetFirstModSize(60);

    CryptoContext<DCRTPoly> cc = GenCryptoContext(parameters);
    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);
    cc->Enable(ADVANCEDSHE);

    // CPU Computation

    std::vector<std::complex<double>> input({0.5, 0.7, 0.9, 0.95, 0.93});
    std::vector<double> coefficients1({0.15, 0.75, 0, 1.25, 0, 0, 1, 0, 1, 2, 0, 1, 0, 0, 0, 0, 1});

    // size_t encodedLength = input.size();
    Plaintext plaintext1 = cc->MakeCKKSPackedPlaintext(input);
    Plaintext plaintext2 = cc->MakeCKKSPackedPlaintext(coefficients1);

    auto keyPair = cc->KeyGen();

    cc->EvalMultKeyGen(keyPair.secretKey);

    const auto ciphertext1 = cc->Encrypt(keyPair.publicKey, plaintext1);
    const auto ciphertext2 = cc->Encrypt(keyPair.publicKey, plaintext2);

    auto depth1 = cc->EvalMult(ciphertext1, ciphertext2);
    // auto sqr1 = cc->EvalMult(ciphertext1, ciphertext1);
    // Ciphertext<DCRTPoly> sqr1(ciphertext1);
    // assert(sqr1 == ciphertext1);
    auto sqr1 = cc->EvalSquare(ciphertext1);
    auto sqr1_sum = cc->EvalAdd(sqr1, ciphertext1);
    // cc->EvalSquareInPlace(sqr1);
    auto result = cc->EvalMult(depth1, sqr1_sum); // = depth1 + sqr1_sum = (ct1 * ct2) + sqr1 + ct1 = (ct1 * ct2) + ct1^2 + ct1

    // GPU Boiler

    const auto cryptoParams = std::dynamic_pointer_cast<CryptoParametersCKKSRNS>(cc->GetCryptoParameters());
    const auto gpu_context = GenGPUContext(cryptoParams);
    const std::string keyTag = keyPair.publicKey->GetKeyTag();
    const auto gpu_evk = LoadEvalMultRelinKey(cc, keyTag);

    const auto scheme = std::dynamic_pointer_cast<LeveledSHECKKSRNS>(cc->GetScheme());

    // GPU Computation

    // const auto ct1 = LoadCiphertext(ciphertext1);
    // const auto ct2 = LoadCiphertext(ciphertext2);
    const auto ct1 = LoadAccurateCiphertext(ciphertext1);
    const auto ct2 = LoadAccurateCiphertext(ciphertext2);

    const auto gpu_depth1 = gpu_context.EvalMultAndRelin(ct1, ct2, gpu_evk);

    {
        assert(gpu_depth1.level == depth1->GetLevel());
        assert(gpu_depth1.noiseScaleDeg == depth1->GetNoiseScaleDeg());
        assert(gpu_depth1.scalingFactor == depth1->GetScalingFactor());
    }

    auto gpu_sq1 = gpu_context.EvalSquareAndRelin(ct1, gpu_evk);

    {
        assert(gpu_sq1.level == sqr1->GetLevel());
        assert(gpu_sq1.noiseScaleDeg == sqr1->GetNoiseScaleDeg());
        assert(gpu_sq1.scalingFactor == sqr1->GetScalingFactor());
    }

    // std::cout << "ciphertext1 " << ciphertext1->GetLevel() << " " << ciphertext1->GetNoiseScaleDeg() << " " << ciphertext1->GetScalingFactor() << std::endl;
    // std::cout << "sq1 " << sqr1->GetLevel() << " " << sqr1->GetNoiseScaleDeg() << " " << sqr1->GetScalingFactor() << std::endl;
    // std::cout << "sqr1_sum " << sqr1_sum->GetLevel() << " " << sqr1_sum->GetNoiseScaleDeg() << " " << sqr1_sum->GetScalingFactor() << std::endl;

    double ct1_adjust_scalar = gpu_context.GetAdjustScalar(gpu_sq1, ct1);
    const uint32_t ct1_numLimbs = ct1.ax__.size()/gpu_context.degree__;
    const auto ct1_adjust_scalar_dcrt = scheme->GetElementForEvalMult(cryptoParams, ct1.level, ct1_numLimbs, ct1.noiseScaleDeg, ct1_adjust_scalar);
    const auto ct1_adjust_scalar_gpu = LoadIntegerVector(ct1_adjust_scalar_dcrt);
    const auto ct1_adjusted_to_rescale = gpu_context.EvalMultConst(ct1, ct1_adjust_scalar_gpu);
    const auto ct1_to_add = gpu_context.Rescale(ct1_adjusted_to_rescale);
    auto gpu_sq1_sum = gpu_context.Add(gpu_sq1, ct1_to_add);

    // std::cout << "gpu_sq1_sum " << gpu_sq1_sum.level << " " << gpu_sq1_sum.noiseScaleDeg << " " << gpu_sq1_sum.scalingFactor << std::endl;

    {
        assert(gpu_sq1_sum.level == sqr1_sum->GetLevel());
        assert(gpu_sq1_sum.noiseScaleDeg == sqr1_sum->GetNoiseScaleDeg());
        assert(gpu_sq1_sum.scalingFactor == sqr1_sum->GetScalingFactor());
        std::cout << "square sum passed\n";
    }

    auto gpu_result = gpu_context.EvalMultAndRelin(gpu_depth1, gpu_sq1_sum, gpu_evk);

    // Check result
    const auto resParams = result->GetElements()[0].GetParams();
    // const auto resParams = depth1->GetElements()[0].GetParams();
    DCRTPoly gpu_res_0 = loadIntoDCRTPoly(gpu_result.bx__, resParams);
    DCRTPoly gpu_res_1 = loadIntoDCRTPoly(gpu_result.ax__, resParams);

    assert(gpu_res_0 == result->GetElements()[0]);
    assert(gpu_res_1 == result->GetElements()[1]);
    // assert(gpu_res_0 == depth1->GetElements()[0]);
    // assert(gpu_res_1 == depth1->GetElements()[1]);
    assert(gpu_result.level == result->GetLevel());
    assert(gpu_result.noiseScaleDeg == result->GetNoiseScaleDeg());
    assert(gpu_result.scalingFactor == result->GetScalingFactor());

    std::cout << "accurate mult test passed\n";
    // std::cout << "depth two mult test passed\n";
}

void test_automorphism() {
    CCParams<CryptoContextCKKSRNS> parameters;
    const size_t levels = 16;
    parameters.SetMultiplicativeDepth(levels);
    parameters.SetScalingModSize(59);
    parameters.SetFirstModSize(60);

    CryptoContext<DCRTPoly> cc = GenCryptoContext(parameters);
    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);
    cc->Enable(ADVANCEDSHE);

    const auto cryptoParams = std::dynamic_pointer_cast<CryptoParametersCKKSRNS>(cc->GetCryptoParameters());
    auto gpu_context = GenGPUContext(cryptoParams);
    gpu_context.EnableMemoryPool();

    std::vector<std::complex<double>> input({0.5, 0.7, 0.9, 0.95, 0.93});

    // size_t encodedLength = input.size();
    Plaintext plaintext1 = cc->MakeCKKSPackedPlaintext(input);

    auto keyPair = cc->KeyGen();

    const int32_t rot_ind = 2;
    // cc->EvalAutomorphismKeyGen(keyPair.secretKey, {rot_ind});
    cc->EvalAtIndexKeyGen(keyPair.secretKey, {rot_ind}); // Generate EvalKeys for rotations

    const std::map<usint, lbcrypto::EvalKey<lbcrypto::DCRTPoly>> evalKeys = cc->GetEvalAutomorphismKeyMap(keyPair.publicKey->GetKeyTag());

    const auto ciphertext1 = cc->Encrypt(keyPair.publicKey, plaintext1);

    const auto result = cc->EvalAtIndex(ciphertext1, rot_ind);


    // usint autoIndex = cc->FindAutomorphismIndex(rot_ind);
    usint M = cryptoParams->GetElementParams()->GetCyclotomicOrder();
    usint autoIndex = FindAutomorphismIndex2nComplex(rot_ind, M);

    std::cout << "index bash: " << rot_ind << " " << autoIndex << " " << cc->FindAutomorphismIndex(rot_ind) << std::endl;

    const auto evalKey = evalKeys.at(autoIndex);

    const auto evk_gpu = LoadRelinKey(evalKey);

    const auto ct1 = LoadAccurateCiphertext(ciphertext1);

    const auto gpu_result = gpu_context.EvalAtIndex(ct1, evk_gpu, rot_ind);

    // Check result
    const auto resParams = result->GetElements()[0].GetParams();
    // const auto resParams = depth1->GetElements()[0].GetParams();
    DCRTPoly gpu_res_0 = loadIntoDCRTPoly(gpu_result.bx__, resParams);
    DCRTPoly gpu_res_1 = loadIntoDCRTPoly(gpu_result.ax__, resParams);

    assert(gpu_res_1 == result->GetElements()[1]);
    assert(gpu_res_0 == result->GetElements()[0]);
    // assert(gpu_res_0 == depth1->GetElements()[0]);
    // assert(gpu_res_1 == depth1->GetElements()[1]);
    assert(gpu_result.level == result->GetLevel());
    assert(gpu_result.noiseScaleDeg == result->GetNoiseScaleDeg());
    assert(gpu_result.scalingFactor == result->GetScalingFactor());
}

int main(int argc, char* argv[]) {

    basic_sub_test();

    mult_and_add_test();
    basic_mult_test();

    test_automorphism();

}