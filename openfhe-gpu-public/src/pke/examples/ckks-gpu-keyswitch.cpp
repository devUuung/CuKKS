#include "openfhe.h"

#include "gpu/Utils.h"

using namespace lbcrypto;

void minimumFailingTest(const std::shared_ptr<lbcrypto::CryptoParametersRNS>& cryptoParams, const ckks::Context& gpu_context, const bool verbose = false) {
    // std::cout << "Minimum failing test\n";
    const auto elementParams = cryptoParams->GetElementParams();
    // using DugType  = typename DCRTPoly::DugType;
    typename DCRTPoly::DugType dug;
    DCRTPoly inverseNTTInput(dug, elementParams, Format::EVALUATION);
    ckks::DeviceVector inverse_ntt_input = loadIntoDeviceVector(inverseNTTInput);
    ckks::HostVector inverse_ntt_host(inverse_ntt_input);
    for (int i = 0; i < gpu_context.alpha__; i++)
        assert(NativeInteger(inverse_ntt_host[i*gpu_context.degree__]) == inverseNTTInput.m_vectors[i].m_values->at(0));

    if (verbose) std::cout << "Basic input scan passed\n";

    assert(inverseNTTInput.GetFormat() == Format::EVALUATION);
    inverseNTTInput.SwitchFormat();
    assert(inverseNTTInput.GetFormat() == Format::COEFFICIENT);
    const auto elemParams = inverseNTTInput.GetParams();
    const auto limbParams = elemParams->GetParams();

    if (verbose) {
        for (size_t i = 0; i < limbParams.size(); i++) {
            std::cout << limbParams[i]->GetModulus() << " " << limbParams[i]->GetRootOfUnity() << std::endl;
            // std::cout << gpu_context.param__.primes_[i] << " " << gpu_context.param__.primes_[i] << std::endl;
        }
    }

    gpu_context.FromNTTInplace(inverse_ntt_input, 0, gpu_context.alpha__);
    ckks::HostVector toCheck(inverse_ntt_input);
    for (int i = 0; i < gpu_context.alpha__; i++) {
        if (verbose) std::cout << i << " " << toCheck[i*gpu_context.degree__] << " " << inverseNTTInput.m_vectors[i].m_values->at(0) << std::endl;
        assert(NativeInteger(toCheck[i*gpu_context.degree__]) == inverseNTTInput.m_vectors[i].m_values->at(0));
    }
    std::cout << "Basic inverse NTT test passed\n";
}

void test_deeper_mult_circuit() {

    // CPU Boiler

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
    Ciphertext<DCRTPoly> sqr1(ciphertext1);
    assert(sqr1 == ciphertext1);
    // auto sqr1 = cc->EvalSquare(ciphertext1);
    cc->EvalSquareInPlace(sqr1);
    auto result = cc->EvalMult(depth1, sqr1);

    // GPU Boiler

    const auto cryptoParams = std::dynamic_pointer_cast<CryptoParametersRNS>(cc->GetCryptoParameters());
    const auto gpu_context = GenGPUContext(cryptoParams);
    minimumFailingTest(cryptoParams, gpu_context);
    const auto gpu_evk = LoadEvalMultRelinKey(cc);

    // GPU Computation

    const auto ct1 = LoadCiphertext(ciphertext1);
    const auto ct2 = LoadCiphertext(ciphertext2);

    const auto gpu_depth1 = gpu_context.EvalMultAndRelin(ct1, ct2, gpu_evk);
    auto gpu_result = gpu_context.EvalSquareAndRelin(ct1, gpu_evk);
    gpu_result = gpu_context.EvalMultAndRelin(gpu_depth1, gpu_result, gpu_evk);

    // Check result
    const auto resParams = result->GetElements()[0].GetParams();
    DCRTPoly gpu_res_0 = loadIntoDCRTPoly(gpu_result.bx__, resParams);
    DCRTPoly gpu_res_1 = loadIntoDCRTPoly(gpu_result.ax__, resParams);

    assert(gpu_res_0 == result->GetElements()[0]);
    assert(gpu_res_1 == result->GetElements()[1]);

    std::cout << "depth two mult test passed\n";
}


int main(int argc, char* argv[]) {

    test_deeper_mult_circuit();

    CCParams<CryptoContextCKKSRNS> parameters;
    // const size_t levels = 6;
    const size_t levels = 16;
    parameters.SetMultiplicativeDepth(levels);
    parameters.SetScalingModSize(50);

    CryptoContext<DCRTPoly> cc = GenCryptoContext(parameters);
    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);
    cc->Enable(ADVANCEDSHE);

    std::vector<std::complex<double>> input({0.5, 0.7, 0.9, 0.95, 0.93});
    std::vector<double> coefficients1({0.15, 0.75, 0, 1.25, 0, 0, 1, 0, 1, 2, 0, 1, 0, 0, 0, 0, 1});

    size_t encodedLength = input.size();
    Plaintext plaintext1 = cc->MakeCKKSPackedPlaintext(input);
    Plaintext plaintext2 = cc->MakeCKKSPackedPlaintext(coefficients1);

    auto keyPair = cc->KeyGen();

    cc->EvalMultKeyGen(keyPair.secretKey);

    const auto ciphertext1 = cc->Encrypt(keyPair.publicKey, plaintext1);
    const auto ciphertext2 = cc->Encrypt(keyPair.publicKey, plaintext2);

    // {
    //     std::cout << "input params:\n";
    //     auto limbParams = ciphertext1->GetElements()[0].GetParams()->GetParams();
    //     for (auto& limb : limbParams) {
    //         std::cout << limb->GetModulus()  << " " << limb->GetRootOfUnity() << std::endl;
    //     }
    // }

    // std::cout << "EvalMult input numLimbs: " << ciphertext1->GetElements()[0].m_vectors.size() << std::endl;
    auto result = cc->EvalMult(ciphertext1, ciphertext2);
    // {
    //     std::cout << "result params:\n";
    //     auto limbParams = result->GetElements()[0].GetParams()->GetParams();
    //     for (auto& limb : limbParams) {
    //         std::cout << limb->GetModulus() << " " << limb->GetRootOfUnity() << std::endl;
    //     }
    // }


    // const std::vector<DCRTPoly> constant_ct_1_elems = ciphertext1->GetElements();
    // const std::vector<DCRTPoly> constant_ct_2_elems = ciphertext2->GetElements();
    Ciphertext<DCRTPoly> toModDown1 = ciphertext1->Clone();
    Ciphertext<DCRTPoly> toModDown2 = ciphertext2->Clone();
    cc->GetScheme()->ModReduceInternalInPlace(toModDown1, 1);
    cc->GetScheme()->ModReduceInternalInPlace(toModDown2, 1);

    const std::vector<DCRTPoly> ct1Elems = toModDown1->GetElements();
    const std::vector<DCRTPoly> ct2Elems = toModDown2->GetElements();
    assert(ct1Elems.size() == 2);
    assert(ct2Elems.size() == 2);

    DCRTPoly toRelin0 = ct1Elems[0] * ct2Elems[0];  // this part has the message
    DCRTPoly toRelin1 = ct1Elems[0] * ct2Elems[1] + ct1Elems[1] * ct2Elems[0];
    DCRTPoly toRelin2 = ct1Elems[1] * ct2Elems[1];  // this is the part that gets decomposed.

    auto algo = ciphertext1->GetCryptoContext()->GetScheme();

    auto evks = cc->GetEvalMultKeyVector(ciphertext1->GetKeyTag());
    // std::cout << "length of eval key vector: " << evks.size() << std::endl;
    EvalKey<DCRTPoly> evk = evks[0];

    std::shared_ptr<std::vector<DCRTPoly>> ab = algo->KeySwitchCore(toRelin2, evk);
    {
        // step through key-switch core steps

        auto ks_input = algo->EvalKeySwitchPrecomputeCore(toRelin2, evk->GetCryptoParameters());  // ModRaise
        auto should_be_ab = algo->EvalFastKeySwitchCore(ks_input, evk, toRelin2.GetParams());

        assert(should_be_ab->size() == 2);
        assert(should_be_ab->at(0) == ab->at(0));
        assert(should_be_ab->at(1) == ab->at(1));
    }

    assert(ab->size() == 2);

    DCRTPoly should_be_result_0 = toRelin0 + (*ab)[0];
    DCRTPoly should_be_result_1 = toRelin1 + (*ab)[1];

    assert(should_be_result_0 == result->GetElements()[0]);
    assert(should_be_result_1 == result->GetElements()[1]);

    Plaintext plaintextDec;
    cc->Decrypt(keyPair.secretKey, result, &plaintextDec);
    plaintextDec->SetLength(encodedLength);
    std::cout << plaintextDec << std::endl;

    // const auto cryptoParams = std::dynamic_pointer_cast<CryptoParametersRNS>(cc->GetCryptoParameters());
    // // const auto gpu_context = cryptoParams->GenGPUContext();
    // auto elementParams = cc->GetElementParams();
    // auto limb_params = elementParams->GetParams();
    // auto ext_params = cryptoParams->GetParamsP()->GetParams();

    // // std::cout << "alpha: " << cryptoParams->GetNumPerPartQ() << std::endl;
    // // std::cout << "real dnum: " << cryptoParams->GetNumberOfQPartitions() << std::endl;

    // // std::cout << "log(n) = " << log2(limb_params[0]->GetRingDimension()) << std::endl;
    // const size_t n = limb_params[0]->GetRingDimension();
    // const size_t logn = log2(n);
    // // const size_t numModuli = limb_params.size();
    // // std::cout << "num moduli = " << numModuli << std::endl;        

    // std::vector<uint64_t> limb_moduli;
    // std::vector<uint64_t> limb_rous;
    // for (const auto& limb : limb_params) {
    //     limb_moduli.push_back((uint64_t)limb->GetModulus());
    //     limb_rous.push_back((uint64_t)limb->GetRootOfUnity());
    //     // std::cout << limb->GetModulus() << " " << limb->GetRootOfUnity() << std::endl;
    // }

    // for (const auto& limb : ext_params) {
    //     limb_moduli.push_back((uint64_t)limb->GetModulus());
    //     limb_rous.push_back((uint64_t)limb->GetRootOfUnity());
    //     // std::cout << limb->GetModulus() << " " << limb->GetRootOfUnity() << std::endl;
    // }

    // ckks::Parameter gpu_params(logn, 8, 3, limb_moduli);
    // ckks::Context gpu_context(gpu_params, limb_rous);

    const auto cryptoParams = std::dynamic_pointer_cast<CryptoParametersRNS>(cc->GetCryptoParameters());
    ckks::Context gpu_context = GenGPUContext(cryptoParams);
    minimumFailingTest(cryptoParams, gpu_context);


    std::vector<DCRTPoly> evk_a = evk->GetAVector();
    std::vector<DCRTPoly> evk_b = evk->GetBVector();
    // std::cout << "evk size: " << evk_a.size() << " " << evk_b.size() << std::endl;

    std::cout << "Step 1: Rescale input ciphertexts\n";

    ckks::Ciphertext ct1, ct2;

    ct1.bx__ = loadIntoDeviceVector(ciphertext1->GetElements()[0]);
    ct1.ax__ = loadIntoDeviceVector(ciphertext1->GetElements()[1]);
    ct2.bx__ = loadIntoDeviceVector(ciphertext2->GetElements()[0]);
    ct2.ax__ = loadIntoDeviceVector(ciphertext2->GetElements()[1]);
    // std::cout << ciphertext1->GetElements()[0].GetRingDimension() << " " << ciphertext1->GetElements()[0].GetParams()->m_params.size() << std::endl;
    // assert(ct1_0.size() == ciphertext1->GetElements()[0].GetRingDimension()*ciphertext1->GetElements()[0].GetParams()->m_params.size());

    ckks::Ciphertext multInput1, multInput2;
    gpu_context.Rescale(ct1, multInput1);
    gpu_context.Rescale(ct2, multInput2);
    {
        DCRTPoly shouldBeModDown1_0 = loadIntoDCRTPoly(multInput1.bx__, ct1Elems[0].GetParams(), Format::EVALUATION);
        assert(shouldBeModDown1_0 == toModDown1->GetElements()[0]);
        DCRTPoly shouldBeModDown1_1 = loadIntoDCRTPoly(multInput1.ax__, ct1Elems[1].GetParams(), Format::EVALUATION);
        assert(shouldBeModDown1_1 == toModDown1->GetElements()[1]);

        DCRTPoly shouldBeModDown2_0 = loadIntoDCRTPoly(multInput2.bx__, ct2Elems[0].GetParams(), Format::EVALUATION);
        assert(shouldBeModDown2_0 == toModDown2->GetElements()[0]);
        DCRTPoly shouldBeModDown2_1 = loadIntoDCRTPoly(multInput2.ax__, ct2Elems[1].GetParams(), Format::EVALUATION);
        assert(shouldBeModDown2_1 == toModDown2->GetElements()[1]);
    }


    std::cout << "Step 2: Multiply to get three-term ciphertext\n";

    ckks::DeviceVector gpu_to_relin_2;
    ckks::Ciphertext orig_elems;
    gpu_context.EvalMult(multInput1, multInput2, orig_elems.bx__, orig_elems.ax__, gpu_to_relin_2);
    {
        DCRTPoly shouldBeToRelin0 = loadIntoDCRTPoly(orig_elems.bx__, toRelin0.GetParams());
        assert(shouldBeToRelin0 == toRelin0);
        DCRTPoly shouldBeToRelin1 = loadIntoDCRTPoly(orig_elems.ax__, toRelin1.GetParams());
        assert(shouldBeToRelin1 == toRelin1);
        DCRTPoly shouldBeToRelin2 = loadIntoDCRTPoly(gpu_to_relin_2, toRelin1.GetParams());
        assert(shouldBeToRelin2 == toRelin2);
    }

    std::cout << "Step 3: Mod Up last element\n";

    ckks::DeviceVector raisedDigits = gpu_context.ModUp(gpu_to_relin_2);
    {
        gpu_context.is_modup_batched = !gpu_context.is_modup_batched;
        ckks::DeviceVector should_be_raisedDigits = gpu_context.ModUp(gpu_to_relin_2);
        assert(should_be_raisedDigits == raisedDigits);
        gpu_context.is_modup_batched = !gpu_context.is_modup_batched;
    }
    {
        // step through the correct ModUp operation
        auto ks_input = algo->EvalKeySwitchPrecomputeCore(toRelin2, evk->GetCryptoParameters());  // ModRaise
        // DCRTPoly raisedDigitsPoly = loadIntoDCRTPoly() 
        const size_t phim = ks_input->at(0).GetRingDimension();
        const size_t numRaisedLimbs = ks_input->at(0).m_params->GetParams().size();
        // std::cout << phim << " " << numRaisedLimbs << " " << ks_input->size() << std::endl;
        ckks::HostVector raisedDigitsHost(raisedDigits);
        // std::cout << raisedDigitsHost.size() / (phim * ks_input->size()) << std::endl;
        assert(raisedDigitsHost.size() == ks_input->size()*phim*numRaisedLimbs);
        for (size_t digit_idx = 0; digit_idx < ks_input->size(); digit_idx++) {
            // std::cout << "checking digit " << digit_idx << std::endl;
            for (size_t limbInd = 0; limbInd < numRaisedLimbs; limbInd++) {
                // std::cout << "checking limb " << limbInd << std::endl;
                for (size_t data_ind = 0; data_ind < phim; data_ind++) {
                    // std::cout << digit_idx << " " << limbInd << " " << data_ind << std::endl;
                    assert(NativeInteger(raisedDigitsHost[digit_idx*(numRaisedLimbs*phim) + limbInd*phim + data_ind]) == ks_input->at(digit_idx).m_vectors[limbInd].m_values->at(data_ind));
                }
            }
        }
    }


    std::cout << "Step 4: Inner product with evaluation key\n";

    ckks::EvaluationKey gpu_evk;
    gpu_evk.ax__ = loadIntoDeviceVector(evk_a);
    gpu_evk.bx__ = loadIntoDeviceVector(evk_b);

    ckks::DeviceVector ks_a, ks_b;

    gpu_context.KeySwitch(raisedDigits, gpu_evk, ks_a, ks_b);
    // {  
        // std::cout << "unfused branch is currently disabled\n"; assert(false);
    //     gpu_context.is_keyswitch_fused = !gpu_context.is_keyswitch_fused;
    //     ckks::DeviceVector should_be_ks_a, should_be_ks_b;
    //     gpu_context.KeySwitch(raisedDigits, gpu_evk, should_be_ks_a, should_be_ks_b);
    //     assert(ks_a == should_be_ks_a);
    //     assert(ks_b == should_be_ks_b);
    //     gpu_context.is_keyswitch_fused = !gpu_context.is_keyswitch_fused;
    // }
    {
        // step through key-switch core steps
        auto ks_input = algo->EvalKeySwitchPrecomputeCore(toRelin2, evk->GetCryptoParameters());  // ModRaise
        auto inner_product_out = algo->EvalFastKeySwitchCoreExt(ks_input, evk, toRelin2.GetParams());

        assert(inner_product_out->size() == 2);

        DCRTPoly ks_b_poly = loadIntoDCRTPoly(ks_b, inner_product_out->at(0).GetParams());
        // this should be index 0
        assert(ks_b_poly == inner_product_out->at(0));

        DCRTPoly ks_a_poly = loadIntoDCRTPoly(ks_a, inner_product_out->at(1).GetParams());
        // this should be index 1
        assert(ks_a_poly == inner_product_out->at(1));
    }

    std::cout << "Step 5: ModDown\n";

    ckks::Ciphertext ks_output;
    gpu_context.ModDown(ks_a, ks_output.ax__);
    gpu_context.ModDown(ks_b, ks_output.bx__);
    {
        DCRTPoly ks_b_poly = loadIntoDCRTPoly(ks_output.bx__, ab->at(0).GetParams(), Format::EVALUATION);
        assert(ks_b_poly == ab->at(0));
        DCRTPoly ks_a_poly = loadIntoDCRTPoly(ks_output.ax__, ab->at(1).GetParams(), Format::EVALUATION);
        assert(ks_a_poly == ab->at(1));
    }

    std::cout << "Step 6: Add\n";

    ckks::Ciphertext toLoad; 
    gpu_context.Add(ks_output, orig_elems, toLoad);

    // just use the correct parameters for now.
    const auto resParams = result->GetElements()[0].GetParams();
    DCRTPoly gpu_res_0 = loadIntoDCRTPoly(toLoad.bx__, resParams);
    DCRTPoly gpu_res_1 = loadIntoDCRTPoly(toLoad.ax__, resParams);

    assert(gpu_res_0 == result->GetElements()[0]);
    assert(gpu_res_1 == result->GetElements()[1]);

    std::cout << "Full test\n";

    ckks::Ciphertext fullRes = gpu_context.EvalMultAndRelin(ct1, ct2, gpu_evk);

    std::vector<DCRTPoly> gpu_res_elems(2);
    // gpu_res_elems[0] = gpu_res_0;
    // gpu_res_elems[1] = gpu_res_1;
    gpu_res_elems[0] = loadIntoDCRTPoly(fullRes.bx__, resParams);
    gpu_res_elems[1] = loadIntoDCRTPoly(fullRes.ax__, resParams);

    Ciphertext<DCRTPoly> gpu_result = result->Clone();
    gpu_result->SetElements(gpu_res_elems);
    // gpu_result->SetElements(gpu_res_elems);

    // std::cout << "created gpu result ciphertext\n";

    Plaintext plaintextDecGPU;
    cc->Decrypt(keyPair.secretKey, gpu_result, &plaintextDecGPU);
    plaintextDecGPU->SetLength(encodedLength);
    std::cout << plaintextDecGPU << std::endl;
}