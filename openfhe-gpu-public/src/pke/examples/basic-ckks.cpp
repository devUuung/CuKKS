#include "openfhe.h"

#include "gpu/Utils.h"

using namespace lbcrypto;


int main(int argc, char* argv[]) {

    CCParams<CryptoContextCKKSRNS> parameters;
    const size_t levels = 6;
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

    auto keyPair = cc->KeyGen();

    cc->EvalMultKeyGen(keyPair.secretKey);

    auto ciphertext1 = cc->Encrypt(keyPair.publicKey, plaintext1);

    auto result = cc->EvalPoly(ciphertext1, coefficients1);

    Plaintext plaintextDec;

    cc->Decrypt(keyPair.secretKey, result, &plaintextDec);

    plaintextDec->SetLength(encodedLength);

    std::cout << "\n Original Plaintext #1: \n";
    std::cout << plaintext1 << std::endl;

    std::cout << "\n Result of evaluating a polynomial with coefficients " << coefficients1 << " \n";
    std::cout << plaintextDec << std::endl;

    std::cout << "\n Expected result: (0.70519107, 1.38285078, 3.97211180, "
                 "5.60215665, 4.86357575) "
              << std::endl;

    {
        DCRTPoly data(cc->GetElementParams(), Format::EVALUATION, true);
        const size_t numLimbs = data.m_vectors.size();
        const size_t phim = data.GetRingDimension();
        std::cout << "numLimbs: " << numLimbs << std::endl;
        // assert(numLimbs == levels);

        for (size_t i = 0; i < numLimbs; i++) {
            for (size_t j = 0; j < phim; j++) {
                (*(data.m_vectors[i].m_values))[j] = 1;
            }
        }

        std::cout << "loaded data\n";

        // std::cout << "data = " << data << std::end;
        data.SwitchFormat();
        // std::cout << "data coeffs = " << data << std::end;

        std::cout << "switched format\n";

        for (size_t i = 0; i < numLimbs; i++) {
            for (size_t j = 0; j < phim; j++) {
                if (j == 0) assert(data.m_vectors[i][j] == 1);
                else assert(data.m_vectors[i][j] == 0);
            }
        }
    }

    std::cout << "Step 1: create parameter object with the correct primes.\n";
    // Parameter(int log_degree, int level, int dnum, const std::vector<word64>& primes)

    // No primes in the top-level parameters. need to go into the Element parameters.

    // using ElementParamsType = typename DCRTPoly::Params;

    auto elementParams = cc->GetElementParams();

    // const size_t numLimbs = a.m_vectors.size();
    // std::cout << "a numlimbs = " << a.GetNumOfElements() << std::endl;

    const auto cryptoParams = std::dynamic_pointer_cast<CryptoParametersRNS>(cc->GetCryptoParameters());

    std::cout << "alpha: " << cryptoParams->GetNumPerPartQ() << std::endl;
    std::cout << "real dnum: " << cryptoParams->GetNumberOfQPartitions() << std::endl;

    auto limb_params = elementParams->GetParams();
    auto ext_params = cryptoParams->GetParamsP()->GetParams();
    // std::cout << "extension params size: " << 

    std::cout << "log(n) = " << log2(limb_params[0]->GetRingDimension()) << std::endl;
    const size_t n = limb_params[0]->GetRingDimension();
    const size_t logn = log2(n);
    const size_t numModuli = limb_params.size();
    std::cout << "num moduli = " << numModuli << std::endl;        
    std::cout << "num extension moduli = " << ext_params.size() << std::endl;        

    std::vector<uint64_t> limb_moduli;
    std::vector<uint64_t> limb_rous;
    for (const auto& limb : limb_params) {
        limb_moduli.push_back((uint64_t)limb->GetModulus());
        limb_rous.push_back((uint64_t)limb->GetRootOfUnity());
        std::cout << limb->GetModulus() << " " << limb->GetRootOfUnity() << std::endl;
    }

    for (const auto& limb : ext_params) {
        limb_moduli.push_back((uint64_t)limb->GetModulus());
        limb_rous.push_back((uint64_t)limb->GetRootOfUnity());
        std::cout << limb->GetModulus() << " " << limb->GetRootOfUnity() << std::endl;
    }

    ckks::Parameter gpu_params(logn, 8, 3, limb_moduli);
    ckks::Context gpu_context(gpu_params, limb_rous);

    // dummy device vector to examine sizes
    ckks::Test test_small(gpu_params);

    {
        const ckks::DeviceVector coeffs = test_small.GetRandomPolyRNS(gpu_params.chain_length_);

        ckks::DeviceVector eval(coeffs);
        gpu_context.ToNTTInplace(eval.data(), 0, gpu_params.chain_length_);
        ckks::DeviceVector should_be_coeffs = gpu_context.FromNTT(eval);

        assert(coeffs == should_be_coeffs);
        std::cout << "NTT inverse inplace test passed\n";
    }

    {
        const ckks::DeviceVector coeffs = test_small.GetRandomPolyRNS(gpu_params.chain_length_);

        ckks::DeviceVector eval = gpu_context.ToNTT(coeffs);
        ckks::DeviceVector should_be_coeffs = gpu_context.FromNTT(eval);

        assert(coeffs == should_be_coeffs);
        std::cout << "NTT inverse test passed\n";
    }

    {
        DCRTPoly data(cc->GetElementParams(), Format::EVALUATION, true);
        const size_t numLimbs = data.m_vectors.size();
        const size_t phim = data.GetRingDimension();
        std::cout << "numLimbs: " << numLimbs << std::endl;
        // assert(numLimbs == levels);

        for (size_t i = 0; i < numLimbs; i++) {
            for (size_t j = 0; j < phim; j++) {
                (*(data.m_vectors[i].m_values))[j] = 1;
            }
        }

        ckks::DeviceVector eval_ones = loadIntoDeviceVector(data);

        std::cout << "loaded data\n";

        // std::cout << "data = " << data << std::end;
        data.SwitchFormat();
        // std::cout << "data coeffs = " << data << std::end;

        std::cout << "switched format\n";

        ckks::DeviceVector coeffs = gpu_context.FromNTT(eval_ones);

        DCRTPoly should_be_one = loadIntoDCRTPoly(coeffs, data.GetParams(), Format::COEFFICIENT);

        assert(data == should_be_one);
    }

    {
        DCRTPoly data(cc->GetElementParams(), Format::COEFFICIENT, true);
        const size_t numLimbs = data.m_vectors.size();
        const size_t phim = data.GetRingDimension();
        std::cout << "numLimbs: " << numLimbs << std::endl;
        // assert(numLimbs == levels);

        for (size_t i = 0; i < numLimbs; i++) {
            for (size_t j = 0; j < phim; j++) {
                (*(data.m_vectors[i].m_values))[j] = 1;
            }
        }

        ckks::DeviceVector coeff_ones = loadIntoDeviceVector(data);

        std::cout << "loaded data\n";

        // std::cout << "data = " << data << std::end;
        data.SwitchFormat();
        // std::cout << "data coeffs = " << data << std::end;

        std::cout << "switched format\n";

        ckks::DeviceVector evals = gpu_context.ToNTT(coeff_ones);

        DCRTPoly should_be_evals = loadIntoDCRTPoly(evals, data.GetParams(), Format::EVALUATION);

        assert(data == should_be_evals);
    }

    std::cout << "Step 2: Load DCRTPoly data into DeviceVectors\n";

    DiscreteUniformGeneratorImpl<lbcrypto::NativeVector> dug;
    DCRTPoly a(dug, elementParams, Format::EVALUATION);
    ckks::DeviceVector a_data = loadIntoDeviceVector(a);

    {
        DCRTPoly shouldBeA = loadIntoDCRTPoly(a_data, a.GetParams());
        assert(a == shouldBeA);
    }

    ckks::DeviceVector a_coeffs = gpu_context.FromNTT(a_data);

    std::cout << "Step 3: Test NTT\n";

    assert(a.GetFormat() == Format::EVALUATION);
    a.SwitchFormat();

    DCRTPoly shouldBeACoeffs = loadIntoDCRTPoly(a_coeffs, a.GetParams(), Format::COEFFICIENT);

    // std::cout << "a: " << *(a.m_vectors[0].m_values)[0] << std::endl;
    // std::cout << "shouldBeACoeffs: " << *(shouldBeACoeffs.m_vectors[0].m_values)[0] << std::endl;

    assert(a == shouldBeACoeffs);

    ckks::DeviceVector a_correct_coeffs = loadIntoDeviceVector(a);

    assert(a_coeffs == a_correct_coeffs);

    std::cout << "inverse NTT test passed\n";

    a.SwitchFormat();

    ckks::DeviceVector a_eval = gpu_context.ToNTT(a_coeffs);

    assert(a_data == a_eval);

    gpu_context.ToNTTInplace(a_coeffs.data(), 0, gpu_params.chain_length_);

    assert(a_data == a_coeffs);

    std::cout << "forward NTT test passed\n";

    DCRTPoly from_dv = loadIntoDCRTPoly(a_data, elementParams);

    assert(from_dv == a);

    a.DropLastElement();
    ckks::DeviceVector a_dropped = loadIntoDeviceVector(a);

    a.SwitchFormat();
    assert(a.GetFormat() == Format::COEFFICIENT);

    ckks::DeviceVector a_dropped_coeffs = loadIntoDeviceVector(a);
    
    const size_t phim = a.GetRingDimension();

    // gpu_context.ToNTTInplace(a_dropped_coeffs.data(), 0, gpu_params.chain_length_-1);

    // assert(a_dropped == a_dropped_coeffs);

    ckks::DeviceVector a_copy; 
    // a_copy.append(a_dropped_coeffs);
    a_copy.resize(phim*(gpu_params.chain_length_-1));
    // cudaMemcpyAsync(a_copy.data(), a_dropped_coeffs.data(), (gpu_params.chain_length_-1)*phim * sizeof(ckks::DeviceVector::Dtype), cudaMemcpyDefault, a_copy.stream_);
    for (int i = 0; i < gpu_params.chain_length_-1; i++) 
        cudaMemcpyAsync(a_copy.data() + i*phim, a_dropped_coeffs.data() + i*phim, phim * sizeof(ckks::DeviceVector::Dtype), 
            cudaMemcpyDefault, a_copy.stream_);

    gpu_context.ToNTTInplace(a_copy.data(), 0, gpu_params.chain_length_-1);

    assert(a_copy == a_dropped);

}