/* Copyright (c) by CryptoLab Inc. and Seoul National University R&DB Foundation.
 * This library is licensed under a
 * Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.
 * You should have received a copy of the license along with this
 * work. If not, see <http://creativecommons.org/licenses/by-nc-sa/4.0/>.
 */


#include <random>

// #include "gtest/gtest.h"
#include "gpu/Ciphertext.h"
#include "gpu/Context.h"
#include "gpu/Define.h"
#include "gpu/EvaluationKey.h"
#include "gpu/MultPtxtBatch.h"
#include "gpu/Parameter.h"
#include "gpu/Test.h"

class FusionTest : public ckks::Test {
                  //  public ::testing::TestWithParam<ckks::Parameter> {
  public:


  FusionTest(ckks::Parameter& param) : ckks::Test(param){};

  void ModUpTest();
  void ModDownTest();
  void KeySwitchTest();
  void PtxtCtxtBatching();

  void runAllTests() {

      ModUpTest();
      ModDownTest();
      KeySwitchTest();
      PtxtCtxtBatching();

  }
};

void FusionTest::ModUpTest() {
  for (int beta = 0; beta < std::min(2 /* else it takes too long*/, param.dnum_); beta++) {
    const int num_moduli = (beta + 1) * param.alpha_;
    // assumed rns-decomposed already.
    ckks::DeviceVector from{GetRandomPolyRNS(num_moduli)};
    context.is_modup_batched = false;
    auto ref = context.ModUp(from);
    context.is_modup_batched = true;
    auto batched = context.ModUp(from);
    // COMPARE(ref, batched);
    assert(ref == batched);
  }

  std::cout << "ModUpTest passed\n";
}

void FusionTest::ModDownTest() {
  auto from = GetRandomPolyRNS(param.max_num_moduli_);
  ckks::DeviceVector from_fused(from);
  ckks::DeviceVector to;
  ckks::DeviceVector to_fused;
  // const int target_chain_length = 2;
  context.is_moddown_fused = true;
  context.ModDown(from_fused, to_fused);
  context.is_moddown_fused = false;
  context.ModDown(from, to);
  assert(to == to_fused);

  std::cout << "ModDownTest passed\n";
}

void FusionTest::KeySwitchTest() {
  int beta = param.dnum_;
  auto key = GetRandomKey();
  auto in = GetRandomPolyAfterModUp(beta);
  ckks::DeviceVector ax, bx;
  ckks::DeviceVector ax_ref, bx_ref;
  context.is_keyswitch_fused = false;
  context.KeySwitch(in, key, ax_ref, bx_ref);
  context.is_keyswitch_fused = true;
  context.KeySwitch(in, key, ax, bx);
  assert(ax == ax_ref);
  assert(bx == bx_ref);

  std::cout << "KeySwitch test passed\n";
}

void FusionTest::PtxtCtxtBatching() {
  using namespace ckks;
  int batch_size = 3;
  vector<Ciphertext> op1(batch_size);
  vector<Plaintext> op2(batch_size);
  // setup
  for (int i = 0; i < batch_size; i++) {
    op1[i] = GetRandomCiphertext();
    op2[i] = GetRandomPlaintext();
  }
  // reference
  Ciphertext accum, out;
  context.PMult(op1[0], op2[0], accum);
  for (int i = 1; i < batch_size; i++) {
    context.PMult(op1[i], op2[i], out);
    context.Add(accum, out, accum);
  }
  // with batching
  Ciphertext accum_new;
  {
    MultPtxtBatch batcher(&context);
    for (int i = 0; i < batch_size; i++) {
      batcher.push(op1[i], op2[i]);
    }
    batcher.flush(accum_new);
  }
  assert(accum_new.getAxDevice() == accum.getAxDevice());
  assert(accum_new.getBxDevice() == accum.getBxDevice());

  std::cout << "PtxtCtxtBatching test passed\n";
}

int main() {
  FusionTest test_small(PARAM_SMALL_DNUM);
  test_small.runAllTests();

  // FusionTest test_large(PARAM_LARGE_DNUM);
  // test_large.runAllTests();
}