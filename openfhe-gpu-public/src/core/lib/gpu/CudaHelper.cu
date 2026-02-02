/* Copyright (c) by CryptoLab Inc. and Seoul National University R&DB Foundation.
 * This library is licensed under a
 * Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.
 * You should have received a copy of the license along with this
 * work. If not, see <http://creativecommons.org/licenses/by-nc-sa/4.0/>.
 */
#include "Define.h"
#if CUDART_VERSION >= 12000
#include <nvtx3/nvToolsExt.h>
#else
#include "nvToolsExt.h"
#endif

namespace ckks {

void CudaNvtxStart(std::string msg) { nvtxRangePushA(msg.c_str()); }
void CudaNvtxStop() { nvtxRangePop(); }
void CudaHostSync() { cudaDeviceSynchronize(); }

}