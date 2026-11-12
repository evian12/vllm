// SPDX-License-Identifier: Apache-2.0
#include <torch/extension.h>
void cdb_exact_qknorm_rope(torch::Tensor qkv, torch::Tensor qw, torch::Tensor kw,
    torch::Tensor cache, torch::Tensor positions, torch::Tensor qout,
    torch::Tensor kout, double eps, int64_t nq, int64_t nk);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &cdb_exact_qknorm_rope, "Exact BF16 QK RMSNorm + NeoX RoPE");
}
