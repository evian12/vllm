// SPDX-License-Identifier: Apache-2.0
// Experimental operator; no production model path uses this kernel.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cmath>
#include <cassert>
#include "cub_helpers.h"
#include "quantization/vectorization_utils.cuh"

using B = c10::BFloat16;

// Preserve layernorm_kernels.cu's vector8 accumulation, CUB1024 valid16
// reduction, BF16 norm product and BF16 weight product. Then preserve
// pos_encoding_kernels.cu's BF16 multiply/subtract/add operations.
__global__ void exact_kernel(const B* qkv, const B* qw, const B* kw,
    const B* cache, const int64_t* positions, B* qout, B* kout,
    float eps, int nq, int nk, int cache_rows, int head_dim) {
  const int heads = nq + nk;
  const int token = blockIdx.x / heads;
  const int head = blockIdx.x % heads;
  const bool isq = head < nq;
  const int h = isq ? head : head - nq;
  const B* input = qkv + token * (nq + 2 * nk) * 128
      + (isq ? h : nq + h) * 128;
  const B* weight = isq ? qw : kw;
  B* output = isq ? qout + (token * nq + h) * 128
                  : kout + (token * nk + h) * 128;
  __shared__ float inverse_rms;
  __shared__ __align__(16) B normalized[128];
  float variance = 0.0f;
  auto vec_op = [&variance](const vllm::vec_n_t<B, 8>& vec) {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      float x = static_cast<float>(vec.val[i]);
      variance += x * x;
    }
  };
  auto scalar_op = [&variance](const B& value) {
    float x = static_cast<float>(value);
    variance += x * x;
  };
  vllm::vectorize_read_with_alignment<8>(
      input, 128, threadIdx.x, blockDim.x, vec_op, scalar_op);
  using Reduce = cub::BlockReduce<float, 1024>;
  __shared__ typename Reduce::TempStorage reduce_store;
  variance = Reduce(reduce_store).Reduce(variance, CubAddOp{}, blockDim.x);
  if (threadIdx.x == 0) inverse_rms = rsqrtf(variance / head_dim + eps);
  __syncthreads();
  auto* in = reinterpret_cast<const vllm::vec_n_t<B, 8>*>(input);
  auto* w = reinterpret_cast<const vllm::vec_n_t<B, 8>*>(weight);
  auto* norm = reinterpret_cast<vllm::vec_n_t<B, 8>*>(normalized);
  vllm::vec_n_t<B, 8> src = in[threadIdx.x], weights = w[threadIdx.x], dst;
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    float x = static_cast<float>(src.val[j]);
    dst.val[j] = ((B)(x * inverse_rms)) * weights.val[j];
  }
  norm[threadIdx.x] = dst;
  __syncthreads();
  const int64_t pos = positions[token];
  assert(pos >= 0 && pos < cache_rows);
  const B* cos = cache + pos * 128;
  const B* sin = cos + 64;
  for (int i = threadIdx.x; i < 64; i += blockDim.x) {
    const B x = normalized[i], y = normalized[i + 64];
    output[i] = x * cos[i] - y * sin[i];
    output[i + 64] = y * cos[i] + x * sin[i];
  }
}

static bool overlap(const torch::Tensor& a, const torch::Tensor& b) {
  const auto ap = reinterpret_cast<uintptr_t>(a.data_ptr());
  const auto bp = reinterpret_cast<uintptr_t>(b.data_ptr());
  return a.numel() && b.numel()
      && ap < bp + b.nbytes() && bp < ap + a.nbytes();
}

void cdb_exact_qknorm_rope(torch::Tensor qkv, torch::Tensor qw, torch::Tensor kw,
    torch::Tensor cache, torch::Tensor positions, torch::Tensor qout,
    torch::Tensor kout, double eps, int64_t nq, int64_t nk) {
  TORCH_CHECK(qkv.is_cuda(), "QKV must be CUDA");
  TORCH_CHECK(nq > 0 && nk > 0 && nq <= 32 && nk <= 8 && nq % nk == 0,
              "Unsupported head counts");
  TORCH_CHECK(std::isfinite(eps) && eps > 0, "Invalid epsilon");
  for (const auto& t : {qkv, qw, kw, cache, qout, kout}) {
    TORCH_CHECK(t.device() == qkv.device() && t.scalar_type() == torch::kBFloat16
                && t.is_contiguous(), "Requires contiguous BF16 tensors on one device");
  }
  TORCH_CHECK(positions.is_cuda() && positions.device() == qkv.device()
      && positions.scalar_type() == torch::kInt64 && positions.is_contiguous()
      && positions.dim() == 1, "Invalid positions");
  TORCH_CHECK(qkv.dim() == 2 && qkv.size(1) == (nq + 2 * nk) * 128
      && qkv.size(0) == positions.numel(), "Invalid QKV shape");
  const auto n = qkv.size(0);
  TORCH_CHECK(n <= 1024 && qw.sizes() == torch::IntArrayRef({128})
      && kw.sizes() == torch::IntArrayRef({128}), "Invalid weight/token shape");
  TORCH_CHECK(cache.dim() == 2 && cache.size(0) >= 1 && cache.size(1) == 128,
              "Requires full head128 NeoX cache");
  TORCH_CHECK(qout.sizes() == torch::IntArrayRef({n, nq * 128})
      && kout.sizes() == torch::IntArrayRef({n, nk * 128}), "Invalid output shape");
  for (const auto& input : {qkv, qw, kw, cache, positions}) {
    TORCH_CHECK(!overlap(qout, input) && !overlap(kout, input), "Output aliases input");
  }
  TORCH_CHECK(!overlap(qout, kout), "Outputs overlap");
  if (!n) return;
  const at::cuda::OptionalCUDAGuard guard(device_of(qkv));
  exact_kernel<<<n * (nq + nk), 16, 0, at::cuda::getCurrentCUDAStream()>>>(
      qkv.data_ptr<B>(), qw.data_ptr<B>(), kw.data_ptr<B>(), cache.data_ptr<B>(),
      positions.data_ptr<int64_t>(), qout.data_ptr<B>(), kout.data_ptr<B>(),
      float(eps), nq, nk, cache.size(0), 128);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
