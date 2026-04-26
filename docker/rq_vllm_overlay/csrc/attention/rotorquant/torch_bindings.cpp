// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the rq-models project
//
// Torch extension bindings for rq-models RotorQuant planar3 KV kernels.
// Exposes two ops to Python:
//   torch.ops.rq_models.rotorquant_planar3_pack(src_f16, dst_uint8, n_blocks)
//   torch.ops.rq_models.rotorquant_planar3_unpack(src_uint8, dst_f16, n_blocks)
//
// These are LOW-LEVEL dispatch ops that operate on flat tensors. The
// vLLM-side glue in vllm/v1/attention/ops/rotorquant_kv.py is responsible
// for unpacking the [num_tokens, num_kv_heads, head_size] -> flat
// [n_blocks * 128] mapping and re-shaping the output back. Keeping this
// op signature minimal lets us iterate on the per-token block-table
// integration without touching the kernel.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace rq_models {
namespace rotorquant {

// Forward declarations of the launchers from planar3_kv.cu.
struct BlockPlanar3;  // opaque to the binding TU

extern "C" {
void rq_planar3_pack_f16(
    const __half* d_src,
    BlockPlanar3* d_dst,
    int64_t n_blocks,
    cudaStream_t stream);

void rq_planar3_unpack_f16(
    const BlockPlanar3* d_src,
    __half* d_dst,
    int64_t n_blocks,
    cudaStream_t stream);

// Forward declarations from planar3_paged_kv.cu (Phase 2.5 fused write/read).
void rq_planar3_pack_and_scatter(
    const __half* d_key,
    const __half* d_value,
    uint8_t* d_key_cache,
    uint8_t* d_value_cache,
    const int64_t* d_slot_mapping,
    int num_tokens,
    int num_kv_heads,
    int head_size,
    int blocks_per_head,
    cudaStream_t stream);

void rq_planar3_gather_and_unpack(
    const uint8_t* d_key_cache,
    const uint8_t* d_value_cache,
    __half* d_key_unpacked,
    __half* d_value_unpacked,
    const int32_t* d_block_table,
    const int32_t* d_seq_lens,
    int num_seqs,
    int max_seq_len,
    int num_kv_heads,
    int head_size,
    int blocks_per_head,
    int block_size,
    int max_blocks_per_seq,
    cudaStream_t stream);
}

constexpr int64_t QK_PLANAR3   = 128;
constexpr int64_t BLOCK_BYTES  = 50;

// Pack: flat fp16 (length n_blocks * 128) -> packed uint8 (length n_blocks * 50).
// Args:
//   src    fp16 CUDA tensor, contiguous, length n_blocks * 128
//   dst    uint8 CUDA tensor, contiguous, length n_blocks * 50
//   n_blocks   number of 128-element blocks to encode
void rotorquant_planar3_pack(
    const torch::Tensor& src,
    torch::Tensor& dst,
    int64_t n_blocks)
{
    TORCH_CHECK(src.is_cuda(), "src must be a CUDA tensor");
    TORCH_CHECK(dst.is_cuda(), "dst must be a CUDA tensor");
    TORCH_CHECK(src.dtype() == torch::kFloat16, "src must be float16");
    TORCH_CHECK(dst.dtype() == torch::kUInt8,   "dst must be uint8");
    TORCH_CHECK(src.is_contiguous(), "src must be contiguous");
    TORCH_CHECK(dst.is_contiguous(), "dst must be contiguous");
    TORCH_CHECK(src.numel() == n_blocks * QK_PLANAR3,
                "src.numel() ", src.numel(),
                " != n_blocks * 128 = ", n_blocks * QK_PLANAR3);
    TORCH_CHECK(dst.numel() == n_blocks * BLOCK_BYTES,
                "dst.numel() ", dst.numel(),
                " != n_blocks * 50 = ", n_blocks * BLOCK_BYTES);

    auto stream = at::cuda::getCurrentCUDAStream(src.get_device()).stream();
    rq_planar3_pack_f16(
        reinterpret_cast<const __half*>(src.data_ptr<at::Half>()),
        reinterpret_cast<BlockPlanar3*>(dst.data_ptr<uint8_t>()),
        n_blocks,
        stream);
}

// Unpack: packed uint8 (length n_blocks * 50) -> fp16 (length n_blocks * 128).
void rotorquant_planar3_unpack(
    const torch::Tensor& src,
    torch::Tensor& dst,
    int64_t n_blocks)
{
    TORCH_CHECK(src.is_cuda(), "src must be a CUDA tensor");
    TORCH_CHECK(dst.is_cuda(), "dst must be a CUDA tensor");
    TORCH_CHECK(src.dtype() == torch::kUInt8,   "src must be uint8");
    TORCH_CHECK(dst.dtype() == torch::kFloat16, "dst must be float16");
    TORCH_CHECK(src.is_contiguous(), "src must be contiguous");
    TORCH_CHECK(dst.is_contiguous(), "dst must be contiguous");
    TORCH_CHECK(src.numel() == n_blocks * BLOCK_BYTES,
                "src.numel() ", src.numel(),
                " != n_blocks * 50 = ", n_blocks * BLOCK_BYTES);
    TORCH_CHECK(dst.numel() == n_blocks * QK_PLANAR3,
                "dst.numel() ", dst.numel(),
                " != n_blocks * 128 = ", n_blocks * QK_PLANAR3);

    auto stream = at::cuda::getCurrentCUDAStream(src.get_device()).stream();
    rq_planar3_unpack_f16(
        reinterpret_cast<const BlockPlanar3*>(src.data_ptr<uint8_t>()),
        reinterpret_cast<__half*>(dst.data_ptr<at::Half>()),
        n_blocks,
        stream);
}

// ============================================================================
// Phase 2.5 fused paged KV ops
// ============================================================================

// pack_and_scatter: fused pack of fp16 K/V into the packed paged cache.
// Args:
//   key, value      [num_tokens, num_kv_heads, head_size]    fp16
//   key_cache, value_cache  packed cache uint8, flat-byte layout
//                   [num_blocks * block_size * num_kv_heads * blocks_per_head * 50]
//   slot_mapping    [num_tokens]                              int64
//   num_kv_heads, head_size, blocks_per_head    int
void rotorquant_planar3_pack_and_scatter(
    const torch::Tensor& key,
    const torch::Tensor& value,
    torch::Tensor& key_cache,
    torch::Tensor& value_cache,
    const torch::Tensor& slot_mapping,
    int64_t num_kv_heads,
    int64_t head_size,
    int64_t blocks_per_head)
{
    TORCH_CHECK(key.is_cuda() && value.is_cuda(), "K, V must be CUDA tensors");
    TORCH_CHECK(key_cache.is_cuda() && value_cache.is_cuda(), "caches must be CUDA");
    TORCH_CHECK(slot_mapping.is_cuda(), "slot_mapping must be CUDA");
    TORCH_CHECK(key.dtype() == torch::kFloat16, "K must be float16");
    TORCH_CHECK(value.dtype() == torch::kFloat16, "V must be float16");
    TORCH_CHECK(key_cache.dtype() == torch::kUInt8, "key_cache must be uint8");
    TORCH_CHECK(value_cache.dtype() == torch::kUInt8, "value_cache must be uint8");
    TORCH_CHECK(slot_mapping.dtype() == torch::kInt64, "slot_mapping must be int64");
    TORCH_CHECK(key.is_contiguous() && value.is_contiguous(),
                "K, V must be contiguous");
    TORCH_CHECK(head_size % 128 == 0,
                "head_size must be a multiple of 128 (planar3 block size); got ",
                head_size);
    TORCH_CHECK(blocks_per_head == head_size / 128,
                "blocks_per_head ", blocks_per_head, " != head_size/128 ",
                head_size / 128);

    const int64_t num_tokens = slot_mapping.size(0);
    auto stream = at::cuda::getCurrentCUDAStream(key.get_device()).stream();
    rq_planar3_pack_and_scatter(
        reinterpret_cast<const __half*>(key.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(value.data_ptr<at::Half>()),
        key_cache.data_ptr<uint8_t>(),
        value_cache.data_ptr<uint8_t>(),
        slot_mapping.data_ptr<int64_t>(),
        static_cast<int>(num_tokens),
        static_cast<int>(num_kv_heads),
        static_cast<int>(head_size),
        static_cast<int>(blocks_per_head),
        stream);
}

// gather_and_unpack: pre-attention materialize of packed cache to dense fp16.
// Args:
//   key_cache, value_cache    packed uint8 cache
//   key_unpacked, value_unpacked  [num_seqs, max_seq_len, num_kv_heads, head_size] fp16
//   block_table  [num_seqs, max_blocks_per_seq] int32
//   seq_lens     [num_seqs] int32
//   num_kv_heads, head_size, blocks_per_head, block_size, max_blocks_per_seq  int
void rotorquant_planar3_gather_and_unpack(
    const torch::Tensor& key_cache,
    const torch::Tensor& value_cache,
    torch::Tensor& key_unpacked,
    torch::Tensor& value_unpacked,
    const torch::Tensor& block_table,
    const torch::Tensor& seq_lens,
    int64_t num_kv_heads,
    int64_t head_size,
    int64_t blocks_per_head,
    int64_t block_size,
    int64_t max_blocks_per_seq)
{
    TORCH_CHECK(key_cache.dtype() == torch::kUInt8, "key_cache must be uint8");
    TORCH_CHECK(key_unpacked.dtype() == torch::kFloat16, "key_unpacked must be fp16");
    TORCH_CHECK(block_table.dtype() == torch::kInt32, "block_table must be int32");
    TORCH_CHECK(seq_lens.dtype() == torch::kInt32, "seq_lens must be int32");
    TORCH_CHECK(key_unpacked.dim() == 4,
                "key_unpacked must be [num_seqs, max_seq_len, num_kv_heads, head_size]; got rank ",
                key_unpacked.dim());

    const int64_t num_seqs    = key_unpacked.size(0);
    const int64_t max_seq_len = key_unpacked.size(1);
    auto stream = at::cuda::getCurrentCUDAStream(key_cache.get_device()).stream();
    rq_planar3_gather_and_unpack(
        key_cache.data_ptr<uint8_t>(),
        value_cache.data_ptr<uint8_t>(),
        reinterpret_cast<__half*>(key_unpacked.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(value_unpacked.data_ptr<at::Half>()),
        block_table.data_ptr<int32_t>(),
        seq_lens.data_ptr<int32_t>(),
        static_cast<int>(num_seqs),
        static_cast<int>(max_seq_len),
        static_cast<int>(num_kv_heads),
        static_cast<int>(head_size),
        static_cast<int>(blocks_per_head),
        static_cast<int>(block_size),
        static_cast<int>(max_blocks_per_seq),
        stream);
}

}  // namespace rotorquant
}  // namespace rq_models

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rotorquant_planar3_pack",
          &rq_models::rotorquant::rotorquant_planar3_pack,
          "rq-models RotorQuant planar3 KV pack (fp16 -> 3-bit packed)");
    m.def("rotorquant_planar3_unpack",
          &rq_models::rotorquant::rotorquant_planar3_unpack,
          "rq-models RotorQuant planar3 KV unpack (3-bit packed -> fp16)");
    m.def("rotorquant_planar3_pack_and_scatter",
          &rq_models::rotorquant::rotorquant_planar3_pack_and_scatter,
          "Phase 2.5: fused pack-and-scatter into the packed paged KV cache");
    m.def("rotorquant_planar3_gather_and_unpack",
          &rq_models::rotorquant::rotorquant_planar3_gather_and_unpack,
          "Phase 2.5: pre-attention gather + unpack from packed paged KV cache");
}
