// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the rq-models project
//
// rq-models RotorQuant planar3 PAGED KV cache kernels (Sprint 004 Phase 2.5).
//
// Phase 2c (the "lossy-passthrough" mode) stored fp16 KV in the cache and
// applied a planar3 round-trip to K/V before storage — quality matches the
// real planar3 scheme but no actual storage savings.
//
// Phase 2.5 (this file) flips the storage to PACKED uint8 — the cache shape
// becomes [num_blocks, block_size, num_kv_heads, blocks_per_head * 50]
// where blocks_per_head = head_size / 128. Per-token storage drops from
// num_kv_heads * head_size * 2 bytes (fp16) to num_kv_heads * (head_size/128) * 50
// bytes. For Qwen3 head_size=128, that's 256→50 bytes per head per token,
// a 5.12× compression. THIS is the actual RotorQuant feature.
//
// Two new kernels:
//   pack_and_scatter:   takes fp16 K, V plus slot_mapping; packs each
//                       128-element segment to a 50-byte block and scatters
//                       into the paged cache at the slot's offset. One
//                       fused launch instead of pack-then-scatter.
//   gather_and_unpack:  the inverse for pre-attention read materialization.
//                       Takes packed cache + block_table + seq_lens, outputs
//                       fp16 K/V tensors sized for the active slots.
//
// Block layout matches the standalone planar3_kv.cu kernel (50 bytes,
// 3.125 bpe) so packed cache contents are interchangeable across the two
// code paths. We share constants by re-declaring with __constant__ here
// rather than #include'ing — nvcc handles per-TU __constant__ memory
// independently.

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

namespace rq_models {
namespace rotorquant {
namespace paged {

constexpr int QK_PLANAR3 = 128;
constexpr int QS_BYTES   = QK_PLANAR3 / 4;  // 32
constexpr int SIGNS_BYTES= QK_PLANAR3 / 8;  // 16
constexpr int BLOCK_BYTES = 2 + QS_BYTES + SIGNS_BYTES;  // 50

// Mirror of planar3_kv.cu's BlockPlanar3 — must match byte-for-byte.
struct __align__(2) BlockPlanar3 {
    __half  norm;
    uint8_t qs[QS_BYTES];
    uint8_t signs[SIGNS_BYTES];
};
static_assert(sizeof(BlockPlanar3) == BLOCK_BYTES, "BlockPlanar3 layout drift");

// ============================================================================
// Constants — same numerical values as planar3_kv.cu (cross-substrate parity
// requirement). Each TU gets its own __constant__ memory copy.
// ============================================================================

__constant__ float p_planar_cos[64] = {
    -0.9095053397f, 0.1535578452f, -0.8537489227f, -0.6827218011f,
    -0.4249387949f, 0.9864510046f, 0.9906673944f, 0.5752363372f,
    -0.9866459035f, 0.9878848090f, -0.6215683804f, -0.9835597698f,
     0.8777263755f, -0.4624640047f, 0.2843135922f, -0.7739960698f,
     0.2385234222f, 0.9121914932f, -0.8815003943f, -0.2639699512f,
    -0.5517087300f, -0.9035294557f, -0.8520543188f, -0.5600635985f,
    -0.7667286376f, -0.9877949369f, -0.9781949787f, -0.9953372831f,
    -0.8622053901f, -0.7382118186f, 0.9136037642f, -0.2558504503f,
    -0.8541000475f, -0.6159335408f, 0.9861256679f, -0.6758560284f,
     0.4249571682f, -0.6219544719f, 0.9130573430f, -0.5948161096f,
     0.5759782996f, 0.9729901203f, 0.6535998325f, 0.9222195491f,
    -0.7668084044f, 0.5116178563f, -0.7848786574f, 0.9902111051f,
     0.1997167840f, 0.7173003220f, -0.9999998006f, -0.9557868691f,
     0.5594852693f, -0.9980111824f, 0.9782398557f, -0.9150004329f,
    -0.4084754305f, 0.0071549185f, 0.9558482753f, -0.0971921648f,
    -0.9469334002f, 0.9999492419f, 0.6100589016f, 0.0350818915f
};

__constant__ float p_planar_sin[64] = {
    -0.4156922383f, 0.9881396603f, 0.5206849114f, -0.7306784124f,
    -0.9052220836f, 0.1640561354f, 0.1363015542f, 0.8179872593f,
     0.1628798979f, 0.1551889303f, 0.7833599099f, -0.1805828875f,
    -0.4791621957f, 0.8866380571f, -0.9587313395f, 0.6331904010f,
    -0.9711367448f, 0.4097641756f, 0.4721832852f, -0.9645309040f,
     0.8340368561f, 0.4285259884f, 0.5234533769f, 0.8284496156f,
     0.6419713361f, -0.1557599517f, -0.2076886701f, 0.0964556523f,
     0.5065588468f, -0.6745689815f, -0.4066056591f, -0.9667163736f,
     0.5201087471f, -0.7877981171f, 0.1660005034f, -0.7370336688f,
     0.9052134584f, 0.7830534049f, -0.4078312009f, -0.8038618014f,
     0.8174649829f, -0.2308467584f, -0.7568403127f, -0.3866666566f,
     0.6418760557f, -0.8592131104f, 0.6196494922f, 0.1395778183f,
     0.9798536657f, 0.6967641265f, -0.0006314605f, 0.2940603015f,
     0.8288402943f, -0.0630371303f, 0.2074771907f, 0.4034528570f,
     0.9127693152f, -0.9999744032f, 0.2938606379f, 0.9952656344f,
     0.3214298299f, 0.0100754012f, -0.7923560668f, -0.9993844410f
};

__constant__ float p_centroids_3bit[8] = {
    -0.190685f, -0.117832f, -0.065717f, -0.021460f,
     0.021460f,  0.065717f,  0.117832f,  0.190685f
};

__constant__ float p_mid_3bit[7] = {
    -0.154259f, -0.091775f, -0.043589f, 0.0f, 0.043589f, 0.091775f, 0.154259f
};

// ============================================================================
// Device helpers — shared between pack and unpack
// ============================================================================

__device__ __forceinline__ uint8_t quantize_3bit_p(float val) {
    if (val < p_mid_3bit[0]) return 0;
    if (val < p_mid_3bit[1]) return 1;
    if (val < p_mid_3bit[2]) return 2;
    if (val < p_mid_3bit[3]) return 3;
    if (val < p_mid_3bit[4]) return 4;
    if (val < p_mid_3bit[5]) return 5;
    if (val < p_mid_3bit[6]) return 6;
    return 7;
}

__device__ __forceinline__ uint8_t unpack_3bit_p(const BlockPlanar3* blk, int j) {
    uint8_t low = (blk->qs[j / 4] >> ((j % 4) * 2)) & 0x3;
    uint8_t hi  = (blk->signs[j / 8] >> (j % 8)) & 0x1;
    return low | (hi << 2);
}

// One-thread-per-block packer. Single-threaded per planar3 block keeps the
// math identical to the standalone packer; can later be parallelized to one
// warp per block if profiling warrants.
__device__ __forceinline__ void pack_block_planar3(
    const __half* __restrict__ src,    // 128 fp16 elements
    BlockPlanar3* __restrict__ dst)
{
    float buf[QK_PLANAR3];
    float norm_sq = 0.0f;
    #pragma unroll
    for (int j = 0; j < QK_PLANAR3; j++) {
        buf[j] = __half2float(src[j]);
        norm_sq += buf[j] * buf[j];
    }
    float grp_norm = sqrtf(norm_sq);
    float inv_norm = (grp_norm > 1e-10f) ? (1.0f / grp_norm) : 0.0f;
    #pragma unroll
    for (int j = 0; j < QK_PLANAR3; j++) buf[j] *= inv_norm;

    float rotated[QK_PLANAR3];
    #pragma unroll
    for (int p = 0; p < 64; p++) {
        float c = p_planar_cos[p], s = p_planar_sin[p];
        rotated[p*2]     = c * buf[p*2] - s * buf[p*2+1];
        rotated[p*2+1]   = s * buf[p*2] + c * buf[p*2+1];
    }

    #pragma unroll
    for (int j = 0; j < QS_BYTES;    j++) dst->qs[j]    = 0;
    #pragma unroll
    for (int j = 0; j < SIGNS_BYTES; j++) dst->signs[j] = 0;

    float recon_sq = 0.0f;
    #pragma unroll
    for (int j = 0; j < QK_PLANAR3; j++) {
        uint8_t idx = quantize_3bit_p(rotated[j]);
        dst->qs[j / 4]    |= (idx & 0x3) << ((j % 4) * 2);
        if (idx & 0x4) dst->signs[j / 8] |= (1 << (j % 8));
        float c = p_centroids_3bit[idx];
        recon_sq += c * c;
    }
    float recon_norm = sqrtf(recon_sq);
    float corrected = (recon_norm > 1e-10f) ? (grp_norm / recon_norm) : grp_norm;
    dst->norm = __float2half(corrected);
}

__device__ __forceinline__ void unpack_block_planar3(
    const BlockPlanar3* __restrict__ src,
    __half* __restrict__ dst)              // 128 fp16 outputs
{
    const float norm = __half2float(src->norm);
    #pragma unroll
    for (int p = 0; p < 64; p++) {
        float q0 = p_centroids_3bit[unpack_3bit_p(src, p*2)];
        float q1 = p_centroids_3bit[unpack_3bit_p(src, p*2 + 1)];
        float c  = p_planar_cos[p];
        float s  = p_planar_sin[p];
        // Inverse Givens (transpose of forward rotation matrix)
        float v0 = ( c * q0 + s * q1) * norm;
        float v1 = (-s * q0 + c * q1) * norm;
        dst[p*2]     = __float2half(v0);
        dst[p*2 + 1] = __float2half(v1);
    }
}

// ============================================================================
// pack_and_scatter: fused write into paged KV cache
// ============================================================================
// Input:
//   key   [num_tokens, num_kv_heads, head_size]              fp16
//   value [num_tokens, num_kv_heads, head_size]              fp16
//   slot_mapping [num_tokens]                                 int64
//     slot_mapping[i] = absolute slot index = block_id * block_size + offset
//     OR -1 for padding tokens (skipped)
//
// Output (in-place):
//   key_cache   [num_blocks, block_size, num_kv_heads, blocks_per_head * 50]
//   value_cache  same shape
//   Both treated as flat uint8 arrays of length
//     (num_blocks * block_size) * num_kv_heads * blocks_per_head * 50
//
// Each thread handles one (token_idx, head_idx, intra_head_block_idx) triple.
// Total thread count = num_tokens * num_kv_heads * blocks_per_head.
//
// Grid is 1D for simplicity; we compute (token, head, intra_block) from
// the thread index. Could go to 3D grid if we ever need >2^31 threads.

__global__ void kernel_pack_and_scatter_planar3(
    const __half* __restrict__ key,         // [num_tokens, num_kv_heads, head_size]
    const __half* __restrict__ value,
    uint8_t* __restrict__ key_cache,        // packed cache
    uint8_t* __restrict__ value_cache,
    const int64_t* __restrict__ slot_mapping,
    int num_tokens,
    int num_kv_heads,
    int head_size,
    int blocks_per_head)                    // = head_size / 128
{
    const int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t total = (int64_t)num_tokens * num_kv_heads * blocks_per_head;
    if (tid >= total) return;

    const int intra_block_idx = tid % blocks_per_head;
    const int head_idx        = (tid / blocks_per_head) % num_kv_heads;
    const int token_idx       = tid / (blocks_per_head * num_kv_heads);

    const int64_t slot = slot_mapping[token_idx];
    if (slot < 0) return;  // padded slot (unused)

    // Source pointer: 128 fp16 elements for this (token, head, intra_block).
    const int64_t src_offset =
        (int64_t)token_idx * num_kv_heads * head_size
      + (int64_t)head_idx  * head_size
      + (int64_t)intra_block_idx * QK_PLANAR3;

    // Destination pointer: 50 bytes for this (slot, head, intra_block).
    // Cache layout flat: slot * (num_kv_heads * blocks_per_head * BLOCK_BYTES)
    //                  + head * (blocks_per_head * BLOCK_BYTES)
    //                  + intra_block * BLOCK_BYTES
    const int64_t per_slot_bytes = (int64_t)num_kv_heads * blocks_per_head * BLOCK_BYTES;
    const int64_t per_head_bytes = (int64_t)blocks_per_head * BLOCK_BYTES;
    const int64_t dst_offset = slot * per_slot_bytes
                              + (int64_t)head_idx * per_head_bytes
                              + (int64_t)intra_block_idx * BLOCK_BYTES;

    // Pack K
    pack_block_planar3(
        key + src_offset,
        reinterpret_cast<BlockPlanar3*>(key_cache + dst_offset));
    // Pack V
    pack_block_planar3(
        value + src_offset,
        reinterpret_cast<BlockPlanar3*>(value_cache + dst_offset));
}

// ============================================================================
// gather_and_unpack: pre-attention read materialization
// ============================================================================
// Input:
//   key_cache  packed cache [num_blocks, block_size, num_kv_heads, blocks_per_head*50]
//   value_cache same
//   block_table [num_seqs, max_blocks_per_seq] int32
//     block_table[s, i] = physical_block_id for the i-th logical block of seq s
//   seq_lens [num_seqs] int32
//
// Output:
//   key_unpacked [num_seqs, max_seq_len, num_kv_heads, head_size]   fp16
//   value_unpacked same shape
//
// Each thread handles one (seq, logical_token, head, intra_block) tuple.
// Padding positions (token_idx >= seq_lens[seq]) write zeros — caller
// passes max_seq_len but seq_lens controls validity.
//
// NOTE: the output tensor is dense (not paged) so the caller can hand it to
// FlashAttention without further modification. This is the conservative
// integration — wastes memory for short sequences in a batch. A fused
// unpack-into-attention kernel is a follow-up optimization.

__global__ void kernel_gather_and_unpack_planar3(
    const uint8_t* __restrict__ key_cache,
    const uint8_t* __restrict__ value_cache,
    __half* __restrict__ key_unpacked,      // [num_seqs, max_seq_len, num_kv_heads, head_size]
    __half* __restrict__ value_unpacked,
    const int32_t* __restrict__ block_table, // [num_seqs, max_blocks_per_seq]
    const int32_t* __restrict__ seq_lens,    // [num_seqs]
    int num_seqs,
    int max_seq_len,
    int num_kv_heads,
    int head_size,
    int blocks_per_head,
    int block_size,
    int max_blocks_per_seq)
{
    const int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t total = (int64_t)num_seqs * max_seq_len * num_kv_heads * blocks_per_head;
    if (tid >= total) return;

    const int intra_block_idx = tid % blocks_per_head;
    const int head_idx        = (tid / blocks_per_head) % num_kv_heads;
    const int token_idx       = (tid / ((int64_t)blocks_per_head * num_kv_heads)) % max_seq_len;
    const int seq_idx         =  tid / ((int64_t)blocks_per_head * num_kv_heads * max_seq_len);

    // Output destination (dense) — write to it for either real or padded slots.
    const int64_t dst_offset =
        (int64_t)seq_idx   * max_seq_len * num_kv_heads * head_size
      + (int64_t)token_idx * num_kv_heads * head_size
      + (int64_t)head_idx  * head_size
      + (int64_t)intra_block_idx * QK_PLANAR3;

    __half* k_out = key_unpacked   + dst_offset;
    __half* v_out = value_unpacked + dst_offset;

    if (token_idx >= seq_lens[seq_idx]) {
        // Padding — zero out so attention masking stays well-defined.
        #pragma unroll
        for (int j = 0; j < QK_PLANAR3; j++) {
            k_out[j] = __float2half(0.0f);
            v_out[j] = __float2half(0.0f);
        }
        return;
    }

    // Resolve logical token -> physical block + offset.
    const int logical_block = token_idx / block_size;
    const int offset_in_block = token_idx % block_size;
    if (logical_block >= max_blocks_per_seq) return;
    const int32_t phys_block = block_table[seq_idx * max_blocks_per_seq + logical_block];
    if (phys_block < 0) {
        // Block not allocated for this slot (shouldn't happen if seq_lens is
        // honored; defensive zero-fill anyway).
        #pragma unroll
        for (int j = 0; j < QK_PLANAR3; j++) {
            k_out[j] = __float2half(0.0f);
            v_out[j] = __float2half(0.0f);
        }
        return;
    }

    // Cache-side address: per_slot_bytes follows the same layout as write.
    const int64_t slot = (int64_t)phys_block * block_size + offset_in_block;
    const int64_t per_slot_bytes = (int64_t)num_kv_heads * blocks_per_head * BLOCK_BYTES;
    const int64_t per_head_bytes = (int64_t)blocks_per_head * BLOCK_BYTES;
    const int64_t src_offset = slot * per_slot_bytes
                              + (int64_t)head_idx * per_head_bytes
                              + (int64_t)intra_block_idx * BLOCK_BYTES;

    unpack_block_planar3(
        reinterpret_cast<const BlockPlanar3*>(key_cache   + src_offset), k_out);
    unpack_block_planar3(
        reinterpret_cast<const BlockPlanar3*>(value_cache + src_offset), v_out);
}

// ============================================================================
// Host launchers
// ============================================================================

extern "C" {

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
    cudaStream_t stream)
{
    if (num_tokens == 0) return;
    constexpr int THREADS = 256;
    const int64_t total = (int64_t)num_tokens * num_kv_heads * blocks_per_head;
    const int64_t grid = (total + THREADS - 1) / THREADS;
    kernel_pack_and_scatter_planar3<<<grid, THREADS, 0, stream>>>(
        d_key, d_value, d_key_cache, d_value_cache, d_slot_mapping,
        num_tokens, num_kv_heads, head_size, blocks_per_head);
}

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
    cudaStream_t stream)
{
    if (num_seqs == 0 || max_seq_len == 0) return;
    constexpr int THREADS = 256;
    const int64_t total = (int64_t)num_seqs * max_seq_len * num_kv_heads * blocks_per_head;
    const int64_t grid = (total + THREADS - 1) / THREADS;
    kernel_gather_and_unpack_planar3<<<grid, THREADS, 0, stream>>>(
        d_key_cache, d_value_cache, d_key_unpacked, d_value_unpacked,
        d_block_table, d_seq_lens,
        num_seqs, max_seq_len, num_kv_heads, head_size, blocks_per_head,
        block_size, max_blocks_per_seq);
}

}  // extern "C"

}  // namespace paged
}  // namespace rotorquant
}  // namespace rq_models
