// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the rq-models project
//
// rq-models RotorQuant planar3 KV cache pack / unpack CUDA kernels for vLLM.
//
// Sprint 004 Phase 2a: ports the planar3 math from
// johndpope/llama-cpp-turboquant @ feature/planarquant-kv-cache (commit
// fc3d1b6 / 20efe75) into vLLM's attention kernel directory. The math is
// near-1:1 with the reference; only the kernel-launch signature and host
// API change.
//
// This file is intentionally standalone and does NOT yet wire into
// vLLM's FlashAttention KV-write call site. Phase 2b adds the torch
// extension build entry; Phase 2c replaces the rotorquant_kv.py
// passthrough stubs with calls into this code.
//
// Block layout (must match llama.cpp's block_planar3_0 exactly so the
// quantization scheme is portable across substrates):
//   struct block_planar3_0 {
//       __half  norm;                 // 2 bytes
//       uint8_t qs[QK_PLANAR3 / 4];   // 32 bytes (2-bit indices, 4 per byte)
//       uint8_t signs[QK_PLANAR3 / 8];// 16 bytes (1-bit signs, 8 per byte)
//   };
//   sizeof = 50 bytes for QK_PLANAR3=128 elements -> 3.125 bpe.
//
// Math summary (Lloyd-Max codebook + 64 successive 2D Givens rotations):
//   Pack:   normalize -> rotate (PI_COS,PI_SIN paired across 64 dims) ->
//           quantize each value to 1 of 8 centroids -> pack to 3 bits ->
//           store norm scaled to recover the per-block scale.
//   Unpack: read norm, decode each 3-bit index to centroid value, apply
//           inverse Givens rotation per pair, scale by norm.

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

namespace rq_models {
namespace rotorquant {

// ============================================================================
// Block layout constants
// ============================================================================

constexpr int QK_PLANAR3 = 128;            // elements per block
constexpr int QS_BYTES   = QK_PLANAR3 / 4; // 32 bytes for 2-bit indices
constexpr int SIGNS_BYTES= QK_PLANAR3 / 8; // 16 bytes for 1-bit signs
constexpr int BLOCK_BYTES = 2 + QS_BYTES + SIGNS_BYTES; // = 50 bytes

// Block layout struct mirrors llama-cpp-turboquant block_planar3_0.
// Use `uint8_t` storage and access via offsets to avoid alignment surprises
// across compilers. The byte offsets are: 0..1 norm, 2..33 qs, 34..49 signs.
struct __align__(2) BlockPlanar3 {
    __half  norm;
    uint8_t qs[QS_BYTES];
    uint8_t signs[SIGNS_BYTES];
};
static_assert(sizeof(BlockPlanar3) == BLOCK_BYTES,
              "BlockPlanar3 layout drifted from llama.cpp planar3_0");

// ============================================================================
// Constants (precomputed, baked at compile time)
// Source: llama-cpp-turboquant ggml/src/ggml-cuda/planar-iso-constants.cuh
// Generated from: torch.manual_seed(42) sequences. Must match Python /
// llama.cpp byte-for-byte or quantization is not interoperable.
// ============================================================================

__constant__ float d_planar_cos[64] = {
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

__constant__ float d_planar_sin[64] = {
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

// 3-bit centroid values (Lloyd-Max levels). Index 0..7 -> reconstructed value.
__constant__ float d_centroids_3bit[8] = {
    -0.190685f, -0.117832f, -0.065717f, -0.021460f,
     0.021460f,  0.065717f,  0.117832f,  0.190685f
};

// 3-bit boundary midpoints for fast quantization (between adjacent centroids).
__constant__ float d_mid_3bit[7] = {
    -0.154259f, -0.091775f, -0.043589f, 0.0f, 0.043589f, 0.091775f, 0.154259f
};

// ============================================================================
// Device helpers
// ============================================================================

__device__ __forceinline__ uint8_t quantize_3bit(float val) {
    // Threshold cascade. ~7 comparisons; fast on Ampere/Ada SMs.
    if (val < d_mid_3bit[0]) return 0;
    if (val < d_mid_3bit[1]) return 1;
    if (val < d_mid_3bit[2]) return 2;
    if (val < d_mid_3bit[3]) return 3;
    if (val < d_mid_3bit[4]) return 4;
    if (val < d_mid_3bit[5]) return 5;
    if (val < d_mid_3bit[6]) return 6;
    return 7;
}

__device__ __forceinline__ uint8_t unpack_3bit(const BlockPlanar3* blk, int j) {
    uint8_t low = (blk->qs[j / 4] >> ((j % 4) * 2)) & 0x3;
    uint8_t hi  = (blk->signs[j / 8] >> (j % 8)) & 0x1;
    return low | (hi << 2);
}

// ============================================================================
// Pack kernel: f16 input -> packed planar3 blocks
// Grid: one block per element-block (groups of 128 elements)
// Block: 1 thread per grid block (could parallelize within if needed)
//
// Input layout: src is flat fp16, length n_elements (= n_blocks * 128).
// Output layout: dst is array of n_blocks BlockPlanar3 records.
// ============================================================================

__global__ void kernel_pack_f16_to_planar3(
    const __half* __restrict__ src,
    BlockPlanar3* __restrict__ dst,
    int64_t n_blocks)
{
    const int64_t ib = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (ib >= n_blocks) return;

    const __half* s = src + ib * QK_PLANAR3;
    BlockPlanar3* blk = &dst[ib];

    // Step 1: load fp16 -> fp32, compute L2 norm
    float buf[QK_PLANAR3];
    float norm_sq = 0.0f;
    #pragma unroll
    for (int j = 0; j < QK_PLANAR3; j++) {
        buf[j] = __half2float(s[j]);
        norm_sq += buf[j] * buf[j];
    }
    float grp_norm = sqrtf(norm_sq);
    float inv_norm = (grp_norm > 1e-10f) ? (1.0f / grp_norm) : 0.0f;
    #pragma unroll
    for (int j = 0; j < QK_PLANAR3; j++) {
        buf[j] *= inv_norm;
    }

    // Step 2: 64 successive 2D Givens rotations on adjacent pairs
    float rotated[QK_PLANAR3];
    #pragma unroll
    for (int p = 0; p < 64; p++) {
        float c = d_planar_cos[p];
        float sn = d_planar_sin[p];
        rotated[p * 2]     = c * buf[p * 2] - sn * buf[p * 2 + 1];
        rotated[p * 2 + 1] = sn * buf[p * 2] + c  * buf[p * 2 + 1];
    }

    // Step 3: clear packed bytes
    #pragma unroll
    for (int j = 0; j < QS_BYTES;    j++) blk->qs[j]    = 0;
    #pragma unroll
    for (int j = 0; j < SIGNS_BYTES; j++) blk->signs[j] = 0;

    // Step 4: quantize each rotated value, pack 3-bit index (2-bit qs + 1-bit sign)
    // Track reconstruction norm so we can correct the stored norm scalar to
    // recover the original magnitude.
    float recon_sq = 0.0f;
    #pragma unroll
    for (int j = 0; j < QK_PLANAR3; j++) {
        uint8_t idx = quantize_3bit(rotated[j]);
        blk->qs[j / 4]    |= (idx & 0x3) << ((j % 4) * 2);
        if (idx & 0x4) {
            blk->signs[j / 8] |= (1 << (j % 8));
        }
        float c = d_centroids_3bit[idx];
        recon_sq += c * c;
    }

    // Step 5: store norm-corrected scalar so unpack(pack(x)) ≈ x in magnitude.
    float recon_norm = sqrtf(recon_sq);
    float corrected = (recon_norm > 1e-10f) ? (grp_norm / recon_norm) : grp_norm;
    blk->norm = __float2half(corrected);
}

// ============================================================================
// Unpack kernel: packed planar3 blocks -> f16 output
// Grid: one block per element-block; pairs handled together.
// ============================================================================

__global__ void kernel_unpack_planar3_to_f16(
    const BlockPlanar3* __restrict__ src,
    __half* __restrict__ dst,
    int64_t n_blocks)
{
    const int64_t ib = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (ib >= n_blocks) return;

    const BlockPlanar3* blk = &src[ib];
    __half* d = dst + ib * QK_PLANAR3;
    const float norm = __half2float(blk->norm);

    // Inverse Givens rotation per pair: each (q0, q1) pair is independent.
    #pragma unroll
    for (int p = 0; p < 64; p++) {
        float q0 = d_centroids_3bit[unpack_3bit(blk, p * 2)];
        float q1 = d_centroids_3bit[unpack_3bit(blk, p * 2 + 1)];
        float c  = d_planar_cos[p];
        float sn = d_planar_sin[p];
        // Inverse of the rotation matrix [[c, -sn], [sn, c]] is its transpose.
        float v0 = ( c * q0 + sn * q1) * norm;
        float v1 = (-sn * q0 + c  * q1) * norm;
        d[p * 2]     = __float2half(v0);
        d[p * 2 + 1] = __float2half(v1);
    }
}

// ============================================================================
// Host-side launchers (C-linkage so a torch extension binding can call them)
// ============================================================================

extern "C" {

void rq_planar3_pack_f16(
    const __half* d_src,
    BlockPlanar3* d_dst,
    int64_t n_blocks,
    cudaStream_t stream)
{
    if (n_blocks == 0) return;
    constexpr int THREADS = 256;
    int64_t grid = (n_blocks + THREADS - 1) / THREADS;
    kernel_pack_f16_to_planar3<<<grid, THREADS, 0, stream>>>(
        d_src, d_dst, n_blocks);
}

void rq_planar3_unpack_f16(
    const BlockPlanar3* d_src,
    __half* d_dst,
    int64_t n_blocks,
    cudaStream_t stream)
{
    if (n_blocks == 0) return;
    constexpr int THREADS = 256;
    int64_t grid = (n_blocks + THREADS - 1) / THREADS;
    kernel_unpack_planar3_to_f16<<<grid, THREADS, 0, stream>>>(
        d_src, d_dst, n_blocks);
}

}  // extern "C"

}  // namespace rotorquant
}  // namespace rq_models
