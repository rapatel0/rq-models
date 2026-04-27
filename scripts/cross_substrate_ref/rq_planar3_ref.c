/* Reference C emulator of llama-cpp-turboquant's CUDA-path planar3
 * encode + decode. Same algorithm as ggml-planar-quant.c but uses the
 * CUDA-path PI_COS / PI_SIN constants from
 * ggml-cuda/planar-iso-constants.cuh, which is the table our rq-vllm
 * CUDA kernel ports byte-for-byte.
 *
 * This is the right oracle for cross-substrate parity against rq-vllm.
 *
 * Build:
 *   gcc -O3 -std=c11 /tmp/rq_ref_cuda_path.c -lm -o /tmp/rq_planar3_ref
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#define QK_PLANAR3 128
#define PLANAR_D 128
#define BLOCK_BYTES 50

typedef struct {
    uint16_t norm;
    uint8_t  qs[QK_PLANAR3 / 4];
    uint8_t  signs[QK_PLANAR3 / 8];
} block_planar3_0_t;

/* Centroids — same on CPU and CUDA path. */
static const float CENT[8] = {
    -0.190685f, -0.117832f, -0.065717f, -0.021460f,
     0.021460f,  0.065717f,  0.117832f,  0.190685f,
};
/* Midpoints — for CUDA-style fast quantize_3bit. */
static const float MID[7] = {
    -0.154259f, -0.091775f, -0.043589f, 0.0f, 0.043589f, 0.091775f, 0.154259f,
};

/* CUDA-path PI_COS / PI_SIN — copied verbatim from
 * ggml-cuda/planar-iso-constants.cuh. */
static const float PI_COS[64] = {
    -0.9095053397f,0.1535578452f,-0.8537489227f,-0.6827218011f,-0.4249387949f,0.9864510046f,0.9906673944f,0.5752363372f,
    -0.9866459035f,0.9878848090f,-0.6215683804f,-0.9835597698f,0.8777263755f,-0.4624640047f,0.2843135922f,-0.7739960698f,
     0.2385234222f,0.9121914932f,-0.8815003943f,-0.2639699512f,-0.5517087300f,-0.9035294557f,-0.8520543188f,-0.5600635985f,
    -0.7667286376f,-0.9877949369f,-0.9781949787f,-0.9953372831f,-0.8622053901f,-0.7382118186f,0.9136037642f,-0.2558504503f,
    -0.8541000475f,-0.6159335408f,0.9861256679f,-0.6758560284f,0.4249571682f,-0.6219544719f,0.9130573430f,-0.5948161096f,
     0.5759782996f,0.9729901203f,0.6535998325f,0.9222195491f,-0.7668084044f,0.5116178563f,-0.7848786574f,0.9902111051f,
     0.1997167840f,0.7173003220f,-0.9999998006f,-0.9557868691f,0.5594852693f,-0.9980111824f,0.9782398557f,-0.9150004329f,
    -0.4084754305f,0.0071549185f,0.9558482753f,-0.0971921648f,-0.9469334002f,0.9999492419f,0.6100589016f,0.0350818915f,
};
static const float PI_SIN[64] = {
    -0.4156922383f,0.9881396603f,0.5206849114f,-0.7306784124f,-0.9052220836f,0.1640561354f,0.1363015542f,0.8179872593f,
     0.1628798979f,0.1551889303f,0.7833599099f,-0.1805828875f,-0.4791621957f,0.8866380571f,-0.9587313395f,0.6331904010f,
    -0.9711367448f,0.4097641756f,0.4721832852f,-0.9645309040f,0.8340368561f,0.4285259884f,0.5234533769f,0.8284496156f,
     0.6419713361f,-0.1557599517f,-0.2076886701f,0.0964556523f,0.5065588468f,-0.6745689815f,-0.4066056591f,-0.9667163736f,
     0.5201087471f,-0.7877981171f,0.1660005034f,-0.7370336688f,0.9052134584f,0.7830534049f,-0.4078312009f,-0.8038618014f,
     0.8174649829f,-0.2308467584f,-0.7568403127f,-0.3866666566f,0.6418760557f,-0.8592131104f,0.6196494922f,0.1395778183f,
     0.9798536657f,0.6967641265f,-0.0006314605f,0.2940603015f,0.8288402943f,-0.0630371303f,0.2074771907f,0.4034528570f,
     0.9127693152f,-0.9999744032f,0.2938606379f,0.9952656344f,0.3214298299f,0.0100754012f,-0.7923560668f,-0.9993844410f,
};

/* IEEE-754 fp16 conversions matching CUDA's __half2float / __float2half.
 * CUDA's __half2float is exact (no IEEE-754 magic); __float2half does
 * round-to-nearest-even. Our implementation matches that contract. */
static float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (h >> 15) & 1u;
    uint32_t exp  = (h >> 10) & 0x1fu;
    uint32_t mant = h & 0x3ffu;
    uint32_t f;
    if (exp == 0) {
        if (mant == 0) f = sign << 31;
        else { int e = -1; do { e++; mant <<= 1; } while ((mant & 0x400u) == 0);
               mant &= 0x3ffu; f = (sign << 31) | (((uint32_t)(127 - 15 - e)) << 23) | (mant << 13); }
    } else if (exp == 0x1fu) f = (sign << 31) | 0x7f800000u | (mant << 13);
    else                     f = (sign << 31) | (((uint32_t)exp - 15 + 127) << 23) | (mant << 13);
    union { uint32_t u; float f; } v; v.u = f; return v.f;
}

static uint16_t fp32_to_fp16(float f) {
    union { float f; uint32_t u; } v; v.f = f;
    uint32_t u = v.u;
    uint32_t sign = (u >> 31) & 1u;
    int32_t exp  = (int32_t)((u >> 23) & 0xffu) - 127 + 15;
    uint32_t mant = u & 0x7fffffu;
    if (exp >= 0x1f) return (uint16_t)((sign << 15) | 0x7c00u | (mant ? 0x200u : 0u));
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)(sign << 15);
        mant |= 0x800000u;
        uint32_t shift = (uint32_t)(14 - exp);
        uint32_t result = mant >> shift;
        uint32_t rem = mant & ((1u << shift) - 1u);
        uint32_t half = 1u << (shift - 1);
        if (rem > half || (rem == half && (result & 1u))) result++;
        return (uint16_t)((sign << 15) | result);
    }
    uint32_t result = ((uint32_t)exp << 10) | (mant >> 13);
    uint32_t rem = mant & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (result & 1u))) result++;
    return (uint16_t)((sign << 15) | result);
}

/* CUDA-style fast 3-bit quantize via midpoint search. */
static uint8_t quantize_3bit(float val) {
    if      (val < MID[0]) return 0;
    else if (val < MID[1]) return 1;
    else if (val < MID[2]) return 2;
    else if (val < MID[3]) return 3;
    else if (val < MID[4]) return 4;
    else if (val < MID[5]) return 5;
    else if (val < MID[6]) return 6;
    else                   return 7;
}

static void encode_block(const float *x, block_planar3_0_t *blk) {
    /* per-block L2 norm */
    float norm_sq = 0.0f;
    for (int j = 0; j < QK_PLANAR3; j++) norm_sq += x[j] * x[j];
    float grp_norm = sqrtf(norm_sq);
    float inv_norm = grp_norm > 1e-10f ? 1.0f / grp_norm : 0.0f;

    float buf[QK_PLANAR3];
    for (int j = 0; j < QK_PLANAR3; j++) buf[j] = x[j] * inv_norm;

    /* forward Givens on (p*2, p*2+1) */
    float rotated[QK_PLANAR3];
    for (int p = 0; p < 64; p++) {
        float c = PI_COS[p], s = PI_SIN[p];
        rotated[p*2]   = c * buf[p*2] - s * buf[p*2+1];
        rotated[p*2+1] = s * buf[p*2] + c * buf[p*2+1];
    }

    memset(blk->qs, 0, sizeof(blk->qs));
    memset(blk->signs, 0, sizeof(blk->signs));

    float recon_sq = 0.0f;
    for (int j = 0; j < QK_PLANAR3; j++) {
        uint8_t idx = quantize_3bit(rotated[j]);
        blk->qs[j/4] |= (idx & 0x3) << ((j%4) * 2);
        if (idx & 0x4) blk->signs[j/8] |= (1u << (j%8));
        recon_sq += CENT[idx] * CENT[idx];
    }

    float recon_norm = sqrtf(recon_sq);
    float corrected = recon_norm > 1e-10f ? grp_norm / recon_norm : grp_norm;
    blk->norm = fp32_to_fp16(corrected);
}

static void decode_block(const block_planar3_0_t *blk, float *y) {
    float norm = fp16_to_fp32(blk->norm);
    for (int p = 0; p < 64; p++) {
        int j0 = p*2, j1 = p*2 + 1;
        uint8_t low0 = (blk->qs[j0/4] >> ((j0%4)*2)) & 0x3;
        uint8_t hi0  = (blk->signs[j0/8] >> (j0%8)) & 0x1;
        uint8_t idx0 = low0 | (hi0 << 2);
        uint8_t low1 = (blk->qs[j1/4] >> ((j1%4)*2)) & 0x3;
        uint8_t hi1  = (blk->signs[j1/8] >> (j1%8)) & 0x1;
        uint8_t idx1 = low1 | (hi1 << 2);
        float q0 = CENT[idx0], q1 = CENT[idx1];

        float c = PI_COS[p], s = PI_SIN[p];
        /* inverse rotation: undo r0 = c*v0 - s*v1; r1 = s*v0 + c*v1
         * → v0 = c*r0 + s*r1; v1 = -s*r0 + c*r1 */
        y[j0] = (c * q0 + s * q1) * norm;
        y[j1] = (-s * q0 + c * q1) * norm;
    }
}

static const char MAGIC[4] = {'R', 'Q', 'K', '1'};

int main(int argc, char **argv) {
    if (argc != 3) { fprintf(stderr, "usage: %s <in.fp16> <out.fp16>\n", argv[0]); return 2; }
    FILE *fin = fopen(argv[1], "rb");
    if (!fin) { perror("input"); return 1; }
    char magic[4];
    if (fread(magic, 1, 4, fin) != 4 || memcmp(magic, MAGIC, 4) != 0) { fprintf(stderr, "bad magic\n"); return 1; }
    uint32_t n_blocks = 0;
    if (fread(&n_blocks, 4, 1, fin) != 1) { fprintf(stderr, "short read\n"); return 1; }

    size_t n_elem = (size_t)n_blocks * QK_PLANAR3;
    uint16_t *src = malloc(n_elem * sizeof(uint16_t));
    if (fread(src, sizeof(uint16_t), n_elem, fin) != n_elem) { fprintf(stderr, "short K read\n"); return 1; }
    fclose(fin);

    float *src_f32 = malloc(n_elem * sizeof(float));
    float *dst_f32 = malloc(n_elem * sizeof(float));
    block_planar3_0_t *packed = malloc(n_blocks * sizeof(block_planar3_0_t));
    uint16_t *dst = malloc(n_elem * sizeof(uint16_t));

    for (size_t i = 0; i < n_elem; i++) src_f32[i] = fp16_to_fp32(src[i]);
    for (uint32_t b = 0; b < n_blocks; b++) {
        encode_block(src_f32 + b * QK_PLANAR3, &packed[b]);
        decode_block(&packed[b], dst_f32 + b * QK_PLANAR3);
    }
    for (size_t i = 0; i < n_elem; i++) dst[i] = fp32_to_fp16(dst_f32[i]);

    FILE *fout = fopen(argv[2], "wb");
    if (!fout) { perror("output"); return 1; }
    fwrite(MAGIC, 1, 4, fout);
    fwrite(&n_blocks, 4, 1, fout);
    fwrite(dst, sizeof(uint16_t), n_elem, fout);
    fclose(fout);

    fprintf(stderr, "OK: %u blocks\n", n_blocks);
    free(src); free(src_f32); free(dst_f32); free(packed); free(dst);
    return 0;
}
