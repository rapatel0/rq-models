/* rq_planar3_ref — minimal cross-substrate reference for the planar3 KV
 * round-trip. Reads a binary blob in the rq-models cross-substrate
 * format, runs llama.cpp-turboquant's planar3 quantize + dequantize on
 * each 128-element block, writes the round-tripped output in the same
 * format.
 *
 * Used by scripts/cross_substrate_parity.py to verify that the
 * rq-vllm CUDA kernel produces output bit-identical (or fp-equivalent)
 * to the original C reference for any input — including the
 * pathological Qwen3 L0 k_norm distributions where the codebook itself
 * loses direction. If a porting bug existed, it would show up here as
 * a non-zero ULP diff.
 *
 * Binary protocol (must match scripts/cross_substrate_parity.py):
 *   bytes 0..3   : magic "RQK1"
 *   bytes 4..7   : little-endian uint32, n_blocks
 *   bytes 8..N   : n_blocks * 128 * 2 bytes of fp16, row-major
 *
 * Build (against a checked-out llama.cpp-turboquant tree):
 *
 *   git clone git@github.com:johndpope/llama-cpp-turboquant.git
 *   cd llama-cpp-turboquant && git checkout fc3d1b6 && cmake -B build && cmake --build build -j
 *   gcc -O3 -I /path/to/llama-cpp-turboquant/ggml/include \\
 *       -I /path/to/llama-cpp-turboquant/ggml/src \\
 *       scripts/cross_substrate_ref/rq_planar3_ref.c \\
 *       /path/to/llama-cpp-turboquant/build/ggml/src/libggml.a \\
 *       -lm -o /tmp/rq_planar3_ref
 *
 * Usage:
 *
 *   /tmp/rq_planar3_ref input.fp16 output.fp16
 *
 * The exact symbol names below (quantize_row_planar3_reference,
 * dequantize_row_planar3) match llama-cpp-turboquant's
 * feature/planarquant-kv-cache branch at commit 20efe75 / fc3d1b6.
 * Adjust the function names if your fork has renamed them.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

/* From ggml.h / ggml-quants.h in llama-cpp-turboquant. The exact
 * include paths depend on your build layout. */
#include "ggml.h"
#include "ggml-quants.h"

#define QK_PLANAR3   128
#define BLOCK_BYTES  50

static const char MAGIC[4] = {'R', 'Q', 'K', '1'};

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s <input.fp16> <output.fp16>\n", argv[0]);
        return 2;
    }

    FILE *fin = fopen(argv[1], "rb");
    if (!fin) { perror("input"); return 1; }

    char magic[4];
    if (fread(magic, 1, 4, fin) != 4 || memcmp(magic, MAGIC, 4) != 0) {
        fprintf(stderr, "bad magic in %s\n", argv[1]);
        return 1;
    }
    uint32_t n_blocks = 0;
    if (fread(&n_blocks, 4, 1, fin) != 1) {
        fprintf(stderr, "short read for n_blocks\n");
        return 1;
    }

    size_t n_elem = (size_t)n_blocks * QK_PLANAR3;
    ggml_fp16_t *src = (ggml_fp16_t *)malloc(n_elem * sizeof(ggml_fp16_t));
    if (!src) { fprintf(stderr, "OOM src\n"); return 1; }
    if (fread(src, sizeof(ggml_fp16_t), n_elem, fin) != n_elem) {
        fprintf(stderr, "short read for K data\n");
        return 1;
    }
    fclose(fin);

    /* Convert to fp32 — the planar3 reference takes float input. */
    float *src_f32 = (float *)malloc(n_elem * sizeof(float));
    float *dst_f32 = (float *)malloc(n_elem * sizeof(float));
    uint8_t *packed = (uint8_t *)malloc((size_t)n_blocks * BLOCK_BYTES);
    if (!src_f32 || !dst_f32 || !packed) { fprintf(stderr, "OOM bufs\n"); return 1; }

    for (size_t i = 0; i < n_elem; ++i) {
        src_f32[i] = ggml_fp16_to_fp32(src[i]);
    }

    /* The exact API names below come from llama-cpp-turboquant's
     * feature/planarquant-kv-cache. Update if your branch differs.
     *
     * Encode: float[k] -> packed bytes
     *   void quantize_row_planar3_reference(const float * x,
     *                                       block_planar3_0 * y,
     *                                       int64_t k);
     *
     * Decode: packed bytes -> float[k]
     *   void dequantize_row_planar3(const block_planar3_0 * x,
     *                               float * y, int64_t k);
     */
    quantize_row_planar3_reference(src_f32,
                                   (struct block_planar3_0 *)packed,
                                   (int64_t)n_elem);
    dequantize_row_planar3((const struct block_planar3_0 *)packed,
                           dst_f32, (int64_t)n_elem);

    /* Convert back to fp16 for byte-identical output framing. */
    ggml_fp16_t *dst = (ggml_fp16_t *)malloc(n_elem * sizeof(ggml_fp16_t));
    if (!dst) { fprintf(stderr, "OOM dst\n"); return 1; }
    for (size_t i = 0; i < n_elem; ++i) {
        dst[i] = ggml_fp32_to_fp16(dst_f32[i]);
    }

    FILE *fout = fopen(argv[2], "wb");
    if (!fout) { perror("output"); return 1; }
    fwrite(MAGIC, 1, 4, fout);
    fwrite(&n_blocks, 4, 1, fout);
    fwrite(dst, sizeof(ggml_fp16_t), n_elem, fout);
    fclose(fout);

    free(src); free(src_f32); free(dst_f32); free(packed); free(dst);
    return 0;
}
