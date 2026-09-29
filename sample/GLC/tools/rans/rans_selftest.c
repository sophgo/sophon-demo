//===----------------------------------------------------------------------===//
//
// Copyright (C) 2026 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
/* aarch64 冒烟测试：不依赖 Python，直接用 C 驱动 glc_rans_encode/decode。
 *
 * 目的是证明交叉编译出的 aarch64 目标代码真的能跑通（在 qemu-aarch64 下执行），
 * 而不是只证明「能编译」。同一份 main 也在 x86_64 上跑，两端结果必须一致。
 *
 * 用法：
 *   gcc -O2 -static -I<rans 目录> -o rans_selftest glc_rans.c rans_selftest.c
 *   ./rans_selftest            # 本机
 *   qemu-aarch64 ./rans_selftest_aarch64
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

uint8_t *glc_rans_encode(const int32_t *symbols, const int32_t *indexes,
                         size_t n, const int32_t *cdf, const int32_t *cdf_len,
                         const int32_t *cdf_off, const int32_t *offsets,
                         size_t n_cdf, size_t *out_len);
int32_t *glc_rans_decode(const uint8_t *payload, size_t payload_len, size_t n,
                         const int32_t *indexes, const int32_t *cdf,
                         const int32_t *cdf_len, const int32_t *cdf_off,
                         const int32_t *offsets, size_t n_cdf);
void glc_rans_free(void *p);
void glc_int32_free(int32_t *p);

/* 一个 4 符号均匀分布（precision=16）的 CDF 行，外加一个更宽的 6 符号行。
 * 行宽不同是为了确认 cdf_off 的跳转正确。 */
static const int32_t CDF_ROW0[] = {0, 16384, 32768, 49152, 65536};           /* 4 值 */
static const int32_t CDF_ROW1[] = {0, 8192, 16384, 32768, 49152, 65536};    /* 5 值 */

int main(void) {
    const int32_t cdf[] = {
        0, 16384, 32768, 49152, 65536,
        0, 8192, 16384, 32768, 49152, 65536,
    };
    const int32_t cdf_len[] = {5, 6};
    const int32_t cdf_off[] = {0, 5};
    const int32_t offsets[] = {0, 0};
    const size_t n_cdf = 2;

    /* 混入大值触发 bypass 逃逸路径 */
    int32_t symbols[] = {0, 1, 2, 1, 0, 3, -2, 7, 5000, -5001, 0, 2};
    int32_t indexes[] = {0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1};
    const size_t n = sizeof(symbols) / sizeof(symbols[0]);
    const int32_t expect[] = {0, 1, 2, 1, 0, 3, -2, 7, 5000, -5001, 0, 2};

    size_t out_len = 0;
    uint8_t *payload = glc_rans_encode(symbols, indexes, n, cdf, cdf_len, cdf_off,
                                       offsets, n_cdf, &out_len);
    if (!payload) {
        printf("FAIL: encode 返回 NULL\n");
        return 1;
    }

    printf("encoded %zu bytes:", out_len);
    for (size_t i = 0; i < out_len; ++i) printf(" %02x", payload[i]);
    printf("\n");

    int32_t *dec = glc_rans_decode(payload, out_len, n, indexes, cdf, cdf_len,
                                   cdf_off, offsets, n_cdf);
    if (!dec) {
        printf("FAIL: decode 返回 NULL\n");
        glc_rans_free(payload);
        return 1;
    }

    int bad = 0;
    for (size_t i = 0; i < n; ++i) {
        if (dec[i] != expect[i]) {
            printf("FAIL: 符号 %zu 期望 %d 实得 %d\n", i, expect[i], dec[i]);
            bad = 1;
        }
    }

    printf("roundtrip: %s\n", bad ? "FAIL" : "PASS");
    printf("decoded:");
    for (size_t i = 0; i < n; ++i) printf(" %d", dec[i]);
    printf("\n");

    glc_rans_free(payload);
    glc_int32_free(dec);
    return bad;
}
