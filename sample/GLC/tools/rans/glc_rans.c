//===----------------------------------------------------------------------===//
//
// Copyright (C) 2026 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
/* 纯 C ABI rANS 编解码器，供 ctypes 调用。
 *
 * 语义与 compressai 1.2.8 的 cpp_exts/rans/rans_interface.cpp 逐位对齐：
 *   - precision = 16，bypass_precision = 4
 *   - 编码端做 value = symbol - offset，越界走 bypass 逃逸路径
 *   - 解码端最后补回 offset
 * 不依赖 torch / pybind11 / Python，可直接交叉编译到 aarch64。
 *
 * CDF 传参采用「扁平数组 + 每 index 起始偏移 + 每 index 有效长度」：
 *   cdf[off[i] .. off[i]+len[i]-1] 是第 i 个 index 的完整 CDF 行。
 * 对应 compressai 的 _quantized_cdf（行内 2 项填充）与 _cdf_length（有效长度），
 * 调用方须按实际有效长度传 len，而不是填充后的行宽。
 */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "rans64.h"

#define PRECISION 16
#define BYPASS_PRECISION 4
#define MAX_BYPASS_VAL ((1 << BYPASS_PRECISION) - 1)

/* 这两个 bypass 辅助函数在 compressai 的 rans_interface.cpp 匿名命名空间里，
 * 不属于 rans64.h，此处按原实现照搬。 */
static inline void rans_enc_put_bits(Rans64State *r, uint32_t **pptr,
                                     uint32_t val, uint32_t nbits) {
    uint64_t x = *r;
    uint32_t freq = 1u << (16 - nbits);
    uint64_t x_max = ((RANS64_L >> 16) << 32) * freq;
    if (x >= x_max) {
        *pptr -= 1;
        **pptr = (uint32_t)x;
        x >>= 32;
    }
    *r = (x << nbits) | val;
}

static inline uint32_t rans_dec_get_bits(Rans64State *r, uint32_t **pptr,
                                         uint32_t n_bits) {
    uint64_t x = *r;
    uint32_t val = x & ((1u << n_bits) - 1);
    x >>= n_bits;
    if (x < RANS64_L) {
        x = (x << 32) | **pptr;
        *pptr += 1;
    }
    *r = x;
    return val;
}

typedef struct {
    uint32_t start;
    uint32_t range;
    int bypass;
} RansSym;

/* 按需扩容。一个符号走逃逸路径时会展开成 1 个主符号 + 若干长度计数 + n_bypass
 * 个 4-bit 块，而 n_bypass 随 |symbol| 增长，无法用与 n 成正比的上界封顶。 */
static int grow(RansSym **buf, size_t *cap, size_t need) {
    if (need <= *cap) return 1;
    size_t next = *cap;
    while (next < need) next *= 2;
    RansSym *p = (RansSym *)realloc(*buf, next * sizeof(RansSym));
    if (!p) return 0;
    *buf = p;
    *cap = next;
    return 1;
}

/* 编码端：symbols/indexes 长度 n，逐符号查 CDF 后再反序做 rANS 编码。
 * 返回 malloc 出来的字节流，调用方用 glc_rans_free 释放；失败返回 NULL。 */
uint8_t *glc_rans_encode(const int32_t *symbols, const int32_t *indexes,
                         size_t n, const int32_t *cdf, const int32_t *cdf_len,
                         const int32_t *cdf_off, const int32_t *offsets,
                         size_t n_cdf, size_t *out_len) {
    if (!symbols || !indexes || !cdf || !cdf_len || !cdf_off || !out_len) {
        return NULL;
    }

    size_t cap = 1024;
    RansSym *syms = (RansSym *)malloc(cap * sizeof(RansSym));
    if (!syms) return NULL;
    size_t ns = 0;

    for (size_t i = 0; i < n; ++i) {
        int32_t ci = indexes[i];
        if (ci < 0 || (size_t)ci >= n_cdf) {
            free(syms);
            return NULL;
        }
        const int32_t *c = cdf + cdf_off[ci];
        const int32_t row_len = cdf_len[ci];
        const int32_t max_value = row_len - 2;
        if (max_value < 0) {
            free(syms);
            return NULL;
        }
        /* 官方 encode_with_indexes：value = symbols[i] - offsets[cdf_idx] */
        int32_t value = symbols[i] - (offsets ? offsets[ci] : 0);

        uint32_t raw_val = 0;
        if (value < 0) {
            raw_val = (uint32_t)(-2 * value - 1);
            value = max_value;
        } else if (value >= max_value) {
            raw_val = (uint32_t)(2 * (value - max_value));
            value = max_value;
        }

        if (!grow(&syms, &cap, ns + 2)) { free(syms); return NULL; }
        syms[ns].start = (uint32_t)c[value];
        syms[ns].range = (uint32_t)(c[value + 1] - c[value]);
        syms[ns].bypass = 0;
        ns++;

        if (value == max_value) {
            int32_t n_bypass = 0;
            while ((raw_val >> (n_bypass * BYPASS_PRECISION)) != 0) n_bypass++;

            int32_t v = n_bypass;
            while (v >= MAX_BYPASS_VAL) {
                if (!grow(&syms, &cap, ns + 1)) { free(syms); return NULL; }
                syms[ns].start = MAX_BYPASS_VAL;
                syms[ns].range = MAX_BYPASS_VAL + 1;
                syms[ns].bypass = 1;
                ns++;
                v -= MAX_BYPASS_VAL;
            }
            if (!grow(&syms, &cap, ns + 1)) { free(syms); return NULL; }
            syms[ns].start = (uint32_t)v;
            syms[ns].range = (uint32_t)(v + 1);
            syms[ns].bypass = 1;
            ns++;

            for (int32_t j = 0; j < n_bypass; ++j) {
                if (!grow(&syms, &cap, ns + 1)) { free(syms); return NULL; }
                uint32_t val = (raw_val >> (j * BYPASS_PRECISION)) & MAX_BYPASS_VAL;
                syms[ns].start = val;
                syms[ns].range = val + 1;
                syms[ns].bypass = 1;
                ns++;
            }
        }
    }

    size_t buf_cap = ns + 8;
    uint32_t *buf = (uint32_t *)calloc(buf_cap, sizeof(uint32_t));
    if (!buf) { free(syms); return NULL; }
    uint32_t *ptr = buf + buf_cap;

    Rans64State st;
    Rans64EncInit(&st);
    for (size_t k = ns; k-- > 0;) {
        if (!syms[k].bypass) {
            Rans64EncPut(&st, &ptr, syms[k].start, syms[k].range, PRECISION);
        } else {
            rans_enc_put_bits(&st, &ptr, syms[k].start, BYPASS_PRECISION);
        }
    }
    Rans64EncFlush(&st, &ptr);

    size_t nbytes = (size_t)((buf + buf_cap) - ptr) * sizeof(uint32_t);
    uint8_t *res = (uint8_t *)malloc(nbytes ? nbytes : 1);
    if (res) memcpy(res, ptr, nbytes);
    free(syms);
    free(buf);
    if (!res) return NULL;
    *out_len = nbytes;
    return res;
}

/* 解码端：payload 为 glc_rans_encode 的字节流原样；indexes 长度 n。
 * 返回 malloc 出来的 int32 数组（已补回 offset），调用方用 glc_int32_free 释放。 */
int32_t *glc_rans_decode(const uint8_t *payload, size_t payload_len, size_t n,
                         const int32_t *indexes, const int32_t *cdf,
                         const int32_t *cdf_len, const int32_t *cdf_off,
                         const int32_t *offsets, size_t n_cdf) {
    if (!payload || !indexes || !cdf || !cdf_len || !cdf_off) return NULL;

    int32_t *out = (int32_t *)malloc((n ? n : 1) * sizeof(int32_t));
    if (!out) return NULL;

    /* Rans64DecInit 按 4 字节宽度读初始状态，payload 必须 4 字节对齐。
     * ctypes 从 Python bytes 取到的指针不保证对齐，这里按需拷一份。 */
    uint32_t *aligned = NULL;
    const uint8_t *src = payload;
    if (((uintptr_t)payload & 3u) != 0) {
        size_t need = payload_len ? payload_len : 4;
        aligned = (uint32_t *)malloc(need);
        if (!aligned) { free(out); return NULL; }
        memcpy(aligned, payload, payload_len);
        src = (const uint8_t *)aligned;
    }

    uint32_t *ptr = (uint32_t *)src;
    Rans64State st;
    Rans64DecInit(&st, &ptr);

    for (size_t i = 0; i < n; ++i) {
        int32_t ci = indexes[i];
        if (ci < 0 || (size_t)ci >= n_cdf) { free(out); free(aligned); return NULL; }
        const int32_t *c = cdf + cdf_off[ci];
        const int32_t row_len = cdf_len[ci];
        const int32_t max_value = row_len - 2;
        if (max_value < 0) { free(out); free(aligned); return NULL; }
        const int32_t offset = offsets ? offsets[ci] : 0;

        uint32_t cum = Rans64DecGet(&st, PRECISION);
        /* 等价于官方 find_if(cdf.begin(), cdf.begin()+size, v > cum)：
         * 上界取 row_len-1，而 cdf[row_len-1] == 1<<PRECISION > cum 必然成立。 */
        int32_t s = 0;
        while (s + 1 < row_len && !((uint32_t)c[s + 1] > cum)) s++;

        Rans64DecAdvance(&st, &ptr, (uint32_t)c[s], (uint32_t)(c[s + 1] - c[s]),
                         PRECISION);

        int32_t value = s;
        if (value == max_value) {
            int32_t val = (int32_t)rans_dec_get_bits(&st, &ptr, BYPASS_PRECISION);
            int32_t n_bypass = val;
            while (val == MAX_BYPASS_VAL) {
                val = (int32_t)rans_dec_get_bits(&st, &ptr, BYPASS_PRECISION);
                n_bypass += val;
            }
            uint32_t raw_val = 0;
            for (int32_t j = 0; j < n_bypass; ++j) {
                val = (int32_t)rans_dec_get_bits(&st, &ptr, BYPASS_PRECISION);
                raw_val |= (uint32_t)val << (j * BYPASS_PRECISION);
            }
            value = (int32_t)(raw_val >> 1);
            if (raw_val & 1) {
                value = -value - 1;
            } else {
                value += max_value;
            }
        }
        /* 官方 decode_with_indexes：output[i] = value + offset */
        out[i] = value + offset;
    }

    free(aligned);
    return out;
}

void glc_rans_free(void *p) { free(p); }
void glc_int32_free(int32_t *p) { free(p); }
