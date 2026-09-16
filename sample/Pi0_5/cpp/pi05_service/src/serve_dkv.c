//===----------------------------------------------------------------------===//
//
// Copyright (C) 2026 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
/* serve_dkv.c — D 列 KV 分离架构服务:
 * dkv(prefix 18 层 KV 常驻) + ddn_0_6/6_12/12_18(denoise 链).
 * 协议(stdin/stdout):
 *   'P' + prefix(968*2048 f32) + p_amask(968*968) + p_cos/p_sin(968*256) + f4d(10*978) + s_cos/s_sin(10*256) + time(1 f32)
 *       -> dkv 前向 -> 36 KV 常驻 -> 'OK'
 *   'S' + x_t(10*32 f32) -> ddn_0_6(x_t) -> ddn_6_12 -> ddn_12_18_final -> v_t(320 f32)
 * env: BMRUNTIME_NEURON_HEAP_MASK=7
 * usage: serve_dkv <devid> <dkv.bmodel> <ddn0> <ddn1> <ddn2>
 */
#include <stdio.h>
#include <malloc.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "bmdef.h"
#include "bmlib_runtime.h"
#include "bmruntime_interface.h"

#define PL 536   /* H5: 968 -> 536(删 436 死 token; 留槽给 t04) */
#define AH 10
#define KV_N 36          /* 18 层 x (k,v) */
#define KV_SZ (PL * 256) /* 每层 k 或 v 的 f32 数 */

typedef struct {
    const bm_net_info_t* net;
    const bm_stage_info_t* stage;
    char* name;
    /* 2026-09-14: 设备侧输入/输出 buffer 缓存.
     * 原实现每次 net_forward 都 bm_malloc_device_mem / bm_free_device_mem 一遍，
     * 一次推理约 245 对(1×P 45 + 10×S 20)，实测 serve_dkv 的 RSS ~1.9MB/次推理
     * 单向增长(34 次推理 4.4MB→68MB)，在 ~1GB 系统 RAM 的设备上最终触发 OOM。
     * 这些 buffer 的尺寸/形状对同一张网是固定的，故按网缓存、进程内复用。 */
    bm_tensor_t* in_t;
    bm_tensor_t* out_t;
    int nin, nout;
    int buf_ready;
} net_t;

static int load_net(void* bmrt, const char* path, net_t* sn) {
    int n_before = bmrt_get_network_number(bmrt);
    if (!bmrt_load_bmodel(bmrt, path)) { fprintf(stderr, "load fail: %s\n", path); return 1; }
    int n_after = bmrt_get_network_number(bmrt);
    const char** names = NULL;
    bmrt_get_network_names(bmrt, &names);
    const char* fixed = "";
    for (int i = n_before; i < n_after && names; i++) { fixed = names[i]; break; }
    sn->net = bmrt_get_network_info(bmrt, fixed);
    if (!sn->net) { fprintf(stderr, "net info fail %s\n", fixed); return 1; }
    sn->stage = &sn->net->stages[0];
    sn->name = strdup(fixed);
    sn->in_t = NULL; sn->out_t = NULL; sn->buf_ready = 0;
    sn->nin = sn->net->input_num; sn->nout = sn->net->output_num;
    fprintf(stderr, "NET %s nin=%d nout=%d\n", fixed, sn->nin, sn->nout);
    return 0;
}

/* 首次调用时为该网分配并初始化设备侧 in/out tensor(含 device_mem); 之后复用. */
static int net_bufs_init(bm_handle_t handle, net_t* sn) {
    if (sn->buf_ready) return 0;
    const bm_net_info_t* net = sn->net;
    sn->in_t = (bm_tensor_t*)calloc(sn->nin, sizeof(bm_tensor_t));
    sn->out_t = (bm_tensor_t*)calloc(sn->nout, sizeof(bm_tensor_t));
    if (!sn->in_t || !sn->out_t) { fprintf(stderr, "buf alloc fail\n"); return -1; }
    for (int i = 0; i < sn->nin; i++) {
        size_t want = net->max_input_bytes[i];
        unsigned long long pa = 0;
        if (bm_malloc_device_mem(handle, &pa, 0, want)) { fprintf(stderr, "in dev mem fail\n"); return -1; }
        sn->in_t[i].dtype = net->input_dtypes[i];
        sn->in_t[i].shape = sn->stage->input_shapes[i];
        sn->in_t[i].st_mode = BM_STORE_1N;
        sn->in_t[i].device_mem = bm_mem_from_device(pa, want);
    }
    for (int i = 0; i < sn->nout; i++) {
        size_t want = net->max_output_bytes[i];
        unsigned long long pa = 0;
        if (bm_malloc_device_mem(handle, &pa, 0, want)) { fprintf(stderr, "out dev mem fail\n"); return -1; }
        sn->out_t[i].dtype = net->output_dtypes[i];
        sn->out_t[i].shape = sn->stage->output_shapes[i];
        sn->out_t[i].st_mode = BM_STORE_1N;
        sn->out_t[i].device_mem = bm_mem_from_device(pa, want);
    }
    sn->buf_ready = 1;
    return 0;
}

/* 复用 sn 的设备 buffer: 只做 host->device 拷贝 + launch + device->host 拷贝.
 * out_bufs 仍由调用方 malloc 并持有(所有权语义与原来一致).
 *
 * 2026-09-14 性能优化 —— 新增「设备常驻张量」支持:
 *   in_dev[i].size  > 0 : 第 i 个输入直接指向该设备内存, 跳过 host->device 拷贝(不读 in_bufs[i]);
 *   keep_out != 0       : 全部输出保留在设备上, 跳过 device->host 拷贝并写入 out_dev[0..nout-1],
 *                         out_bufs[i] 置 NULL (调用方不得 free; 该内存属 sn->out_t, 进程内常驻).
 *                         —— 输出侧必须用显式标志: 调用方的 out_dev 数组本身就是清零的,
 *                         不能用 "size>0" 当"请保留"的信号(第一版就踩了这个坑, 表现为 dkv1 fail).
 * 两者都不改变任何算术: 用的是同一块设备内存、同一份字节, 只是不再在 host/device 之间来回搬.
 * 传 NULL 即完全退化为原行为.
 * 注意: sn->in_t[i].device_mem 不被就地修改 —— 每次 launch 用本地 bm_tensor_t 副本,
 * 否则一次设备直连会永久污染该网后续的 host 输入路径(会打破 K 主机 KV 模式). */
#define MAX_NIO 64
static int net_forward_ex(bm_handle_t handle, void* bmrt, net_t* sn,
                          void** in_bufs, const bm_device_mem_t* in_dev,
                          void** out_bufs, size_t* out_szs,
                          int keep_out, bm_device_mem_t* out_dev) {
    const bm_net_info_t* net = sn->net;
    int nin = sn->nin, nout = sn->nout;
    if (nin > MAX_NIO || nout > MAX_NIO) { fprintf(stderr, "too many io\n"); return -1; }
    if (net_bufs_init(handle, sn)) return -1;
    bm_tensor_t tin[MAX_NIO], tout[MAX_NIO];
    for (int i = 0; i < nin; i++) {
        tin[i] = sn->in_t[i];
        if (in_dev && in_dev[i].size > 0) {
            tin[i].device_mem = in_dev[i];          /* 设备常驻: 零拷贝 */
        } else {
            if (bm_memcpy_s2d(handle, sn->in_t[i].device_mem, in_bufs[i])) return -1;
        }
    }
    for (int i = 0; i < nout; i++) tout[i] = sn->out_t[i];
    int ok = bmrt_launch_tensor_ex(bmrt, sn->name, tin, nin, tout, nout, true, false);
    bm_thread_sync(handle);
    /* 静态网下 runtime 不改写用户提供的 tensor; 这里回写只是保险(动态网若重定位输出地址也能接住).
     * 输入侧刻意不回写 —— 那会把本次的设备直连别名永久留在 sn->in_t 上. */
    for (int i = 0; i < nout; i++) sn->out_t[i].device_mem = tout[i].device_mem;
    if (ok) {
        for (int i = 0; i < nout; i++) {
            out_szs[i] = net->max_output_bytes[i];
            if (keep_out) {
                out_dev[i] = tout[i].device_mem;    /* 留在设备上 */
                out_bufs[i] = NULL;
            } else {
                out_bufs[i] = malloc(out_szs[i]);
                if (bm_memcpy_d2s(handle, out_bufs[i], tout[i].device_mem)) ok = 0;
            }
        }
    }
    return ok ? 0 : -1;
}

static int net_forward(bm_handle_t handle, void* bmrt, net_t* sn,
                       void** in_bufs, void** out_bufs, size_t* out_szs) {
    return net_forward_ex(handle, bmrt, sn, in_bufs, NULL, out_bufs, out_szs, 0, NULL);
}

int main(int argc, char** argv) {
    setenv("BMRUNTIME_NEURON_HEAP_MASK", "7", 1);  /* npu+vpu+vpp 三 heap */
    if (argc < 8) { fprintf(stderr, "usage: serve_dkv <devid> <siglip> <dkv0> <dkv1> <ddn0> <ddn1> <ddn2>\n"); return 2; }
    int devid = atoi(argv[1]);
    bm_handle_t handle;
    if (bm_dev_request(&handle, devid)) { fprintf(stderr, "dev req fail\n"); return 1; }
    void* bmrt = bmrt_create(handle);
    if (!bmrt) { fprintf(stderr, "bmrt fail\n"); return 1; }
    net_t nets[6];
    for (int i = 0; i < 6; i++) if (load_net(bmrt, argv[2+i], &nets[i])) return 1;
    net_t* sig = &nets[0];
    net_t* dkv[2];           /* dkv[0]=dkv0_9: prefix_embs -> 18 KV + hidden; dkv[1]=dkv9_18: hidden -> 18 KV */
    net_t* dseg[3] = {&nets[3], &nets[4], &nets[5]};
    dkv[0] = &nets[1]; dkv[1] = &nets[2];
    fprintf(stderr, "LOADED siglip + dkv x2 + 3 ddn, dev%d\n", devid);

    size_t sz_prefix = (size_t)PL * 2048 * 4;
    size_t sz_pam = (size_t)PL * PL * 4;
    size_t sz_pcossin = (size_t)PL * 256 * 4;
    size_t sz_f4d = (size_t)AH * (PL + AH) * 4;   /* H5: 978 -> PL+AH */
    size_t sz_scossin = (size_t)AH * 256 * 4;
    float* prefix = (float*)malloc(sz_prefix);
    float* p_amask = (float*)malloc(sz_pam);
    float* p_cos = (float*)malloc(sz_pcossin);
    float* p_sin = (float*)malloc(sz_pcossin);
    float* f4d = (float*)malloc(sz_f4d);
    float* s_cos = (float*)malloc(sz_scossin);
    float* s_sin = (float*)malloc(sz_scossin);
    float* time_ = (float*)malloc(4);
    float* x_t = (float*)malloc((size_t)AH * 32 * 4);
    /* KV 常驻 host 缓冲(每层 k/v 交错, 与 dkv 输出名顺序一致) */
    float** kv = (float**)calloc(KV_N, sizeof(float*));  /* 清零, 首次 P 不 free 野指针 */
    /* 2026-09-14 性能优化: P 段算出的 36 个 KV 常驻设备内存.
     * 原实现把 KV 回拷 host(P 段 d2s 52ms), 再在每条 S 命令里重新上传(dn2 共 48ms),
     * 一次推理白搬 ~110MB @ ~1.5GB/s. 改为设备常驻后 ddn 段直接引用该设备内存.
     * kv_dev_ok=0 时(K 主机-KV 调试模式)仍走原来的 host 缓冲 + s2d 路径. */
    bm_device_mem_t kv_dev[KV_N];
    int kv_dev_ok = 0;
    memset(kv_dev, 0, sizeof(kv_dev));
    if (!prefix||!p_amask||!p_cos||!p_sin||!f4d||!s_cos||!s_sin||!time_||!x_t||!kv) return 1;

    char cmd;
    int memdbg = getenv("PI05_MEMDBG") != NULL;
    float* obs3 = (float*)malloc((size_t)3 * 3 * 224 * 224 * 4);  /* I: 3 obs CHW f32 */
    if (!obs3) return 1;
    for (;;) {
        if (fread(&cmd, 1, 1, stdin) != 1) break;
        if (memdbg) {
            /* 直接读自身 RSS: 更贴近"会不会 OOM"的判据, 且不依赖 glibc 版本 */
            long rss_pages = 0, dummy;
            FILE* sf = fopen("/proc/self/statm", "r");
            if (sf) { if (fscanf(sf, "%ld %ld", &dummy, &rss_pages) != 2) rss_pages = -1; fclose(sf); }
            fprintf(stderr, "MEMDBG cmd=%c rss_kb=%ld\n", cmd, rss_pages * 4);
        }
        if (cmd == 'I' || cmd == 'J') {
            /* I: 3 obs -> 3 x siglip feats;  J: 2 obs -> 2 x siglip feats.
             * 2026-09-14: 第 3 路(补零黑图)的 siglip 结果在 host 侧被常量 z3h 整段覆盖
             * (见 tcp_serve_dkv.c), 即设备白跑 37ms/次推理. 改用 'J' 只跑前两路,
             * prefix 的 [512:768] 段仍由 z3h 填充 —— 数值路径完全不变(见等价性验证). */
            int nimg = (cmd == 'I') ? 3 : 2;
            if (fread(obs3, 4, (size_t)nimg * 3 * 224 * 224, stdin) != (size_t)nimg * 3 * 224 * 224) return 1;
            /* 2026-09-15: siglip 模型改成 batch=2(输入 [2,3,224,224])。
             * 生产路径(cmd='J', 2 图)一次调用出两张图的 feats —— 省掉一次 571MB 权重加载,
             * 实测 75.7 -> 61.5 ms(-14.2ms/-18.7%), 且两张图的输出与 batch1 逐位完全相同
             * (相对 L2 = 0.000000%, max|diff| = 0), 所以数值路径不变。
             * 旧的 3 图调试通道(cmd='I')没有 batch=3 的模型, 用两次 batch2 调用拼出来:
             * 先跑 [img0,img1], 再跑 [img2,img2] 取第一份。 */
            if (nimg == 2) {
                void* ib[1] = { obs3 };
                void* ob[1]; size_t os[1];
                if (net_forward(handle, bmrt, sig, ib, ob, os)) { fprintf(stderr, "sig fail\n"); return 1; }
                fwrite(ob[0], 1, 2 * 256 * 2048 * 4, stdout); free(ob[0]);
            } else {
                const size_t ONE = (size_t)3 * 224 * 224;
                void* ib[1] = { obs3 };
                void* ob[1]; size_t os[1];
                if (net_forward(handle, bmrt, sig, ib, ob, os)) { fprintf(stderr, "sig fail\n"); return 1; }
                fwrite(ob[0], 1, 2 * 256 * 2048 * 4, stdout); free(ob[0]);
                memcpy(obs3, obs3 + 2 * ONE, ONE * 4);              /* img2 -> 槽 0 */
                memcpy(obs3 + ONE, obs3, ONE * 4);                  /* img2 -> 槽 1 */
                if (net_forward(handle, bmrt, sig, ib, ob, os)) { fprintf(stderr, "sig fail\n"); return 1; }
                fwrite(ob[0], 1, 256 * 2048 * 4, stdout); free(ob[0]);   /* 只取槽 0 */
            }
            fflush(stdout);
        } else if (cmd == 'K') { /* host KV 模式: 直接收 36 个 KV(各 968*256 f32) + 资产, 跳过 dkv 前向 */
            if (fread(p_amask, 1, sz_pam, stdin) != sz_pam) return 1;
            if (fread(f4d, 1, sz_f4d, stdin) != sz_f4d) return 1;
            if (fread(s_cos, 1, sz_scossin, stdin) != sz_scossin) return 1;
            if (fread(s_sin, 1, sz_scossin, stdin) != sz_scossin) return 1;
            for (int i = 0; i < KV_N; i++) free(kv[i]);
            size_t sz_kv1 = (size_t)PL * 256 * 4;
            for (int i = 0; i < KV_N; i++) {
                kv[i] = (float*)malloc(sz_kv1);
                if (fread(kv[i], 1, sz_kv1, stdin) != sz_kv1) { fprintf(stderr, "K read %d fail\n", i); return 1; }
            }
            kv_dev_ok = 0;   /* 主机 KV 模式: kv[] 是 host 缓冲, S 段仍走 s2d 路径 */
            fprintf(stderr, "K host-KV mode: 36 KV latched\n");
            fwrite("OK", 1, 2, stdout); fflush(stdout);
        } else if (cmd == 'P') {
            if (fread(prefix, 1, sz_prefix, stdin) != sz_prefix) return 1;
            if (fread(p_amask, 1, sz_pam, stdin) != sz_pam) return 1;
            if (fread(p_cos, 1, sz_pcossin, stdin) != sz_pcossin) return 1;
            if (fread(p_sin, 1, sz_pcossin, stdin) != sz_pcossin) return 1;
            if (fread(f4d, 1, sz_f4d, stdin) != sz_f4d) return 1;
            if (fread(s_cos, 1, sz_scossin, stdin) != sz_scossin) return 1;
            if (fread(s_sin, 1, sz_scossin, stdin) != sz_scossin) return 1;
            if (fread(time_, 4, 1, stdin) != 1) return 1;
            struct timespec t0, t1; clock_gettime(CLOCK_MONOTONIC, &t0);
            /* dkv 链式: dkv0(层0-8: k0..k8,v0..v8,hidden) -> dkv1(层9-17: k9..k17,v9..v17)
             * 2026-09-14: dkv0 的 19 个输出全部保留在设备上 —— hidden 直接以设备指针喂给
             * dkv1(免去 7.9MB 回拷+重传), 18 个 KV 存入 kv_dev[0..17] 供 ddn 段直接引用. */
            for (int i = 0; i < KV_N; i++) { free(kv[i]); kv[i] = NULL; }
            kv_dev_ok = 0;
            bm_device_mem_t kdev0[19], kdev1[18];
            memset(kdev0, 0, sizeof(kdev0)); memset(kdev1, 0, sizeof(kdev1));
            void* ib0[4] = {prefix, p_amask, p_cos, p_sin};
            void* ob0[19]; size_t os0[19];
            if (net_forward_ex(handle, bmrt, dkv[0], ib0, NULL, ob0, os0, 1, kdev0)) { fprintf(stderr, "dkv0 fail\n"); return 1; }
            bm_device_mem_t ib1_dev[4];
            memset(ib1_dev, 0, sizeof(ib1_dev));
            ib1_dev[0] = kdev0[18];                  /* hidden: 设备直连, 零拷贝 */
            void* ib1[4] = {NULL, p_amask, p_cos, p_sin};
            void* ob1[18]; size_t os1[18];
            if (net_forward_ex(handle, bmrt, dkv[1], ib1, ib1_dev, ob1, os1, 1, kdev1)) { fprintf(stderr, "dkv1 fail\n"); return 1; }
            /* ob 交错序 [k0,v0,...]: 前 9 层=dkv0 的 0..17, 后 9 层=dkv1 的 0..17 */
            for (int i = 0; i < 18; i++) kv_dev[i] = kdev0[i];
            for (int i = 0; i < 18; i++) kv_dev[18 + i] = kdev1[i];
            kv_dev_ok = 1;
            clock_gettime(CLOCK_MONOTONIC, &t1);
            float dbgk[3] = {0.f, 0.f, 0.f};
            bm_memcpy_d2s_partial_offset(handle, dbgk, kv_dev[0], sizeof(dbgk), 0);  /* 仅 12B, 供日志核对 */
            fprintf(stderr, "KV %.1fms kv0[:3]=%.4f %.4f %.4f | am=%.1f f4=%.1f sc=%.4f tm=%.3f dev_kv=%d\n", (t1.tv_sec-t0.tv_sec)*1e3 + (t1.tv_nsec-t0.tv_nsec)/1e6, dbgk[0], dbgk[1], dbgk[2], p_amask[0], f4d[0], s_cos[0], time_[0], kv_dev_ok);
            fwrite("OK", 1, 2, stdout); fflush(stdout);
        } else if (cmd == 'S') {
            if (fread(x_t, 4, AH * 32, stdin) != AH * 32) return 1;
            if (fread(time_, 4, 1, stdin) != 1) return 1;  /* 每步 time */
            struct timespec t0, t1; clock_gettime(CLOCK_MONOTONIC, &t0);
            /* denoise 链: 段输入 = (suffix_in, time, f4d, s_cos, s_sin, 段层 KV...)
             * 输入名匹配: suffix_in/time/f4d/s_cos/s_sin + p_k{L}/p_v{L}(本段层号) */
            float* carry = x_t; size_t carry_is_vt = 0;
            float* v = NULL; size_t vb = 0;
            for (int s = 0; s < 3; s++) {
                const bm_net_info_t* net = dseg[s]->net;
                int nin = net->input_num;
                void* in_bufs[64];
                bm_device_mem_t in_dev[64];
                memset(in_dev, 0, sizeof(in_dev));
                int use_dev = 0;
                for (int i = 0; i < nin; i++) {
                    const char* nm = net->input_names[i];
                    if (strstr(nm, "suffix_in")) in_bufs[i] = carry;
                    else if (strstr(nm, "time")) in_bufs[i] = time_;
                    else if (strstr(nm, "f4d")) in_bufs[i] = f4d;
                    else if (strcmp(nm, "s_cos") == 0) in_bufs[i] = s_cos;
                    else if (strcmp(nm, "s_sin") == 0) in_bufs[i] = s_sin;
                    else if (strncmp(nm, "p_k", 3) == 0 || strncmp(nm, "p_v", 3) == 0) {
                        int li = atoi(nm + 3);
                        int idx = (nm[2] == 'k') ? li * 2 : li * 2 + 1;
                        if (kv_dev_ok) { in_dev[i] = kv_dev[idx]; use_dev = 1; }  /* 设备常驻: 零拷贝 */
                        else in_bufs[i] = kv[idx];
                    } else { fprintf(stderr, "unmatched %s\n", nm); return 1; }
                }
                void* ob[4]; size_t os[4];
                if (net_forward_ex(handle, bmrt, dseg[s], in_bufs, use_dev ? in_dev : NULL, ob, os, 0, NULL)) { fprintf(stderr, "dn%d fail\n", s); return 1; }
                if (s < 2) {
                    if (carry != x_t) free(carry);   /* 释放前段输出, 防泄漏 */
                    carry = (float*)ob[0];   /* suffix_out */
                    fprintf(stderr, "seg%d out[:3]=%.4f %.4f %.4f\n", s, carry[0], carry[1], carry[2]);
                    for (int i = 1; i < dseg[s]->net->output_num; i++) free(ob[i]);
                } else {
                    v = (float*)ob[0]; vb = os[0];  /* v_t */
                    for (int i = 1; i < dseg[s]->net->output_num; i++) free(ob[i]);
                }
                if (carry_is_vt) { /* unused */ }
            }
            clock_gettime(CLOCK_MONOTONIC, &t1);
            fprintf(stderr, "DN3 %.1fms x0=%.4f t=%.3f\n", (t1.tv_sec-t0.tv_sec)*1e3 + (t1.tv_nsec-t0.tv_nsec)/1e6, x_t[0], time_[0]);
            fwrite(v, 1, AH * 32 * 4, stdout); fflush(stdout);  /* 固定 1280B, 防 vb 对齐余量错位 */
            free(v);
            /* 2026-09-14 修复内存泄漏: 三段链里 s=1 的输出被赋给 carry 作为 s=2 的输入,
             * 但循环结束后只 free 了 v, carry 从未释放 —— 每条 S 命令泄漏一个
             * AH*1024*4=40960B 的 ddn_6_12 suffix_out。实测每步 +40KB、每次推理
             * (dn10) +400KB, 长跑后 serve_dkv 的 RSS 涨到数百 MB 并把只有 ~1GB
             * 系统 RAM 的设备推入 OOM。 */
            if (carry != x_t) free(carry);
        } else {
            fprintf(stderr, "bad cmd %c\n", cmd); return 1;
        }
    }
    bmrt_destroy(bmrt);
    bm_dev_free(handle);
    return 0;
}