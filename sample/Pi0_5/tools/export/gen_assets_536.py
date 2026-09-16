#===----------------------------------------------------------------------===#
#
# Copyright (C) 2026 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
#!/usr/bin/env python3
"""gen_assets_536.py — 生成 PL=536 的通用 per-task 资产(支持存活数 <536 的 task).

每个 task 的存活数 n 在 528~533 之间(512 图 + L 个语言, L=16~21), 全部 <= 536。
补位槽用 p_amask/f4d 整段屏蔽, 取值不影响输出(§2 已证)。
"""
import os as _os

# All inputs and outputs live under PI05_ROOT, the working tree that holds the
# official openpi checkout (openpi-ref/), the official checkpoint (ckpt_official/)
# and the generated artifacts. Set it before running:
#     export PI05_ROOT=/path/to/pi05-work
_PI05_ROOT = _os.environ.get("PI05_ROOT")
if not _PI05_ROOT:
    raise SystemExit("PI05_ROOT is not set; see tools/export/README.md")

import os
import numpy as np

PL = 536
SRC = _os.environ.get("PI05_ASSETS_IN", f"{_PI05_ROOT}/assets/dkva_npy")
DST = _os.environ.get("PI05_ASSETS_OUT", f"{_PI05_ROOT}/assets/dkva536_npy")
os.makedirs(DST, exist_ok=True)
skipped, done = [], []
for t in range(10):
    m = np.load(f'{SRC}/t{t:02d}_p_amask.npy').reshape(968, 968)
    a = np.where(~((m == -10000).all(axis=0)))[0]
    n = len(a)
    assert (a[:512] == np.arange(512)).all(), t
    assert (a[512:] == np.arange(768, 768 + n - 512)).all(), t
    if n > PL:
        skipped.append((t, n)); continue
    # p_amask: 有效块全 0(原掩码的存活子块本就全可见), 补位行/列全 -10000
    am = np.full((1, 1, PL, PL), -10000.0, np.float32)
    am[0, 0, :n, :n] = 0.0
    np.save(f'{DST}/t{t:02d}_p_amask.npy', am)
    # p_cos/p_sin: 有效行取原 alive 行(已验证 == rope(0..n-1)), 补位行任意(被屏蔽)
    for nm in ('p_cos', 'p_sin'):
        v = np.load(f'{SRC}/t{t:02d}_{nm}.npy')[0][a]      # (n,256)
        out = np.zeros((1, PL, 256), np.float32); out[0, :n] = v
        np.save(f'{DST}/t{t:02d}_{nm}.npy', out)
    # f4d: 列 = [n 个有效 prefix][PL-n 个屏蔽][10 个 suffix]
    f4 = np.load(f'{SRC}/t{t:02d}_f4d.npy').reshape(1, 1, 10, 978)
    pre = f4[:, :, :, :968][:, :, :, a]                    # (1,1,10,n) 全 0
    pad = np.full((1, 1, 10, PL - n), -10000.0, np.float32)
    suf = f4[:, :, :, 968:]
    np.save(f'{DST}/t{t:02d}_f4d.npy', np.concatenate([pre, pad, suf], axis=3))
    # s_cos/s_sin: 原样(其位置本就按 n 紧凑算)
    for nm in ('s_cos', 's_sin'):
        np.save(f'{DST}/t{t:02d}_{nm}.npy', np.load(f'{SRC}/t{t:02d}_{nm}.npy'))
    # 语言 token: 取前 L 个
    p200 = np.load(f'{SRC}/prompt200_t{t:02d}.npy').reshape(200, 2048)
    np.save(f'{DST}/promptL_t{t:02d}.npy', np.ascontiguousarray(p200[:n - 512]))
    done.append((t, n))
print('已生成:', done)
print('跳过(存活>%d):' % PL, skipped)
np.save(f'{DST}/alive.npy', np.load(f'{SRC}/t00_p_amask.npy').reshape(968,968)[0] if False else
        np.where(~((np.load(f'{SRC}/t00_p_amask.npy').reshape(968,968) == -10000).all(axis=0)))[0])
print('GEN532B_DONE')
