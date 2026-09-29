#!/usr/bin/env python3
"""在真机上跑单个子图（指定精度），把输出存成 npz 供跨设备比对。

用法（设备侧）：
    python3 subgraph_probe.py <models_dir> <subgraph> <precision> <probes.npz> <out.npz>

subgraph: prior_init | stage_1 | stage_2 | stage_3 | tail | encoder
precision: bf16 | fp32

输出按子图输出张量的顺序存成 out_0, out_1, ...，不做任何解释。
一致性判据是「两台机器同一子图同一精度的输出逐位相同」，
因此这里只负责忠实落盘，比较放在主机侧做。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bmrt import BmrtEngine  # noqa: E402

# 子图 -> (bmodel 名前缀, bmodel 输入名, 探针 npz 里的键名)
FEEDS = {
    "prior_init": ("glc_prior_init", ["z_hat"], ["z_hat"]),
    "stage_1": ("glc_stage_1", ["y_hat_so_far", "common"],
                ["stage1_y", "stage1_common"]),
    "stage_2": ("glc_stage_2", ["y_hat_so_far", "common"],
                ["stage2_y", "stage2_common"]),
    "stage_3": ("glc_stage_3", ["y_hat_so_far", "common"],
                ["stage3_y", "stage3_common"]),
    "tail": ("glc_tail", ["y_hat", "q_dec"], ["tail_y", "tail_q_dec"]),
    "encoder": ("glc_encoder", ["x", "q_enc"], ["x", "q_enc"]),
}


def main():
    if len(sys.argv) != 6:
        print(__doc__)
        return 2
    models, sub, prec, probe_path, out_path = sys.argv[1:6]
    if sub not in FEEDS:
        print("未知子图 %s，可选：%s" % (sub, ",".join(FEEDS)))
        return 2
    name, feed_names, probe_keys = FEEDS[sub]

    d = np.load(probe_path)
    feeds = {}
    for fn, pk in zip(feed_names, probe_keys):
        if pk not in d.files:
            raise SystemExit("探针文件缺 %s（有 %s）" % (pk, d.files))
        feeds[fn] = np.ascontiguousarray(d[pk], dtype=np.float32)

    bmodel = os.path.join(models, "%s_%s.bmodel" % (name, prec))
    if not os.path.isfile(bmodel):
        raise SystemExit("缺少 %s" % bmodel)
    engine = BmrtEngine(bmodel)
    try:
        out = engine.infer(feeds)
    finally:
        engine.close()

    ks = list(out.keys())
    np.savez(out_path, **{"out_%d" % i: np.ascontiguousarray(out[k], dtype=np.float32)
                          for i, k in enumerate(ks)})
    print("%s/%s -> %s  张量 %s" % (name, prec, out_path, ks))
    return 0


if __name__ == "__main__":
    sys.exit(main())
