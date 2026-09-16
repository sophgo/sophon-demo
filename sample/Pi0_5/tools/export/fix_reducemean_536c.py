#===----------------------------------------------------------------------===#
#
# Copyright (C) 2026 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
#!/usr/bin/env python3
"""fix_reducemean_536.py — 把 536 版 5 个 ONNX 的 ReduceMean axes 从 input 改写为 attribute
(tpu-mlir 1.28 只认 attribute 形式, 否则报 "Unsupport opset for ReduceMean")。

opset 18 下 axes 有两种来源, 都要处理:
  - dkv: axes 来自 graph.initializer
  - ddn: axes 来自 Constant 节点(dynamo=False 导出)

⚠️ Constant 节点必须**保留** —— 同一个 Constant 可能同时被 Reshape 等算子当 shape 用。
只有"被 ReduceMean 吃掉且无其他消费者"的才删。
"""
import os as _os

# All inputs and outputs live under PI05_ROOT, the working tree that holds the
# official openpi checkout (openpi-ref/), the official checkpoint (ckpt_official/)
# and the generated artifacts. Set it before running:
#     export PI05_ROOT=/path/to/pi05-work
_PI05_ROOT = _os.environ.get("PI05_ROOT")
if not _PI05_ROOT:
    raise SystemExit("PI05_ROOT is not set; see tools/export/README.md")

import numpy as np
import onnx
from onnx import helper, numpy_helper


def const_values(graph):
    out = {}
    for n in graph.node:
        if n.op_type == 'Constant' and len(n.output) == 1:
            for a in n.attribute:
                if a.name == 'value':
                    out[n.output[0]] = numpy_helper.to_array(a.t)
    return out


def fix(nm):
    d = _os.environ.get("PI05_ONNX_DIR") or f"{_PI05_ROOT}/onnx"
    pi = f'{d}/{nm}.onnx'
    po = f'{d}/{nm}_fx.onnx'
    m = onnx.load(pi, load_external_data=False)
    g = m.graph
    inits = {i.name: numpy_helper.to_array(i) for i in g.initializer
             if len(i.raw_data) and len(i.raw_data) <= 4096}
    consts = const_values(g)

    consumed = {}          # axes 输入名 -> 该 ReduceMean 节点
    new_nodes = []
    n_rm = n_fix = 0
    for n in g.node:
        if n.op_type == 'ReduceMean':
            n_rm += 1
            if len(n.input) > 1:
                src = n.input[1]
                val = inits.get(src)
                if val is None:
                    val = consts.get(src)
                if val is not None:
                    consumed[src] = n
                    kd = [helper.get_attribute_value(a) for a in n.attribute if a.name == 'keepdims'] or [1]
                    new_nodes.append(helper.make_node('ReduceMean', [n.input[0]], n.output,
                                                      axes=np.asarray(val).ravel().tolist(),
                                                      keepdims=int(kd[0])))
                    n_fix += 1
                    continue
        new_nodes.append(n)

    # 被吃掉的 axes 输入还有没有别的消费者?
    still_used = set()
    for n in new_nodes:
        if n.op_type == 'ReduceMean' and len(n.input) == 1 and id(n) in [id(x) for x in new_nodes]:
            pass
        still_used.update(n.input)
    for src, rmnode in consumed.items():
        still_used.discard(src) if False else None
    # 重新统计: 排除 ReduceMean 自己吃掉的那一份
    ref = set()
    for n in new_nodes:
        if n.op_type == 'ReduceMean' and any(
                helper.get_attribute_value(a).__class__ is list and a.name == 'axes'
                for a in n.attribute[:0]):
            pass
        ref.update(n.input)
    # 简化: 直接看除 ReduceMean 外还有谁引用
    others = set()
    for n in new_nodes:
        if n.op_type == 'ReduceMean':
            continue
        others.update(n.input)

    kept = []
    for n in new_nodes:
        if n.op_type == 'Constant' and n.output[0] in consumed and n.output[0] not in others:
            continue
        kept.append(n)
    del g.node[:]
    g.node.extend(kept)
    kept_init = [i for i in g.initializer
                 if not (i.name in consumed and i.name not in others)]
    del g.initializer[:]
    g.initializer.extend(kept_init)

    # 校验: 所有节点输入都能找到生产者
    produced = {o for n in g.node for o in n.output}
    produced |= {i.name for i in g.initializer}
    produced |= {x.name for x in g.input}
    missing = []
    for n in g.node:
        for inp in n.input:
            if inp and inp not in produced:
                missing.append((n.op_type, inp))
    onnx.save(m, po)
    print('%s: ReduceMean %d, 改写 %d, 悬空输入 %d' % (nm, n_rm, n_fix, len(missing)))
    # opset17 导出的图 axes 本就是 attribute(无第二输入), 无需改写 ——
    # 只要不存在"还带着未转换 input"的 ReduceMean 即可
    leftover = [n for n in g.node if n.op_type == 'ReduceMean' and len(n.input) > 1]
    assert not leftover, '%s 仍有带 input 的 ReduceMean %d 个' % (nm, len(leftover))
    assert not missing, '悬空输入: %s' % missing[:3]


for nm in ['pi05_dkv0_9', 'pi05_dkv9_18', 'pi05_ddn0_6', 'pi05_ddn6_12', 'pi05_ddn12_18']:
    fix(nm)
print('FIX536_DONE')
