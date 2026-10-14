---
name: precision-alignment
description: 精度对齐验证。在 SoC/PCIe 设备上对比 C++ 例程与 Python 参考实现的检测结果，确认 bbox/score/category 完全对齐。说「按照 precision-alignment skill 对齐 XXX 的 C++ 和 Python」即可。
---

# 精度对齐验证

> 目的：将 C++ 例程的推理结果与 Python 参考实现逐图对比，量化 bbox/score/category 差异，判定是否对齐通过。输出结构化对比报告。

## 适用场景

- C++ 例程移植完成后，需验证与 Python 参考的精度一致性
- 修改预处理/后处理后，需确认未引入精度回退
- bmodel 精度切换（F16→INT8）后，需确认 C++ 仍与 Python 对齐

## 前置条件

| 检查 | 命令 | 期望 |
|------|------|------|
| Python 参考可跑 | `python3 python/xxx_bmcv.py --input=... --bmodel=...` | 生成 JSON 结果文件 |
| C++ 例程已编译 | `ls xxx_bmcv.soc` 或 `ls xxx_bmcv.pcie` | 可执行文件存在 |
| 同一 bmodel + 同一测试集 | 两端用完全相同的模型和输入图片 | 消除模型/数据差异变量 |

## 流程（4 阶段）

### 阶段 1：准备对齐环境

1. 确认 C++ 和 Python 使用**同一个 bmodel 文件**和**同一组测试图片**。
2. 确认置信度阈值一致（C++ `--conf_thresh` vs Python `thrh`，默认通常 0.4 或 0.25）。
3. 确认 NMS 策略一致（有些模型如 D-FINE 内部解码无需 NMS，有些如 YOLOv5 需要）。
4. 确认类别映射一致（COCO 80 类的 category_id 编号方案，是否含 background=0）。

### 阶段 2：分别运行 + 收集 JSON

1. 运行 Python 参考实现，生成 JSON 结果文件：
   ```bash
   python3 python/xxx_bmcv.py --input=datasets/test --bmodel=models/BM1684X/xxx_f16_1b.bmodel --dev_id=0
   # 结果通常在 python/results/xxx_bmcv_python_result.json
   ```

2. 运行 C++ 例程，生成 JSON 结果文件：
   ```bash
   ./xxx_bmcv.soc --input=datasets/test --bmodel=models/BM1684X/xxx_f16_1b.bmodel --dev_id=0
   # 结果通常在 results/xxx_test_bmcv_cpp_result.json
   ```

3. 确认两个 JSON 文件都存在且非空。

### 阶段 3：逐图对比

用 Python 脚本对比两个 JSON，逐图逐框比较：

```python
import json

with open('python_result.json') as f:
    py = json.load(f)
with open('cpp_result.json') as f:
    cpp = json.load(f)

py_map = {r['image_name']: r['bboxes'] for r in py}
cpp_map = {r['image_name']: r['bboxes'] for r in cpp}

for img in sorted(set(py_map.keys()) | set(cpp_map.keys())):
    pb = sorted(py_map.get(img, []), key=lambda x: -x['score'])
    cb = sorted(cpp_map.get(img, []), key=lambda x: -x['score'])
    print(f'{img}: Python={len(pb)}, C++={len(cb)}')
    for i in range(max(len(pb), len(cb))):
        if i < len(pb) and i < len(cb):
            bbox_diff = [abs(pb[i]['bbox'][j]-cb[i]['bbox'][j]) for j in range(4)]
            score_diff = abs(pb[i]['score'] - cb[i]['score'])
            cat_match = pb[i]['category_id'] == cb[i]['category_id']
            print(f'  [{i}] cat={"OK" if cat_match else "MISMATCH"} '
                  f'score_diff={score_diff:.6f} bbox_diff={bbox_diff}')
```

### 阶段 4：判定 + 报告

**对齐通过判据：**

| 指标 | 容差 | 说明 |
|------|------|------|
| 检测框数量 | 完全一致 | 每张图 C++ 和 Python 检出的框数相同 |
| category_id | 完全一致 | 每个对应框的类别 ID 相同 |
| score 差异 | < 1e-4 | F16 精度下浮点误差正常范围 |
| bbox 坐标差异 | ≤ 2 px | 整数截断导致的 1px 差异可接受 |

**未对齐时的排查路径：**

| 症状 | 可能原因 | 排查方法 |
|------|----------|----------|
| 框数不一致 | 阈值不同 / NMS 参数不同 | 确认 conf_thresh、iou_thresh 两端一致 |
| category_id 不一致 | 类别偏移（background=0 vs 0-indexed） | 检查 C++ class_id 映射逻辑 |
| score 差异大 | 预处理不一致（mean/scale/归一化） | 对比 C++ converto_attr 的 alpha/beta 与 Python 的 ab 数组 |
| bbox 差异大 | letterbox 参数不一致 / 坐标映射错误 | 对比 ratio、pad_w、pad_h 计算逻辑 |
| 某些图对齐某些不对齐 | 特定分辨率对齐问题 | 检查 stride 对齐（64-byte alignment）、odd-width 图片 |

## 常见预处理对齐检查点

| 项目 | Python | C++ (BMCV) |
|------|--------|------------|
| 归一化 | `ab = [x/255 for x in [1,0,1,0,1,0]]` | `converto_attr.alpha = input_scale/255, beta = 0` |
| mean | `ab = [scale, -mean*scale, ...]` | `alpha = input_scale*scale, beta = -mean*input_scale*scale` |
| letterbox | `r=min(net_w/w, net_h/h)`, 居中黑边 | `bmcv_image_vpp_convert_padding` + `padding_attr` |
| 颜色空间 | `convert_format(FORMAT_RGB_PLANAR)` | 同左 |
| resize 插值 | `BMCV_INTER_LINEAR` | 同左 |

## 报告格式

```
## <SAMPLE> 精度对齐报告

结论：✅ 对齐通过 / ❌ 未对齐

对比条件：bmodel=xxx_f16_1b.bmodel, 测试集=datasets/test (N 张图), conf_thresh=0.4

| 图片 | Python 框数 | C++ 框数 | 框数一致 | max score_diff | max bbox_diff |
|------|------------|---------|---------|---------------|--------------|
| 001.jpg | 5 | 5 | ✅ | 0.000003 | 0 |
| 002.jpg | 12 | 12 | ✅ | 0.000005 | 1 |

汇总：所有图片框数一致 ✅，所有 category_id 一致 ✅，
max score_diff = 5e-6 (< 1e-4 ✅)，max bbox_diff = 1px (≤ 2px ✅)
```

## 执行纪律

- 两端必须用同一 bmodel、同一测试集、同一阈值——这是对齐前提，不满足则对比无意义。
- 对比脚本按 score 降序排列后逐框对应，假设两端排序一致。若框数一致但对应关系错乱，改用 IoU 匹配。
- 清理：对比完成后删除临时 JSON 和输出图，保留对比报告。
- 汇报用中文，结论先行。
