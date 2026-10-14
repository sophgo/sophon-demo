---
name: perf-profiling
description: 性能 profiling 与优化。在 SoC/PCIe 设备上对 bmodel 推理做性能剖析，定位瓶颈（引擎占用/MAC 利用率/显存），针对性优化后验证收益。说「按照 perf-profiling skill 测 XXX 的性能」即可。
---

# 性能 Profiling

> 目的：对 bmodel 推理做端到端性能剖析，定位瓶颈是解码、预处理、推理还是后处理，再针对推理本身定位是 TPU 计算瓶颈还是访存/调度瓶颈，针对性优化并量化收益。**先 Profile 定位瓶颈，再优化——别用聚合数挥手。**

## 适用范围

- bmodel 推理性能评测（单帧耗时、FPS、吞吐量）
- TPU 利用率与瓶颈分析
- 多模型级联（如 SAM3 五 part 链）的全链路性能分解
- SoC 平台显存约束下的性能-精度权衡

## 前置条件

| 检查 | 命令 | 期望 |
|------|------|------|
| TPU 驱动 | `bm-smi` | 列出芯片，无 error |
| bmodel | `ls *.bmodel` | 模型已编译 |
| bmrt_test | `which bmrt_test` | 可用（tpu-motor 性能测试） |
| C++/Python 例程 | 编译/可运行 | 能跑推理并打印耗时 |

## 流程（5 阶段）

### 阶段 1：端到端耗时分解

1. 用例程自带的时间戳（`TimeStamp`）分解各阶段耗时。Python 例程通常打印 `decode/preprocess/inference/postprocess` 时间；C++ 例程通过 `ts->save("xxx time")` 记录。

   ```
   ================
   dfine test summary
   ================
   decode time      5.977 ms
   preprocess time  0.847 ms
   inference time   9.044 ms
   postprocess time 0.030 ms
   ```

2. **判断瓶颈阶段：**
   - 推理占比 > 60% → 阶段 2 深入 TPU 分析
   - 解码占比高 → 硬件解码调优（阶段 4）
   - 后处理占比高 → 后处理 BMCV 化 / 量化（阶段 4）
   - 预处理占比高 → BMCV 流水优化（阶段 4）

3. 多次运行取中位数（首次运行有 JIT/缓存开销，从第 2 次起取值）。

### 阶段 2：TPU 计算瓶颈分析

**工具 1：bmrt_test（最快，看引擎耗时分解）**

```bash
# 编译时打开 --opt 2 或保持默认
bmrt_test --bmodel xxx.bmodel --dev_id 0
```

输出包含：
- `Create bmodel handle success` → 加载无问题
- 子图耗时分解（每个 subnet 的计算时间）
- 总推理时间

**工具 2：TPU 利用率（bm-smi 实时监控）**

```bash
# 后台跑推理时监控
watch -n 0.5 bm-smi
```

观察：
- TPU 利用率：持续 > 80% 说明计算密集，优化空间在模型结构
- TPU 利用率低 + 显存带宽高 → 访存瓶颈
- TPU 利用率低 + 显存带宽低 → 调度/数据搬运瓶颈

**工具 3：逐层 Profile（model_deploy 带 profiling）**

```bash
# 编译 bmodel 时加 --debug，或用 npz_tool 逐层计时
model_deploy.py ... --model xxx.bmodel --debug
```

逐层 Profile 输出每层计算时间，找出耗时 Top-10 层：
- Conv/MatMul 占比高 → 计算瓶颈，考虑 INT8 量化
- Concat/Reshape/Transpose 占比高 → 调度瓶颈，考虑算子融合
- 无明显单层瓶颈 → 全局量化或结构裁剪

### 阶段 3：显存分析

**SoC 平台显存分区：**

| 平台 | Heap 结构 | 默认 TPU 可用 | 调优方法 |
|------|----------|--------------|---------|
| BM1684X SoC (SE7) | npu/vpu/vpp 三 heap | npu ~2.4G | `memory_edit.sh` 调整 npu 比例 |
| BM1684X PCIe | 统一显存 | 全部 | 无需调整 |
| BM1688 SoC (SE9) | 统一 TPU heap | ~2G | 无 memory_edit |

```bash
# SE7 查看显存
bm-smi  # 看 npu/vpu/vpp heap
# 调整 npu heap（需 root）
sudo /opt/sophon/libsophon-0.5.3/bin/memory_edit.sh -npu 6144 -vpu 2048 -vpp 2048
# 重启生效
sudo reboot
```

**显存不足处理：**
- INT8 全常驻 > 5G → 调整 npu heap 到 6G（SE7）或改用流式推理
- 多模型同时加载 → 评估各模型常驻显存之和是否超 npu heap
- 显存碎片 → 分批加载/卸载

### 阶段 4：针对性优化

| 瓶颈类型 | 优化手段 | 预期收益 |
|----------|---------|---------|
| TPU 计算密集 | INT8 量化 | 2-4x 加速（精度需验证，见 quantization-calibration skill） |
| TPU 计算密集 | W8F16（INT8 不达标时） | 1.5-2x |
| TPU 调度瓶颈 | `--opt 1`（优化搜索级别） | 减少调度开销，但不一定更快 |
| `--opt 2` 崩溃 | 切换 `--opt 1` | opt=2 的 layer-group 搜索可能产非法 shape_secs → `llvm_unreachable` |
| 显存不足 | 调整 npu heap / 流式推理 | 解除 OOM |
| 解码瓶颈 | 硬件解码（ff_decode 默认尝试 HW） | 解码从 ~10ms 降到 ~2ms |
| 后处理瓶颈 | BMCV 化（draw/resize 走 BMCV） | 减少 CPU↔TPU 数据搬运 |
| 预处理瓶颈 | BMCV 流水（convert_to + resize 合并） | 减少中间拷贝 |
| 多模型级联 | 流水并行（前一模型推理时后一模型预处理） | 逼近最慢模型耗时 |

**`--opt` 级别选择：**
- `--opt 0`：不做优化（最快编译，最慢推理）
- `--opt 1`：基础优化（稳妥，推荐默认）
- `--opt 2`：layer-group 搜索（理论最快，但部分模型触发 `LmemAllocator.cpp:1645 llvm_unreachable` → 改用 opt=1）

### 阶段 5：优化验证

1. 优化后重新跑阶段 1 的端到端耗时分解，对比优化前后：
   ```
   优化前：decode 5.9ms, preprocess 0.8ms, inference 9.0ms, postprocess 0.03ms (15.7ms)
   优化后：decode 5.8ms, preprocess 0.8ms, inference 5.2ms, postprocess 0.03ms (11.8ms)
   推理加速 1.73x，端到端加速 1.33x
   ```

2. **注意精度回归**：INT8 优化后必须做精度对齐（见 precision-alignment skill），确认无精度回退。

3. 性能数据报告需包含：
   - 优化前后各阶段耗时对比
   - TPU 利用率变化
   - 精度变化（如有）
   - 显存占用变化

## 多模型级联性能分解

对于级联模型（如 SAM3 的 ViT part0-4 + Neck + decoder），逐 part 测量：

```bash
# 逐 part 单独测
for part in part0 part1 part2 part3 part4; do
  bmrt_test --bmodel ${part}.bmodel --dev_id 0 2>&1 | grep "sub_net"
done
```

找出最慢 part（瓶颈），优先优化。全链路耗时 ≈ Σ(各 part 耗时)（无流水时）或 ≈ max(各 part)（完美流水时）。

## 常见性能数字参考（BM1684X SoC）

| 模型类型 | 模型 | F16 单帧 | INT8 单帧 |
|----------|------|---------|----------|
| 目标检测 | YOLOv5s 640 | ~8ms | ~4ms |
| 目标检测 | D-FINE 640 | ~9ms | — |
| 实例分割 | SAM3 ViT-L | ~25ms/part | ~15ms/part |
| 开词检测 | YOLO_world | ~12ms | ~6ms |
| 语音识别 | Whisper-base | ~50ms | — |

（以上为参考量级，实际取决于 batch/精度/设备状态）

## 执行纪律

- **先 Profile 再优化**：用 `bmrt_test` 和逐层 Profile 定位真实瓶颈，不要凭聚合耗时猜测。
- 多次运行取中位数，排除首次 JIT 开销和系统噪声。
- 优化后必须验证精度未回退——性能和精度是一对权衡，不能只看一边。
- 报告需含优化前后对比数据，量化收益（加速比），不要只说"快了"。
- 清理：Profile 产生的中间文件（.profile/.json）测完删除。
- 汇报用中文，结论先行。
