# Python例程

## 目录

* [1. 环境准备](#1-环境准备)
    * [1.1 x86/arm/riscv PCIe平台](#11-x86armriscv-pcie平台)
    * [1.2 SoC平台](#12-soc平台)
    * [1.3 SE9 内存布局调整](#13-se9-内存布局调整)
* [2. 推理测试](#2-推理测试)
    * [2.1 参数说明](#21-参数说明)
    * [2.2 视频跟踪](#22-视频跟踪)

python目录下提供 `tapnext_infer.py`，使用 SAIL 对 TAPNext++ 两图循环架构进行逐帧点跟踪推理。

| 序号 | Python例程        | 说明                                          |
| ---- | ----------------- | --------------------------------------------- |
| 1    | tapnext_infer.py  | OpenCV 解码+前处理，SAIL SYSIO 推理 init/step 双图 |

## 1. 环境准备

### 1.1 x86/arm/riscv PCIe平台

如果您在x86/arm/riscv平台安装了PCIe加速卡（如SC系列加速卡），并使用它测试本例程，您需要安装libsophon、sophon-opencv、sophon-ffmpeg和sophon-sail，具体请参考[x86-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#3-x86-pcie平台的开发和运行环境搭建)或[arm-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#5-arm-pcie平台的开发和运行环境搭建)或[riscv-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#6-riscv-pcie平台的开发和运行环境搭建)。

此外您可能还需要安装其他第三方库：
```bash
pip3 install opencv-python-headless
```

### 1.2 SoC平台

如果您使用SoC平台（如SE、SM系列边缘设备），并使用它测试本例程，刷机后在`/opt/sophon/`下已经预装了相应的libsophon、sophon-opencv和sophon-ffmpeg运行库包。您还需要交叉编译安装sophon-sail，具体可参考[交叉编译安装sophon-sail](../../../docs/Environment_Install_Guide.md#42-交叉编译安装sophon-sail)。

此外您可能还需要安装其他第三方库：
```bash
pip3 install opencv-python-headless
```

> **注:** 运行前需设置环境变量：
> ```bash
> export LD_LIBRARY_PATH=/opt/sophon/libsophon-current/lib:$LD_LIBRARY_PATH
> ```

### 1.3 SE9 内存布局调整

SE9 SoC 的总物理内存固定，但 CPU 与 TPU 之间的划分可调。`free` 看到的是 CPU 内存，`bm-smi` 看到的是 TPU 内存（`tpu总内存 = npu + vpu + vpp`），两者独立。

TAPNext++ 的 bmodel 较小（init 363 MB / step 310 MB），TPU 侧内存需求不大；但 SAIL 加载 bmodel 时会在 CPU 侧分配与 bmodel 等大的瞬时读缓冲，叠加 24 个循环 cache 张量（~144 MB）和 Python 开销，默认布局下 CPU 内存（~850 MB）偏紧，加载 step 图时可能触发 OOM（exit 137）。此时需要**减小 npu、把内存让给 CPU**。

用 `memory_edit.sh`（sophon-tools 提供）调整布局，改完需 reboot 生效：

```bash
cd /data/
mkdir memedit && cd memedit
wget -nd https://github.com/sophgo/sophon-tools/releases/download/v24.09.21/memory_edit_v2.10.tar.xz
tar xvf memory_edit_v2.10.tar.xz
cd memory_edit
./memory_edit.sh -p                         # 打印当前内存布局，确认总内存和 SE9 变体

# SE9-16 / SE9-8 8G 版本：npu 2048 MB 足够放权重+激活，CPU 得到 ~6 GB
./memory_edit.sh -c -npu 2048 -vpu 0 -vpp 40
sudo cp /data/memedit/DeviceMemoryModificationKit/memory_edit/boot.itb /boot/boot.itb && sync
sudo reboot

# SE9-8 4G 版本：npu 1024 MB，CPU 得到 ~3 GB
./memory_edit.sh -c -npu 1024 -vpu 0 -vpp 40
sudo cp /data/memedit/DeviceMemoryModificationKit/memory_edit/boot.itb /boot/boot.itb && sync
sudo reboot
```

> **说明:**
> 1. npu 需保留放得下最大 bmodel 权重（~363 MB）+ 激活（~300 MB），TAPNext++ 给 1024 MB 即可，无需像 LLM 那样给 5 GB+。
> 2. 改完 reboot 后用 `free` 确认 CPU 内存已增大，再用 `bm-smi` 确认 TPU 内存仍够用。
> 3. 更多教程请参考[SoC内存修改工具](https://doc.sophgo.com/sdk-docs/v23.07.01/docs_latest_release/docs/SophonSDK_doc/zh/html/appendix/2_mem_edit_tools.html)。

## 2. 推理测试

python例程不需要编译，可以直接运行，PCIe平台和SoC平台的测试参数和运行方式是相同的。

TAPNext++ 采用两图循环架构：`init` 图处理第 0 帧并初始化 24 个循环 cache 张量，`step` 图处理后续帧并将 cache 回馈。SE9 SoC 默认 CPU 内存仅 ~850 MB，脚本通过子进程隔离 init 图（`--init_only`）保证 bmodel 系数内存彻底回收，并将 144 MB cache 以 lazy npz 延迟加载，避免 init/step 两图同时驻留。若仍 OOM，请先按 [1.3 节](#13-se9-内存布局调整) 调整内存布局增大 CPU 内存。

### 2.1 参数说明

```bash
usage: tapnext_infer.py [--input INPUT] [--init_bmodel INIT_BMODEL] [--step_bmodel STEP_BMODEL]
                        [--dev_id DEV_ID] [--query QUERY] [--query_file QUERY_FILE]
                        [--max_frames MAX_FRAMES] [--output_dir OUTPUT_DIR] [--visualize]
--input:       输入视频文件或图像目录路径；
--init_bmodel: init 图 bmodel 路径（第 0 帧）；
--step_bmodel: step 图 bmodel 路径（后续帧）；
--dev_id:      TPU 设备 id；
--query:       查询点 "y1,x1;y2,x2"，原图像素坐标（t=0，自动缩放到模型尺寸）；
--query_file:  查询点 JSON 文件 [[t,y,x],...]，模型像素坐标（不缩放）；
--max_frames:  最大处理帧数，0 = 全部；
--output_dir:  输出目录；
--visualize:   输出跟踪可视化视频。
```

### 2.2 视频跟踪

```bash
# FP16：跟踪原图 (y=50, x=100) 处的点，处理 10 帧
python3 python/tapnext_infer.py \
    --input datasets/test.mp4 \
    --init_bmodel ../models/BM1688/tapnext_init_fp16_1b.bmodel \
    --step_bmodel ../models/BM1688/tapnext_step_fp16_1b.bmodel \
    --query 50,100 --max_frames 10 \
    --output_dir results/fp16 --visualize

```

测试结束后，跟踪结果保存在 `output_dir/tracks.json`（逐帧 `{y, x, visible}`）和 `tracks.npz`，同时打印各阶段耗时。

> **坐标系说明:** 模型 track 输出为 `[y, x]`（行、列），范围为 [0, 255.5] 模型像素。查询点格式为 `[t, y, x]`。`--query` 给的是原图像素坐标，脚本会按视频分辨率自动缩放到 256×256 模型像素。
>
> **精度说明:** **生产精度使用 FP16。** FP16 跟踪结果正确（查询点 (50,100) → 跟踪 (50.06, 99.75)），且在 BM1688 上比 INT8 更快。INT8 经 6 种变体充分验证均不可用：
>
> | 精度 | 第 0 帧跟踪 (y,x) | 速度 (ms/frame) | 模型大小 (init/step) | 结论 |
> | --- | --- | --- | --- | --- |
> | **FP16** | **(50.06, 99.75)** | **1496** | **542/488 MB** | **✅ 正确，生产精度** |
> | 纯 INT8 | (1.0, 1.0) | 1694 | 363/310 MB | ❌ 输出崩塌 |
> | INT8-mix v1（预测头 F16，53 op） | (244.75, 146.62) | — | 364/310 MB | ❌ 散射 garbage |
> | INT8-mix v2（非 MatMul 全 F16，1235 op） | (245.75, 181.62) | — | 381/327 MB | ❌ 散射 garbage |
> | INT8 + per-channel | (1.0, 1.0) | 1694 | 365/311 MB | ❌ 输出崩塌 |
> | INT8-mix + per-channel | (179.75, 208.62) | 1356 | 365/311 MB | ❌ 散射 garbage |
> | W8F16 | — | — | — | ❌ 固件 assert 崩溃 |
>
> **根因分析:** TAPNext++ 的核心是 ViT self-attention + RG-LRU/Conv1D 状态空间模型，对 MatMul 的 INT8 量化极度敏感。即使将 67% 算子保留 F16（mix v2，仅 174 个 MatMul 为 INT8），attention 模式仍被破坏导致 garbage 输出。per-channel 量化（transformer INT8 标准修复）亦无法缓解。同时 INT8 在此模型上**无速度优势**——纯 INT8 反而比 FP16 慢（1694 vs 1496 ms/frame），因 backbone 中大量 element-wise RG-LRU 门控/Conv1D 算子不受益于 INT8，且量化/反量化开销抵消 MatMul 加速。综上 FP16 在精度、速度、模型大小三轴均优于 INT8，选为生产精度。

### 2.3 性能测试

在 SE9（BM1688）上对 FP16 生产精度进行性能测试，测试视频 30 帧、Q=1 查询点。

**理论 TPU 推理时间**（`bmrt_test`，随机输入，不含前后处理，loopnum=10 取均值）：

| 图 | 1-core (ms) | 2-core (ms) | 2-core 加速比 |
| --- | --- | --- | --- |
| init | 590.4 | 376.0 | 1.57× |
| step | 615.7 | 398.5 | 1.55× |

**端到端性能**（真实视频，含 OpenCV 解码 + 前处理 + SAIL 推理 + 后处理，30 帧）：

| 核数 | init (ms) | step (ms) | 总均摊 (ms/frame) | 稳态吞吐 (FPS) |
| --- | --- | --- | --- | --- |
| 1-core | 4381 | 910 | 1106 | 1.10 |
| 2-core | 4282 | 685 | 889 | 1.46 |

> **性能说明:**
> 1. **init 图**仅在首帧运行一次，4381 ms 中 TPU 推理仅 590 ms，其余为 SAIL 引擎创建 + 542 MB bmodel 加载 + 24 个 cache 输出提取。init 在子进程中运行以避免 init/step 双图同时驻留导致 OOM。
> 2. **step 图**是稳态逐帧推理，端到端 910 ms（1-core）/ 685 ms（2-core），其中 TPU 推理 616 ms / 399 ms，差额 ~294 ms 为 SAIL 27 输入张量灌入 + 26 输出张量拷贝开销。
> 3. 前后处理耗时可忽略（preprocess 2.4 ms、postprocess 0.03 ms），瓶颈在 TPU 推理。
> 4. 2-core 模式 TPU 加速 1.55×，端到端 step 加速 1.33×（SAIL I/O 开销不随核数缩减）。
> 5. 该模型 244M 参数、12 层 Transformer+SSM 循环架构，逐帧串行推理，1.46 FPS（2-core）符合此类模型在边缘 SoC 上的预期。
