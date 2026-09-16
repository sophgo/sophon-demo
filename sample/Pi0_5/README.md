# Pi0_5

## 目录
- [Pi0\_5](#pi0_5)
  - [目录](#目录)
  - [1. 简介](#1-简介)
  - [2. 特性](#2-特性)
  - [3. 运行环境准备](#3-运行环境准备)
  - [4. 准备模型](#4-准备模型)
    - [4.1 使用提供的模型](#41-使用提供的模型)
    - [4.2 自行编译BModel模型](#42-自行编译bmodel模型)
  - [5. 例程测试](#5-例程测试)
  - [6. 精度测试](#6-精度测试)
  - [7. 程序性能测试](#7-程序性能测试)

## 1. 简介
π0.5 是 Physical Intelligence 开源的视觉-语言-动作（Vision-Language-Action, VLA）模型，约 3.3B 参数，输入图像与语言指令、输出连续机器人动作轨迹。模型可见[π0.5](https://www.physicalintelligence.company/research)。

本例程对 π0.5（pi05_libero）进行移植，使其可在 Sophon BM1684X（SE7 系列）SoC 上运行。主干为 PaliGemma（SigLIP So400m/14 视觉编码器 + Gemma-2B 双专家），动作头为 flow-matching 连续动作专家，**按迭代去噪求解**（本 sample 默认 2 步），输出 10 步动作 chunk。整条链路拆成 6 个子模型，KV 常驻设备内存，因此不是"一张图进、一个结果出"的单次前向，而是**多子模型串联 + 迭代循环 + 有状态 KV 复用**。

在 SoC 上运行需要额外进行环境配置，请参照[运行环境准备](#3-运行环境准备)完成环境部署。

## 2. 特性

* 支持BM1684X（SoC，SE7系列）
* 支持W8BF16 + BF16混合精度
* 支持单次推理的C++例程，只依赖libsophon
* 支持常驻推理服务与LIBERO闭环评测（进阶）
* 支持两路相机输入，接口按官方三路图像槽预留（详见[FAQ](./FAQ.md#12-那要接真第三路相机怎么办)）
* 6个子模型合计约3.45GB，主干KV常驻设备内存
* 前缀536token：完整前缀968token中有436个被注意力掩码整段屏蔽

## 3. 运行环境准备

本例程只支持 **BM1684X 的 SoC 模式**（SE7 系列）。

对于 SE7 设备，需要把 NPU 堆预留调大：6 个 bmodel 合计约 3.45 GB，且需要
TPU/VPU/VPP 三个 heap 同时可用（合计约 13 GB）。默认 ION 预留下的 Linux 可用内存
可能只有约 1 GB，bmodel 会加载失败。

```bash
cd /data/
mkdir memedit && cd memedit
wget -nd https://github.com/sophgo/sophon-tools/releases/download/v24.09.21/memory_edit_v2.10.tar.xz
tar xvf memory_edit_v2.10.tar.xz
cd memory_edit
./memory_edit.sh -p                      # 打印当前内存布局
./memory_edit.sh -c -npu 7615 -vpu 2048 -vpp 2048
sudo cp output/emmcboot.itb /boot/emmcboot.itb && sync
sudo reboot
```

运行环境要求：

| 组件 | 版本 | 说明 |
|---|---|---|
| libsophon | **0.5.3**（SDK 26.03.01） | 必须与编译 bmodel 用的 TPU-MLIR 版本匹配，否则 bmodel 加载失败 |
| g++ | **9.x** | SE7 自带，可在设备上直接原生编译 |
| numpy | 任意 | 仅 `tools/compare_acc.py` 需要 |

> **注意：**
> 1. 交叉编译可行，但**宿主交叉工具链必须与设备运行时匹配**，否则编出来的二进制在设备上起不来，详见[FAQ 4.1](./FAQ.md#41-交叉编译出来的二进制为什么在设备上跑不起来)。
> 2. 推荐直接在设备上原生编译 —— SE7 自带 g++ 9.x 与 libsophon，不需要交叉工具链。

## 4. 准备模型

模型目前只提供 BM1684X（SoC）编译好的 bmodel。数据集（观测样本、初始噪声与官方动作真值）
随模型一起提供，用于精度测试。

### 4.1 使用提供的模型

本例程在`scripts`目录下提供了相关模型和数据的下载脚本：

```bash
└── scripts
    └──download.sh                                        # 通过该脚本下载Pi0_5的BModel与数据集
```

> **注意：**
> 1. 下载bmodel和数据集之前，应该保证存储空间大于8G（bmodel文件约3.45G，解压后另需约3.45G）

```bash
chmod -R +x scripts/
./scripts/download.sh bm1684x      # 提供了all|bm1684x
```

执行下载脚本后，目录结构如下：

```bash
├── models
|   └── BM1684X
|       ├── pi05_siglip_w8bf16_2b.bmodel        # 视觉编码器（SigLIP，2路batch）
|       ├── pi05_dkv0_9_w8bf16_1b.bmodel        # 主干KV前向 第0-8层
|       ├── pi05_dkv9_18_w8bf16_1b.bmodel       # 主干KV前向 第9-17层
|       ├── pi05_ddn0_6_bf16_1b.bmodel          # 动作专家去噪 第0-5层
|       ├── pi05_ddn6_12_bf16_1b.bmodel         # 动作专家去噪 第6-11层
|       └── pi05_ddn12_18_bf16_1b.bmodel        # 动作专家去噪 第12-17层+动作头
└── datasets
    └── pi05_libero_sample
        ├── obs/tXX_initY/{agentview.npy, wrist.npy}   # 观测：uint8 [224,224,3] RGB
        ├── noise/tXX_initY.npy                        # 初始噪声：float32 [10,32]
        ├── actions_gt/tXX_initY.npy                   # 官方动作真值：float32 [10,7]
        ├── prefix_assets/                             # 逐任务前缀张量
        ├── action_unnorm.npz                          # 反归一化系数
        └── index.json
```

### 4.2 自行编译BModel模型

从官方权重到 bmodel 的**完整流程**（JAX→PyTorch 转换、ONNX 导出、分段切分、图修补、
TPU-MLIR 编译）见[导出指南](./docs/Pi0_5_Export_Guide.md)，导出脚本在
[`tools/export/`](./tools/export/README.md)。这里只给编译命令：

```bash
cd scripts
./gen_siglip_bmodel_mlir.sh bm1684x
./gen_dkv_bmodel_mlir.sh bm1684x
./gen_ddn_bmodel_mlir.sh bm1684x
```

> **注意：**
> 1. 编译需要 [TPU-MLIR](https://github.com/sophgo/tpu-mlir) docker 环境，本例程使用 `tpu_mlir_dev:1.28`。
> 2. 编译脚本消费的 ONNX 放在 `models/onnx/`：siglip 用导出产物 `pi05_siglip.onnx`，
>    其余 5 段用**图修补后**的 `pi05_*_fx.onnx`。

## 5. 例程测试

- [C++例程](./cpp/pi05_bmcv/README.md) —— 单次推理（观测 → 动作 chunk）
- [进阶服务与LIBERO环境部署](./cpp/pi05_service/README.md) —— 常驻推理服务 + 闭环评测环境

在设备上运行（首次运行会自动编译）：

```bash
cd sample/Pi0_5
./scripts/run_demo.sh                 # 用task 0、2步去噪跑一条观测

# 换任务 / 改步数 / 计次测速
./scripts/run_demo.sh -t 3 -n 2
./scripts/run_demo.sh -l 20
```

`run_demo.sh` 会逐项检查工具链、6 个 bmodel、数据集与观测文件，缺哪一项就打印出对应的
修复命令，而不是跑到一半才失败。跑完会打印精度对照命令。

一键跑完整测试（bmrt_test 理论延迟 + 全数据集精度对照）：

```bash
./scripts/auto_test.sh -m soc_test -t BM1684X -d 0
```

## 6. 精度测试

π0.5 是策略模型，**不用 Top-1/mAP 类指标**，按同类工作的一致做法报 cosine similarity
与闭环任务成功率。

### 6.1 数值级（单次推理例程）

与官方参考实现的 cosine similarity，门限 **≥ 0.999（最低可接受 0.99）**。

对比口径：参考实现为官方 PyTorch 权重 + openpi 自带模型代码（**fp32**，与 ONNX 导出同精度）；
两边使用**同一份观测、同一份噪声**（`noise/tXX_initY.npy`，设备端 `--noise` 下发）、
同样的 2 步去噪，输出都取 unnorm 空间前 7 维。

| 测试平台 | 精度档 | 用例数 | Overall cos (mean / min / max) | per-timestep cos (mean / min / max) | rel L2 |
|---|---|---|---|---|---|
| SE7-32 | W8BF16 + BF16 | 50 | **0.999978** / 0.999960 / 0.999990 | 0.999983 / 0.999759 / 1.000000 | 0.689% |

> **测试说明：**
> 1. 50条全部达标，最差一条 cos 0.999960 仍在 0.999 门限之上；
> 2. 50条 = LIBERO-Spatial 10个任务 × 5个初始状态，固定 seed 采集；
> 3. per-timestep 指动作 chunk 里 10 个时间步各自的 cos（不是去噪步）；
> 4. 逐条结果可复跑：`./scripts/auto_test.sh -m soc_test` 产出 `scripts/acc.txt`。

### 6.2 任务级（闭环，需 LIBERO 环境）

| 测试平台 | 精度档 | 配置 | 任务成功率 | 95% Wilson CI |
|---|---|---|---|---|
| SE7-32 | W8BF16 + BF16 | LIBERO-Spatial 10 task × 2 init = 20 episode，replan=5 / dn=2 / seed7 | **20/20 = 100%** | 83.9% – 100% |

> **测试说明：**
> 1. 跑法见[进阶服务部署说明](./cpp/pi05_service/README.md#4-跑闭环评测)；
> 2. 20 episode 是抽样不是结论 —— Wilson 下界只有 83.9%，要下"成功率达标"的结论需要增加 episode 数；
> 3. 单次推理中位 445 ms，20 case 墙钟 472 s（EGL 硬渲染）。

## 7. 程序性能测试

### 7.1 bmrt_test

```bash
bmrt_test --bmodel models/BM1684X/pi05_dkv0_9_w8bf16_1b.bmodel
```

| 测试平台 | 子模型 | calculate time (ms) |
|---|---|---|
| SE7-32 | pi05_siglip_w8bf16_2b | **56.9** |
| SE7-32 | pi05_dkv0_9_w8bf16_1b | **155.7** |
| SE7-32 | pi05_dkv9_18_w8bf16_1b | **139.5** |
| SE7-32 | pi05_ddn0_6_bf16_1b | **8.5** |
| SE7-32 | pi05_ddn6_12_bf16_1b | **8.3** |
| SE7-32 | pi05_ddn12_18_bf16_1b | **8.4** |

### 7.2 程序运行性能

测试命令：

```bash
./scripts/run_demo.sh -l 20
```

单次推理 = 1 次 siglip + 1 次 dkv（两段）+ N 步去噪（每步 3 段 ddn）：

| 测试平台 | 测试程序 | 去噪步数 | preprocess (ms) | siglip (ms) | dkv (ms) | ddn (ms) | 整chunk合计 (ms) | 每动作 (ms) |
|---|---|---|---|---|---|---|---|---|
| SE7-32 | pi05_bmcv.soc | 2 | 6.8 | 68.2 | 312.5 | 58.0 | **445.6** | **89** |

> **测试说明：**
> 1. 延迟为**不含前后处理**的纯推理延迟，用C/C++ API测量；
> 2. 「每动作」= 整chunk延迟 ÷ replan步数，本sample默认chunk=10、replan=5；
> 3. 表中的 `dkv` 比 bmrt_test 两段之和（295.2 ms）多约 17 ms，是 host↔device 张量搬运与段间同步的开销。

### 7.3 内存占用

| 测试平台 | bmodel合计 | 进程峰值RSS | NPU堆预留 |
|---|---|---|---|
| SE7-32 | 3.45 GB | **20.9 MB** | 需三heap（`BMRUNTIME_NEURON_HEAP_MASK=7`） |

> **测试说明：**
> 1. 主机侧只放前缀张量与段间中间结果，权重常驻 NPU 侧，不占主机内存；
> 2. 进程峰值RSS为单次推理进程的 `/proc/<pid>/status` 中 `VmHWM`。
