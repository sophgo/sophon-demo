# Python例程

- [Python例程](#python例程)
  - [1. 环境准备](#1-环境准备)
    - [1.1 x86/arm PCIe平台](#11-x86arm-pcie平台)
    - [1.2 SoC平台](#12-soc平台)
  - [2. 推理测试](#2-推理测试)
    - [2.1 参数说明](#21-参数说明)
    - [2.2 测试音频](#22-测试音频)
    - [2.3 批量测试](#23-批量测试)
  - [3. 性能测试](#3-性能测试)

python目录下提供了一系列Python例程，具体情况如下：

| 序号 | Python例程           | 说明                         |
| ---- | ------------------- | ---------------------------- |
| 1    | funasr_nano_infer.py | 端到端语音识别（ONNX/TPU/验证）  |
| 2    | utils/              | 工具模块（参考 WeNet 结构）     |

## 1. 环境准备

### 1.1 x86/arm PCIe平台

如果您在x86/arm平台安装了PCIe加速卡（如SC系列加速卡），并使用它测试本例程，您需要安装libsophon、sophon-sail，具体请参考[x86-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#3-x86-pcie平台的开发和运行环境搭建)或[arm-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#5-arm-pcie平台的开发和运行环境搭建)。

此外您还需要安装其他第三方库：

```bash
pip3 install torch torchaudio -i https://pypi.tuna.tsinghua.edu.cn/simple
pip3 install funasr transformers -i https://pypi.tuna.tsinghua.edu.cn/simple
pip3 install numpy -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 1.2 SoC平台

如果您使用SoC平台（如SE、SM系列边缘设备），并使用它测试本例程，刷机后在`/opt/sophon/`下已经预装了相应的libsophon运行库包。您还需要交叉编译安装sophon-sail，具体可参考[交叉编译安装sophon-sail](../../../docs/Environment_Install_Guide.md#42-交叉编译安装sophon-sail)。

此外您还需要安装其他第三方库：

```bash
pip3 install funasr numpy -i https://pypi.tuna.tsinghua.edu.cn/simple
```

> ⚠️ SoC平台内存有限（如SE9仅3.3GB），加载完整 PyTorch 模型可能 OOM。建议在 x86 主机上运行 LLM 解码部分。

## 2. 推理测试

python例程不需要编译，可以直接运行，PCIe平台和SoC平台的测试参数和运行方式是相同的。

### 2.1 参数说明

```bash
usage: funasr_nano_infer.py [-h] --input INPUT [--encoder ENCODER]
                            [--adapter ADAPTER] [--backend {onnx,tpu}]
                            [--dev_id DEV_ID] [--verify] [--benchmark]
                            [--loops LOOPS]
--input:     输入 WAV 文件路径 (16kHz 单声道)；
--encoder:   编码器 bmodel/onnx 路径，默认 models/BM1688/funasr_encoder_f16_1b.bmodel；
--adapter:   适配器 bmodel/onnx 路径，默认 models/BM1688/funasr_adapter_f16_1b.bmodel；
--backend:   推理后端，可选 onnx 或 tpu，默认 onnx；
--dev_id:    TPU 设备 ID，默认 0；
--verify:    ONNX vs PyTorch 精度验证模式；
--benchmark: 编码器+适配器性能基准测试；
--loops:     基准测试循环次数，默认 10。
```

### 2.2 测试音频

```bash
python3 python/funasr_nano_infer.py \
    --input datasets/aishell_S0764/BAC009S0764W0121.wav \
    --backend tpu
```

测试结束后，会打印识别文本和分阶段耗时：
- `encode`: TPU 编码器+适配器耗时
- `llm`: LLM 解码耗时
- `total`: 端到端总耗时

ONNX 精度验证：

```bash
python3 python/funasr_nano_infer.py \
    --input datasets/aishell_S0764/BAC009S0764W0121.wav \
    --verify
```

### 2.3 批量测试

```bash
for wav in datasets/aishell_S0764/*.wav; do
    echo "Processing: $(basename $wav)"
    python3 python/funasr_nano_infer.py \
        --input "$wav" --backend tpu
done
```

## 3. 性能测试

编码器+适配器基准测试：

```bash
python3 python/funasr_nano_infer.py \
    --input datasets/aishell_S0764/BAC009S0764W0121.wav \
    --backend tpu \
    --benchmark \
    --loops 10
```

BM1688 性能参考：

| 阶段 | 精度 | 耗时 |
|------|------|------|
| Encoder | F16 | ~692 ms |
| Adapter | F16 | ~34 ms |
| LLM Decode (CPU) | — | ~7 s |
| 端到端 | — | ~7.7 s |
