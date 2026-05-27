# Python例程

## 目录

- [Python例程](#python例程)
  - [目录](#目录)
  - [1. 环境准备](#1-环境准备)
    - [1.1 x86/arm/riscv PCIe平台](#11-x86armriscv-pcie平台)
    - [1.2 SoC平台](#12-soc平台)
  - [2. 推理测试](#2-推理测试)
    - [2.1 参数说明](#21-参数说明)
    - [2.2 使用方式](#22-使用方式)
    - [2.3 测试音频](#23-测试音频)

python目录下提供了一系列Python例程，具体情况如下：

| 序号 |  Python例程     | 说明                |
| ---- | ------------- | -------------------  |
| 1    | silero_vad.py | 使用SAIL推理 |

## 1. 环境准备
### 1.1 x86/arm/riscv PCIe平台

如果您在x86/arm/riscv平台安装了PCIe加速卡（如SC系列加速卡），并使用它测试本例程，您需要安装libsophon、sophon-sail，具体请参考[x86-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#3-x86-pcie平台的开发和运行环境搭建)或[arm-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#5-arm-pcie平台的开发和运行环境搭建)或[riscv-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#6-riscv-pcie平台的开发和运行环境搭建)。

此外您可能还需要安装其他第三方库：
```bash
pip3 install soundfile numpy
# soundfile 用于读取音频文件，若无法安装可改用标准库 wave
```

### 1.2 SoC平台

如果您使用SoC平台（如SE、SM系列边缘设备），并使用它测试本例程，刷机后在`/opt/sophon/`下已经预装了相应的libsophon运行库包。您还需要交叉编译安装sophon-sail，具体可参考[交叉编译安装sophon-sail](../../../docs/Environment_Install_Guide.md#42-交叉编译安装sophon-sail)。

此外您可能还需要安装其他第三方库：
```bash
pip3 install soundfile numpy
```

## 2. 推理测试
python例程不需要编译，可以直接运行，PCIe平台和SoC平台的测试参数和运行方式是相同的。

### 2.1 参数说明
```bash
usage: silero_vad.py [--bmodel BMODEL] [--input INPUT] [--dev_id DEV_ID]
                     [--threshold THRESHOLD] [--min_speech_duration_ms MS]
                     [--min_silence_duration_ms MS] [--speech_pad_ms MS]

--bmodel: 用于推理的bmodel路径，默认使用 models/BM1684X/silero_vad_bm1684x_f16.bmodel；
--input: 输入WAV音频文件路径（16kHz采样率），必填；
--dev_id: 用于推理的tpu设备id，默认为0；
--threshold: 语音概率阈值(0.0~1.0)，大于此值判定为语音，默认为0.5；
--min_speech_duration_ms: 最小语音段时长(ms)，小于此值的语音段被丢弃，默认为250；
--min_silence_duration_ms: 最小静音段时长(ms)，小于此值的静音会被合并，默认为100；
--speech_pad_ms: 语音段边界扩展(ms)，默认为30；
--save_segments: 是否将检测到的语音段保存为独立WAV文件。启用后结果保存在 results/segments/ 目录下；
```

### 2.2 使用方式
模型加载使用SAIL的`sail.Engine`类，采用`SYSIO`模式（numpy数组输入输出）。每帧处理流程为：
1. 从音频流中取512个采样点+64个历史上下文拼接为[1, 576]输入
2. 逐帧送入TPU推理，每帧得到语音概率
3. 将概率序列通过阈值平滑处理转为语音段起止时间

```python
import sophon.sail as sail

# 加载模型
net = sail.Engine(bmodel_path, dev_id, sail.IOMode.SYSIO)
graph_name = net.get_graph_names()[0]

# 逐帧推理
inputs = {'x': x_frame, 'h': h_state, 'c': c_state}
outputs = net.process(graph_name, inputs)
prob = outputs['out']        # 语音概率 [1, 1]
h_state = outputs['h_new']   # LSTM隐藏状态 [1, 128]
c_state = outputs['c_new']   # LSTM细胞状态 [1, 128]
```

### 2.3 测试音频
WAV音频测试实例如下：
```bash
cd python
python3 silero_vad.py \
    --bmodel ../models/BM1684X/silero_vad_bm1684x_f16.bmodel \
    --input ../datasets/test.wav \
    --dev_id 0
```
测试结束后，输出VAD结果，内容包括：
- 检测到的语音段数量和每个段落的起止时间
- 每帧的预处理/推理/后处理平均耗时
- 实时因子（real_time_factor）

如需同时保存检测到的语音段为独立WAV文件，可加上`--save_segments`参数：
```bash
python3 silero_vad.py --input ../datasets/test.wav --save_segments
```

语音段输出示例：
```
INFO:root:Frames: 1875, speech segments: 19
INFO:root:  seg 0:    0.00s →    2.05s (2.04s)
INFO:root:  seg 1:    2.63s →    4.70s (2.07s)
...
```

结果会保存为JSON文件至`./results/`目录。