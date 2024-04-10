# Python例程 <!-- omit in toc -->

## 目录 <!-- omit in toc -->
- [1. 环境准备](#1-环境准备)
  - [1.1 x86/arm PCIe平台](#11-x86arm-pcie平台)
  - [1.2 SoC平台](#12-soc平台)
- [2. 推理测试](#2-推理测试)
  - [2.1 参数说明](#21-参数说明)
  - [2.2 使用方式](#22-使用方式)

python目录下提供了一系列Python例程，具体情况如下：

| 序号  |  Python例程       | 说明                            |
| ---- | ----------------  | ------------------------------- |
| 1    |    whisper.py     |         使用SAIL推理             |


## 1. 环境准备
### 1.1 x86/arm PCIe平台

如果您在x86/arm平台安装了PCIe加速卡（如SC系列加速卡），并使用它测试本例程，您需要安装libsophon、sophon-opencv、sophon-ffmpeg，具体请参考[x86-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#3-x86-pcie平台的开发和运行环境搭建)或[arm-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#5-arm-pcie平台的开发和运行环境搭建)。

此外您还需要安装其他第三方库：
```bash
pip3 install -r requirements.txt
```
您还需要安装sophon-sail，由于本例程需要的sophon-sail版本较新，相关功能还未发布，这里暂时提供一个可用的sophon-sail版本，x86/arm PCIe环境可以通过下面的命令下载：
```bash
pip3 install dfss --upgrade #安装dfss依赖

#x86 pcie, py38
python3 -m dfss --url=open@sophgo.com:sophon-demo/Whisper/sail/pcie/sophon-3.7.0-py3-none-any.whl
pip3 install sophon-3.7.0-py3-none-any.whl

#arm pcie, py38
python3 -m dfss --url=open@sophgo.com:sophon-demo/Whisper/sail/arm_pcie/sophon_arm_pcie-3.7.0-py3-none-any.whl
pip3 install sophon_arm_pcie-3.7.0-py3-none-any.whl
```
如果您需要其他版本的sophon-sail，或者遇到glibc版本问题（pcie环境常见），可以通过以下命令下载源码，参考[sophon-sail编译安装指南](https://doc.sophgo.com/sdk-docs/v23.07.01/docs_latest_release/docs/sophon-sail/docs/zh/html/1_build.html#)自己编译sophon-sail。
```bash
python3 -m dfss --url=open@sophgo.com:sophon-demo/Whisper/sail/sophon-sail_20240226.tar.gz
tar xvf sophon-sail_20240226.tar.gz
```
### 1.2 SoC平台

如果您使用SoC平台（如SE、SM系列边缘设备），并使用它测试本例程，刷机后在`/opt/sophon/`下已经预装了相应的libsophon、sophon-opencv和sophon-ffmpeg运行库包。

此外您还需要安装其他第三方库：
```bash
pip3 install -r requirements.txt
```
由于本例程需要的sophon-sail版本较新，这里提供一个可用的sophon-sail whl包，SoC环境可以通过下面的命令下载：
```bash
pip3 install dfss --upgrade
python3 -m dfss --url=open@sophgo.com:sophon-demo/Whisper/sail/soc/sophon_arm-3.7.0-py3-none-any.whl #arm soc, py38
```
如果您需要其他版本的sophon-sail，可以参考上一小节，下载源码自己编译。

## 2. 推理测试
python例程不需要编译，可以直接运行，PCIe平台和SoC平台的测试参数和运行方式是相同的。
### 2.1 参数说明

```bash
usage: whisper.py wavfile/path [--model MODEL] [--bmodel_dir BMODEL_DIR] [--dev_id DEV_ID] [--output_dir OUTPUT_DIR] [--output_format OUTPUT_FORMAT] [--verbose VERBOSE] [--task TASK] [--language LANGUAGE] [--temperature TEMPERATURE] [--best_of BEST_OF] [--beam_size BEAM_SIZE] [--patience PATIENCE] [--length_penalty LENGTH_PENALTY] [--suppress_tokens SUPPRESS_TOKENS] [--initial_prompt INITIAL_PROMPT] [--condition_on_previous_text CONDITION_ON_PREVIOUS_TEXT] [--temperature_increment_on_fallback TEMPERATURE_INCREMENT_ON_FALLBACK] [--compression_ratio_threshold COMPRESSION_RATIO_THRESHOLD] [--logprob_threshold LOGPROB_THRESHOLD] [--no_speech_threshold NO_SPEECH_THRESHOLD] [--word_timestamps WORD_TIMESTAMPS] [--prepend_punctuations PREPEND_PUNCTUATIONS] [--append_punctuations APPEND_PUNCTUATIONS] [--highlight_words HIGHLIGHT_WORDS] [--max_line_width MAX_LINE_WIDTH] [--max_line_count MAX_LINE_COUNT] [--threads THREADS] [--padding_size PADDING_SIZE] [--loop_profile LOOP_PROFILE]
--model: 选择模型尺寸，可选项为 small/base/medium。默认为 "small"。
--bmodel_dir: 用于推理的 bmodel 文件夹路径。默认为 "../models/BM1684X/"。
--dev_id: 用于推理的 TPU 设备 ID。默认为 0。
--output_dir: 模型输出的存放路径。默认为当前目录 "."。
--output_format: 模型输出的保存格式，可选项为 txt, vtt, srt, tsv, json, all。若未指定，则生成所有可用格式。默认为 "all"。
--verbose: 是否打印进度和调试信息。接受布尔值。默认为 True。
--task: 指定执行转录（'transcribe'）或翻译（'translate'）。默认为 "transcribe"。
--language: 音频中的语言。指定 None 以执行语言检测。默认为 None。可用选项取决于支持的语言。
--temperature: 用于采样的温度。默认为 0。
--best_of: 在非零温度下采样时考虑的候选数量。默认为 5。
--beam_size: 束搜索中的束（beam）数量，仅当温度为零时适用。默认为 5。
--patience: 在束解码中使用的可选耐心值。默认为 None。
--length_penalty: 使用的可选令牌长度惩罚系数。默认为 None。
--suppress_tokens: 在采样过程中要抑制的令牌 ID 的逗号分隔列表。默认为 "-1"。
--initial_prompt: 提供给第一个窗口的可选提示文本。默认为 None。
--condition_on_previous_text: 如果为 True，则为下一个窗口提供模型的前一次输出作为提示。默认为 True。
--temperature_increment_on_fallback: 在回退时增加的温度，用于解码失败。默认为 0.2。
--compression_ratio_threshold: 如果 gzip 压缩比高于此值，则将解码视为失败。默认为 2.4。
--logprob_threshold: 如果平均对数概率低于此值，则将解码视为失败。默认为 -1.0。
--no_speech_threshold: 如果 <|nospeech|> 令牌的概率高于此值且解码因 logprob_threshold 失败，则将该部分视为静默。默认为 0.6。
--word_timestamps: （实验性功能）提取单词级时间戳并根据它们优化结果。默认为 False。
--prepend_punctuations: 如果启用了 word_timestamps，则将这些标点符号与下一个单词合并。默认为 ''"'“¿([{—"'。
--append_punctuations: 如果启用了 word_timestamps，则将这些标点符号与前一个单词合并。默认为 '""'.。,，!！?？:：”)]}、'。
--highlight_words: （需要 --word_timestamps 为 True）在 srt 和 vtt 格式中为每个单词加下划线，随着它们的发音。默认为 False。
--max_line_width: （需要 --word_timestamps 为 True）在换行前一行中的最大字符数。默认为 None。
--max_line_count: （需要 --word_timestamps 为 True）一个片段中的最大行数。默认为 None。
--threads: PyTorch 在 CPU 推理中使用的线程数；取代 MKL_NUM_THREADS/OMP_NUM_THREADS。默认为 0。
--padding_size: 键值缓存的最大预分配大小。默认为 448。
--loop_profile: 是否打印循环时间以用于性能分析。默认为 False。
```

### 2.2 使用方式
测试单个语音文件
```bash
python3 whisper.py ../datasets/test/demo.wav --model base --bmodel_dir ../models/BM1684X --dev_id 0  --output_dir ./result/ --output_format txt
```

测试语音数据集
```bash
python3 whisper.py ../datasets/aishell_S0764/ --model base --bmodel_dir ../models/BM1684X --dev_id 0  --output_dir ./result/ --output_format txt
```
