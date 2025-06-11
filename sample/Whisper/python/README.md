# Python例程 <!-- omit in toc -->

## 目录 <!-- omit in toc -->
- [1. 环境准备](#1-环境准备)
  - [1.1 x86 PCIe平台](#11-x86-pcie平台)
- [2. 推理测试](#2-推理测试)
  - [2.1 参数说明](#21-参数说明)
  - [2.2 使用方式](#22-使用方式)

python目录下提供了一系列Python例程，具体情况如下：

| 序号  |  Python例程       | 说明                            |
| ---- | ----------------  | ------------------------------- |
| 1    |    whisper.py     |         使用SAIL推理             |


## 1. 环境准备
### 1.1 x86 PCIe平台

目前仅支持在x86 PCIe平台测试本例程。除了安装tpuv7-driver、tpuv7-runtime和sophon-sail之外，您还需要配置其他第三方库：

```bash
pip3 install -r requirements.txt
```

## 2. 推理测试
python例程不需要编译，可以直接运行，PCIe平台和SoC平台的测试参数和运行方式是相同的。

### 2.1 参数说明

```bash
usage: whisper.py wavfile/path [--model MODEL] [--bmodel_dir BMODEL_DIR] [--dev_id DEV_ID] [--num_core NUM_CORE] [--output_dir OUTPUT_DIR] [--output_format OUTPUT_FORMAT] [--verbose VERBOSE] [--task TASK] [--language LANGUAGE] [--temperature TEMPERATURE] [--best_of BEST_OF] [--beam_size BEAM_SIZE] [--patience PATIENCE] [--length_penalty LENGTH_PENALTY] [--suppress_tokens SUPPRESS_TOKENS] [--initial_prompt INITIAL_PROMPT] [--condition_on_previous_text CONDITION_ON_PREVIOUS_TEXT] [--temperature_increment_on_fallback TEMPERATURE_INCREMENT_ON_FALLBACK] [--compression_ratio_threshold COMPRESSION_RATIO_THRESHOLD] [--logprob_threshold LOGPROB_THRESHOLD] [--no_speech_threshold NO_SPEECH_THRESHOLD] [--word_timestamps WORD_TIMESTAMPS] [--prepend_punctuations PREPEND_PUNCTUATIONS] [--append_punctuations APPEND_PUNCTUATIONS] [--highlight_words HIGHLIGHT_WORDS] [--max_line_width MAX_LINE_WIDTH] [--max_line_count MAX_LINE_COUNT] [--threads THREADS] [--padding_size PADDING_SIZE] [--loop_profile LOOP_PROFILE]
--model: 选择模型尺寸，可选项为 small/base/medium。默认为 "small"。
--bmodel_dir: 用于推理的 bmodel 文件夹路径。默认为 "../models/BM1690/"。
--dev_id: 用于推理的 TPU 设备 ID。默认为 0。
--num_core: 模型的核数，默认为8。
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
python3 whisper.py ../datasets/test/demo.wav --model base --bmodel_dir ../models/BM1690 --dev_id 0 --num_core 8 --output_dir ./result/ --output_format txt
```

测试语音数据集
```bash
python3 whisper.py ../datasets/aishell_S0764/ --model base --bmodel_dir ../models/BM1690 --dev_id 0 --num_core 8 --output_dir ./result/ --output_format txt
```
