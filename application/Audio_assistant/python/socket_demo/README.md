# Audio assistant websocket应用例程 <!-- omit in toc -->

## 目录 <!-- omit in toc -->
- [1. Server端环境准备](#1-Server端环境准备)
  - [1.1 x86/arm PCIe平台](#11-x86arm-pcie平台)
  - [1.2 SoC平台](#12-soc平台)
- [2. Client端环境准备](#2-Client端环境准备)
  - [2.1 x86/arm PCIe、SoC平台](#21-x86/arm-pcie、soc平台)
- [3. Server端准备模型](#3-Server端准备模型)
- [4. Client端准备数据](#4-Client端准备数据)
- [5. 推理测试](#5-推理测试)
  - [5.1 参数说明](#51-参数说明)
  - [5.2 使用方式](#52-使用方式)
- [6. 程序运行性能](#6-程序运行性能)
- [7. 流程图](#7-流程图)

websocket_demo目录下提供了一系列Python例程，具体情况如下：

| 序号  |             Python例程                    |             说明                |
| ---- | ----------------------------------------  | ------------------------------- |
| 1    |    service/server.py                      |         服务端代码               |
| 2    |    client/client.py                       |         客户端代码               |


## 1. Server端环境准备

### 1.1 x86/arm PCIe平台

如果您在x86/arm平台安装了PCIe加速卡（如SC系列加速卡），并使用它测试本例程，您需要安装libsophon、sophon-opencv、sophon-ffmpeg和sophon-sail，具体请参考[x86-pcie平台的开发和运行环境搭建](../../../../docs/Environment_Install_Guide.md#3-x86-pcie平台的开发和运行环境搭建)或[arm-pcie平台的开发和运行环境搭建](../../../../docs/Environment_Install_Guide.md#5-arm-pcie平台的开发和运行环境搭建)。

- Python >= 3.8.2环境，SDK >= v23.09。

- 如果您使用Llama3作为LLM，则需要执行如下步骤进行编译
```bash
sudo apt-get install pybind11-dev
# 编译库文件
cd ../Llama3/python_demo
python3 -m dfss --url=open@sophgo.com:sophon-demo/application/Audio_assistant/llama3/support.zip
unzip support.zip -d ../
rm -f support.zip
mkdir build
cd build && cmake -DTARGET_ARCH=pcie .. && make && mv *cpython* ../..
cd ../../../socket_demo
```

- 接着需要下载Whisper的依赖文件
```bash
cd ../whisper-TPU_py/bmwhisper
python3 -m dfss --url=open@sophgo.com:sophon-demo/application/Audio_assistant/whisper-TPU_py/third_party.zip
unzip third_party.zip -d ./
rm -f third_party.zip
python3 -m dfss --url=open@sophgo.com:sophon-demo/application/Audio_assistant/whisper-TPU_py/assets.zip
unzip assets.zip -d ./
rm -f assets.zip
cd ../../socket_demo
```

- 此外您还需要安装其他python第三方库：
```bash
sudo apt install portaudio19-dev
sudo apt-get install libsndfile1
pip3 install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cpu
pip3 install -r service/server_requirements.txt
```

### 1.2 SoC平台
如果您使用SoC平台（如SE、SM系列边缘设备），并使用它测试本例程，刷机后在`/opt/sophon/`下已经预装了相应的libsophon、sophon-opencv和sophon-ffmpeg运行库包。

- SDK >= v23.09

- 如果您使用Llama3作为LLM，则需要在SOC平台执行如下步骤进行编译
```bash
sudo apt-get install pybind11-dev
# 编译库文件
cd ../Llama3/python_demo
python3 -m dfss --url=open@sophgo.com:sophon-demo/application/Audio_assistant/llama3/support.zip
unzip support.zip -d ../
rm -f support.zip
mkdir build
cd build && cmake -DTARGET_ARCH=soc .. && make && mv *cpython* ../..
cd ../../../socket_demo
```

- 如果您使用MiniCPM作为LLM，则需要在SOC平台执行如下步骤进行编译
```bash
# 编译库文件
cd ../MiniCPM/demo
python3 -m dfss --url=open@sophgo.com:sophon-demo/application/Audio_assistant/minicpm/support.zip
unzip support.zip -d ../
rm -f support.zip
mkdir build
cd build && cmake -DTARGET_ARCH=soc .. && make && mv minicpm ..
cd ../../../socket_demo
```

- 接着需要下载Whisper的依赖文件
```bash
cd ../whisper-TPU_py/bmwhisper
python3 -m dfss --url=open@sophgo.com:sophon-demo/application/Audio_assistant/whisper-TPU_py/third_party.zip
unzip third_party.zip -d ./
rm -f third_party.zip
python3 -m dfss --url=open@sophgo.com:sophon-demo/application/Audio_assistant/whisper-TPU_py/assets.zip
unzip assets.zip -d ./
rm -f assets.zip
cd ../../socket_demo
```

- 此外您还需要在SOC平台安装其他python第三方库：
```bash
pip3 install dfss -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade
# 对于SE9平台
python3 -m dfss --url=open@sophgo.com:sophon-demo/application/Audio_assistant/sophon_arm-3.8.0-py3-none-any.whl
pip3 install sophon_arm-3.8.0-py3-none-any.whl --force-reinstall
rm -f sophon_arm-3.8.0-py3-none-any.whl
# 对于SE7平台
python3 -m dfss --url=open@sophgo.com:sophon-demo/application/Audio_assistant/84x_soc_sail/sophon-3.8.0-py3-none-any.whl
pip3 install sophon-3.8.0-py3-none-any.whl --force-reinstall
rm -f sophon-3.8.0-py3-none-any.whl
# 对于SE7、SE9平台
sudo apt install portaudio19-dev
sudo apt install libsndfile1
pip3 install networkx==2.8.8
pip3 install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cpu
pip3 install requests==2.26.0
pip3 install -r service/server_requirements.txt
```

- 对于SE9平台，运行前需要下载额外动态库，然后设置环境变量
```bash
python3 -m dfss --url=open@sophgo.com:sophon-demo/application/Audio_assistant/libfirmware_core.so
export BMRUNTIME_USING_FIRMWARE=/path-to-current-dir/libfirmware_core.so
```

## 2. Client端环境准备

### 2.1 x86/arm PCIe、SoC平台

您只需要执行如下命令安装其他第三方库：
```bash
sudo apt install portaudio19-dev
pip3 install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cpu
pip3 install -r client/client_requirements.txt
```

## 3. Server端准备模型
目前只支持在BM1684X和BM1688上运行，已提供编译好的BModel。

​本例程在`scripts`目录下提供了相关模型的下载脚本
```bash
└── scripts
    ├── download_bm1684x_whisper_llama3_vits.sh                                      # 通过该脚本下载BM1684X平台的Whisper/Llama3/VITS BModels
    └── download_bm1688_whisper_minicpm_vits.sh                                      # 通过该脚本下载BM1688平台的Whisper/MiniCPM/VITS BModels
```

- 对于BM1688平台，执行如下命令下载模型
```bash
# 安装unzip，若已安装请跳过
sudo apt install unzip
cd ../..
chmod -R +x scripts/
./scripts/download_bm1688_whisper_minicpm_vits.sh
cd python/socket_demo
```

下载的模型包括：
```
./models
└── BM1688
    ├── minicpm
    |   ├── minicpm-2b_int4_1core.bmodel                                                                 # MiniCPM-2B int4，1core BM1688 BModel
    |   └── tokenizer.model                                                                              # MiniCPM-2B 的Tokenizer模型
    ├── vits
    |   ├── bert_1688_f32_1core.bmodel                                                                   # VITS BERT fp32，1core BM1688 BModel
    |   └── vits_chinese_128_bm1688_f16_1core.bmodel                                                     # VITS fp16，1core BM1688 BModel
    └── whisper
        ├── all_quant_decoder_loop_with_kvcache_and_rearrange_base_5beam_128pad_bm1688_f16.bmodel        # Whisper-Base Decoder模型，beam size为5，输出最大token长度为128，fp16 BM1688 BModel
        ├── all_quant_decoder_loop_with_kvcache_and_rearrange_small_5beam_128pad_bm1688_f16.bmodel       # Whisper-Small Decoder模型，beam size为5，输出最大token长度为128，fp16 BM1688 BModel
        ├── all_quant_decoder_loop_with_kvcache_and_rearrange_small_5beam_448pad_bm1688_f16.bmodel       # Whisper-Small Decoder模型，beam size为5，输出最大token长度为448，效率相比128模型更低，fp16 BM1688 BModel
        ├── all_quant_decoder_main_with_kvcache_base_5beam_128pad_bm1688_f16.bmodel                      # Whisper-Base Decoder模型，beam size为5，输出最大token长度为128，fp16 BM1688 BModel
        ├── all_quant_decoder_main_with_kvcache_small_5beam_128pad_bm1688_f16.bmodel                     # Whisper-Small Decoder模型，beam size为5，输出最大token长度为128，fp16 BM1688 BModel
        ├── all_quant_decoder_main_with_kvcache_small_5beam_448pad_bm1688_f16.bmodel                     # Whisper-Small Decoder模型，beam size为5，输出最大token长度为448，效率相比128模型更低，fp16 BM1688 BModel
        ├── all_quant_decoder_post_base_5beam_128pad_bm1688_f16.bmodel                                   # Whisper-Base Decoder模型，beam size为5，fp16 BM1688 BModel
        ├── all_quant_decoder_post_small_5beam_128pad_bm1688_f16.bmodel                                  # Whisper-Small Decoder模型，beam size为5，fp16 BM1688 BModel
        ├── all_quant_decoder_post_small_5beam_448pad_bm1688_f16.bmodel                                  # Whisper-Small Decoder模型，beam size为5，fp16 BM1688 BModel
        ├── all_quant_encoder_base_5beam_128pad_bm1688_f16.bmodel                                        # Whisper-Base Encoder模型，beam size为5，fp16 BM1688 BModel
        ├── all_quant_encoder_small_5beam_128pad_bm1688_f16.bmodel                                       # Whisper-Small Encoder模型，beam size为5，fp16 BM1688 BModel
        ├── all_quant_encoder_small_5beam_448pad_bm1688_f16.bmodel                                       # Whisper-Small Encoder模型，beam size为5，fp16 BM1688 BModel
        ├── all_quant_logits_decoder_base_5beam_128pad_bm1688_f16.bmodel                                 # Whisper-Base 预测层，beam size为5，fp16 BM1688 BModel
        ├── all_quant_logits_decoder_small_5beam_128pad_bm1688_f16.bmodel                                # Whisper-Small 预测层，beam size为5，fp16 BM1688 BModel
        └── all_quant_logits_decoder_small_5beam_448pad_bm1688_f16.bmodel                                # Whisper-Small 预测层，beam size为5，fp16 BM1688 BModel
```

- 对于BM1684X平台，执行如下命令下载模型
```bash
# 安装unzip，若已安装请跳过
sudo apt install unzip
cd ..
chmod -R +x scripts/
./scripts/download_bm1684x_whisper_llama3_vits.sh
cd python/socket_demo
```

下载的模型包括：
```
./models
└── BM1684X
    ├── llama3
    |   ├── token_config                                                                                 # Llama3-8B Tokenizer配置文件 
    |   └── llama3-8b_int4_1dev_256.bmodel                                                               # Llama3-8B 最大输出长度为256个tokens，int4 Bmodel
    ├── vits
    |   ├── bert_1684x_f32.bmodel                                                                        # VITS BERT fp32 BModel
    |   └── vits_chinese_128_f16.bmodel                                                                  # VITS fp16 BModel
    └── whisper
        ├── all_quant_decoder_loop_with_kvcache_and_rearrange_base_5beam_448pad_1684x_f16.bmodel         # Whisper-Base Decoder模型，beam size为5，输出最大token长度为448，fp16 BM1684X BModel
        ├── all_quant_decoder_loop_with_kvcache_and_rearrange_large-v2_5beam_448pad_1684x_f16.bmodel     # Whisper-Large-v2 Decoder模型，beam size为5，输出最大token长度为448，fp16 BM1684X BModel
        ├── all_quant_decoder_loop_with_kvcache_and_rearrange_large-v3_5beam_448pad_1684x_f16.bmodel     # Whisper-Large-v3 Decoder模型，beam size为5，输出最大token长度为448，fp16 BM1684X BModel
        ├── all_quant_decoder_loop_with_kvcache_and_rearrange_medium_5beam_448pad_1684x_f16.bmodel       # Whisper-Medium Decoder模型，beam size为5，输出最大token长度为448，fp16 BM1684X BModel
        ├── all_quant_decoder_loop_with_kvcache_and_rearrange_small_5beam_448pad_1684x_f16.bmodel        # Whisper-Small Decoder模型，beam size为5，输出最大token长度为448，fp16 BM1684X BModel
        ├── all_quant_decoder_main_with_kvcache_base_5beam_448pad_1684x_f16.bmodel                       # Whisper-Base Decoder模型，beam size为5，输出最大token长度为448，fp16 BM1684X BModel
        ├── all_quant_decoder_main_with_kvcache_medium_5beam_448pad_1684x_f16.bmodel                     # Whisper-Medium Decoder模型，beam size为5，输出最大token长度为448，fp16 BM1684X BModel
        ├── all_quant_decoder_main_with_kvcache_small_5beam_448pad_1684x_f16.bmodel                      # Whisper-Small Decoder模型，beam size为5，输出最大token长度为448，fp16 BM1684X BModel
        ├── all_quant_decoder_post_base_5beam_448pad_1684x_f16.bmodel                                    # Whisper-Base Decoder模型，beam size为5，fp16 BM1684X BModel
        ├── all_quant_decoder_post_medium_5beam_448pad_1684x_f16.bmodel                                  # Whisper-Medium Decoder模型，beam size为5，fp16 BM1684X BModel
        ├── all_quant_decoder_post_small_5beam_448pad_1684x_f16.bmodel                                   # Whisper-Small Decoder模型，beam size为5，fp16 BM1684X BModel
        ├── all_quant_encoder_base_5beam_448pad_1684x_f16.bmodel                                         # Whisper-Base Encoder模型，beam size为5，fp16 BM1684X BModel
        ├── all_quant_encoder_medium_5beam_448pad_1684x_f16.bmodel                                       # Whisper-Medium Encoder模型，beam size为5，fp16 BM1688 BModel
        ├── all_quant_encoder_small_5beam_448pad_1684x_f16.bmodel                                        # Whisper-Small Encoder模型，beam size为5，fp16 BM1684X BModel
        ├── all_quant_logits_decoder_base_5beam_448pad_1684x_f16.bmodel                                  # Whisper-Base 预测层，beam size为5，fp16 BM1684X BModel
        ├── all_quant_logits_decoder_medium_5beam_448pad_1684x_f16.bmodel                                # Whisper-Medium 预测层，beam size为5，fp16 BM1684X BModel
        └── all_quant_logits_decoder_small_5beam_448pad_1684x_f16.bmodel                                 # Whisper-Small 预测层，beam size为5，fp16 BM1684X BModel
```

## 4. Client端准备数据
已提供测试数据，​同时，您可以自行准备用于测试的数据集。

​本例程在`scripts`目录下提供了相关数据的下载脚本`download_datasets.sh`

```bash
# 安装unzip，若已安装请跳过
sudo apt install unzip
cd ../../
chmod -R +x scripts/
# 在客户端执行
./scripts/download_datasets.sh
cd python/socket_demo
```

下载的数据包括：
```
./datasets
└── ai_zh.wav                                 # 测试用音频文件，包含中文问题“什么是人工智能？”
```

## 5. 推理测试
python例程不需要编译，可以直接运行，PCIe平台和SoC平台的测试参数和运行方式是相同的。
### 5.1 参数说明

服务端参数说明如下：
```bash
usage: server.py [-h] [--host HOST] [--port PORT] [--profile] [--streaming_output] [--llm_type LLM_TYPE] [--min_tts_input_len MIN_TTS_INPUT_LEN]
--host: 主机的IP，默认本机
--port: 服务的端口
--streaming_output: 是否使用流式输出，即并行LLM和TTS两个模块
--llm_type: LLM的类型, 目前支持minicpm-2b, llama3-8b
--min_tts_input_len: 最小的TTS输入文本的长度，默认为6，可减小长度以减小延时。
--bmodel_dir: Whisper模型文件夹路径
--chip: 指定Whisper运行的平台，支持1684x、bm1688
--padding_size: 指定Whisper所使用模型的padding尺寸，可从模型的文件名获得
--chip_mode: 指定平台的模式，支持pcie、soc
--vits_model: VITS模型路径
--bert_model: VITS所使用的BERT模型路径
```

客户端参数说明如下：
```bash
usage: client.py [-h] [--host HOST] [--port PORT] [--chunk_duration_ms CHUNK_DURATION_MS] [--vad_level VAD_LEVEL] [--vad_type VAD_TYPE] [--audio_in AUDIO_IN] [--audio_fs AUDIO_FS]
                     [--microphone_dev_id MICROPHONE_DEV_ID] [--output_file]
--host: 服务端的IP，默认为本机
--port: 服务端端口
--chunk_duration_ms: VAD一次性输入的音频长度，单位为毫秒，默认为200
--vad_level: webrtcvad的粒度，支持[0,3], 值越大粒度越细
--vad_type: VAD模型的类型，支持webrtcvad、fsmn-vad
--audio_in: 输入音频文件，如果未给出则使用麦克风输入
--audio_fs: 程序使用的音频sample rate，包括输出音频，默认16000
--microphone_devid: 麦克风设备id
--audio_devid: 喇叭设备id
--output_file: 是否输出音频文件，如果未给出则输出到喇叭
```

### 5.2 使用方式

### 5.2.1 启动服务器
若使用非流式输出，则需要等待回答完全结束才输出音频数据到客户端
```bash
# 对于BM1688 SOC平台可以使用如下命令
python3 service/server.py --port 10095 --bmodel_dir=../../BM1688/whisper
# 对于BM1684X SOC平台可以使用如下命令
python3 service/server.py --bmodel_dir ../../BM1684X/whisper --vits_model ../../BM1684X/vits/vits_chinese_128_f16.bmodel --bert_model ../../BM1684X/vits/bert_1684x_f32.bmodel --llm_type llama3-8b --chip 1684x --padding_size 448 --chip_mode soc
# 对于BM1684X PCIE平台可以使用如下命令
python3 service/server.py --bmodel_dir ../../BM1684X/whisper --vits_model ../../BM1684X/vits/vits_chinese_128_f16.bmodel --bert_model ../../BM1684X/vits/bert_1684x_f32.bmodel --llm_type llama3-8b --chip 1684x --padding_size 448 --chip_mode pcie
```

若使用流式输出，则边回答边输出音频数据到客户端，会有更好的实时音频输出效果
```bash
# 对于BM1688 SOC平台可以使用如下命令
python3 service/server.py --port 10095 --bmodel_dir=../../BM1688/whisper --streaming_output
# 对于BM1684X SOC平台可以使用如下命令
python3 service/server.py --bmodel_dir ../../BM1684X/whisper --vits_model ../../BM1684X/vits/vits_chinese_128_f16.bmodel --bert_model ../../BM1684X/vits/bert_1684x_f32.bmodel --llm_type llama3-8b --chip 1684x --padding_size 448 --chip_mode soc --streaming_output
# 对于BM1684X PCIE平台可以使用如下命令
python3 service/server.py --bmodel_dir ../../BM1684X/whisper --vits_model ../../BM1684X/vits/vits_chinese_128_f16.bmodel --bert_model ../../BM1684X/vits/bert_1684x_f32.bmodel --llm_type llama3-8b --chip 1684x --padding_size 448 --chip_mode pcie --streaming_output
```

### 5.2.2 启动客户端
若使用音频文件作为输入输出，则使用如下命令，根据实际情况修改`--host`参数为服务器IP地址，`--port`参数修改为服务器端口
```bash
python3 client/client.py --host 127.0.0.1 --port 10095 --audio_in ../../datasets/ai_zh.wav --output_file
```

若使用麦克风、喇叭作为输入输出，使用如下命令，需根据实际情况修改麦克风设备id，`--host`参数修改为服务器IP地址，`--port`参数修改为服务器端口
```bash
python3 client/client.py --host 127.0.0.1 --port 10095 --microphone_devid 0
```

若使用文件作为输入、喇叭作为输出，使用如下命令，根据实际情况修改`--host`参数为服务器IP地址，`--port`参数修改为服务器端口
```bash
python3 client/client.py --host 127.0.0.1 --port 10095 --audio_in ../../datasets/ai_zh.wav 
```

> **注意**
> 1. 如果没有麦克风或喇叭设备，请使用文件的形式
> 2. 若使用麦克风作为输入，当终端输出`microphone running ...`，就可以通过麦克风说出问题，等待回答即可。

## 6 程序运行性能
在不同的测试平台上，测试`../datasets/ai_zh.wav`音频文件，使用5.2.1节中流式输出、5.2.2节中文件作为输入喇叭作为输出的命令进行测试，性能测试结果如下：
|    服务端测试平台  |  客户端测试平台  |  Latency  | 
| ----------------- | --------------- | --------- |
|   SE7-32          |    SE7-32       |    8.5    |
|   SE9-16          |    SE9-16       |    9.3    |

> **测试说明**：  
> 1. 性能Latency指标是指问题说完到开始输出回答音频的时间（包括网络传输时间），单位为秒(s)。
> 2. 不同PCIE平台有差异，以实际性能为准。
> 3. 性能结果受LLM输出的第一段话长度、参数`--min_tts_input_len`、网络的影响，前两者长度越长延时越高，实际性能可以通过实际数据测试得到。
> 4. 对于SE7，不同SDK版本性能可能存在较大差异，以实测为准。

## 7. 流程图
`socket_demo`中的处理流程，遵循以下流程图：

![flowchart](../../pics/assistant2.png)