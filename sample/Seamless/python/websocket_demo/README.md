# Seamless websocket应用例程 <!-- omit in toc -->

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

websocket_demo目录下提供了一系列Python例程，具体情况如下：

| 序号  |             Python例程                    |             说明                |
| ---- | ----------------------------------------  | ------------------------------- |
| 1    |    service/wss_server.py                  |         服务端代码               |
| 2    |    client/wss_client.py                   |         客户端代码               |


## 1. Server端环境准备

### 1.1 x86/arm PCIe平台

如果您在x86/arm平台安装了PCIe加速卡（如SC系列加速卡），并使用它测试本例程，您需要安装libsophon、sophon-opencv、sophon-ffmpeg，具体请参考[x86-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#3-x86-pcie平台的开发和运行环境搭建)或[arm-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#5-arm-pcie平台的开发和运行环境搭建)。

由于本例程依赖[fairseq2n](https://github.com/facebookresearch/fairseq2/blob/v0.2.0/INSTALL_FROM_SOURCE.md)，该库目前只提供了x86_64架构的预编译好的安装包。为了快速进行运行环境搭建，我们提供了arm64平台预编译好的`fairseq2n==0.2.0`的whl包，arm64平台需要单独执行如下命令安装：
```bash
pip3 install dfss --upgrade #安装dfss依赖
python3 -m dfss --url=open@sophgo.com:test/seamless_bmodel/0415/fairseq2n-0.2.0-cp38-cp38-linux_aarch64.whl
pip3 install fairseq2n-0.2.0-cp38-cp38-linux_aarch64.whl
rm -f fairseq2n-0.2.0-cp38-cp38-linux_aarch64.whl
```
对于其他平台（例如arm32），可以在平台上从源码编译安装：
```bash
git clone https://github.com/facebookresearch/fairseq2.git
cd fairseq2
# 需要使用0.2.0版本
git checkout v0.2.0
```
然后参考[INSTALL_FROM_SOURCE.md](https://github.com/facebookresearch/fairseq2/blob/main/INSTALL_FROM_SOURCE.md)进行安装。

此外您还需要安装其他第三方库：
```bash
sudo apt install libsndfile1
pip3 install -r service/server_requirements.txt
pip3 install -r ../requirements.txt
```
您还需要安装sophon-sail，这里提供一个可用于libsophon-0.4.9、sophon-opencv_0.7.0、sophon-ffmpeg_0.7.0的sophon-sail whl安装包，x86/arm PCIe环境可以通过下面的命令下载：
```bash
pip3 install dfss --upgrade #安装dfss依赖

#x86 pcie, py38
python3 -m dfss --url=open@sophgo.com:sophon-demo/Seamless/sophon-3.8.0-py3-none-any.whl
pip3 install sophon-3.8.0-py3-none-any.whl --force-reinstall

#arm pcie, py38
python3 -m dfss --url=open@sophgo.com:sophon-demo/Seamless/sophon_arm-3.8.0-py3-none-any.whl
pip3 install sophon_arm-3.8.0-py3-none-any.whl --force-reinstall
```
如果您需要在其他版本的libsophon、sophon-opencv、sophon-ffmpeg上安装sophon-sail，或者遇到glibc版本问题（pcie环境常见），可以通过以下命令下载源码，参考[sophon-sail编译安装指南](https://doc.sophgo.com/sdk-docs/v23.07.01/docs_latest_release/docs/sophon-sail/docs/zh/html/1_build.html#)自行编译sophon-sail。
```bash
python3 -m dfss --url=open@sophgo.com:sophon-demo/Seamless/sophon-sail.zip
unzip sophon-sail.zip
```
### 1.2 SoC平台

如果您使用SoC平台（如SE、SM系列边缘设备），并使用它测试本例程，刷机后在`/opt/sophon/`下已经预装了相应的libsophon、sophon-opencv和sophon-ffmpeg运行库包。

此外您还需要安装其他第三方库：
```bash
pip3 install dfss --upgrade #安装dfss依赖
python3 -m dfss --url=open@sophgo.com:test/seamless_bmodel/0415/fairseq2n-0.2.0-cp38-cp38-linux_aarch64.whl
pip3 install fairseq2n-0.2.0-cp38-cp38-linux_aarch64.whl
rm -f fairseq2n-0.2.0-cp38-cp38-linux_aarch64.whl

sudo apt install libsndfile1
pip3 install -r service/server_requirements.txt
pip3 install -r ../requirements.txt
```
您还需要安装sophon-sail，这里提供一个可用于libsophon-0.4.9、sophon-opencv_0.7.0、sophon-ffmpeg_0.7.0的sophon-sail whl安装包，SoC环境可以通过下面的命令下载和安装：
```bash
pip3 install dfss --upgrade
# SE7和SE5使用如下whl安装包
python3 -m dfss --url=open@sophgo.com:sophon-demo/Seamless/sophon_arm-3.8.0-py3-none-any.whl #arm soc, py38
pip3 install sophon_arm-3.8.0-py3-none-any.whl --force-reinstall
```
如果您需要在其他版本的libsophon、sophon-opencv、sophon-ffmpeg上安装sophon-sail，可以参考上一小节，下载源码自行编译。

## 2. Client端环境准备

### 2.1 x86/arm PCIe、SoC平台

您只需要执行如下命令安装其他第三方库：
```bash
pip3 install -r client/client_requirements.txt
```

## 3. Server端准备模型
该模型目前只支持在1684X上运行，已提供编译好的bmodel。​同时，您也可以重新编译BModel，可参考[模型编译](../../README.md#4-模型编译)。

​本例程在`scripts`目录下提供了相关模型的下载脚本`download_bmodel.sh`

```bash
# 安装unzip，若已安装请跳过
sudo apt install unzip
chmod -R +x scripts/
# 在服务端执行
./scripts/download_bmodel.sh
```

下载的模型包括：
```
./models
├── BM1684X
|   ├── m4t_encoder_frontend_fp16_s2t.bmodel                                                         # M4t(s2t任务) Encoder的前端模块，fp16 BModel
|   ├── m4t_decoder_frontend_beam_size_fp16_s2t.bmodel                                               # M4t(s2t任务) Decoder的前端模块，fp16 BModel
|   ├── m4t_decoder_final_proj_beam_size_fp16_s2t.bmodel                                             # M4t(s2t任务) Decoder的线性模块，fp16 BModel
|   ├── m4t_encoder_fp16_s2t.bmodel                                                                  # M4t(s2t任务) Encoder模型，fp16 BModel
|   ├── m4t_decoder_beam_size_fp16_s2t.bmodel                                                        # M4t(s2t任务) Decoder模块，fp16 BModel
|   ├── seamless_streaming_encoder_frontend_fp16_s2t.bmodel                                          # SeamlessStreaming(s2t任务) Encoder的前端模块，fp16 BModel
|   ├── seamless_streaming_decoder_frontend_fp16_s2t.bmodel                                          # SeamlessStreaming(s2t任务) Decoder的前端模块，fp16 BModel
|   ├── seamless_streaming_decoder_final_proj_fp16_s2t.bmodel                                        # SeamlessStreaming(s2t任务) Decoder的线性模块，fp16 BModel
|   ├── seamless_streaming_encoder_fp16_s2t.bmodel                                                   # SeamlessStreaming(s2t任务) Encoder模型，fp16 BModel
|   ├── seamless_streaming_decoder_step_bigger_1_fp16_s2t.bmodel                                     # SeamlessStreaming(s2t任务) Decoder模块，大于第一步解码的fp16 BModel
|   └── seamless_streaming_decoder_step_equal_1_fp32_s2t.bmodel                                      # SeamlessStreaming(s2t任务) Decoder模块，第一步解码的fp32 BModel
├── punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727                                        # 标点符号恢复模型目录
└── tokenizer.model                                                                                  # SeamlessStreaming(s2t任务)和M4t(s2t任务)的 tokenizer
```

## 4. Client端准备数据
已提供测试数据，​同时，您可以自行准备用于测试的数据集。

​本例程在`scripts`目录下提供了相关数据的下载脚本`download_datasets.sh`

```bash
# 安装unzip，若已安装请跳过
sudo apt install unzip
chmod -R +x scripts/
# 在客户端执行
./scripts/download_datasets.sh
```

下载的数据包括：
```
./datasets
|── aishell_S0764                             # 从aishell数据集中抽取的用于测试的音频文件
|   └── *.wav
├── aishell_S0764.list                        # 从aishell数据集的文件列表
├── ground_truth.txt                          # 从aishell数据集的预测真实值
└── test                                      # 测试使用的音频文件
    ├── long_audio.wav                        # 2分58秒的长语音音频文件
    └── demo.wav
```

## 5. 推理测试
python例程不需要编译，可以直接运行，PCIe平台和SoC平台的测试参数和运行方式是相同的。
### 5.1 参数说明

服务端参数说明如下：
```bash
usage: service/wss_server.py [-h] [--host HOST] [--port PORT] [--use_offline] [--offline_encoder_frontend_bmodel OFFLINE_ENCODER_FRONTEND_BMODEL] [--offline_encoder_bmodel OFFLINE_ENCODER_BMODEL]
                     [--offline_decoder_frontend_bmodel OFFLINE_DECODER_FRONTEND_BMODEL] [--offline_decoder_bmodel OFFLINE_DECODER_BMODEL] [--offline_decoder_final_proj_bmodel OFFLINE_DECODER_FINAL_PROJ_BMODEL]
                     [--max_output_seq_len MAX_OUTPUT_SEQ_LEN] [--beam_size BEAM_SIZE] [--use_online] [--online_encoder_frontend_bmodel ONLINE_ENCODER_FRONTEND_BMODEL]
                     [--online_encoder_bmodel ONLINE_ENCODER_BMODEL] [--tokenizer_model TOKENIZER_MODEL] [--online_decoder_frontend_bmodel ONLINE_DECODER_FRONTEND_BMODEL]
                     [--online_decoder_step_bigger_1_bmodel ONLINE_DECODER_STEP_BIGGER_1_BMODEL] [--online_decoder_step_equal_1_bmodel ONLINE_DECODER_STEP_EQUAL_1_BMODEL]
                     [--online_decoder_final_proj_bmodel ONLINE_DECODER_FINAL_PROJ_BMODEL] [--chunk_duration_ms CHUNK_DURATION_MS] [--consecutive_segments_num CONSECUTIVE_SEGMENTS_NUM]
                     [--fbank_min_input_length FBANK_MIN_INPUT_LENGTH] [--fbank_min_starting_wait FBANK_MIN_STARTING_WAIT] [--tgt_lang TGT_LANG] [--dev_id DEV_ID] [--punc_model PUNC_MODEL] [--punc_model_revision PUNC_MODEL_REVISION] [--device DEVICE] [--ncpu NCPU] [--certfile CERTFILE] [--keyfile KEYFILE]

--host: websocket服务ip，一般保持默认为本地主机ip
--port: websocket服务端口
--use_offline: 是否使用离线模型
--offline_encoder_frontend_bmodel: 离线m4t模型的Encoder Frontend Bmodel路径
--offline_encoder_bmodel: 离线m4t模型的Encoder Bmodel路径
--offline_decoder_frontend_bmodel: 离线m4t模型的Decoder Frontend Bmodel路径
--offline_decoder_bmodel: 离线m4t模型的Decoder Bmodel路径
--offline_decoder_final_proj_bmodel: 离线m4t模型的最终线性变换层 Bmodel路径
--max_output_seq_len: 离线m4t模型输出序列的最大长度
--beam_size: 离线m4t模型的beam search的长度
--use_online: 是否使用流式模型
--online_encoder_frontend_bmodel: 流式SeamlessStreaming模型的Encoder Frontend Bmodel路径
--online_encoder_bmodel: 流式SeamlessStreaming模型的Encoder Bmodel路径
--tokenizer_model: Seamless模型的tokenizer路径
--online_decoder_frontend_bmodel: 流式SeamlessStreaming模型的Decoder Frontend Bmodel路径
--online_decoder_step_bigger_1_bmodel: 流式SeamlessStreaming模型的Decoder解码大于第一步的 Bmodel路径
--online_decoder_step_equal_1_bmodel: 流式SeamlessStreaming模型的Decoder解码等于第一步的 Bmodel路径
--online_decoder_final_proj_bmodel: 流式SeamlessStreaming模型的最终线性变换层 Bmodel路径
--chunk_duration_ms: 流式SeamlessStreaming模型输入切片的大小，单位为毫秒，大小会影响精度和性能
--consecutive_segments_num: 流式SeamlessStreaming模型输入切片的数量，大小会影响精度和性能
--fbank_min_input_length: 流式SeamlessStreaming模型将切片转为fbank，模型Encoder能够接受的fbank最小长度，若末尾的语音片段小于该值，将被丢弃
--fbank_min_starting_wait: 流式SeamlessStreaming模型将切片转为fbank，输入到模型的fbank最小长度，若末尾的语音片段小于该值，会保留，当它大于fbank_min_input_length的值时才有效
--tgt_lang: 输出的目标语言
--dev_id: 设备id
--punc_model: PUNC标点符号恢复模型的路径，cpu执行
--punc_model_revision: PUNC标点符号恢复模型的版本，暂无用
--device: 设备，仅仅支持cpu
--ncpu: 用于PUNC模型的cpu核心数
--certfile: ssl的证书文件，空字符串表示不使用，使用可以设置service/server.crt
--keyfile: ssl的密钥文件，空字符串表示不使用，使用可以设置service/server.key
```

客户端参数说明如下：
```bash
usage: client/wss_client.py [-h] [--hosts HOSTS [HOSTS ...]] [--ports PORTS [PORTS ...]] [--chunk_duration_ms CHUNK_DURATION_MS] [--vad_level VAD_LEVEL] [--online_use_vad ONLINE_USE_VAD]
                     [--encoder_chunk_look_back ENCODER_CHUNK_LOOK_BACK] [--decoder_chunk_look_back DECODER_CHUNK_LOOK_BACK]
                     [--chunk_interval CHUNK_INTERVAL] [--hotword HOTWORD] [--audio_in AUDIO_IN] [--audio_fs AUDIO_FS] [--microphone_dev_id MICROPHONE_DEV_ID] [--thread_num THREAD_NUM] [--words_max_print WORDS_MAX_PRINT]
                     [--output_dir OUTPUT_DIR] [--ssl SSL] [--use_itn USE_ITN] [--mode MODE]
--hosts: 服务端ip，对于parallel2pass模式，需要分别给出流式服务器ip和离线服务器ip
--ports: 服务端端口，对于parallel2pass模式，需要分别给出流式服务器端口和离线服务器端口
--chunk_duration_ms: 流式模型的输入切片大小，单位为毫秒
--vad_level: webrtcvad识别语音和非语音的敏感度，取值范围为[0,3]，值越大越敏感
--online_use_vad: 是否在音频传入流式模型前使用vad过滤无人声片段，默认不使用
--encoder_chunk_look_back: 暂无用，用于后期扩展
--decoder_chunk_look_back: 暂无用，用于后期扩展
--chunk_interval: 暂无用，用于后期扩展
--hotword: 暂无用，用于后期扩展
--audio_in: 输入的音频文件，若为None则输入为麦克风
--audio_fs: 输入音频的采样率
--microphone_dev_id: 麦克风输入时的设备id
--thread_num: 暂无用，用于后期扩展
--words_max_print: 客户端显示识别结果的最大字符串序列长度
--output_dir: 输出结果文件夹，若为空则不输出文件
--ssl: 是否使用ssl加密传输，0表示不使用，1表示使用
--use_itn: 暂无用，用于后期扩展
--mode: 模式，支持offline、online、parallel2pass，分别表示仅vad断句+离线识别+标点符号恢复、仅流式识别、流式识别+vad断句+离线纠正+标点符号恢复
```

### 5.2 使用方式
### 5.2.1 流式+离线修正并行方式
该方式会并行运行流式算法和离线算法，离线算法会纠正流式算法的识别结果，流式和离线算法并行运行。

为了测试实时中文语音转中文文字并进行修正，可在Server端使用如下命令先启动流式服务端进行监听
```bash
# 启动流式推理进程
python3 service/wss_server.py --port 10095 --use_online
```

然后在Server端使用如下命令启动离线服务端进行监听
```bash
# 启动离线推理进程
python3 service/wss_server.py --port 10096 --use_offline
```

然后在Client端使用如下命令启动客户端，客户端会发送指令和音频数据给服务端进行语音识别，可根据实际情况修改IP
```bash
# 若使用本地音频文件，使用如下命令
python3 client/wss_client.py --hosts 127.0.0.1 127.0.0.1 --ports 10095 10096 --mode parallel2pass --audio_in ../../datasets/test/long_audio.wav
# 若使用麦克风输入，使用如下命令，需根据实际情况修改麦克风设备id
python3 client/wss_client.py --hosts 127.0.0.1 127.0.0.1 --ports 10095 10096 --mode parallel2pass --microphone_dev_id 1 --online_use_vad
```

### 5.2.2 流式方式
为了测试实时中文语音转中文文字，可使用如下命令先在Server端启动流式服务端进行监听
```bash
# 启动流式推理进程
python3 service/wss_server.py --port 10095 --use_online
```

然后在Client端使用如下命令启动客户端，客户端会发送指令和音频数据给服务端进行语音识别，可根据实际情况修改IP

```bash
# 若使用本地音频文件，使用如下命令
python3 client/wss_client.py --hosts 127.0.0.1 --ports 10095 --mode online --audio_in ../../datasets/test/long_audio.wav
# 若使用麦克风输入，使用如下命令，需根据实际情况修改麦克风设备id
python3 client/wss_client.py --hosts 127.0.0.1 --ports 10095 --mode online --microphone_dev_id 1 --online_use_vad
```

### 3.2.3 离线方式
为了测试离线中文语音转中文文字，可在Server端使用如下命令先启动离线服务端进行监听

```bash
# 启动离线推理进程
python3 service/wss_server.py --port 10095 --use_offline
```

然后在Client端使用如下命令在Server端启动客户端，客户端会发送指令和音频数据给服务端进行语音识别，可根据实际情况修改IP

```bash
# 若使用本地音频文件，使用如下命令
python3 client/wss_client.py --hosts 127.0.0.1 --ports 10095 --mode offline --audio_in ../../datasets/test/long_audio.wav
# 若使用麦克风输入，使用如下命令，需根据实际情况修改麦克风设备id
python3 client/wss_client.py --hosts 127.0.0.1 --ports 10095 --mode offline --microphone_dev_id 1
```
