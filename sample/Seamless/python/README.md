# Python例程 <!-- omit in toc -->

## 目录 <!-- omit in toc -->
- [1. 环境准备](#1-环境准备)
  - [1.1 x86/arm PCIe平台](#11-x86arm-pcie平台)
  - [1.2 SoC平台](#12-soc平台)
- [2. 准备模型与数据](#2-准备模型与数据)
- [3. 推理测试](#3-推理测试)
  - [3.1 参数说明](#31-参数说明)
  - [3.2 使用方式](#32-使用方式)

python目录下提供了一系列Python例程，具体情况如下：

| 序号  |             Python例程                    |             说明                |
| ---- | ----------------------------------------  | ------------------------------- |
| 1    |    pipeline_seamless_streaming_s2t.py     |         使用SAIL推理             |


## 1. 环境准备

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
pip3 install -r requirements.txt
```
您还需要安装sophon-sail，由于本例程需要的sophon-sail版本较新，相关功能还未发布，这里暂时提供一个可用的sophon-sail版本，x86/arm PCIe环境可以通过下面的命令下载：
```bash
pip3 install dfss --upgrade #安装dfss依赖

#x86 pcie, py38
python3 -m dfss --url=open@sophgo.com:sophon-demo/Seamless/sophon-3.8.0-py3-none-any.whl
pip3 install sophon-3.8.0-py3-none-any.whl --force-reinstall

#arm pcie, py38
python3 -m dfss --url=open@sophgo.com:sophon-demo/Seamless/sophon_arm-3.8.0-py3-none-any.whl
pip3 install sophon_arm-3.8.0-py3-none-any.whl --force-reinstall
```
如果您需要其他版本的sophon-sail，或者遇到glibc版本问题（pcie环境常见），可以通过以下命令下载源码，参考[sophon-sail编译安装指南](https://doc.sophgo.com/sdk-docs/v23.07.01/docs_latest_release/docs/sophon-sail/docs/zh/html/1_build.html#)自己编译sophon-sail。
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
pip3 install -r requirements.txt
```
由于本例程需要的sophon-sail版本较新，这里提供一个可用的sophon-sail whl包，SoC环境可以通过下面的命令下载和安装：
```bash
pip3 install dfss --upgrade
# SE7和SE5使用如下whl安装包
python3 -m dfss --url=open@sophgo.com:sophon-demo/Seamless/sophon_arm-3.8.0-py3-none-any.whl #arm soc, py38
# SE9使用如下whl安装包
python3 -m dfss --url=open@sophgo.com:sophon-demo/Seamless/bm1688/sophon_arm-3.8.0-py3-none-any.whl
pip3 install sophon_arm-3.8.0-py3-none-any.whl --force-reinstall
```
如果您需要其他版本的sophon-sail，可以参考上一小节，下载源码自己编译。

## 2. 准备模型与数据
该模型目前只支持在BM1684X和BM1688上运行，已提供编译好的bmodel和测试数据，​同时，您也可以自行准备用于测试的数据集，以及重新编译模型，可参考[模型编译](../../README.md#4-模型编译)。

​本例程在`scripts`目录下提供了相关模型和数据的下载脚本
```bash
└── scripts
    ├── download_bm1684x_bmodel.sh                                           # 通过该脚本下载BM1684X平台的SeamlessStreaming(s2t任务)和M4t(s2t任务)的BModel
    ├── download_bm1688_1core_bmodel.sh                                      # 通过该脚本下载BM1688平台的SeamlessStreaming(s2t任务)和M4t(s2t任务)的单core BModel
    ├── download_bm1688_2core_bmodel.sh                                      # 通过该脚本下载BM1688平台的SeamlessStreaming(s2t任务)和M4t(s2t任务)的2core BModel
    └── download_datasets.sh                                                 # 通过该脚本下载测试数据
```

```bash
# 安装unzip，若已安装请跳过
sudo apt install unzip
chmod -R +x scripts/
# 若使用BM1684X平台模型
./scripts/download_bm1684x_bmodel.sh
# 若使用BM1688平台单core模型
./scripts/download_bm1688_1core_bmodel.sh
# 若使用BM1688平台2core模型
./scripts/download_bm1688_2core_bmodel.sh
# 下载测试数据
./scripts/download_datasets.sh
```

下载的模型包括：
```
./models
├── BM1684X或BM1688
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
├── campplus_cn_common.bin                                                                           # 说话人识别模型CAM++文件
└── tokenizer.model                                                                                  # SeamlessStreaming(s2t任务)和M4t(s2t任务)的 tokenizer
或
./models
├── BM1688
|   ├── m4t_encoder_frontend_fp16_s2t_2core.bmodel                                                   # M4t(s2t任务) Encoder的前端模块，fp16 2core BModel
|   ├── m4t_decoder_frontend_beam_size_fp16_s2t_2core.bmodel                                         # M4t(s2t任务) Decoder的前端模块，fp16 2core BModel
|   ├── m4t_decoder_final_proj_beam_size_fp16_s2t_2core.bmodel                                       # M4t(s2t任务) Decoder的线性模块，fp16 2core BModel
|   ├── m4t_encoder_fp16_s2t_2core.bmodel                                                            # M4t(s2t任务) Encoder模型，fp16 2core BModel
|   ├── m4t_decoder_beam_size_fp16_s2t_2core.bmodel                                                  # M4t(s2t任务) Decoder模块，fp16 2core BModel
|   ├── seamless_streaming_encoder_frontend_fp16_s2t_2core.bmodel                                    # SeamlessStreaming(s2t任务) Encoder的前端模块，fp16 2core BModel
|   ├── seamless_streaming_decoder_frontend_fp16_s2t_2core.bmodel                                    # SeamlessStreaming(s2t任务) Decoder的前端模块，fp16 2core BModel
|   ├── seamless_streaming_decoder_final_proj_fp16_s2t_2core.bmodel                                  # SeamlessStreaming(s2t任务) Decoder的线性模块，fp16 2core BModel
|   ├── seamless_streaming_encoder_fp16_s2t_2core.bmodel                                             # SeamlessStreaming(s2t任务) Encoder模型，fp16 2core BModel
|   ├── seamless_streaming_decoder_step_bigger_1_fp16_s2t_2core.bmodel                               # SeamlessStreaming(s2t任务) Decoder模块，大于第一步解码的fp16 2core BModel
|   └── seamless_streaming_decoder_step_equal_1_fp32_s2t_2core.bmodel                                # SeamlessStreaming(s2t任务) Decoder模块，第一步解码的fp32 2core BModel
├── punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727                                        # 标点符号恢复模型目录
├── campplus_cn_common.bin                                                                           # 说话人识别模型CAM++文件
└── tokenizer.model    
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

## 3. 推理测试
python例程不需要编译，可以直接运行，PCIe平台和SoC平台的测试参数和运行方式是相同的。
### 3.1 参数说明

流式算法配置参数说明：
```bash
usage: pipeline_seamless_streaming_s2t.py [-h] [--input INPUT] [--tgt_lang TGT_LANG] [--encoder_frontend_bmodel ENCODER_FRONTEND_BMODEL]
                                                                                                         [--encoder_bmodel ENCODER_BMODEL] [--tokenizer_model TOKENIZER_MODEL]
                                                                                                         [--decoder_frontend_bmodel DECODER_FRONTEND_BMODEL]
                                                                                                         [--decoder_step_bigger_1_bmodel DECODER_STEP_BIGGER_1_BMODEL] [--decoder_step_equal_1_bmodel DECODER_STEP_EQUAL_1_BMODEL]
                                                                                                         [--decoder_final_proj_bmodel DECODER_FINAL_PROJ_BMODEL] [--dev_id DEV_ID] [--sample_rate SAMPLE_RATE]
                                                                                                         [--use_slience_remover] [--chunk_duration_ms CHUNK_DURATION_MS] 
                                                                                                         [--consecutive_segments_num CONSECUTIVE_SEGMENTS_NUM]
                                                                                                         [--fbank_min_input_length FBANK_MIN_INPUT_LENGTH] [--fbank_min_starting_wait FBANK_MIN_STARTING_WAIT]
--input: 输入音频文件目录，文件夹中为.wav格式文件。
--tgt_lang: 输出的目标语言。支持的语言可以参考https://github.com/facebookresearch/seamless_communication/blob/main/docs/m4t/README.md#supported-languages，其中的'Target'列为'Tx'的语言，填上对应的'code'即可完成对应语言的识别或翻译
--encoder_frontend_bmodel: Encoder前端模型。
--encoder_bmodel: Encoder模型。
--tokenizer_model: tokenizer模型。
--decoder_frontend_bmodel: Decoder前端模型，用于step大于1的情况。
--decoder_step_bigger_1_bmodel: Decoder模型，用于step大于1的情况。
--decoder_step_equal_1_bmodel: Decoder模型，用于step等于1的情况。
--decoder_final_proj_bmodel: Decoder最终的线性模型。
--dev_id: 设备id。默认为 0。
--sample_rate: 输入音频的采样率。默认为 16000。
--use_slience_remover: 是否对输入音频进行自动的无效片段的删除。默认为 False。
--chunk_duration_ms: 音频切片的长度，可根据具体数据修改，单位为毫秒。默认为 1600。
--consecutive_segments_num: 一次性处理的切片数量，数量越多精度越高，但不能超过长度限制，默认为1。
--fbank_min_input_length: 模型Encoder能够接受的fbank最小长度，若末尾的语音片段小于该值，将被丢弃，默认为80。
--fbank_min_starting_wait: 输入到模型Encoder的fbank最小长度，若末尾的语音片段小于该值，会保留，当它大于--fbank_min_input_length的值时才有效，默认为48。
```

离线算法配置参数说明：
```bash
usage: pipeline_m4t_s2t.py [-h] [--input INPUT] [--tgt_lang TGT_LANG] [--encoder_frontend_bmodel ENCODER_FRONTEND_BMODEL] [--encoder_bmodel ENCODER_BMODEL]
                                                                                              [--tokenizer_model TOKENIZER_MODEL] [--decoder_frontend_bmodel DECODER_FRONTEND_BMODEL] [--decoder_bmodel DECODER_BMODEL]
                                                                                              [--decoder_final_proj_bmodel DECODER_FINAL_PROJ_BMODEL] [--max_output_seq_len MAX_OUTPUT_SEQ_LEN] [--beam_size BEAM_SIZE] [--dev_id DEV_ID]
--input: 输入音频文件目录，文件夹中为.wav格式文件。
--tgt_lang: 输出的目标语言。支持的语言可以参考https://github.com/facebookresearch/seamless_communication/blob/main/docs/m4t/README.md#supported-languages，其中的'Target'列为'Tx'的语言，填上对应的'code'即可完成对应语言的识别或翻译
--encoder_frontend_bmodel: Encoder前端模型。
--encoder_bmodel: Encoder模型。
--tokenizer_model: tokenizer模型。
--decoder_frontend_bmodel: Decoder前端模型。
--decoder_bmodel: Decoder模型。
--decoder_final_proj_bmodel: Decoder最终的线性模型。
--max_output_seq_len: 模型输出序列的最大长度，超过此长度会被截断，默认为50。
--beam_size: beam search的长度，值越大精度越高性能越差，最大支持5，默认为1。
--dev_id: 设备id，默认为 0。
```

### 3.2 使用方式
这里以BM1684X平台为例，其他平台需要修改模型路径参数。

为了测试实时中文语音转英文文字，可使用如下命令

```bash
cd python

python3 pipeline_seamless_streaming_s2t.py --input=../datasets/aishell_S0764 --tgt_lang=eng --dev_id=0
```

为了测试实时中文语音转中文文字，可使用如下命令

```bash
cd python

python3 pipeline_seamless_streaming_s2t.py --input=../datasets/aishell_S0764 --tgt_lang=cmn --dev_id=0
```

为了测试离线中文语音转中文文字，可使用如下命令

```bash
cd python

python3 pipeline_m4t_s2t.py --input ../datasets/aishell_S0764 --tgt_lang=cmn --dev_id=0 --beam_size 5
```