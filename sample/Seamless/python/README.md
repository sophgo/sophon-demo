# Python例程 <!-- omit in toc -->

## 目录 <!-- omit in toc -->
- [1. 环境准备](#1-环境准备)
  - [1.1 x86/arm PCIe平台](#11-x86arm-pcie平台)
  - [1.2 SoC平台](#12-soc平台)
- [2. 推理测试](#2-推理测试)
  - [2.1 参数说明](#21-参数说明)
  - [2.2 使用方式](#22-使用方式)

python目录下提供了一系列Python例程，具体情况如下：

| 序号  |             Python例程                    |             说明                |
| ---- | ----------------------------------------  | ------------------------------- |
| 1    |    pipeline_seamless_streaming_s2t.py     |         使用SAIL推理             |


## 1. 环境准备

### 1.1 x86/arm PCIe平台

如果您在x86/arm平台安装了PCIe加速卡（如SC系列加速卡），并使用它测试本例程，您需要安装libsophon、sophon-opencv、sophon-ffmpeg，具体请参考[x86-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#3-x86-pcie平台的开发和运行环境搭建)或[arm-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#5-arm-pcie平台的开发和运行环境搭建)。

由于本例程依赖[fairseq2n](https://github.com/facebookresearch/fairseq2/blob/v0.2.0/INSTALL_FROM_SOURCE.md)，该库目前只提供了x86_64架构的预编译好的安装包。为了快速进行运行环境搭建，我们提供了arm64平台预编译好的`fairseq2n==0.2.0`的whl包，arm64平台需要单独执行如下命令安装：
```bash
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
python3 -m dfss --url=open@sophgo.com:test/seamless_bmodel/0415/fairseq2n-0.2.0-cp38-cp38-linux_aarch64.whl
pip3 install fairseq2n-0.2.0-cp38-cp38-linux_aarch64.whl
rm -f fairseq2n-0.2.0-cp38-cp38-linux_aarch64.whl

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
usage: pipeline_seamless_streaming_s2t.py [-h] [--input INPUT] [--tgt_lang TGT_LANG] [--encoder_frontend_bmodel ENCODER_FRONTEND_BMODEL]
                                                                                                         [--encoder_bmodel ENCODER_BMODEL] [--tokenizer_model TOKENIZER_MODEL]
                                                                                                         [--decoder_frontend_step_bigger_1_bmodel DECODER_FRONTEND_STEP_BIGGER_1_BMODEL]
                                                                                                         [--decoder_frontend_step_equal_1_bmodel DECODER_FRONTEND_STEP_EQUAL_1_BMODEL]
                                                                                                         [--decoder_step_bigger_1_bmodel DECODER_STEP_BIGGER_1_BMODEL] [--decoder_step_equal_1_bmodel DECODER_STEP_EQUAL_1_BMODEL]
                                                                                                         [--decoder_final_proj_bmodel DECODER_FINAL_PROJ_BMODEL] [--dev_id DEV_ID] [--sample_rate SAMPLE_RATE]
                                                                                                         [--use_slience_remover] [--source_segment_size SOURCE_SEGMENT_SIZE]
--input: 输入音频文件目录，文件夹中为.wav格式文件。
--tgt_lang: 输出的目标语言。支持的语言可以参考https://github.com/facebookresearch/seamless_communication/blob/main/docs/m4t/README.md#supported-languages，其中的'Target'列为'Tx'的语言，填上对应的'code'即可完成对应语言的识别或翻译
--encoder_frontend_bmodel: Encoder前端模型。
--encoder_bmodel: Encoder模型。
--tokenizer_model: tokenizer模型。
--decoder_frontend_step_bigger_1_bmodel: Decoder前端模型，用于step大于1的情况。
--decoder_frontend_step_equal_1_bmodel: Decoder前端模型，用于step等于1的情况。
--decoder_step_bigger_1_bmodel: Decoder模型，用于step大于1的情况。
--decoder_step_equal_1_bmodel: Decoder模型，用于step等于1的情况。
--decoder_final_proj_bmodel: Decoder最终的线性模型。
--dev_id: 设备id。默认为 0。
--sample_rate: 输入音频的采样率。默认为 16000。
--use_slience_remover: 是否对输入音频进行自动的无效片段的删除。默认为 False。
--source_segment_size: 音频切片的长度，可根据具体数据修改，基于ffmpeg解码出来的序列长度。默认为 320。
```

### 2.2 使用方式
为了测试中文语音转英文文字，可使用如下命令

```bash
cd python

python3 pipeline_seamless_streaming_s2t.py --input=../datasets/aishell_S0764 --encoder_frontend_bmodel=../models/BM1684X/seamless_streaming_encoder_frontend_fp16_s2t.bmodel --encoder_bmodel=../models/BM1684X/seamless_streaming_encoder_fp16_s2t.bmodel --tokenizer_model=../models/tokenizer.model --decoder_frontend_step_bigger_1_bmodel=../models/BM1684X/seamless_streaming_decoder_frontend_step_bigger_1_fp16_s2t.bmodel --decoder_frontend_step_equal_1_bmodel=../models/BM1684X/seamless_streaming_decoder_frontend_step_equal_1_fp16_s2t.bmodel --decoder_step_bigger_1_bmodel=../models/BM1684X/seamless_streaming_decoder_step_bigger_1_fp16_s2t.bmodel --decoder_step_equal_1_bmodel=../models/BM1684X/seamless_streaming_decoder_step_equal_1_fp32_s2t.bmodel --decoder_final_proj_bmodel=../models/BM1684X/seamless_streaming_decoder_final_proj_fp16_s2t.bmodel --tgt_lang=eng --dev_id=0
```

为了测试中文语音转中文文字，可使用如下命令

```bash
cd python

python3 pipeline_seamless_streaming_s2t.py --input=../datasets/aishell_S0764 --encoder_frontend_bmodel=../models/BM1684X/seamless_streaming_encoder_frontend_fp16_s2t.bmodel --encoder_bmodel=../models/BM1684X/seamless_streaming_encoder_fp16_s2t.bmodel --tokenizer_model=../models/tokenizer.model --decoder_frontend_step_bigger_1_bmodel=../models/BM1684X/seamless_streaming_decoder_frontend_step_bigger_1_fp16_s2t.bmodel --decoder_frontend_step_equal_1_bmodel=../models/BM1684X/seamless_streaming_decoder_frontend_step_equal_1_fp16_s2t.bmodel --decoder_step_bigger_1_bmodel=../models/BM1684X/seamless_streaming_decoder_step_bigger_1_fp16_s2t.bmodel --decoder_step_equal_1_bmodel=../models/BM1684X/seamless_streaming_decoder_step_equal_1_fp32_s2t.bmodel --decoder_final_proj_bmodel=../models/BM1684X/seamless_streaming_decoder_final_proj_fp16_s2t.bmodel --tgt_lang=cmn --dev_id=0  --source_segment_size=640
```
