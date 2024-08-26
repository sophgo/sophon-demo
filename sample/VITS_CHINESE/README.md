# VITS_CHINESE

## 目录
  - [1. 简介](#1-简介)
  - [2. 特性](#2-特性)
  - [3. 准备模型](#3-准备模型)
  - [4. 例程测试](#4-例程测试)
  - [5. 程序性能测试](#5-程序性能测试)

## 1. 简介
VITS 是一种并行的端到端TTS方法，该方法比当前的两阶段模型产生更自然的声音。关于它的特性，请前往源repo查看：https://github.com/PlayVoice/vits_chinese  。本例程对vits_chinese进行移植，使之能在SOPHON BM1684X上进行推理测试。为了生成更自然的语音，本例程需要使用BERT模型生成字符嵌入。

BM1684X系列：该例程支持在V24.04.01及以上的SDK上运行，支持在插有1684X加速卡(SC7系列)的x86主机上运行，也可以在1684X SoC设备（如SE7）上运行。

BM1688/CV186X系列：该例程支持V1.7及以上的SDK上运行，支持SE9-16/SE9-8
## 2. 特性
* 支持BM1684X(x86 PCIe、SoC)、BM1688/CV186X(SoC)
* 支持FP16模型编译和推理
* 支持基于SAIL推理的Python例程

## 3. 准备模型
该模型目前支持在bm1684X、bm1688/cv186x上运行，已提供编译好的bmodel。
### 3.1 使用提供的模型

​本例程在`scripts`目录下提供了下载脚本`download.sh`

```bash
chmod -R +x scripts/
./scripts/download.sh
```

执行下载脚本后，当前目录下的文件如下：

```bash
.
├── datasets
│   └── vits_infer_item.txt              #文本测试数据
├── docs
│   └── VITS_CHINESE_Export_Guide.md     # VITS_CHINESE ONNX导出和 bmodel 编译指南
├── models                               # download.sh下载的 bmodel
│   ├── BM1684X
│   │   ├── bert_f32_1core.bmodel
│   │   └── vits_chinese_f16.bmodel
│   ├── BM1688
│   │   ├── bert_f16_1core.bmodel
│   │   └── vits_chinese_f16.bmodel
│   └── CV186X
│       ├── bert_f16_1core.bmodel
│       └── vits_chinese_f16.bmodel
├── python
│   ├── bert
│   │   ├── config.json
│   │   ├── __init__.py
│   │   ├── ProsodyModel.py
│   │   ├── prosody_tool.py
│   │   └── vocab.txt
│   ├── image.png                         #流程图
│   ├── monotonic_align
│   │   ├── core.c
│   │   ├── core.pyx
│   │   ├── __init__.py
│   │   └── setup.py
│   ├── README.md                         # python 例程执行指南
│   ├── requirements_soc.txt              # python 例程的依赖模块(适用于SoC模式)
│   ├── requirements.txt                  # python 例程的依赖模块(适用于PCIe模式)
│   ├── text
│   │   ├── __init__.py
│   │   ├── __init__.pyc
│   │   ├── pinyin-local.txt
│   │   ├── symbols.py
│   │   └── symbols.pyc
│   └── vits_infer_sail.py                # VITS_CHINESE python 推理脚本
├── README.md                             # VITS_CHINESE 例程指南
├── scripts
│   ├── auto_test.sh                      
│   ├── download.sh                       #下载脚本
│   └── gen_bmodel.sh                     # bmodel编译脚本
└── tools
    ├── compare_statis.py                
    ├── configs
    │   └── bert_vits.json
    ├── model_onnx.py                     # VITS_CHINESE ONNX导出脚本
    ├── requirements_model.txt            # ONNX导出的依赖模块
    └── utils.py
```

### 3.2 自行编译模型

此部分请参考[VITS模型导出与编译](./docs/VITS_CHINESE_Export_Guide.md)

## 4. 例程测试

- [Python例程](./python/README.md)

## 5. 程序性能测试
例程测试后能在终端看到前处理时间、推理时间、后处理时间，以及从数据载入到生成音频文件的总时间
|    测试平台  |     测试程序      |             测试模型               |preprocess_time |inference_time   |postprocess_time| 
| ----------- | ---------------- | -----------------------------------| ---------------| --------------- | -------------- | 
|   SE7-32    |vits_infer_sail.py|      vits_chinese_f16.bmodel       |     46.26      |     232.07      |      69.75     |
|   SE9-16    |vits_infer_sail.py|      vits_chinese_f16.bmodel       |     89.15      |     1203.62     |     96.80      |
|    SE9-8    |vits_infer_sail.py|      vits_chinese_f16.bmodel       |     87.28      |     1185.91     |     99.84      |
- 测试说明
1. 性能测试结果具有一定的波动性，建议多次测试取平均值；
2. SE7-32(BM1684X) SDK版本:V24.04.01；SE9-16(BM1688)和SE9-8(CV186X) SDK版本:V1.7；
3. SE7-32的主控处理器为8核CA53@2.3GHz，SE9-16为8核CA53@1.6GHz，SE9-8为6核CA53@1.6GHz，PCIe上的性能由于处理器的不同可能存在较大差异；