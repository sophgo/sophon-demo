# Wan2.2

## 目录

* [1. 简介](#1-简介)
* [2. 文件路径结构](#2-文件路径结构)
* [3. 特性](#3-特性)
* [4. 准备模型与数据](#4-准备模型与数据)
* [5. 测试环境准备](#5-测试环境准备)
* [6. 例程测试](#6-例程测试)

## 1. 简介

本例程基于[Wan2.2](https://github.com/Wan-Video/Wan2.2)中算法进行适配，使其可以在SOPHON BM1690设备上进行使用。

## 2. 文件路径结构

```bash
├── docs                  # 存放本例程使用文档，如测试环境配置方法
├── python                # 存放Python例程及其README
|   ├──README.md          # Wan2.2官方README
|   ├──README_TPU.md      # 本例程的指南，同外部README文件
|   ├──generate.py        # Wan2.2例程启动文件
|   └──...                # Python例程共用功能的封装。
├── README.md             # 本例程的指南
└── scripts               # 存放镜像以及TORCH_TPU下载脚本
```

## 3. 例程特性

* 支持BM1690(PCIe)，各版本修改可查看[CHANGELOG](./docs/CHANGELOG.md)，具体修改细节可在代码中搜索DEVICE以查看注释。
* 支持SP，TP并行推理
* 支持BF16模型
* 支持T2V-A14B以及TI2V-5B

## 4. 准备模型与数据

使用huggingface-cli下载模型：

```bash
pip install "huggingface_hub[cli]"
huggingface-cli download Wan-AI/Wan2.2-T2V-A14B --local-dir ./Wan2.2-T2V-A14B
huggingface-cli download Wan-AI/Wan2.2-TI2V-5B-BF16 --local-dir ./Wan2.2-TI2V-5B-BF16
```

使用modelscope-cli下载模型：

```bash
pip install modelscope
modelscope download Wan-AI/Wan2.2-T2V-A14B --local_dir ./Wan2.2-T2V-A14B
modelscope download Wan-AI/Wan2.2-TI2V-5B-BF16 --local_dir ./Wan2.2-TI2V-5B-BF16
```

## 5. 测试环境准备

在测试前参考[INSTALL_TPU](./docs/INSTALL_TPU.md)文件进行环境配置。

## 6. 例程测试

1. 单芯推理（以TI2V例程为例）

    ```bash
    python3 generate.py --task ti2v-5B --size 1280*704 --ckpt_dir ./Wan2.2-TI2V-5B-BF16/ --convert_model_dtype --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage"
    ```

    可在python指令前添加**CHIP_MAP**用于指定推理使用设备

    参数部分与原Wan2.2例程保持一致

2. 多芯并行推理（以TI2V例程为例）

    SP 并行：

    ```bash
    CHIP_MAP=0,1 CONV_SHARD=ic torchrun --nproc_per_node 2 --nnodes 1 generate.py --task ti2v-5B --size 1280*704 --ckpt_dir ./Wan2.2-TI2V-5B-BF16/ --convert_model_dtype --ulysses_size 2 --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage"
    ```

    TP 并行：

    ```bash
    CHIP_MAP=0,1 CONV_SHARD=ic torchrun --nproc_per_node 2 --nnodes 1 generate.py --task ti2v-5B --size 1280*704 --ckpt_dir ./Wan2.2-TI2V-5B-BF16/ --convert_model_dtype --tp_size 2 --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage"
    ```

    **CHIP_MAP**用于指定多芯推理使用设备。
    **CONV_SHARD=ic**用于启动vae阶段TP并行，可不添加。
