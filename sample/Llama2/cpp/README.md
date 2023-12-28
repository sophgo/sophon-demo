# Llama2 C++例程

## 目录
- [Llama2 C++例程](#llama2-C++例程)
  - [目录](#目录)
  - [1. 编译程序(C++版本)](#1-编译程序(C++版本))
  - [2. 例程测试](#2-例程测试)

如果在使用或者测试过程中遇到问题，可以先参考[常见问题说明](../docs/Llama2_Guide.md)

## 1. 编译程序(C++版本)
PCIE环境下直接编译可以在之前使用的tpuc的docker下继续执行；[交叉编译所需要的工具请参考](../../../docs/Environment_Install_Guide.md#4-soc平台的开发和运行环境搭建)

1. PCIE环境下直接编译
请先确认之前已经执行过`Llama2/scripts/download.sh`, 并且在`Llama2/tools/libsophon-distributed`对应文件。
考虑到所使用的`libsophon-distributed`是在GLIBC_2.33+版本下编译得到的，因此建议编译可执行程序的步骤也在之前编译模型(sophgo-tpuc:latest)的docker中执行，可以参考[常见问题](../docs/常见问题.doc)。
当前路径/Llama2/cpp

```shell
mkdir build
cd build
cmake -DTARGET_ARCH=pcie ..
make -j
```
 
2. SOC环境下直接编译
```shell
mkdir build
cd build
cmake -DTARGET_ARCH=soc_base ..
make -j
```

## 2. 例程测试
在编译完成后，会根据pcie或soc模式在项目路径下生成llama2的可执行文件,将llama2可执行文件,tokenizer.model放在同一路径下, bmodel路径可以通过model参数来指定,设置好以后即可运行

多芯(仅在多卡PCIE模式下可用)
```shell
./llama2.pcie --model=your_bmodel_name --dev_id=your_tpu_id(如0,1 表示tpu0和tpu1双芯推理)
```

单芯(pcie和soc均支持, 以soc为例)
```shell
./llama2.soc --model=your_bmodel_name
```