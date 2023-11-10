# Llama2 C++例程

## 目录
- [Llama2 C++例程](#Llama2 C++例程)
  - [目录](#目录)
  - [1. 编译程序(C++版本)](#1-编译程序(C++版本))
  - [2. 例程测试](#2-例程测试)


## 1. 编译程序(C++版本)
1. PCIE环境下直接编译
(请先确认Llama2/Llama2-pcie 中 LIBSOPHON 指定到了tools当中的libsophon_distributed/install/libsophon_0.4.9中). 当前路径/Llama2/cpp

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

3. 交叉编译
首先准备一台x86机器

```shell
mkdir build
cd build
cmake -DTARGET_ARCH=soc ..
make -j
```

如果交叉编译后的模型是在SE7上运行，请确认当前Ubuntu版本小于22.04. 否则会出现GLIBC版本过高无法执行的问题


## 2. 例程测试
在编译完成后，会在build路径下生成llama2的可执行文件,将llama2可执行文件,tokenizer.model,以及对应的bmodel模型文件放在同一路径下,即可运行

多芯(仅在多卡PCIE模式下可用)
```shell
./llama2.pcie --model==your_bmodel_name --dev_id=0,1
```

单芯(pcie和soc均支持, 以soc为例)
```shell
./llama2.soc --model==your_bmodel_name
```