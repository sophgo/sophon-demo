# cv Demo

## 目录
- [cv Demo](#cv_demo)
  - [目录](#目录)
  - [1. 简介](#1-简介)
  - [2. 特性](#2-特性)
  - [3. 准备数据](#3-准备数据)
  - [4. 例程测试](#4-例程测试)
  - [5. 性能测试](#5-性能测试)

## 1. 简介

本例程用于说明如何使用BM1688快速构建双目鱼眼或者广角拼接应用。

本例程中，cv_demo算法的dwa展开、blend拼接分别在两个线程上进行运算，保证了一定的运行效率。

## 2. 特性

* 支持BM1688(SoC)
* 支持二路视频流
* 支持多线程

## 3. 准备数据

​在`scripts`目录下提供了相关数据的下载脚本 [download.sh](./scripts/download.sh)。

```bash
# 安装unzip，若已安装请跳过，非ubuntu系统视情况使用yum或其他方式安装
sudo apt install unzip
chmod -R +x scripts/
./scripts/download.sh
```

脚本执行完毕后，会在当前目录下生成`data`目录,子目录如下。

.
├── gridinfo # 用于dwa模块的参数文件
├── images   # 测试图片
└── wgt     # 用于拼接的权重文件


## 4. 例程测试

- [C++例程](./cpp/README.md)

## 5. 性能测试

目前，鱼眼和广角拼接算法只支持在BM1688 SOC模式下进行推理。按照默认设置可以达到30fps。

