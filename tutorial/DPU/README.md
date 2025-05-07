[简体中文](./README.md) | [English](./README_EN.md)

# DPU例程

## 目录

- [DPU例程](#dpu例程)
  - [目录](#目录)
  - [1. 简介](#1-简介)
  - [2. 交叉编译及准备数据集](#2-交叉编译及准备数据集)
    - [2.1 程序编译](#21-程序编译)
    - [2.2 准备数据集](#22-准备数据集)
  - [3. 推理测试](#3-推理测试)
    - [3.1 参数说明](#31-参数说明)
    - [3.2 测试图片](#32-测试图片)
    - [3.3 DPU参数配置](#33-dpu参数配置)

## 1. 简介
本例程是双目深度的DPU实现。支持在BM1688/CV186X上测试。
DPU（Depth Process Unit）是BM1688/CV186X的深度处理单元：利用双目校正后的左、右图，计算出图像的视差/深度信息。具有如下两种功能。
- SGBM(Semi-Global Block Matching) :半全局块匹配算法，计算视差图。
- FGS（Fast Global Smooth）:快速全局平滑滤波，平滑视差图。视差转深度功能包含在FGS模块里。
当前支持两种处理模式： SGBM和Online(SGBM+FGS).

## 2. 交叉编译及准备数据集

### 2.1 程序编译
如果您使用SoC平台（如SE、SM系列边缘设备），刷机后在`/opt/sophon/`下已经预装了相应的libsophon、sophon-opencv和sophon-ffmpeg运行库包，可直接使用它作为运行环境。还需要一台x86主机作为开发环境，用于交叉编译C++程序。


C++程序运行前需要编译可执行文件。在x86主机上交叉编译程序，您需要在x86主机上使用SOPHON SDK搭建交叉编译环境，将程序所依赖的头文件和库文件打包至soc-sdk目录中，具体请参考[交叉编译环境搭建](../../../docs/Environment_Install_Guide.md#41-交叉编译环境搭建)。本例程主要依赖libsophon、sophon-opencv和sophon-ffmpeg运行库包。

交叉编译环境搭建好后，使用交叉编译工具链编译生成可执行文件：
```bash
cd cpp/dpu_bmcv
mkdir build && cd build
# 请根据实际情况修改-DSDK的路径，需使用绝对路径。
cmake -DTARGET_ARCH=soc -DSDK=/path_to_sdk/soc-sdk ..  
make
```
编译完成后，会在dpu_bmcv目录下生成dpu_bmcv.soc。

### 2.2 准备数据集

```bash
pip3 install dfss -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade
python3 -m dfss --url=open@sophgo.com:sophon-demo/LightStereo/KITTI12.tar.gz
tar xvf KITTI12.tar.gz && rm KITTI12.tar.gz
```

## 3. 推理测试
### 3.1 参数说明
可执行程序默认有一套参数，请注意根据实际情况进行传参，具体参数说明如下：
```bash
Usage:  dpu_demo.soc [params] 

        --dev_id (value:0)
                TPU device id
        --help (value:true)
                print help information.
        --mode (value:0)
                0: SGBM, 1: Online(SGBM+FGS)
        --input (value:KITTI12/kitti12_train194.txt)
                input path
        --output (value: results/images)
                output path
        --debug (value:0)
                0:do not save debug image, 1: save debug image
```

**注意：** CPP传参与python不同，需要用等于号，例如`./dpu_demo.soc --mode=xxx`。

### 3.2 测试图片

图片测试实例如下：
```bash
# 测试整个文件夹  
./dpu_demo.soc --input=../../KITTI12/kitti12_train194.txt
```
测试结束后，会将预测结果保存在`results/images`下，同时会打印推理时间等信息。


### 3.3 DPU参数配置

DPU支持以下参数配置，接口详情请参考算能官网--技术资料--BMCV文档。

1. SGBM参数：
```cpp
bmcv_dpu_sgbm_attrs sgbm_params;
sgbm_params.bfw_mode_en = DPU_BFW_MODE_5x5;        // 块匹配窗口大小
sgbm_params.disp_range_en = BMCV_DPU_DISP_RANGE_128; // 视差范围
sgbm_params.disp_start_pos = 0;                     // 视差起始位置
sgbm_params.dcc_dir_en = BMCV_DPU_DCC_DIR_A13;     // DCC方向
sgbm_params.dpu_census_shift = 1;                   // Census变换移位
sgbm_params.dpu_rshift1 = 0;                        // 右移1参数
sgbm_params.dpu_rshift2 = 2;                        // 右移2参数
sgbm_params.dpu_ca_p1 = 2880;                       // 代价聚合P1参数
sgbm_params.dpu_ca_p2 = 14400;                      // 代价聚合P2参数
sgbm_params.dpu_uniq_ratio = 0;                     // 唯一性比率
sgbm_params.dpu_disp_shift = 4;                     // 视差移位

dpu.setSGBMParams(sgbm_params);
```

2. FGS参数：
```cpp
bmcv_dpu_fgs_attrs fgs_params;
fgs_params.depth_unit_en = BMCV_DPU_DEPTH_UNIT_MM; // 深度单位
fgs_params.fgs_max_count = 19;                      // 最大迭代次数
fgs_params.fgs_max_t = 3;                           // 最大阈值
fgs_params.fxbase_line = 864000;                    // 基线参数

dpu.setFGSParams(fgs_params);
```

