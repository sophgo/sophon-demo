# DWA 例程


- [DWA 例程](#dwa-例程)
  - [1. 说明](#1-说明)
  - [2. 相机标定，数据准备](#2-相机标定数据准备)
  - [3. 样例测试](#3-样例测试)


## 1. 说明

DWA 是算能BM1688/CV186AH上的硬件去畸变仿射模块；具有几何畸变校正功能，通过校正镜头引起的图像畸变（针对桶形畸变 (Barrel Distortion) 及枕形畸变 (Pincushion Distortion) ），使图像中的直线变得更加准确和几何正确，提高图像的质量和可视化效果。

本例程是调用 bmcv_dwa_gdc 接口的示例，接口的具体表述请参考算能官网--技术资料--BM1688/CV186AH的BMCV手册。

## 2. 相机标定，数据准备
相机标定可以参考 https://github.com/sophgo/sophon-stream/blob/master/samples/dwa_dpu_encode/Calibration.md.

本demo也提供了标定测试图片及标定后的参数文件:

```bash
# 下载文件
pip3 install dfss -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade
python3 -m dfss --url=open@sophgo.com:/sophon-demo/DWA/data.zip

# 安装unzip，若已安装请跳过，非ubuntu系统视情况使用yum或其他方式安装
sudo apt-get install unzip
# 解压文件
unzip data.zip
```

`data`目录如下。

.
├── left            # 测试组1 
│   ├── left.jpg
│   └── LL.dat
└── right           # 测试组2
    ├── right.jpg
    └── RR.dat

## 3. 样例测试

- [C++例程](./cpp/README.md)

