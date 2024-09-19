# YOLOv5_fuse_multi_QT
  
## 1. 简介

YOLOv5_fuse_multi_QT 是在例程 [YOLOv5_multi](../YOLOv5_multi/README.md) 实现 pipeline 的基础上，进一步适配了 [YOLOv5_fuse](../../sample/YOLOv5_fuse/README.md) 模型，并接入 QT 显示模块（显示模块参考了例程 [YOLOv5_multi_QT](../YOLOv5_multi_QT/README.md)），从而在算能SE9系列上实现了低延时的视频流解码+QT显示功能。

## 2. 特性
* 支持 SE9-16/SE9-8
* 全流程实现方式针对低延时做了特殊优化
* 支持FP32、FP16、INT8模型编译和推理
* 解码、前处理、推理、后处理通过pipeline的形式实现
* SDK 版本 V1.8
 
## 3. 准备模型和依赖库

​本例程在`scripts`目录下提供了相关模型、数据集以及公版QT库（5.14.2）的下载脚本`download.sh`。

您也可以自己准备模型和数据集，具体转模型方法请参考 sophon-demo/sample 中例程 [YOLOv5_fuse](../../sample/YOLOv5_fuse/README.md#4-模型编译) 模型编译步骤。

```bash
# 安装unzip，若已安装请跳过，非ubuntu系统视情况使用yum或其他方式安装
sudo apt install unzip
chmod -R +x scripts/
./scripts/download.sh
```

执行下载脚本后，YOLOv5_fuse_multi_QT例程文件结构如下：
```
.
├── cpp
│   ├── install                             # 解压后的公版QT库（5.12.8）
│   ├── README.md                           # C++ 例程文档
│   ├── run_hdmi_show.sh                    # 例程执行脚本
│   ├── workflow.png                        # 程序流程图
│   └── yolov5_bmcv                         # C++ bmcv 例程 
├── datasets
│   ├── coco
│   ├── coco128
│   ├── coco.names                          # 类别名
│   ├── test
│   └── test_car_person_1080P.mp4           # 测试视频
├── models                                  # BModel 模型
│   ├── BM1688
│   └── CV186X
├── README.md                               # C++ BMCV 例程文档
└── scripts
    └── download.sh                         # 数据、模型、QT库下载脚本
```


## 4. 例程测试

- [C++例程](./cpp/README.md)

## 5. FAQ

其他问题请参考[FAQ](../../docs/FAQ.md)查看一些常见的问题与解答。