# YOLOv5_multi_QT
  
## 1. 简介

YOLOv5_multi_QT在算能SE7上实现了低延时的视频流解码+QT显示模块，可选择性的开启YOLOv5算法模块对视频流进行实时目标检测。

main.cpp 中 #define OPEN_YOLOV5 1 即为开始YOLOv5检测，0 即为纯解码+QT显示。

## 2. 特性

* 全流程实现方式针对低延时做了特殊优化
* 使用tpu_kernel进行后处理加速，仅支持BM1684X设备
* 支持FP32、FP16(BM1684X)、INT8模型编译和推理，仅支持1batch模型
 
## 3. 准备模型和依赖库

​本例程在`scripts`目录下提供了相关模型和数据集的下载脚本`download.sh`。

您也可以自己准备模型和数据集，具体转模型方法请参考sophon-demo中sample/YOLOv5_opt示例的模型编译步骤。

```bash
# 安装unzip，若已安装请跳过，非ubuntu系统视情况使用yum或其他方式安装
sudo apt install unzip
chmod -R +x scripts/
./scripts/download.sh
```

下载的模型包括：
```
./models
├── BM1684X
│   ├── yolov5s_tpukernel_fp32_1b.bmodel   # 使用TPU-MLIR编译，用于BM1684X的FP32 BModel，batch_size=1
│   ├── yolov5s_tpukernel_fp16_1b.bmodel   # 使用TPU-MLIR编译，用于BM1684X的FP16 BModel，batch_size=1
│   └── yolov5s_tpukernel_int8_1b.bmodel   # 使用TPU-MLIR编译，用于BM1684X的INT8 BModel，batch_size=1
└── onnx
    └── yolov5s_tpukernel.onnx             # 导出的onnx动态模型       
```

下载的tools和依赖库包括：
```
./tools
├── install     # sophon-qt 的 qmake编译工具
├── lib         # 低延时版本的SDK库
└── tpu_kernel_module  # 后处理加速依赖库，demo中自带，请勿删除。  
```

## 4. 例程测试

- [C++例程](./cpp/README.md)

## 5. FAQ

其他问题请参考[FAQ](../../docs/FAQ.md)查看一些常见的问题与解答。