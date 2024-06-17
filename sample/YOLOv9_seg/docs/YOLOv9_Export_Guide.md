# YOL0v9模型导出
## 1. 准备工作
YOL0v9模型导出是在Pytorch模型的生产环境下进行的，需提前根据[​YOLOv9官方开源仓库](https://github.com/WongKinYiu/yolov9)的要求安装好Pytorch环境，准备好相应的代码和模型，并保证模型能够在Pytorch环境下正常推理运行。本例程导出环境版本为：`torch==2.0.1+cpu, onnx==1.15.0`。

## 2. 导出onnx模型
如果使用tpu-mlir编译模型，则必须先将Pytorch模型导出为onnx模型。YOL0v9官方仓库提供了模型导出接口，可以直接使用它导出onnx模型：

```bash
python3 export.py --weights yolov9-c-seg-converted.pt --batch-size 1 --include onnx # 导出batch_size=1的ONNX模型
```

上述脚本会在原始pt模型所在目录下生成导出的onnx模型`yolov9-c-seg-converted.onnx`。

## 4. 常见问题
TODO
