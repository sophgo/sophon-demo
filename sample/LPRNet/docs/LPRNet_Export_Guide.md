# LPRNet模型导出
## 1. 准备工作
LPRNet模型导出是在Pytorch模型的生产环境下进行的，需提前根据[​LPRNet官方开源仓库](https://github.com/sirius-ai/LPRNet_Pytorch)的要求安装好Pytorch环境，准备好相应的代码和模型，并保证模型能够在Pytorch环境下正常推理运行。

## 2. 主要步骤
### 2.1 修改LPRNet的maxpool_3d

目前工具链暂不支持maxpool_3d，可用maxpool_2d实现maxpool_3d。修改方式可参考tools/LPRNet.py


### 2.2 导出onnx模型
如果使用tpu-mlir编译模型，则必须先将Pytorch模型导出为onnx模型。导出模型的方法可参考tools/export_onnx.py。

## 3. 常见问题
TODO