# Real-ESRGAN模型导出

## 1.准备源模型

从https://github.com/xinntao/Real-ESRGAN获取你想要的模型。

## 2.安装依赖

建议在tpu-mlir提供的sophgo/tpuc_dev:v3.2以后版本的docker中进行模型导出，这样只需要安装basicsr。

```bash
pip3 install basicsr==1.4.2
```

## 3.导出onnx

本例程基于源仓库的导出脚本进行了一些修改，使之能导出`realesr-general-x4v3`版本onnx，使用示例参考：

```bash
python3 tools/pytorch2onnx.py --input xxx.pth --output xxx.onnx --batch_size 1
```

如果你使用的源模型与本例程的不同，您可能需要对`tools/pytorch2onnx.py`进行修改。