# SuperGlue模型导出

SuperGlue算法分为superpoint和superglue两个模型，`tools/`目录提供了两个模型的导出脚本，然后进入`tools/`目录。
```bash
cd tools
```

需要安装torch和onnxruntime的依赖：
```bash
pip3 install torch onnxruntime
```

# superpoint导出

由于mlir暂不支持superpoint simple_nms之后的动态输出部分，本例程对源码的`superpoint.py`做了一些修改，您可以直接运行该命令导出`superpoint_to_nms.onnx`
```bash
python3 export_superpoint.py
```

# superglue导出

由于mlir不支持某些算子，本例程在输出不变的前提下对源码的`superglue.py`做了一些修改，您可以直接运行该命令导出`superglue_indoor_iter20_1024.onnx`
```bash
python3 export_superglue.py
```