# LightStereo ONNX导出

注：导出onnx需要依赖cuda，如果您没有cuda，可以下载我们提供的onnx，也可以自己更改导出脚本。

本例程将OpenStereo仓库中LightStereo的modeling部分单独摘了出来，存放在`tools/core`目录下。并且提供了导出脚本`export_onnx.py`。它的使用方法如下：
```bash
cd tools
python3 export_onnx.py --loadckpt ../models/ckpt/LightStereo-S-SceneFlow.ckpt --onnx_path ../models/onnx/LightStereo-S-SceneFlow.onnx
```
`--loadckpt`对应的文件请自行从OpenStereo源仓库获取，如果您使用不同版本的模型，您还需要更改其他参数，具体请查看`export_onnx.py`脚本中的参数定义。

运行完成后，`--onnx_path`指定的目录下会生成LightStereo-S-SceneFlow.onnx，用户可以使用该onnx进行bmodel编译。