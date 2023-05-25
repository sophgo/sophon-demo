# centernet模型导出
## 1. 准备工作
centernet模型导出是在Pytorch模型的生产环境下进行的，需提前根据[​centernet官方开源仓库](https://github.com/xingyizhou/CenterNet)的要求安装好Pytorch环境，准备好相应的代码和模型，并保证模型能够在Pytorch环境下正常推理运行。
**注意**：建议使用`1.8.0+cpu`的torch版本，避免因pytorch版本导致模型编译失败。

## 2. 主要步骤
### 2.1 dlav0.py网络修改说明
tools/目录下dlav0.py，是从[CenterNet源码](https://github.com/xingyizhou/CenterNet)中，修改dlav0.py中DLASeg类forward方法的返回值后得到的。
```python
#return [ret]
return torch.cat((ret['hm'], ret['wh'], ret['reg']), 1) 
```
将heatmap, wh, reg三个head的特征图concat到一起，方便后续bmodel的转换


### 2.2 导出torchscript模型
​Pytorch模型在编译前要经过`torch.jit.trace`，trace后的模型才能使用tpu-nntc编译BModel。tools中提供了`export_torchscript.py`，可以直接使用它导出torchscript模型。

**注意：** 导出的模型建议以`.pt`为后缀，以免在后续模型编译量化中发生错误。

### 2.3 导出onnx模型
如果使用tpu-mlir编译模型，则必须先将Pytorch模型导出为onnx模型。tools中提供了模型导出脚本`export_onnx.py`，可以直接使用它导出onnx模型。


## 3. 常见问题
TODO
