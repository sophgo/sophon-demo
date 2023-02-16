# LPRNet模型导出
## 1. 准备工作
LPRNet模型导出是在Pytorch模型的生产环境下进行的，需提前根据[​LPRNet官方开源仓库](https://github.com/sirius-ai/LPRNet_Pytorch)的要求安装好Pytorch环境，准备好相应的代码和模型，并保证模型能够在Pytorch环境下正常推理运行。

## 2. 主要步骤
### 2.1 修改LPRNet的maxpool_3d

目前工具链暂不支持maxpool_3d，可用maxpool_2d实现maxpool_3d。修改方式可参考tools/LPRNet.py


### 2.2 导出torchscript模型
​Pytorch模型在编译前要经过`torch.jit.trace`，trace后的模型才能使用tpu-nntc编译BModel。使用以下方式导出torchscript模型：

```python

model = LPRNet.build_lprnet(class_num=68)
model.load_state_dict(
    torch.load(
        "../models/torch/Final_LPRNet_model.pth", map_location=torch.device("cpu")
    )
)
model.eval()
input_var = torch.zeros([1, 3, 24, 94], dtype=torch.float32)
traced_script_module = torch.jit.trace(model, input_var)
traced_script_module.save(save_script_pt)  # save_script_pt为保存路径

```

上述脚本会按照save_script_pt路径生成导出的torchscript模型。


### 2.3 导出onnx模型
如果使用tpu-mlir编译模型，则必须先将Pytorch模型导出为onnx模型。导出模型的方法可参考tools/export_onnx.py。

## 3. 常见问题
TODO