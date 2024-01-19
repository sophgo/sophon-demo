[简体中文](./YOLOv34_Export_Guide.md)

# YOLOv34模型导出
## 1. 准备工作
本例程采用的YOLOv34模型导出是在Pytorch模型的生产环境下进行的，需提前根据[​YOLOv3 Pytorch开源仓库](https://github.com/bubbliiiing/yolo3-pytorch)[​YOLOv4 Pytorch开源仓库](https://github.com/bubbliiiing/yolov4-pytorch)的要求安装好Pytorch环境，准备好相应的代码和模型，并保证模型能够在Pytorch环境下正常推理运行。
> **注意**：建议使用`1.8.0+cpu`的torch版本，避免因pytorch版本导致模型编译失败。

## 2. 主要步骤
### 2.1 修改nets/yolo.py

YOLOv34模型的输出形状一般是[batch_size,3*(classes+5),h,w]，主要取决于nets/yolo.py文件中的YoloBody的forward函数。建议修改YoloBody的forward函数的最后的输出语句，将模型的输出形状修改为[batch_size,3,h,w,(classes+5)]。原始代码为：

```python
    ....
    
    def forward(self, x):
            ....
    
        #---------------------------------------------------#
        #   第三个特征层
        #   y3=(batch_size,75,52,52)
        #---------------------------------------------------#
        out2 = self.yolo_head3(P3)
        #---------------------------------------------------#
        #   第二个特征层
        #   y2=(batch_size,75,26,26)
        #---------------------------------------------------#
        out1 = self.yolo_head2(P4)
        #---------------------------------------------------#
        #   第一个特征层
        #   y1=(batch_size,75,13,13)
        #---------------------------------------------------#
        out0 = self.yolo_head1(P5)

        return out0, out1, out2
        ....
```
为了改变模型输出形状，建议修改代码为：
```python
    ....
    
    def forward(self, x):
            ....
    
        #---------------------------------------------------#
        #   第三个特征层
        #   y3=(batch_size,75,52,52)
        #---------------------------------------------------#
        out2 = self.yolo_head3(P3)
        out2 = torch.reshape(out2,[out2.shape[0], 3, int(out2.shape[1]/3), out2.shape[2], out2.shape[3]])
        #---------------------------------------------------#
        #   第二个特征层
        #   y2=(batch_size,75,26,26)
        #---------------------------------------------------#
        out1 = self.yolo_head2(P4)
        out1 = torch.reshape(out1,[out1.shape[0], 3, int(out1.shape[1]/3), out1.shape[2], out1.shape[3]])
        #---------------------------------------------------#
        #   第一个特征层
        #   y1=(batch_size,75,13,13)
        #---------------------------------------------------#
        out0 = self.yolo_head1(P5)
        out0 = torch.reshape(out0,[out0.shape[0], 3, int(out0.shape[1]/3), out0.shape[2], out0.shape[3]])

        return out2.permute(0,1,3,4,2), out1.permute(0,1,3,4,2), out0.permute(0,1,3,4,2)
        ....
```

### 2.2 导出onnx模型
如果使用tpu-mlir编译模型，则必须先将Pytorch模型导出为onnx模型。YOLOv4 Pytorch开源仓库提供了模型导出脚本`predict.py`，可以使用它导出onnx模型：
首先，需要在`predict.py`中将模式参数设置为'export_onnx'，即`mode = "export_onnx"`。还需在`yolo.py`文件中设置cuda不可用，即`"cuda"              : False`。最后运行脚本即可：

```bash
python3 predict.py
```

上述脚本会在原始pth模型所在目录下生成导出的onnx模型。
