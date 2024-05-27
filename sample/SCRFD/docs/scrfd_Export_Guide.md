[简体中文](./scrfd_Export_Guide.md)

# scrfd模型导出
## 1. 准备工作
scrfd模型导出是在mmdetection模型的生产环境下进行的，需提前根据[​scrfd官方开源仓库](https://github.com/deepinsight/insightface/tree/master/detection/scrfd)的要求安装好mmdetection环境，准备好相应的代码和模型，并保证模型能够在mmdetection环境下正常推理运行。

## 2. 模型转出
运行路径是insightface/detection/scrfd，并使用mmdetection环境:

```bash
python ./tools/scrfd2onnx.py \
./configs/scrfd/scrfd_10g_bnkps.py \ #配置文件
./runs/scrfd_10g_bnkps/latest.pth \ #pth模型
--shape 640 640 \ #输入图片的shape，H*W，需要按照实际情况来设置
--output-file ./onnx/scrfd.onnx #保存为scrfd.onnx,放在路径onnx下
```

## 多batch模型导出
如果您想导出4batch的模型，请修改
scrfd2onnx.py的源文件的第71行，为 `0: '4'` ， 如下：
```python
    if dynamic:
        dynamic_axes = {out: {0: '?', 1: '?'} for out in output_names}
        dynamic_axes[input_names[0]] = {
            0: '4',
            2: '?',
            3: '?'
        }
```
还需要修改第164行为，`dynamic = True` , 在170添加多batch参数如下：
```python
    dynamic = True
    if input_shape[2]<=0 or input_shape[3]<=0:
        input_shape = (1,3,640,640)
        dynamic = True
        #simplify = False
        print('set to dynamic input with dummy shape:', input_shape)
    else:
        input_shape = (4, 3, 640, 640)  # 这里将批次大小设置为4
```