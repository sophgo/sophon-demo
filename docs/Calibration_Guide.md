[简体中文](./Calibration_Guide.md) | [English](./Calibration_Guide_EN.md.md)

# 模型量化
更多模型量化教程请参考
《TPU-MLIR开发参考手册》和《TPU-NNTC开发参考手册》的“模型量化”(请从[算能官网](https://developer.sophgo.com/site/index/material/28/all.html)相应版本的SDK中获取)。

## 1. 注意事项
### 1.1 量化数据集
建议从训练集随机抽取100~500张样本作为量化数据集，量化数据集应尽量涵盖测试场景和类别，量化时可尝试不同的iterations进行量化以获得最优的量化精度。

### 1.2 前处理对齐
量化数据集的预处理应该和推理测试的预处理保持一致，否则会导致较大的精度损失。在mlir制作npz/npy数据集、nntc制作lmdb量化数据集时，应当预先完成数据的预处理。

### 1.3 特定模型优化技巧
#### 1.3.1 YOLO系列模型
由于yolo系列的输出既有分类又有回归，致使输出在统计学上的分布不均匀，所以通常不量化最后三个conv层及其之后的所有层，有时候最开始的几个conv层也不量化，具体效果如何需要实际操作下。

MLIR具体步骤如下：
1. 可以先用mlir2onnx.py这个工具，将model_transform生成的mlir文件转化成onnx，然后通过netron查看onnx网络结构。
```
mlir2onnx.py -m xxx.mlir -o xxx.onnx
```
2. 使用fp_forward.py生成qtable，--fpfwd_outputs、--fpfwd_inputs功能与以前nntc一样，指定层名即可将对应的所有层指定对应的fp_type。类似如下命令：
```
fp_forward.py yolov7_1b.mlir --fpfwd_outputs 1376_Conv,1046_Transpose --chip bm1684 --fp_type F32 -o yolov7_qtable
```
3. 生成的qtable传给model_deploy.py，配合加入test_input和test_reference来验证混精度策略是否有效。

NNTC具体步骤如下：
1. 生成fp32 umodel的prototxt文件；
2. 用Netron打开fp32 umodel的prototxt文件，选择后面三个branch（大目标，中目标，小目标）的conv层，记下名字；
3. 在分步量化或一键量化中通过--fpfwd_outputs指定步骤2所获得的conv层。可通过--help查看具体方法或参考YOLOv5的量化脚本。

## 2. 常见问题
### 2.1 量化后检测框偏移严重
在以上注意事项都确认无误的基础上，尝试不同的门限策略th_method，推荐ADMM, MAX,PERCENTILE9999。

### 2.2 NNTC量化过程中精度对比不通过导致量化中断
NNTC相关报错：
```bash
w0209 14:47:33.992739 3751 graphTransformer.cpp:515] max diff = 0.00158691 max diff blob id : 4 blob name : out put
Fail: only one compare!
```
原因：量化过程中fp32精度对比超过设定阈值(默认值为0.001)。
解决办法：修改fp32精度对比阈值，如-fp32_diff=0.01。
