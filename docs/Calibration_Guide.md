# 模型量化
更多模型量化教程请参考《TPU-NNTC开发参考手册》的“模型量化”(请从[算能官网](https://developer.sophgo.com/site/index/material/28/all.html)相应版本的SDK中获取)。

## 1. 注意事项
### 1.1 量化数据集
建议从训练集随机抽取100~500张样本作为量化数据集，量化数据集应尽量涵盖测试场景和类别，量化时可尝试不同的iterations进行量化以获得最优的量化精度。

### 1.2 前处理对齐
量化数据集的预处理应该和推理测试的预处理保持一致，否则会导致较大的精度损失。建议在制作lmdb量化数据集时，通过convert_imageset.py完成数据的预处理。

### 1.3 特定模型优化技巧
#### 1.3.1 YOLO系列模型
由于yolo系列的输出既有分类又有回归，致使输出在统计学上的分布不均匀，所以通常不量化最后三个conv层及其之后的所有层。具体步骤如下：
1. 生成fp32 umodel的prototxt文件；
2. 用Netron打开fp32 umodel的prototxt文件，选择后面三个branch（大目标，中目标，小目标）的conv层，记下名字；
3. 在分步量化或一键量化中通过--fpfwd_outputs指定步骤2所获得的conv层。可通过--help查看具体方法或参考YOLOv5的量化脚本。

## 2. 常见问题
### 2.1 量化后检测框偏移严重
在以上注意事项都确认无误的基础上，尝试不同的门限策略th_method，推荐ADMM, MAX,PERCENTILE9999。

### 2.2 量化过程中精度对比不通过导致量化中断
相关报错：
```bash
w0209 14:47:33.992739 3751 graphTransformer.cpp:515] max diff = 0.00158691 max diff blob id : 4 blob name : out put
Fail: only one compare!
```
原因：量化过程中fp32精度对比超过设定阈值(默认值为0.001)。
解决办法：修改fp32精度对比阈值，如-fp32_diff=0.01。