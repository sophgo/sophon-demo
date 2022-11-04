# CenterNet

## 目录

[TOC]

## 1. 简介

CenterNet 是一种 anchor-free 的目标检测网络，不仅可以用于目标检测，还可以用于其他的一些任务，如姿态识别或者 3D 目标检测等等。

**文档:** [CenterNet论文](https://arxiv.org/pdf/1904.07850.pdf)

**参考repo:** [CenterNet](https://github.com/xingyizhou/CenterNet)



## 2. 数据集

[MS COCO](http://cocodataset.org/#home)，是微软构建的一个包含分类、检测、分割等任务的大型的数据集。使用[CenterNet](https://github.com/xingyizhou/CenterNet)基于COCO Detection 2017预训练好的80类通用目标检测模型。

## 3. 准备环境与数据


### 3.1 准备开发环境

开发环境是指用于模型转换或验证以及程序编译等开发过程的环境，目前只支持x86，需要使用我们提供的基于Ubuntu18.04的docker镜像。

运行环境是具备Sophon设备的平台上实际使用设备进行算法应用部署的环境，有PCIe加速卡、SM5模组、SE5边缘计算盒子等，所有运行环境上的bmodel都是一样的，SDK中各模块的接口也是一致的。

开发环境与运行环境可能是统一的（如插有SC5加速卡的x86主机，既是开发环境又是运行环境），也可能是分离的（如使用x86主机作为开发环境转换模型和编译程序，使用SE5盒子部署运行最终的算法应用）。

但是，无论使用的产品是SoC模式还是PCIe模式，都需要一台x86主机作为开发环境，模型的转换工作必须在开发环境中完成。

#### 3.1.1 开发主机准备：

- 开发主机：一台安装了Ubuntu16.04/18.04/20.04的x86主机，运行内存建议12GB以上
- 软件环境：libsophon、sophon-mw、sophon-sail，可以通过[算能官网](https://developer.sophgo.com/site/index/material/21/all.html)下载安装对应版本

### 3.2 准备模型

在后续步骤中使用scripts/路径下download.sh脚本下载所需的pt模型。

> **注意：**本示例展示的是使用CenterNet进行目标检测。由于工具链目前对DeformConv可变卷积还未支持，所以选用dlav0作为主干网, 下载对应的pt文件。



#### 3.2.1 JIT环境准备
SophonSDK中的PyTorch模型编译工具BMNETP只接受PyTorch的JIT模型（TorchScript模型）。

JIT（Just-In-Time）是一组编译工具，用于弥合PyTorch研究与生产之间的差距。它允许创建可以在不依赖Python解释器的情况下运行的模型，并且可以更积极地进行优化。在已有PyTorch的Python模型（基类为torch.nn.Module）的情况下，通过torch.jit.trace就可以得到JIT模型，如`torch.jit.trace(python_model, torch.rand(input_shape)).save('jit_model')`。BMNETP暂时不支持带有控制流操作（如if语句或循环）的JIT模型，因此不能使用torch.jit.script，而要使用torch.jit.trace，它仅跟踪和记录张量上的操作，不会记录任何控制流操作。可在源码导入CPU模型后通过添加以下代码导出符合要求的JIT模型：

```bash
# 下载dlav0作为主干网的预训练模型
cd sample/CenterNet/scripts/
./download.sh
# download.sh 同时会下载原始模型，转换及量化后的bmodel，量化数据集以及测试图片
# 原始模型下载成功后，文件位于../data/models/torch/ctdet_coco_dlav0_1x.pth
```

#### dlav0.py网络修改说明
tools/目录下dlav0.py，是从[CenterNet源码](https://github.com/xingyizhou/CenterNet)中，修改dlav0.py中DLASeg类forward方法的返回值后得到的。
```python
#return [ret]
return torch.cat((ret['hm'], ret['wh'], ret['reg']), 1) 
```
将heatmap, wh, reg三个head的特征图concat到一起，方便后续bmodel的转换


### 3.3 准备量化集

不量化模型可跳过本节。

量化集使用COCO Detection 2017的验证集
我们选取其中的200张图片进行量化

```bash
在前置步骤中使用download.sh已下载量化所需数据集
# 下载成功后，JPG文件位于../data/images文件夹中
```


## 4. 模型编译与量化

建议模型转换的过程在tpu-nntc提供的x86下的docker开发环境中完成。模型编译前需要安装TPU-NNTC(>=3.1.0)，具体可参考[tpu-nntc环境搭建](../docs/Environment_Install_Guide.md#1-tpu-nntc环境搭建)。

#### JIT模型生成
进入docker以后直接运行export.py即可
```bash
pip3 install torch torchvision
cd ../tools
python3 export.py
```
在`../data/models/torch`目录下生成了一份`ctdet_coco_dlav0_1x.torchscript.pt`文件

### 4.1 生成FP32 BModel

```bash
cd ../scripts
./gen_fp32bmodel.sh BM1684
```

在参数中指定目标名称（BM1684/BM684X），上述脚本会在`../data/models/BM1684`下生成`centernet_fp32_1b.bmodel`与`centernet_fp32_4b.bmodel`文件，即转换好的FP32 BModel，使用`tpu_model --info`查看的模型具体信息如下：

```bash
bmodel version: B.2.2
chip: BM1684
create time: Fri Sep  9 17:36:25 2022

==========================================
net 0: [ctdet_dlav0]  static
------------
stage 0:
input: input.1, [1, 3, 512, 512], float32, scale: 1, zero point: 0
output: 40, [1, 84, 128, 128], float32, scale: 1, zero point: 0

device mem size: 115830408 (coeff: 72003592, instruct: 48768, runtime: 43778048)
host mem size: 0 (coeff: 0, runtime: 0)
```

### 4.2 生成INT8 BModel

不量化模型可跳过本节。


执行以下命令，使用一键量化工具cali_model，生成INT8 BModel：

```shell
./gen_int8bmodel.sh BM1684X
```
在参数中指定目标名称（BM1684/BM684X），上述脚本会在`../data/models/BM1684X`下生成`centernet_int8_1b.bmodel`与`centernet_int8_4b.bmodel`，即转换好的INT8 BModel，使用`tpu_model --info`查看的模型具体信息如下


```bash
bmodel version: B.2.2
chip: BM1684X
create time: Fri Sep  9 10:21:22 2022

==========================================
net 0: [centernet]  static
------------
stage 0:
input: input.1, [4, 3, 512, 512], int8, scale: 60.233, zero point: 0
output: 40, [4, 84, 128, 128], float32, scale: 1, zero point: 0

device mem size: 87904264 (coeff: 18698248, instruct: 0, runtime: 69206016)
host mem size: 0 (coeff: 0, runtime: 0)
```

## 5. 例程测试

例程测试环节请参照`./cpp`以及`./python`目录下`README.md`文档进行

