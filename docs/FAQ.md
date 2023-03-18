# 常见问题解答

## 目录

* [1 环境安装相关问题](#1-环境安装相关问题)
* [2 模型导出相关问题](#2-模型导出相关问题)
* [3 模型编译和量化相关问题](#3-模型编译和量化相关问题)
* [4 算法移植相关问题](#4-算法移植相关问题)
* [5 精度测试相关问题](#5-精度测试相关问题)
* [6 性能测试相关问题](#6-性能测试相关问题)
* [7 其他问题](#7-其他问题)

我们列出了一些用户和开发者在开发过程中会遇到的常见问题以及对应的解决方案，如果您发现了任何问题，请随时联系我们或创建相关issue，非常欢迎您提出的任何问题或解决方案。

## 1 环境安装相关问题
### 1.1 sophon-demo涉及的开发和运行环境安装可以参考[相关文档](./Environment_Install_Guide.md)。

### 1.2 SoC模式下如何使用SD卡刷更新固件
下载解压v22.09.02以后的SophonSDK，找到sophon-img文件夹下的sdcard刷机包，并参考[相关文档](https://doc.sophgo.com/docs/3.0.0/docs_latest_release/faq/html/devices/SOC/soc_firmware_update.html#id6)刷新固件。

## 2 模型导出相关问题
### 2.1 Pytorch模型如何进行jit.trace
部分开源仓库提供了模型导出的脚本，如果没有则需要参考[相关文档](./torch.jit.trace_Guide.md)或者[Pytorch官方文档](https://pytorch.org/docs/stable/jit.html)编写jit.trace脚本。jit.trace通常是在模型开发(torch)环境中进行，不需要安装SophonSDK。
- YOLOv5模型jit.trace的方法可参考[相关文档](../sample/YOLOv5/docs/YOLOv5_Export_Guide.md)

### 2.2 ONNX模型导出过程使用的算子集
如果pytorch版本过低，可能无法支持高版本ONNX算子集，导致ONNX输出失败。
需要根据当前环境使用pytorch版本，设置支持的ONNX算子集版本。以设置使用ONNX算子集版本13为例，设置方式如下：
```python
torch.onnx.export(
    ...
    opset_version=13,
    ...
)
```

## 3 模型编译和量化相关问题
### 3.1 使用TPU-NNTC量化的相关问题
详见[相关文档](./Calibration_Guide.md)。

### 3.2 使用TPU-NNTC编译大模型(如resnet260)耗时长
resnet260编译会这么耗时有可能是二次搜索时间太长的原因。我们在做编译优化的时候对已经做完初次layergroup后又结果做一次再分组搜索，这个搜索结果对于resnet这类网络性能收益不大，反倒会增加编译优化的时间。为了解决这类模型问题，我们增加了二次搜索的上限值，通过以下环境变量来控制二次搜索的上限：
```bash
export BMCOMPILER_GROUP_SEARCH_COUNT=1
```

## 4 算法移植相关问题
### 4.1 Sophon OpenCV和原生OpenCV、BMCV的关系
Sophon OpenCV继承和优化了原生OpenCV，修改了原生mat，增加了设备内存，部分接口（如imread，videocapture，videowriter，resize，convert等）增加了硬件加速支持，保持了跟原生OpenCV操作的接口一致性，修改内容具体信息请查看[相关文档](https://doc.sophgo.com/sdk-docs/v22.12.01/docs_latest_release/docs/sophon-mw/guide/html/1_guide.html)。

BMCV是我们提供的一套基于硬件VPP和TPU进行图像处理以及部分数学运算的加速库，是C接口的库。在我们修改的Sophon OpenCV底层，相关操作的硬件加速调用的也是BMCV的接口。

Sophon OpenCV使用芯片中的硬件加速单元进行解码，相比原生OpenCV采用了不同的upsample算法，解码和前后处理的方式与原生的OpenCV存在一定差异，可能影响最终的预测结果，但通常不会对鲁棒性好的模型造成明显影响。

### 4.2 基于OpenCV的Python例程如何调用Sophon OpenCV进行加速
使用v22.09.02以后的SophonSDK，不管是PCIe，还是SoC模式，基于OpenCV的Python例程默认都使用安装的原生OpenCV，可以通过设置环境变量使用Sophon OpenCV：
```bash
export PYTHONPATH=$PYTHONPATH:/opt/sophon/sophon-opencv-latest/opencv-python/
```
注意使用sophon-opencv可能会导致推理结果的差异。

### 4.3 基于OpenCV的C++例程使用Sophon OpenCV还是原生OpenCV
基于OpenCV的C++例程在CMakeLists.txt中设置了OpenCV_DIR路径，链接Sophon OpenCV的相关头文件和库文件。如果需要调用原生OpenCV，需自行安装原生OpenCV，并修改相关链接路径。

### 4.4 C++例程使用ff_decode解码视频提示“vid reset unlock flock failed”，但不影响正常运行
这是由于缓存没有及时清除导致的，不影响推理结果。可以通过运行如下命令解决：
```bash
sudo rm /tmp/vid_*
```

### 4.5 C++例程使用ff_decode解码，输入图片的后缀是大写时报"not support pic format, only support jpg and png"
目前的解决方案是将输入图片的后缀改小写，后续将直接根据图像输入来判断图像格式，而非根据名称来判断。

## 5 精度测试相关问题
### 5.1 FP32 BModel的推理结果与原模型的推理结果不一致
在前后处理与原算法对齐的前提下，FP32 BModel的精度与原模型的最大误差通常在0.001以下，不会对最终的预测结果造成影响。FP32 BModel精度对齐的方法可以参考[相关文档](./FP32BModel_Precise_Alignment.md)。

## 6 性能测试相关问题
### 6.1 部分FP32 BModel的BM1684X性能低于BM1684
BM1684X的local memory相比BM1684少了一半，参数量较大的模型有可能放不下，导致gdma访问ddr次数大量增加。

### 6.2 基于opencv-python的例程int8bmodel推理时间没有比fp32bmodel快
int8bmodel的输入层数据类型是int8，scale不等1，基于opencv-python的例程以numpy.array为输入，推理接口内部需要进行乘scale操作，而fp32bmodel输入层的scale是1，推理接口内部不需要进行乘scale操作，这部分时间可能会抵掉模型推理优化的时间。可以在代码中添加`sail.set_print_flag(1)`，打印推理接口的具体耗时。  
如果要使用opencv-python例程进行部署，建议将int8bmodel的输入、输出层保留为浮点计算。

## 7 其他问题
