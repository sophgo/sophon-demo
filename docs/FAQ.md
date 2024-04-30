[简体中文](./FAQ.md) | [English](./FAQ_EN.md)

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
**如果您使用TPU-NNTC来编译BModel，注意在TPU-NNTC支持的Pytorch版本`1.8.0+cpu`的环境下进行torchscript模型导出。**
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

Sophon OpenCV使用TPU中的硬件加速单元进行解码，相比原生OpenCV采用了不同的upsample算法，解码和前后处理的方式与原生的OpenCV存在一定差异，可能影响最终的预测结果，但通常不会对鲁棒性好的模型造成明显影响。

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

### 4.5 bm_free_device失败
可能的原因：
1. 这块device_mem没有被分配过内存，或者已经被释放过。

2. 如果一个bm_image的内存是attach过来的，注意只调用detach函数就好，不要调用bm_image_free_contiguous_mem释放它，否则后续那块device_mem会释放失败。

## 5 精度测试相关问题
### 5.1 FP32 BModel的推理结果与原模型的推理结果不一致
在前后处理与原算法对齐的前提下，FP32 BModel的精度与原模型的最大误差通常在0.001以下，不会对最终的预测结果造成影响。FP32 BModel精度对齐的方法可以参考[相关文档](./FP32BModel_Precise_Alignment.md)。

## 6 性能测试相关问题
### 6.1 部分FP32 BModel的BM1684X性能低于BM1684
BM1684X的local memory相比BM1684少了一半，参数量较大的模型有可能放不下，导致gdma访问ddr次数大量增加。

### 6.2 基于opencv-python的例程int8bmodel推理时间没有比fp32bmodel快
int8bmodel的输入层数据类型是int8，scale不等1，基于opencv-python的例程以numpy.array为输入，推理接口内部需要进行乘scale操作，而fp32bmodel输入层的scale是1，推理接口内部不需要进行乘scale操作，这部分时间可能会抵掉模型推理优化的时间。可以在代码中添加`sail.set_print_flag(1)`，打印推理接口的具体耗时。  
如果要使用opencv-python例程进行部署，建议将int8bmodel的输入、输出层保留为浮点计算。

### 6.3 sophon-demo能否用于性能压测
不建议。sophon-demo提供一系列主流算法的移植例程，用户可以根据sophon-demo进行模型算法移植和精度测试。但sophon-demo的前处理/推理/后处理是串行的，即使开多个进程也很难将TPU的性能充分发挥出来。[sophon-pipeline](https://github.com/sophgo/sophon-pipeline)能够将前处理/推理/后处理分别运行在3个线程上，最大化的实现并行，因此建议使用sophon-pipeline进行性能压测。

### 6.4 测试性能相比README下降
如果性能下降超过10%，需要确认产品型号，sophon-demo的各个例程都是基于标准版的产品（如SE-16）来测试的，如果您使用的是低配版产品（如SE5-8），性能下降是正常的，SE5-16的int8算力是17.6TOPS，SE5-8是10.6TOPS，所以大概会有2/5的性能损失。
如果您使用的产品也是标准版，也遇到了性能下降的问题，可以将问题反馈给算能工作人员或者在github上创建issue。

### 6.5 性能测试表格的作用
sophon-demo提供的性能测试表格一般包含解码、前处理、推理、后处理这四个部分，业界对于算法部署也大多是这么划分的。对于算能的AI处理器而言，这四个部分都是可以并发的，因为它们各自依赖的器件不同，比如解码依赖VPU、前处理依赖VPP、推理依赖TPU、后处理则依赖中央处理器，这些元器件是可以并发工作且不会受到各自影响的。sophon-demo提供的性能测试表格将这些信息列出来，可以直观地分析出当前算法的瓶颈以及它能达到的理论最高每秒处理次数。

## 7 其他问题
### 7.1 编译时出现`Unkown CMake command "add_compile_definitions".`
这是你的cmake没有add_compile_definition这个函数，可以修改成add_definitions，或者升级cmake到3.12之后。

### 7.2 有些图片的数据重叠到一起，有的图片框上面没数据，有些图片没有框
图片和视频测试结果存在少数的漏检、误检，这是正常的，sophon-demo的例程只需要保证移植后的模型和源模型的精度可以对齐就行了，判断模型效果应当以精度指标测试为准。

### 7.3 程序运行时出现`bm_ion_alloc failed`等报错
首先排查程序是否出现设备内存泄漏，代码中是否存在分配设备内存没有及时释放的问题。如果确认没有内存泄漏，那么可能是由于某个heap太小导致的，可以观察一下设备内存各个heap的使用情况：
```bash
#在SE5/SE7系列上使用以下命令查看：
sudo cat /sys/kernel/debug/ion/bm_vpu_heap_dump/summary
sudo cat /sys/kernel/debug/ion/bm_vpp_heap_dump/summary
sudo cat /sys/kernel/debug/ion/bm_npu_heap_dump/summary
#在SE9上使用以下命令查看：
sudo cat /sys/kernel/debug/ion/cvi_vpp_heap_dump/summary
sudo cat /sys/kernel/debug/ion/cvi_npu_heap_dump/summary
#会打印类似这种信息
Summary:
[1] vpp heap size:4294967296 bytes, used:144646144 bytes
usage rate:4%, memory usage peak 144646144 bytes #memory usage peak是该heap的内存使用峰值
```
如果某个heap的`memory usage peak`已经接近`heap size`了，那么可以考虑用这个工具调整设备内存各个heap的大小：[sophon内存修改工具](https://doc.sophgo.com/sdk-docs/v23.07.01/docs_latest_release/docs/SophonSDK_doc/zh/html/appendix/2_mem_edit_tools.html)。

### 7.4 刚开机首次执行某个函数慢(比如解码)，重启进程再次运行程序，时间正常

可能是因为文件还没有缓存到内存中导致的刚开始比较慢。
可以做个验证，如果不重启可复现，就说明是刚开机运行程序比较慢，是文件还没缓存的原因，步骤如下：
    
1. 上电后执行程序，第一次执行慢，第二次执行正常。

2. 之后进入root用户清除cache，运行命令`echo 3 > /proc/sys/vm/drop_caches`。
    
3. 再次执行程序，运行慢，即可确定是cache导致的。

### 7.5 运行下载脚本`scripts/download.sh`报错

可能有以下原因：

1. 网络环境不好或者有防火墙。

2. 一般下载脚本会最先安装dfss，如果是非ubuntu系统出现dfss安装失败，中间可能会打印类似`ERROR: Failed building wheel for cff`这种报错，这是因为缺少libffi-devel这个依赖，可以通过以下命令安装：

    ```bash
    sudo yum install libffi-devel
    #然后重新运行下载脚本
    ```
    