[简体中文](./FP32BModel_Precise_Alignment.md) | [English](./FP32BModel_Precise_Alignment_EN.md)

# FP32 BModel精度对齐参考方法(Python)

## 1 前言

使用算丰平台部署深度学习模型时，我们通常需要进行模型和算法移植，并进行精度测试。如果原框架的预测结果与移植后的预测结果不一致，我们经常需要排查产生误差的原因。本文主要以Python接口为例，介绍产生精度误差的常见原因和误差排查的参考方法。

## 2 注意事项

关于精度对齐，有几点需要说明：

- 原始模型转FP32 BModel存在一定的精度误差，如果开启精度对比(cmp=True)，且精度对比通过，FP32 BModel的最大误差通常在0.001以下，不会对最终的预测结果造成影响。
- Sophon OpenCV使用TPU中的硬件加速单元进行解码，相比原生OpenCV采用了不同的upsample算法。若移植的Python代码中使用Sophon OpenCV或SAIL进行解码和前后处理，解码和前后处理的方式与原生的OpenCV存在一定差异，可能影响最终的预测结果，但通常不会对鲁棒性好的模型造成明显影响。
- debian9系统的SoC模式(如SE5盒子)默认使用Sophon OpenCV，ubuntu系统的SoC模式和PCIe模式默认使用原生OpenCV，可以通过设置环境变量使用Sophon OpenCV。
- 我们通常使用静态尺寸的bmodel，如果原程序中的模型输入尺寸是不固定的，建议修改原程序中的预处理操作，使原程序的模型输入尺寸与bmodel一致，再进行精度对齐。

## 3 精度误差常见原因

- 没有正确导出模型，导出后的模型与原始模型推理结果不一致。

- 移植的python代码使用Sophon OpenCV或sail进行解码和预处理，解码和前后处理的方式与原生OpenCV存在一定差异，可能影响最终的预测结果。可以参考步骤七进行对齐评估影响，通常不会对鲁棒性较好的模型造成明显影响，如果影响较大，可以尝试通过增加原模型的鲁棒性来减少影响。

- 移植程序resize的接口或插值方式与原程序不同，导致精度差异。比如原程序中使用`transforms.Resize`，而移植程序使用`cv2.resize`，`transforms.Resize`除了默认的双线性插值，还会进行antialiasing，与`cv2.resize`处理后的数据有较大差异。建议原程序中使用opencv进行解码和预处理。

- 移植程序中的填充策略和像素值与原程序不同，可能导致细微差别。

- 移植程序中预处理图片的通道排序(BGR或RGB)与原程序不一致，可能导致较大差异。

- 移植程序中第三方库(如opencv)的版本与原程序不一致，可能导致精度差异。

- 移植程序中标准化的参数设置错误，导致预测结果存在较大差异。比如原程序的标准化操作是`transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])`，移植程序中对应的标准化如下：

  ```python
  # 使用numpy进行标准化
  mean = np.array([0.485, 0.456, 0.406]).reshape((1, 1, 3)) * 255.0
  scale = np.array([1/0.229, 1/0.224, 1/0.225]).reshape((1, 1, 3)) * 1 / 255.0
  img = img - mean
  img = img * scale
  # 使用sail.Bmcv.convert_to进行标准化
  self.ab = [x * self.input_scale for x in [1/0.229/255.0, -0.485/0.229, 1/0.224/255.0, -0.456/0.224, 1/0.225/255.0, -0.406/0.225]]
  self.bmcv.convert_to(input_bmimg, output_bmimg, ((self.ab[0], self.ab[1]), \
                                                   (self.ab[2], self.ab[3]), \
                                                   (self.ab[4], self.ab[5])))
  ```

  **注意**：标准化之前应先对齐通道的排序是BGR还是RGB。

- 多输出模型的输出是以字典形式保存的，如果将输出字典values转list，可能出现输出顺序错误的问题，建议使用输出名称获取对应的输出数据。比如：
  ```python
  # 将输出字典values转list
  outputs_sail = list(outputs_sail.values())
  # 根据输出名称获取对应的输出数据
  outputs_sail_0 = outputs_sail['17']
  outputs_sail_1 = outputs_sail['18']
  outputs_sail_2 = outputs_sail['28']
  outputs_sail_3 = outputs_sail['29']
  ```

- 目标检测算法中的nms接口不一致，导致预测结果差异。如原程序使用`torchvision.ops.nms`，移植程序使用`cv2.dnn.NMSBoxes`，会导致细微的精度差异。

- 在使用coco2017验证数据集进行精度测试时，注意保存检测数据到JSON文件时：
  1. bbox坐标的精度是否需要保留3位浮点数并进行四舍五入操作
  2. score的精度是否需要保留5位浮点数并进行四舍五入操作

  ```python
    bbox_dict['bbox'] = [float(round(x1, 3)), float(round(y1, 3)), float(round(x2 - x1,3)), float(round(y2 -y1, 3))]
    bbox_dict['score'] = float(round(score,5))
  ```
  
## 4 精度对齐参考步骤

以CV算法模型为例，从输入到获取预测结果通常可以分为解码、预处理、网络推理、后处理四个阶段，每个阶段包含一个或多个操作。每个操作的差异都有可能造成最后预测结果的不同，因此我们需要保证各个操作的输入数据相同，对比各个操作的输出数据，来排查各个操作所产生的精度误差。具体可参考以下步骤：

**步骤一**：检查解码和前后处理的接口和参数，尽量保持移植前后各个操作的一致性，避免常见的移植问题带来精度误差。具体可参考[3 精度误差常见原因](#3-精度误差常见原因)。

**步骤二**：使用batch_size=1的FP32 BModel进行测试，如果测试结果与原程序的测试结果相差较大，则需要找出测试结果不一致的样本，作为精度对齐的测试样本。

**步骤三**：设置环境变量`BMRT_SAVE_IO_TENSORS=1`，然后使用`batch_size=1`的FP32 BModel测试步骤二挑选的样本，会在当前目录下自动保存bmodel推理前后的数据，`input_ref_data.dat.bmrt`和`output_ref_data.dat.bmrt`。

```bash
export BMRT_SAVE_IO_TENSORS=1
```

**步骤四**：使用原程序测试同一张样本，在原程序中加载`input_ref_data.dat.bmrt`，并对比原模型推理前的数据。若推理前的数据一致，可认为解码和预处理已对齐，否则需要对解码和预处理的各个操作进行对齐。以Pytorch的原程序为例：

```python
import numpy as np
# 加载input_ref_data.dat.bmrt，并转成numpy.array
# 根据bmodel的输入尺寸修改reshape参数，可通过bm_model.bin --info *.bmodel查看模型输入信息
input_bmrt_array = np.fromfile("input_ref_data.dat.bmrt", dtype=np.float32).reshape(1,3,224,224)
# 将原模型的输入转成numpy.array
input_torch_array = input_tensor.detach().numpy()
# 对比input_bmrt_array和input_torch_array的最大误差
print("input_diff_max:", np.abs(input_bmrt_array - input_torch_array).max())
```

**步骤五**：使用`input_bmrt_array`作为原模型的输入，对比输出数据，若最大误差在0.001以下，通常不会对最终的预测结果造成影响。以Pytorch的原程序为例：

```python
# 将input_bmrt_array转成torch.tensor
input_tensor = torch.from_numpy(input_bmrt_array).to(device)
# 使用原模型进行推理
output_torch_tensor = model_torch(input_tensor)
# 将原模型的输出转成numpy.array
output_torch_array = output_torch_tensor.detach().numpy()
# 加载output_ref_data.dat.bmrt，并转成numpy.array
# 根据bmodel的输出尺寸修改reshape参数，可通过bm_model.bin --info *.bmodel查看bmodel输出信息
output_bmrt_array = np.fromfile("output_ref_data.dat.bmrt", dtype=np.float32).reshape(1,8)
# 对比原模型输出和bmodel输出的最大误差
print("output_diff_max:", np.abs(output_bmrt_array - output_torch_array).max())
```

**步骤六**：使用`output_bmrt_array`在原程序中进行后处理，对比原程序与移植程序的预测结果，若预测结果不一致，则说明后处理操作存在精度误差，需要对后处理的各个操作进行对齐。

**步骤七**：根据步骤四至步骤六可初步确定解码和前后处理是否对齐，若没有对齐，可以保存移植程序中各个操作的数据，在原程序对应的操作中加载并进行对比，若输入相同但输出不一致，则该操作可能存在误差，视情况进行对齐或优化。

以对齐resize操作为例，原程序使用`transforms.Resize`，而移植程序使用`cv2.resize`，可参考以下步骤进行排查：

1.在移植程序中保存resize前后的数据。

```python
# 保存resize前的数据
np.save('img_read_cv.npy', img_read_cv)
# 移植程序中的resize操作
img_resize_cv = cv2.resize(img_read_cv, (resize_w, resize_h))
# 保存resize后的数据
np.save('img_resize_cv.npy', img_resize_cv)
```

2.在原程序使用`img_read_cv.npy`进行resize，并对比resize的结果。

```python
# 加载img_read_cv.npy
img_read_cv = np.load('img_read_cv.npy')
# 将img_read_cv转换成相应格式，并进行resize
img_read_PIL = transforms.ToPILImage()(img_read_cv)
img_resize_PIL = transforms.Resize((resize_w, resize_h))(img_read_PIL)
# 将resize后的数据转为numpy.array
img_resize_transforms = np.array(img_resize_PIL)
# 加载img_resize_cv.npy
img_resize_cv = np.load('img_resize_cv.npy')
# 对比resize操作的最大误差
print("resize_diff_max:", np.abs(img_resize_cv - img_resize_transforms).max())
```

若移植的Python代码使用sail.Decoder和sail.Bmcv，可以将sail.BMImage转成numpy.array，再进行保存和对比：

```python
# 初始化sail.Tensor，根据实际情况修改初始化参数
sail_tensor = sail.Tensor(self.handle, (h, w), sail.Dtype.BM_FLOAT32, True, True)
# 将sail.BMImage转成sail.Tensor
self.bmcv.bm_image_to_tensor(bmimage, sail_tensor)
# 将sail.Tensor转成numpy.array
sail_array = sail_tensor.asnumpy()
```

## 5 Resnet分类模型精度对齐示例
**环境准备：**

- pytorch原代码运行环境：需安装opencv-python、torchvision>=0.11.2、pytorch>=1.10.1
- 移植代码运行环境：在PCIe模式下使用官网所提供的Docker镜像和SophonSDK，并安装Sophon Inference。

**相关文件可通过以下方式下载：**
```bash
pip3 install dfss -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade
python3 -m dfss --url=open@sophgo.com:sophon-demo/common/bmodel_align.zip
unzip bmodel_align.zip
```
**`bmodel_align`文件夹中的各个文件介绍如下：**
```bash
├── resnet_torch.py #pytorch源代码
├── resnet_torch_new.py #用于加载和对比数据的源代码
├── resnet_sail.py #对齐前的移植代码
├── resnet_sail_new.py #用于保存操作数据的移植代码
├── resnet_sail_align.py #对齐后的移植代码
├── resnet18_fp32_b1.bmodel #转换后的FP32 BModel
├── resnet18_traced.pt #trace后的原模型
└── test.jpg #测试图片
```

**复现步骤：**

- 步骤一：下载相关文件，可在`TPU-NNTC`开发环境中编译FP32 BModel模型，也可直接使用转换后的FP32 BModel。
  ```bash
  python3 -m bmnetp --model=resnet18_traced.pt --shapes=[1,3,224,224] --target="BM1684"
  ```

- 步骤二：分别运行pytorch原代码和对齐前的移植代码，发现测试结果差别较大。
  
  ```bash
  # 运行原代码
  python resnet_torch.py
  # 原代码运行结果
  INFO:root:filename: test.jpg, res: 751, score: 9.114608764648438
  # 运行对齐前的移植代码
  python3 resnet_sail.py
  # 移植代码运行结果
  INFO:root:filename: test.jpg, res: 867, score: 9.796974182128906
  ```
  
- 步骤三：修改移植代码，在移植代码中保存各个操作的数据（参考resnet_sail_new.py），设置环境变量，运行修改后的移植代码。
  
  ``` bash
  export BMRT_SAVE_IO_TENSORS=1
  python3 resnet_sail_new.py
  ```
  
  生成以下文件：
  ```bash
  img_read_cv.npy：解码后数据
  img_resize_cv.npy：resize后数据
  img_normalize_cv.npy：标准化后数据
  input_ref_data.dat.bmrt：模型推理的输入数据
  output_ref_data.dat.bmrt：模型推理的输出数据
  ```
  
- 步骤四：将以上文件拷贝至与原代码同一目录下，修改原代码，在原代码中加载移植程序生成的数据并进行对比（参考resnet_torch_new.py）。
  ```bash
  python resnet_torch_new.py
  ```
  打印对比结果如下：
  ```bash
  DEBUG:root:read aligned.
  DEBUG:root:PIL aligned.
  WARNING:root:resize unaligned!
  WARNING:root:resize_diff_max:255
  WARNING:root:normalize unaligned!
  WARNING:root:normalize_diff_max:0.9098039269447327
  WARNING:root:input unaligned!
  WARNING:root:input_diff_max:0.9098039269447327
  DEBUG:root:model infer aligned.
  DEBUG:root:res aligned.
  DEBUG:root:score aligned.
  DEBUG:root:sail: res=867, score=9.796974182128906
  INFO:root:filename: test.jpg, res: 751, score: 9.114608764648438
  ```
  
- 步骤五：从对比结果来看，图片解码、网络推理和后处理的操作基本对齐，而预处理从resize操作开始没有对齐，因此需要修改移植代码的resize操作使之对齐（详见resnet_sail_align.py）。修改内容如下：

  ```bash
  # 对齐前
  # h, w, _ = img.shape
  # if h != self.net_h or w != self.net_w:
  #     img = cv2.resize(img, (self.net_w, self.net_h))
  # 对齐后
  img = transforms.ToPILImage()(img)
  img = transforms.Resize((self.net_w, self.net_h))(img)
  img = np.array(img)
  ```

- 步骤六：运行对齐后的移植代码，保存各个操作的数据，并重复步骤四进行验证。

  ```bash
  # 运行对齐后的移植代码，保存各个操作的数据
  python3 resnet_sail_align.py
  # 对齐后的移植代码运行结果
  INFO:root:filename: test.jpg, res: 751, score: 9.114605903625488
  # 拷贝生成的数据，并运行修改后的原程序
  python resnet_torch_new.py
  # 打印对比结果
  DEBUG:root:read aligned.
  DEBUG:root:PIL aligned.
  DEBUG:root:resize aligned.
  DEBUG:root:normalize aligned.
  DEBUG:root:input aligned.
  DEBUG:root:model infer aligned.
  DEBUG:root:res aligned.
  DEBUG:root:score aligned.
  DEBUG:root:sail: res=751, score=9.114605903625488
  INFO:root:filename: test.jpg, res: 751, score: 9.114608764648438
  ```

**结果分析：**

- 误差原因：`transforms.Resize`除了默认的双线性插值，还会进行antialiasing，与`cv2.resize`处理后的数据有较大差异，导致明显的精度误差。FP32 BModel与原模型的推理结果存在细微差别，基本不会影响预测精度。
- 对齐后，移植程序和原程序的预测结果基本一致。
