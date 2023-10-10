[简体中文](./FP32BModel_Precise_Alignment.md) | [English](./FP32BModel_Precise_Alignment_EN.md)

# FP32 BModel Precision Alignment Reference Method (Python)

- [FP32 BModel Precision Alignment Reference Method (Python)](#fp32-bmodel-precision-alignment-reference-method-python)
  - [1 Introduction](#1-introduction)
  - [2 Attention Notes](#2-attention-notes)
  - [3 Common Causes of Accuracy Errors](#3-common-causes-of-accuracy-errors)
  - [4 Reference Steps for Accuracy Alignment](#4-reference-steps-for-accuracy-alignment)
  - [5 Resnet Classification Model Accuracy Alignment Examples](#5-resnet-classification-model-accuracy-alignment-examples)

## 1 Introduction

When deploying a deep learning model on the Sophon platform, it is often necessary to perform model and algorithm porting as well as accuracy testing. If the predicted results of the original framework differ from those of the ported model, it is often necessary to identify the cause of the error. This article mainly uses the Python interface as an example to introduce common causes of accuracy errors and reference methods for error troubleshooting.

## 2 Attention Notes

There are several points to note about precision alignment:

- There may be some precision errors when converting the original model to FP32 BModel. If the precision comparison (cmp=True) is enabled and passes, the maximum error of the FP32 BModel is usually less than 0.001, which will not affect the final prediction results.
- Sophon OpenCV uses the hardware acceleration unit in the chip for decoding, and adopts a different upsample algorithm compared with the native OpenCV. If Sophon OpenCV or SAIL is used for decoding and pre- and post-processing in the ported Python code, the decoding and pre- and post-processing methods may differ from those of the native OpenCV, which may affect the final prediction results, but usually does not have a significant impact on robust models.
- The SoC mode of the debian9 system (such as SE5 box) defaults to use Sophon OpenCV, while the SoC mode and PCIe mode of the ubuntu system default to use native OpenCV. Sophon OpenCV can be used by setting environment variables.
- We usually use a static size bmodel. If the model input size in the original program is not fixed, it is recommended to modify the original program's pre-processing operation to make the model input size consistent with the bmodel before performing precision alignment.

## 3 Common Causes of Accuracy Errors

- Incorrectly tracing the model, resulting in inconsistent inference results between the traced model and the original model.

- The Python code used for decoding and preprocessing in the ported program uses Sophon OpenCV or SAIL, and the decoding and preprocessing methods differ from those of the native OpenCV, which may affect the final prediction results. You can refer to step 7 to evaluate the impact and alignment. Generally, it will not have a significant impact on models with good robustness. If the impact is significant, you can try to increase the original model's robustness to reduce the impact.

- The interface or interpolation method for resizing the ported program differs from that of the original program, resulting in differences in accuracy. For example, the original program uses `transforms.Resize`, while the ported program uses `cv2.resize`. `transforms.Resize` performs antialiasing in addition to the default bilinear interpolation, which differs significantly from the data processed by `cv2.resize`. It is recommended to use OpenCV for decoding and preprocessing in the original program.

- The padding strategy and pixel values in the ported program may differ from the original program, which may cause subtle differences.

- The channel ordering (BGR or RGB) of the preprocessed images in the ported program may be inconsistent with that of the original program, which may cause significant differences.

- Incorrect parameter settings for normalization in the ported program may result in significant differences in prediction results. For example, the normalization operation in the original program is `transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])`, and the corresponding normalization in the ported program is as follows:

  ```python
  # Perform normalization using ’numpy‘
  mean = np.array([0.485, 0.456, 0.406]).reshape((1, 1, 3)) * 255.0
  scale = np.array([1/0.229, 1/0.224, 1/0.225]).reshape((1, 1, 3)) * 1 / 255.0
  img = img - mean
  img = img * scale
  # Perform normalization using ‘sail.Bmcv.convert_to’
  self.ab = [x * self.input_scale for x in [1/0.229/255.0, -0.485/0.229, 1/0.224/255.0, -0.456/0.224, 1/0.225/255.0, -0.406/0.225]]
  self.bmcv.convert_to(input_bmimg, output_bmimg, ((self.ab[0], self.ab[1]), \
                                                   (self.ab[2], self.ab[3]), \
                                                   (self.ab[4], self.ab[5])))
  ```

**Note**: Channel order should be aligned as BGR or RGB before normalization.

- The output of a multi-output model is saved in dictionary form. If you convert the output dictionary values to a list, there may be a problem with the order of the output. It is recommended to use the output name to obtain the corresponding output data. For example:

  ```python
  # Convert the values of a dictionary to a list
  outputs_sail = list(outputs_sail.values())
  # Get the corresponding output data based on the output name
  outputs_sail_0 = outputs_sail['17']
  outputs_sail_1 = outputs_sail['18']
  outputs_sail_2 = outputs_sail['28']
  outputs_sail_3 = outputs_sail['29']
  ```

- Inconsistent NMS interfaces in object detection algorithms can cause differences in prediction results. For example, using `torchvision.ops.nms` in the original program and `cv2.dnn.NMSBoxes` in the ported program can lead to minor differences in accuracy.

- When using the coco2017 validation dataset for accuracy testing, please pay attention to the following when saving detection data to a JSON file:
  1. Whether the precision of the bbox coordinates needs to be rounded to 3 decimal places.
  2. Whether the precision of the score needs to be rounded to 5 decimal places.

  ```python
    bbox_dict['bbox'] = [float(round(x1, 3)), float(round(y1, 3)), float(round(x2 - x1,3)), float(round(y2 -y1, 3))]
    bbox_dict['score'] = float(round(score,5))
  ```

## 4 Reference Steps for Accuracy Alignment

In the case of CV algorithm models, the process from input to obtaining the predicted results can usually be divided into four stages: decoding, preprocessing, network inference, and post-processing. Each stage includes one or more operations. Differences in each operation can potentially cause differences in the final predicted results. Therefore, we need to ensure that the input data for each operation is the same, compare the output data of each operation, and identify any precision errors caused by each operation. The following steps can be followed for reference:

**Step 1**: Check the interfaces and parameters of decoding and pre/post processing, and try to maintain consistency between each operation before and after transplantation to avoid accuracy errors caused by common transplantation problems. Refer to [3 Common Causes of Accuracy Errors](#3-common-causes-of-accuracy-errors).

**Step 2**: Test with a `batch_size=1` FP32 BModel. If the test result is significantly different from the original program, find the samples where the test results are inconsistent as the test samples for accuracy alignment.

**Step 3**: Set the environment variable `BMRT_SAVE_IO_TENSORS=1`, and then use a `batch_size=1` FP32 BModel to test the samples selected in Step 2. The data before and after the BModel inference will be automatically saved in the current directory as `input_ref_data.dat.bmrt` and `output_ref_data.dat.bmrt`.

```bash
export BMRT_SAVE_IO_TENSORS=1
```

**Step 4**: Test the same sample in the original program, load `input_ref_data.dat.bmrt` in the original program, and compare it with the data before inference of the original model. If the data before inference is consistent, it can be considered that the decoding and preprocessing have been aligned, otherwise alignment of the individual operations of decoding and preprocessing is needed. Taking the PyTorch original program as an example:

```python
import numpy as np
# Load input_ref_data.dat.bmrt and convert it to a numpy array
# Adjust the reshape parameter based on the input size of the bmodel. You can check the input information of the model by using bm_model.bin --info *.bmodel
input_bmrt_array = np.fromfile("input_ref_data.dat.bmrt", dtype=np.float32).reshape(1,3,224,224)
# Convert the input of the original model to numpy.array
input_torch_array = input_tensor.detach().numpy()
# Compare the maximum error between input_bmrt_array and input_torch_array
print("input_diff_max:", np.abs(input_bmrt_array - input_torch_array).max())
```

**Step 5**: Use `input_bmrt_array` as the input for the original model and compare the output data. If the maximum error is below 0.001, it usually will not affect the final prediction results. Taking Pytorch as an example:

```python
# Convert input_bmrt_array to torch.tensor
input_tensor = torch.from_numpy(input_bmrt_array).to(device)
# Perform inference using the original model
output_torch_tensor = model_torch(input_tensor)
# Converting the output of the original model to a numpy array
output_torch_array = output_torch_tensor.detach().numpy()
# Load output_ref_data.dat.bmrt and convert it to a numpy array
# Modify the reshape parameter according to the output size of the bmodel. You can check the output information of the bmodel through bm_model.bin --info *.bmodel
output_bmrt_array = np.fromfile("output_ref_data.dat.bmrt", dtype=np.float32).reshape(1,8)
# Compare the maximum error between the output of the original model and the output of the bmodel
print("output_diff_max:", np.abs(output_bmrt_array - output_torch_array).max())
```

**Step 6**: Use `output_bmrt_array` for post-processing in the original program, compare the prediction results between the original program and the ported program. If the prediction results are inconsistent, it indicates that there is a precision error in the post-processing operation, and each operation of the post-processing needs to be aligned.

**Step 7**: Based on steps 4 to 6, the alignment of decoding, pre-processing, post-processing can be preliminarily determined. If they are not aligned, the data of each operation in the transplant program can be saved, and loaded and compared in the corresponding operation of the original program. If the input is the same but the output is different, the operation may have an error, and alignment or optimization should be performed as appropriate.

As an example of aligning the resize operation, if the original program uses `transforms.Resize` and the ported program uses `cv2.resize`, you can follow these steps for troubleshooting:

1. Save the data before and after resizing in the porting program.

```python
# Save the data before resize operation
np.save('img_read_cv.npy', img_read_cv)
# Resize operation in the ported program
img_resize_cv = cv2.resize(img_read_cv, (resize_w, resize_h))
# Save the data after resize operation
np.save('img_resize_cv.npy', img_resize_cv)
```

2. Use `img_read_cv.npy` in the original program for resizing and compare the results.

```python
# Load img_read_cv.npy
img_read_cv = np.load('img_read_cv.npy')
# Convert img_read_cv to the corresponding format and resize
img_read_PIL = transforms.ToPILImage()(img_read_cv)
img_resize_PIL = transforms.Resize((resize_w, resize_h))(img_read_PIL)
# Convert the resized data to a numpy array
img_resize_transforms = np.array(img_resize_PIL)
# Load img_resize_cv.npy
img_resize_cv = np.load('img_resize_cv.npy')
# Compare the maximum error of the resize operation
print("resize_diff_max:", np.abs(img_resize_cv - img_resize_transforms).max())
```

If the ported Python code uses `sail.Decoder` and `sail.Bmcv`, you can convert `sail.BMImage` to `numpy.array` for saving and comparison:

```python
# Initialize sail.Tensor and modify initialization parameters according to actual situation
sail_tensor = sail.Tensor(self.handle, (h, w), sail.Dtype.BM_FLOAT32, True, True)
# Convert sail.BMImage to sail.Tensor
self.bmcv.bm_image_to_tensor(bmimage, sail_tensor)
# Converting sail.Tensor to numpy.array
sail_array = sail_tensor.asnumpy()
```

## 5 Resnet Classification Model Accuracy Alignment Examples
**Environment Preparation**

- PyTorch original code running environment: opencv-python, torchvision>=0.11.2, pytorch>=1.10.1 are required.
- Porting code running environment: SophonSDK and Sophon Inference need to be installed using the Docker image provided on the official website, and it needs to be run in PCIe mode.

**Download our test pack：**
```bash
pip3 install dfss -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade
python3 -m dfss --url=open@sophgo.com:sophon-demo/common/bmodel_align.zip
unzip bmodel_align.zip
```
**files in directory `bmodel_align`：**
```bash
├── resnet_torch.py #pytorch source code
├── resnet_torch_new.py #source code for load and compare data
├── resnet_sail.py #original sail code for resnet
├── resnet_sail_new.py #code for saving operation data
├── resnet_sail_align.py #i/o data aligned code
├── resnet18_fp32_b1.bmodel #FP32 BModel
├── resnet18_traced.pt # torchscript model
└── test.jpg #test image
```

**Reproduction Steps**

- Step 1: Download the relevant files. You can compile the FP32 BModel model in the `TPU-NNTC` development environment, or use the converted FP32 BModel directly.

  ```bash
  python3 -m bmnetp --model=resnet18_traced.pt --shapes=[1,3,224,224] --target="BM1684"
  ```

- Step 2: Run the original PyTorch code and the pre-aligned porting code separately, and observe a significant difference in test results.

  ```bash
  # Running original code
  python resnet_torch.py
  # Original code running results
  INFO:root:filename: test.jpg, res: 751, score: 9.114608764648438
  # Running the ported code before alignment
  python3 resnet_sail.py
  # Result of running the unaligned ported code
  INFO:root:filename: test.jpg, res: 867, score: 9.796974182128906
  ```

- Step 3: Modify the ported code to save data for each operation (refer to resnet_sail_new.py), set environment variables, and run the modified ported code.

  ``` bash
  export BMRT_SAVE_IO_TENSORS=1
  python3 resnet_sail_new.py
  ```

  Generates the following files:

  ```bash
  img_read_cv.npy：Decoded data
  img_resize_cv.npy：Resized data
  img_normalize_cv.npy：Normalized data
  input_ref_data.dat.bmrt：Input data for model inference
  output_ref_data.dat.bmrt：Output data for model inference
  ```

- Step 4: Copy the above files to the same directory as the original code and modify the original code. Load the data generated by the ported program in the original code and perform a comparison (refer to resnet_torch_new.py).

  ```bash
  python resnet_torch_new.py
  ```
  The comparison result is as follows:
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

- Step 5: From the comparison results, it can be seen that the operations of image decoding, network inference, and post-processing are basically aligned, but the preprocessing is not aligned starting from the resize operation. Therefore, it is necessary to modify the resizing operation in the transplant code to make it aligned (see resnet_sail_align.py for details). The modifications are as follows:

  ```bash
  # Before alignment
  # h, w, _ = img.shape
  # if h != self.net_h or w != self.net_w:
  #     img = cv2.resize(img, (self.net_w, self.net_h))
  # After alignment
  img = transforms.ToPILImage()(img)
  img = transforms.Resize((self.net_w, self.net_h))(img)
  img = np.array(img)
  ```

- Step 6: Run the migrated code after alignment, save the data of each operation, and repeat step 4 to verify.

  ```bash
  # Run the aligned porting code to save the data for each operation
  python3 resnet_sail_align.py
  # The result of running the ported code after alignment
  INFO:root:filename: test.jpg, res: 751, score: 9.114605903625488
  # Copy the generated data and run the modified original program
  python resnet_torch_new.py
  # Print the comparison results
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

**Analysis of results:**

- Error reason: `transforms.Resize` not only uses the default bilinear interpolation but also performs antialiasing, which leads to significant differences from the data processed by `cv2.resize`, resulting in obvious accuracy errors. There is a slight difference between the inference results of the FP32 BModel and the original model, but it does not affect the prediction accuracy significantly.
- After alignment, the prediction results of the ported program and the original program are basically the same.
