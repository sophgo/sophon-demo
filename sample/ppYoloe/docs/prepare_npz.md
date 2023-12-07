# 多输入模型npz验证文件制作

## 1.简介

使用MLIR转仅有图像作为输入的模型时，可以添加一个验证文件（图像或npz格式的文件）作为参考输入，使用各算子的计算结果辅助判断模型转换的正确性。对于多输入模型，则需要基于数据集制作npz格式的参考输入文件，且生成npz文件时，需要进行图像的预处理。本文档以ppYoloe为例，说明npz文件的制作要点。

## 2.bmodel模型输入信息

以ppYoloe为例，安装libsophon后，可以使用bmrt_test读取bmodel的相关信息。

```bash
# 在ppyoloe_1684_fp32_1b.bmodel所在路径下，使用bmrt_test查看模型输入信息
bmrt_test --bmodel ppyoloe_1684_fp32_1b.bmodel --devid 0

# 打印出的模型输入输出信息部分
[BMRT][show_net_info:1523] INFO: ---- stage 0 ----
[BMRT][show_net_info:1532] INFO:   Input 0) 'image' shape=[ 1 3 640 640 ] dtype=FLOAT32 scale=1 zero_point=0
[BMRT][show_net_info:1532] INFO:   Input 1) 'scale_factor' shape=[ 1 2 ] dtype=FLOAT32 scale=1 zero_point=0
[BMRT][show_net_info:1542] INFO:   Output 0) 'p2o.Concat.29_Concat' shape=[ 1 80 8400 ] dtype=FLOAT32 scale=1 zero_point=0
[BMRT][show_net_info:1542] INFO:   Output 1) 'p2o.Div.1_Div' shape=[ 1 8400 4 ] dtype=FLOAT32 scale=1 zero_point=0
```

可以发现，ppYoloe模型的输入为两个，第一个是'image'，形状为[1, 3, 640, 640]，第二个为'scale_factor'，即输入图像的高宽和640的比例， 形状为[1, 2]，制作npz参考输入文件时，需要包含这两个输入。

## 3.预处理信息

根据官方PaddlePaddle的ppYoloe读取图像数据的[配置文件](https://github.com/PaddlePaddle/PaddleYOLO/blob/develop/configs/ppyoloe/_base_/ppyoloe_reader.yml)，可知官方模型的图像预处理数据为*mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225]*，MLIR对npz格式的参考输入不会做归一化预处理，因此要在生成npz文件时进行预处理。

## 4. 关键代码解释

读取图像为numpy数据需要opencv-python库，可以用如下指令安装：

```bash
pip3 install opencv-python
```

生成npz的python代码位于tools/prepare_npz.py。在tools文件夹下执行python3 prepare_npz.py即可。需要配置图像数据路径，输出npz文件的路径，用numpy合并预处理后的图像数据和scale_factor数据。代码核心部分解释如下：

```python
# 数据集文件夹路径下需要转换的图像格式，需要opencv-python库支持
img_extension = ['.jpg','.jpeg','.png']

# 归一化所需的均值和方差，ppYoloe模型的数据来自官网配置文件
channel_means = np.array([0.485, 0.456, 0.406])
channel_stds = np.array([0.229, 0.224, 0.225])

# 读取图像文件，resize为高宽640，数据类型转为float32，除以255，使数据[0, 255]变为[0, 1]
img_bgr_uint8 = cv2.imread(img_path)
img_rgb_uint8 = cv2.cvtColor(img_bgr_uint8, cv2.COLOR_BGR2RGB)
img_resized_uint8 = cv2.resize(img_rgb_uint8, img_target_shape)
img_resized_fp32 = img_resized_uint8.astype("float32")  # change data format
img = img_resized_fp32.transpose((2, 0 , 1))
img = img/255.

# 根据mean和std进行归一化
normalized_img = (img - channel_means[:, np.newaxis, np.newaxis])/channel_stds[:, np.newaxis, np.newaxis]
normalized_img = normalized_img[np.newaxis, ...]

# 模型输入高宽(640)与图像原始高宽的比例
img_ratio_h = img_target_shape[0]/float(img_rgb_uint8.shape[0])
img_ratio_w = img_target_shape[1]/float(img_rgb_uint8.shape[1])

# 构建npz数据
img_npz = {}
img_npz['image'] = normalized_img
img_npz['scale_factor'] = np.array(([img_ratio_h, img_ratio_w],)).astype('float32')
npz_save_path = os.path.join(npz_path, img_name_withoutEXT+".npz")
np.savez(npz_save_path, **img_npz)
```

## 5.转换模型时的使用方法

转换模型指令的详细参数可参考[MLIR使用](https://doc.sophgo.com/sdk-docs/v23.07.01/docs_latest_release/docs/tpu-mlir/quick_start/html/03_onnx.html)，此处以ppYoloe生成BM1684x上的fp16模型为例，转换时用test_input指定验证文件：

```bash
model_transform --model_name ppyoloe \
--model_def ppyoloe.onnx \
--input_shapes [[1,3,640,640],[1,2]] \
--output_names p2o.Div.1,p2o.Concat.29 \
--test_input your_npz_file.npz \
--test_result ppyoloe_top_outputs.npz \
--mlir ppyoloe_1b.mlir

model_deploy --mlir ppyoloe_1b.mlir\
--quantize F16 \
--chip bm1684x \
--test_input your_npz_file.npz \
--test_reference ppyoloe_top_outputs.npz \
--model ppyoloe_1684x_f16.bmodel
```

