PP-OCR Python例程

python目录下提供了一系列Python例程，具体情况如下：

| 序号   | Python例程            | 说明                        | 功能 |
| ----   | ----------------     | --------------------------- |-|
| 1      | ppocr_det_opencv.py | 使用OpenCV前处理、SAIL推理   |文本检测|
| 2      | ppocr_cls_opencv.py | 使用OpenCV前处理、SAIL推理   |文本方向分类|
| 3      | ppocr_rec_opencv.py | 使用OpenCV前处理、SAIL推理   |文本识别|
| 4      | ppocr_system_opencv.py | 使用OpenCV前处理、SAIL推理   |全流程测试|

## 1. x86 PCIe平台
## 1.1 环境准备
如果您在x86平台安装了PCIe加速卡，并使用它测试本例程，您需要安装libsophon(>=0.3.0)、sophon-opencv(>=0.2.4)、sophon-ffmpeg(>=0.2.4)和sophon-sail(>=3.1.0),具体请参考[x86-pcie平台的开发和运行环境搭建](../../../docs/Environment_Install_Guide.md#2-x86-pcie平台的开发和运行环境搭建)。

需要执行以下命令安装所需的python库:
```bash
pip3 install -r requirements.txt
```

如果在运行python例程的过程中遇到 "`No module named '_ctypes'`"的问题需要
```bash
apt-get install libffi-dev
# 进入python安装路径内重新编译安装python
cd python<version>
make && make install
```

## 1.2 测试命令
以BM1684为例，如要测试BM1684X请更换bmodel文件夹

ppocr_det_opencv.py参数说明如下：
```bash
usage:ppocr_det_opencv.py [--tpu_id] [--img_path] [--det_model] [--det_batch_size] [--det_limit_side_len]
--tpu_id: 用于推理的tpu设备id;
--img_path: 输入图片文件夹的路径;
--det_model: 用于推理的文本检测bmodel路径;
--det_batch_size: 模型输入的batch_size，本例程可支持1或4;
--det_limit_side_len: 网络输入尺寸列表，本例程的模型支持960。
```

文本检测测试实例如下：
```bash
# 以测试图片中的文本检测为例，测试同时支持1batch和4batch的FP32BModel模型。
python3 ppocr_det_opencv.py --tpu_id 0  --img_path ../data/images/ppocr_img/test --det_model ../data/models/BM1684/ch_PP-OCRv2_det_fp32_b1b4.bmodel --det_batch_size 1
# BM1684X目前只支持1batch
python3 ppocr_det_opencv.py --tpu_id 0  --img_path ../data/images/ppocr_img/test --det_model ../data/models/BM1684X/ch_PP-OCRv2_det_1b.bmodel --det_batch_size 1
```

执行完成后，会将预测图片保存在`results/det_results`文件夹下。

ppocr_cls_opencv.py参数说明如下：
```bash
usage:ppocr_cls_opencv.py [--tpu_id] [--img_path] [--cls_model] [--cls_batch_size] [--cls_thresh] [--label_list]
--tpu_id: 用于推理的tpu设备id;
--img_path: 输入图片文件夹的路径;
--cls_model: 用于推理的文本方向分类bmodel路径;
--cls_batch_size: 模型输入的batch_size，本例程可支持1或4;
--cls_thresh: 预测阈值，模型预测结果为180度，且得分大于该阈值时，认为最终预测结果为180度，需要翻转;
--label_list: class id对应的角度值。
```

文本方向分类测试实例如下：
```bash
# 以图片中的文本方向分类为例测试，测试同时支持1batch和4batch的FP32BModel模型。
python3 ppocr_cls_opencv.py --tpu_id 0  --img_path ../data/images/ppocr_img/imgs_words/ch --cls_model ../data/models/BM1684/ch_ppocr_mobile_v2.0_cls_fp32_b1b4.bmodel --cls_batch_size 1 --cls_thresh 0.9 --label_list "0, 180"
# BM1684X目前只支持1batch
python3 ppocr_cls_opencv.py --tpu_id 0  --img_path ../data/images/ppocr_img/imgs_words/ch --cls_model ../data/models/BM1684X/ch_ppocr_mobile_v2.0_cls_1b.bmodel --cls_batch_size 1 --cls_thresh 0.9 --label_list "0, 180"
```

执行完成后，会打印预测的类别及置信度如下：
```bash
INFO:root:img_name:word_1.jpg, pred:0, conf:0.9998781681060791
INFO:root:img_name:word_2.jpg, pred:0, conf:0.9999998807907104
INFO:root:img_name:word_3.jpg, pred:0, conf:0.9999998807907104
INFO:root:img_name:word_4.jpg, pred:0, conf:0.9999982118606567
INFO:root:img_name:word_5.jpg, pred:0, conf:0.9999988079071045
```

ppocr_rec_opencv.py参数说明如下：
```bash
usage:ppocr_rec_opencv.py [--tpu_id] [--img_path] [--rec_model] [--rec_batch_size] [--char_dict_path] [--use_space_char]
--tpu_id: 用于推理的tpu设备id;
--img_path: 输入图片文件夹的路径;
--rec_model: 用于推理的文本识别bmodel路径;
--rec_batch_size: 模型输入的batch_size，本例程可支持1或4;
--char_dict_path: 识别的字符字典文件路径;
--use_space_char: 是否包含空格如果为True，则会在最后字符字典中补充空格字符。
```

文本识别测试实例如下：
```bash
# 由于需要处理不同的图片尺寸，需要使用组合后的 ch_PP-OCRv2_rec_fp32_b1b4.bmodel 进行测试。以图片中的文本识别为例测试，测试同时支持1batch和4batch的FP32BModel模型: 
python3 ppocr_rec_opencv.py --tpu_id 0  --img_path ../data/images/ppocr_img/imgs_words/ch --rec_model ../data/models/BM1684/ch_PP-OCRv2_rec_fp32_b1b4.bmodel --rec_batch_size 4 --char_dict_path ../data/ppocr_keys_v1.txt  --use_space_char True
# BM1684X目前只支持1batch: ch_PP-OCRv2_rec_fp32_b1.bmodel
python3 ppocr_rec_opencv.py --tpu_id 0  --img_path ../data/images/ppocr_img/imgs_words/ch --rec_model ../data/models/BM1684X/ch_PP-OCRv2_rec_fp32_b1.bmodel --rec_batch_size 1 --char_dict_path ../data/ppocr_keys_v1.txt  --use_space_char True
```

执行完成后，会打印预测的文本内容及置信度如下：
```bash
INFO:root:img_name:word_1.jpg, conf:0.997443, pred:韩国小馆
INFO:root:img_name:word_3.jpg, conf:0.936186, pred:电话：15952301928
INFO:root:img_name:word_4.jpg, conf:0.966046, pred:实力活力
INFO:root:img_name:word_5.jpg, conf:0.980753, pred:西湾监管
INFO:root:img_name:word_2.jpg, conf:0.994984, pred:汉阳鹦鹉家居建材市场E区25-26号
```

ppocr_system_opencv.py参数说明如下：
```bash
usage:ppocr_system_opencv.py [--tpu_id] [--img_path] [--batch_size] [--det_model] [--det_batch_size] [--det_limit_side_len] [--rec_model] [--img_size] [--rec_batch_size] [--char_dict_path] [--use_space_char] [--cls_model] [--cls_batch_size] [--cls_thresh] [--label_list]
--tpu_id: 用于推理的tpu设备id;
--img_path: 输入图片文件夹的路径;
--batch_size: 模型输入的batch_size，本例程可支持1或4;

--det_model: 用于推理的文本检测bmodel路径;
--det_batch_size: 模型输入的batch_size，本例程可支持1或4;
--det_limit_side_len: 网络输入尺寸列表，本例程的模型支持960。

--rec_model: 用于推理的文本识别bmodel路径;
--img_size: 用于推理的图片尺寸, 默认[[320, 32],[640, 32],[1280, 32]];
--rec_batch_size: 模型输入的batch_size，本例程可支持1或4;
--char_dict_path: 识别的字符字典文件路径;
--use_space_char: 是否包含空格如果为True，则会在最后字符字典中补充空格字符。
--drop_score:识别得分小于该值的结果会被丢弃，不会作为返回结果;

--use_angle_cls: 是否对文本方向分类;
--cls_model: 用于推理的文本方向分类bmodel路径;
--cls_batch_size: 模型输入的batch_size，本例程可支持1或4;
--cls_thresh: 预测阈值，模型预测结果为180度，且得分大于该阈值时，认为最终预测结果为180度，需要翻转;
--label_list: class id对应的角度值。
```

全流程测试：
```bash
# 以图片中的文本检测，文字方向分类以及文本识别为例，测试同时支持1batch和4batch的FP32BModel模型。
python3 ppocr_system_opencv.py --batch_size 4 --use_angle_cls True --drop_score 0.5 
# BM1684X目前只支持1batch
python3 ppocr_system_opencv.py --batch_size 1 --det_model ../data/models/BM1684X/ch_PP-OCRv2_det_1b.bmodel --det_batch_size 1 --rec_model ../data/models/BM1684X/ch_PP-OCRv2_rec_fp32_b1.bmodel --rec_batch_size 1 --cls_model ../data/models/BM1684X/ch_ppocr_mobile_v2.0_cls_1b.bmodel --cls_batch_size 1 --use_angle_cls True --drop_score 0.5 
```

执行完成后，会打印预测的字段，同时会将预测的可视化结果保存在`results/inference_results`文件夹下。

## 2. arm SoC平台
### 2.1 环境准备
如果您使用SoC平台测试本例程，您需要交叉编译安装sophon-sail(>=3.1.0)，具体可参考[交叉编译安装sophon-sail](../../../docs/Environment_Install_Guide.md#32-交叉编译安装sophon-sail)。

在SoC平台上需要执行以下命令安装所需的文件及python库:
```bash
sudo apt-get install libgeos-dev
pip3 install -r requirements.txt
```

### 2.2 测试命令
Soc平台的测试方法与x86 PCIe平台相同，请参考1.2测试命令。
