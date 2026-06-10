# Skill 6: 精度测试

## 目标
对比 TPU BModel 与参考框架（PyTorch/TensorFlow/ONNX Runtime）的推理结果，根据算法类型选择对应的评价指标，验证精度无损。

## 精度指标速查表

不同算法类型使用的精度指标差异很大，请根据模型类型选择对应的指标和评估方法：

| 算法类别 | 典型精度指标 | 参考样例 |
|---------|------------|---------|
| 图像分类 | ACC(%) Top-1/Top-5 准确率 | `sample/ResNet`, `sample/C3D`, `sample/SlowFast` |
| 目标检测 | COCO mAP (AP@IoU=0.5:0.95, AP@IoU=0.5) | `sample/YOLOv5`, `sample/YOLOX`, `sample/CenterNet` |
| 实例分割 | COCO mAP (bbox + segm) | `sample/yolact`, `sample/YOLO26_seg` |
| 语义分割 | mIoU (Mean Intersection over Union) | `sample/segformer`, `sample/Unet` |
| 姿态估计 | COCO keypoints mAP | `sample/HRNet_pose`, `sample/OpenPose` |
| 人脸检测 | WiderFace mAP | `sample/RetinaFace`, `sample/SCRFD` |
| 人脸识别 | 余弦相似度 (Cosine Similarity) | `sample/ArcFace` |
| 语音识别 (ASR) | CER (字符错误率) / WER (词错误率) | `sample/WeNet`, `sample/Whisper`, `sample/SeAcoParaformer` |
| OCR 文字识别 | F-score / Precision / Recall / 字符准确率 | `sample/PP-OCR`, `sample/LPRNet` |
| 立体匹配 | D1 (3-px) / EPE (End-Point Error) | `sample/LightStereo` |
| 超分辨率 | PSNR / SSIM | `sample/Real-ESRGAN` |
| 多目标跟踪 (MOT) | MOTA / MOTP / IDF1 | `sample/ByteTrack`, `sample/DeepSORT` |
| 人群计数 | MAE / MSE | `sample/P2PNet` |
| 图像生成 | FID / CLIP Score / 无正式指标 | `sample/StableDiffusionV1_5` |

## 执行步骤

### 6.1 准备参考推理（各算法不同）

根据算法类型选择合适的参考推理方式：

**方式 A: 原始框架推理作为参考（推荐）**
用于分类、检测、分割、ASR 等模型，使用 PyTorch/TensorFlow 在 CPU/GPU 上的推理结果作为 ground truth：

```python
# 分类模型 - 获取参考 Top-1 预测
# 参考: sample/ResNet/tools/eval_imagenet.py
ref_model = OriginalModel(checkpoint_path)
ref_prediction = ref_model.infer(image)  # → class_id

# 检测模型 - 获取参考 bbox 和类别
# 参考: sample/YOLOv5 (使用 PyTorch 模型推理 COCO val)
ref_bboxes = ref_model.detect(image)  # → [[x,y,w,h, conf, cls], ...]

# ASR 模型 - 获取参考文本
# 参考: sample/WeNet/tools/eval_aishell.py
ref_text = ref_model.transcribe(audio)  # → "识别结果文本"

# OCR 模型 - 获取参考文字
# 参考: sample/LPRNet/tools/eval_ccpd.py
ref_text = ref_model.recognize(image)  # → "识别结果"

# 人脸识别 - 获取参考 embedding
# 参考: sample/ArcFace
ref_embedding = ref_model.extract_feature(face_image)  # → [1, 512]
```

**方式 B: 使用 ONNX Runtime 作为参考**
适用于 ONNX 导出后的中间验证：

```python
import onnxruntime

ort_session = onnxruntime.InferenceSession("model.onnx")
ort_outputs = ort_session.run(None, {"input": input_data})
ref_result = ort_outputs[0]
```

**方式 C: 无需参考（部分生成类模型）**
超分辨率、图像生成类模型不对比参考模型，直接计算输出质量指标（如 PSNR/FID），或通过人工评估。

### 6.2 运行 TPU 推理

参考各 sample 的 C++ 或 Python 例程推理测试数据集，生成预测结果文件：

```bash
# Python 例程推理（以 目标检测 为例）
cd python
python3 yolo_detect.py \
    --model ../models/BM1684X/yolov5s_fp32_1b.bmodel \
    --input ../datasets/coco/val2017_1000 \
    --output results/yolo_fp32_result.json

# C++ 例程推理
cd cpp/yolo_bmcv
./yolo_bmcv.pcie \
    --model ../../models/BM1684X/yolov5s_fp32_1b.bmodel \
    --input ../../datasets/coco/val2017_1000 \
    --output results/yolo_fp32_result.json
```

### 6.3 计算精度指标（按算法类型选择）

#### 图像分类: Top-1 / Top-5 准确率

参考: `sample/ResNet/tools/eval_imagenet.py`

```python
# 将 TPU 推理预测的 class_id 与 ground truth label 对比
# 输入: gt_path (标签文件), result_json (TPU 推理结果)
correct = sum(1 for k, gt in labels.items() if predictions[k] == gt)
acc = correct / len(labels)  # Top-1 Accuracy
```

```bash
python3 tools/eval_imagenet.py \
    --gt_path datasets/imagenet_val_1k/label.txt \
    --result_json results/resnet50_fp32_result.json
```

#### 目标检测: COCO mAP

参考: `sample/YOLOv5/tools/eval_coco.py`

```python
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

cocoGt = COCO(gt_json_path)       # instances_val2017_1000.json
cocoDt = cocoGt.loadRes(pred_json)  # TPU 推理结果
cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()  # 输出 AP@IoU=0.5:0.95, AP@IoU=0.5 等
```

```bash
pip3 install pycocotools
python3 tools/eval_coco.py \
    --gt_path datasets/coco/instances_val2017_1000.json \
    --result_json results/yolov5s_fp32_result.json \
    --ann_type bbox  # 或 keypoints / segm
```

#### ASR 语音识别: CER / WER

参考: `sample/WeNet/tools/eval_aishell.py`, `sample/SeAcoParaformer/python/eval_accuracy.py`

```python
import editdistance

def char_error_rate(ref: str, hyp: str) -> float:
    """字符错误率 CER (Character Error Rate)"""
    dist = editdistance.eval(list(ref), list(hyp))
    return dist / max(len(ref), 1)

def word_error_rate(ref: str, hyp: str) -> float:
    """词错误率 WER (Word Error Rate)"""
    # 对中文通常分词后计算，或直接按字符级
    ref_words = list(ref)  # 或用分词工具切词
    hyp_words = list(hyp)
    dist = editdistance.eval(ref_words, hyp_words)
    return dist / max(len(ref_words), 1)
```

```bash
# 使用工具脚本计算 WER（每个 ASR 项目可能有自定义工具）
python3 tools/eval_aishell.py --char=1 \
    datasets/aishell/ground_truth.txt \
    python/result.txt > online_wer
cat online_wer | grep "Overall"
```

#### OCR 文字识别: 字符准确率 / F-score

参考: `sample/LPRNet/tools/eval_ccpd.py`, `sample/PP-OCR/tools/eval_icdar.py`

```python
# 车牌/文字识别 - 完全匹配准确率
tp = sum(1 for k in labels if predictions[k] == labels[k])
acc = tp / len(labels)

# OCR 端到端 - F-score / Precision / Recall
# 使用 ICDAR 评估工具
```

#### 人脸识别: 余弦相似度

```python
import numpy as np

def cosine_similarity(emb1, emb2):
    """归一化后的 embedding 点积即为余弦相似度"""
    return np.dot(emb1, emb2)
    # 或: from scipy.spatial.distance import cosine
    # similarity = 1 - cosine(emb1, emb2)
```

#### 超分辨率: PSNR / SSIM

参考: `sample/Real-ESRGAN/tools/eval_psnr.py`

```python
import cv2
import numpy as np

def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    pixel_max = 255.0
    return 20 * np.log10(pixel_max / np.sqrt(mse))
```

#### 立体匹配: D1-all (3-px Error Rate)

参考: `sample/LightStereo/tools/eval.py`

```python
# 评估视差图中像素误差大于3px且大于5%真实值的比例
# D1 = (error > 3px && error > 0.05*gt) 的像素数 / 总像素数
# 值越小越好
```

### 6.4 单文件测试 (快速验证)

```bash
cd python
python3 eval_accuracy.py \
    --model_dir ../models/BM1684X \
    --input ../data/test_sample \
    --ref "预期的参考结果"
```

### 6.5 批量测试 (完整评估)

```bash
# 根据算法类型使用对应的评估脚本
python3 eval_accuracy.py \
    --model_dir ../models/BM1684X \
    --test_manifest test_manifest.txt \
    --input_data_base /path/to/test_dataset \
    --max_samples 1000 \
    --output results/accuracy.json
```

## 精度目标（一般性指导）

| 模型精度 | 分类/检测/分割 | ASR (WER) | OCR (Acc) | 人脸 (CosSim) |
|---------|-------------|-----------|-----------|--------------|
| FP32 BModel | 与 PyTorch 一致 (diff <0.01%) | 与 PyTorch 一致 | 与 PyTorch 一致 | > 0.99 |
| FP16 BModel | 与 FP32 一致 或 diff <0.01% | diff <0.5% | diff <1% | > 0.99 |
| INT8 BModel | diff <1% (mAP) | diff <1% | diff <2% | > 0.98 |
| INT8_4b BModel | diff <2% (mAP) | diff <2% | diff <3% | > 0.97 |

## 调试方法

### 如果精度不达标:

1. **对比中间输出**: 逐层/逐子模型对比 TPU 和参考框架的输出
   ```python
   # 对比每个子模型或关键层的输出
   pt_out = pt_model.layer_N(input_data)
   tpu_out = tpu_model.submodel_N(input_data)
   diff = np.abs(pt_out - tpu_out).max()
   print(f"Max diff: {diff}, Mean diff: {np.abs(pt_out - tpu_out).mean()}")
   ```

2. **检查预处理一致性**: 确保 TPU 侧预处理与参考完全一致
   ```python
   # 对比预处理输出
   pt_preprocessed = pytorch_preprocess(input_data)  # resize, normalize, etc.
   tpu_preprocessed = cpp_preprocess(input_data)      # 对照实现
   diff = np.abs(pt_preprocessed - tpu_preprocessed).max()
   ```

3. **检查后处理一致性**: NMS、解码等算法实现的差异
   ```python
   # 检测模型: 对比 NMS 前的 bbox
   # ASR 模型: 对比 logits 和 beam search 中间状态
   # OCR 模型: 对比 CTC decode 结果
   ```

4. **检查数据类型**: FP32 → FP16 可能有精度损失，必要时使用混合精度

5. **INT8 校准问题**: 检查校准数据集是否具有代表性，可尝试增大校准样本数

## 检查清单

- [ ] 参考模型能正常推理，输出格式已确认
- [ ] TPU 模型能正常推理
- [ ] 预处理输出已验证与参考一致
- [ ] 单文件测试指标符合预期（FP32 应与参考完全一致）
- [ ] 批量测试完成
- [ ] 结果 JSON 已保存
- [ ] 详细 per-sample 结果已审查（找出异常样本）

## 示例结果

### 图像分类 (ResNet)
```
gt_path: datasets/imagenet_val_1k/label.txt
pred_path: results/resnet50_fp32_result.json
ACC: 80.10%
```

### 目标检测 (YOLOv5)
```
Running per image evaluation...
Evaluate annotation type *bbox*
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all ] = 0.377
 Average Precision  (AP) @[ IoU=0.50      | area=   all ] = 0.580
```

### 语音识别 (WeNet)
```
Overall -> 2.70 %  N=123456  C=120100  S=1856  D=800  I=700
```

### OCR 车牌识别 (LPRNet)
```
ACC = 0.894
```

### 立体匹配 (LightStereo)
```
D1: 0.454
```
