# SOPHON TPU 模型移植申请模板

> 填写此模板后，提交给 Claude Code，即可自动完成 BM1684X 上的模型移植。
> 包含：ONNX 导出 → BModel 编译 → Python/C++ 推理 → 精度/性能测试 → 文档更新。

---

## 1. 基本信息

| 项目 | 内容 |
|------|------|
| 模型名称 | `【填写: 如 ArcFace / YOLOv8 / BERT】` |
| 算法类别 | `【填写: 如 人脸识别 / 目标检测 / 语音识别】` |
| 原始框架 | `【填写: 如 PyTorch / TensorFlow / PaddlePaddle】` |
| 原始代码仓库 | `【填写: GitHub 链接或本地路径】` |
| 预训练模型路径 | `【填写: .pth / .ckpt / .pb 文件路径】` |
| 论文/参考文档 | `【填写: 论文链接或文档链接】` |

---

## 2. 模型架构

### 2.1 模型结构

```
【填写模型 pipeline 流程，例如:】
输入 → 预处理 → 骨干网络(TPU) → 后处理(CPU) → 输出
```

### 2.2 子模型拆分（如果是多模型串联）

> 如果只有一个模型，只填"模型"行即可。

| 子模型 | 名称 | 功能描述 | 输入 shape | 输出 shape |
|--------|------|---------|-----------|------------|
| 子模型1 | `【如 backbone】` | `【如 特征提取】` | `【如 [1,3,112,112]】` | `【如 [1,512]】` |
| 子模型2 | `【如 decoder】` | `【如 解码】` | `【如 [1,512]】` | `【如 [1,1000]】` |
| 子模型3 | `【如 空】` | `【如 空】` | `【如 空】` | `【如 空】` |

### 2.3 关键算子

```
【列出模型中使用的特殊算子，可能影响 ONNX 导出和 TPU-MLIR 编译:】
- 标准卷积 / 全连接 / BatchNorm
- 【填写其他特殊算子，如 MultiHeadAttention, LayerNorm, Gelu 等】
```

---

## 3. 输入规格

| 参数 | 值 |
|------|-----|
| 输入类型 | `【填写: 图像 / 音频 / 文本 / 视频】` |
| 输入格式 | `【填写: .jpg / .wav / .txt 等】` |
| 输入尺寸 | `【填写: 如 112x112 RGB 图像】` |
| 输入 shape (模型) | `【填写: 如 [batch, 3, 112, 112]】` |
| 输入 dtype | `【填写: float32 / int32 / uint8】` |
| 动态维度 | `【填写: 哪些维度是动态的，如 batch 维度】` |
| 归一化方式 | `【填写: 如 mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5] 或 除以255】` |

---

## 4. 预处理详情

### 4.1 预处理流程

```
【用伪代码或步骤描述预处理流程，例如:】
1. 读取图片，BGR/RGB 格式
2. Resize 到 112x112 (保持宽高比 + padding / 直接拉伸)
3. 转换到 RGB 格式
4. 归一化: (pixel - mean) / std，值域变为 [-1, 1] 或 [0, 1]
5. 转换为 NCHW 格式
```

### 4.2 预处理参数

| 参数 | 值 |
|------|-----|
| 目标尺寸 | `【如 112x112】` |
| 保持宽高比 | `【是/否】` |
| Padding 填充值 | `【如 [114, 114, 114]】` |
| 颜色格式 | `【RGB / BGR】` |
| 均值 (mean) | `【如 [127.5, 127.5, 127.5]】` |
| 方差/缩放 (std/scale) | `【如 [0.0078125, 0.0078125, 0.0078125]】` |
| 输出值域 | `【如 [-1, 1] 或 [0, 1]】` |

---

## 5. 后处理详情

### 5.1 后处理流程

```
【用伪代码描述后处理步骤，例如:】
1. 取模型输出 embedding 向量
2. L2 归一化: embedding = embedding / ||embedding||
3. 与注册库中的 embedding 做余弦相似度比对
4. 输出最高相似度的人脸 ID 和置信度
```

### 5.2 输出规格

| 输出 | shape | dtype | 含义 |
|------|-------|-------|------|
| 输出1 | `【如 [1, 512]】` | `【如 float32】` | `【如 人脸特征向量】` |
| 输出2 | `【如 空】` | `【如 空】` | `【如 空】` |

---

## 6. BModel 编译需求

### 6.1 目标芯片和精度

| 参数 | 值 |
|------|-----|
| 目标芯片 | `【填写: BM1684X / BM1684 / BM1688 / CV186X】` |
| 需要编译的精度 | `【填写: FP32 / FP16 / INT8 / INT8_4b，多选用逗号分隔】` |
| 需要编译的 batch | `【填写: 1b / 4b / 1b和4b 等】` |
| 最大输入长度 | `【填写: 动态模型需指定 max_len，如 1000】` |
| 是否启用动态 shape | `【是/否】` |

### 6.2 INT8 校准数据（仅 INT8 需要）

| 参数 | 值 |
|------|-----|
| 校准数据集路径 | `【填写: 校准图片/数据目录】` |
| 校准样本数 | `【填写: 如 100】` |
| 校准数据格式 | `【填写: 与模型输入一致的格式】` |

---

## 7. 精度测试需求

### 7.1 精度指标（根据算法类型选择）

请根据算法类别选择对应的精度指标：

| 算法类别 | 推荐指标 | 典型目标值 |
|---------|---------|-----------|
| 图像分类 | Top-1/Top-5 准确率 ACC(%) | FP32 与参考一致，INT8 diff<1% |
| 目标检测 | COCO mAP (AP@IoU=0.5:0.95) | FP32 与参考一致，INT8 diff<1% |
| 语义分割 | mIoU | FP32 与参考一致 |
| 姿态估计 | COCO keypoints mAP | FP32 与参考一致 |
| 人脸识别 | 余弦相似度 (Cosine Similarity) | > 0.99 (FP32), > 0.98 (INT8) |
| 语音识别 | CER (字符错误率) / WER (词错误率) | FP32 与参考一致 (diff=0) |
| OCR / 车牌识别 | 字符准确率 / F-score | FP32 与参考一致，INT8 diff<2% |
| 立体匹配 | D1 (3-px error rate) / EPE | FP32 与参考一致 |
| 超分辨率 | PSNR / SSIM | 越高越好 |
| 多目标跟踪 | MOTA / MOTP / IDF1 | FP32 与参考一致 |
| LLM / 图像生成 | 通常无正式指标，或使用 FID / PPL | — |

### 7.2 精度指标详情

| 指标名称 | `【填写: 如 余弦相似度 / CER+WER / COCO mAP】` |
|---------|------|
| 计算方式 | `【填写: 如 与参考模型 embedding 的余弦距离 / editdistance CER / pycocotools mAP】` |
| 目标值 (FP32) | `【填写: 如 > 0.99 / CER=0.00% / mAP=0.377】` |
| 目标值 (FP16) | `【填写: 如 > 0.99 / CER<0.5% / mAP diff<0.01】` |
| 目标值 (INT8) | `【填写: 如 > 0.98 / CER<1% / mAP diff<1%】` |

### 7.3 测试数据

| 参数 | 值 |
|------|-----|
| 测试集名称 | `【填写: 如 ImageNet val / COCO val2017 / AISHELL-1 test / WiderFace val】` |
| 测试集路径 | `【填写: 测试数据目录】` |
| 测试样本数 | `【填写: 如 1000】` |
| 标签文件路径 | `【填写: 如 datasets/coco/instances_val2017.json / datasets/test_label.json】` |
| 参考模型 | `【填写: 用于对比的 PyTorch/ONNX 模型】` |
| 参考推理环境 | `【填写: PyTorch CPU / PyTorch GPU / ONNX Runtime】` |
| 评估脚本路径 | `【填写: 如 tools/eval_coco.py / tools/eval_aishell.py】` |

---

## 8. 性能测试需求

### 8.1 性能指标（根据算法类型选择）

请根据算法类别选择对应的性能指标：

| 算法类别 | 推荐性能指标 | 阶段分解 |
|---------|------------|---------|
| 图像分类/检测/分割 | FPS 或 单帧耗时(ms) | decode + preprocess + inference + postprocess |
| 人脸识别 | 单帧耗时(ms) 或 FPS | decode + preprocess + inference + postprocess |
| 语音识别 | RTF (Real Time Factor) | preprocess + encoder + decoder + postprocess |
| LLM 文本生成 | tokens/s (吞吐量) | prefill + decode_per_token |
| OCR 级联模型 | 总耗时(ms) / FPS | 各子模型分别计时 |
| 图像生成 | 单次生成耗时(s) | preprocess + 迭代推理 + postprocess |

### 8.2 性能测试详情

| 参数 | 值 |
|------|-----|
| 测试平台 | `【填写: x86 PCIE / SE7-32 / SE5-16 / SE9-16 / 全部】` |
| 端到端性能指标 | `【填写: 如 FPS(帧/秒) / RTF(real-time factor) / tokens/s / 单帧耗时(ms)】` |
| 阶段拆解方式 | `【填写: 如 decode→preprocess→inference→postprocess 或 encoder→decoder→...】` |
| 测试次数 | `【填写: 如 5 次取平均】` |
| 业务性能要求 | `【填写: 如 FPS > 100 / latency < 10ms / RTF < 0.5 / tokens/s > 50】` |
| bmrt_test 理论性能 | `【填写: 是否已测试，记录 calculate time】` |

---

## 9. 部署形态

| 参数 | 值 |
|------|-----|
| Python 推理 | `【需要/不需要】` |
| C++ 推理 | `【需要/不需要，如果两者都要填"两者都要"】` |
| C++ 推理 SDK | `【BMRT (bmrt) / SAIL (sail)，多选】` |
| 前后处理方式 | `【BMCV / OpenCV / 自定义，多选】` |
| SoC 部署 | `【需要/不需要】` |
| SoC 设备型号 | `【SE7-32 / SE5-16 / 其他】` |
| 目标设备     | `【SC7 (用户名@ip:密码) / SE7-32 (用户名@ip:密码) / SE9-16 (用户名@ip:密码) / SE9-8 (用户名@ip:密码)】`|
---

## 10. 依赖和环境

### 10.1 Python 依赖

```
【列出 Python 依赖包，如:】
- torch >= 1.9.0
- opencv-python
- numpy
- 【其他】
```

### 10.2 C++ 依赖

```
【列出 C++ 依赖库，如:】
- libsophon (bmrt, bmlib, bmcv)
- OpenCV
- FFmpeg
- 【其他第三方库，如 Armadillo, libsndfile】
```

### 10.3 特殊依赖

```
【填写模型特有的依赖，如:】
- onnx >= 1.10.0
- onnxruntime
- onnx-simplifier
- 【自定义 Python 包】
```

---

## 11. 其他信息

### 11.1 已知问题或注意事项

```
【填写: 已知的 ONNX 导出问题、不支持的算子、精度敏感层等】
```

### 11.2 参考已有 Sample（重要）

```
【填写: sophon-demo 中类似模型的 sample 路径，作为参考模板】
如: sample/RetinaFace, sample/SCRFD, sample/ResNet
```

> **⚠️ 重要原则：开发新 Sample 时，务必参考 sophon-demo 中已有的类似 Sample。**
>
> 在开始任何开发工作前，请先执行以下步骤：
>
> **1. 找到最相似的已有 Sample**
> - 根据算法类别找到同类的 sample（如目标检测 → 参考 `sample/YOLOv5`，人脸识别 → 参考 `sample/ArcFace`/`sample/RetinaFace`，ASR → 参考 `sample/WeNet`/`sample/SeAcoParaformer`）
> - 根据模型架构找到最接近的 sample（单模型 → `sample/ResNet`，编码器-解码器 → `sample/WeNet`，级联多模型 → `sample/PP-OCR`）
> - 如果存在多个候选，选择最新、最完善的 sample 作为主要参考
>
> **2. 参考已有 Sample 的以下内容：**
>
> | 需要开发的内容 | 参考已有 Sample 的什么 |
> |--------------|---------------------|
> | **README.md** | 整体结构、章节划分（简介/特性/准备模型与数据/模型编译/例程测试/精度测试/性能测试/FAQ）、表格格式、测试说明措辞 |
> | **scripts/download.sh** | 下载脚本的结构、模型/数据集下载方式、目录组织方式 |
> | **scripts/gen_*bmodel_mlir.sh** | 模型编译脚本的参数（model_transform/model_deploy 参数）、batch/精度组合 |
> | **scripts/auto_test.sh** | 自动化测试流程、各精度/batch 的测试循环 |
> | **python/*.py** | 推理代码结构、预处理/后处理实现、sophon.sail API 使用方式、命令行参数 |
> | **cpp/\*\*/\*** | C++ 工程结构（CMakeLists.txt）、bmrt API 使用方式、预处理实现 |
> | **tools/\*\*/\*** | 精度/性能评估脚本结构和实现 |
> | **docs/\*\*/\*** | 模型导出文档、补充说明文档的结构 |
>
> **3. 适配而非照搬**
> - 保留已有 Sample 的整体结构和风格一致
> - 根据新模型的差异（输入尺寸、预处理方式、输出格式等）修改具体逻辑
> - 变量名、函数名、文件名等使用与新模型匹配的命名
>
> **4. 算法类别 → 推荐参考 Sample 速查表：**
>
> | 算法类别 | 推荐参考 Sample |
> |---------|---------------|
> | 图像分类（单模型） | `sample/ResNet`, `sample/C3D` |
> | 目标检测 | `sample/YOLOv5`, `sample/YOLOX` |
> | 实例分割 | `sample/yolact`, `sample/YOLO26_seg` |
> | 语义分割 | `sample/segformer`, `sample/Unet` |
> | 姿态估计 | `sample/HRNet_pose`, `sample/OpenPose` |
> | 人脸检测 | `sample/RetinaFace`, `sample/SCRFD` |
> | 人脸识别 | `sample/ArcFace` |
> | 语音识别 (Encoder-Only) | `sample/WeNet`, `sample/SeAcoParaformer` |
> | 语音识别 (Encoder-Decoder) | `sample/Whisper`, `sample/SeAcoParaformer` |
> | LLM / 文本生成 | `sample/Qwen`, `sample/ChatGLM4`, `sample/Llama2` |
> | OCR（级联模型） | `sample/PP-OCR`, `sample/LPRNet` |
> | 多目标跟踪 | `sample/ByteTrack`, `sample/DeepSORT` |
> | 超分辨率 | `sample/Real-ESRGAN` |
> | 立体匹配 | `sample/LightStereo` |
> | 视觉语言模型 (VLM) | `sample/InternVL2`, `sample/Qwen2-VL` |
> | 图像生成 | `sample/StableDiffusionV1_5`, `sample/StableDiffusionXL` |
>
> **5. 文件结构规范（所有 Sample 统一）**
> ```
> sample/NewModel/
> ├── README.md              # 必须，参考同类 sample 的结构
> ├── docs/                  # 可选，模型导出指南等补充文档
> ├── models/                # 模型存放目录（BM1684/BM1684X/BM1688/CV186X/onnx/torch）
> ├── python/                # Python 推理代码
> ├── cpp/                   # C++ 推理代码（如有）
> │   └── newmodel_bmcv/     # C++ 子目录以模型名+预处理方式命名
> ├── scripts/               # 下载/编译/自动化测试脚本
> │   ├── download.sh        # 模型和数据集下载脚本
> │   ├── gen_fp32bmodel_mlir.sh  # FP32 模型编译脚本
> │   ├── gen_fp16bmodel_mlir.sh  # FP16 模型编译脚本
> │   ├── gen_int8bmodel_mlir.sh  # INT8 模型编译脚本
> │   └── auto_test.sh       # 自动化测试脚本
> ├── tools/                 # 精度/性能评估工具脚本
> ├── datasets/              # 测试数据集
> └── results/               # 测试结果输出
> ```

### 11.3 额外需求

```
【填写: 除标准流程外的特殊需求，如视频流输入、多线程、GRPC 服务等】
```

---

## 使用说明

1. **填写模板**: 按照上述章节，尽可能详细地填写模型信息
2. **提交给 Claude Code**: 将填好的模板内容粘贴到对话中，并声明"请按照 skills 帮我移植这个模型"
3. **Claude 自动执行**: Claude 会按 10 个 Skill 的流程自动完成：
   - Skill 1: 分析模型架构
   - Skill 2: 搭建环境
   - Skill 3: 导出 ONNX
   - Skill 4: 编译 BModel
   - Skill 5: Python 推理验证
   - Skill 6: 精度测试
   - Skill 7: 性能测试
   - Skill 8: C++ 移植
   - Skill 9: SoC 部署
   - Skill 10: 文档更新
4. **结果交付**: 移植完成后，会得到：
   - `models/BM1684X/` 下的 BModel 文件
   - `python/` 下的 Python 推理代码
   - `cpp/` 下的 C++ 推理代码
   - 精度和性能测试结果
   - 更新后的 README

---

## 简化版模板（最小信息集）

如果暂时不想填写完整模板，至少提供以下核心信息：

```
模型名称:          【填写】
算法类别:          【分类/目标检测/语义分割/姿态估计/人脸识别/语音识别/OCR/立体匹配/超分辨率/多目标跟踪/LLM/图像生成/其他】
原始框架:          【PyTorch/TensorFlow/PaddlePaddle/其他】
模型文件位置:       【本地路径或下载链接】
输入尺寸:          【如 112x112 图像 / 16kHz 音频】
输入通道数:         【如 3 (RGB) / 80 (FBANK)】
输出规格:          【如 512维特征向量 / [1,68,18] CTC输出 / [N,8404] token logits】
预处理方式:         【如图像: Resize→Normalize: mean/std; 音频: FBANK特征提取→CMVN归一化】
模型架构:          【单模型 / 编码器-解码器 / 级联多模型 / Transformer Decoder-Only】
子模型个数:         【1 / 2 / 3 / 更多】
目标芯片:          【BM1684X / BM1688 / CV186X】
目标设备：         【SC7 (用户名@ip:密码) / SE7-32 (用户名@ip:密码) / SE9-16 (用户名@ip:密码) / SE9-8 (用户名@ip:密码)】
需要精度:          【FP32 / FP16 / INT8 / INT8_4b，多选用逗号分隔】
需要 batch:        【1b / 4b / 10b】
需要 Python:       【是/否】
需要 C++:          【是/否】
C++ 前后处理方式:   【bmcv / opencv / 自定义】
测试数据集:         【路径或描述】
精度指标:          【如 ACC(%) / COCO mAP / CER+WER(%) / 余弦相似度 / PSNR / D1 / MOTA】
性能指标:          【如 FPS / RTF / tokens/s / 单帧耗时(ms)】
```
