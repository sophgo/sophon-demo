# Skill 7: 性能测试

## 目标
测试 BModel 在 TPU 上的理论性能（bmrt_test）和程序端到端性能，根据算法类型和模型架构分解各阶段耗时。

## 性能指标速查表

不同算法类型的性能指标和吞吐量计算方式不同：

| 算法类别 | 端到端吞吐指标 | 关键阶段分解 | 参考样例 |
|---------|-------------|------------|---------|
| 图像分类/检测/分割 | FPS (帧/秒) 或 单帧耗时(ms) | decode → preprocess → inference → postprocess | `sample/ResNet`, `sample/YOLOv5`, `sample/segformer` |
| 人脸识别 | 单帧耗时(ms) 或 FPS | decode → preprocess → inference → postprocess | `sample/ArcFace`, `sample/RetinaFace` |
| 语音识别 (Encoder 独立) | RTF (Real Time Factor) | preprocess → encoder_inference → decoder_inference → postprocess | `sample/WeNet`, `sample/SeAcoParaformer` |
| LLM / 文本生成 | tokens/s (吞吐量) | prefill_time + decode_per_token_time | `sample/Qwen`, `sample/ChatGLM`, `sample/Llama2` |
| OCR (级联模型) | FPS 或 单帧总耗时 | 各子模型分别计时 (det→cls→rec) | `sample/PP-OCR` |
| 多目标跟踪 | FPS (含 tracker) | detector_time + tracker_update_time | `sample/ByteTrack`, `sample/DeepSORT` |
| 超分辨率/图像生成 | 单帧耗时(s) 或 iter/s | preprocess → inference → postprocess | `sample/Real-ESRGAN`, `sample/StableDiffusionV1_5` |
| 立体匹配 | 单帧耗时(ms) | decode → preprocess → inference → postprocess | `sample/LightStereo` |

## 执行步骤

### 7.1 bmrt_test 理论性能

测试每个 BModel 的纯 TPU 推理性能（不含前后处理），获得理论性能上限：

```bash
# 单模型 (如 分类/检测/OCR 单子模型)
bmrt_test --bmodel models/BM1684X/model_fp32_1b.bmodel --devid 0

# 多子模型 (如 语音识别 encoder + decoder + predictor)
bmrt_test --bmodel models/BM1684X/encoder_fp32.bmodel --devid 0
bmrt_test --bmodel models/BM1684X/decoder_fp32.bmodel --devid 0
bmrt_test --bmodel models/BM1684X/predictor_fp32.bmodel --devid 0
```

bmrt_test 输出关注指标:
- `calculate time`: 纯推理时间 (ms)
- 多 batch 模型需除以 batch_size 得到每个 sample 的推理时间

### 7.2 程序端到端性能

参考各 sample 的 Python/C++ 例程运行程序，程序内部应打印各阶段耗时：

```bash
# Python 例程
cd python
python3 model_inference.py \
    --model_dir ../models/BM1684X \
    --input ../data/test_input

# C++ PCIe 例程
cd cpp/model_bmrt
./model_bmrt.pcie \
    --model_dir ../../models/BM1684X \
    --input ../../data/test_input

# C++ SoC 例程
cd /data/model/model_bmrt
./model_bmrt.soc \
    --model_dir /data/model/models/BM1684X \
    --input /data/model/test_input
```

### 7.3 各阶段耗时分解（按模型架构）

程序需要打印各阶段耗时。不同模型架构的阶段拆分不同：

#### 架构 A: 单模型 pipeline（分类/检测/分割/OCR 单子模型）

参考: `sample/ResNet`, `sample/YOLOv5`, `sample/LPRNet`, `sample/LightStereo`

```
decode_time (CPU):       图片解码 / 视频帧读取
preprocess_time (CPU):   Resize / Normalize / HWC→CHW
inference_time (TPU):    模型推理 (可能是多子模型串联)
postprocess_time (CPU):  Argmax / NMS / CTC decode
─────────────────────────────────────────
total:                   总耗时 (ms)
```

#### 架构 B: 编码器-解码器 pipeline（ASR / Seq2Seq 模型）

参考: `sample/WeNet`, `sample/SeAcoParaformer`, `sample/Whisper`

```
preprocess_time (CPU):           特征提取 (FBANK/MFCC) + CMVN 归一化
encoder_inference_time (TPU):    编码器推理 (或 "none" 如果仅用 decoder)
decoder_inference_time (TPU):    解码器推理 (attention/CTC)
postprocess_time (CPU):          CTC解码 / Beam Search
─────────────────────────────────────────
total:                           总耗时
RTF:                             total_time / audio_duration (越小越好)
```

> **注意**: WeNet 等支持 encoder-only 模式 (CTC prefix beam search)，此时 `decoder_inference_time` = none

#### 架构 C: LLM 文本生成

参考: `sample/Qwen`, `sample/ChatGLM`, `sample/Llama2`

```
prefill_time (TPU):         预填充阶段 (处理 prompt tokens)
decode_time_per_token (TPU): 每 token 解码时间
total_decode_time (TPU):    总解码时间
total_tokens:                生成的 token 总数
─────────────────────────────────────────
throughput:                 total_tokens / total_time (tokens/s)
```

#### 架构 D: 级联多模型 pipeline（OCR 多阶段）

参考: `sample/PP-OCR`

```
det_preprocess_time:        检测模型预处理
det_inference_time:         文字检测推理
det_postprocess_time:       检测后处理
rec_preprocess_time:        识别模型预处理 (crop/resize)
rec_inference_time:         文字识别推理
rec_postprocess_time:       识别后处理 (CTC decode)
─────────────────────────────────────────
total:                      总耗时
FPS:                        1.0 / avg_total_time
```

#### 架构 E: 多目标跟踪 pipeline

参考: `sample/ByteTrack`, `sample/DeepSORT`

```
detector_time (每帧):        目标检测时间 (decode + preprocess + inference + postprocess)
tracker_time (每帧):        Tracker 状态更新 (Kalman/匈牙利匹配)
─────────────────────────────────────────
total_per_frame:            平均每帧总耗时
FPS:                        1.0 / total_per_frame
```

### 7.4 多次测试取平均

```bash
# 运行 N 次取平均，减少波动
for i in $(seq 1 5); do
    echo "--- Run $i ---"
    ./model_bmrt.soc \
        --model_dir ... --input ... 2>&1 | \
        grep -E "preprocess|inference|postprocess|total|FPS|RTF"
done
```

## 通用性能指标

| 指标 | 说明 | 适用算法 |
|------|------|---------|
| FPS | 每秒处理帧数 (1/avg_time_per_frame) | 图像分类/检测/分割 |
| RTF | Real Time Factor (total_time/audio_duration) | 语音识别 (越小越好，<1 为实时) |
| tokens/s | 每秒生成 token 数 | LLM 文本生成 |
| 延迟 (ms) | 单次推理延迟 (含/不含前后处理) | 所有模型 |
| 吞吐量 | QPS (Queries Per Second) | 所有模型 |

## 性能对比表模板

### 单模型架构 (分类/检测/分割/单子模型 OCR)

| 测试平台 | 测试程序 | 测试模型 | decode_time | preprocess_time | inference_time | postprocess_time |
|----------|---------|---------|------------|----------------|---------------|-----------------|
| SE7-32 | model.py | model_fp32_1b | ? | ? | ? | ? |
| x86 PCIE | model.py | model_fp32_1b | ? | ? | ? | ? |
| SE7-32 | model_bmrt.soc | model_fp32_1b | ? | ? | ? | ? |

> 时间单位: ms，已折算为每个样本的处理时间

### 编码器-解码器架构 (ASR)

| 测试平台 | 测试程序 | 测试模型 | preprocess | encoder_inference | decoder_inference | postprocess |
|----------|---------|---------|-----------|-------------------|-------------------|------------|
| SE7-32 | asr.py | encoder_fp32 | ? | ? | ? | ? |
| x86 PCIE | asr.py | encoder_fp32 | ? | ? | ? | ? |

> 时间单位: ms，RFT = total_time / audio_duration

### LLM 架构

| 测试平台 | 测试程序 | 测试模型 | prefill | decode_per_token | total_tokens | throughput(tokens/s) |
|----------|---------|---------|--------|------------------|-------------|----------------------|
| SE7-32 | llm.py | model_fp16 | ? | ? | ? | ? |

## 性能优化建议

1. **BMCV 预处理加速**: 将 OpenCV 预处理替换为 BMCV API，可显著减少 CPU 耗时
   - 参考: `sample/YOLOv5/cpp/yolov5_bmcv` vs `sample/YOLOv5/cpp/yolov5_opencv`
2. **批量推理 (multi-batch)**: 使用 batch>1 的 BModel，一次处理多个样本
   - 参考: `sample/ResNet` 的 `_4b.bmodel`
3. **流水线并行**: 预处理和 TPU 推理并行（当前样本推理时预处理下一个）
   - 参考: `sophon-pipeline` / `sophon-stream`
4. **INT8 量化**: 从 FP32 转为 INT8，通常可获得 2-4x 推理加速（需验证精度）
5. **模型融合 (fuse)**: 将前后处理融合进 BModel，减少 CPU-TPU 交互
   - 参考: `sample/YOLOv5_fuse`, `sample/PP-OCR` 的 `fuse` 模型
6. **SoC zero-copy**: 在 SoC 上使用 `bm_mem_mmap_device_mem` 避免 CPU-TPU 数据拷贝
7. **多核并行**: 使用 `num_core=2` 的 BModel，利用多 TPU 核心
   - 参考: `sample/ResNet` 的 `_2core.bmodel`
8. **后处理 CPU 加速**: 使用多线程或 TPU 实现 NMS 等耗时后处理
   - 参考: `sample/YOLOv5_opt` (TPU NMS)

## 检查清单

- [ ] bmrt_test 对所有 BModel 完成（记录 calculate time）
- [ ] Python 程序端到端完成 (x86 PCIe)
- [ ] Python 程序端到端完成 (SoC)
- [ ] C++ 程序端到端完成 (x86 PCIe)
- [ ] C++ 程序端到端完成 (SoC)
- [ ] N 次测试取平均 (N≥5)
- [ ] 各阶段耗时分析合理（推理时间应接近 bmrt_test 的 calculate time）
- [ ] 性能指标满足业务需求（FPS/QPS/RTF 达标）

## 测试说明通用模板

> **测试说明**:
> 1. 时间单位均为毫秒(ms)，统计的时间均已折算为平均每个样本的耗时；
> 2. 性能测试结果具有一定的波动性，建议多次测试取平均值；
> 3. SE5-16/SE7-32 的主控处理器均为 8 核 CA53@2.3GHz，SE9-16 为 8 核 CA53@1.6GHz，SE9-8 为 6 核 CA53@1.6GHz，PCIe 上的性能由于 CPU 性能差异可能存在较大不同；
> 4. 当 SoC 测试结果优于 PCIe 时，一般是 TPU-CPU 间 zero-copy 带来的收益；
> 5. bmrt_test 的 calculate time 为纯推理时间，程序端到端的 inference_time 包含了数据搬运和同步耗时，会比 calculate time 略高。

## 示例结果

### 单模型架构 (YOLOv5)

| 测试平台 | 测试程序 | 测试模型 | decode_time | preprocess_time | inference_time | postprocess_time |
|----------|---------|---------|------------|----------------|---------------|-----------------|
| SE7-32 | yolov5_bmcv.py | yolov5s_fp32_1b | 3.09 | 2.35 | 28.98 | 103.87 |
| SE7-32 | yolov5_bmcv.soc | yolov5s_fp32_1b | 4.32 | 0.74 | 21.63 | 15.91 |
| SE9-16 | yolov5_bmcv.py | yolov5s_fp32_1b | 4.53 | 4.85 | 107.05 | 143.24 |

### 编码器-解码器架构 (WeNet)

| 测试平台 | 测试程序 | 测试模型 | preprocess | encoder_inference | decoder_inference | postprocess |
|----------|---------|---------|-----------|-------------------|-------------------|------------|
| SE7-32 | wenet.py | encoder_streaming_fp32 | 3.32 | 23.69 | none | 8.69 |
| SE7-32 | wenet.py | encoder_streaming_fp32 + decoder_fp32 | 3.24 | 23.70 | 66.98 | 10.34 |
| SE7-32 | wenet.soc | encoder_streaming_fp16 | 25.73 | 5.13 | none | 1.00 |

### ASR RTF 示例 (SeACoParaformer)

| 测试平台 | 测试程序 | 测试模型 | preprocess(s) | encoder(s) | decoder(s) | total(s) | RTF |
|----------|---------|---------|-------------|-----------|-----------|---------|-----|
| x86 PCIE | seaco.py | encoder/decoder/predictor FP32 | 1.307 | 0.106 | 0.031 | 1.462 | 0.323 |
| SE7-32 | seaco.py | encoder/decoder/predictor FP32 | 4.046 | 0.113 | 0.051 | 4.234 | 0.937 |
| SE7-32 | seaco_bmrt.soc | encoder/decoder/predictor FP32 | 5.338 | 0.136 | 0.058 | 5.562 | 1.230 |
