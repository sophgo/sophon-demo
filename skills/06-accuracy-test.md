# Skill 6: 精度测试

## 目标
对比 TPU BModel 与 PyTorch 参考模型的推理结果，计算 CER/WER，验证精度无损。

## 执行步骤

### 6.1 准备参考推理
```python
# 使用 FunASR PyTorch 模型作为参考
from funasr import AutoModel

ref_model = AutoModel(model="./model", device="cpu")

# 对每个测试音频生成参考文本
for wav in test_wavs:
    result = ref_model.generate(input=wav)
    ref_text = result[0]["text"]  # 参考文本
```

### 6.2 运行 TPU 推理
```python
# 使用 sophon.sail BModel 推理
from seaco_paraformer import SeacoParaformer

tpu_model = SeacoParaformer("./models/BM1684X", dev_id=0)

for wav in test_wavs:
    audio = read_audio(wav)
    result = tpu_model.infer(audio)
    hyp_text = result["text"]  # TPU 推理文本
```

### 6.3 计算指标
```python
import editdistance

def char_error_rate(ref, hyp):
    """字符错误率 CER"""
    ref_chars = list(ref.replace(" ", ""))
    hyp_chars = list(hyp.replace(" ", ""))
    dist = editdistance.eval(ref_chars, hyp_chars)
    return dist / max(len(ref_chars), 1)

def word_error_rate(ref, hyp):
    """词错误率 WER (中文字符级)"""
    return char_error_rate(ref, hyp)  # 中文以字符为单位
```

### 6.4 单文件测试 (快速验证)
```bash
cd python
python3 eval_accuracy.py \
    --model_dir ../models/BM1684X \
    --input ../model/example/asr_example.wav \
    --ref "欢迎大家来到么哒社区进行体验"
```

### 6.5 批量测试 (完整评估)
```bash
# 准备 test manifest (JSONL 格式)
# {"audio_filepath": "path/to/audio.wav", "text": "reference text"}

python3 eval_accuracy.py \
    --model_dir ../models/BM1684X \
    --test_manifest test_manifest.txt \
    --audio_base /path/to/aishell \
    --max_samples 100 \
    --output results/accuracy.json
```

## 精度目标

| 模型类型 | CER 目标 | WER 目标 |
|----------|---------|---------|
| FP32 BModel | 与 PyTorch 一致 (0% diff) | 与 PyTorch 一致 |
| INT8 BModel | < 0.5% 相对退化 | < 1.0% 相对退化 |

## 调试方法

### 如果精度不达标:

1. **对比 logits**: 检查 TPU 和 PyTorch 的 logits 是否一致
   ```python
   # PyTorch logits
   pt_logits = pt_model.forward(audio)
   # TPU logits
   tpu_logits = tpu_model.decoder_forward(...)
   # 对比
   diff = np.abs(pt_logits - tpu_logits).max()
   ```

2. **检查预处理**: 确保 FBANK/LFR/CMVN 输出一致
   ```python
   pt_fbank = torchaudio.fbank(waveform, ...)
   cpp_fbank = armadillo_fbank(waveform, ...)
   diff = np.abs(pt_fbank.numpy() - cpp_fbank).max()
   ```

3. **逐层对比**: 对比每个子模型的输入输出
   - encoder 输出
   - CIF alphas 和 hidden
   - decoder logits
   - predictor us_alphas

4. **检查数据类型**: 确认 FP32 vs FP16 没有精度损失

## 检查清单

- [ ] 参考模型能正常推理
- [ ] TPU 模型能正常推理
- [ ] 单文件测试 CER=0% (FP32)
- [ ] 批量测试完成
- [ ] 结果 JSON 已保存
- [ ] 详细 per-sample 结果已审查

## 示例结果 (SeACoParaformer)

```
============================================================
Evaluation complete: 4 samples (0 skipped)
Overall CER: 0.0000 (0.00%)
Overall WER: 0.0000 (0.00%)
============================================================

Details:
  asr_example.wav: CER=0.0000, WER=0.0000
  asr_example_hotword.wav: CER=0.0000, WER=0.0000
  mid.wav: CER=0.0000, WER=0.0000
  long.wav: CER=0.0000, WER=0.0000
```
