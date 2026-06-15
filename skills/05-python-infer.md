# Skill 5: Python 推理验证

## 目标
使用 sophon.sail 加载 BModel 进行 Python 推理，验证模型在 TPU 上能正常输出。

## 开发前准备

> **重要**: 在编写 Python 推理代码前，先找到 sophon-demo 中已有的类似 Sample 作为参考模板。
> - 根据算法类别（分类/检测/ASR/OCR/人脸等）查找最相似的 sample
> - 参考其 `python/` 目录下的代码结构、sophon.sail API 使用方式、命令行参数设计
> - 参考其预处理和后处理实现
> - 保留参考代码的整体结构，根据新模型的输入输出规格修改具体逻辑
> - 算法类别与推荐参考 Sample 的对应关系，见 `model-porting-template.md` 第 11.2 节

## 执行步骤

### 5.1 加载 BModel
```python
import sophon.sail as sail

子模型1 = sail.Engine("submodel1_fp32.bmodel", dev_id=0, mode=sail.IOMode.SYSIO)
子模型2 = sail.Engine("submodel2_fp32.bmodel", dev_id=0, mode=sail.IOMode.SYSIO)
子模型3 = sail.Engine("submodel3_fp32.bmodel", dev_id=0, mode=sail.IOMode.SYSIO)
```

### 5.2 实现预处理 (CPU)
```python
# 1. 读取输入数据 (符合模型要求的格式)
input_data = read_input_data("test_input")

# 2. 特征提取 特征提取 (80 维 mel 滤波器组)
features = framework.preprocess(input_data,
                    frame_shift=10, sample_frequency=16000)

# 3. 降采样 (Low Frame Rate): m=7, n=6
features = apply_preprocessing(raw_features)  # → [T, feature_dim]

# 4. 归一化 (倒谱均值方差归一化)
features = normalize(features)
```

### 5.3 逐模型推理
```python
# 编码器
enc_in = {"input_data": input_data[None,:,:], "input_lengths": np.array([input_len])}
submodel1_out = 子模型1.process(graph_name, enc_in)
# 输出: submodel1_out, hidden_state, intermediate_values, output_length

# 中间处理 (CPU)
intermediate_embeds, _ = intermediate_process(submodel1_out)
intermediate_embeds = intermediate_embeds[:, :int(output_length), :]

# 子模型2
dec_in = {"enc": submodel1_out, "enc_len": ..., "embeds": intermediate_embeds, ...}
submodel2_out = 子模型2.process(graph_name, dec_in)
# 输出: logits, dec_hidden_state

# 预测器 V3
pred_in = {"enc": submodel1_out, "enc_len": ...}
submodel3_out = 子模型3.process(graph_name, pred_in)
# 输出: aux_data, submodel3_output_length
```

### 5.4 解码 (CPU)
```python
# Greedy 解码
output_ids = np.argmax(logits[0, :N, :], axis=-1)
output_ids = [t for t in output_ids if t not in special_tokens]
results = [vocab[oid] for oid in output_ids]
final_output = post_process(results)

# 辅助输出预测
aux_data = intermediate_values2 * (reference_output_length / submodel3_output_length)
aux_peaks = secondary_process(aux_data, threshold=1.0 - 1e-4)
aux_outputs = aux_output_prediction(aux_data, tokens)
```

### 5.5 运行测试
```bash
cd python
python3 model_inference.py \
    --model_dir ../models/BM1684X \
    --input ../data/test_input
```

## 关键验证点

1. **输入 shape 匹配**: 确保输入 tensor shape 和 dtype 与 BModel 的输入描述一致
2. **输出解析**: 确认每个输出 tensor 的含义和 shape
3. **文本解码**: 验证 输出ID → 输出符号 → 最终结果的映射正确
4. **辅助输出**: 验证 中间处理峰值检测和辅助输出计算的正确性
5. **内存管理**: SoC 模式下使用 zero-copy (mmap)，PCIe 模式使用 d2s 拷贝

## 检查清单

- [ ] BModel 加载成功 (无 BMRT_ASSERT)
- [ ] 预处理输出 shape 正确
- [ ] 编码器输出 shape 符合预期
- [ ] 中间处理正确计算
- [ ] 解码器输出 logits shape 正确
- [ ] Greedy 解码输出有效文本
- [ ] 辅助输出在合理范围内
- [ ] 无 NaN 或 Inf

## 常见问题

1. **BMRT_ASSERT 错误**: bmodel 与 libsophon 版本不兼容
2. **输出全空**: 中间处理未产生有效输出，检查输入数据和 归一化
3. **解码乱码**: 检查 model_config_files 与模型是否匹配
4. **内存溢出**: 检查动态 shape 的实际大小是否超过编译时的 max
