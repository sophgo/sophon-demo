# Skill 1: 阅读和分析 README

## 目标
从项目 README 中提取关键信息，理解模型架构、数据流和测试要求。

## 执行步骤

### 1.1 识别模型架构
- 阅读 README 中的模型架构说明
- 确认模型类型（编码器-解码器 / 纯编码器 / 非自回归等）
- 确认输入输出规格（采样率、特征维度、token 数）

### 1.2 确认推理 pipeline
```
输入数据 → 预处理 → 编码器(TPU) → 中间处理(CPU) → 解码器(TPU) → 预测器(TPU) → 解码(CPU) → 文本
```

### 1.3 确认测试要求
- 精度测试：参考模型（原始推理框架 PyTorch）、测试集（标准测试集）、指标（推理误差指标）
- 性能测试：bmrt_test 理论性能 + 程序运行 延迟/吞吐指标

### 1.4 确认文件结构
- `models/BM1684X/` — BModel 文件
- `python/` — Python SAIL 推理代码
- `cpp/` — C++ bmrt 推理代码
- `scripts/` — 下载/编译脚本

### 1.5 查找相似已有 Sample（重要）
在开发新 Sample 之前，必须先找到 sophon-demo 中已有最相似的 Sample 作为参考：
- **按算法类别查找**: 进入 `sample/` 目录，根据模型类型（分类/检测/ASR/OCR/人脸等）找到同类 sample
- **按模型架构查找**: 确认是单模型/编码器-解码器/级联多模型，找到架构最接近的 sample
- **阅读参考 Sample 的 README**：了解其章节结构、表格格式、测试说明写法
- **阅读参考 Sample 的 scripts**：了解 download.sh、gen_*bmodel_mlir.sh、auto_test.sh 的脚本结构和参数模式
- **阅读参考 Sample 的 python/cpp 代码**：了解推理代码结构、前后处理实现方式

> 算法类别与推荐参考 Sample 的对应关系，见 `model-porting-template.md` 第 11.2 节。

## 检查清单

- [ ] 理解模型架构（子模型1/子模型2/子模型3 结构）
- [ ] 确认输入格式（模型要求的输入格式, 适配模型输入的特征参数）
- [ ] 确认输出格式（文本 + 词级别辅助输出）
- [ ] 确认依赖（libsophon, sophon-sail, torch, 框架预处理工具, numpy）
- [ ] 理解精度/性能测试方法
- [ ] 确认目标平台（BM1684X, x86 PCIe / SE7-32 SoC）

## 示例: 目标模型

```
输入: 模型要求的输入格式 → 特征提取(特定参数) → 降采样 → 归一化 → [1, T, 560]
编码器: [1, T, 560] → submodel1_out[1, T, 512], hidden_state[1, T+1, 512], intermediate_values[1, T+1], output_length[1]
中间处理: hidden_state + intermediate_values → intermediate_embeds[1, N, 512]
解码器: [submodel1_out, intermediate_embeds] → logits[1, N, 8404]
预测器: submodel1_out → aux_data[1, T*3], submodel3_output_length[1]
解码: argmax(logits) → output_ids → results → final_output
输出: text + [start_ms, end_ms, token] 辅助输出列表
```
