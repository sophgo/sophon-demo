---
name: model-porting
description: 模型移植全流程。按顺序执行10个子步骤，从ONNX导出到SoC部署。使用前先填写 skills/model-porting-template.md 模板，然后说「按照 model-porting skill 移植[模型名]」即可。
---

# 模型移植

> 使用流程：填模板 → 一句话启动 → AI 自动按序执行 10 步。

## 前置条件

1. 确认 `skills/model-porting-template.md` 模板已按要求填写完整
2. 原始模型权重已下载到本地
3. 测试数据集已准备好

## 启动方式

```
按照 model-porting skill，移植 [模型名] 到 SOPHON TPU。
移植模板在 skills/model-porting-template.md，请先读取模板内容。
```

或者如果模板就是你当前在编辑的文件：

```
我已经填好了 skills/model-porting-template.md，按照 model-porting skill 开始移植。
```

## 执行流程

启动后，AI 自动按以下顺序调用子 Skill：

| 步骤 | 子 Skill | 做什么 |
|------|----------|--------|
| 01 | `model-porting-01-analyze` | 读取模板 + 原始模型 README → 输出 model_info.json |
| 02 | `model-porting-02-env` | 搭建 TPU-MLIR Docker 环境 |
| 03 | `model-porting-03-export` | PyTorch → ONNX，处理算子兼容 |
| 04 | `model-porting-04-compile` | ONNX → BModel，多芯片多精度 |
| 05 | `model-porting-05-python` | 生成 Python 推理代码 |
| 06 | `model-porting-06-accuracy` | 精度测试，输出对比表 |
| 07 | `model-porting-07-perf` | 性能测试，延迟+吞吐 |
| 08 | `model-porting-08-cpp` | Python → C++ 移植 |
| 09 | `model-porting-09-soc` | SoC 交叉编译 + 部署 |
| 10 | `model-porting-10-docs` | 生成 README + 文档 |

每个步骤 AI 会：
1. 调用对应子 Skill 加载上下文（只加载当前步骤的知识，不浪费 token）
2. 根据子 Skill 中的提示词执行任务
3. 完成后展示输出，确认通过后进入下一步

## 执行规则

- **串行执行**：每步通过后才进入下一步
- **失败处理**：遇到问题用 Debug 四步曲（编号→写TC→分析→修复验证）
- **产物提交**：每步完成后产物提交 Git
