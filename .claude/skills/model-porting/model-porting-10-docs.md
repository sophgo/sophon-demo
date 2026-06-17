---
name: model-porting-10-docs
description: 文档更新。基于精度/性能报告和移植经验，生成 README 和相关文档。
---

# 步骤 10：文档更新

## 前置

所有前序步骤已完成，精度报告和性能报告已生成。

## 提示词

```
生成和更新文档：

1. README.md（主文档）：
   - 模型简介（算法类别、框架、精度规格、支持的芯片）
   - 精度对比表（Python/C++，按芯片+精度分列）
   - 性能测试表（延迟+吞吐，按芯片+精度+batch 分列）
   - 编译运行指南（环境准备→模型下载→编译→推理）
   - FAQ（本次移植中遇到的典型问题+解决方案，至少 3 条）
   - 交叉引用：../../docs/Environment_Install_Guide.md、../../docs/FAQ.md
   - 交叉引用：python/README.md、cpp/README.md

2. docs/[ModelName]_Export_Guide.md：
   - ONNX 导出步骤、算子替换说明、验证方法

3. python/README.md：
   - 目录、环境准备、推理测试（参数说明、使用方式）、性能测试

4. cpp/README.md（如有 C++）：
   - 目录、环境准备（PCIe+SoC）、编译方法、推理测试、性能测试

格式参考已有 sample 的 README 风格，表格用统一格式。
待补充数据标为「待补充」。
```

## 预期输出

- `README.md`
- `docs/[ModelName]_Export_Guide.md`
- `python/README.md`
- `cpp/README.md`

## 内联知识：README 标准结构

```markdown
# [SampleName]

## 目录
## 简介
## 特性
## 准备模型与数据
## 模型编译
## 例程测试
### Python 例程
### C++ 例程
## 精度测试
## 性能测试
## FAQ
```

### FAQ 最小集

至少包含以下问题：
1. 如何下载模型和数据集？（引用 download.sh）
2. 如何编译 BModel？（引用 gen_*bmodel_mlir.sh）
3. 精度不达标怎么办？（对齐预处理参数、确认校准数据）
4. （本次移植遇到的特有问题）
