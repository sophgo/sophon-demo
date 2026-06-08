# Skill 10: 文档更新与代码提交

## 目标
整理测试结果，更新 README 文档，提交代码到 Git/Gerrit。

## 执行步骤

### 10.1 整理测试结果

#### 精度测试结果
```
| 测试平台  | 测试程序          | 测试模型                       | 推理误差   | 推理误差   |
|-----------|-------------------|-------------------------------|-------|-------|
| x86 PCIE  | eval_accuracy.py  | 子模型1/子模型2/子模型3 FP32 | 0.00% | 0.00% |
```

#### 性能测试结果
```
| 测试平台  | 测试程序                    | preprocess(s) | 子模型1(s) | 子模型2(s) | total(s) | 延迟/吞吐指标   |
|-----------|----------------------------|--------------|------------|------------|----------|-------|
| x86 PCIE  | model_inference.py        | 1.307        | 0.106      | 0.031      | 1.462    | 0.323 |
| SE7-32    | model_inference.py        | 4.046        | 0.113      | 0.051      | 4.234    | 0.937 |
| SE7-32    | model_inference_bmrt.soc  | 5.338        | 0.136      | 0.058      | 5.562    | 1.230 |
```

### 10.2 更新 README 各节

#### 必填项
- [ ] 精度测试结果表（第 6.2 节）
- [ ] 性能测试结果表（第 7.2 节）
- [ ] 测试说明（确保描述准确）

#### 建议项
- [ ] 添加已知问题和解决方案
- [ ] 添加 C++ 编译说明
- [ ] 添加 SoC 部署步骤

### 10.3 README 更新模板

```markdown
## 6. 精度测试

### 6.2 测试结果

在目标模型的原始推理框架源码模型上，精度测试结果如下：
|   测试平台  |    测试程序               |              测试模型              | 推理误差 | 推理误差 |
| ---------- | ----------------------- | --------------------------------- | --- | --- |
| x86 PCIE   | eval_accuracy.py         | 子模型1/子模型2/子模型3 FP32     | 0.00% | 0.00% |
| SE7-32     | model_inference.py     | 子模型1/子模型2/子模型3 FP32     | TBD | TBD |

> **测试说明**：
> 1. 推理误差指标使用原始推理框架 PyTorch模型在CPU上的推理结果作为参考；
> 2. 测试使用公开测试集（如标准测试集 test）进行评估；
> 3. FP32精度应与PyTorch参考模型完全一致；

## 7. 性能测试

### 7.2 程序运行性能

|    测试平台  |     测试程序               | preprocess(s) | 子模型1(s) | 子模型2(s) | total(s) | 延迟/吞吐指标   |
| ----------- | ------------------------- | ------------- | ---------- | ---------- | -------- | ----- |
|   SE7-32    | model_inference.py       |  4.046       |  0.113     |  0.051     |  4.234   | 0.937 |
|   SE7-32    | model_inference_bmrt.soc |  5.338       |  0.136     |  0.058     |  5.562   | 1.230 |
|   x86 PCIE  | model_inference.py       |  1.307       |  0.106     |  0.031     |  1.462   | 0.323 |

> **测试说明**：
> 1. 延迟/吞吐指标 = total_time / input_data_duration；
> 2. 5次测试取平均值；
> 3. 测试输入数据：典型输入文件；
```

### 10.4 Git 提交

```bash
# 1. 暂存更改
git add sample/ModelName/README.md

# 2. 提交（amend 到上次提交或新建提交）
git commit --amend --no-edit  # 合并到上一个 commit
# 或
git commit -m "docs(ModelName): update accuracy and performance test results"

# 3. Rebase（如果远程有新提交）
git fetch origin developer
git rebase origin/developer

# 4. 推送（Gerrit 推荐使用 refs/for/ 分支）
git push origin HEAD:refs/for/developer

# 如果直接推送被拒绝，使用 force-with-lease（仅限自己的分支）
# git push origin developer --force-with-lease
```

### 10.5 Gerrit 提交规范

Gerrit 使用 `refs/for/` 方式提交代码审查：

```bash
# 标准 Gerrit push
git push origin HEAD:refs/for/developer

# 如果需要在 commit message 中添加 Change-Id
# Gerrit 会自动在 commit message 中插入 Change-Id
```

### 10.6 Commit Message 格式
```
<type>(<scope>): <short subject>

<longer body - optional>

Change-Id: <Gerrit auto-generated>
Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
```

类型 (type):
- `feat`: 新功能
- `fix`: 修复
- `docs`: 文档更新
- `perf`: 性能优化
- `refactor`: 重构
- `test`: 测试

## 检查清单

- [ ] 精度测试结果完整准确
- [ ] 性能测试结果完整准确
- [ ] README 表格格式正确
- [ ] 测试说明清晰
- [ ] git status 只包含预期文件
- [ ] Commit message 符合规范
- [ ] 推送成功 (Gerrit link 已生成)
- [ ] 如果有 Gerrit review link，发给 reviewer

## 示例提交记录

```
abc123 feat(ModelName): add model sample with SAIL inference
0792fa94 feat(FearTracker): add visual tracking sample with Python SAIL inference
b3985794 feat(Silero): add VAD sample with C++ bmrt and Python SAIL
```
