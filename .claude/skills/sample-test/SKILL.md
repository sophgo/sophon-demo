---
name: sample-test
description: 通用 sample 例程测试。在 x86 PCIe + TPU 环境下按 sample 的 README 验证例程可用性：跑 download.sh 拉数据/模型、用下载的模型跑推理、用 gen_*.sh 重编译后再跑一遍并对比，输出 pass/fail 报告。说「按照 sample-test skill 测试 XXX sample」即可。
---

# 通用 Sample 例程测试

> 目的：在当前 x86 PCIe + TPU 环境下，按某个 sample 的 README 把例程完整跑一遍，验证交付链路可用——下载脚本、下载的模型推理、编译脚本重编译、重编译后推理对比。输出结构化 pass/fail 报告。

## 适用范围

绝大多数 sophon-demo sample 共享同一套骨架：
- `scripts/download.sh` — 从 dfss 拉数据集 + 预编译 bmodel
- `scripts/gen_*.sh` — 在 tpu-mlir Docker 内把 ONNX 编译成 bmodel
- `README.md` — 文档化运行命令
- `python/*.py` — 推理脚本

本 skill 适配这套骨架。少数 sample 缺 download.sh 或 gen 脚本（如纯 LLM、ByteTrack），按「通用性适配」一节降级处理。

## 前置条件（环境自检，先跑这一步）

| 检查 | 命令 | 期望 |
|------|------|------|
| TPU 驱动 | `bm-smi` | 列出芯片（BM1684X / BM1688 等），无 error |
| SAIL | `python3 -c "import sophon.sail; print(sophon.sail.__version__)"` | 打印版本号 |
| tpu-mlir Docker | `docker ps -a --filter name=tpu_mlir` | 存在 `sophgo/tpuc_dev:v3.4` 容器 |
| unzip / dfss | `which unzip && python3 -m dfss --help` | 都可用 |

任一缺失→报告缺什么、怎么装，不要擅自往下跑。交叉环境意识：tpu-mlir 工具在 Docker 内（`/workspace == host ~/work/git_commits`），`sophon.sail` 在 host 跑。

## 流程（5 阶段）

### 阶段 1：探查 sample + 读 README

1. 确认 sample 目录 `sample/<NAME>/`，列出 `scripts/`、`python/`、`README.md`、`datasets/`、`models/`。
2. 读 README，定位「例程测试 / 运行」章节，提取：
   - 下载命令（通常是 `scripts/download.sh`）
   - 编译命令（`scripts/gen_*.sh` + 参数，如 `--res 504 --chip bm1684x --mode f16`）
   - 推理命令（`python3 python/xxx.py --model_dir ... --image ...` 等，含默认参数和测试图）
   - 期望输出（README 给的 demo 结果：检测框坐标、类别、输出图等）
3. 记下 sample 的芯片/精度约定（如 504→SoC、1008→PCIe），确认当前芯片匹配。不匹配→报告并停下，不要硬跑。

### 阶段 2：download.sh 验证

1. 备份现有 `datasets/`、`models/`（若有）到 `/tmp/<NAME>_bak/`，确保测的是全新下载。
2. 在 `sample/<NAME>/scripts/` 下跑 `bash download.sh`，全程记录日志到 `results/download.log`。
3. 校验：`datasets/`、`models/` 目录出现且文件非空；每个 bmodel 用 `bmrt_test --bmodel <path> --dev_id 0` 能加载（看 "Run ok" / 输入输出 shape 正确）。
4. download.sh 报错或文件缺失→阶段 FAIL，记下错误，不进阶段 3。

### 阶段 3：下载模型推理

1. 按 README 的推理命令，用**下载的**模型在 host 上跑（`sophon.sail`）。
2. 至少跑 README 示例用的那张测试图（通常在 `datasets/` 下）。多类 sample（检测/分割/OCR）跑 2-3 张代表性图。
3. 抓 stdout 到 `results/infer_<image>.log`，保留输出图到 `results/`。
4. Sanity check：输出非空、检测框/分割 mask 在合理范围、不报异常。和 README 给的 demo 结果对比——数值大致吻合即 PASS，明显偏离记为可疑并继续（不一定是错，编译/版本差异）。

### 阶段 4：重编译

1. 进 tpu-mlir Docker：`docker exec -it tpu_mlir bash`，`source /workspace/git_commits/tpu-mlir/envsetup.sh`。
2. `cd /workspace/git_commits/developer/sophon-demo/sample/<NAME>/scripts`，跑 README 提取的编译命令（如 `bash gen_bmodel.sh --res 504 --chip bm1684x --mode f16`）。
3. 日志记到 `results/gen_bmodel.log`。检查：无 UNREACHABLE / core dumped / Aborted / Traceback；产物 bmodel 落到 `models/` 对应子目录；`bmrt_test` 能加载。
4. 编译失败→阶段 FAIL，记错误。注意：gen 脚本常需 `--model NAME.bmodel` 带后缀（model_deploy.py context_dir 碰撞坑），具体见各 sample CLAUDE.md。

### 阶段 5：重编译后推理 + 对比

1. 用**重编译的** bmodel 重复阶段 3 的推理命令，输出记 `results/infer_<image>_recompiled.log`。
2. 逐图对比下载版 vs 重编译版：检测框坐标 / 分数 / 分割 IoU / 输出图。一致或差异在容差内→PASS；发散→记差异量级，按 sample 既有精度结论判断是否已知限制。
3. 清理：重编译产生的中间产物（.mlir/.npz/.profile/.json/per-model 目录）可删，保留 bmodel + results 日志。

## 报告格式（结论先行）

```
## <NAME> 例程测试报告

结论：✅ 全通过 / ⚠️ 部分通过 / ❌ 失败

| 阶段 | 结果 | 证据 |
|------|------|------|
| 环境自检 | ✅ | bm-smi 见 1 颗 BM1684X，sail 0.x.x |
| download.sh | ✅ | datasets/ 3 图，models/ 9 bmodel，bmrt_test 全 Run ok |
| 下载模型推理 | ✅ | truck 0.71 (.51,.46,.85,.46)，与 README demo 吻合 |
| 重编译 | ✅ | 9 bmodel 产出，无 core dump |
| 重编译推理对比 | ✅ | 与下载版逐图一致 |

失败阶段详情 / 已知限制 / 遗留问题……
```

## 通用性适配

| 样本特征 | 处理 |
|----------|------|
| 无 `download.sh`（如 DeepSeek、InternVL2） | 跳过阶段 2，README 指明模型来源就按其手动准备，否则报告"需手动备模型"并停 |
| 无 `gen_*.sh`（如 ByteTrack、InternVL3） | 跳过阶段 4-5，只测下载模型推理 |
| LLM / 对话类 | 推理命令是交互式或长文本，阶段 3 改跑 README 给的非交互示例 prompt，sanity check 改为"有合理文本输出、不报错" |
| 多芯片多精度 | 默认只测 README 主推的那组（通常 f16 + 当前芯片），其余精度按需在命令里加 |
| SoC only sample 在 PCIe 机上 | README 标 SoC 交付的也能在 PCIe 上跑（PCIe 兼容 SoC bmodel），照测；反过来 PCIe-only（如需大显存）在 SoC 上跑不了，报告并停 |

## 常见坑

| 现象 | 原因 | 解决 |
|------|------|------|
| `gen_*.sh` 报 context_dir 碰撞 / Save:424 | `--model` 没带 .bmodel 后缀 | 带后缀，见 sample CLAUDE.md |
| Docker 内找不到 model_transform.py | 没 `source envsetup.sh` | 进容器先 source `/workspace/git_commits/tpu-mlir/envsetup.sh` |
| 编译产物 root 属主，host 删不掉 | Docker 内以 root 跑 | 用 `docker run --rm -v <dir>:/tgt` 以容器 root 删，或 `sudo rm` |
| `download.sh` 卡在 dfss 下载 | 网络问题 | 重试，或检查 `python3 -m dfss --help` 能否连 |
| bmodel 加载失败 "magic number" | 文件没下全 / 被截断 | 重下，校验字节数 |
| 推理报 "device not found" | dev_id 不对或驱动没装 | `bm-smi` 确认芯片号，改 `--dev_id` |

## 执行纪律

- 不动 git：不 add/commit/push（除非用户明确要提交测试结果）。
- 改未被跟踪的文件前先备份到 `/tmp/<NAME>_bak/`。
- 测试后及时清理：SoC/host 上的临时 bmodel、解压临时目录、`/tmp` 中间产物删掉，保留 `results/` 日志和输出图。
- 重大基础设施（tpu-mlir 等）修复：先评估 update+rebuild，报告用户后再动。
- 汇报用中文，结论先行。
