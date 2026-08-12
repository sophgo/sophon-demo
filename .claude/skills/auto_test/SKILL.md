---
name: auto_test
description: Use when the user wants to README-walkthrough-test sophon-demo samples on a TPU device — triggers like「自动测试例程」「README 走查」「按 README 测例程」「测所有例程」「sample 走查」. auto_test.sh 已全部过时，以各 sample 的 README 为唯一事实来源。远程设备先索取账号/IP/密码/设备类型；本地直跑（用户明确说本机）则免。Applies when a BM1684X/BM1688/CV186X/BM1684 device is available.
---

# sophon-demo 例程 README 走查测试

> 目的：在目标设备上，**以每个 sample 的 README 为唯一事实来源**，按文档把例程完整走一遍，验证"照 README 能否复现"，并产出 **README 缺陷清单**（这是走查的核心价值）。
> **关键认知：各 sample 的 `scripts/auto_test.sh` 已全部过时**（用例集/数值基线与现状不符），**一律不作为判据，仅作参考**。判定时只信 README。
> 第一动作：远程设备先索取 账号/IP/密码/设备类型；**本地直跑**（用户明确说在本机测）跳过索取。

## 1. 要设备（远程才要，本地跳过）

远程设备用这段话索取（4 个必填项不能少）：

> 我需要一台设备来跑 sophon-demo 例程 README 走查。请提供：
> 1. **设备类型**：BM1684X / BM1688 / CV186X / BM1684（知道具体型号 SC7/SE7-32/SE9-16/SE9-8/SE5-16 也告诉我）
> 2. **设备 IP**  3. **登录账号**  4. **登录密码**
> 5. **连接形态**：PCIe（x86 主机插卡，编译+运行都在该主机）还是 SoC（边缘设备）？不确定就问。
> 6. （可选）TPU dev id，默认 0
> 7. （可选）只走查某些 sample，默认全量

**Red Flag — STOP：** 远程设备凭据未齐就动手 = 违反本 skill。本地直跑（用户已说明"本机/本地，型号 XXX"）不需要密码，直接进入走查。

## 2. 设备信息 → 走查参数映射

| 用户提供 | 映射到 |
|---|---|
| 设备类型 BM1684X/BM1688/CV186X/BM1684 | 决定读 README 的哪一节（BM1684X PCIe 看 PCIe/x86 段与 `models/BM1684X/`；SoC 看对应段） |
| 具体型号（SoC） | `PLATFORM`：BM1684X→SE7-32、BM1684→SE5-16、BM1688→SE9-16、CV186X→SE9-8 |
| PCIe 形态 | 本机原生 `cmake .. && make` 编 C++ |
| SoC 形态 | 需 x86 主机交叉编译（`SOPHON_SDK_SOC`），或设备自托管原生编译 |
| TPU dev id | 推理命令的 `--dev_id` |

## 3. 连接设备 + 准备代码（本地直跑跳过 1-3 的 SSH 部分）

1. （远程）SSH 探活：`sshpass -p '<pwd>' ssh -o StrictHostKeyChecking=no <user>@<ip> 'uname -a; bm-smi; ls /opt/sophon'`。
2. 建工作目录 `~/code-review/`，取代码（联网 `git clone`，否则从本地 scp）。
3. 记录受测 commit：`git -C <repo> log -1 --format='%H %s'`。
4. 基础依赖：`unzip p7zip-full libeigen3-dev`；`pip3 install dfss pycocotools opencv-python-headless`（清华源）。确认 `which bmrt_test`、`ls /opt/sophon/sophon-sail/lib` 在。
5. **Python 环境隔离**：各 sample 依赖可能互相冲突，**优先用 `python3 -m venv` 按 sample 建隔离环境**，不要全局乱装（LLM 类对 transformers/torch 版本敏感）。

**凭据安全：** 密码只出现在 `sshpass` 命令行，用 SSH ControlMaster 复用连接，任何日志/报告不回显密码。

## 4. 走查范围（不限于有 auto_test.sh 的样例）

对象 = **所有有 `README.md` 的 sample**（`sample/<NAME>/README.md`），不只是有 auto_test.sh 的那 47 个。先 triage 分桶：

```bash
cd <repo>/sophon-demo/sample
for d in */; do
  n=${d%/}
  [ -f "$n/README.md" ] || continue
  chips=$(grep -oiE "BM1684X|BM1688|CV186X|BM1684" "$n/README.md" | sort -u | tr '\n' ',' )
  echo "$n :: $chips"
done
```

- **本机可测**：README 声明支持本机芯片 → 进走查队列。
- **SKIP-for-chip**：README 只声明其它芯片（如纯 BM1688/CV186X SoC-only、无 PCIe 段）→ SKIP，注明。
- **SKIP-硬件**：需摄像头/HDMI 显示/麦克风/多卡/特定外设而本机没有 → SKIP，注明。
- **超重 LLM/VLM/AIGC**（Qwen、Llama2、MiniCPM、DeepSeek、InternVL、FLUX、StableDiffusion、Whisper、FunASR…）：模型 GB 级 + 专用 Python 环境，**先问用户是否纳入**（默认纳入但放最后，逐个 venv）。

triage 后给用户一份"将走查 N 个 / SKIP 哪些"的清单再开跑（全量时）。

## 5. 每个 sample 的走查流程（核心）

严格按 README 步骤执行，逐步记录"文档说的 vs 实际发生的"。要读的 README：`sample/<N>/README.md` + `sample/<N>/python/README.md` + `sample/<N>/cpp/*/README.md`（存在才读）。

### 5.1 读 README，提取走查清单
从 README 提取：
- 支持芯片/平台声明（本机是否在列）
- 依赖安装命令（apt / pip / requirements.txt / 环境变量 / docker）
- 数据+模型下载命令（通常 `scripts/download.sh`，或手动 dfss/wget/编译）
- Python 推理命令 + **README 给的期望输出**（demo 结果、检测框、数值、输出图）
- C++ 编译命令 + 推理命令 + 期望输出
- 精度测试命令 + **README 给的期望精度数值**
- 性能测试命令 + README 给的性能数值（如有）

### 5.2 芯片/平台门禁
README 支持芯片不含本机 → SKIP-for-chip，注明，不进后续。

### 5.3 装依赖（按 README）
- 严格按 README 命令装（venv 内）。README 没写但运行报缺的依赖 → 记 **DOC_DEFECT（依赖未文档化）**。
- 装不上（版本冲突/源不可达）→ 记录实际报错，按 README 意图尽量继续，记偏差。

### 5.4 下载数据/模型（按 README）
- 按 README 的下载命令跑。校验产物存在且非空；bmodel 用 `bmrt_test --bmodel <p> --dev_id <id>` 能加载（"Run ok"）。
- 下载失败/缺文件 → FAIL（附错误）；README 指向的 URL/路径失效 → **DOC_DEFECT**。

### 5.5 Python 推理（按 README）
- 严格用 README 的命令 + README 指定的测试输入跑。
- 对照 README 期望输出：一致/容差内 → OK；明显不符但能跑 → **DOC_DEFECT（数值/结果过时）**；跑不通 → FAIL。

### 5.6 C++ 编译+推理（按 README）
- 按 README 编译命令（PCIe 原生 cmake/make；SoC 交叉 SDK）。
- 编译失败：README 命令/路径写错 → **DOC_DEFECT**；源码/环境缺东西 → FAIL。
- 按 README 跑 C++ 推理，对照期望输出，同 5.5 判定。

### 5.7 精度 / 性能（README 有则跑）
- 精度：按 README 命令跑，对照 README 给的期望数值。明显偏离 → DOC_DEFECT 或 FAIL。
- 性能：可选，只作参考（随环境波动），**不作 FAIL 依据**。

### 5.8 清理
每个 sample 测完删其 `datasets/` `models/` `results/` 中间产物（磁盘有限），保留顶层日志。

### 判定（每 sample 一个结论）
- ✅ **PASS**：README 全链路（下载→Python→C++→精度）按文档复现，输出与文档一致。
- ⚠️ **DOC_DEFECT**：例程能跑通，但 README 有缺陷（错命令/错路径/过时数值/缺步骤/失效链接/缺依赖）。逐条列缺陷——**走查核心产出**。
- ❌ **FAIL**：按 README（及合理推断）跑不通。附真实报错。
- ⏭️ **SKIP**：本机硬件/平台不支持（注明：纯 SoC、需摄像头/HDMI、需多卡、模型超显存等）。

## 6. 驱动循环

- 逐个 sample 走 §5 流程（本机单 TPU，**串行**，避免 TPU/磁盘/带宽争用）。
- 单 sample 超时（下载卡死/死循环）→ 记 TIMEOUT，kill 后继续，不卡死全流程。
- **流式报进度**：每完成一个报 `[i/N] <NAME> ✅/⚠️/❌/⏭️ 一句话`。
- 不改样例源码/脚本去凑通过；不动 git；密码不落盘。

## 7. 报告格式（结论先行，中文）

```
## sophon-demo README 走查报告

结论：PASS a / DOC_DEFECT b / FAIL c / SKIP d（共 N 个有 README 的 sample）

- 设备：<型号> @ <ip>（<类型>，<PCIe|SoC>，dev_id=<id>）
- 受测 commit：<hash>   形态/目标：<…>   耗时：<hh:mm:ss>
- 判据：只信 README；auto_test.sh 已过时仅参考。TIMEOUT≠FAIL。

| 样例 | 结论 | 说明 |
|------|------|------|
| ResNet | ✅ PASS | 下载/Python/C++/精度全复现，数值与文档一致 |
| YOLOv5 | ⚠️ DOC_DEFECT | 能跑通，但 README 精度数值过时（见缺陷清单 #3） |
| CLIP | ❌ FAIL | bmodel 加载 invalid bmodel |
| SAM3 | ⏭️ SKIP | 纯 SoC，本机 PCIe 不支持 |

## README 缺陷清单（核心产出）
| # | 样例 | 文件:行 | 缺陷 | 正确应为 |
|---|------|---------|------|----------|
| 1 | YOLOv5 | README.md:120 | 精度数值与实测不符 | 更新为实测值 X |
| … | | | | |

FAIL 详情（每个摘关键报错）：…
SKIP 清单（含原因）：…
```

## 8. 常见坑

| 现象 | 原因 | 处理 |
|------|------|------|
| `import sophon.sail` 失败 | sophon-sail 未装或 `LD_LIBRARY_PATH` 缺 `/opt/sophon/sophon-sail/lib` | 装包 + export；venv 里也要能 import |
| README 命令直接报错 | README 过时（这正是走查要抓的） | 记 DOC_DEFECT，按 README 意图修正后继续，记偏差 |
| 依赖装不上/版本冲突 | README 依赖清单过时或与全局冲突 | 用 venv 隔离；记 DOC_DEFECT（依赖未文档化/版本未锁） |
| dfss 下载慢/卡 | 网络慢（~2.8MiB/s），大包久 | 耐心等；超时记 TIMEOUT；无外网则从本地 scp |
| bmodel 加载 "magic number" / invalid | 下载被截断/文件坏 | 重下并校验字节数；README 模型路径错 → DOC_DEFECT |
| C++ cmake 找不到 SAIL/opencv | 缺 `-DSAIL_PATH` 或 `LD_LIBRARY_PATH` | 按 README 补；README 没写 → DOC_DEFECT |
| 需摄像头/HDMI/麦克风 | 本机无外设 | SKIP-硬件，注明 |
| LLM 环境互相污染 | transformers/torch 版本各 sample 不同 | 每 sample 独立 venv |
| 磁盘满 | 多样例累积 datasets/models | 每 sample 测完即清 §5.8 |

## 9. 执行纪律

- **以 README 为唯一事实来源**；auto_test.sh 已过时，仅参考不作判据。
- 远程设备凭据未齐不动手；本地直跑免密码（见 §1 Red Flag）。
- 不改源码/脚本/模型去凑通过；不改 git。
- 走查核心价值是 **README 缺陷清单**——能跑通但文档有错的，一定逐条记。
- 测完清理每 sample 大文件，保留日志。
- 密码不进任何日志/报告。
- 结论先行、中文、流式报进度。
