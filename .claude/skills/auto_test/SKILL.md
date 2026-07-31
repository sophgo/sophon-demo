---
name: auto_test
description: Use when the user wants to automatically test sophon-demo routines on a remote TPU device — triggers like「自动测试例程」「跑 auto_test」「回归测试 sophon-demo」「测所有例程」, and device 账号/IP/密码/设备类型 will be requested first. Applies when a BM1684X/BM1688/CV186X/BM1684 device is available for SSH access.
---

# sophon-demo 例程自动测试（远程设备）

> 目的：在一台 TPU 设备上，把 sophon-demo 里**自带 `scripts/auto_test.sh` 的样例**逐个跑通，输出 per-sample pass/fail 报告。
> **第一动作永远是向用户要设备**（账号/IP/密码/设备类型），拿到之前不连机器、不跑任何测试、不改任何文件。

## 1. 第一步：向用户要设备（拿到前禁止动手）

用这段话向用户索取（可按上下文精简，但 4 个必填项不能少）：

> 我需要一台设备来跑 sophon-demo 例程自动测试。请提供：
> 1. **设备类型**：BM1684X / BM1688 / CV186X / BM1684（若知道具体型号如 SC7/SE7-32/SE9-16/SE9-8/SE5-16 也一并告诉我，用于 SoC 模式选 PLATFORM 与双核变体）
> 2. **设备 IP**
> 3. **登录账号**
> 4. **登录密码**
> 5. **连接形态**：PCIe（x86 主机插加速卡，编译+测试都在该主机）还是 SoC（边缘设备，本机直接跑 `soc_test`）？默认按设备类型推断：BM1684X 可能是 PCIe 也可能是 SoC，BM1688/CV186X/BM1684 通常按 SoC 处理——**不确定就问**。
> 6. （可选）TPU dev id，默认 0
> 7. （可选）只跑某些样例，默认全跑

**Red Flag — STOP：** 用户没给齐账号/IP/密码/设备类型就动手连机器、跑测试、或改文件 = 违反本 skill。先要设备，再做事。

## 2. 设备信息 → 测试参数映射

| 用户提供 | 映射到 |
|---|---|
| 设备类型 BM1684X/BM1688/CV186X/BM1684 | `TARGET`（传给 `auto_test.sh -t`） |
| 具体型号（SoC 模式） | `PLATFORM`：BM1684X→SE7-32、BM1684→SE5-16、BM1688→SE9-16、CV186X→SE9-8（部分脚本会按 `nproc` 把 6 核 BM1688 映射为 SE9-8，非所有脚本都有此分支，不影响 `int8_4b_2core` 用例的触发——那由 `TARGET=BM1688` 决定） |
| PCIe 形态 | `MODE=pcie_test`（先 `pcie_build` 编 C++） |
| SoC 形态 | `MODE=soc_test`（若有 x86 交叉编译主机则先 `soc_build`，否则要求设备自托管可原生编译/或已有预编译产物） |
| TPU dev id | `-d` |

**SoC 交叉编译坑：** SoC 的 C++ 例程通常需在 x86 主机用 `SOPHON_SDK_SOC` 交叉编译再拷到设备。若用户给的是纯 SoC 设备且无 x86 主机，先确认设备上是否已能 `cmake .. -DTARGET_ARCH=soc` 原生编译或已有预编译 `.so`/二进制；都不行就只跑 Python 用例（`auto_test.sh` 的 Python 部分不依赖 C++ 编译），并如实报告 C++ 用例被跳过。

## 3. 连接设备 + 准备代码（凭据到位后才执行）

1. SSH 探活：`sshpass -p '<pwd>' ssh -o StrictHostKeyChecking=no <user>@<ip> 'uname -a; bm-smi -V; ls /opt/sophon'`，确认芯片型号与 libsophon 在位。
2. 建工作目录 `~/code-review/`。
3. 取代码（优先级）：
   - 设备能联网 → `git clone`（用与本地副本相同的远端）。
   - 否则 → 从本地 `scp -r /home/lihengfang/work/git_commits/code-review/sophon-demo <user>@<ip>:~/code-review/`。
4. 记录受测 commit：`git -C ~/code-review/sophon-demo log -1 --format='%H %s'`。
5. 装依赖（参照 `scripts/auto_test_regression.sh`）：`unzip p7zip p7zip-full`，PCIe/soc_build 还需 `libeigen3-dev`；`pip3 install pycocotools opencv-python-headless`（用清华源 `-i https://pypi.tuna.tsinghua.edu.cn/simple`）。**另需 `pip3 install dfss`**（各样例 `download.sh` 依赖它拉数据/模型，regression 脚本未含但 download 必需）。确认 `which bmrt_test`、`ls /opt/sophon/sophon-sail/lib` 存在。

**凭据安全：** 密码会出现在 `sshpass` 的命令行（`ps` 可见）。尽量用 SSH ControlMaster 复用连接（`-M -S /tmp/ssh-<ip>`），减少带密码命令次数；任何日志/报告里都不要回显密码。

## 4. 测试范围（关键：不是所有样例都能自动测）

`sophon-demo/sample/` 下约 99 个目录，**只有约 47 个自带 `scripts/auto_test.sh`**。用这条命令在设备上列出可测样例：
```bash
find ~/code-review/sophon-demo/sample -maxdepth 3 -name auto_test.sh | sed 's#.*/sample/##; s#/scripts/auto_test.sh##' | sort
```
- **只跑有 `auto_test.sh` 的样例。**
- **没有 `auto_test.sh` 的样例（多为 LLM/VLM：Qwen、Llama2、ChatGLM2、DeepSeek、FLUX.1、SAM、InternVL2 等，及新增样例）一律 SKIP**，并在报告里按名列出"无自动测试脚本，已跳过"。不要为它们临时手搓测试。
  - 注意 `SAM` 无脚本（SKIP），但 **`SAM2` 有 `auto_test.sh`，必须跑**——别因前缀相同误跳。最终以 `find` 命令的输出为准，不要凭名字猜。

## 5. 每个样例的测试流程（sophon-demo）

`auto_test.sh` 已封装好下载+推理+精度+性能。驱动方式复刻 `scripts/auto_test_regression.sh`，但**遍历全部 47 个有脚本的样例**（不局限于 regression 脚本里硬编码的那十几个）。

### 模式选择

- **PCIe**：先 `pcie_build` 再 `pcie_test`：
  ```bash
  cd ~/code-review/sophon-demo/sample/<NAME>
  chmod +x scripts/auto_test.sh
  ./scripts/auto_test.sh -m pcie_build -t <TARGET> -d <TPUID>    # 编 C++
  ./scripts/auto_test.sh -m pcie_test  -t <TARGET> -d <TPUID> -c fully
  ```
- **SoC**：有 x86 交叉编译主机则先 `soc_build`，再 `soc_test`：
  ```bash
  ./scripts/auto_test.sh -m soc_build -t <TARGET> -s <SOCSDK> -d <TPUID>     # 编 C++（-s 仅此模式用）
  ./scripts/auto_test.sh -m soc_test  -t <TARGET>              -d <TPUID> -c fully
  ```
  （`-s <SOCSDK>` 仅 `soc_build` 用；`soc_test` 不需要，传了也被忽略）

### sail_list 样例需多传 `-a`

这 9 个 C++ 用例用 SAIL 接口，必须传 SAIL 路径 `/opt/sophon/sophon-sail`：
`YOLOv5、CenterNet、BERT、ppYOLOv3、YOLOv34、YOLOX、segformer、ppYoloe、YOLOv5_opt`
→ 对它们加 `-a /opt/sophon/sophon-sail`（其余样例不要加）。

### 通过判定（不要自己造标准）

> 注：`*_test` 模式内部会先调 `download.sh` 拉数据/模型（依赖 dfss + 外网 + 磁盘，耗时较长），不要事先再单独跑一次 download。

`auto_test.sh` 内部用 `ALL_PASS` 累积，结尾二选一打印：
- `Test cases all pass!` → **PASS**
- `Some process produced unexpected results, please look out their logs!` → **FAIL**

只认这两行；不要自行判断精度数值是否合格（脚本内 `compare_acc.py`/`compare_statis.py` 已对比基线）。

### 驱动循环（每个样例）

```bash
cd ~/code-review/sophon-demo/sample/<NAME>
chmod +x scripts/auto_test.sh
# 按 MODE / 是否在 sail_list 组装参数
./scripts/auto_test.sh -m <MODE> -t <TARGET> [-s <SOCSDK>] [-a <SAIL>] -d <TPUID> -c fully \
  > ~/code-review/logs_demo/<NAME>.log 2>&1
tail -n 4 ~/code-review/logs_demo/<NAME>.log   # 取 PASS/FAIL 行
```
- PASS → 记录 PASS，进入下一个。
- FAIL → `tail -n 40` 该日志，并 `grep -E "Failed:|Error|Traceback|core|Aborted" logs_demo/<NAME>.log | head`，把真实错误摘进报告。
- 单个样例超时（如卡在下载或死循环）→ 记 TIMEOUT，kill 后继续下一个，不要卡死全流程。
- 每个样例测完清理其 `datasets/`、`models/`、`results/`、中间产物（磁盘有限），保留 `log/` 与顶层 `logs_demo/<NAME>.log`。

### 不要做的事

- 不跑 `compile_mlir`/`compile_nntc` 模式（重编 bmodel 需 MLIR 工具链，耗时且易环境问题）——只用 `download.sh` 拉的预编译 bmodel。用户明确要重编时再加。
- 不 `git add/commit/push`（除非用户明确要提交测试结果）。
- 不改样例源码/脚本去"让它过"。

## 6. 报告格式（结论先行，中文）

```
## sophon-demo 例程自动测试报告

结论：<PASS数>/<总数> 通过；FAIL <n>；SKIP <n>

- 设备：<型号> @ <ip>（<设备类型>，<PCIe|SoC>，dev_id=<id>）
- 受测 commit：<hash>
- 模式：<MODE>，TARGET=<TARGET>
- 耗时：<hh:mm:ss>

| 样例 | 结果 | 说明 | 日志 |
|------|------|------|------|
| YOLOv8_plus_det | ✅ PASS | 全 16 个 model×精度用例通过 | logs_demo/YOLOv8_plus_det.log |
| ResNet | ❌ FAIL | cpp int8_4b eval 精度偏离基线 | logs_demo/ResNet.log（tail 见下） |
| … | | | |

SKIP（无 auto_test.sh，共 <n> 个）：ArcFace、Baichuan2、BLIP、ChatGLM2、DeepSeek、FLUX.1、InternVL2、Llama2、SAM、Qwen…（按 `find` 输出列全；注意 SAM2 有脚本、不在本列）

FAIL 详情（每个 FAIL 摘最后 ~20 行关键报错）：
### ResNet
...
```
流式汇报：每跑完一个样例就向用户报一句进度（`[12/47] YOLOv8_plus_det PASS`），不要闷头跑完才出声。

## 7. 常见坑

| 现象 | 原因 | 处理 |
|------|------|------|
| `bmrt_test` not found / `import sophon.sail` 失败 | libsophon/sophon-sail 未装或 `LD_LIBRARY_PATH` 缺 `/opt/sophon/sophon-sail/lib` | 装包；`auto_test.sh` 已 export 该路径，手动跑 Python 时记得也 export |
| `download.sh` 卡住 / dfss 连不上 | 设备无外网或 dfss 未装 | 先 `pip3 install dfss`；无外网则从本地 scp 数据，或跳过该样例并报原因 |
| C++ 编译 `cmake` 找不到 SAIL | sail_list 样例没传 `-a` | 对这 9 个样例加 `-a /opt/sophon/sophon-sail` |
| SoC 上 C++ 编译失败 | 需交叉编译，设备上缺 SDK 头文件 | 改只跑 Python 用例，C++ 记 SKIP-需交叉编译 |
| BM1688 上没跑双核变体 | `auto_test.sh` 在 `TARGET=BM1688` 时会自动加 `int8_4b_2core` 用例 | 无需干预；2core 用例由 `TARGET=BM1688` 触发，与设备核数无关。6 核设备的 PLATFORM 映射仅部分脚本有分支 |
| bmodel 加载 "magic number" | 下载未完成/被截断 | 重跑 download；校验字节数 |
| 磁盘满 | 多样例累积 datasets/models 数十 GB | 每样例测完即清理其 datasets/models/results |

## 8. 执行纪律

- **凭据未齐前不动手**（见 §1 Red Flag）。
- 只跑有 `auto_test.sh` 的样例；其余 SKIP 并列名。
- 不改源码/脚本/模型去凑通过；不改 git。
- 测完清理每样例的大文件，保留日志。
- 密码不进任何日志/报告。
- 结论先行、中文、流式报进度。
