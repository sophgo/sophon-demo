---
name: delivery-checklist
description: 交付检查清单。在提交/推送 sample 的新增代码（C++ 例程、Python 推理脚本、bmodel 编译脚本）前，按清单逐项验证：编译通过、SoC 实跑通过、精度对齐、README 完整、ff_decode 四修、无构建产物、脚本可用。说「按照 delivery-checklist skill 检查 XXX 的交付」即可。
---

# 交付检查清单

> 目的：在提交/推送 sample 的新增或修改代码前，逐项验证交付质量，避免"能编译但跑不了""README 说的和实际不一致""构建产物混入提交"等常见问题。**提交前过一遍这个清单，每项 ✅ 才推。**

## 适用范围

- 新增 C++ 例程（`sample/<NAME>/cpp/`）
- 新增 Python 推理脚本
- 修改 bmodel 编译脚本（`scripts/gen_*.sh`）
- sample 整体交付前自检

## 检查清单

### A. 代码完整性

| # | 检查项 | 验证方法 | 通过标准 |
|---|--------|---------|---------|
| A1 | 源文件齐全 | `find sample/<NAME>/cpp/ -type f` | 有 CMakeLists.txt、main.cpp、<name>.cpp、<name>.hpp、README.md |
| A2 | dependencies 齐全 | `ls sample/<NAME>/cpp/dependencies/include/` | 有 bmnn_utils.h、bm_wrapper.hpp、ff_decode.hpp、json.hpp、utils.hpp |
| A3 | ff_decode.cpp 四修 | 见下方专项检查 | 全部 4 项修复在位 |
| A4 | 无构建产物混入 | `git status sample/<NAME>/cpp/` | build/ 目录、*.o、*.so、*.soc、*.pcie 不在 untracked 列表 |
| A5 | 无临时结果文件 | `git status` | results/、*.json、*.log 不在提交范围 |

### B. ff_decode.cpp 四修专项检查

每个新 C++ sample 自带一份 ff_decode.cpp，必须包含以下 4 项修复。用 grep 逐项验证：

```bash
FF=sample/<NAME>/cpp/dependencies/src/ff_decode.cpp

# 修1: SOF2 marker 0xFFC2 → return false in determine_hardware_decode
grep -n "0xFFC2\|SOF2" $FF  # 应有命中

# 修2: data_on_device_mem=false for SW-decoded frames in picDec
grep -n "data_on_device_mem.*false" $FF  # 应有命中

# 修3: RGBA/BGRA→RGB24 conversion in miscDec
grep -n "RGB24\|RGBA.*RGB\|BGRA.*RGB" $FF  # 应有命中

# 修4: num_extra_framebuffers = hardware_decode ? 0 : 2
grep -n "num_extra_framebuffers" $FF  # 应有条件表达式
```

### C. 编译验证

| # | 检查项 | 验证方法 | 通过标准 |
|---|--------|---------|---------|
| C1 | SoC 交叉编译/原生编译通过 | `cmake -DTARGET_ARCH=soc -DSDK=... .. && make` | 无 error，生成 *.soc |
| C2 | PCIe 编译通过（如有 PCIe 环境） | `cmake .. && make` | 无 error，生成 *.pcie |
| C3 | CMakeLists.txt 格式正确 | 对比同类 sample（如 YOLOv5）的 CMakeLists.txt | 结构一致，binary name 正确 |

### D. 运行验证

| # | 检查项 | 验证方法 | 通过标准 |
|---|--------|---------|---------|
| D1 | SoC 实跑通过 | `./<name>.soc --input=datasets/test --bmodel=... --dev_id=0` | 无 crash/abort，有检测输出 |
| D2 | 输出结果合理 | 检查 JSON 输出和可视化图 | 检测框位置合理，类别正确，score > 0 |
| D3 | 多图测试 | 至少跑 3-4 张不同图片 | 每张图都有合理输出 |
| D4 | 视频测试（如支持） | `./<name>.soc --input=test.mp4 --bmodel=...` | 帧率正常，输出图序列 |

### E. 精度对齐

| # | 检查项 | 验证方法 | 通过标准 |
|---|--------|---------|---------|
| E1 | 与 Python 参考对齐 | 参考 precision-alignment skill | ✅ |
| E2 | 框数一致 | 逐图对比 C++ 和 Python JSON | 每张图检测框数相同 |
| E3 | category_id 一致 | 逐框对比 | 全部匹配 |
| E4 | score 差异 < 1e-4 | 逐框对比 | 在浮点误差范围内 |
| E5 | bbox 差异 ≤ 2px | 逐框对比 | 整数截断导致的 1px 可接受 |

### F. README 完整性

| # | 检查项 | 通过标准 |
|---|--------|---------|
| F1 | 目录结构 | 有环境准备、程序编译、推理测试三节 |
| F2 | 编译指令 | PCIe 和 SoC 两种方式都写了 |
| F3 | 参数说明 | 列出 --bmodel/--input/--dev_id/--conf_thresh 等参数 |
| F4 | 测试图片指令 | 有完整的图片测试命令示例 |
| F5 | 测试视频指令 | 有视频测试命令示例 |
| F6 | 链接有效 | README 中的相对路径链接（如 `../../../docs/...`）指向存在的文件 |

### G. 提交规范

| # | 检查项 | 通过标准 |
|---|--------|---------|
| G1 | commit message 格式 | `feat(<SAMPLE>): add C++ sample (<name>_bmcv)` 或类似规范格式 |
| G2 | 每个 feature 独立 commit | 不混入其他 sample 的改动 |
| G3 | 不含调试代码 | 无 `#if DEBUG` 下的大段调试输出残留、无临时 cout |
| G4 | attribution | commit message 末尾有 `Co-Authored-By: Claude Code <noreply@anthropic.com>` |
| G5 | 推送到正确分支 | `git push ssh://... HEAD:refs/for/developer` |

## 执行流程

1. **写完代码后立即自检**：按 A→B→C→D→E→F 逐项过一遍。
2. **有 ❌ 的项必须修完再提交**。特别是 A4（构建产物混入）和 B（ff_decode 四修）是高频遗漏。
3. **全部 ✅ 后执行提交**：
   ```bash
   git add sample/<NAME>/cpp/
   git commit -m "feat(<NAME>): add C++ sample (<name>_bmcv)

   <简要描述>

   Co-Authored-By: Claude Code <noreply@anthropic.com>"
   git push ssh://liheng.fang@gerrit-ai.sophgo.vip:29418/sophon-demo HEAD:refs/for/developer
   ```
4. **推完后清理设备空间**：删除 SoC 上的临时编译目录、测试 bmodel、中间产物。

## 报告格式

```
## <SAMPLE> 交付检查报告

结论：✅ 全部通过，可提交 / ❌ 有 N 项未通过

| 类别 | 项数 | 通过 | 未通过 |
|------|------|------|--------|
| A 代码完整性 | 5 | 5 | 0 |
| B ff_decode 四修 | 4 | 4 | 0 |
| C 编译验证 | 3 | 3 | 0 |
| D 运行验证 | 4 | 4 | 0 |
| E 精度对齐 | 5 | 5 | 0 |
| F README 完整性 | 6 | 6 | 0 |
| G 提交规范 | 5 | 5 | 0 |

未通过项详情：
- （如有）...

Gerrit: https://gerrit-ai.sophgo.vip:8443/XXXXXX
```

## 执行纪律

- 这份清单是提交前的硬性门禁——任何 ❌ 都不能跳过直接提交。
- 最易遗漏的 3 项：A4（build/ 混入）、B3/B4（ff_decode 修 3/修 4）、E2（框数不一致没深查原因）。
- 清单不是形式主义：每一项都对应过往踩过的坑。
- 汇报用中文，结论先行。
