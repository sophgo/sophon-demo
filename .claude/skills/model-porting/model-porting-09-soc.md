---
name: model-porting-09-soc
description: SoC 交叉编译和部署。适配多芯片 g++ toolchain，生成编译和自动化测试脚本。
---

# 步骤 09：SoC 部署

## 前置

步骤 08 已完成（C++ 代码），模板中设备信息（ip、soc-sdk 路径）已填写。

## 提示词

```
适配 SoC 交叉编译和部署：

1. 根据目标芯片选择 g++ 版本（见内联知识）
2. 生成 scripts/build.sh：
   - PCIe 编译：mkdir build && cd build && cmake .. -DTARGET_ARCH=pcie && make -j
   - SoC 交叉编译：cmake .. -DTARGET_ARCH=soc -DSDK=[soc-sdk路径] && make -j
   - 产出物后缀 .pcie 和 .soc
3. 生成 scripts/auto_test.sh：
   - 遍历所有 BModel × batch × chip 组合
   - 自动执行推理并汇总结果
4. 编译产物部署到目标设备（scp），运行验证
```

## 预期输出

- `scripts/build.sh`
- `scripts/auto_test.sh`
- SoC 设备上验证通过的日志

## 内联知识：交叉编译 g++ 版本

| 芯片 | 设备型号 | 编译器 | g++ 版本 |
|------|---------|--------|---------|
| BM1684X | SE7-32 | aarch64-linux-gnu-g++ | **g++ 9.x** |
| BM1688 | SE9-16 | aarch64-linux-gnu-g++ | **g++ 11.x** |
| CV186X | SE9-8 | aarch64-linux-gnu-g++ | **g++ 11.x** |

> SoC SDK 路径从模板第 9 节「目标设备」字段获取。

### auto_test.sh 结构

```bash
#!/bin/bash
# 遍历所有模型文件
for bmodel in models/BM1684X/*.bmodel; do
    for batch in 1 4; do
        echo "Testing $bmodel batch=$batch"
        ./cpp/[name]_bmcv/build/[name]_bmcv.pcie \
            --bmodel $bmodel --input $dataset --loops 100
    done
done
```

## Debug

| 问题 | 排查方向 |
|------|---------|
| 交叉编译链接错误 (undefined reference) | 检查 g++ 版本：BM1684X→9.x, BM1688/CV186X→11.x |
| SoC 运行时库找不到 | 确认 LD_LIBRARY_PATH 包含 libsophon 路径 |
| SoC 内存不足 | 减小 batch size 或用 zero-copy mmap 模式 |
