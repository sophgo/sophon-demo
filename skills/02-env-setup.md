# Skill 2: 环境准备

## 目标
搭建 SOPHON TPU 开发和运行环境。

## x86 PCIe 平台

### 安装 libsophon
```bash
# 从 SDK 安装或从算能官网下载
# 检查安装
bm-smi
```

### 安装 TPU-MLIR (如需编译模型)
```bash
# 参考 docs/Environment_Install_Guide.md
# 需要 Docker 或直接安装
```

### 安装 Python 依赖
```bash
pip3 install sophon-sail numpy torch 框架预处理工具 soundfile scipy
```

### 安装 C++ 依赖 (如需 C++ 移植)
```bash
# 数值计算库 (线性代数)
sudo apt-get install numerical-lib-dev
# 数据I/O库 (输入数据读取)
sudo apt-get install data-io-dev
```

## SoC 平台 (SE7-32)

### 检查预装软件
```bash
# libsophon 预装在 /opt/sophon/
ls /opt/sophon/libsophon-current/
# 检查版本
bm-smi  # (可能需要 source 环境)
```

### 安装额外依赖
```bash
sudo apt-get install data-io-dev numerical-lib-dev
```

### 交叉编译环境 (在 PC 上)
```bash
# 安装 ARM 交叉编译工具链
sudo apt-get install gcc-aarch64-linux-gnu g++-aarch64-linux-gnu
# 准备 SDK (需设置 SDK 环境变量)
export SDK=/path/to/sophon-sdk
```

## 检查清单

- [ ] bm-smi 能显示 TPU 设备
- [ ] Python 能 import sophon.sail
- [ ] Python 能 import torch, 框架预处理工具
- [ ] bmrt_test 可用 (在 SDK 中)
- [ ] C++ 编译器可用 (gcc/g++)
- [ ] 数据I/O库 和 numerical_lib 已安装
- [ ] SoC 设备可通过 SSH 访问

## 常见问题

1. **sophon-sail 安装失败**: 确认 Python 版本与 SDK 兼容
2. **bm-smi 找不到**: 需要 source SDK 环境变量或添加到 PATH
3. **SoC 设备 apt 依赖冲突**: 尝试 `sudo apt --fix-broken install` 后重试
