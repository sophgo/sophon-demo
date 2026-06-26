---
name: model-porting-02-env
description: 搭建 TPU-MLIR Docker 编译环境。基于 model_info.json 中的框架信息安装依赖。
---

# 步骤 02：搭建环境

## 前置

步骤 01 已完成，`model_info.json` 已生成。

## 提示词

```
基于 model_info.json 中的框架和依赖信息，搭建编译环境：

1. 拉取 TPU-MLIR Docker 镜像（最新 release 版本）
2. 启动 Docker 容器，挂载 sophon-demo 目录到 /workspace
3. 在容器内安装模型特定的 Python 依赖（torch、timm、onnx、onnx-simplifier 等）
4. 验证：model_transform.py --help 和 model_deploy.py --help 可正常输出

不需要写代码，输出环境安装命令。
```

## 预期输出

```bash
# 环境安装命令
docker pull sophgo/tpuc_dev:latest
docker run --rm -v $(pwd):/workspace -it sophgo/tpuc_dev:latest
cd /workspace
pip install torch torchvision timm onnx onnx-simplifier onnxruntime
```

## 内联知识

| 工具 | 用途 |
|------|------|
| `model_transform.py` | ONNX → MLIR |
| `model_deploy.py` | MLIR → BModel |
| `run_calibration.py` | INT8 校准表生成 |
| `bmrt_test` | BModel 加载测试和理论性能 |

Docker 镜像必须与目标 libsophon 版本匹配，否则 BModel 可能加载失败。

### TPU-MLIR 容器准备流程

完整步骤参考 [Environment_Install_Guide.md#1-tpu-mlir环境搭建](../../../docs/Environment_Install_Guide.md#1-tpu-mlir环境搭建)。要点：

1. **先询问用户**是否已有可用的 TPU-MLIR 容器（用户指定容器名则直接 `docker exec` 进入）。
2. 若用户未提供，检查本机是否有 `sophgo/tpuc_dev` 镜像：`docker images | grep tpuc_dev`；没有则 `docker pull sophgo/tpuc_dev:latest`。
3. 启动容器（挂载仓库到 /workspace）：
   ```bash
   docker run --privileged --name mymlir --network host -v $PWD:/workspace -it sophgo/tpuc_dev:latest
   ```
4. 进入容器后安装 pip 版 TPU-MLIR：
   ```bash
   pip install tpu_mlir -i https://pypi.tuna.tsinghua.edu.cn/simple
   # 或按需: pip install tpu_mlir[onnx]
   ```
5. 验证：`model_transform.py --help` 能输出版本号即就绪。后续编译用 `docker exec mymlir bash -lc 'cd /workspace/sample/XXX && bash scripts/gen_xxx.sh bm1684x'`。

> ⚠️ `source envsetup.sh`（若用源码版）不能接管道 `| tail`，否则在子 shell 中导出不生效。`model_transform.py`/`model_deploy.py` 在 pip 安装或 source envsetup 后才进 PATH。
