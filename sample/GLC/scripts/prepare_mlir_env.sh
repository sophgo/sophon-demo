#!/bin/bash
# 准备 tpu-mlir 环境。
#
# 在较新 glibc 的主机上，tpu-mlir 1.28.1 官方 wheel 有三个坑，本脚本处理掉：
#
# 1. wheel 自带 lib/third_party/libc.so.6 与 libm.so.6，而 tpu_mlir/__init__.py 会把该
#    目录插到 LD_LIBRARY_PATH 最前面，导致 Python 解释器本身被那个旧 libc 加载，
#    直接报 GLIBC_2.36 / GLIBC_ABI_DT_RELR not found。解法：把这两个文件移进
#    _bundled_glibc/ 子目录（不在搜索路径里），让 tpu-mlir 用系统的 glibc。
# 2. wheel 没声明 psutil / flatbuffers / onnxruntime / onnxsim 依赖，
#    但它们被 model_transform.py 的 import 链强制加载。
# 3. 版本匹配：tpu-mlir 1.28.1 依赖 numpy 1.24.x 与 onnx 1.16.x，
#    所以必须用独立 venv，不能装进主环境。
#
# 用法：
#   scripts/prepare_mlir_env.sh [venv 路径] [tpu_mlir wheel 路径]
#
# 之后导出 ONNX 与编译 bmodel 的脚本都用 <venv>/bin/python 与 <venv>/bin/model_*.py。

set -euo pipefail

venv="${1:-.venv-mlir}"
wheel="${2:-}"

if [ -z "$wheel" ] || [ ! -f "$wheel" ]; then
    echo "用法: $0 <venv 路径> <tpu_mlir wheel 路径>" >&2
    echo "wheel 可从 PyPI 取: pip download tpu-mlir==1.28.1 --no-deps -d ." >&2
    exit 1
fi

if [ ! -d "$venv" ]; then
    python3.10 -m venv "$venv"
fi

"$venv/bin/python" -m pip install --upgrade pip
# torch 与 onnx 用于导出 ONNX；tpu-mlir 自身会拉 numpy/onnx 的匹配版本
"$venv/bin/python" -m pip install "$wheel" torch==2.5.1+cpu \
    --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple
"$venv/bin/python" -m pip install "onnx==1.16.2" onnxruntime onnxsim psutil flatbuffers

third_party=$("$venv/bin/python" -c \
    "import tpu_mlir, os; print(os.path.join(os.path.dirname(tpu_mlir.__file__), 'lib', 'third_party'))")
if [ -f "$third_party/libc.so.6" ]; then
    mkdir -p "$third_party/_bundled_glibc"
    mv "$third_party/libc.so.6" "$third_party/libm.so.6" \
       "$third_party/libpthread.so.0" "$third_party/libgcc_s.so.1" \
       "$third_party/_bundled_glibc/" 2>/dev/null || true
    echo "已移开 wheel 自带的 glibc 组件，避免与主机 glibc 冲突"
else
    echo "wheel 未带 libc.so.6，无需处理"
fi

echo "tpu-mlir 环境就绪：$venv"
echo "  自检：$venv/bin/python -c 'import tpu_mlir, onnx, torch; print(\"ok\")'"
