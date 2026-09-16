#!/bin/bash
#===----------------------------------------------------------------------===#
#
# Copyright (C) 2026 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
# eglrun.sh — 在带 NVIDIA 设备 + EGL 库的一次性容器里跑命令.
#
# 宿主机上没装 nvidia-container-toolkit（--gpus 不可用），所以设备节点与 EGL 库
# 都是手工传进容器的；原理与三个易错点见 ../README.md §2.2。
#
# 用法:
#     PI05_REPO=<工作树> bash eglrun.sh '<容器内命令>'
#
# 环境变量:
#     PI05_REPO  评测工作树，需含 openpi-ref/（挂到 /app）与 nvlib/（挂到 /nvlib）。
#                默认 /mnt/sda1/zzt/pi05-s1
#     GL         egl（默认，GPU 硬渲染）| osmesa（CPU 软渲染）
set -u
GL=${GL:-egl}
PI05_REPO=${PI05_REPO:-/mnt/sda1/zzt/pi05-s1}
HERE=$(cd "$(dirname "$0")" && pwd)
CMD=${1:?用法: PI05_REPO=<工作树> bash eglrun.sh '<cmd>'}

for d in "$PI05_REPO/openpi-ref" "$PI05_REPO/nvlib" "$HERE"; do
  [ -d "$d" ] || { echo "缺少目录: $d"; echo "用 PI05_REPO=<工作树> 指定 openpi-ref/ 与 nvlib/ 所在位置"; exit 1; }
done

DEV="--device /dev/nvidia0 --device /dev/nvidiactl --device /dev/nvidia-modeset
     --device /dev/nvidia-uvm --device /dev/nvidia-uvm-tools
     --device /dev/nvidia-caps/nvidia-cap1 --device /dev/nvidia-caps/nvidia-cap2
     --device /dev/dri/card1 --device /dev/dri/renderD128"

if [ "$GL" = "egl" ]; then
  GLENV="-e LD_LIBRARY_PATH=/nvlib -e MUJOCO_GL=egl -e PYOPENGL_PLATFORM=egl
         -e __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json"
else
  GLENV="-e MUJOCO_GL=osmesa -e PYOPENGL_PLATFORM=osmesa"
fi

# shellcheck disable=SC2086
docker run --rm --name pi05glrun \
  $DEV \
  -v "$PI05_REPO/nvlib":/nvlib:ro \
  -v "$PI05_REPO/openpi-ref":/app \
  -v "$PI05_REPO":/work \
  -v "$HERE":/svc/scripts:ro \
  $GLENV \
  -e PYTHONPATH=/app/third_party/libero:/app/packages/openpi-client/src:/svc/scripts \
  -e TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
  libero:local bash -lc "source /.venv/bin/activate; cd /work; $CMD" \
  2>&1 | grep -vE "^=|^CUDA Version|Copyright|governed by|By pulling|developer.nvidia|A copy of this|^$|WARNING: The NVIDIA Driver|Use the NVIDIA Container Toolkit|docs.nvidia.com"
