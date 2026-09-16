#!/bin/bash
#===----------------------------------------------------------------------===#
#
# Copyright (C) 2026 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
# run_h5_20_gpu.sh — 一键：GPU(EGL) 硬渲染 + H5 服务全量 20 case 闭环.
#
# 与 osmesa 链路相比只换了渲染后端，模型 / 客户端 / 噪声表完全不变：
#   env.step  203 ms -> 47 ms（256x256，同容器同参数实测）
#   replan=5 周期  5x203+457 -> 5x47+457，即 1472 -> 692 ms
#
# 用法: PI05_REPO=<工作树> bash run_h5_20_gpu.sh [dn] [tag]
#
# 前置：
#   1. 设备上常驻推理服务在跑（README §3），HOST/PORT 默认 172.26.166.88:9201；
#   2. 宿主机装好 NVIDIA 驱动（/dev/nvidia*），nvlib/ 内有与驱动版本一致的 EGL 库；
#   3. PI05_REPO 指向含 openpi-ref/ 与 nvlib/ 的工作树。
set -eu
DN=${1:-2}; TAG=${2:-h5gpu}
HERE=$(cd "$(dirname "$0")" && pwd)
PI05_REPO=${PI05_REPO:-/mnt/sda1/zzt/pi05-s1}
export PI05_REPO

GL=egl bash "$HERE/eglrun.sh" "timeout 1400 bash /svc/scripts/run_h5_20_egl.sh $DN $TAG"
echo
LOG="$PI05_REPO/logs/h5_${TAG}.log"
echo "日志: $LOG"
grep -c "ok=1$" "$LOG" 2>/dev/null | xargs -I{} echo "通过: {}/20"
