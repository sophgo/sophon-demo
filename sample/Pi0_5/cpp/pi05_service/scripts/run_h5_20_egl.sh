#!/bin/bash
#===----------------------------------------------------------------------===#
#
# Copyright (C) 2026 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
# run_h5_20_egl.sh — 容器内脚本：全量 10 task x 2 init = 20 case 闭环.
#
# 由 run_h5_20_gpu.sh 通过 eglrun.sh 拉起；渲染后端（osmesa / egl）由外层注入。
#
# 用法: run_h5_20_egl.sh [dn] [tag] [replan]
set -u
DN=${1:-2}; TAG=${2:-h5egl}; REPLAN=${3:-5}
HOST=${HOST:-172.26.166.88}
PORT=${PORT:-9201}
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
cd /work || exit 1
mkdir -p logs
LOG=/work/logs/h5_${TAG}.log; : > "$LOG"
echo "H5EGL_START dn=$DN tag=$TAG replan=$REPLAN gl=${MUJOCO_GL:-?} $(date +%F' '%H:%M:%S)" | tee -a "$LOG"
for t in 0 1 2 3 4 5 6 7 8 9; do
  for i in 0 1; do
    timeout 900 python /svc/scripts/client_one_task_dn_seeded.py \
      "$t" "$i" "$HOST" "$PORT" 7 "$REPLAN" 220 1 "$DN" 2>&1 \
      | grep -E "RESULT|SUMMARY" | tee -a "$LOG"
  done
done
echo "H5EGL_DONE $(date +%F' '%H:%M:%S) cases=$(grep -c RESULT "$LOG") ok=$(grep -c 'ok=1$' "$LOG")" | tee -a "$LOG"
