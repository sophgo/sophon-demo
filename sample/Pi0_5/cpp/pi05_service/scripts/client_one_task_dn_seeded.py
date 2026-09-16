#!/usr/bin/env python3
"""client_one_task_dn_seeded.py — LIBERO-Spatial 单 task 闭环客户端.

它连设备上的常驻推理服务，在仿真里真正跑一遍任务：渲染观测 -> 请求动作 chunk ->
按 replan 步数执行 -> 看仿真器有没有判定成功。

用法:
    python client_one_task_dn_seeded.py <task> <init> <host> <port> <seed> \
                                       <replan> <max_steps> <num_ep> <dn>

其中 <dn> 是去噪步数。渲染后端由外层 eglrun.sh 用 MUJOCO_GL 注入（osmesa 或 egl）。
一次推理用哪份噪声由 (case, 第几次推理) 决定，所以整条闭环可复现、可逐位比对。
"""
import datetime
if not hasattr(datetime, "UTC"): datetime.UTC = datetime.timezone.utc
import sys, os, pathlib, logging, collections, time
logging.basicConfig(level=logging.INFO, force=True)
import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openpi_client import image_tools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tcp_client import TcpClient

TASK = int(sys.argv[1]); INIT = int(sys.argv[2])
HOST = sys.argv[3] if len(sys.argv) > 3 else "172.26.166.88"
PORT = int(sys.argv[4]) if len(sys.argv) > 4 else 9200
SEED = int(sys.argv[5]) if len(sys.argv) > 5 else 7
REPLAN = int(sys.argv[6]) if len(sys.argv) > 6 else 5
MAX_STEPS = int(sys.argv[7]) if len(sys.argv) > 7 else 220
N_EP = int(sys.argv[8]) if len(sys.argv) > 8 else 1
DN = int(sys.argv[9]) if len(sys.argv) > 9 else 10
DUMMY = [0.0] * 6 + [-1.0]
NUM_WAIT = 10

bd = benchmark.get_benchmark_dict(); suite = bd["libero_spatial"]()
task = suite.get_task(TASK); inits = suite.get_task_init_states(TASK)
bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
client = TcpClient(HOST, port=PORT, quality=int(os.environ.get("JQ", "85")))
np.random.seed(SEED)

# 2026-09-14: 确定性噪声表 —— 原实现每次推理都用无种子的 np.random.RandomState(),
# 导致整条闭环不可复现, A/B 对照只能靠统计, 无法逐位比对.
# 现在噪声只由 (case, 第几次推理) 决定: 同一份代码在新旧设备二进制上跑,
# 每一步的输入完全一致, 结果可直接逐位比较. 见 run_h1_dn_seeded.sh.
NOISE_TABLE = {}
def _noise_for(case_idx, infer_idx):
    k = (case_idx, infer_idx)
    if k not in NOISE_TABLE:
        NOISE_TABLE[k] = np.random.RandomState(20260914 + case_idx * 1000 + infer_idx
                                              ).standard_normal((10, 32)).astype(np.float32)
    return NOISE_TABLE[k]
ok = 0
for ep in range(N_EP):
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
    env.seed(SEED); env.reset()
    obs = env.set_init_state(np.asarray(inits[INIT + ep]))
    plan = collections.deque(); t = 0; done = False; inf_ms = []
    while t < MAX_STEPS + NUM_WAIT and not done:
        if t < NUM_WAIT:
            obs, r, d, info = env.step(DUMMY); t += 1; continue
        img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
        wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
        img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, 224, 224))
        wrist = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist, 224, 224))
        if not plan:
            t0 = time.time()
            ac = client.infer(img, wrist, tidx=TASK, steps=DN, seed=SEED,
                              noise=_noise_for(INIT + ep, len(inf_ms)))[0][:, :7]
            inf_ms.append((time.time() - t0) * 1000)
            plan.extend(ac[:REPLAN])
        act = plan.popleft()
        obs, r, done, info = env.step(act.tolist()); t += 1
    ok += 1 if done else 0
    med = float(np.median(inf_ms)) if inf_ms else -1.0
    print(f"RESULT task{TASK} init{INIT+ep} dn={DN} done={done} exec_steps={t-NUM_WAIT} "
          f"n_infer={len(inf_ms)} inf_med={med:.0f}ms ok={ok}", flush=True)
    env.close()
print(f"SUMMARY task{TASK} dn={DN} inits={N_EP} ok={ok}/{N_EP}", flush=True)
print("CLIENT_DN_DONE", flush=True)
