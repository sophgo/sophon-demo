# π0.5 进阶用法：常驻推理服务 + LIBERO 闭环评测环境

> 本目录**不是** sample 的主例程。主例程是 [`../pi05_bmcv/`](../pi05_bmcv/README.md) —— 单次推理、无外部依赖。
> 这里放的是**把 π0.5 真正接到机器人上跑闭环**所需要的部署说明，供需要复现"任务成功率"指标的读者使用。

## 目录

- [1. 两个例程的区别](#1-两个例程的区别)
- [2. 宿主机部署：LIBERO 评测环境](#2-宿主机部署libero-评测环境)
- [3. 设备侧：常驻推理服务](#3-设备侧常驻推理服务)
- [4. 跑闭环评测](#4-跑闭环评测)
- [5. 该报哪些指标](#5-该报哪些指标)
- [6. 已知坑](#6-已知坑)

## 1. 两个例程的区别

| | `../pi05_bmcv/`（主例程） | 本目录（进阶） |
|---|---|---|
| 做什么 | 读一条观测 → 出动作 chunk | 常驻服务 + 仿真环境闭环控制 |
| 依赖 | 只有 bmodel 与观测数据 | **需要 robosuite / MuJoCo / EGL 的完整仿真环境** |
| 精度指标 | 与官方参考的 cosine similarity | **任务成功率（含 95% Wilson CI）** |
| 适合谁 | 判断"移植得对不对" | 判断"在真实控制回路里能不能用" |

**先跑主例程。** 数值不对的话，闭环结果没有解释力。

## 2. 宿主机部署：LIBERO 评测环境

### 2.1 镜像

```
libero:local     # robosuite 1.4.0 + mujoco 3.2.3 + python 3.9.20 + EGL 离屏渲染
```

镜像内需包含：`openpi`（官方 ref 分支）、`libero` 第三方包、评测客户端脚本。
`python` 环境位于 `/.venv`，使用时需 `source /.venv/bin/activate`。

### 2.2 渲染后端：osmesa 还是 GPU(EGL)

这是**决定评测吞吐的关键开关**。同容器、同镜像、同参数，只改 `MUJOCO_GL`：

| 渲染后端 | `env.step` 中位 | 20 case 全量墙钟 | TPU 占空比（稳态循环） |
|---|---|---|---|
| `osmesa`（CPU 软渲染，llvmpipe） | 203 ms | 804 s | 31% |
| **`egl`（GPU 硬渲染，T400）** | **47 ms（4.3×）** | **402 s（2.0×）** | **66%** |

**为什么默认的 EGL 跑不起来**：容器里只有 `/dev/dri/*`（DRM 节点），没有 `/dev/nvidia*`；
宿主机也没装 `nvidia-container-toolkit`，`--gpus` 不可用。
解法是**手工把设备节点与 EGL 库传进容器**：

```bash
docker run --rm \
  --device /dev/nvidia0 --device /dev/nvidiactl --device /dev/nvidia-modeset \
  --device /dev/nvidia-uvm --device /dev/nvidia-uvm-tools \
  --device /dev/nvidia-caps/nvidia-cap1 --device /dev/nvidia-caps/nvidia-cap2 \
  --device /dev/dri/card1 --device /dev/dri/renderD128 \
  -v <repo>/nvlib:/nvlib:ro \
  -e LD_LIBRARY_PATH=/nvlib \
  -e MUJOCO_GL=egl -e PYOPENGL_PLATFORM=egl \
  -e __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json \
  -e PYTHONPATH=<repo>/openpi-ref/third_party/libero:<repo>/openpi-ref/packages/openpi-client/src \
  -e TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
  libero:local bash -lc "source /.venv/bin/activate; <cmd>"
```

完整可运行版本见 [`scripts/eglrun.sh`](./scripts/eglrun.sh)。

**三个容易看错的地方**：

1. **`/dev/dri/card1` 就是那块 T400**（`DRIVER=nvidia`，PCI `10DE:1FB2`），**不是核显** —— 这点最容易看错。
2. `nvlib/` 里的 EGL 库**版本必须与宿主机驱动一致**（本 sample 环境为 610.57.04）。
3. 不设 `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` 时，torch ≥ 2.6 会拒绝加载 LIBERO 的 `init_states.pt`。

### 2.3 代价：EGL 不是无损提速

EGL 与 osmesa 的渲染**不是逐像素相同**：约 3% 的像素差 >4/255（`max|diff|=135`、`mean|diff|=1.34`）。
图像进策略，因此**轨迹会移动** —— 20 个 case 里只有 4 个的步数与 osmesa 完全一致，其余差 1~2 步。
**两种后端都是 20/20，成功率没有退化。** 但若要与历史结果严格逐位对齐，仍需用 osmesa 复跑。

## 3. 设备侧：常驻推理服务

设备（SE7, BM1684X）上跑两个进程：

```
评测容器 (libero:local)
   │  TCP :9201（自定义二进制协议）
   ▼
宿主 tcp_serve_dkv536          ← 进程管理 + 协议解析 + host 侧 10 步 Euler 积分
   │  stdio 管道
   ▼
设备子进程 serve_dkv536        ← 加载 6 个 bmodel，跑实际前向（自动 respawn）
```

源码见 [`src/tcp_serve_dkv.c`](./src/tcp_serve_dkv.c) 与 [`src/serve_dkv.c`](./src/serve_dkv.c)。

**bmodel 与子进程二进制都由环境变量配置，不重编即可换档**：

```ini
# /etc/systemd/system/pi05-h5.service
[Unit]
Description=pi0.5 SE7 H5-tier infer service (tcp_serve_dkv536 :9201)
After=network.target

[Service]
Type=simple
User=linaro
WorkingDirectory=/data2/pi05s/d536
Environment=BMRUNTIME_NEURON_HEAP_MASK=7
Environment=PI05_SERVE_BIN=/data2/pi05s/d536/serve_dkv536
Environment=PI05_DKV0=/data2/pi05s/d536/dkv0_9_bm1684x_W8BF16_H5.bmodel
Environment=PI05_DKV1=/data2/pi05s/d536/dkv9_18_bm1684x_W8BF16_H5.bmodel
Environment=PI05_DDN0=/data2/pi05s/d536/ddn_0_6_bm1684x_BF16_H5.bmodel
Environment=PI05_DDN1=/data2/pi05s/d536/ddn_6_12_bm1684x_BF16_H5.bmodel
Environment=PI05_DDN2=/data2/pi05s/d536/ddn_12_18_final_bm1684x_BF16_H5.bmodel
ExecStart=/data2/pi05s/d536/tcp_serve_dkv536 0 /data2/pi05s/demo_work 9201 /data2/pi05s/demo_work/siglip_visual_bm1684x_w4bf16.bmodel
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
```

> ⚠️ **`PI05_SERVE_BIN` 必须显式给**。`tcp_serve_dkv536` 只是宿主侧的协议进程，
> 真正跑前向的是它 fork 出来的子进程；子进程路径没有默认值可依赖 ——
> 源码里的兜底是一个历史遗留的绝对路径（`/data2/pi05s/demo_work/opt2/serve_dkv`），
> 指到旧二进制上就会父子进程双双卡在 `pipe_wait`，客户端 300 s 超时，
> 而且日志里看不出子进程到底起了谁。**换档时 bmodel 和这个子进程二进制要一起换。**

> ⚠️ **强杀持有 NPU 的进程会把 TPU 驱动卡死，只能重启设备恢复。**
> 服务切换一律走 `systemctl`（它发 SIGTERM，进程会优雅释放 NPU 句柄）。

> ⚠️ **NPU 堆内存必须预留足够**：SE7 默认 ION 预留可能只有约 1 GB Linux 可用内存。
> 需要把 TPU/VPU/VPP 三个 heap 调大（合计约 13 GB），否则 bmodel 加载失败。

## 4. 跑闭环评测

```bash
# 宿主机侧（EGL 渲染 + H5 档）。PI05_REPO 指向含 openpi-ref/ 与 nvlib/ 的工作树
PI05_REPO=/path/to/pi05-work bash scripts/run_h5_20_gpu.sh
```

该脚本在容器内跑 LIBERO-Spatial 10 task × 2 init state = 20 case，
`seed7 / replan5 / max220 / dn2`，输出每 case 的 `exec_steps` / `n_infer` / 成功与否。
日志落在 `$PI05_REPO/logs/h5_<tag>.log`。

需要 `nvlib/`（与宿主驱动版本一致的 EGL 库）与 `openpi-ref/`；两者都不在本 sample 里，
用 `PI05_REPO` 指过去。没有 NVIDIA 驱动时把 `GL=osmesa` 传给 `eglrun.sh` 退回软渲染
（慢 4.3×，结果会略有不同，见 §2.3）。

## 5. 该报哪些指标

**任务成功率**，并**必须写清 episode 数**：

| 测试平台 | 精度档 | 配置 | 任务成功率 | 95% Wilson CI |
|---|---|---|---|---|
| SE7-32 | W8BF16 + BF16 | LIBERO-Spatial 10 task × 2 init = 20 episode，replan=5 / dn=2 / seed7 | **20/20 = 100%** | 83.9% – 100% |

实测：单次推理中位 445 ms，20 case 墙钟 472 s（EGL）。逐 case 记录见
`$PI05_REPO/logs/h5_h5gpu.log`。

> **20 episode 是抽样，不是结论**：Wilson 下界只有 83.9%。
> 同类工作里，vla.cpp 用 "ten tasks twenty episodes per architecture" 并给出 95% Wilson 区间；
> FlashRT 用 "10 × 50 = 500 episodes"；Isaac-GR00T 明确声明 run-to-run 存在 5-6% variance。
> 要下"成功率达标"的结论，需要把 episode 数加上去。

## 6. 已知坑

| 现象 | 根因 | 处理 |
|---|---|---|
| 父子进程双双卡在 `pipe_wait`，客户端 300 s 超时 | 没设 `PI05_SERVE_BIN`，子进程落到源码里历史遗留的兜底路径，起了旧二进制 | 显式设 `PI05_SERVE_BIN`（见 §3）；换档时 bmodel 与该二进制要一起换 |
| 客户端卡到 300 s 超时 | 服务端把 `ok` 与 `traj` 分两次 `write`，客户端 `_recv_exact` 用局部 buf 收、丢掉余量 | 服务端合并成一次 write；客户端改**实例级持久缓冲**。**两边必须同步上线** |
| 服务端每帧固定慢 200 ms | 上面那个问题的旧补丁（响应后 `usleep(200000)`） | 已删除；背靠背 667 → 460 ms（1.45×） |
| 长时间跑内存持续增长 | 去噪链 `carry` 未释放，每 `S` 命令泄漏 40960 B | 已修复并验证 RSS 持平 |
| 评测跑一阵后设备 load 暴涨 | web 控制台的 NPU 采样循环泄漏（ssh 断线不回收远端 `while true`），累积数百个进程 | 已修根因（`-tt` + `ServerAlive` + 远端 `timeout`）。**该窗口内测的性能数字偏高 10-15%** |
| 同一 case 步数忽多忽少 | 噪声采样随机性（非确定性） | 对比必须固定 seed，C/D 成对（同 obs 同 seed） |

## 7. 来源

本目录内容由 π0.5 移植项目的复现包迁入，详情见该项目的里程碑报告与复现包文档。
