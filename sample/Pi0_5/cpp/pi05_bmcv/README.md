# π0.5 C++ 例程（单次推理）

读一条观测（相机图像 + 任务 id），跑完整链路，输出动作 chunk，并给出与官方参考的数值对照。
相机接口按官方 **3 个图像槽**预留，本 sample 的 bmodel 只编了其中 2 个（见 §5）。

**不做的事**：不连仿真环境、不跑闭环、不常驻服务 —— 那些在 [`../pi05_service/`](../pi05_service/README.md)。

## 1. 目录

```
cpp/pi05_bmcv/
├── CMakeLists.txt      # 只保留 soc 分支（本期只做 BM1684X SoC）
├── main.cpp            # 入口：命令行参数解析 + 流程编排 + 写出 action.npy
├── pi05.h / pi05.cpp   # 推理类：加载 6 个 bmodel、拼前缀、跑去噪循环、反归一化
├── npy_io.h            # .npy / .npz 读取 + .npy 写出（header-only）
└── README.md
```

## 2. 编译

只支持 **BM1684X 的 SoC 模式**（SE7 系列）。**唯一依赖是 libsophon** ——
观测数据是 `.npy` 数组，例程不需要任何图像解码库。

**推荐：在设备上原生编译**（SE7 自带 g++ 9.x 与 libsophon，省掉交叉工具链）：

```bash
mkdir -p build && cd build
cmake .. -DTARGET_ARCH=soc -DSDK=/opt/sophon/libsophon-current
make -j
# 产物：cpp/pi05_bmcv/pi05_bmcv.soc
```

**或者：在 x86 主机上交叉编译**，再把产物拷到设备：

```bash
mkdir -p build && cd build
cmake .. -DTARGET_ARCH=soc -DSDK=<soc-sdk 绝对路径>     # 或 export SOC_SDK=...
make -j
scp pi05_bmcv.soc <user>@<device>:~/
```

BM1684X 用 `aarch64-linux-gnu-g++`（**g++ 9.x**）。注意工具链版本要与设备运行时匹配，
否则二进制拷过去会因 `GLIBCXX` / `GLIBC` 版本过新而起不来 —— 详见
[FAQ 4.1](../../FAQ.md#41-交叉编译出来的二进制为什么在设备上跑不起来)。

## 3. 命令行

```bash
./pi05_bmcv.soc --bmodel_dir ../models/BM1684X \
                --data_dir ../datasets/pi05_libero_sample \
                --input ../datasets/pi05_libero_sample/obs/t00_init0 \
                --task_id 0 \
                --num_steps 2 \
                --loops 1 \
                --seed 0 \
                --dev_id 0 \
                --output results/t00_init0.npy
```

| 参数 | 默认 | 说明 |
|---|---|---|
| `--bmodel_dir` | `../models/BM1684X` | 6 个 bmodel 所在目录 |
| `--input` | 必填 | 一条观测的目录，需含 `agentview.npy` 与 `wrist.npy`（uint8 `[224,224,3]` RGB） |
| `--cam3` | 无 | 第三路相机（官方槽位 `right_wrist_0_rgb`），同样的 `.npy` 布局。**当前 bmodel 只有两个图像槽，传了会直接报错** —— 见 §5 的相机槽位说明 |
| `--data_dir` | `--input` 的上级目录 | 数据集根目录，内含 `prefix_assets/` 与 `action_unnorm.npz` |
| `--task_id` | 0 | 任务编号 0–9，决定加载哪一套前缀资产 |
| `--num_steps` | 2 | 去噪步数 |
| `--loops` | 1 | 重复推理次数，用于性能测试 |
| `--seed` | 0 | 噪声种子；固定值可复现（SplitMix64 + Box-Muller，不依赖标准库实现） |
| `--noise` | 无 | 从 `.npy`（float32 `[10,32]`）读初始噪声，替代 `--seed` 生成；
与参考实现喂同一个文件，比出来的才是移植误差 |
| `--dev_id` | 0 | TPU 设备号 |
| `--output` | `results/action.npy` | 动作输出路径 |

输出：

- `results/*.npy` —— 反归一化后的动作，形状 `[10,7]`
- 标准输出打印分阶段耗时：`preprocess / siglip / dkv / ddn / postproc / total`

## 4. 精度对照

数据集里每个 case 都带一份官方真值 `actions_gt/` 和一份初始噪声 `noise/`。
逐 case 跑一遍再整体比对：

```bash
# 在设备上，从 sample 根目录
mkdir -p results
for d in datasets/pi05_libero_sample/obs/*/; do
  name=$(basename "$d")
  ./cpp/pi05_bmcv/pi05_bmcv.soc --bmodel_dir models/BM1684X \
      --data_dir datasets/pi05_libero_sample --input "$d" \
      --task_id "$((10#$(echo "$name" | cut -c2-3)))" --num_steps 2 \
      --noise "datasets/pi05_libero_sample/noise/$name.npy" \
      --output "results/$name.npy"
done

python3 tools/compare_acc.py --pred_dir results \
        --gt_dir datasets/pi05_libero_sample/actions_gt
```

或者直接 `./auto_test.sh -m soc_test`（bmrt_test 理论延迟 + 全数据集精度一次跑完）。

按 [主 README §6](../../README.md) 的口径报 **cosine similarity（Overall + per-timestep Mean/Min/Max）**，
门限 `cos ≥ 0.999` 为正常、`≥ 0.99` 为可接受下限。

## 5. 实现要点

1. **输入契约**：每一路观测各为 **uint8 `[224,224,3]`、RGB、行主序、224×224**。
   数据集里的图已是 `rotate180 → resize_with_pad(224)` 之后的（与官方 client 一致），
   例程只做 `rgb/127.5 - 1.0 → CHW [-1,1]`。换成原始相机帧时，预处理在 client 侧完成，
   见 [../pi05_service/README.md](../pi05_service/README.md)。
2. **相机槽位（3 路，本 sample 只用前 2 路）**：pi0.5 的 PaliGemma 固定吃 **3 个图像槽**，
   官方命名与顺序是

   | 槽位 | 本 sample | 说明 |
   |---|---|---|
   | `base_0_rgb` | `agentview.npy` | 第三人称主视角 |
   | `left_wrist_0_rgb` | `wrist.npy` | 腕部视角 |
   | `right_wrist_0_rgb` | **无** | 官方 LIBERO 策略**补零 + 掩码置 False**，256 个 token 结构性死亡 |

   第三路的判据在 openpi 自己的代码里（`src/openpi/policies/libero_policy.py`）：
   `"right_wrist_0_rgb": np.zeros_like(base_image)` 配 `image_mask ... np.False_`。
   所以本 sample 的 bmodel 只编了两个图像槽、前缀是 **536** 而不是 968（详见
   [Export Guide §2.4](../../docs/Pi0_5_Export_Guide.md)）。

   **接口已经按 3 路预留**：`pi05.h` 里 `kNumViews / kTokensPerView / kVisTokens / kPrefixLen`
   是一条推导链（`kPrefixLen = kNumViews * kTokensPerView + kLangSlots`），
   `infer()` 收第三路（可为空，空则补零 —— 与官方行为一致），命令行对应 `--cam3`。
   接真第三路相机需要：把 `kNumViews` 改成 3、重编 siglip（batch=3）与 dkv（前缀 792），
   并重新生成前缀资产。**在那之前传 `--cam3` 会直接报错而不是被静默忽略** ——
   悄悄丢掉一路输入会让结果看起来正常但其实是错的。
3. **siglip 是 batch=kNumViews**：所有相机合成一个 batch，权重只加载一遍（2 路时省约 14 ms/次）。
4. **denoise 循环**：`for step in range(num_steps)` —— 每步依次跑 3 段 ddn，宿主侧做 Euler 积分
   `x_{t+1} = x_t + dt·v_t`，`dt = -1/num_steps`，`time` 每步下发。
   与官方 `sample_actions` 的 `while time >= -dt/2` 等价：`num_steps=2` 时取 `time ∈ {1.0, 0.5}`。
5. **初始噪声是 N(0,1)**：官方采样器抽的是标准正态，不是均匀分布 —— 换成均匀会让收敛到的
   动作 chunk 变样。`make_noise()` 用 SplitMix64 + Box-Muller 复现同一分布，并保证跨平台一致。
6. **没有 state 输入**：pi0.5 与 pi0 的差别之一就是不把机器人状态拼进 suffix
   （openpi 里 `embed_suffix` 的 state 分支带 `if not self.pi05`）。所以 suffix 就是 10 个动作
   token，例程也不需要读状态量。
7. **KV 常驻设备**：dkv 两段产出的 36 个 KV 留在设备内存，ddn 各段直接以设备指针引用，不来回搬运；
   dkv0 的 `hidden` 输出也直接以设备指针喂给 dkv1。注意这 36 个 KV **就是 dkv 两段自己的输出缓冲**
   （每个网络在 `net_bufs_init` 里分配一次、反复复用），例程只是别名引用、不拥有它们 ——
   重复释放同一批设备地址会在退出时炸掉。
8. **反归一化**：分位数仿射 `action = x·scale + bias`，取前 7 维。
   系数从 `action_unnorm.npz` 读（键名沿用上游的 `mean`/`std`，**语义实为 scale/bias**）。
9. **内存**：设备 in/out tensor 按网缓存、进程内复用 —— 逐次 malloc/free 会让常驻进程内存单向增长。
   段间中间结果用完即 free，否则每步泄漏 40 KB。

## 6. 已验证 / 待验证

| 项 | 状态 |
|---|---|
| **设备实跑（端到端）** | ✅ **已在 SE7-32 上跑通**：6 个 bmodel 加载正常，单次推理 **445.6 ms**，输出 `[10,7]` 有效动作 |
| **设备原生编译** | ✅ `cmake .. -DTARGET_ARCH=soc -DSDK=/opt/sophon/libsophon-current && make` 通过（g++ 9.4.0） |
| 6 个 bmodel 的加载与前向链路 | ✅ 网络输入/输出数与 Export Guide 记录一致（dkv0_9 4→19、dkv9_18 4→18、ddn 17→1） |
| **与官方参考的 cosine similarity** | ✅ **50 条全过**：overall cos 均值 0.999978（最低 0.999960），per-timestep 均值 0.999983（最低 0.999759） |
| **进程退出码** | ✅ 全部 50 次运行 `EXIT=0`（退出时不再 SIGSEGV） |
| `.npy` / `.npz` 读取、`.npy` 写出 | **已单测**：与 numpy 逐位一致；DEFLATE 压缩的 `.npz` 被显式拒绝；写出的 `.npy` 能被 numpy 正确读回 |
| 固定 seed 噪声的确定性与分布 | **已单测**：同 seed 逐位相同、服从 N(0,1)（与官方采样器一致） |
| 三路相机接口 | ✅ 实测：不传 `--cam3` 时输出与改动前**逐位相同**；传了会以 `EXIT=1` 明确拒绝（不静默丢弃）；`--help` 列出该参数 |
| SoC 交叉编译 | ✅ 实测：`cmake .. -DTARGET_ARCH=soc -DSDK=<soc-sdk>` + `make` 产出 aarch64 ELF。**注意工具链版本** —— 宿主是 GCC 16 时二进制要求 glibc 2.38 / GLIBCXX 3.4.32，而 SE7 只有 2.31 / 3.4.28，拷过去起不来；`scripts/build.sh` 会自动检查并告警 |
