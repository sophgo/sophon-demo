# dfss 上传清单（需人工执行）

## 要传什么、传到哪

**只有两个文件**，都已打包好放在 SE7 设备 `/data2/pi05s/` 下：

| # | 设备上的文件 | 上传到 dfss 的路径 |
|---|---|---|
| 1 | `/data2/pi05s/BM1684X.tar.gz` | `open@sophgo.com:sophon-demo/Pi0_5/models/BM1684X.tar.gz` |
| 2 | `/data2/pi05s/pi05_libero_sample.tar.gz` | `open@sophgo.com:sophon-demo/Pi0_5/datasets/pi05_libero_sample.tar.gz` |

上传后校验：

| 文件 | 体积 | md5 |
|---|---|---|
| `BM1684X.tar.gz` | 3,082,299,317 B | `6faff97b38e748c75af753f433e26685` |
| `pi05_libero_sample.tar.gz` | 11,886,111 B | `51b711f2054c2644a1915d64c867009b` |

这两个路径与 `scripts/download.sh` 里的 URL 一一对应，传完 `./scripts/download.sh bm1684x`
就能跑通。**在这之前 download.sh 必然失败** —— 这是当前唯一的阻断项。

## 一、目录布局（上传后在 dfss 上的样子）

```
open@sophgo.com:sophon-demo/Pi0_5/
├── models/
│   └── BM1684X.tar.gz          # 6 个 bmodel，解包进 models/BM1684X/
└── datasets/
    └── pi05_libero_sample.tar.gz   # 观测样本 + 官方动作真值 + 前缀资产
```

## 二、`models/BM1684X.tar.gz`

> ✅ **已打包好，可直接上传**：SE7 设备上 `/data2/pi05s/BM1684X.tar.gz`
> （3,082,299,317 B，md5 `6faff97b38e748c75af753f433e26685`，顶层就是 6 个 `.bmodel`，
> 与 `download.sh` 的解包位置对应）。

| 打包后文件名（**需改名**） | 设备上原路径 | 体积 | md5 |
|---|---|---|---|
| `pi05_siglip_w8bf16_2b.bmodel` | `/data2/pi05s/demo_work/siglip_visual_bm1684x_w4bf16.bmodel` | 571,981,824 B | `a44cb26930d30c155b7d0d7461413f68` |
| `pi05_dkv0_9_w8bf16_1b.bmodel` | `/data2/pi05s/d536/dkv0_9_bm1684x_W8BF16_H5.bmodel` | 1,045,651,456 B | `1ee11ce7d9cbbeaa25ba94485a0ea6ea` |
| `pi05_dkv9_18_w8bf16_1b.bmodel` | `/data2/pi05s/d536/dkv9_18_bm1684x_W8BF16_H5.bmodel` | 931,155,968 B | `fb77908097d53fe7acb418adf8cc3e6a` |
| `pi05_ddn0_6_bf16_1b.bmodel` | `/data2/pi05s/d536/ddn_0_6_bm1684x_BF16_H5.bmodel` | 297,168,896 B | `84db682b4f8d6914259d7e1b3a30d8a8` |
| `pi05_ddn6_12_bf16_1b.bmodel` | `/data2/pi05s/d536/ddn_6_12_bm1684x_BF16_H5.bmodel` | 297,099,264 B | `f72e663ca3673ccb9f610136dcc4e4b4` |
| `pi05_ddn12_18_bf16_1b.bmodel` | `/data2/pi05s/d536/ddn_12_18_final_bm1684x_BF16_H5.bmodel` | 303,476,736 B | `b057ad8aaf675044de9e1c107ace1725` |

合计约 **3.45 GB**。

> **这 6 个文件已用文档里的流程从官方权重重新导出+编译验证过**：体积逐个一致，
> 详见 [`../tools/export/README.md`](../tools/export/README.md) 的「验证状态」。

> ⚠️ **`pi05_siglip_w8bf16_2b` 的设备原文件名有误导** —— 它叫 `siglip_visual_bm1684x_w4bf16.bmodel`，
> 但实际是 **W8BF16、batch=2**（md5 `a44cb269…` = `siglip_visual_b2_w8.bmodel`，官方 cos 0.99992）。
> 打包时**必须按上表左列改名**，否则使用者会以为拿到的是 W4 档。

## 三、`datasets/pi05_libero_sample.tar.gz`

> ✅ **已打包好，可直接上传**：SE7 设备上 `/data2/pi05s/pi05_libero_sample.tar.gz`
> （11,886,111 B，md5 `51b711f2054c2644a1915d64c867009b`，328 个条目，顶层目录即
> `pi05_libero_sample/`，与 `download.sh` 的解包位置对应）。

| 内容 | 来源 | 体积 | 说明 |
|---|---|---|---|
| `obs/` 50 条固定 seed 观测 | LIBERO-Spatial 评测导出（见第四节） | 约 15 MB | 每条含 `agentview.npy` + `wrist.npy`，uint8 `[224,224,3]` RGB |
| `noise/` 50 条初始噪声 | `tools/make_reference.py` 产出 | 约 64 KB | 每条一个 `.npy`，float32 `[10,32]`；设备端用 `--noise` 喂入，保证与真值同源 |
| `actions_gt/` 官方动作真值 | 官方策略在同一观测、同一噪声上的输出 | 约 64 KB | 每条一个 `.npy`，形状 `[10,7]`（unnorm 空间前 7 维） |
| `index.json` | `tools/make_sample_dataset.py` 产出 | 约 10 KB | 逐 case 记录 task_id / init_state / seed / prompt |
| `prefix_assets/`（即设备上的 `dkva536_npy/`） | `/data2/pi05s/demo_work/dkva536_npy/` | 24 MB，71 个文件 | 每 task 6 个张量 `tXX_{p_amask,p_cos,p_sin,f4d,s_cos,s_sin}.npy` + 语言前缀 `promptL_tXX.npy`（**单次推理例程实际读的就是这 7 类**）；另有 `alive.npy` 仅作调试参照，例程不读 |
| `cond_weights_libero.npz` | `/data2/pi05s/demo_work/cond_weights_libero.npz` | 8,533,554 B | action_in_proj + time_mlp 权重；**仅进阶服务用**，单次推理例程不读 |
| `action_unnorm.npz` | `/data2/pi05s/demo_work/action_unnorm.npz` | 556 B | 分位数仿射反归一化系数 |

> ⚠️ **两个必须遵守的资产约束**（踩过的坑，会导致"精度静静退化"且不报错）：
> 1. `action_unnorm.npz` 必须是**分位数仿射**版（`x*(q99-q01)/2 + (q01+q99)/2`），不是 `x*std + mean` 版；
>    期望 `mean[:7] = [0.094875, 0.031688, -0.000188, 0.012212, 0.005802, 0.058522, -0.0002]`。
> 2. 所有 `.npz` 必须是 **STORED（非压缩）** 打包 —— 设备侧加载器手工解析 ZIP local header，
>    DEFLATE 压缩的会读出垃圾。用 `np.savez`，**不要** `np.savez_compressed`。

## 四、观测样本集的生成方式（50 条，固定 seed）

bmodel 只有 3.45 GB，但"官方动作真值"必须现算，不能凭空造。生成分两步，
因为两步需要的环境不同（脚本都在 `tools/`）：

1. `make_sample_dataset.py` —— 在官方 LIBERO-Spatial 环境（`libero:local` docker）里，
   按**官方评测协议**采集 50 条观测（10 任务 × 5 初始状态）：固定 seed → reset →
   载入记录的初始状态 → 空动作走 10 步 → 旋转 180° → `resize_with_pad(224)`；
2. `make_reference.py` —— 在 openpi 环境里，用**官方 PyTorch 权重 + openpi 模型代码**
   在同一观测上推理，得到 `actions_gt/`；同时把该 case 的初始噪声写进 `noise/`。

> **噪声必须一起发**：去噪从噪声开始，噪声是输入的一部分。两边各抽各的，
> 比出来的差异里就混着采样器差异。`noise/` 与 `actions_gt/` 必须由同一次运行产出。

推荐 50 条，与业界做法一致（见 README §6 的指标口径说明）。

## 五、上传后的自检

```bash
# 在任意一台有网机器上
cd sample/Pi0_5/scripts && ./download.sh bm1684x
cd .. && md5sum models/BM1684X/*.bmodel   # 与本文档第二节的 md5 逐项比对
```

md5 不一致说明上传/下载损坏，**不要继续**。
