# tools/export —— 从官方权重导出 6 个 ONNX

这些脚本把**官方 π0.5 权重**导出成本 sample 需要的 6 个 ONNX。
完整背景与验收口径见 [`../../docs/Pi0_5_Export_Guide.md`](../../docs/Pi0_5_Export_Guide.md)。

## 环境变量

| 变量 | 必需 | 说明 |
|---|---|---|
| `PI05_ROOT` | ✅ | 工作树根目录，需含下面「目录约定」里的内容 |
| `PI05_ONNX_DIR` | | **所有**导出与修补脚本的 ONNX 读写目录，默认 `${PI05_ROOT}/onnx` |
| `PI05_ASSETS_IN` | | 968-token 逐任务资产目录（`gen_kvseg_pertask.py` 产出），默认 `${PI05_ROOT}/assets/dkva_npy` |
| `PI05_ASSETS_OUT` | | 536-token 资产目录（`gen_assets_536.py` 产出），默认 `${PI05_ROOT}/assets/dkva536_npy` |
| `PI05_UNNORM_NPZ` | | `fix_unnorm_quantile.py` 要修正的 `action_unnorm.npz` 路径 |

```bash
export PI05_ROOT=/path/to/pi05-work
```

## 目录约定（`PI05_ROOT` 之下）

```
$PI05_ROOT/
├── openpi-ref/                       # 官方 openpi 仓库（ref 分支；叫 openpi/ 也行）
├── ckpt_official/pi05_libero/params  # 官方 Orbax/ocdbt 权重
├── models/pi05_base_pytorch/         # 基础 PyTorch 权重（safetensors）
├── models/pi05_libero_official_pt/   # 官方转换脚本产出的 PyTorch checkpoint
├── obs/t00_init0/                    # 一条样例观测（agentview.npy + wrist.npy，uint8 [224,224,3]）
└── onnx/                             # ONNX 产出目录（= PI05_ONNX_DIR）
```

> **无项目内部依赖**：早先版本需要 `c_onnx/denoise_step.onnx`、`d_seg/d_inputs.npz`、
> `d532/alive.npy`、`devdata/diag_*.npy` 等移植期间的中间产物，现已全部消除 ——
> `c_onnx` 基线在 536 下本就不可比（脚本里 `v_c = None`），`d_inputs` / `alive`
> 只是形状载体与可由结构推导的索引，样例观测改为由 --obs 指定的 .npy 数组。

> openpi 检出目录叫 `openpi/` 或 `openpi-ref/` **都可以** —— 脚本会自动探测，
> 并把 `src/`、`packages/openpi-client/src/`、`third_party/libero/` 依次加入 `sys.path`。

## 脚本

| 脚本 | 作用 | 产出 |
|---|---|---|
| `extract_siglip.py` | 从 Orbax 权重抽出视觉塔，存为 npz | `$PI05_ROOT/siglip/siglip_visual_weights.npz` |
| `export_vision_hf2.py` | 组装 SigLIP PyTorch 模块并导出 | `$PI05_ONNX_DIR/pi05_siglip.onnx`（**opset 17，不做图修补**） |
| `export_d_kvseg_536.py` | 分段导出 KV 主干与动作专家 | `$PI05_ONNX_DIR/pi05_{dkv0_9,dkv9_18,ddn0_6,ddn6_12,ddn12_18}.onnx` |
| `fix_reducemean_536c.py` | ReduceMean → 显式 Mean | `pi05_dkv*_fx.onnx` / `pi05_ddn*_fx.onnx` |
| `patch_mean_axes.py` | Mean 轴修正（仅 dkv 两段）⚠️ 见下 | 同上，就地修补 |
| `patch_cumsum_fast.py` | cumsum 位置补丁（**换 prefix 长度必查**）⚠️ 见下 | 同上 |
| `gen_libero_prefix.py` | 前缀构建工具：tokenize / build_prefix（被 `gen_kvseg_pertask` 导入） | 模块 |
| `gen_kvseg_pertask.py` | 生成 968-token 逐任务前缀资产与语言嵌入（`--obs` / `--tokenizer` / `--policy-dir`） | `$PI05_ASSETS_IN/tXX_{p_amask,p_cos,p_sin,f4d,s_cos,s_sin}.npy`、`$PI05_ASSETS_IN/prompt200_tXX.npy` |
| `gen_assets_536.py` | 由 968 资产导出 536 资产（供设备加载） | `$PI05_ASSETS_OUT/tXX_*.npy` |
| `export_denoise_pp.py` | 导出侧配置与辅助（被 `gen_libero_prefix` 导入） | 模块 |
| `fix_unnorm_quantile.py` | 把 `action_unnorm.npz` 修正为分位数仿射系数 | 就地修补，旧文件备份为 `.bak_meanstd` |

> ⚠️ `patch_mean_axes.py` 与 `patch_cumsum_fast.py` 是从项目的导出目录一并带过来的：
> 移植时确实撞上过 tpu-mlir 对 `ReduceMean` / `CumSum` 常量链的限制，但**无法确认 H5（prefix 536）
> 这条链一定需要它们**。请当作"编译报错时再试"，而不是必跑步骤。`fix_reducemean_536c.py`
> 则是确定的 —— 它的段列表就是 H5 的 5 段，`_fx` 后缀由它产出。

## 调用顺序

```bash
export PI05_ROOT=/path/to/pi05-work

python3 extract_siglip.py           # 1. 抽视觉塔权重
python3 export_vision_hf2.py        # 2. -> pi05_siglip.onnx
python3 export_d_kvseg_536.py       # 3. -> pi05_dkv*_fx 的输入（5 个 ONNX）
python3 fix_reducemean_536c.py      # 4. 图修补（确定需要，产出 _fx）
python3 patch_mean_axes.py          #    按需：编译报 ReduceMean 相关错误时再试
python3 patch_cumsum_fast.py        #    按需：编译报 CumSum 相关错误时再试
python3 gen_assets_536.py           # 5. 前缀资产
python3 gen_kvseg_pertask.py
python3 fix_unnorm_quantile.py      # 6. 反归一化系数

# 7. 把 ONNX 放到编译脚本读取的位置（复制或软链）
ln -sfn "$PI05_ONNX_DIR"/*.onnx ../models/onnx/     # 或 cp

# 8. 编译（回到 sample 根目录）
cd ../../scripts && ./gen_siglip_bmodel_mlir.sh bm1684x
./gen_dkv_bmodel_mlir.sh bm1684x
./gen_ddn_bmodel_mlir.sh bm1684x
```

产出对应关系：`gen_siglip_*` 吃 `pi05_siglip.onnx`（siglip 不做图修补），
`gen_dkv_*` / `gen_ddn_*` 吃 `pi05_*_fx.onnx`（**图修补后**的版本）。

## 资产链的输入与产出

```
gen_kvseg_pertask.py                     gen_assets_536.py
  ┌─────────────────────────┐              ┌──────────────────────────┐
  │ tXX_p_amask/p_cos/p_sin │─────────────▶│ tXX_*.npy  (536-token)   │
  │ tXX_f4d / s_cos / s_sin │              │ promptL_tXX.npy          │
  │ prompt200_tXX.npy       │─────────────▶│                          │
  └─────────────────────────┘              └──────────────────────────┘
       968-token 逐任务资产                    设备实际加载的 536-token 集
```

`prompt200_tXX.npy` 是前缀里**语言那一半**（3×256 个图像 token 之后的 200 槽），
逐任务不同、必须固化；图像那一半由设备用 SigLIP bmodel 按真实观测现算，所以
`--obs` 给的样例观测只是个形状载体，换一张图不影响产出。

## 前置资产

| 需要什么 | 从哪来 |
|---|---|
| 官方 Orbax 权重 | `gs://openpi-assets/checkpoints/pi05_libero` |
| PyTorch checkpoint | 官方 `convert_jax_model_to_pytorch.py` 产出 |
| PaliGemma tokenizer | 官方 PaliGemma 资产，用 `--tokenizer` 指定 |
| 一条样例观测 | 任意 224×224 双路观测；`--obs` 指定，默认取数据集里的 `obs/t00_init0` |

## 验证状态

**整条链已在独立工作树实跑通过**（`PI05_ROOT` 指向临时目录 + 软链输入），并一路做到 bmodel：

| 阶段 | 状态 |
|---|---|
| `extract_siglip.py` | ✅ 实跑通过：439 个张量 |
| `export_vision_hf2.py` | ✅ 实跑通过，产出 `pi05_siglip.onnx`（1.66 GB）。与**线上那份 bmodel 的源 ONNX** 在同一输入下 cos = 1.000000 / rel L2 0.06% |
| `export_d_kvseg_536.py --export` | ✅ 实跑通过，5 个 ONNX 全出（dkv 带外部数据） |
| `fix_reducemean_536c.py` | ✅ 实跑通过：dkv 两段各改写 18 / 17 个 ReduceMean，ddn 三段 0 个 |
| `gen_kvseg_pertask.py` → `gen_assets_536.py` | ✅ 实跑通过，产出 71 个 536 资产，与设备在服务的那份**逐文件 md5 完全一致** |
| 编译（`scripts/gen_*bmodel_mlir.sh`，tpu_mlir_dev:1.28） | ✅ 实跑通过，6 个 bmodel 体积与线上逐个一致；拿到设备上跑同一条观测，**动作输出与线上 bmodel 逐位相同** |

**怎样算"导出正确"——两道自检**：

1. **siglip feats norm**：官方 siglip 的 feats norm ≈ 3776。错误的导出会得到 3020 或 487。
2. **与线上 bmodel 的源 ONNX 对比**：手上有可用的旧 ONNX 时，喂同一张图比 cos，比绝对值可靠得多。

**审计中发现并修复的问题**（这就是为什么值得真跑一遍）：

1. `export_vision_hf2.py` 假设检出目录叫 `openpi/`，而部分脚本假设 `openpi-ref/` —— 已统一为自动探测两者。
2. 三个导出脚本原先各自写到 `siglip/`、`vlm/vision/`、`d536/out/` 三个不同目录，
   而编译脚本只读 `models/onnx/` —— 使用者会拿到散落三处的 ONNX 然后编译报文件不存在。
   已统一到 `PI05_ONNX_DIR`，并在调用顺序里补上「放到 models/onnx/」这一步。
3. `gen_kvseg_pertask.py` 依赖同目录的 `gen_libero_prefix.py`，而后者又依赖 `export_denoise_pp.py`
   —— 这两个都没随首批迁入，链在资产生成这一步会直接 ImportError。已补齐。
4. `fix_onnx_opset.py` 看着像通用 opset 修补，实际只服务 `vlm/repro/`（旧架构的 scratch 目录），
   **不属于这条链**，留着会让使用者跑出莫名的 IndexError。已移除。
5. **脚本里硬编码了设备 IP 与登录密码**（`linaro@172.26.166.88` / `-p linaro`）—— 迁入公开仓库
   前必须清掉。已随设备 `scp` 步骤一并移除，改为只写本地目录，设备分发交给使用者。
6. （上一轮）`gen_siglip` 引用了 siglip 永远不会产出的 `_fx` 版本。
7. **`export_vision_hf2.py` 读错了 checkpoint**：它加载的是 `pi05_base_pytorch`（π0.5 **基座**模型，
   `action_horizon=50`），而其余导出脚本读的都是 `pi05_libero_official_pt`。LIBERO 微调把视觉塔
   挪得很远 —— 两个 checkpoint 的图像特征 **cos 只有 0.977、rel L2 29%**，用基座权重编出来的
   siglip 会让整条链精度崩掉而且不报错。已改为读 LIBERO checkpoint。
8. **siglip 的 batch 维必须导出成动态的**：`torch.jit.trace` 用 batch=1 跟踪会把
   `Reshape(..., [1,256,16,72])` 烘进 attention，编译时按 `[[2,3,224,224]]` 做 shape 推断直接失败
   （`Input shape:{2,256,1152}, requested shape:{1,256,16,72}`）。加 `dynamic_axes` 后正常。
9. **torch ≥ 2.9 的默认导出器拒收 ScriptModule**（"Exporting a ScriptModule is not supported"），
   必须显式 `dynamo=False` 走回旧的 TorchScript 导出路径。
10. `export_d_kvseg_536.py` 里有一处引用已删变量的调试 print（`d['prefix_embs']`），
   一跑就 NameError —— 只有真正执行到那一行才会暴露。

若你在干净环境上跑通了全流程，欢迎把踩到的坑补进
[`../../docs/Pi0_5_Export_Guide.md`](../../docs/Pi0_5_Export_Guide.md) 的「常见失败与定位」。
