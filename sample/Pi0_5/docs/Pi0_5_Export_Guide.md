# π0.5 导出指南：从官方权重到 6 个 bmodel

> 本文给出**从零复现**整条编译链的完整流程。官方权重 → PyTorch → ONNX（6 个子模型）→ bmodel。
> 每一步都标注了**为什么这么做**，以及踩过的坑；只给命令、不给原因的文档在换配置时会失效。

---

## 总览

```
官方 JAX 权重 (Orbax/ocdbt, 12.4 GB)
   │  ① openpi 官方转换脚本
   ▼
PyTorch checkpoint (bf16, 7.2 GB)
   │  ② 分段导出（本 sample 自有的 graph surgery）
   ▼
ONNX × 6
   ├── pi05_siglip.onnx           视觉编码器
   ├── pi05_dkv0_9.onnx           主干 KV 前向 L0–8
   ├── pi05_dkv9_18.onnx          主干 KV 前向 L9–17
   ├── pi05_ddn0_6.onnx           动作专家去噪 L0–5
   ├── pi05_ddn6_12.onnx          去噪 L6–11
   └── pi05_ddn12_18.onnx         去噪 L12–17 + 动作头
   │  ③ ONNX 图修补（ReduceMean / opset / cumsum）
   ▼
ONNX (_fx) ── ④ TPU-MLIR ──▶ bmodel × 6
```

**为什么一开始就要分段**：主干与动作专家都是 18 层。单图在 W8BF16/BF16 下编译会失败（OOM / 编译器崩溃）。
分段是编译期约束，**不改变数学语义** —— 分段导出的输出必须与官方单图前向逐值对齐（见 §6）。

---

## 阶段 0：环境

| 组件 | 版本 | 用途 |
|---|---|---|
| `openpi` 官方仓库 | ref 分支 | 模型定义、官方转换脚本、`PI0Pytorch`。来源 <https://github.com/Physical-Intelligence/openpi> |
| PyTorch | 2.10（CPU 即可） | 导出 ONNX |
| JAX + orbax | jax 0.5.3 | 还原官方 ocdbt 权重 |
| onnxruntime | 1.23.2 | ONNX 数值校验 |
| TPU-MLIR docker | `tpu_mlir_dev:1.28` | 编译 bmodel |
| libsophon | 0.5.3（SDK 26.03.01） | **必须与 TPU-MLIR 版本匹配**，否则 bmodel 加载失败 |

> ⚠️ torch ≥ 2.6 默认 `weights_only=True`，加载 LIBERO 的 `init_states.pt` 会失败。
> 需设 `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`。

---

## 阶段 1：官方 JAX 权重 → PyTorch

```bash
# 下载官方权重（ocdbt 格式，约 12.4 GB）
#   gs://openpi-assets/checkpoints/pi05_libero
python openpi/examples/convert_jax_model_to_pytorch.py \
    --config-name pi05_libero \
    --checkpoint-dir <official_ckpt> \
    --output-path models/pi05_libero_official_pt
```

产物：`models/pi05_libero_official_pt`（bf16，约 7.2 GB，含 assets）。

> ⚠️ ocdbt 是多分片格式，**下载必须完整**。分片有空洞（sparse hole）时转换会在非零区域报错，
> 且错误信息不指向根因。建议下载后逐分片校验大小与 md5。

**为什么必须走官方转换脚本**：官方脚本内部执行 `to_bfloat16_for_selected_params("bfloat16")` ——
只把主干参数降到 bf16，`patch_embedding` / `position_embedding` 等少数参数**保留 fp32**。
自己写转换会漏掉这个豁免列表，导致视觉编码器精度悄悄退化。

---

## 阶段 2：分段导出 ONNX

导出脚本在 [`tools/export/`](../tools/export/README.md) 下。所有脚本读同一个环境变量 `PI05_ROOT`
（指向含官方权重与 openpi 检出目录的工作树），ONNX 统一产出到 `PI05_ONNX_DIR`
（默认 `${PI05_ROOT}/onnx`）。**编译前需把 `PI05_ONNX_DIR/*.onnx` 复制或软链到
`sample/Pi0_5/models/onnx/`** —— 编译脚本只从那里读。目录约定与调用顺序见该目录的 README。

### 2.1 视觉编码器（SigLIP）

```bash
python3 tools/export/extract_siglip.py          # 从 orbax ckpt 抽出视觉塔权重 → npz
python3 tools/export/export_vision_hf2.py       # 组装 PyTorch 模块并导出 onnx
```

产物 `pi05_siglip.onnx`（**opset 17，不做图修补**，直接进编译）。

- 结构：patch embed（Conv 3→1152, k/s 14）+ position embedding（1,256,1152）+ LayerNorm + MLP projector（1152→2048）
- 输出：`(1,256,2048)`

**导出侧三个必须遵守的点**（实跑踩出来的，缺一个都编不过或精度崩）：

| 点 | 不遵守会怎样 |
|---|---|
| 权重读 **`models/pi05_libero_official_pt`**，不是 `pi05_base_pytorch` | 基座与 LIBERO 微调的视觉塔差很远（图像特征 cos 0.977 / rel L2 29%），编出来的 siglip 会让整条链精度崩掉，**且不报错** |
| `torch.onnx.export(..., dynamic_axes={"images": {0: "batch"}, ...})` | trace 用 batch=1 会把 `Reshape(..., [1,256,16,72])` 烘进 attention；按 `[[2,3,224,224]]` 编译时 shape 推断直接失败 |
| `torch.onnx.export(..., dynamo=False)` | torch ≥ 2.9 默认的 torch.export 导出器拒收 ScriptModule（"Exporting a ScriptModule is not supported"） |

**判定导出是否正确的金标准**：fp32 ONNX 的 feats **norm ≈ 3776**（等于官方 siglip 的 3775）。
错误的导出会得到 norm=3020 或 487（随机权重）。**每次重新导出后必须先跑这个检查**，不合格直接废弃。
> 注意 norm 随输入而变：3776 是对**真实观测**的值；拿随机噪声当输入会得到三千零几，
> 那不代表导出错了 —— 拿不准时就用同一张图跟已知正确的 ONNX 比 cos。

### 2.2 主干 KV 前向（dkv，2 段）

```bash
python3 tools/export/export_d_kvseg_536.py
```

- 段 1（`pi05_dkv0_9`）：L0–8，输入 `prefix_embs [1,536,2048]` + `p_amask [1,1,536,536]` + `p_cos/p_sin [1,536,256]`，
  输出 18 个 KV + `hidden`
- 段 2（`pi05_dkv9_18`）：L9–17，同样输入，输出 18 个 KV

### 2.3 动作专家去噪（ddn，3 段）

一份脚本同时产出主干与动作专家的 5 个 ONNX（`pi05_dkv0_9` / `pi05_dkv9_18` /
`pi05_ddn0_6` / `pi05_ddn6_12` / `pi05_ddn12_18`）。

- 段 1（`pi05_ddn0_6`）：suffix 输入 32 维
- 段 2（`pi05_ddn6_12`）：suffix 输入 1024 维
- 段 3（`pi05_ddn12_18`）：含 final norm + `action_out_proj`，输出 `v_t [1,10,32]`

每段输入 = `suffix_in` + `time` + `f4d` + `s_cos/s_sin` + 本段负责的 12 个 KV。

### 2.4 为什么是 536 而不是 968

完整前缀 968 token，其中 **436 个是结构性死 token**：

| 区段 | token 数 | 内容 | 是否存活 |
|---|---|---|---|
| `[0,256)` | 256 | 主视角图像特征 | ✅ |
| `[256,512)` | 256 | 腕部图像特征 | ✅ |
| `[512,768)` | 256 | **第 3 路补零黑图特征** | ❌ 整段死 |
| `[768,968)` | 200 | 语言 token（真实 16–21 个 + padding） | ⚠️ 仅前 L 个存活 |

"死"的判据来自 `p_amask`：这些 token 对应的**行与列**在注意力掩码里都取 `-10000`，既不参与输出也不被注意到。

**删除前必须证明无影响**（两步实验）：
1. 宿主 float32 下把死 token 取 8 组扰动（含确定性置零、随机化），输出与原 968 版**逐位相同**；
2. 设阳性对照（改动存活 token）验证实验本身灵敏。

删除后后缀起点从 968 变 536，**所有依赖位置索引的地方都要跟着改** —— 见 §3.3。

---

## 阶段 3：ONNX 图修补（三个必做项）

导出的 ONNX 不能直接进 TPU-MLIR，需先做图修补。修补后产物以 `_fx` 结尾（如 `dkv0_9_fx.onnx`）。

```bash
python3 tools/export/fix_reducemean_536c.py    # ReduceMean → 显式 Mean（5 段）
python3 tools/export/patch_mean_axes.py        # Mean 轴修正（仅 dkv 两段）
python3 tools/export/patch_cumsum_fast.py      # cumsum 位置补丁（按需，见 tools/export/README.md）
```

修补产物统一加 `_fx` 后缀（如 `pi05_dkv0_9_fx.onnx`）—— **编译脚本吃的是 `_fx` 版本**，
只有 siglip 例外（不做修补）。

### 3.1 ReduceMean

TPU-MLIR 对某些 `ReduceMean` 形态会编译崩溃或产出错误结果。改成显式 `Mean` 算子后正常。

### 3.2 opset

导出后统一对齐到 TPU-MLIR 支持良好的 opset（本 sample 用 **opset 18** 导出再修补）。

### 3.3 ★ cumsum：最隐蔽的一个坑

官方 `DenoisePP._rope` 用的是 `0..L-1` 的朴素索引，而**真实前向用 `cumsum(pad) - 1`**
（因为 suffix 的起始位置是 prefix 长度，不是 0）。

- 不改：`v_t` 的 cos 明显偏离 1.0；
- 改对：`v_t` cos = 1.0。

**换 prefix 长度（968 → 536）时必须重新检查这个补丁** —— 它的正确性依赖当前位置索引。

---

## 阶段 4：编译 bmodel

见 `scripts/gen_siglip_bmodel_mlir.sh` / `gen_dkv_bmodel_mlir.sh` / `gen_ddn_bmodel_mlir.sh`。

```bash
cd scripts
./gen_siglip_bmodel_mlir.sh bm1684x
./gen_dkv_bmodel_mlir.sh    bm1684x
./gen_ddn_bmodel_mlir.sh    bm1684x
```

编译完成后**逐个 `bmrt_test` 验证可加载**：

```bash
bmrt_test --bmodel ../models/BM1684X/pi05_dkv0_9_w8bf16_1b.bmodel
```

### 精度档位选择依据（都经过实测，不是拍的）

| 段 | 档位 | 依据 |
|---|---|---|
| siglip | **W8BF16** | 相对 F32 参考 cos 0.99992；W4 档只有 0.99936 |
| dkv | **W8BF16** | 实测 W4 两段全压偏差 **13.38%**、只压一段 **5.68%**，而验收线是 cos ≥ 0.99997 —— 差 100~1000 倍。**W8 就是 dkv 的精度地板** |
| ddn | **BF16** | W4 收益仅约 11 ms，且 `ddn_12_18_final` 段在 W4F16/W4BF16 下**编译直接失败**（TPU-MLIR layer-group 阶段限制） |

> **不提供 F32 / INT8 / FP8 / FP4 档**：F32 约 13 GB 超出 SE7 可用内存；INT8 对连续动作的激活量化过于敏感；
> BM1684X **没有 FP4/FP8 计算单元**，实测 F4F16DYN / F8E4M3BF16DYN 反而慢 2.3~4.4%。

---

## 阶段 5：前缀资产生成

π0.5 的语言指令在训练时是固定模板 + 任务文本，且 prefix 结构随任务变化。
因此**按 task 预计算**前缀相关资产（设备端直接加载，不做运行时 tokenizer）：

```bash
python3 tools/export/gen_assets_536.py         # 由 968 资产导出 536 版
python3 tools/export/gen_kvseg_pertask.py      # 逐任务前缀资产
python3 tools/export/fix_unnorm_quantile.py    # 修正 action_unnorm.npz 为分位数仿射
```

产物落在 `datasets/pi05_libero_sample/prefix_assets/`，与设备端 `dkva536_npy/` 同构。

> ⚠️ **`.npz` 必须是 STORED（非压缩）**：设备侧加载器手工遍历 ZIP local header 并把**压缩长度**当数据长度用。
> 用 `np.savez_compressed` 生成会读出垃圾 —— 症状是 `load fail` 或**模型输出全零**。请用 `np.savez`。

> ⚠️ **`action_unnorm.npz` 必须是分位数仿射版**：官方 `_output_transform` 是
> `action = x*(q99-q01)/2 + (q01+q99)/2`，不是 `x*std + mean`。用错会让前 6 维尺度差 2.1~3.5 倍，
> 症状是**闭环步数翻倍、gripper 该松不松**，极易误判为"模型精度不够"。

---

## 阶段 6：数值验收

每一步导出后都要做受控数值对照，**不允许用"看起来合理"代替验证**：

| 对照 | 口径 | 门限 |
|---|---|---|
| siglip fp32 ONNX feats norm | 对官方 siglip | ≈ **3776** |
| 分段 ONNX vs 官方单图前向 | 同输入、同噪声下的逐步 diff | 逐段 cos ≥ 0.9999 |
| 设备 bmodel vs ONNX | **同一输入**、**同一数值空间** | cos ≥ 0.999 |
| 端到端动作轨迹 vs 宿主真值 | 10 步去噪积分后 | 相对偏差 ≤ 5% |

**两条口径纪律**（本项目踩过，会导致假差异）：

1. **比精度前必须统一数值空间**：设备返回的是 **unnorm（反归一化）空间**，离线参考常算在 **norm 空间**。
   混比会得到 24.8% / 51% / 56.7% 这类假差异。
2. **去噪步数必须对齐再比**：本 sample 默认 dn2，竞品多为 dn10。不对齐会把差距少算约 1.5 倍。

---

## 附：常见失败与定位

| 现象 | 根因 | 处理 |
|---|---|---|
| bmodel 加载失败 | libsophon 与 TPU-MLIR 版本不匹配 | 对齐版本（本 sample：libsophon 0.5.3 / SDK 26.03.01） |
| 编译崩溃 / OOM | 18 层单图过大 | 按 §2 的分段切分 |
| `unsupported op` | ONNX 里残留 TPU-MLIR 不支持的算子 | 回阶段 3 做图修补 |
| 模型输出全零 | 资产 `.npz` 被压缩存储 | 用 `np.savez` 重新打包 |
| `v_t` cos 明显偏离 1 | cumsum 位置补丁未做或 prefix 长度变了没跟着改 | 见 §3.3 |
| 动作尺度差 2~3 倍 | 反归一化用了 mean/std 而非分位数仿射 | 见阶段 5 的警告 |
| 精度莫名退化 | siglip ONNX 源不对 | 跑 feats norm 检查（应 ≈3776） |
