# DINOv2 模型导出

本文档说明如何把 DINOv2（`dinov2_vitb14_reg`）从 PyTorch 权重导出为 ONNX，再编译为 SOPHON BModel，以及如何准备评估/标定数据集。

> **权重许可**：DINOv2 的原始 PyTorch 权重（骨干 `.pth` 与官方 linear head `.pth`）按 Meta 的 [DINOv2 License](https://github.com/facebookresearch/dinov2/blob/main/LICENSE) 分发，**不随本仓库提供**，需自行下载（官方仓库 README 提供直链，见 §1.4）。本仓库的 `scripts/download.sh` 只提供已导出的 ONNX、编译好的 BModel、测试数据集与由官方 linear head 转换出的 `.npz/.bin` 权重。

## 1. 准备工作

> **适用范围**：只有“自行导出 ONNX / 自行编译 bmodel”才需要本节环境；直接使用 `scripts/download.sh` 下发产物的用户可跳过。导出**不依赖 TPU-MLIR**，也不要直接使用 TPU-MLIR 容器自带的 Python 环境（官方 tpuc_dev 镜像自带的 torch 通常为 2.0/2.1，低于本脚本要求的 2.6）。

### 1.1 环境要求

- x86_64 Linux（实测 Ubuntu 22.04 容器），Python 3.10（实测）
- 无需 GPU，CPU 版 PyTorch 即可
- 磁盘：成品 ONNX 约 343MB；导出时会另生成一个同量级的 `_raw.onnx`，默认在自检后删除

### 1.2 安装依赖

建议使用独立的 venv，避免与 TPU-MLIR / 其他例程的 Python 环境互相污染（若 `python3 -m venv` 报 `ensurepip is not available`，先 `sudo apt install python3-venv`）：

```bash
python3 -m venv ~/dinov2_export
source ~/dinov2_export/bin/activate
python3 -m pip install -U pip

# torch >= 2.6 才支持 torch.onnx.export(..., dynamo=True)；
# --index-url 选 CPU wheel，避免默认源拉取 2GB+ 的 CUDA 运行时（导出不需要 GPU）
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# ONNX 导出/自检与标定数据生成所需
pip3 install "onnx>=1.14" "onnxsim>=0.4" "onnxruntime>=1.16" "pillow>=10"
```

> 实测 PyTorch **2.12.1+cpu**（成品 ONNX 的 `producer` 字段也记录该版本）；`torch 2.6` 起才提供 `dynamo=True` 导出路径，更低版本会在 `torch.onnx.export` 处报参数错误。国内网络访问 `download.pytorch.org` 慢时可换可达的 PyTorch 镜像源，装完按 §1.5 自检。
>
> 其余依赖也可用 `pip3 install -r scripts/requirements-export.txt` 安装。该文件**故意不含 torch/torchvision**：requirements 无法指定 CPU 索引，混在一起安装可能把已装好的 CPU 版 torch 替换成 CUDA 版。

### 1.3 获取官方 DINOv2 源码（仅作源码依赖，不安装）

```bash
git clone https://github.com/facebookresearch/dinov2.git ~/dinov2
cd ~/dinov2
git checkout 7764ea0        # 本例程验证所用 commit
```

导出脚本通过 `--dinov2_src ~/dinov2` 把该目录加入 `sys.path`，只用到模型注册与 `interpolate_pos_encoding()`，因此：

- **不要** `pip install -e ~/dinov2`；
- **不要**安装官方 `requirements.txt`：它把 `torch==2.0.0`、`xformers==0.0.18`（cu117）钉死，会覆盖 1.2 装好的 torch>=2.6，导致 `dynamo=True` 导出失败；
- 源码中的 xformers 是可选依赖，导出脚本已强制 `XFORMERS_DISABLED=1` 并替换 Attention 实现，无需安装。

### 1.4 获取预训练权重（Meta License）

官方仓库 README 提供直链（按 Meta 的 [DINOv2 License](https://github.com/facebookresearch/dinov2/blob/main/LICENSE) 分发，下载即表示同意）：

```bash
mkdir -p models/torch
wget -P models/torch https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_reg4_pretrain.pth
wget -P models/torch https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_reg4_linear_head.pth
```

两个文件对应后续命令里的 `models/torch/dinov2_vitb14_reg4_pretrain.pth` 与 `models/torch/dinov2_vitb14_reg4_linear_head.pth`。若网络无法直连，需自行向 Meta 申请/走代理获取；文件名与上述一致即可。

### 1.5 环境自检

```bash
python3 - <<'PY'
import inspect, torch, torchvision, onnx, onnxsim, onnxruntime, PIL
from importlib.metadata import version
print("torch", torch.__version__, "| torchvision", torchvision.__version__)
print("onnx", onnx.__version__, "| onnxsim", version("onnxsim"),
      "| onnxruntime", onnxruntime.__version__, "| pillow", PIL.__version__)
assert tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2]) >= (2, 6), "需要 torch >= 2.6"
assert "dynamo" in inspect.signature(torch.onnx.export).parameters, "该 torch 不支持 dynamo 导出"
print("export deps OK")
PY

# 只验证源码可导入（不下载权重）
PYTHONPATH=~/dinov2 python3 -c "from dinov2.hub.backbones import dinov2_vitb14_reg; print('dinov2 source OK')"
```

最后一条失败时，检查 clone 是否完整、commit 是否切换成功、`--dinov2_src` 是否指向该目录。

### 1.6 各脚本的依赖范围

| 脚本 | 需要的依赖 | 运行位置 |
| --- | --- | --- |
| `export_onnx.py` | torch>=2.6、onnx、onnxsim、onnxruntime、numpy + 官方 dinov2 源码 | 上述 venv（任意 x86 Python 环境） |
| `export_linear_head.py` | torch、numpy | 同上 |
| `gen_val_1k.py` | 仅 Python 标准库（zipfile/json） | 任意 Python 3 |
| `gen_cali_data.py` | torch、torchvision、pillow、numpy | 上述 venv |
| `gen_qtable.py` | `tpu_mlir`（`from tpu_mlir.python.utils.mlir_parser import MlirParser`） | **必须在 TPU-MLIR 容器内** |

> `gen_qtable.py` 由 `gen_int8bmodel_mlir.sh` 调用，不需要 torch；其余四个脚本不需要 TPU-MLIR。

## 2. 模型说明

- 模型：`dinov2_vitb14_reg`（ViT-B/14 + 4 个 register token，86M 参数）。
- **输入口径**：官方 ImageNet-1k linear-probe 评估用的是 `dinov2/data/transforms.py` 的 `make_classification_eval_transform()`，其默认参数为 `Resize(256, BICUBIC)` -> `CenterCrop(224)`，即**模型输入 224x224**（官方 MODEL_CARD 的 ViT-B/14-reg4 精度 84.6 就是这个口径；仓库 README 另一张“已发布线性头”表对该 reg4 头记 84.4，同属官方数字，见 `README.md` §6.2 注）。因此本例程导出 `1x3x224x224` 的静态输入，单 batch。
- **输出口径**：官方线性探针特征（`dinov2/eval/linear.py: create_linear_input`，`use_n_blocks=1`、`use_avgpool=True`）：

  ```
  concat([x_norm_clstoken, mean(x_norm_patchtokens, dim=1)])   # [B, 1536]
  ```

  推理时在 CPU 侧再接官方 1 层线性头 `dinov2_vitb14_reg4_linear_head`（`weight[1000,1536]` + `bias[1000]`）得到 ImageNet-1k 1000 类 logits，取 Top-1。线性头不编进 BModel。

## 3. ONNX 导出

```bash
python3 scripts/export_onnx.py \
    --dinov2_src <path/to/dinov2_repo> \
    --weights models/torch/dinov2_vitb14_reg4_pretrain.pth \
    --out_onnx models/onnx/dinov2_vitb14_reg_224_1b.onnx
```

要点（详见 `scripts/export_onnx.py` 顶部注释）：

- **位置编码烘焙**：预训练权重的 `pos_embed` 是按 518x518（37x37 网格）训出来的，形状 `(1, 1370, 768)`。官方在喂 224 图时会在 forward 内调用 `interpolate_pos_encoding()` 把 37x37 网格 bicubic 插值到 16x16（`antialias=True`、`offset=0.0`）。导出前用官方同一个函数把插值结果算好并替换 `model.pos_embed`（一次性烘焙，数值与官方逐次插值完全一致），此后 forward 中 `npatch == N and w == h` 短路，**图中不含 Resize 算子**（TPU-MLIR 对 bicubic 支持有限）。脚本会断言图中没有 `Resize` 节点。
- **Attention**：DINOv2 默认走 torch>=2.0 的 `F.scaled_dot_product_attention`（xformers 可用时走 `memory_efficient_attention`），二者均无法导出为 ONNX。这里强制 `XFORMERS_DISABLED=1` 并 monkey-patch `Attention.forward` 为手写 `q·kᵀ/√d -> softmax -> ·v`（eval 下数值等价）。
- **wrapper**：原始 forward 用 `def forward(self, *args, ...)`，dynamo 导出器会因 variadic 结构报错，因此包一层显式签名 `def forward(self, x)` 的 wrapper；patch token 均值用 `MatMul` 实现（避免 opset18 的 `ReduceMean`，TPU-MLIR 的 OnnxOpt 不支持 axes 作为输入的 ReduceMean）。
- **导出后自检**：`onnx.checker` + onnxsim + 与 PyTorch 输出的 cosine 相似度（`> 0.9999`）+ 图中无 `Resize`。

`--out_onnx` 是 onnxsim 简化后的成品（权重内嵌、单文件），编译脚本只认这一个文件；未简化的中间产物默认写到 `<out_onnx 去后缀>_raw.onnx` 并在自检后删除（需要保留时加 `--keep_raw`）。

### 3.1 线性头导出

推理脚本不上 torch，故把官方线性头转成 numpy 格式（`datasets/dinov2_vitb14_reg4_linear_head.npz` 供 Python 例程，`_bin` 供 C++ 例程）：

```bash
python3 scripts/export_linear_head.py \
    --weights models/torch/dinov2_vitb14_reg4_linear_head.pth \
    --out datasets/dinov2_vitb14_reg4_linear_head.npz \
    --bin_out datasets/dinov2_vitb14_reg4_linear_head.bin
```

## 4. 数据集准备

- **ImageNet-1k 验证子集**（精度/性能测试用）：从 val zip 每类取 1 张，共 1000 张平铺到 `img/`，并生成 `label.txt`（`文件名\t类别下标`，对齐 ResNet 例程与官方 eval 约定）。要求 zip 内按 synset 文件夹组织（如 `n01440764/xxx.JPEG`）：官方原始 `ILSVRC2012_img_val.tar` 是扁平结构且无标签，需先用 devkit 的 ground-truth + synsets 整理成上述结构。

  ```bash
  python3 scripts/gen_val_1k.py \
      --val_zip <path/to/imagenet_validation.zip> \
      --out_dir datasets/imagenet_val_1k
  ```

- **INT8 标定集**（量化用）：把 val 里的图片按文件名排序后**跨类别均匀采样 N 张**（`names[::len(names)//N][:N]`，脚本实测取满 200 张），过官方 eval 前处理（`Resize(256, BICUBIC)` + `CenterCrop(224)`）后存成 tpu_mlir 标定 npz；同时生成 `datasets/test_input/dinov2_1b_in_f32.npz` 供 `model_transform.py`/`model_deploy.py` 做数值校验。

  ```bash
  python3 scripts/gen_cali_data.py \
      --val_zip <path/to/imagenet_validation.zip> \
      --num 200
  ```

## 5. BModel 编译

在 TPU-MLIR 开发容器内进入 `scripts` 目录执行（TPU-MLIR 环境安装见 [Environment_Install_Guide](../../../docs/Environment_Install_Guide.md#1-tpu-mlir环境搭建)；若直接使用 `download.sh` 下发的预编译 BModel，可跳过本节）。本例程目前只适配并验证了 `bm1684x` / `bm1688` / `cv186x` 三种芯片（对应 SE7-32 / SE9-16 / SE9-8）；SE5-16(BM1684) 未做适配。

> **工具链版本**：请使用 TPU-MLIR `v1.30.beta.0-46-gb2e0030-20261029`（whl `tpu_mlir-1.30b0`）或更新版本。该版本包含 BM1686（BM1688/CV186X）dequant/requant 连续 staging 修复（上游 commit `bb0cb8e`），修复在编译期烤进 bmodel，盒子无需刷固件；旧版 TPU-MLIR 编译的 INT8 bmodel 在 BM1688/CV186X 上仍会明显慢于 FP16，需要重新编译才能获得新性能。

```bash
cd scripts
./gen_fp32bmodel_mlir.sh bm1684x   # 生成 models/BM1684X/dinov2_vitb14_reg_fp32_1b.bmodel
./gen_fp16bmodel_mlir.sh bm1684x
./gen_int8bmodel_mlir.sh bm1684x
# bm1688 / cv186x 同理，参数换成对应芯片名
```

`bm1688` 会额外生成 `--num_core 2` 的双核版本（`*_1b_2core.bmodel`）；`cv186x` 为单核芯片，只生成单核版本。三颗芯片统一使用同一份 INT8 配方，不再生成 tail16/tail24 精度档。编译产物：

```
models
├── BM1684X
│   ├── dinov2_vitb14_reg_fp32_1b.bmodel
│   ├── dinov2_vitb14_reg_fp16_1b.bmodel
│   └── dinov2_vitb14_reg_int8_1b.bmodel
├── BM1688   （fp32/fp16/int8 另含 *_1b_2core.bmodel）
└── CV186X   （单核）
```

> **精度档选择**：三颗芯片都提供 FP32 / FP16 / INT8 三种档位。INT8 延迟最低，Top-1 比 FP16 低 0.8~1.9 个点；要求精度与 FP16 对齐时直接用 FP16。实测数据见 `README.md` 第 6、7 节与 FAQ 第 4 条。

> **INT8 量化配方（关键）**：DINOv2 的 linear-probe 特征对量化极敏感，**全 INT8 会直接毁掉特征**（SE9-16 实测 Top-1 仅 0.1%，与随机相当）。正确配方是「只把 12 个 block 的 48 个带 bias 线性层（每 block 4 个：qkv 融合投影 `768→2304`、`attn.proj` `768→768`、`mlp.fc1` `768→3072`、`mlp.fc2` `3072→768`）留在 INT8，其余算子（24 个无 bias 的 `QK^T`/`attn@V`、patch 均值、LayerNorm / Softmax / GELU / 残差 / 布局）全部 F16」——`run_calibration.py --sq --cali_method mse`（200 张官方 eval 口径标定图）+ `--quantize_table`（`scripts/gen_qtable.py` 生成的白名单）+ `--matmul_perchannel`。详见 `scripts/gen_int8bmodel_mlir.sh` 顶部注释与 `scripts/gen_qtable.py`。

## 6. 一次性跑通

```bash
# 下载数据集/ONNX/BModel（或按上文自行编译）
bash scripts/download.sh

# 编译（TPU-MLIR 开发机）
bash scripts/auto_test.sh -m compile_mlir -t BM1684X

# PCIe 编译 + 测试
bash scripts/auto_test.sh -m pcie_build -t BM1684X
bash scripts/auto_test.sh -m pcie_test -t BM1684X -p pytest

# SoC 交叉编译（交叉编译机上）+ 设备端测试
bash scripts/auto_test.sh -m soc_build -t BM1688 -s <soc-sdk 路径>
bash scripts/auto_test.sh -m soc_test -t BM1688 -p pytest
```

> **注**：本版本仅在 SoC 上实测；上面的 `pcie_build` / `pcie_test` 命令沿用仓库通用流程，PCIe 卡上未实测。
