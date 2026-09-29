# GLC 真熵编码与编解码分离

本文说明 GLC 例程新增的**真熵编码**路径，以及它与原有「理论值路径」的区别。

## 一、原有 bpp 不是真实码流长度

原有例程与 GLC 官方脚本打印的 bpp 都是**信息论下界**，不是实际要传输的字节数：

- 官方 `src/models/common_model.py:36` 的 `get_y_gaussian_bits()` 只把 y 的高斯概率按
  `-log2(P(y-0.5 < Y < y+0.5))` 求和；
- 现有例程 `python/glc_infer.py` 的等价实现同口径，README 也写明是「按高斯熵模型就地估算」。

这条路径下编码端输出的不是码流，而是原始张量 `y`。真实部署场景（压缩端与还原端分离、
码流过无线信道）不能用它——还原端手里只有码流，没有 `y`。

## 二、真熵编码路径的模型边界必须重切

关键在于**解码端的输入变了**。原有 `glc_decoder.bmodel` 的输入是原始未量化的 `y`；
真熵编码的解码端要从码流里把量化后的 `y` 解出来，所以它看不到 `y`。

GLC 的四段空间先验是自回归结构（`common_model.py:179` `forward_four_part_prior`）：
第 N 段的 scales/means 依赖前 N-1 段重建出的 `y_hat_so_far`。因此**两端都要能逐段算
scales/means**，先验子图两端各一套。

切分结果：

    编码端                                解码端
    glc_encoder      -> y, z             解 z_payload -> z_hat
    码本 argmin       -> z_indices         glc_prior_init -> 初始 scales/means, common
    glc_prior_init   -> 初始 scales/means glc_stage_1..3 -> 逐段 scales/means
    glc_stage_1..3   -> 逐段 scales       rANS 解码       -> 重建 y_hat_so_far
    rANS 编码         -> 四段码流          glc_tail       -> x_hat

码流协议（与客户参考实现 `test_image_complement2.py` 的 `HEADER` 逐字段对齐）：

    header(76B) + z_payload(14-bit 定长) + 四段 y rANS 码流

## 三、rANS 走纯 C ABI，设备端不需要 compressai

设备端只装 `numpy`，不装 compressai、不装 torch：

- `tools/rans/glc_rans.c` 是纯 C 实现，语义与 compressai 1.2.8 的
  `cpp_exts/rans/rans_interface.cpp` 逐位对齐；`ctypes` 调用，绕开 pybind11
  与设备端 Python 版本耦合。
- CDF 表只依赖固定的 64 档 `scale_table`（`SCALE_MIN=0.11`, `SCALE_MAX=256.0`），
  与图像无关。在主机用 compressai 预算一次存成资产，设备端只做 `build_indexes`
  （64 次比较的查表），因此完全避开 scipy 依赖。

两个容易踩的值，取错会导致码流与官方不一致、两端互相解不出来：

- `precision = 16`，**`bypass_precision = 4`（不是 8）**；
- `Rans64EncPutBits` / `Rans64DecGetBits` 这两个 bypass 辅助函数在
  `rans_interface.cpp` 的匿名命名空间里，**不在** `rans64.h`，需要照搬。

编码端还有一处容易漏：官方是 `value = symbols[i] - offsets[cdf_idx]`，
解码端最后 `output[i] = value + offset`。漏掉 offset 时，在 offset 恰好为 0 的
CDF 行上测试会「看起来通过」，换一组参数就错。

## 四、怎么跑

两个可独立部署的进程，完整步骤与预期输出见
[Python例程 第 4 节](../python/README.md)。摘要：

```bash
bash tools/rans/build.sh                       # 编译 rANS（设备上原生编 aarch64）
python3 python/glc_encode.py --precision bf16 \
    --input datasets/kodak24/kodim01.png --out /tmp/out.glc    # 压缩端
python3 python/glc_decode.py --precision bf16 \
    --input /tmp/out.glc --out /tmp/recon.png                  # 还原端
```

**两端必须用同一个 `--precision` 档案**——同样的字面值，不是「同样的精度水平」。
`bf16` 配 `mixed`、`bf16` 配 `fp32` 这类混搭，还原端都会报错，原因见
[Python例程 第 4.2 节](../python/README.md)。

同芯片内闭环用 `bf16` 即可；**跨芯片（SE7-32 ↔ SE9-16）必须两端都用 `mixed`**，
用 `bf16` 会互不可解，见本文第九节。

主机端（需要 `vendor/` 下的 compressai 最小子集，见 `tools/rans/build.sh` 注释）：

```bash
PYTHONPATH=tools/vendor python3 tools/tests/test_rans_parity.py       # rANS 与官方字节级一致
PYTHONPATH=tools/vendor:. python3 tools/tests/test_split_equivalence.py  # 边界重切语义不变
```

`test_split_equivalence.py` 用随机权重即可判定切分是否正确：只要两边数学路径一致，
无论权重取什么值，输出都必须逐位相同。这样能把「切分切错了」与「权重没下对」
两种失败区分开。

`tools/rans/build.sh` 会同时编 x86_64 与 aarch64 两个版本，并用同一份 `tools/rans/rans_selftest.c`
在两个架构上实跑（aarch64 走 `qemu-aarch64`），比对两端码流必须逐字节一致——
证明交叉编译产物真的能跑，而不只是能编译。

## 五、ONNX 子图与 bmodel

### 5.1 直接下载（推荐）

`scripts/download.sh` 会把真熵编码需要的全部 bmodel 与静态资产拉到
`models/BM1684X/` 与 `models/BM1688/`（两平台内容相同，按实际芯片选目录），
无需自行编译：

- 子图 bmodel：`glc_prior_init`、`glc_stage_1..3`、`glc_tail`
  （`glc_encoder` 与基础路径共用），**BF16 与 FP32 两套都含**
- 静态资产：`entropy_assets.npz`（CDF 表 + scale_table）、`codebook.npy`、
  `q_enc.npy`、`q_dec.npy`

下载后三个精度档案都开箱可用：

| 档案 | 需要的子图 | 下载包已含 |
|------|-----------|-----------|
| `bf16` | 全部 BF16 | 是 |
| `mixed`（跨芯片推荐） | 熵模型子图 FP32 + `encoder`/`tail` BF16 | 是 |
| `fp32` | 全部 FP32 | 是 |

`mixed` 与 `fp32` 所需的 FP32 子图已随包下发，**不需要自行编译**
（早期版本的包只含 BF16，需按 5.2 补编，现已补齐）。

### 5.2 自行编译（需要改精度或改 shape 时）

熵编码静态资产（CDF 表 + scale_table，设备端只读）：

```bash
PYTHONPATH=tools/vendor:. python3 tools/make_entropy_assets.py \
    --out models/BM1684X/entropy_assets.npz
```

导出新子图（需要 tpu-mlir 那个 venv 里的 torch）：

```bash
scripts/prepare_mlir_env.sh .venv-mlir <tpu_mlir-*.whl>
PYTHONPATH=tools/vendor:. .venv-mlir/bin/python tools/export_subgraphs.py \
    --ckpt models/GLC_image.pth.tar
```

编译 bmodel（默认 BF16，`--fp32` 可切 FP32 做逐步对账）：

```bash
PATH=$PWD/.venv-mlir/bin:$PATH scripts/gen_entropy_bmodel.sh bm1684x
```

> 自编译后务必用 `tools/tests/test_bmodel_accuracy.py` 对一遍新产物。TPU-MLIR 在
> 本模型上不是逐位可复现的（同一份 ONNX 两次编译出的 bmodel md5 不同），且观察到
> 过 `glc_stage_2` 出现两个输出张量指向同一块内存、`scales` 与 `means` 取到相同值
> 的情况；重编一次即恢复。
>
> 注意该测试目前只覆盖 `glc_prior_init`（`glc_stage_1..3` 与 `glc_tail` 尚未纳入）。
> 要确认 stage 子图，可用 `tools/make_bmodel_probes.py` + 比对同一输入在两芯片上
> 的输出是否逐位相同——跨芯片部署时这一步是必需的。

tpu-mlir 官方 wheel 在较新 glibc 主机上有三个坑，`prepare_mlir_env.sh` 已处理：
wheel 自带的 `libc.so.6`/`libm.so.6` 会因为 `tpu_mlir/__init__.py` 把它插到
`LD_LIBRARY_PATH` 最前面而破坏 Python 解释器本身；缺 `psutil`/`flatbuffers`/
`onnxruntime`/`onnxsim` 声明；依赖 numpy 1.24 与 onnx 1.16 的特定版本。
因此必须用独立 venv，不要装进主环境。

## 六、实测结果

主机 x86_64，5 张 Kodak 512x768，q_index=0，真实权重：

- 真实 bpp **0.02679**，理论 bpp（现有例程口径）0.02461，**差值 +8.96%**
- 构成：y 0.02183 + z 0.00342 + header 0.00155
- 码流总长 1176 ~ 1520 字节；往返 y_hat 逐位一致 5/5

真实 bpp 高于理论值的归因（三个来源，量级与预期一致）：

- **header 固定 76 字节 = 0.00155 bpp**，占差值的 71%
- **z 走 14-bit 定长打包**，而理论值按 log2(16384)=14 bit 算，两者口径接近；
  z_payload 168 字节是四段之外的真实开销
- **rANS 状态开销与 CDF 量化到 16 bit**：残余部分，约 0.0006 bpp

rANS 串行耗时（393216 个符号，主机 5 次中位数）：编码 14.6 ms、解码 5.5 ms，
合计约 20 ms，相对现有 BF16 端到端 688 ms 基线可忽略。
SE7 的 aarch64 主频低于主机，绝对耗时需在真机复测。

## 七、SE7 真机实测

SE7（BM1684X，Ubuntu 20.04，aarch64，8 核）真机数字，均为实测：

### 7.1 交叉编译产物可用

`tools/rans/build.sh` 产出的 aarch64 `.so` 在真机跑 `rans_selftest`，输出
`01 00 19 5c 00 00 02 00 13 48 2e 7c 42 11 9f 00`，与 x86_64、qemu-aarch64
三者逐字节一致。交叉编译不是「能编译」，是能跑且结果相同。

### 7.2 真机 rANS 串行耗时

393216 符号（5 次中位数）：

| 项 | 真机 aarch64 | 主机 x86_64 |
|----|--------------|-------------|
| 编码 | 41.4 ms | 14.6 ms |
| 解码 | 15.2 ms | 5.5 ms |
| 合计 | 56.5 ms | 20 ms |

约为主机 x86_64 的 3 倍，与主频差异吻合。合计占端到端 1027 ms 的 **5.5%**，
**不构成瓶颈**——rANS 串行开销至此排除。

### 7.3 真机端到端

kodim01（512x768，q0），全程 bmodel，编解码之间只传 `.glc` 码流文件：

- 码流 **1416 字节**，真实 bpp **0.028809**
  （y 0.023844 + z 0.003418 + header 0.001546）
- 编码端 502.5 ms，解码端 524.3 ms
- **两端 y_hat 逐位相同**（`np.array_equal`）——编解码分离成立
- 重建图 PSNR **18.98 dB**（主机同参数 18.95 dB，差 0.03 dB）

真机与主机码流长度差 4 字节（1416 vs 1420），源于 bmodel 是 BF16：
量化后的 y 符号个别越界，走上 bypass 逃逸路径。bpp 差 0.28%。

耗时分解（真机，单张）：

| 编码端 | ms | 解码端 | ms |
|--------|----|--------|----|
| encoder bmodel | 234.6 | tail bmodel（dec+generator） | 381.6 |
| 码本 argmin | 66.9 | prior_init + 3x stage | 42.6 |
| rANS 编码 | 41.2 | rANS 解码 | 15.2 |
| **小计** | **343** | **小计** | **440** |
| Python/码本查表等零散开销 | ~160 | 同上（含码本查表） | ~84 |
| **实测总计** | **502.5** | **实测总计** | **524.3** |

### 7.4 bmodel 数值精度

`glc_prior_init`（5 输出子图）在真机跑出的误差与主机 cmodel 完全一致
（q_enc 9.06e-3、q_dec 7.14e-3、scales 5.86e-3、means 8.63e-3、common 1.28e-2，
均按参考量级归一），符合 BF16 尾数 2^-8 的预期。

### 7.5 与客户端的对齐情况

客户样例 test_image_complement2.py 里给了全部影响码流的参数，没给 compressai 版本号。
我们据此做了对齐验证（tools/tests/test_param_alignment.py）：

- **参数已对齐**：按样例参数（0.11 / 256.0 / 64 档 / tail_mass 1e-9）构造的 CDF
  与我们下发的资产逐元素相同；header 常量（GLC2 / v2 / >4sBBH17I / 14-bit）也一致。
- **参数确实敏感**：单独改动上述任一参数，码流都会变（已实测），所以对齐是必要的。
- **版本不敏感**：compressai 1.1.9 / 1.2.0 / 1.2.4 / 1.2.7 / 1.2.8 五个版本，
  相同参数下 CDF 与码流 md5 完全相同。

**仍待客户确认**：若客户还原端不是用这套参数（例如自训模型用了不同的 scale_table
边界），两端仍对不上。建议直接交换下发的 entropy_assets.npz，而不是各自重算。

## 八、SE9 真机实测

SE9-16（BM1688，Ubuntu 22.04，aarch64，8 核，2GB 内存）真机数字，均为实测。
BM1688 与 BM1684X 的 bmodel 不可互换，需用 `bm1688` 重新编译（见第五节）。

### 8.1 交叉编译产物可用

`tools/rans/build.sh` 在 SE9 上原生编译（gcc 11.4，符合 SE9 的 g++ 11.x 要求），
`rans_selftest` 跑通，落盘 `libglc_rans_aarch64.so`。

### 8.2 真熵编码往返一致

编解码在**两个独立进程**里跑，两端 `y_hat` 逐位相同：

```
y_hat 形状 (1, 256, 32, 48)
逐位相同: True
最大绝对差: 0.0
```

### 8.3 实测数字

精度（Kodak24 中尺寸匹配的 18 张，512x768）：

| 档位 | SE7-32 bpp / PSNR / SSIM | SE9-16 bpp / PSNR / SSIM |
|------|--------------------------|--------------------------|
| q0 | 0.0244 / 21.01 / 0.4952 | 0.0244 / 21.01 / 0.4953 |
| q1 | 0.0282 / 21.41 / 0.5112 | 0.0282 / 21.40 / 0.5106 |
| q2 | 0.0328 / 21.79 / 0.5251 | 0.0328 / 21.78 / 0.5244 |
| q3 | 0.0370 / 21.99 / 0.5343 | 0.0370 / 21.98 / 0.5337 |

单张端到端（kodim01，q0）：

| 项 | SE7-32（BM1684X） | SE9-16（BM1688） |
|----|-------------------|------------------|
| 基础路径 bpp / PSNR | 0.0268 / 18.93 dB | 0.0267 / 19.00 dB |
| 基础路径端到端 | 683.7 ms | 5363.6 ms |
| 真熵编码码流 | 1416 字节 | 1412 字节 |
| 真实 bpp | 0.028809 | 0.028727 |
| 编码 / 解码耗时 | 510 / 551 ms | 4731 / 1960 ms |

> 上表的编码/解码耗时是 `glc_encode.py` / `glc_decode.py` 自报的**纯计算段**（不含
> 进程启动、加载 bmodel 与读图）。第九节的 mixed 对比表用的是 `cross_chip_run.py`，
> 把建引擎与预处理一并计入，同一台 SE7 上同口径实测 1330 ms（bf16）/ 1452 ms（mixed），
> 与本节 510 ms 不是同一口径，两组数字不可直接相减。

> 两平台的 bpp / PSNR 非常接近（差 0.1% / 0.07 dB），说明量化与熵编码口径一致；
> 码流长度差 4 字节（1412 vs 1416）来自 BF16 下个别 y 符号越界触发 rANS bypass 逃逸，
> 与 BM1684X 上观察到的跨精度差异同源。

下面这段的倍数与分项耗时取自主 README 第 7.2 节的 **18 张平均**（不是上表的单张）：

SE9 端到端为 SE7 的 **7.5 倍**，但拆开看两部分差异很大：

- **bmodel 部分只慢约 4 倍**（编码 232 → 980 ms，解码 395 → 1754 ms），与 BM1688 算力低于 BM1684X 相符；
- **CPU 码本查找慢 40 倍**（61 → 2441 ms），是端到端延迟的最大单项，占 SE9 总耗时的 47%（SE7 上只占 9%）。

码本查找是纯 numpy 的 16384 类 argmin，SE9 的 CPU 主频与内存带宽都低于 SE7，在 2GB 内存的设备上这一项会成为主要优化方向。

### 8.4 与 SE7 的适配差异

代码无需改动——`--models` 参数指定芯片目录即可。差异只在资产侧：

- 编译时传 `bm1688` 而非 `bm1684x`，产物落到 `models/BM1688/`
- 两个平台的 bmodel **不可互换**，跨平台部署需各自带对应目录

## 九、跨芯片混合精度（mixed）

### 9.1 为什么不能整体 BF16

跨芯片部署时，SE7-32（BM1684X）与 SE9-16（BM1688）编出的 BF16 码流**互不可解**：
两芯片的 BF16 先验子图输出并不逐位相同，个别 scale 落到不同 CDF 档位，rANS 随即失步。

逐子图实测（同一份真实链路探针，两芯片分别跑 bf16 与 fp32，比对输出是否逐位相同）：

| 子图 | BF16 跨芯片 | FP32 跨芯片 | 是否影响互解 |
|------|------------|------------|-------------|
| `glc_prior_init` | 不同 | **逐位相同** | 是（scales/means/common/q_dec） |
| `glc_stage_1` | 不同 | **逐位相同** | 是 |
| `glc_stage_2` | 不同 | **逐位相同** | 是 |
| `glc_stage_3` | 不同 | **逐位相同** | 是 |
| `glc_encoder` | 不同 | — | 否（符号由熵编码保证） |
| `glc_tail` | 不同 | — | 否（只影响重建图外观） |

关键点：只有「决定 CDF 行选择与 y_hat 重建」的四个子图必须逐位一致。`encoder` 输出的
y 只是被编码成符号，解码端照着符号重建；`tail` 只决定最终像素。这两个恰好是耗时大头，
留 BF16 即可。

### 9.2 切分与实测

`--precision mixed` = `prior_init` / `stage_1..3` 用 FP32，`encoder` / `tail` 用 BF16。
两芯片双向实测：

```
SE7-32 编码 -> SE9-16 解码   码流 1416 字节  两端 y_hat sha256 相同  互解正常
SE9-16 编码 -> SE7-32 解码   码流 1416 字节  两端 y_hat sha256 相同  互解正常
```

对照（未修复前的行为）：

```
SE7-32 BF16 编码 -> SE9-16 BF16 解码
  ValueError: 解码出的 y_hat 含 NaN/Inf：码流与当前 --precision 模型不匹配
```

耗时与质量（kodim01，q0）：

| 档案 | SE7 编码 | SE9 编码 | 码流 | PSNR |
|------|---------|---------|------|------|
| `bf16` | 1312 ms | 5302 ms | 1412~1416 字节 | 18.98 dB |
| `mixed` | 1438 ms | 5725 ms | 1416 字节 | 18.95 dB |
| `fp32` | — | 9579 ms | 1420 字节 | — |

mixed 只比纯 BF16 慢约 10%，重建质量不变，却消除了跨芯片不可解问题。
**跨芯片部署用 `mixed`；同芯片内闭环继续用 `bf16`。**

> 上表的编码耗时用 `tools/cross_chip_run.py` 实测，**含进程启动、加载 bmodel 与
> 读图**，因此比 8.3 节 `glc_encode.py` 自报的纯计算段（SE7 510 ms）大一倍多。
> 两处口径不同，只用于**同一张表内横向对比档案**，不要跨节相减。

### 9.3 W4BF16 尝试（结论：不适用）

按要求进一步试了 `W4BF16` 混合精度，实测**不可用**，三条依据：

1. **数值上与 BF16 完全等价（静默退化）**：W4BF16 编译的 `glc_prior_init` 与 BF16 版
   在真机上输出**逐位相同**（5 个输出张量 max|d| 全为 0）。即权重并未真正压到 4 bit，
   与 FAQ Q9 记录的现象一致。
2. **产物大小不变**：BM1684X 24207360 字节、BM1688 23646208 字节，与 BF16 版
   **完全同尺寸**，没有体积收益。
3. **仍不解决跨芯片问题**：W4BF16 两芯片输出仍不逐位相同（max|d| 0.0078~0.0225），
   rANS 依然会失步。

结论：W4BF16 在本模型上既无精度收益也无体积收益，跨芯片一致性仍依赖 mixed 档案中
的 FP32 熵模型子图。

### 9.4 怎么复现这套结论

三个工具配套使用，缺一不可：

```bash
# 1) 主机侧：用真实图像走一遍编码链，导出每个子图的真实输入
PYTHONPATH=tools/vendor:python:. .venv-mlir/bin/python tools/make_mix_probes.py
#    -> tools/mix_probes.npz（z_hat / stage{N}_y / stage{N}_common / tail_y / x ...）

# 2) 每台设备上：跑指定子图、指定精度，把输出落盘
python3 tools/subgraph_probe.py <models_dir> <subgraph> <bf16|fp32> \
        tools/mix_probes.npz /tmp/out.npz
#    同一子图同一精度在两台机器上各跑一次，比对输出是否逐位相同
#    —— 这是判定「该子图能不能留 BF16」的唯一依据

# 3) 每台设备上：完整编解码往返（验证互解）
python3 tools/cross_chip_run.py <models_dir> <mixed|bf16|fp32> encode kodim01.png /tmp/x
python3 tools/cross_chip_run.py <models_dir> <mixed|bf16|fp32> decode /tmp/x.glc /tmp/y
#    两端的 y_hat sha256 相同即互解成立
```

`tools/mix_probes.npz` 不入库（含真实权重派生的中间量），由第 1 步现场生成。
子图输入必须走真实图像链路，不能用 randn：这些子图含 GroupNorm，非真实分布的输入
会产生 1e10 量级激活，测不出真实差异。

> **编译产物必须逐张量验证**：实测遇到过 BM1688 的 `glc_stage_2_fp32.bmodel` 两个输出
> 张量别名到同一块内存（`scales` 与 `means` 取到完全相同的值）。该产物即使「编译成功」
> 也不能用，重编后才恢复正常。用 `tools/subgraph_probe.py` 在两台机器上跑同一子图、
> 比对输出，是发现这类问题的可靠手段。
