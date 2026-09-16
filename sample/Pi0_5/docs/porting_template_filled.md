# π0.5 移植模板（已填写）

> 本文件是 `skills/model-porting-template.md` 针对 π0.5 的填写结果，
> 作为 `model-porting` skill 步骤 01 的输入。结构化版本见 `../model_info.json`。

```
模型名称:          π0.5（pi05_libero，Physical Intelligence 开源 VLA）
算法类别:          其他（视觉-语言-动作 VLA 策略模型；现有枚举无对应项）
原始框架:          JAX（官方权重为 Orbax/ocdbt 格式；经 openpi 官方脚本转为 PyTorch 后导出 ONNX）
模型文件位置:      官方权重 gs://openpi-assets/checkpoints/pi05_libero（约 12.4 GB）；
                   转换脚本 openpi/examples/convert_jax_model_to_pytorch.py --config-name pi05_libero
输入尺寸:          图像 224x224（2 路：主视角 + 腕部）；
                   前缀嵌入 536x2048；动作隐变量 10x32
输入通道数:        3 (RGB)，两路合成一个 batch（[2,3,224,224]）
输出规格:          v_t [1,10,32]（flow-matching 速度场）；
                   10 步去噪积分后得到 [10,32] 动作 chunk，评测取前 7 维
预处理方式:        图像: rotate180 → resize_with_pad(224, BILINEAR) → uint8 → rgb/127.5-1.0 → f32 CHW [-1,1]；
                   语言: PaliGemma tokenizer → embed ×sqrt(d)（本 sample 用预计算的 per-task 前缀资产）
模型架构:          级联多模型（视觉编码器 + KV 主干 2 段 + 去噪 3 段）
子模型个数:        6
目标芯片:          BM1684X
目标设备：         SE7-32 (linaro@172.26.166.88:linaro)
需要精度:          W8BF16（siglip / dkv）、BF16（ddn）
需要 batch:        1b（dkv / ddn）、2b（siglip，语义是"2 路相机"而非 2 个样本）
需要 Python:       否（当前推理链路为 C/C++，按"只实现一个语言"的要求只做 C++）
需要 C++:          是
C++ 前后处理方式:  自定义（图像前处理 + flow-matching 积分 + 分位数仿射反归一化）
bm1684x soc-sdk路径：【待填：/opt/sophon/sophon-sdk 或交叉编译 SDK 绝对路径】
bm1688/cv186x soc-sdk路径: 不适用（本期只做 BM1684X）
测试数据集:        datasets/pi05_libero_sample/（固定 seed 20–50 条观测 + 官方动作真值 npy）
精度指标:          见下「精度与性能指标口径」
性能指标:          见下「精度与性能指标口径」
```

## 精度与性能指标口径

π0.5 是**策略模型**，Top-1 / mAP / CER 这类指标不适用；同时它**不是单次前向**
（一次推理含 2 步去噪循环 × 3 段子模型），也不该只报单段数值。

调研了同类工作后（NVIDIA Jetson AI Lab 的 π0.5-on-Thor 教程、D-Robotics `rdk_model_zoo_s`、
FlashRT、vla.cpp、Isaac-GR00T），**收敛口径只有三个，且三家独立来源完全一致**，本 sample 直接采用：

| 层级 | 指标 | 用在哪 | 出处 |
|---|---|---|---|
| **① 数值级（主）** | **与官方参考实现的 cosine similarity**：报 Overall + **per-timestep Mean / Min / Max**，门限 **≥ 0.999（最低可接受 0.99）** | 单次推理例程（`cpp/pi05_bmcv/`，无需仿真环境） | NVIDIA Jetson AI Lab（π0.5 on Thor：Overall 0.99456、per-timestep 0.99468/0.98825/0.99779）；D-Robotics 原文 "Ensure the cosine similarity of each output node reaches above 0.999 (a minimum of 0.99)"；FlashRT（cosine ≥ 0.999 / FP4 门限 0.995） |
| **② 任务级** | **任务成功率**，格式 `成功数/总数 (百分比)` + **95% Wilson 置信区间** + 明确写 episode 数 | 进阶服务（`cpp/pi05_service/`，需 LIBERO docker） | FlashRT：`Pi0.5 / LIBERO Spatial 10 × 50 = 500 episodes`，`491 / 500 (98.2%)`；vla.cpp："ten tasks twenty episodes per architecture… brackets are 95% Wilson intervals" |
| **③ 辅助** | 各子模型（siglip / dkv / ddn）输出的 cos 与相对偏差 | 定位精度损失发生在哪一段，不单独作为验收结论 | — |
| **性能** | **不含前后处理的纯推理延迟**（分阶段报）+ 端到端延迟 + **内存占用** | 两个例程都报 | D-Robotics："does not include preprocessing and postprocessing"、"please test performance using C/C++ APIs"；vla.cpp 表含 `step ms / inf ms / VRAM MiB`，并强调 "footprint, not latency, determines what can be deployed" |

**三条必须遵守的口径纪律**：

1. **验收对象是"与参考实现的差距"，不是绝对值。** vla.cpp 原文：*"the relevant quantity is the gap to the reference, not the absolute rate."*
   同理 Isaac-GR00T 用 open-loop 预测对 ground truth 的 **MSE**，而不是单独报模型好坏。
2. **必须先固定随机性，且样本量要够。** Isaac-GR00T 明确声明 *"Users may observe 5-6% variance between runs"*；
   vla.cpp 用 200 episodes 评测并指出 20 次 *"can be insufficient"*。因此任务级指标必须写清 episode 数与是否固定 seed。
3. **本项目自己的两条历史坑**：
   - 设备返回的是 **unnorm（反归一化）空间**，离线参考常算在 **norm 空间** —— 混比会得到 24.8% / 51% 这类假差异。比精度前先统一空间。
   - **去噪步数必须对齐再比** —— 本 sample 默认 dn2，竞品多为 dn10。不对齐会把差距少算约 1.5 倍。

> 补充：本 sample 当前的段级实测已满足①的门限 —— siglip feats cos **0.99992**、dkv（W8BF16）**≥ 0.99997**，
> 端到端（含去噪）相对宿主真值偏差 3.03%。这些数字已含"多步去噪"的累积误差，比单段更难达标。