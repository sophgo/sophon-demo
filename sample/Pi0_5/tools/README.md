# tools

| 脚本 | 用途 | 状态 |
|---|---|---|
| `compare_acc.py` | 动作轨迹与官方参考的数值对照（cos / 相对 L2 / MAE / per-timestep cos），按主 README §6 的门限给 OK/WARN/FAIL | ✅ 已实现并自测 |
| [`export/`](./export/README.md) | 官方权重 → 6 个 ONNX 的分段导出与图修补脚本 | ✅ 已迁入（路径已参数化，未在干净环境重跑） |
| [`make_sample_dataset.py`](./make_sample_dataset.py) | 采集固定 seed 观测并配对官方动作真值 | ✅ 已实现（需 LIBERO 环境，未端到端验证） |

## compare_acc.py

```bash
# 单条
python3 compare_acc.py --pred results/t00_init0.npy --gt datasets/pi05_libero_sample/actions_gt/t00_init0.npy

# 整个目录（按文件名配对）
python3 compare_acc.py --pred_dir results/ --gt_dir datasets/pi05_libero_sample/actions_gt/
```

判定门限（与 NVIDIA Jetson AI Lab / D-Robotics / FlashRT 口径一致）：

| cos | 结论 |
|---|---|
| ≥ 0.999（默认 `--warn`） | OK |
| ≥ 0.99（默认 `--floor`） | WARN |
| < 0.99 | FAIL（退出码 1） |

per-timestep 指 10 步去噪积分后动作轨迹的逐步 cos —— 多步累积误差比整体更难达标，
只报 Overall 会掩盖尾部步的退化。
