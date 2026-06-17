---
name: model-porting-08-cpp
description: C++ 移植。将 Python SAIL 推理代码移植为 C++ BMRT/BMCV，对齐精度，支持 PCIe 和 SoC 编译。
---

# 步骤 08：C++ 移植

## 前置

步骤 05 已完成（Python 推理代码可作为参考实现）。

## 提示词

```
将 Python 推理代码移植为 C++（BMRT/BMCV），放在 cpp/[model_name]_bmcv/ 目录：

1. 工程结构：
   cpp/[model_name]_bmcv/
   ├── CMakeLists.txt       # PCIe/SoC 双模式
   ├── main.cpp             # 入口 + argparse
   ├── [model_name].h       # 推理类声明
   ├── [model_name].cpp     # 推理实现
   └── README.md

2. 初始化：
   bm_dev_request + bmrt_create + bmrt_load_bmodel + bmrt_get_network_info

3. 预处理（BMCV，严格按照 Python 的同名链路）：
   - bm_image_create（解码后的图像）
   - bm_image_create_batch（resized 图像 + converto 图像）
   - bmcv_image_vpp_convert（resize，如果输入尺寸不一致）
   - bmcv_image_convert_to（减均值除方差，alpha/beta 见内联知识）
   - bm_image_attach_contiguous_mem（绑定设备连续内存）

4. 推理：bmrt_launch_tensor_ex + bm_thread_sync

5. 后处理：与 Python 逻辑对齐

6. CMakeLists.txt：PCIe (x86) 和 SoC (aarch64) 双模式

⚠️ 关键约束 — cv::Mat 生命周期：
   batch 模式下，cv::bmcv::toBMI 返回的 bm_image 是零拷贝引用 cv::Mat 内存。
   必须新增 std::vector<cv::Mat> batch_mats 成员变量：
   - 每帧 batch_mats.push_back(mat) 保持引用
   - 推理结束后 batch_mats.clear()
   - 否则前 N-1 帧数据被破坏（只最后一帧正确），精度掉到 ~20%
```

## 预期输出

```
cpp/[model_name]_bmcv/
├── CMakeLists.txt
├── main.cpp
├── [model_name].h
├── [model_name].cpp
└── README.md
```

## 内联知识：Python→C++ API 映射

| 操作 | Python (SAIL/BMCV) | C++ (BMRT/BMCV) |
|------|-------------------|------------------|
| 加载模型 | `sail.Engine(bmodel, dev_id, SYSIO)` | `bm_dev_request` + `bmrt_create` + `bmrt_load_bmodel` |
| 创建 batch 图像 | `sail.BMImageArray` | `bm_image_create_batch` |
| resize | `bmcv_image_vpp_convert` | `bmcv_image_vpp_convert` |
| 归一化 | `bmcv_image_convert_to` | `bmcv_image_convert_to` |
| 绑定连续内存 | SDK 自动 | `bm_image_attach_contiguous_mem` |
| 推理 | `engine.process(graph, inputs)` | `bmrt_launch_tensor_ex` + `bm_thread_sync` |
| 获取输出 | `tensor.asnumpy()` | `bm_memcpy_d2s_partial` |
| SoC zero-copy | SDK 自动 | `bm_mem_mmap_device_mem` + `bm_mem_invalidate_device_mem` |

### convert_to alpha/beta 计算

```cpp
// alpha = 1.0 / (255.0 * std) * scale
// beta  = (-mean / std) * scale
// 其中 scale = 1.0（通常）
float alpha = 1.0f / (255.0f * std_val);
float beta  = -mean_val / std_val;
```

## Debug

| 问题 | 排查方向 |
|------|---------|
| C++ 结果与 Python 不一致 | 对齐 mean/std/resize/通道顺序；检查 alpha/beta 计算 |
| batch 模式精度异常低（~20%） | **BUG-005**：缺少 batch_mats，cv::Mat 析构导致数据损坏 |
| 编译链接错误 | 检查 CMakeLists.txt 中 libsophon 路径 |
| 推理结果 NaN | 确认 bm_image_attach_contiguous_mem 之后才做 convert_to |
