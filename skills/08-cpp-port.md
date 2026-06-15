# Skill 8: C++ 移植 (bmrt SDK)

## 目标
将 Python SAIL 推理代码移植为 C++ bmrt 原生推理程序，实现更高的部署效率。

## 开发前准备

> **重要**: 在编写 C++ 推理代码前，先找到 sophon-demo 中已有的类似 Sample 作为参考模板。
> - 根据算法类别查找最相似的 sample，如果该 sample 有 C++ 实现则直接参考
> - 参考其 `cpp/` 目录的工程结构（CMakeLists.txt、文件命名、目录组织）
> - 参考其 bmrt API 使用方式、动态 shape 处理、SoC zero-copy 实现
> - 保留参考代码的整体结构，根据新模型的输入输出规格修改具体逻辑
> - 建议参考代码: 图像类模型参考 `sample/YOLOv5/cpp/yolov5_bmcv`，ASR 类参考 `sample/WeNet/cpp`，人脸类参考 `sample/RetinaFace/cpp`
> - 算法类别与推荐参考 Sample 的对应关系，见 `model-porting-template.md` 第 11.2 节

## 执行步骤

### 8.1 准备 C++ 工程结构
```
cpp/model_inference_bmrt/
├── CMakeLists.txt        # 构建配置
├── main.cpp              # 入口，命令行参数解析
├── model_inference.h    # 模型类声明
├── model_inference.cpp  # 模型类实现 (推理逻辑)
├── input_data_process.h       # 输入数据预处理声明
├── input_data_process.cpp     # 特征提取 + 归一化 实现
├── utils.hpp             # 工具函数 (TimeStamp 等)
├── json.hpp              # nlohmann::json (header-only)
└── build/                # 构建输出目录
```

### 8.2 CMakeLists.txt 配置
```cmake
project(model_inference_bmrt)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "-O3")

# PCIe 模式
if(${TARGET_ARCH} STREQUAL "pcie")
    find_package(libsophon REQUIRED)
    find_package(数值计算库 REQUIRED)
    pkg_check_modules(DATA_IO REQUIRED data_io)
    # ... 标准链接
endif()

# SoC 模式 (交叉编译)
if(${TARGET_ARCH} STREQUAL "soc")
    set(CMAKE_C_COMPILER aarch64-linux-gnu-gcc)
    set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)
    # SDK 路径从 ${SDK} 环境变量获取
endif()
```

### 8.3 核心 API 移植映射

| Python (SAIL) | C++ (bmrt) |
|---------------|-----------|
| `sail.Engine(bmodel, dev_id, SYSIO)` | `bm_dev_request() + bmrt_create() + bmrt_load_bmodel()` |
| `engine.process(graph, inputs)` | `bm_memcpy_s2d() + bmrt_launch_tensor_ex() + bm_thread_sync()` |
| `numpy array` 输入输出 | `bm_tensor_t` + `bm_device_mem_t` |
| `np.ascontiguousarray(arr)` | `std::vector<float>` 或直接填充设备内存 |

### 8.4 动态 Shape 处理
```cpp
// C++ 动态 shape 推理核心流程：
// 1. 分配 MAX shape 的设备内存 (预分配)
// 2. 拷贝实际数据到设备内存
// 3. bm_set_device_mem 缩小到实际大小
// 4. 设置 tensor.shape.dims 为实际维度
// 5. bmrt_launch_tensor_ex(user_mem=true) 启动推理

struct DynamicDim {
    int tensor_idx;   // 第几个输入 tensor
    int dim_idx;      // 第几个维度
    int actual_val;   // 实际值
};

// 示例: 将 batch=10,T=1000 的动态模型以 batch=1,T=75 运行
std::vector<DynamicDim> dyn_dims = {
    {0, 0, 1},       // 输入0 dim[0] batch=1
    {0, 1, input_len}, // 输入0 dim[1] seq_len=75
    {1, 0, 1},       // 输入1 dim[0] batch=1
};
```

### 8.5 SoC Zero-Copy 优化
```cpp
// SoC 模式：直接 mmap 设备内存，避免拷贝
if (misc_info_.pcie_soc_mode == 1) {  // SoC
    unsigned long long addr;
    bm_mem_mmap_device_mem(handle_, &tensor->device_mem, &addr);
    bm_mem_invalidate_device_mem(handle_, &tensor->device_mem);
    float* data = (float*)addr;  // 直接使用
    // ... 使用 data ...
    bm_mem_unmap_device_mem(handle_, data, size);
} else {  // PCIe
    float* data = new float[count];
    bm_memcpy_d2s_partial(handle_, data, tensor->device_mem, count * sizeof(float));
    // ... 使用 data ...
    delete[] data;
}
```

### 8.6 预处理移植
```cpp
// 特征提取 特征提取: Python 框架预处理工具 → C++ 数值计算库
features_type features(const input_type& input_data, int n_features,
                  int frame_length, int frame_shift, int sample_rate,
                  float dither, float energy_floor,
                  bool param1, bool param2, bool param3);

// 降采样: Python torch.as_strided → C++ 手动实现
auto apply_postprocess(const features_type& inputs, int param1, int param2);

// 归一化
void apply_normalize(features_type& features, const NormConfig& config);
```

### 8.7 构建与运行
```bash
# x86 PCIe 构建
cd cpp/model_inference_bmrt
mkdir build && cd build
cmake .. -DTARGET_ARCH=pcie
make -j4

# SoC 交叉编译
cmake .. -DTARGET_ARCH=soc -DSDK=/path/to/sophon-sdk
make -j4

# 运行
./model_inference_bmrt.pcie --model_dir ../../models/BM1684X --input test_input
```

## 常见移植问题

### 1. 内存管理差异
- Python: 自动 GC
- C++: 手动管理设备内存，注意 bm_free_device_mem / delete[]

### 2. 数据类型映射
- Python int32 → C++ int32_t (注意不是 int)
- Python float32 → C++ float
- numpy (T, D) → C++ row-major: data[t * D + d]

### 3. 动态 Shape 陷阱
- 需要 `bm_set_device_mem` 缩小内存
- tensor shape 要和实际数据大小一致
- 输出 tensor 的 shape 可能和输入不同

### 4. 预处理精度差异
- Python 框架预处理工具中的特征提取有特定实现细节
- C++ 数值计算库 实现可能产生微小数值差异
- 需要对比验证以确认差异不导致识别结果变化

## 检查清单

- [ ] C++ CMakeLists.txt 配置正确
- [ ] PCIe 版本编译通过
- [ ] SoC 版本交叉编译通过
- [ ] C++ 推理结果与 Python 一致
- [ ] 动态 shape 推理正常
- [ ] SoC zero-copy 正常工作
- [ ] 内存无泄漏 (valgrind 检查)
- [ ] 各阶段计时器正常工作
