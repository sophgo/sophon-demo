# Skill 9: SoC 设备部署测试

## 目标
将 C++ 程序部署到 SoC 设备 (SE7-32 / BM1684X) 上运行和测试。

## 执行步骤

### 9.1 连接 SoC 设备
```bash
# SSH 连接
ssh linaro@<device_ip>

# 检查 TPU 状态
bm-smi  # 或 source /opt/sophon/libsophon-current/setup.sh 后执行

# 检查磁盘空间
df -h
# /data 分区通常有较大空间，建议使用 /data 存放模型和数据
```

### 9.2 准备文件

#### 方案 A: 交叉编译 + 上传 (推荐)
```bash
# 在 PC 上交叉编译
export SDK=/path/to/sophon-sdk
mkdir build_soc && cd build_soc
cmake .. -DTARGET_ARCH=soc -DSDK=$SDK
make -j4

# 上传到 SoC
scp seaco_paraformer_bmrt.soc linaro@<device>:/data/seaco_paraformer/
scp -r models/ linaro@<device>:/data/seaco_paraformer/
scp audio/*.wav linaro@<device>:/data/seaco_paraformer/audio/
```

#### 方案 B: 源码上传 + 本地编译
```bash
# 上传源码
scp -r cpp/seaco_paraformer_bmrt/*.cpp linaro@<device>:/data/seaco_paraformer/
scp -r cpp/seaco_paraformer_bmrt/*.h linaro@<device>:/data/seaco_paraformer/
scp -r cpp/seaco_paraformer_bmrt/*.hpp linaro@<device>:/data/seaco_paraformer/

# SSH 到设备编译
ssh linaro@<device>
cd /data/seaco_paraformer
mkdir build && cd build
cmake ..
make -j4
```

### 9.3 安装依赖 (SoC)
```bash
# 在 SoC 设备上安装编译/运行依赖
sudo apt-get update
sudo apt-get install -y libsndfile1-dev libarmadillo-dev

# 如果 apt 有问题
sudo apt --fix-broken install
# 或手动下载 .deb 包
sudo dpkg --force-depends -i libsndfile1*.deb
```

### 9.4 上传模型和数据
```bash
# 模型文件较大（~860MB），注意磁盘空间
scp encoder_fp32_10b.bmodel linaro@<device>:/data/seaco_paraformer/models/BM1684X/
scp decoder_fp32_10b.bmodel linaro@<device>:/data/seaco_paraformer/models/BM1684X/
scp predictor_fp32_10b.bmodel linaro@<device>:/data/seaco_paraformer/models/BM1684X/
scp tokens.json am.mvn config.yaml seg_dict linaro@<device>:/data/seaco_paraformer/models/BM1684X/

# 测试音频
scp asr_example.wav linaro@<device>:/data/seaco_paraformer/audio/
```

### 9.5 运行测试
```bash
# 在 SoC 设备上
export LD_LIBRARY_PATH=/opt/sophon/libsophon-current/lib:$LD_LIBRARY_PATH

cd /data/seaco_paraformer
./seaco_paraformer_bmrt.soc \
    --model_dir ./models/BM1684X \
    --input ./audio/asr_example.wav

# 预期输出:
# TEXT: 欢迎大家来到...
# preprocess: X.XXX s
# encoder: X.XXX s
# total: X.XXX s
# RTF: X.XXX
```

### 9.6 多次测试取平均
```bash
for i in 1 2 3 4 5; do
    echo "--- Run $i ---"
    ./seaco_paraformer_bmrt.soc \
        --model_dir ./models/BM1684X \
        --input ./audio/asr_example.wav \
        2>&1 | grep -E "preprocess|encoder|decoder|total|RTF"
done
```

## SoC vs PCIe 差异

| 特性 | PCIe | SoC |
|------|------|-----|
| 架构 | x86_64 | aarch64 |
| TPU 连接 | PCIe 插卡 | 片上集成 |
| 数据拷贝 | bm_memcpy_d2s (PCIe 传输) | bm_mem_mmap (zero-copy) |
| CPU 性能 | 强 (x86) | 弱 (ARM A53/A55) |
| 预处理耗时 | 1.3s | 4-5s |
| RTF | 0.32 | 0.94-1.23 |
| 编译器 | gcc (native) | gcc (native) 或 aarch64-gcc (cross) |

## 常见问题

### 磁盘空间不足
```bash
# 检查大文件
du -sh /*/
# 清理 pip 缓存
rm -rf ~/.cache/pip/*
# 使用 /data 分区
df -h /data  # 通常有 30GB+ 空间
```

### 库加载失败
```bash
# 检查依赖
ldd seaco_paraformer_bmrt.soc
# 设置库路径
export LD_LIBRARY_PATH=/opt/sophon/libsophon-current/lib:/usr/lib:$LD_LIBRARY_PATH
```

### 权限问题
```bash
# 如果 /data 需要 sudo
sudo mkdir -p /data/seaco_paraformer
sudo chown -R linaro:linaro /data/seaco_paraformer
```

### bmrt 加载失败
```bash
# 检查 libsophon 版本
cat /opt/sophon/libsophon-current/VERSION
# bmodel 的 SDK 版本需要与当前 libsophon 兼容
```

## 检查清单

- [ ] SSH 连接正常
- [ ] bm-smi 显示 TPU 设备
- [ ] 模型文件上传完整（所有文件非 0 字节）
- [ ] 库依赖满足 (ldd 检查)
- [ ] C++ 程序运行正常
- [ ] 5 次测试完成
- [ ] 结果记录到 README
