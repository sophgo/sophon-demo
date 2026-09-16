# Pi0_5 常见问题解答

## 目录

* [1 相机与输入](#1-相机与输入)
* [2 精度相关](#2-精度相关)
* [3 运行与部署](#3-运行与部署)
* [4 编译与模型](#4-编译与模型)

通用问题（环境安装、模型导出、量化等）请参考[仓库通用 FAQ](../../docs/FAQ.md)。

---

## 1 相机与输入

### 1.1 为什么本 sample 只用两路相机？第三路去哪了？

pi0.5 的 PaliGemma **固定吃 3 个图像槽**，顺序与官方命名是
`base_0_rgb` / `left_wrist_0_rgb` / `right_wrist_0_rgb`。

第三路不是本 sample 砍掉的，而是**官方 LIBERO 策略自己就把它补零并屏蔽**：

```python
# openpi/src/openpi/policies/libero_policy.py
"image": {
    "base_0_rgb": base_image,
    "left_wrist_0_rgb": wrist_image,
    "right_wrist_0_rgb": np.zeros_like(base_image),     # 补零
},
"image_mask": {
    ...
    "right_wrist_0_rgb": np.False_,                     # 且掩码置 False
},
```

它的 256 个 token 既不参与输出、也不会被别的 token 注意到，属于结构性死 token，
所以编译时没有为它留位置 —— 完整前缀 968 token 里因此去掉了 436 个
（第三路 256 + 语言 padding 180），得到 **536**。删除不改变数学结果。

### 1.2 那要接真第三路相机怎么办？

接口已经按 3 路预留，不是写死两路。`cpp/pi05_bmcv/pi05.h` 里是一条推导链：

```cpp
static constexpr int kNumViews = 2;                    // ← 接第三路相机时改这里
static constexpr int kTokensPerView = 256;
static constexpr int kVisTokens = kNumViews * kTokensPerView;
static constexpr int kLangSlots = 24;
static constexpr int kPrefixLen = kVisTokens + kLangSlots;   // kNumViews=2 -> 536
```

`infer()` 已经收第三路参数（传空则补零，与官方行为一致），命令行对应 `--cam3 FILE`。
换成 3 路需要同步做三件事：

1. 把 `kNumViews` 改成 3；
2. 重新导出并编译 siglip（输入 `[3,3,224,224]`，即 `scripts/gen_siglip_bmodel_mlir.sh` 里的 `--input_shapes`）；
3. 重新生成前缀资产并重编 dkv 两段 —— 前缀长度从 536 变成 792，
   `gen_kvseg_pertask.py` / `gen_assets_536.py` 里的 `PL` 要跟着改。

**在没做完这三步之前传 `--cam3` 会直接报错退出**，而不是被静默忽略：
悄悄丢掉一路输入会让输出看起来完全正常，但其实是错的。

### 1.3 观测为什么存 `.npy` 而不是 PNG？

为了让例程只依赖 libsophon 与一个 C++ 编译器。仓库里其它 sample 常用 OpenCV，
但目标设备通常只装了 sophon-opencv 的**运行库、没有开发头文件**，会直接卡住编译。

观测的约定：uint8、`[224,224,3]`、RGB、行主序，且已经是
`rotate180 → resize_with_pad(224)` 之后的（与官方 client 一致）；例程只做
`rgb/127.5 - 1.0 → CHW [-1,1]`。换成真实相机帧时，这些预处理在 client 侧完成。

---

## 2 精度相关

### 2.1 为什么精度对比要连噪声一起发？

去噪是从噪声开始的，**噪声本身就是输入的一部分**。两边各抽各的噪声时，
比出来的差异里混着采样器差异，而不是移植误差。

所以数据集里每个 case 都带一份 `noise/tXX_initY.npy`，设备端用 `--noise` 喂入，
两边就是同一个输入。实测：用不同噪声对比会把 rel L2 从 0.60% 抬到 2.12%。

同理，两边的**去噪步数必须一致**（本 sample 默认 2 步，官方评测默认 10 步），
否则比的是采样器而不是移植。

### 2.2 初始噪声为什么必须是 N(0,1)？

官方采样器抽标准正态。抽成均匀分布 `[-1,1)` 不会报错、也能跑出动作，
但收敛到的是另一个动作 chunk —— 拿它去和官方比，差异会被误读成移植误差。

### 2.3 `action_unnorm.npz` 用错会怎样？

官方用**分位数仿射** `x*(q99-q01)/2 + (q01+q99)/2`，不是 `x*std + mean`。
用错会让前 6 维尺度差 2.1~3.5 倍，症状是**闭环步数翻倍、gripper 该松不松**，
极易误判为"模型精度不够"。

### 2.4 比精度前为什么要统一数值空间？

设备返回的是 unnorm 空间，离线参考常算在 norm 空间，混比会得到 24.8% 这类假差异。

---

## 3 运行与部署

### 3.1 为什么结果全对，进程退出却是 139（SIGSEGV）？

dkv 两段产出的 36 个 KV **就是它们各自的输出缓冲**（每个网络在 `net_bufs_init` 里
分配一次、此后反复复用），例程只是别名引用、不拥有它们。若在析构里连同 KV 一起
`bm_free_device_mem`，同一批设备地址会被还两次。

症状很隐蔽：**动作数值全部正确，进程在退出时 SIGSEGV**。在 shell 里
`if`/`&&` 会把它判成失败，`auto_test.sh` 会误报 SOME TESTS FAILED。

### 3.2 强杀持有 NPU 的进程会怎样？

会把 TPU 驱动卡死，只能重启设备恢复。服务切换一律走 `systemctl`
（它发 SIGTERM，进程会优雅释放 NPU 句柄）。

### 3.3 非交互执行时找不到 `bmrt_test`？

`bmrt_test` 随 libsophon 安装，但设备只在**登录 shell** 里把它的 `bin/` 加进 PATH
（`/etc/profile.d/libsophon-bin-path.sh`）。`ssh host '<cmd>'`、CI 这类非交互执行
要用全路径 `/opt/sophon/libsophon-current/bin/bmrt_test`。

`auto_test.sh` 会依次尝试 PATH、`-s` 指定的 SDK、`/opt/sophon/libsophon-current/bin`，
两种环境都能跑。

### 3.4 喂给设备端加载器的 `.npz` 有什么要求？

必须是 **STORED（非压缩）**。加载器手工遍历 ZIP local header 并把压缩长度当数据长度用，
`np.savez_compressed` 生成的会读出垃圾 —— 症状是 `load fail` 或**模型输出全零**。
请用 `np.savez`。

---

## 4 编译与模型

### 4.1 交叉编译出来的二进制为什么在设备上跑不起来？

能编过 ≠ 能跑起来。宿主的 `aarch64-linux-gnu-g++` 通常比设备新得多
（实测宿主是 GCC 16，SE7 是 GCC 9.4 / glibc 2.31），链接出的二进制会要求设备上
不存在的运行时符号：

```
version `GLIBCXX_3.4.32' not found
version `GLIBC_2.38' not found
```

`scripts/build.sh` 会在编译后读二进制里记录的版本需求并与 SE7 的运行时对比，
不匹配就告警。**省事的办法是在设备上原生编译** —— SE7 自带 GCC 9.x 与 libsophon。

### 4.2 为什么没有 FP32 / FP16 / INT8 与 4batch 档？

- 3.3B 参数全 F32 约 13 GB，超出 SE7 的 16 GB 可用内存；
- 连续动作对激活量化敏感，实测更低比特档偏差远超验收线（W4 两段全压偏差 13.38%）；
- batch 维在这里的语义是"相机路数"，固定为 2，不是可调的 1b/4b。

### 4.3 为什么拆成 6 个子模型？

主干与动作专家都是 18 层，单图在 W8BF16/BF16 下编译会失败（OOM / 编译器崩溃）。
按 9 层与 6 层切分后可正常编译加载。**切分是编译期约束，不改变数学语义。**
