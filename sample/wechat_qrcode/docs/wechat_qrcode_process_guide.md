# WeChatQRCode 流程解析

WeChatQRCode（微信二维码检测 + 识别）的推理流程可以概括为：
`解码` -> `detect 前处理` -> `detect 推理` -> `detect 后处理` -> `裁剪 + 填充` -> `超分/缩放` -> `zxing 解码` -> `坐标回投影 + 去重`。

其中 detect（SSD-MobileNet 目标检测）与 sr（超分）两个模型在 TPU 上推理，zxing 二维码解码在 CPU 上完成。下面解释每个阶段分别在代码的哪些位置，以及关键代码的含义。以下文件路径均相对于 `cpp/wechat_qrcode_bmcv/`（“纯 bmrt + bmcv、不依赖 sail”的 C++ 例程）；`cpp/wechat_qrcode_sail/` 与 Python 两个例程只是换了加载/解码方式，算法主线完全一致。

## 1. 解码

对应 `main.cpp`，`wechat_qrcode_bmcv` 有两条读图+处理路径。注意：两者都用 sophon-opencv 的 `cv::imread`，它在 SoC 上默认走 **JPU 硬件解码**（第 3 参 `dev_id` 缺省 0、SoC 上单设备被忽略；软解只是可选的 `IMREAD_RETRY_SOFTDEC` 回退）。两条路径的区别在解码之后：

- **目录/精度模式**（`iters <= 0` 的图像目录迭代）：`cv::imread(path, cv::IMREAD_COLOR | cv::IMREAD_IGNORE_ORIENTATION)` JPU 硬解得到 `cv::Mat`，走**主机算法路径** `detectAndDecode(img)`（`cv::Mat` 路径，detect 前处理在 CPU 上 `cv::resize` + `INTER_CUBIC`）。精度评测（README §5.2）走这条路，是为了与 sail 交付接口 / `opencv.py` 的参考实现保持同一算法口径（`INTER_CUBIC`），使四角点与 BoofCV 真值可比。
- **bench 模式**（`iters > 0`）：`cv::imread(path, cv::IMREAD_COLOR | cv::IMREAD_IGNORE_ORIENTATION, dev_id)` 同样是 JPU 硬解，接着 `cv::bmcv::toBMI(img, &bmimg, true)` **零拷贝附加**为设备 `bm_image`（不落回主机），交给**设备直传**的 `detectAndDecode(handle, bmimg)`（detect resize 用 VPP `LINEAR`）。

```cpp
// 目录/精度模式：JPU 硬解 -> 主机 cv::Mat -> 主机算法路径（INTER_CUBIC）
cv::Mat img = cv::imread(input_path,
                         cv::IMREAD_COLOR | cv::IMREAD_IGNORE_ORIENTATION);
qr->detectAndDecode(img, &points);

// bench 模式：JPU 硬解 -> toBMI 零拷贝 -> 设备直传（VPP LINEAR）
cv::Mat img = cv::imread(input_path,
                         cv::IMREAD_COLOR | cv::IMREAD_IGNORE_ORIENTATION, dev_id);
cv::bmcv::toBMI(img, &bmimg, true);
qr->detectAndDecode(handle, bmimg);
```

统一加 `IMREAD_IGNORE_ORIENTATION` 是为了让两代 SDK（OpenCV 4.1 会自动应用 EXIF 旋转、4.8 不会）下四角点都落在与真值一致的原始像素坐标系（详见主 README §5.1）。Python 的 `wechat_qrcode_opencv.py` 用 `cv2.imdecode(np.fromfile(...), IMREAD_IGNORE_ORIENTATION)`（pip opencv 的 libjpeg 软解）解码成 ndarray，等价于本机目录模式的主机算法路径；`wechat_qrcode_bmcv.py` 用 `sail.Decoder` 解码成 `sail.BMImage`（ffmpeg 硬解，等价于本机 bench 模式的设备直传路径）。

## 2. detect 前处理与推理

对应 `detector/ssd_detector.cpp` 的 `SSDDetector::forward`。

detect 模型是 SSD-MobileNet，输入为 **单通道灰度、uint8**，且 bmodel 由 19 个静态 shape 的子网合并而成（`detect_<W>_<H>`，覆盖横竖不同长宽比）。因此在 `init` 中先枚举全部子网名并解析出各自的 W/H：

```cpp
bmrt_get_network_names(p_bmrt_, &net_names);
// parseGraphName 从 "detect_<W>_<H>" 中解析出 w/h，构造 DetectGraph 列表
```

主机 Mat 路径（`SSDDetector::forward(const cv::Mat &img, ...)`，即目录/精度模式）中，`selectGraph` 依据目标宽高比在 19 个子网里挑出长宽比最接近的一个；随后把整张图按比例 `cv::resize` 到该子网的静态尺寸（`INTER_CUBIC`），再交给 `inferGray` 上传到 TPU 推理：

```cpp
cv::resize(img, input, cv::Size(in_w, in_h), 0, 0, cv::INTER_CUBIC);
const float *out_data = inferGray(p_bmrt_, handle_, input, in_shape, graph_name, out_buf_);
```

`inferGray` 在 `infer.hpp` 中封装：把 resize 后的 8UC1 灰度图放入 `FORMAT_GRAY + DATA_TYPE_EXT_1N_BYTE` 的 `bm_image`，调用 `bm_wrapper.hpp` 的 `bm_inference(bmrt, &input, out_buf.data(), in_shape, net_name)` 完成 attach 输入、`bmrt_launch_tensor`、`bm_thread_sync`、`bm_memcpy_d2s_partial` 全流程。因为 bmodel 在编译时已 `--fuse_preprocess`（mean=0、scale=0.0039216、pixel_format=gray），归一化被烘焙进 TPU 图里，这里无需在 CPU 上再做任何归一化。

设备直传路径（`SSDDetector::forward(bm_handle_t, ...)` 重载，即 bench 模式实际走的路径）不经过上面这条主机 `cv::resize`：整帧已是设备 `bm_image`，在设备上用 VPP 等比例缩放到 `detect_<W>_<H>` 静态尺寸——因 BM1684X/CV186X 的 vpp 不支持 BICUBIC，这里用 `BMCV_INTER_LINEAR`（见 `ssd_detector.cpp`）——再剥离 VPP 行对齐填充后，把 uint8 灰度直接喂给同一个 fused 检测网。两条路径仅 resize 插值（主机 `INTER_CUBIC` vs 设备 `LINEAR`）与数据落点不同，选网、`--fuse_preprocess` 归一化、后处理 `parseOutput` 完全一致。

## 3. detect 后处理

对应 `detector/ssd_detector.cpp` 的 `SSDDetector::parseOutput`。输出为 `(1, 1, 100, 7)` 的 F32 张量，每行 7 个值：`[unused, is_qrcode, confidence, x0, y0, x1, y1]`。这里只保留 `is_qrcode == 1` 且 `confidence > 1e-5` 的候选框（1e-5 是 opencv#2877 的安全阈值），把归一化的角点映射回原图坐标，得到每个二维码的轴对齐矩形（4 个角点）。

## 4. 裁剪 + 填充

对应 `wechat_qrcode.cpp` 的 `WeChatQRCode::Impl::decode` 与 `computeCropBox`。对每个检测框按 `padding_w = padding_h = 0.1`（最小 15 像素）外扩后裁剪出小图（`cv::Rect` + `clone()`），供后续超分/解码使用。外扩是为了给二维码四周留白，避免 finder pattern 贴着裁剪边缘导致解码失败。

## 5. 超分 / 缩放（sr）

对应 `scale/super_scale.cpp` 的 `SuperScale::processImageScale`。裁剪小图需要放大 2 倍再送 zxing 解码，但放大方式按小图尺寸分两条路：

```cpp
if (use_sr && sqrt(width * height) < sr_max_size) {   // sr_max_size == 160
    superResoutionScale(src, dst);                    // TPU 超分
} else {
    cv::resize(src, dst, ..., INTER_CUBIC);           // CPU 双三次
}
```

即裁剪图尺寸 `sqrt(w*h) < 160` 时才走 TPU 超分，否则直接用 `INTER_CUBIC` 放大（`scale < 1` 时用 `INTER_AREA` 缩小）。超分网络输入 224x224、输出 447x447，`superResoutionScale` 中先把小图 resize 到 224x224，`inferGray` 推理后把 F16 输出转回 uint8（`out * 255`），再 resize 回 2 倍原始裁剪尺寸。sr 模型同样 `--fuse_preprocess`，输出为 F16，`out_buf_` 里读回的是半精度字节，转 float 后再乘 255。

## 6. zxing 解码

对应 `decodermgr.cpp`（配合 `binarizermgr.cpp`）。`WeChatQRCode::Impl::decodeCrop` 对裁剪图计算缩放列表 `getScaleList`（通常 scale = 2、比例缩放等），依次尝试：对每个 scale 先做上一步的超分/缩放，再调用 zxing 解码：

```cpp
DecoderMgr decodemgr;
const int ret = decodemgr.decodeImage(scaled_img, true, texts, zxing_points);
if (ret != 0) continue;   // 该 scale 失败，尝试下一个
```

`decodeImage` 内部用 `BinarizerMgr` 对灰度图做自适应二值化（`adaptive_threshold_mean_binarizer` 等），再用 zxing 的 QRCodeReader 完成 finder pattern 定位 + Reed-Solomon 纠错 + 码字解析，得到文本（texts）与二维码四个角点（zxing_points，位于缩放后图像的坐标系）。zxing 是 vendored 第三方库（原 Apache-2.0 License 头保留，位于 `zxing/` 目录），本轮在 CPU 上串行执行。

## 7. 坐标回投影 + 去重

对应 `wechat_qrcode.cpp` 的 `WeChatQRCode::Impl::decodeCrop`。zxing 返回的角点在缩放后的小图坐标系里，需要映射回原图：先除以当前 scale，再累加裁剪偏移：

```cpp
pt /= cur_scale;   // 回到裁剪图尺度
pt.x += crop_x;    // 回到原图
pt.y += crop_y;
```

随后按四个角点坐标去重（与 OpenCV 一致，`eps = 10px`）：若已存在一个结果的四个角点都与新结果相差小于 10 像素，则视为重复丢弃。最终每个成功解码的二维码输出一条 `text` + 一组 `4x2 float32` 角点，由 `emit` 转成 `detectAndDecode` 的返回值。

## 8. 多尺度与 sr 的取舍说明

`getScaleList` 会为较大的裁剪图生成多个 scale（例如 1.0、0.5、0.25……）以适配不同密度的二维码；每个候选会依次尝试这些 scale，一旦某个 scale 解码成功即停止。因此对同一张裁剪图，TPU 超分可能被调用多次（每个 scale 一次），`getBenchStats()` 中的 `sr_calls`/`zxing_calls` 就是这两段的累计调用次数，这也是 bench 输出里 `sr`/`zxing` 的耗时含义。

## 9. 模型输入输出口径

| 模型 | 输入 | 输出 | 说明 |
|------|------|------|------|
| detect_f32_fused.bmodel | `(1,1,H,W)` uint8 灰度，fused | `(1,1,100,7)` F32 | 19 个静态 shape 子网合并，按 `detect_<W>_<H>` 选网 |
| sr_f16_fused.bmodel | `(1,1,224,224)` uint8 灰度，fused | `(1,1,447,447)` F16 | 超分，仅 `sqrt(w*h)<160` 时触发 |

两个模型都使用 `--fuse_preprocess`（均值 0、scale 0.0039216、pixel_format=gray）在 TPU 图内完成归一化，故代码侧一律直接喂 uint8 灰度图。