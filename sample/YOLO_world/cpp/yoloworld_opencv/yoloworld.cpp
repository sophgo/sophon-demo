//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "yoloworld.hpp"
#include <fstream>
#include <cstring>
#include <algorithm>

// ---------------------------------------------------------------------------
// fp16 ↔ fp32 conversion (bmodel outputs may be F16; BMNNTensor::get_cpu_data()
// does not handle BM_FLOAT16, so we copy raw bytes and convert ourselves)
// ---------------------------------------------------------------------------
static float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (h >> 15) & 0x1u;
    uint32_t exp  = (h >> 10) & 0x1fu;
    uint32_t frac =  h         & 0x3ffu;
    uint32_t f;

    if (exp == 0) {
        if (frac == 0) {
            f = sign << 31;
        } else {
            int e = 1;
            do { frac <<= 1; e--; } while ((frac & 0x400u) == 0);
            frac &= 0x3ffu;
            f = (sign << 31) | ((uint32_t)(127 + e - 15) << 23) | (frac << 13);
        }
    } else if (exp == 31) {
        f = (sign << 31) | (0xffu << 23) | (frac << 13);
    } else {
        f = (sign << 31) | ((uint32_t)(exp + 127 - 15) << 23) | (frac << 13);
    }
    float result;
    memcpy(&result, &f, sizeof(float));
    return result;
}

static uint16_t fp32_to_fp16(float f) {
    uint32_t x;
    memcpy(&x, &f, sizeof(uint32_t));
    uint32_t sign = (x >> 16) & 0x8000u;
    int32_t  exp  = (int32_t)((x >> 23) & 0xffu) - 127 + 15;
    uint32_t frac =  x & 0x7fffffu;

    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        frac |= 0x800000u;
        uint32_t shift = (uint32_t)(14 - exp);
        uint16_t h = (uint16_t)(sign | (frac >> shift));
        if ((frac >> (shift - 1)) & 1) h++;
        return h;
    } else if (exp >= 31) {
        return (uint16_t)(sign | 0x7c00u);
    }
    return (uint16_t)(sign | ((uint32_t)exp << 10) | (frac >> 13));
}

// ---------------------------------------------------------------------------
// Minimal numpy .npy reader for float32 (little-endian) arrays.
// Used to load precomputed txt_feats [1, N, 512].
// ---------------------------------------------------------------------------
static bool read_npy_f32(const std::string& path,
                         std::vector<float>& data,
                         std::vector<int>& shape) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) return false;

    // Magic: \x93NUMPY
    char magic[6];
    f.read(magic, 6);
    if (memcmp(magic, "\x93NUMPY", 6) != 0) return false;

    // Version
    uint8_t major, minor;
    f.read((char*)&major, 1);
    f.read((char*)&minor, 1);

    // Header length
    uint32_t header_len;
    if (major == 1) {
        uint16_t hlen;
        f.read((char*)&hlen, 2);
        header_len = hlen;
    } else {
        f.read((char*)&header_len, 4);
    }

    // Header string (ASCII dict)
    std::string header(header_len, '\0');
    f.read(&header[0], header_len);

    // Parse shape from header: "'shape': (1, 80, 512),"
    shape.clear();
    size_t pos = header.find("'shape':");
    if (pos == std::string::npos) return false;
    pos = header.find('(', pos);
    if (pos == std::string::npos) return false;
    pos++;
    while (pos < header.size() && header[pos] != ')') {
        // skip non-digit
        while (pos < header.size() && header[pos] != ')' &&
               (header[pos] < '0' || header[pos] > '9'))
            pos++;
        if (pos >= header.size() || header[pos] == ')') break;
        int dim = 0;
        while (pos < header.size() && header[pos] >= '0' && header[pos] <= '9') {
            dim = dim * 10 + (header[pos] - '0');
            pos++;
        }
        shape.push_back(dim);
    }

    // Compute total elements
    size_t total = 1;
    for (int d : shape) total *= d;

    // Read raw float32 data
    data.resize(total);
    f.read((char*)data.data(), total * sizeof(float));
    return f.good() || f.eof();
}

// ---------------------------------------------------------------------------
// Constructor / Destructor
// ---------------------------------------------------------------------------
YoloWorld::YoloWorld(std::shared_ptr<BMNNContext> context, int dev_id)
    : m_bmContext(context), m_dev_id(dev_id) {
}

YoloWorld::~YoloWorld() {
    if (m_input_buf) delete[] m_input_buf;
    if (m_tensors_allocated) {
        bm_free_device(m_bmContext->handle(), m_images_dev.device_mem);
        bm_free_device(m_bmContext->handle(), m_txt_feats_dev.device_mem);
    }
}

// ---------------------------------------------------------------------------
// Init — load network, identify I/O, allocate tensors
// ---------------------------------------------------------------------------
int YoloWorld::Init(float confThresh, float nmsThresh,
                    const std::string& class_names_file) {
    m_confThreshold = confThresh;
    m_nmsThreshold  = nmsThresh;

    m_bmNetwork = m_bmContext->network(0);

    // Input 0: images [1, 3, H, W]
    m_images_tensor = m_bmNetwork->inputTensor(0);
    auto img_shape = m_images_tensor->get_shape();
    assert(img_shape->num_dims == 4);
    max_batch = img_shape->dims[0];
    m_net_h   = img_shape->dims[2];
    m_net_w   = img_shape->dims[3];

    // Input 1: txt_feats [1, N, 512]
    m_txt_feats_tensor = m_bmNetwork->inputTensor(1);
    auto txt_shape = m_txt_feats_tensor->get_shape();
    assert(txt_shape->num_dims == 3);
    m_num_classes = txt_shape->dims[1];
    assert(txt_shape->dims[2] == 512);

    // Output 0: [1, 4+nc, anchors] (channel-major). OPT models export the
    // transposed layout [1, anchors, 4+nc], which this sample does not
    // support — same check as the Python example.
    auto out_shape = m_bmNetwork->outputTensor(0)->get_shape();
    assert(out_shape->num_dims == 3);
    m_output_nout     = out_shape->dims[1];
    m_output_anchors  = out_shape->dims[2];
    if (m_output_nout > m_output_anchors) {
        std::cerr << "[YoloWorld] OPT model is not supported "
                     "(output [" << m_output_nout << "," << m_output_anchors
                  << "])" << std::endl;
        return -1;
    }
    if (m_output_nout != 4 + m_num_classes) {
        std::cerr << "[YoloWorld] output channels (" << m_output_nout
                  << ") != 4 + num_classes (" << 4 + m_num_classes << ")"
                  << std::endl;
        return -1;
    }

    std::cout << "[YoloWorld] net: " << max_batch << "x" << m_net_h << "x"
              << m_net_w << ", num_classes=" << m_num_classes
              << ", output [1," << m_output_nout << "," << m_output_anchors
              << "]" << std::endl;

    // Allocate host buffer for preprocessed image
    m_input_buf = new float[3 * m_net_h * m_net_w];

    // Allocate device memory for images input
    bmrt_tensor(&m_images_dev, m_bmContext->bmrt(),
                m_images_tensor->get_dtype(), *img_shape);
    m_images_tensor->set_device_mem(&m_images_dev.device_mem);

    // Allocate device memory for txt_feats input
    bmrt_tensor(&m_txt_feats_dev, m_bmContext->bmrt(),
                m_txt_feats_tensor->get_dtype(), *txt_shape);
    m_txt_feats_tensor->set_device_mem(&m_txt_feats_dev.device_mem);

    m_tensors_allocated = true;

    // Load class names
    loadClassNames(class_names_file);

    return 0;
}

// ---------------------------------------------------------------------------
// loadClassNames — read coco.names (one class name per line)
// ---------------------------------------------------------------------------
int YoloWorld::loadClassNames(const std::string& class_names_file) {
    m_class_names.clear();
    std::ifstream ifs(class_names_file);
    if (ifs.is_open()) {
        std::string line;
        while (std::getline(ifs, line)) {
            // strip trailing \r or \n
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (!line.empty()) m_class_names.push_back(line);
        }
    }
    std::cout << "[YoloWorld] loaded " << m_class_names.size()
              << " class names from " << class_names_file << std::endl;
    return 0;
}

// ---------------------------------------------------------------------------
// loadTxtFeats — load precomputed text embeddings from .npy and copy to device
// The .npy must be float32 [1, N, 512] where N == m_num_classes.
// ---------------------------------------------------------------------------
int YoloWorld::loadTxtFeats(const std::string& npy_path) {
    std::vector<float> data;
    std::vector<int> shape;
    if (!read_npy_f32(npy_path, data, shape)) {
        std::cerr << "[YoloWorld] Failed to read npy: " << npy_path << std::endl;
        return -1;
    }

    // Squeeze leading singleton dims; expect [1, N, 512] or [N, 512]
    size_t total = data.size();
    int n_classes = m_num_classes;
    size_t expected = (size_t)n_classes * 512;
    if (total < expected) {
        std::cerr << "[YoloWorld] txt_feats too small: " << total
                  << " < " << expected << std::endl;
        return -1;
    }

    std::cout << "[YoloWorld] loaded txt_feats shape: [";
    for (size_t i = 0; i < shape.size(); i++) {
        std::cout << shape[i] << (i + 1 < shape.size() ? ", " : "");
    }
    std::cout << "], " << total << " floats" << std::endl;

    auto dtype = m_txt_feats_tensor->get_dtype();
    if (dtype == BM_FLOAT32) {
        bm_memcpy_s2d(m_bmContext->handle(), m_txt_feats_dev.device_mem,
                      (void*)data.data());
    } else if (dtype == BM_FLOAT16) {
        // Convert F32 → F16 before copying
        size_t count = (size_t)max_batch * n_classes * 512;
        std::vector<uint16_t> h16(count);
        for (size_t i = 0; i < count; i++)
            h16[i] = fp32_to_fp16(data[i]);
        bm_memcpy_s2d(m_bmContext->handle(), m_txt_feats_dev.device_mem,
                      (void*)h16.data());
    } else {
        std::cerr << "[YoloWorld] unsupported txt_feats dtype: " << dtype
                  << std::endl;
        return -1;
    }

    return 0;
}

// ---------------------------------------------------------------------------
// letterbox — resize + center-pad to new_h × new_w, keeping aspect ratio.
// Mirrors Python's letterbox(scaleup=True, stride=32, color=(114,114,114)).
// ---------------------------------------------------------------------------
void YoloWorld::letterbox(const cv::Mat& src, cv::Mat& dst, int new_h, int new_w,
                          float& ratio, int& tx, int& ty) {
    int h = src.rows, w = src.cols;
    float r = std::min((float)new_h / h, (float)new_w / w);
    // scaleup=True: do not cap at 1.0

    int new_unpad_w = (int)std::round(w * r);
    int new_unpad_h = (int)std::round(h * r);

    float dw = new_w - new_unpad_w;
    float dh = new_h - new_unpad_h;
    dw /= 2.0f;
    dh /= 2.0f;

    int top    = (int)std::round(dh - 0.1f);
    int bottom = (int)std::round(dh + 0.1f);
    int left   = (int)std::round(dw - 0.1f);
    int right  = (int)std::round(dw + 0.1f);

    cv::Mat resized;
    if (w != new_unpad_w || h != new_unpad_h)
        cv::resize(src, resized, cv::Size(new_unpad_w, new_unpad_h), 0, 0,
                   cv::INTER_LINEAR);
    else
        resized = src;

    cv::copyMakeBorder(resized, dst, top, bottom, left, right,
                       cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));

    ratio = r;
    tx = left;
    ty = top;
}

// ---------------------------------------------------------------------------
// pre_process — letterbox + BGR→RGB + HWC→CHW + /255, copy to device
// ---------------------------------------------------------------------------
int YoloWorld::pre_process(const cv::Mat& image, float ratio[2], int txy[2]) {
    cv::Mat letterboxed;
    float r;
    int tx, ty;
    letterbox(image, letterboxed, m_net_h, m_net_w, r, tx, ty);
    ratio[0] = ratio[1] = r;
    txy[0] = tx;
    txy[1] = ty;

    // BGR → RGB, HWC → CHW, /255
    int H = m_net_h, W = m_net_w;
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            cv::Vec3b px = letterboxed.at<cv::Vec3b>(y, x);
            // BGR in OpenCV → write as RGB in CHW
            m_input_buf[0 * H * W + y * W + x] = (float)px[2] / 255.0f;  // R
            m_input_buf[1 * H * W + y * W + x] = (float)px[1] / 255.0f;  // G
            m_input_buf[2 * H * W + y * W + x] = (float)px[0] / 255.0f;  // B
        }
    }

    // Copy to device (input dtype is F32 for this model)
    auto dtype = m_images_tensor->get_dtype();
    if (dtype == BM_FLOAT32) {
        bm_memcpy_s2d(m_bmContext->handle(), m_images_dev.device_mem,
                      (void*)m_input_buf);
    } else if (dtype == BM_FLOAT16) {
        size_t count = (size_t)3 * H * W;
        std::vector<uint16_t> h16(count);
        for (size_t i = 0; i < count; i++)
            h16[i] = fp32_to_fp16(m_input_buf[i]);
        bm_memcpy_s2d(m_bmContext->handle(), m_images_dev.device_mem,
                      (void*)h16.data());
    } else {
        std::cerr << "[YoloWorld] unsupported image input dtype: " << dtype
                  << std::endl;
        return -1;
    }

    return 0;
}

// ---------------------------------------------------------------------------
// retrieveOutput — copy output tensor to host as F32 (handles F32/F16/INT8)
// ---------------------------------------------------------------------------
void YoloWorld::retrieveOutput(std::vector<float>& out_buf) {
    auto tensor = m_bmNetwork->outputTensor(0);
    auto shape  = tensor->get_shape();
    int count   = bmrt_shape_count(shape);
    auto dtype  = tensor->get_dtype();
    auto handle = m_bmContext->handle();
    const bm_device_mem_t* dmem = tensor->get_device_mem();

    out_buf.resize(count);

    if (dtype == BM_FLOAT32) {
        bm_memcpy_d2s_partial(handle, out_buf.data(), *dmem,
                              count * sizeof(float));
    } else if (dtype == BM_FLOAT16) {
        std::vector<uint16_t> h16(count);
        bm_memcpy_d2s_partial(handle, h16.data(), *dmem,
                              count * sizeof(uint16_t));
        for (int i = 0; i < count; i++) out_buf[i] = fp16_to_fp32(h16[i]);
    } else if (dtype == BM_INT8) {
        int8_t* p = new int8_t[count];
        bm_memcpy_d2s_partial(handle, p, *dmem, count * sizeof(int8_t));
        float scale = tensor->get_scale();
        for (int i = 0; i < count; i++) out_buf[i] = (float)p[i] * scale;
        delete[] p;
    } else {
        std::cerr << "[YoloWorld] unsupported output dtype: " << dtype
                  << std::endl;
        assert(0);
    }
}

// ---------------------------------------------------------------------------
// post_process — NMS matching Python postprocess_numpy.py
// Output layout: [1, 4+nc, anchors] = [batch, 4+nc, anchors]
//   channel 0-3: xywh (center x, center y, width, height)
//   channel 4-(3+nc): class scores
// ---------------------------------------------------------------------------
int YoloWorld::post_process(float ratio[2], int txy[2], int img_w, int img_h,
                             YoloWorldBoxVec& boxes) {
    // Output is [1, 4+nc, anchors]
    int nc = m_num_classes;
    int num_anchors = m_output_anchors;
    int max_wh = 7680;

    const float* output = m_output_buf.data();

    // For single-batch, offset is 0
    YoloWorldBoxVec candidates;
    candidates.reserve(256);

    for (int a = 0; a < num_anchors; a++) {
        // Find max class score for this anchor
        float max_score = -1.0f;
        int   max_class = 0;
        for (int c = 0; c < nc; c++) {
            float s = output[(4 + c) * num_anchors + a];
            if (s > max_score) {
                max_score = s;
                max_class = c;
            }
        }

        if (max_score > m_confThreshold) {
            // xywh → xyxy
            float cx = output[0 * num_anchors + a];
            float cy = output[1 * num_anchors + a];
            float w  = output[2 * num_anchors + a];
            float h  = output[3 * num_anchors + a];

            YoloWorldBox box;
            box.x1 = cx - w / 2.0f;
            box.y1 = cy - h / 2.0f;
            box.x2 = cx + w / 2.0f;
            box.y2 = cy + h / 2.0f;
            box.score = max_score;
            box.class_id = max_class;

            // Offset for batched NMS (class-aware)
            if (!m_agnostic) {
                box.x1 += max_class * max_wh;
                box.y1 += max_class * max_wh;
                box.x2 += max_class * max_wh;
                box.y2 += max_class * max_wh;
            }
            candidates.push_back(box);
        }
    }

    // Sort by confidence descending
    std::sort(candidates.begin(), candidates.end(),
              [](const YoloWorldBox& a, const YoloWorldBox& b) {
                  return a.score > b.score;
              });

    // NMS
    NMS(candidates, m_nmsThreshold);

    // Remove offset and rescale to original image coords
    for (auto& box : candidates) {
        if (!m_agnostic) {
            float off = box.class_id * max_wh;
            box.x1 -= off;
            box.y1 -= off;
            box.x2 -= off;
            box.y2 -= off;
        }
        // Subtract padding
        box.x1 -= txy[0];
        box.x2 -= txy[0];
        box.y1 -= txy[1];
        box.y2 -= txy[1];
        // Divide by ratio
        box.x1 /= ratio[0];
        box.x2 /= ratio[0];
        box.y1 /= ratio[1];
        box.y2 /= ratio[1];
        // Clip to original image
        box.x1 = std::max(0.0f, std::min(box.x1, (float)(img_w - 1)));
        box.x2 = std::max(0.0f, std::min(box.x2, (float)(img_w - 1)));
        box.y1 = std::max(0.0f, std::min(box.y1, (float)(img_h - 1)));
        box.y2 = std::max(0.0f, std::min(box.y2, (float)(img_h - 1)));

        boxes.push_back(box);
    }

    return 0;
}

// ---------------------------------------------------------------------------
// NMS — greedy NMS on xyxy boxes, matching Python nms_boxes: dets must be
// sorted by score descending; the highest-score box survives and suppresses
// lower-score boxes with IoU > threshold (same +0.00001 eps as Python).
// ---------------------------------------------------------------------------
void YoloWorld::NMS(YoloWorldBoxVec& dets, float nmsConfidence) {
    int n = (int)dets.size();
    if (n == 0) return;

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
        areas[i] = (dets[i].x2 - dets[i].x1) * (dets[i].y2 - dets[i].y1);

    std::vector<char> suppressed(n, 0);
    for (int i = 0; i < n; i++) {
        if (suppressed[i]) continue;
        for (int j = i + 1; j < n; j++) {
            if (suppressed[j]) continue;
            float left   = std::max(dets[i].x1, dets[j].x1);
            float top    = std::max(dets[i].y1, dets[j].y1);
            float right  = std::min(dets[i].x2, dets[j].x2);
            float bottom = std::min(dets[i].y2, dets[j].y2);
            float overlap = std::max(0.0f, right - left + 0.00001f) *
                             std::max(0.0f, bottom - top + 0.00001f);
            float iou = overlap / (areas[i] + areas[j] - overlap);
            if (iou > nmsConfidence) suppressed[j] = 1;
        }
    }

    int keep = 0;
    for (int i = 0; i < n; i++) {
        if (!suppressed[i]) dets[keep++] = dets[i];
    }
    dets.resize(keep);

    // max_det = 300, same as the Python example
    if (dets.size() > 300) dets.resize(300);
}

// ---------------------------------------------------------------------------
// Detect — preprocess + forward + postprocess
// ---------------------------------------------------------------------------
int YoloWorld::Detect(const std::vector<cv::Mat>& images,
                      std::vector<YoloWorldBoxVec>& boxes) {
    for (size_t i = 0; i < images.size(); i++) {
        float ratio[2];
        int txy[2];

        LOG_TS(m_ts, "yoloworld preprocess");
        int ret = pre_process(images[i], ratio, txy);
        if (ret != 0) return ret;
        LOG_TS(m_ts, "yoloworld preprocess");

        LOG_TS(m_ts, "yoloworld inference");
        ret = m_bmNetwork->forward();
        assert(ret == 0);
        LOG_TS(m_ts, "yoloworld inference");

        LOG_TS(m_ts, "yoloworld postprocess");
        retrieveOutput(m_output_buf);
        YoloWorldBoxVec frame_boxes;
        post_process(ratio, txy, images[i].cols, images[i].rows, frame_boxes);
        boxes.push_back(frame_boxes);
        LOG_TS(m_ts, "yoloworld postprocess");
    }
    return 0;
}

// ---------------------------------------------------------------------------
// drawPred — draw a single detection on the frame
// ---------------------------------------------------------------------------
void YoloWorld::drawPred(cv::Mat& frame, const YoloWorldBox& box) {
    const std::vector<std::vector<int>> colors = {
        {255, 0, 0}, {0, 255, 0}, {0, 0, 255}, {255, 255, 0},
        {0, 255, 255}, {255, 0, 255}, {255, 165, 0}, {128, 0, 128}
    };
    int color_idx = box.class_id % (int)colors.size();
    cv::rectangle(frame,
                  cv::Point((int)box.x1, (int)box.y1),
                  cv::Point((int)box.x2, (int)box.y2),
                  cv::Scalar(colors[color_idx][0], colors[color_idx][1],
                             colors[color_idx][2]),
                  2);

    std::string label = cv::format("%.2f", box.score);
    if ((int)m_class_names.size() > box.class_id)
        label = m_class_names[box.class_id] + ":" + label;

    int baseLine;
    cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX,
                                         0.5, 1, &baseLine);
    int top = std::max((int)box.y1, labelSize.height);
    cv::putText(frame, label, cv::Point((int)box.x1, top),
                cv::FONT_HERSHEY_SIMPLEX, 0.5,
                cv::Scalar(colors[color_idx][0], colors[color_idx][1],
                           colors[color_idx][2]),
                1);
}
