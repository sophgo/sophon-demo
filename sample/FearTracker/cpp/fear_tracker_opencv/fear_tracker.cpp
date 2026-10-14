//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include "fear_tracker.hpp"

// ---------------------------------------------------------------------------
// fp16 → fp32 conversion (bmodel outputs are F16; BMNNTensor::get_cpu_data()
// does not handle BM_FLOAT16, so we copy the raw bytes and convert ourselves)
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
            // subnormal: normalise
            int e = 1;
            do { frac <<= 1; e--; } while ((frac & 0x400u) == 0);
            frac &= 0x3ffu;
            f = (sign << 31) | ((uint32_t)(127 + e - 15) << 23) | (frac << 13);
        }
    } else if (exp == 31) {
        // inf / nan
        f = (sign << 31) | (0xffu << 23) | (frac << 13);
    } else {
        // normalised
        f = (sign << 31) | ((uint32_t)(exp + 127 - 15) << 23) | (frac << 13);
    }
    float result;
    memcpy(&result, &f, sizeof(float));
    return result;
}

// ---------------------------------------------------------------------------
// Constructor / Destructor
// ---------------------------------------------------------------------------
FearTracker::FearTracker(std::shared_ptr<BMNNContext> context, int dev_id)
    : m_bmContext(context), m_dev_id(dev_id) {
}

FearTracker::~FearTracker() {
    if (m_template_buf) delete[] m_template_buf;
    if (m_search_buf)   delete[] m_search_buf;
    if (m_tensors_allocated) {
        bm_free_device(m_bmContext->handle(), m_template_dev.device_mem);
        bm_free_device(m_bmContext->handle(), m_search_dev.device_mem);
    }
}

// ---------------------------------------------------------------------------
// Init — load network, identify I/O, allocate tensors, precompute grid/window
// ---------------------------------------------------------------------------
void FearTracker::Init() {
    m_bmNetwork = m_bmContext->network(0);

    m_template_tensor = m_bmNetwork->inputTensor(0);  // [1,3,128,128]
    m_search_tensor   = m_bmNetwork->inputTensor(1);  // [1,3,256,256]

    // Identify outputs: bbox_pred has 4 channels, cls_pred has 1
    int out_num = m_bmNetwork->outputTensorNum();
    assert(out_num == 2);
    for (int i = 0; i < out_num; i++) {
        auto t = m_bmNetwork->outputTensor(i);
        auto shape = t->get_shape();
        int channels = shape->dims[1];
        if (channels == 4) m_bbox_idx = i;
        else if (channels == 1) m_cls_idx = i;
    }
    assert(m_bbox_idx >= 0 && m_cls_idx >= 0);

    // Allocate host buffers
    m_template_buf = new float[3 * template_size * template_size];
    m_search_buf   = new float[3 * instance_size * instance_size];

    // Allocate device memory for both inputs
    bmrt_tensor(&m_template_dev, m_bmContext->bmrt(),
                m_template_tensor->get_dtype(), *m_template_tensor->get_shape());
    m_template_tensor->set_device_mem(&m_template_dev.device_mem);

    bmrt_tensor(&m_search_dev, m_bmContext->bmrt(),
                m_search_tensor->get_dtype(), *m_search_tensor->get_shape());
    m_search_tensor->set_device_mem(&m_search_dev.device_mem);

    m_tensors_allocated = true;

    // Precompute output buffers
    m_bbox_output.resize(4 * score_size * score_size);
    m_cls_output.resize(1 * score_size * score_size);

    makeGrid();
    makeWindow();
}

// ---------------------------------------------------------------------------
// Grid & Window precomputation (mirrors make_grid / _make_window in Python)
// ---------------------------------------------------------------------------
void FearTracker::makeGrid() {
    int ss = score_size;
    m_grid_x.resize(ss * ss);
    m_grid_y.resize(ss * ss);
    int half = ss / 2;  // floor(16/2) = 8
    for (int r = 0; r < ss; r++) {
        for (int c = 0; c < ss; c++) {
            int idx = r * ss + c;
            m_grid_x[idx] = (float)((c - half) * total_stride + instance_size / 2);
            m_grid_y[idx] = (float)((r - half) * total_stride + instance_size / 2);
        }
    }
}

void FearTracker::makeWindow() {
    // cosine window = outer(hanning(ss), hanning(ss))
    // hanning(n) in numpy = 0.5 - 0.5*cos(2*pi*j/(n-1)), j=0..n-1
    int ss = score_size;
    m_window.resize(ss * ss);
    std::vector<float> hanning(ss);
    for (int j = 0; j < ss; j++) {
        hanning[j] = 0.5f - 0.5f * std::cos(2.0f * M_PI * (float)j / (float)(ss - 1));
    }
    for (int r = 0; r < ss; r++) {
        for (int c = 0; c < ss; c++) {
            m_window[r * ss + c] = hanning[r] * hanning[c];
        }
    }
}

// ---------------------------------------------------------------------------
// Preprocess: BGR/RGB uint8 → float32, normalize, CHW layout into buf
// ---------------------------------------------------------------------------
void FearTracker::preprocessToBuffer(const cv::Mat& image_rgb, float* buf, int H, int W) {
    // image_rgb is already RGB (caller ensures). Convert to float, normalize.
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            cv::Vec3b px = image_rgb.at<cv::Vec3b>(y, x);
            for (int ch = 0; ch < 3; ch++) {
                float val = (float)px[ch] / 255.0f;
                val = (val - m_mean[ch]) / m_std[ch];
                buf[ch * H * W + y * W + x] = val;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// extendBbox — expand bbox by offset on all sides
// Python's extend_bbox returns .astype("int32"), so we truncate to int here
// to keep the crop/pad/rescale math identical across frames.
// ---------------------------------------------------------------------------
Bbox FearTracker::extendBbox(const Bbox& bbox, float offset) {
    Bbox ctx;
    ctx.x = (float)(int)(bbox.x - bbox.w * offset);
    ctx.y = (float)(int)(bbox.y - bbox.h * offset);
    ctx.w = (float)(int)(bbox.w * (1.0f + 2.0f * offset));
    ctx.h = (float)(int)(bbox.h * (1.0f + 2.0f * offset));
    return ctx;
}

// ---------------------------------------------------------------------------
// ensureBboxBoundaries — clamp bbox into image bounds
// ---------------------------------------------------------------------------
Bbox FearTracker::ensureBboxBoundaries(const Bbox& bbox, int img_h, int img_w) {
    int x1 = std::min(std::max(0, (int)bbox.x), img_w);
    int y1 = std::min(std::max(0, (int)bbox.y), img_h);
    int x2 = std::min(std::max(0, (int)(bbox.x + bbox.w)), img_w);
    int y2 = std::min(std::max(0, (int)(bbox.y + bbox.h)), img_h);
    Bbox r;
    r.x = (float)x1; r.y = (float)y1;
    r.w = (float)(x2 - x1); r.h = (float)(y2 - y1);
    return r;
}

// ---------------------------------------------------------------------------
// clampBbox — ensure boundaries + enforce minimum side length
// ---------------------------------------------------------------------------
Bbox FearTracker::clampBbox(const Bbox& bbox, int img_h, int img_w, int min_side) {
    Bbox b = ensureBboxBoundaries(bbox, img_h, img_w);
    if (b.w < min_side) {
        b.w = min_side;
        if (b.x + b.w > img_w) b.x = (float)(img_w - min_side);
        if (b.x < 0) b.x = 0;
    }
    if (b.h < min_side) {
        b.h = min_side;
        if (b.y + b.h > img_h) b.y = (float)(img_h - min_side);
        if (b.y < 0) b.y = 0;
    }
    return b;
}

// ---------------------------------------------------------------------------
// getExtendedCrop — crop around bbox with context padding, resize to crop_size
// Returns the resized crop, fills resized_bbox (in crop coords) and context
// (the extended bbox in original image coords).
// ---------------------------------------------------------------------------
cv::Mat FearTracker::getExtendedCrop(const cv::Mat& image_rgb, const Bbox& bbox,
                                      int crop_size, float offset,
                                      const cv::Scalar& padding_value,
                                      Bbox& resized_bbox, Bbox& context) {
    context = extendBbox(bbox, offset);

    // Python casts context to int32, so truncate here to match.
    int cx0 = (int)context.x;
    int cy0 = (int)context.y;
    int cx1 = (int)(context.x + context.w);
    int cy1 = (int)(context.y + context.h);

    int pad_left   = std::max(-cx0, 0);
    int pad_top    = std::max(-cy0, 0);
    int pad_right  = std::max(cx1 - image_rgb.cols, 0);
    int pad_bottom = std::max(cy1 - image_rgb.rows, 0);

    int crop_x0 = cx0 + pad_left;
    int crop_y0 = cy0 + pad_top;
    int crop_x1 = cx1 - pad_right;
    int crop_y1 = cy1 - pad_bottom;

    // Clamp crop region to image (defensive)
    crop_x0 = std::max(0, std::min(crop_x0, image_rgb.cols));
    crop_y0 = std::max(0, std::min(crop_y0, image_rgb.rows));
    crop_x1 = std::max(0, std::min(crop_x1, image_rgb.cols));
    crop_y1 = std::max(0, std::min(crop_y1, image_rgb.rows));

    cv::Rect roi(crop_x0, crop_y0, crop_x1 - crop_x0, crop_y1 - crop_y0);
    cv::Mat crop = image_rgb(roi).clone();

    cv::Mat padded;
    cv::copyMakeBorder(crop, padded, pad_top, pad_bottom, pad_left, pad_right,
                       cv::BORDER_CONSTANT, padding_value);

    // padded_bbox = [bbox.x - context.x, bbox.y - context.y, bbox.w, bbox.h]
    Bbox padded_bbox;
    padded_bbox.x = bbox.x - context.x;
    padded_bbox.y = bbox.y - context.y;
    padded_bbox.w = bbox.w;
    padded_bbox.h = bbox.h;
    padded_bbox = ensureBboxBoundaries(padded_bbox, padded.rows, padded.cols);

    cv::Mat resized;
    cv::resize(padded, resized, cv::Size(crop_size, crop_size));

    float w_scale = (float)crop_size / (float)padded.cols;
    float h_scale = (float)crop_size / (float)padded.rows;
    resized_bbox.x = padded_bbox.x * w_scale;
    resized_bbox.y = padded_bbox.y * h_scale;
    resized_bbox.w = padded_bbox.w * w_scale;
    resized_bbox.h = padded_bbox.h * h_scale;

    return resized;
}

// ---------------------------------------------------------------------------
// postprocess — sigmoid, argmax, ltrb → xywh decode
// ---------------------------------------------------------------------------
Bbox FearTracker::postprocess(const float* bbox_pred, const float* cls_pred,
                              float& max_score) {
    int ss = score_size;
    // cls_sigmoid = 1 / (1 + exp(-cls))
    std::vector<float> cls_sigmoid(ss * ss);
    for (int i = 0; i < ss * ss; i++) {
        cls_sigmoid[i] = 1.0f / (1.0f + std::exp(-cls_pred[i]));
    }

    // argmax over the classification map
    int max_idx = 0;
    float best = cls_sigmoid[0];
    for (int i = 1; i < ss * ss; i++) {
        if (cls_sigmoid[i] > best) {
            best = cls_sigmoid[i];
            max_idx = i;
        }
    }
    max_score = best;

    // regression_map layout: [4, ss, ss] — channel-major (CHW)
    // pred_location = [grid_x - reg[0], grid_y - reg[1],
    //                   grid_x + reg[2], grid_y + reg[3]]
    int r_max = max_idx / ss;
    int c_max = max_idx % ss;
    float left   = m_grid_x[max_idx] - bbox_pred[0 * ss * ss + max_idx];
    float top    = m_grid_y[max_idx] - bbox_pred[1 * ss * ss + max_idx];
    float right  = m_grid_x[max_idx] + bbox_pred[2 * ss * ss + max_idx];
    float bottom = m_grid_y[max_idx] + bbox_pred[3 * ss * ss + max_idx];

    Bbox pred;
    pred.x = left;
    pred.y = top;
    pred.w = right - left;
    pred.h = bottom - top;
    return pred;
}

// ---------------------------------------------------------------------------
// rescaleBbox — map bbox from search-crop coords back to original image coords
// ---------------------------------------------------------------------------
Bbox FearTracker::rescaleBbox(const Bbox& bbox, const Bbox& padded_box) {
    float w_scale = padded_box.w / (float)instance_size;
    float h_scale = padded_box.h / (float)instance_size;
    Bbox r;
    r.x = std::round(bbox.x * w_scale + padded_box.x);
    r.y = std::round(bbox.y * h_scale + padded_box.y);
    r.w = (float)std::max(3, (int)std::round(bbox.w * w_scale));
    r.h = (float)std::max(3, (int)std::round(bbox.h * h_scale));
    return r;
}

// ---------------------------------------------------------------------------
// retrieve output as float32 (handles FP32 and FP16; INT8/INT32 via scale)
// ---------------------------------------------------------------------------
void FearTracker::retrieveOutput(int tensor_idx, std::vector<float>& out_buf) {
    auto tensor = m_bmNetwork->outputTensor(tensor_idx);
    auto shape  = tensor->get_shape();
    int count   = bmrt_shape_count(shape);
    auto dtype  = tensor->get_dtype();
    auto handle = m_bmContext->handle();
    const bm_device_mem_t* dmem = tensor->get_device_mem();

    out_buf.resize(count);

    if (dtype == BM_FLOAT32) {
        bm_memcpy_d2s_partial(handle, out_buf.data(), *dmem, count * sizeof(float));
    } else if (dtype == BM_FLOAT16) {
        std::vector<uint16_t> h16(count);
        bm_memcpy_d2s_partial(handle, h16.data(), *dmem, count * sizeof(uint16_t));
        for (int i = 0; i < count; i++) out_buf[i] = fp16_to_fp32(h16[i]);
    } else if (dtype == BM_INT8) {
        int8_t* p = new int8_t[count];
        bm_memcpy_d2s_partial(handle, p, *dmem, count * sizeof(int8_t));
        float scale = tensor->get_scale();
        for (int i = 0; i < count; i++) out_buf[i] = (float)p[i] * scale;
        delete[] p;
    } else if (dtype == BM_INT32) {
        int32_t* p = new int32_t[count];
        bm_memcpy_d2s_partial(handle, p, *dmem, count * sizeof(int32_t));
        float scale = tensor->get_scale();
        for (int i = 0; i < count; i++) out_buf[i] = (float)p[i] * scale;
        delete[] p;
    } else {
        std::cerr << "Unsupported output dtype: " << dtype << std::endl;
        assert(0);
    }
}

// ---------------------------------------------------------------------------
// initialize — set template from first frame
// ---------------------------------------------------------------------------
void FearTracker::initialize(const cv::Mat& image_rgb, const Bbox& bbox) {
    int img_h = image_rgb.rows;
    int img_w = image_rgb.cols;

    Bbox clamped = clampBbox(bbox, img_h, img_w);
    m_state_bbox = clamped;

    // mean color for padding (RGB)
    cv::Scalar mc = cv::mean(image_rgb);
    m_mean_color = cv::Scalar(mc[0], mc[1], mc[2]);

    Bbox resized_bbox, context;
    cv::Mat template_crop = getExtendedCrop(image_rgb, clamped,
                                            template_size, template_bbox_offset,
                                            m_mean_color, resized_bbox, context);

    preprocessToBuffer(template_crop, m_template_buf, template_size, template_size);
    bm_memcpy_s2d(m_bmContext->handle(), m_template_dev.device_mem,
                  (void*)m_template_buf);

    m_initialized = true;
}

// ---------------------------------------------------------------------------
// update — track object in a new frame
// ---------------------------------------------------------------------------
Bbox FearTracker::update(const cv::Mat& image_rgb) {
    assert(m_initialized);
    int img_h = image_rgb.rows;
    int img_w = image_rgb.cols;

    Bbox resized_bbox, context;
    cv::Mat search_crop = getExtendedCrop(image_rgb, m_state_bbox,
                                          instance_size, (float)search_context,
                                          m_mean_color, resized_bbox, context);

    // Store mapping = context (extended bbox in original image coords)
    m_state_mapping = context;
    m_state_prev_size.w = resized_bbox.w;
    m_state_prev_size.h = resized_bbox.h;

    preprocessToBuffer(search_crop, m_search_buf, instance_size, instance_size);
    bm_memcpy_s2d(m_bmContext->handle(), m_search_dev.device_mem,
                  (void*)m_search_buf);

    if (m_ts) m_ts->save("fear_tracker inference");
    int ret = m_bmNetwork->forward();
    assert(ret == 0);
    if (m_ts) m_ts->save("fear_tracker inference");

    // Retrieve outputs
    retrieveOutput(m_bbox_idx, m_bbox_output);
    retrieveOutput(m_cls_idx, m_cls_output);

    float max_score;
    Bbox pred = postprocess(m_bbox_output.data(), m_cls_output.data(), max_score);
    pred = rescaleBbox(pred, m_state_mapping);
    pred = clampBbox(pred, img_h, img_w);
    m_state_bbox = pred;

    return pred;
}
