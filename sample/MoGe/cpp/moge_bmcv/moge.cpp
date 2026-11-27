//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include "moge.hpp"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <functional>
#include <stdexcept>

namespace {
constexpr double kPi = 3.14159265358979323846;
}

namespace {

// --- normalized_view_plane_uv -------------------------------------------------
// Mirrors moge.utils.geometry_torch.normalized_view_plane_uv (dtype=float32):
// UV with left-top corner (-width/diag, -height/diag) and right-bottom
// (width/diag, height/diag). The linspace is evaluated in float32 (like numpy's
// ``np.linspace(..., dtype=np.float32)``) and then promoted to double, so the
// recovered focal/shift matches the Python/reference implementation exactly.
void normalized_view_plane_uv(int width, int height, std::vector<double>& uv_u,
                              std::vector<double>& uv_v) {
    double aspect = (double)width / height;
    double span_x = aspect / std::sqrt(1.0 + aspect * aspect);
    double span_y = 1.0 / std::sqrt(1.0 + aspect * aspect);
    // np.linspace(start, stop, N, dtype=float32): endpoints cast to float32
    float lo_u = (float)(-span_x * (width - 1) / width);
    float hi_u = (float)(span_x * (width - 1) / width);
    float lo_v = (float)(-span_y * (height - 1) / height);
    float hi_v = (float)(span_y * (height - 1) / height);
    float step_u = (width > 1) ? ((hi_u - lo_u) / (float)(width - 1)) : 0.0f;
    float step_v = (height > 1) ? ((hi_v - lo_v) / (float)(height - 1)) : 0.0f;
    uv_u.resize((size_t)width * height);
    uv_v.resize((size_t)width * height);
    for (int i = 0; i < height; i++) {
        float v = (height > 1) ? lo_v + step_v * (float)i : lo_v;
        if (i == height - 1) v = hi_v;  // numpy forces last element == stop
        for (int j = 0; j < width; j++) {
            float u = (width > 1) ? lo_u + step_u * (float)j : lo_u;
            if (j == width - 1) u = hi_u;
            uv_u[(size_t)i * width + j] = (double)u;
            uv_v[(size_t)i * width + j] = (double)v;
        }
    }
}

// --- closed-form focal + residual cost for a given shift ----------------------
double focal_shift_cost(const std::vector<double>& uv_u, const std::vector<double>& uv_v,
                        const std::vector<double>& px, const std::vector<double>& py,
                        const std::vector<double>& pz, double shift, double* focal_out) {
    int n = (int)pz.size();
    for (int k = 0; k < n; k++)
        if (pz[k] + shift <= 1e-6) {   // infeasible: keep the fit feasible (depth > 0)
            if (focal_out) *focal_out = 0.0;
            return (2.0 * n) * 1e6;     // mirror residual = [1e3]*2n
        }

    double num = 0.0, den = 0.0;
    for (int k = 0; k < n; k++) {
        double zs = pz[k] + shift;
        double x = px[k] / zs;
        double y = py[k] / zs;
        num += x * uv_u[k] + y * uv_v[k];
        den += x * x + y * y;
    }
    if (den <= 1e-12) {
        if (focal_out) *focal_out = 0.0;
        return (2.0 * n) * 1e6;
    }
    double focal = num / den;
    double cost = 0.0;
    for (int k = 0; k < n; k++) {
        double zs = pz[k] + shift;
        double x = px[k] / zs;
        double y = py[k] / zs;
        double ru = focal * x - uv_u[k];
        double rv = focal * y - uv_v[k];
        cost += ru * ru + rv * rv;
    }
    if (focal_out) *focal_out = focal;
    return cost;
}

double gss_min(const std::function<double(double)>& f, double a, double b, double tol = 1e-6,
               int max_iter = 200) {
    const double gr = (std::sqrt(5.0) - 1.0) / 2.0;
    double c = b - gr * (b - a);
    double d = a + gr * (b - a);
    double fc = f(c), fd = f(d);
    for (int it = 0; it < max_iter; it++) {
        if (fc < fd) {
            b = d; d = c; fd = fc;
            c = b - gr * (b - a);
            fc = f(c);
        } else {
            a = c; c = d; fc = fd;
            d = a + gr * (b - a);
            fd = f(d);
        }
        if (std::abs(b - a) < tol * (std::abs(a) + std::abs(b) + 1e-6))
            break;
    }
    return (a + b) / 2.0;
}

void solve_optimal_focal_shift(const std::vector<double>& uv_u, const std::vector<double>& uv_v,
                               const std::vector<double>& px, const std::vector<double>& py,
                               const std::vector<double>& pz, double& shift, double& focal) {
    double zmin = *std::min_element(pz.begin(), pz.end());
    double zmax = *std::max_element(pz.begin(), pz.end());
    double za = std::max(std::abs(zmin), std::abs(zmax));
    double lo = -zmin + std::max(1e-3, 0.05 * std::abs(zmin));
    double hi = std::max(4.0, 4.0 * za);

    auto cost = [&](double s) {
        return focal_shift_cost(uv_u, uv_v, px, py, pz, s, (double*)nullptr);
    };
    // expand the right boundary if the minimum sits on it
    if (cost(hi) < cost((lo + hi) / 2.0)) {
        while (cost(hi) < cost(hi - (hi - lo) / 2.0) && hi < 1e4)
            hi *= 2.0;
    }
    shift = gss_min(cost, lo, hi);
    focal_shift_cost(uv_u, uv_v, px, py, pz, shift, &focal);
}

// recover (focal, shift). mirrors moge.utils.geometry_torch.recover_focal_shift with
// the same 64x64 nearest downsample (src = floor(dst * src_size / dst_size)).
void recover_focal_shift(const float* points, const uint8_t* mask, int H, int W,
                         double& focal, double& shift) {
    std::vector<double> uv_u, uv_v;
    normalized_view_plane_uv(W, H, uv_u, uv_v);

    const int TH = 64, TW = 64;
    std::vector<int> ys(TH), xs(TW);
    for (int k = 0; k < TH; k++) ys[k] = std::min(k * H / TH, H - 1);
    for (int k = 0; k < TW; k++) xs[k] = std::min(k * W / TW, W - 1);

    std::vector<double> px, py, pz, uu, vv;
    px.reserve(TH * TW); py.reserve(TH * TW); pz.reserve(TH * TW);
    uu.reserve(TH * TW); vv.reserve(TH * TW);
    for (int i = 0; i < TH; i++) {
        for (int j = 0; j < TW; j++) {
            int si = ys[i], sj = xs[j];
            if (mask != NULL && !mask[si * W + sj])
                continue;
            const float* p = points + ((size_t)si * W + sj) * 3;
            px.push_back(p[0]); py.push_back(p[1]); pz.push_back(p[2]);
            uu.push_back(uv_u[(size_t)si * W + sj]);
            vv.push_back(uv_v[(size_t)si * W + sj]);
        }
    }
    if (px.size() < 2) {
        focal = 1.0; shift = 0.0;
        return;
    }
    solve_optimal_focal_shift(uu, vv, px, py, pz, shift, focal);
}

}  // namespace

MoGe::MoGe(std::string bmodel_file, int dev_id) {
    // get handle
    if (bm_dev_request(&handle, dev_id) != BM_SUCCESS)
        throw std::runtime_error("bm_dev_request failed");

    if (bm_get_misc_info(handle, &misc_info) != BM_SUCCESS)
        throw std::runtime_error("bm_get_misc_info failed");

    // create bmrt
    bmrt = bmrt_create(handle);
    if (!bmrt || !bmrt_load_bmodel(bmrt, bmodel_file.c_str()))
        throw std::runtime_error("load bmodel(" + bmodel_file + ") failed");

    // get network names
    const char** names = NULL;
    int num = bmrt_get_network_number(bmrt);
    bmrt_get_network_names(bmrt, &names);
    for (int i = 0; i < num; ++i)
        network_names.push_back(names[i]);
    free(names);

    netinfo = bmrt_get_network_info(bmrt, network_names[0].c_str());
    if (netinfo->stage_num > 1)
        std::cout << "This bmodel have " << netinfo->stage_num
                  << " stages, and this program will only take stage 0." << std::endl;

    batch_size = netinfo->stages[0].input_shapes[0].dims[0];
    net_h = netinfo->stages[0].input_shapes[0].dims[2];
    net_w = netinfo->stages[0].input_shapes[0].dims[3];
    if (netinfo->input_dtypes[0] != BM_FLOAT32)
        throw std::runtime_error("MoGe bmodel input must be fp32");

    // locate the four outputs by name
    out_idx_points = out_idx_normal = out_idx_mask = out_idx_metric_scale = -1;
    for (int i = 0; i < netinfo->output_num; i++) {
        std::string n = netinfo->output_names[i];
        std::string low;
        low.reserve(n.size());
        for (char c : n) low.push_back((char)std::tolower((unsigned char)c));
        if (out_idx_metric_scale < 0 && low.find("metric_scale") != std::string::npos)
            out_idx_metric_scale = i;
        else if (out_idx_mask < 0 && low.find("mask") != std::string::npos)
            out_idx_mask = i;
        else if (out_idx_normal < 0 && low.find("normal") != std::string::npos)
            out_idx_normal = i;
        else if (out_idx_points < 0 && low.find("points") != std::string::npos)
            out_idx_points = i;
    }
    if (out_idx_points < 0 || out_idx_normal < 0 || out_idx_mask < 0 || out_idx_metric_scale < 0)
        throw std::runtime_error("could not locate all four outputs in bmodel");

    m_ts = &tmp_ts;
}

MoGe::~MoGe() {
    if (bmrt != NULL) {
        bmrt_destroy(bmrt);
        bmrt = NULL;
    }
    bm_dev_free(handle);
}

int MoGe::pre_process(const bm_image& src, bm_tensor_t& input_tensor) {
    // 1. BGR packed -> RGB planar byte, direct resize (bilinear). MoGe input is square
    //    (aspect ratio = 1), no letter-box.
    bm_image rgb_resized;
    bm_image_create(handle, net_h, net_w, FORMAT_RGB_PLANAR, DATA_TYPE_EXT_1N_BYTE, &rgb_resized);
    bm_image_alloc_dev_mem(rgb_resized, BMCV_IMAGE_FOR_IN);
    bm_status_t ret = bmcv_image_vpp_convert(handle, 1, src, &rgb_resized);
    if (ret != BM_SUCCESS) {
        bm_image_destroy(rgb_resized);
        throw std::runtime_error("bmcv_image_vpp_convert failed");
    }

    // 2. allocate fp32 input tensor; convert uint8 -> fp32 with an identity scale.
    //    the /255 normalization is done in fp32 on the host so it is exact
    //    (bmcv convert_to fixed-point scale would quantize 1/255 to 1/256).
    bool ok = bmrt_tensor(&input_tensor, bmrt, netinfo->input_dtypes[0], netinfo->stages[0].input_shapes[0]);
    if (!ok) {
        bm_image_destroy(rgb_resized);
        throw std::runtime_error("bmrt_tensor failed");
    }

    bm_image converto;
    bm_image_create(handle, net_h, net_w, FORMAT_RGB_PLANAR, DATA_TYPE_EXT_FLOAT32, &converto);
    bm_image_attach_contiguous_mem(1, &converto, input_tensor.device_mem);

    bmcv_convert_to_attr attr;
    memset(&attr, 0, sizeof(attr));
    attr.alpha_0 = attr.alpha_1 = attr.alpha_2 = 1.0f;
    ret = bmcv_image_convert_to(handle, 1, attr, &rgb_resized, &converto);
    if (ret != BM_SUCCESS) {
#if BMCV_VERSION_MAJOR > 1
        bm_image_detach_contiguous_mem(1, &converto);
#else
        bm_image_dettach_contiguous_mem(1, &converto);
#endif
        bm_image_destroy(converto);
        bm_image_destroy(rgb_resized);
        bm_free_device(handle, input_tensor.device_mem);
        throw std::runtime_error("bmcv_image_convert_to failed");
    }

    // 3. read 0..255 fp32, normalize to [0, 1], write back
    if (bm_thread_sync(handle) != BM_SUCCESS) {
#if BMCV_VERSION_MAJOR > 1
        bm_image_detach_contiguous_mem(1, &converto);
#else
        bm_image_dettach_contiguous_mem(1, &converto);
#endif
        bm_image_destroy(converto);
        bm_image_destroy(rgb_resized);
        bm_free_device(handle, input_tensor.device_mem);
        throw std::runtime_error("bm_thread_sync failed");
    }
    int count = bmrt_shape_count(&netinfo->stages[0].input_shapes[0]);
    float* host = new float[count];
    bm_memcpy_d2s_partial(handle, host, input_tensor.device_mem, (size_t)count * sizeof(float));
    for (int i = 0; i < count; i++)
        host[i] *= (1.0f / 255.0f);
    bm_memcpy_s2d(handle, input_tensor.device_mem, host);
    delete[] host;

    // cleanup (converto detach before destroy so the tensor device_mem is not freed)
#if BMCV_VERSION_MAJOR > 1
    bm_image_detach_contiguous_mem(1, &converto);
#else
    bm_image_dettach_contiguous_mem(1, &converto);
#endif
    bm_image_destroy(converto);
    bm_image_destroy(rgb_resized);
    return 0;
}

int MoGe::forward(bm_tensor_t& input_tensor, std::vector<bm_tensor_t>& output_tensors) {
    bool ok = bmrt_launch_tensor(bmrt, netinfo->name, &input_tensor, netinfo->input_num,
                                 output_tensors.data(), netinfo->output_num);
    if (!ok)
        throw std::runtime_error("bmrt_launch_tensor failed");
    if (bm_thread_sync(handle) != BM_SUCCESS)
        throw std::runtime_error("bm_thread_sync failed");
    bm_free_device(handle, input_tensor.device_mem);
    return 0;
}

int MoGe::post_process(std::vector<bm_tensor_t>& output_tensors, MoGeResult& result) {
    int H = net_h, W = net_w;
    int out_h = netinfo->stages[0].output_shapes[out_idx_mask].dims[1];
    int out_w = netinfo->stages[0].output_shapes[out_idx_mask].dims[2];
    (void)out_h; (void)out_w;   // == H, W for a static bmodel

    // copy the four outputs to host
    auto read_tensor = [&](int idx) -> std::vector<float> {
        bm_tensor_t& t = output_tensors[idx];
        int count = bmrt_shape_count(&t.shape);
        std::vector<float> buf(count);
        bm_memcpy_d2s_partial(handle, buf.data(), t.device_mem, (size_t)count * sizeof(float));
        bm_free_device(handle, t.device_mem);
        return buf;
    };
    std::vector<float> points = read_tensor(out_idx_points);   // (1,H,W,3)
    std::vector<float> normal = read_tensor(out_idx_normal);   // (1,H,W,3)
    std::vector<float> mask_f  = read_tensor(out_idx_mask);    // (1,H,W)
    std::vector<float> scale   = read_tensor(out_idx_metric_scale);  // (1,)
    float metric_scale = scale[0];

    // binary mask
    result.mask.resize((size_t)H * W);
    for (int i = 0; i < H * W; i++)
        result.mask[i] = mask_f[i] > 0.5f ? 1 : 0;

    // recover focal/shift from the affine point map (same as the official repo)
    double focal, shift;
    recover_focal_shift(points.data(), result.mask.data(), H, W, focal, shift);

    double aspect = (double)W / H;
    double fx = focal / 2.0 * std::sqrt(1.0 + aspect * aspect) / aspect;
    double fy = focal / 2.0 * std::sqrt(1.0 + aspect * aspect);

    result.focal = (float)focal;
    result.shift = (float)shift;
    result.fx = (float)fx;
    result.fy = (float)fy;
    result.fov_x_deg = (float)(2.0 * std::atan(0.5 / fx) * 180.0 / kPi);
    result.fov_y_deg = (float)(2.0 * std::atan(0.5 / fy) * 180.0 / kPi);

    // depth (masked, metrically scaled) + reprojected points (pixel-center UV)
    result.depth.assign((size_t)H * W, 0.0f);
    result.points.assign((size_t)H * W * 3, 0.0f);
    result.depth_min = 1e30f;
    result.depth_max = -1e30f;
    for (int i = 0; i < H; i++) {
        for (int j = 0; j < W; j++) {
            size_t id = (size_t)i * W + j;
            const float* p = points.data() + id * 3;
            double z = (double)p[2] + shift;
            bool valid = result.mask[id] && z > 0.0;
            if (valid) {
                double u = (j + 0.5) / (double)W;
                double v = (i + 0.5) / (double)H;
                result.points[id * 3 + 0] = (float)((u - 0.5) / fx * z * metric_scale);
                result.points[id * 3 + 1] = (float)((v - 0.5) / fy * z * metric_scale);
                result.points[id * 3 + 2] = (float)(z * metric_scale);
                result.depth[id] = (float)(z * metric_scale);
                if (result.depth[id] < result.depth_min) result.depth_min = result.depth[id];
                if (result.depth[id] > result.depth_max) result.depth_max = result.depth[id];
            }
        }
    }
    if (result.depth_min > result.depth_max) { result.depth_min = 0.0f; result.depth_max = 0.0f; }

    // normal (already L2-normalized by the network)
    result.normal = std::move(normal);

    // NOTE: MoGe bmodels here are fp32/fp16 (graph outputs fp32, output_scales == 1.0),
    // so no output-scale dequantization is required.

    return 0;
}

int MoGe::Process(const bm_image& input, MoGeResult& result) {
    int ret = 0;
    bm_tensor_t input_tensor;
    std::vector<bm_tensor_t> output_tensors;
    output_tensors.resize(netinfo->output_num);
    std::fill(output_tensors.begin(), output_tensors.end(), bm_tensor_t());

    m_ts->save("moge preprocess");
    ret = pre_process(input, input_tensor);
    m_ts->save("moge preprocess");
    if (ret != 0) return ret;

    m_ts->save("moge inference");
    ret = forward(input_tensor, output_tensors);
    m_ts->save("moge inference");
    if (ret != 0) return ret;

    m_ts->save("moge postprocess");
    ret = post_process(output_tensors, result);
    m_ts->save("moge postprocess");
    return ret;
}