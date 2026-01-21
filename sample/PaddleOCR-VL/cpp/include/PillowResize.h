/*
* Copyright 2021 Zuru Tech HK Limited
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* istributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

#ifndef PILLOWRESIZE_H
#define PILLOWRESIZE_H

#include <array>
#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

#ifdef _WIN32
#define _USE_MATH_DEFINES
#endif
#include <cmath>

#include <opencv2/opencv.hpp>

/**
 * \brief PillowResize 类，将 Pillow 库的 resize 方法移植到 OpenCV 上。
 * 实现仅依赖于 OpenCV，因此所有 Pillow 代码都已转换为使用 cv::Mat 和 OpenCV 结构。
 * 由于 Pillow 不支持所有 cv::Mat 类型，此实现扩展了对几乎所有 OpenCV 像素类型的支持。
 */
class PillowResize {
protected:
    /**
     * \brief precision_bits 8 位精度。滤波器可能有负值区域。
     * 在某些情况下，系数之和将为负值，或者大于 1.0。
     * 这就是为什么我们需要额外的两位用于溢出和整数类型。
     */
    static constexpr uint32_t precision_bits = 32 - 8 - 2;

    static constexpr double box_filter_support = 0.5;
    static constexpr double bilinear_filter_support = 1.;
    static constexpr double hamming_filter_support = 1.;
    static constexpr double bicubic_filter_support = 2.;
    static constexpr double lanczos_filter_support = 3.;

    /**
     * \brief Filter 抽象类，用于处理不同插值方法使用的滤波器。
     */
    class Filter {
    private:
        double _support; /** 支持区域大小（重采样滤波器的长度）。 */

    public:
        /**
         * \brief 构造函数。
         * 
         * \param[in] support 支持区域大小。
         */
        explicit Filter(double support) : _support{support} {};

        /**
         * \brief filter 应用滤波器。
         * 
         * \param[in] x 输入值。
         * 
         * \return 滤波后的值。
         */
        [[nodiscard]] virtual double filter(double x) const = 0;

        /**
         * \brief support 获取支持区域大小。
         * 
         * \return 支持区域大小。
         */
        [[nodiscard]] double support() const { return _support; };
    };

    class BoxFilter : public Filter {
    public:
        BoxFilter() : Filter(box_filter_support){};
        [[nodiscard]] double filter(double x) const override {
            constexpr double half_pixel = 0.5;
            if (x > -half_pixel && x <= half_pixel) {
                return 1.0;
            }
            return 0.0;
        }
    };

    class BilinearFilter : public Filter {
    public:
        BilinearFilter() : Filter(bilinear_filter_support){};
        [[nodiscard]] double filter(double x) const override {
            if (x < 0.0) {
                x = -x;
            }
            if (x < 1.0) {
                return 1.0 - x;
            }
            return 0.0;
        }
    };

    class HammingFilter : public Filter {
    public:
        HammingFilter() : Filter(hamming_filter_support){};
        [[nodiscard]] double filter(double x) const override {
            if (x < 0.0) {
                x = -x;
            }
            if (x == 0.0) {
                return 1.0;
            }
            if (x >= 1.0) {
                return 0.0;
            }
            x = x * M_PI;
            return sin(x) / x * (0.54 + 0.46 * cos(x));    // NOLINT
        }
    };

    class BicubicFilter : public Filter {
    public:
        BicubicFilter() : Filter(bicubic_filter_support){};
        [[nodiscard]] double filter(double x) const override {
            // https://en.wikipedia.org/wiki/Bicubic_interpolation#Bicubic_convolution_algorithm
            constexpr double a = -0.5;
            if (x < 0.0) {
                x = -x;
            }
            if (x < 1.0) {                                         // NOLINT
                return ((a + 2.0) * x - (a + 3.0)) * x * x + 1;    // NOLINT
            }
            if (x < 2.0) {                                 // NOLINT
                return (((x - 5) * x + 8) * x - 4) * a;    // NOLINT
            }
            return 0.0;
        }
    };

    class LanczosFilter : public Filter {
    protected:
        [[nodiscard]] static double _sincFilter(double x) {
            if (x == 0.0) {
                return 1.0;
            }
            x = x * M_PI;
            return sin(x) / x;
        }

    public:
        LanczosFilter() : Filter(lanczos_filter_support){};
        [[nodiscard]] double filter(double x) const override {
            // Truncated sinc.
            // According to Jim Blinn, the Lanczos kernel (with a = 3)
            // "keeps low frequencies and rejects high frequencies better
            // than any (achievable) filter we've seen so far."[3]
            // (https://en.wikipedia.org/wiki/Lanczos_resampling#Advantages)
            constexpr double lanczos_a_param = 3.0;
            if (-lanczos_a_param <= x && x < lanczos_a_param) {
                return _sincFilter(x) * _sincFilter(x / lanczos_a_param);
            }
            return 0.0;
        }
    };

#if __cplusplus >= 201703L
    /**
     * \brief _lut 生成查找表。
     * \reference https://joelfilho.com/blog/2020/compile_time_lookup_tables_in_cpp/
     * 
     * \tparam Length 表的元素数量。
     * \param[in] f 用于生成表中每个元素的函数对象。
     * 
     * \return 长度为 Length 的数组，类型由 Generator 的输出确定。
     */
    template <size_t Length, typename Generator>
    static constexpr auto _lut(Generator&& f) {
        using content_type = decltype(f(size_t{0}));
        std::array<content_type, Length> arr{};
        for (size_t i = 0; i < Length; ++i) {
            arr[i] = f(i);
        }
        return arr;
    }

    /**
     * \brief _clip8_lut Clip 查找表。
     * 
     * \tparam Length 表的元素数量。
     * \tparam min_val 表的起始元素值。
     */
    template <size_t Length, intmax_t min_val>
    static inline constexpr auto _clip8_lut =
        _lut<Length>([](size_t n) -> uint8_t {
            intmax_t saturate_val = static_cast<intmax_t>(n) + min_val;
            if (saturate_val < 0) {
                return 0;
            }
            if (saturate_val > UINT8_MAX) {
                return UINT8_MAX;
            }
            return static_cast<uint8_t>(saturate_val);
        });
#endif

    /**
     * \brief _clip8 优化的裁剪函数。
     * 
     * \param[in] in 输入值。
     * 
     * \return 裁剪后的值。
     */
    [[nodiscard]] static uint8_t _clip8(double in) {
#if __cplusplus >= 201703L
        // 使用查找表来加速裁剪方法。
        // 处理从 -640 到 639 的值。
        const uint8_t* clip8_lookups =
            &_clip8_lut<1280, -640>[640];    // NOLINT
        // NOLINTNEXTLINE
        return clip8_lookups[static_cast<intmax_t>(in) >> precision_bits];
#else
        auto saturate_val = static_cast<intmax_t>(in) >> precision_bits;
        if (saturate_val < 0) {
            return 0;
        }
        if (saturate_val > UINT8_MAX) {
            return UINT8_MAX;
        }
        return static_cast<uint8_t>(saturate_val);
#endif
    }

    /**
     * \brief _roundUp 取整函数。
     * 输出值将被转换为类型 T。
     *      
     * \param[in] f 输入值。
     * 
     * \return 四舍五入后的值。
     */
    template <typename T>
    [[nodiscard]] static T _roundUp(double f) {
        return static_cast<T>(std::round(f));
    }

    /**
     * \brief _getPixelType 返回矩阵元素的类型。
     * 如果矩阵具有多个通道，该函数返回元素的类型，而不包含通道信息。
     * 例如，如果类型为 CV_16SC3，该函数返回 CV_16S。
     * 
     * \param[in] img 输入图像。
     * 
     * \return 矩阵元素类型。
     */
    [[nodiscard]] static int32_t _getPixelType(const cv::Mat& img) {
        return img.type() & CV_MAT_DEPTH_MASK;    // NOLINT
    }

    /**
     * \brief _precomputeCoeffs 计算一维插值系数。
     * 如果有一个图像（或二维矩阵），调用该方法两次，以分别计算行和列的系数。
     * 系数针对范围 [0, out_size) 中的每个元素计算。
     * 
     * \param[in] in_size 输入大小（例如图像宽度或高度）。
     * \param[in] in0 输入起始索引。
     * \param[in] in1 输入结束索引。
     * \param[in] out_size 输出大小。
     * \param[in] filterp 滤波器对象的指针。
     * \param[out] bounds 边界向量。边界是 xmin 和 xmax 的一对值。
     * \param[out] kk 系数向量。每个元素对应函数返回的一组系数。
     * 
     * \return 滤波器系数的大小。
     */
    [[nodiscard]] static int32_t _precomputeCoeffs(
        int32_t in_size,
        double in0,
        double in1,
        int32_t out_size,
        const std::shared_ptr<Filter>& filterp,
        std::vector<int32_t>& bounds,
        std::vector<double>& kk) {
        // 准备水平拉伸。
        const double scale = (in1 - in0) / static_cast<double>(out_size);
        double filterscale = scale;
        if (filterscale < 1.0) {
            filterscale = 1.0;
        }

        // 确定支持区域大小（重采样滤波器的长度）。
        const double support = filterp->support() * filterscale;

        // 最大系数数量。
        const auto k_size = static_cast<int32_t>(ceil(support)) * 2 + 1;

        // 检查溢出
        if (out_size >
            INT32_MAX / (k_size * static_cast<int32_t>(sizeof(double)))) {
            throw std::runtime_error("Memory error");
        }

        // 系数缓冲区。
        kk.resize(out_size * k_size);

        // 边界向量。
        bounds.resize(out_size * 2);

        int32_t x = 0;
        constexpr double half_pixel = 0.5;
        for (int32_t xx = 0; xx < out_size; ++xx) {
            double center = in0 + (xx + half_pixel) * scale;
            double ww = 0.0;
            double ss = 1.0 / filterscale;
            // 取整值。
            auto xmin = static_cast<int32_t>(center - support + half_pixel);
            if (xmin < 0) {
                xmin = 0;
            }
            // 取整值。
            auto xmax = static_cast<int32_t>(center + support + half_pixel);
            if (xmax > in_size) {
                xmax = in_size;
            }
            xmax -= xmin;
            double* k = &kk[xx * k_size];
            for (x = 0; x < xmax; ++x) {
                double w = filterp->filter((x + xmin - center + half_pixel) * ss);
                k[x] = w;    // NOLINT
                ww += w;
            }
            for (x = 0; x < xmax; ++x) {
                if (ww != 0.0) {
                    k[x] /= ww;    // NOLINT
                }
            }
            // 如果 xmax 之后的值被使用，剩余的值应保持为空。
            for (; x < k_size; ++x) {
                k[x] = 0;    // NOLINT
            }
            bounds[xx * 2 + 0] = xmin;
            bounds[xx * 2 + 1] = xmax;
        }
        return k_size;
    }

    /**
     * \brief _normalizeCoeffs8bpc 为每像素 8 位的矩阵归一化系数。
     * 
     * \param[in] prekk 滤波器系数。
     * 
     * \return 归一化的滤波器系数。
     */
    [[nodiscard]] static std::vector<double> _normalizeCoeffs8bpc(
        const std::vector<double>& prekk) {
        std::vector<double> kk;
        kk.reserve(prekk.size());

        constexpr auto shifted_coeff = static_cast<double>(1U << precision_bits);

        constexpr double half_pixel = 0.5;
        for (const auto& k : prekk) {
            if (k < 0) {
                kk.emplace_back(trunc(-half_pixel + k * shifted_coeff));
            }
            else {
                kk.emplace_back(trunc(half_pixel + k * shifted_coeff));
            }
        }
        return kk;
    }

    /**
     * \brief _resampleHorizontal 在水平轴上应用重采样。
     * 根据 cv::Mat::type() 的返回值，调用具有正确像素类型的 _resampleHorizontal。
     * 
     * \param[in, out] im_out 输出调整大小的矩阵。
     *                        矩阵必须事先使用正确的大小进行初始化。
     * \param[in] im_in 输入矩阵。
     * \param[in] offset 垂直偏移量（源图像中第一个使用的行）。
     * \param[in] ksize 插值滤波器大小。
     * \param[in] bounds 插值滤波器边界（要考虑的最小和最大列的值）。
     * \param[in] prekk 插值滤波器系数。
     */
    static void _resampleHorizontal(cv::Mat& im_out,
                                    const cv::Mat& im_in,
                                    int32_t offset,
                                    int32_t ksize,
                                    const std::vector<int32_t>& bounds,
                                    const std::vector<double>& prekk) {
        // 检查像素类型并确定像素大小（元素大小 * 通道数）。
        switch (_getPixelType(im_in)) {
            case CV_8U:
                _resampleHorizontal<uint8_t>(
                    im_out, im_in, offset, ksize, bounds, prekk,
                    _normalizeCoeffs8bpc,
                    static_cast<double>(1U << (precision_bits - 1U)), _clip8);
                break;
            case CV_8S:
                _resampleHorizontal<int8_t>(im_out, im_in, offset, ksize,
                                            bounds, prekk, nullptr, 0.,
                                            _roundUp<int8_t>);
                break;
            case CV_16U:
                _resampleHorizontal<uint16_t>(im_out, im_in, offset, ksize,
                                              bounds, prekk);
                break;
            case CV_16S:
                _resampleHorizontal<int16_t>(im_out, im_in, offset, ksize,
                                             bounds, prekk, nullptr, 0.,
                                             _roundUp<int16_t>);
                break;
            case CV_32S:
                _resampleHorizontal<int32_t>(im_out, im_in, offset, ksize,
                                             bounds, prekk, nullptr, 0.,
                                             _roundUp<int32_t>);
                break;
            case CV_32F:
                _resampleHorizontal<float>(im_out, im_in, offset, ksize,
                                           bounds, prekk);
                break;
            default:
                throw std::runtime_error("Pixel type not supported");
        }
    }

    /**
     * \brief _resampleVertical 在垂直轴上应用重采样。
     * 根据 cv::Mat::type() 的返回值，调用具有正确像素类型的 _resampleVertical。
     * 
     * \param[in, out] im_out 输出调整大小的矩阵。
     *                        矩阵必须事先使用正确的大小进行初始化。
     * \param[in] im_in 输入矩阵。
     * \param[in] ksize 插值滤波器大小。
     * \param[in] bounds 插值滤波器边界（要考虑的最小和最大行的值）。
     * \param[in] prekk 插值滤波器系数。
     */
    static void _resampleVertical(cv::Mat& im_out,
                                  const cv::Mat& im_in,
                                  int32_t ksize,
                                  const std::vector<int32_t>& bounds,
                                  const std::vector<double>& prekk) {
        im_out = im_out.t();
        _resampleHorizontal(im_out, im_in.t(), 0, ksize, bounds, prekk);
        im_out = im_out.t();
    }

    using preprocessCoefficientsFn =
        std::vector<double> (*)(const std::vector<double>&);

    template <typename T>
    using outMapFn = T (*)(double);

    /**
     * \brief _resampleHorizontal 在水平轴上应用重采样。
     *      
     * \param[in, out] im_out 输出调整大小的矩阵。
     *                        矩阵必须事先使用正确的大小进行初始化。
     * \param[in] im_in 输入矩阵。
     * \param[in] offset 垂直偏移量（源图像中第一个使用的行）。
     * \param[in] ksize 插值滤波器大小。
     * \param[in] bounds 插值滤波器边界（滤波器要考虑的最小和最大像素的索引）。
     * \param[in] prekk 插值滤波器系数。
     * \param[in] preprocessCoefficients 用于处理滤波器系数的函数。
     * \param[in] init_buffer 像素缓冲区的初始值（默认值：0.0）。
     * \param[in] outMap 用于将插值后像素值转换为输出像素的函数。
     */
    template <typename T>
    static void _resampleHorizontal(
        cv::Mat& im_out,
        const cv::Mat& im_in,
        int32_t offset,
        int32_t ksize,
        const std::vector<int32_t>& bounds,
        const std::vector<double>& prekk,
        preprocessCoefficientsFn preprocessCoefficients = nullptr,
        double init_buffer = 0.,
        outMapFn<T> outMap = nullptr) {
        std::vector<double> kk(prekk.begin(), prekk.end());
        // 如果需要，预处理系数。
        if (preprocessCoefficients != nullptr) {
            kk = preprocessCoefficients(kk);
        }

        for (int32_t yy = 0; yy < im_out.size().height; ++yy) {
            for (int32_t xx = 0; xx < im_out.size().width; ++xx) {
                const int32_t xmin = bounds[xx * 2 + 0];
                const int32_t xmax = bounds[xx * 2 + 1];
                const double* k = &kk[xx * ksize];
                for (int32_t c = 0; c < im_in.channels(); ++c) {
                    double ss = init_buffer;
                    for (int32_t x = 0; x < xmax; ++x) {
                        // NOLINTNEXTLINE
                        ss += static_cast<double>(
                                  im_in.ptr<T>(yy + offset, x + xmin)[c]) *
                              k[x];
                    }
                    // NOLINTNEXTLINE
                    im_out.ptr<T>(yy, xx)[c] =
                        (outMap == nullptr ? static_cast<T>(ss) : outMap(ss));
                }
            }
        }
    }

    /**
     * \brief _resample 使用指定的插值方法调整矩阵大小。
     * 
     * \param[in] im_in 输入矩阵。
     * \param[in] x_size 期望的输出宽度。
     * \param[in] y_size 期望的输出高度。
     * \param[in] filter_p 插值滤波器的指针。
     * \param[in] rect 要调整大小的输入区域。
     *                 区域由四个点 x0, y0, x1, y1 定义的向量。
     * 
     * \return 调整大小的矩阵。矩阵的类型将与 im_in 相同。
     */
    [[nodiscard]] static cv::Mat _resample(
        const cv::Mat& im_in,
        int32_t x_size,
        int32_t y_size,
        const std::shared_ptr<Filter>& filter_p,
        const cv::Vec4f& rect) {
        cv::Mat im_out;
        cv::Mat im_temp;

        std::vector<int32_t> bounds_horiz;
        std::vector<int32_t> bounds_vert;
        std::vector<double> kk_horiz;
        std::vector<double> kk_vert;

        const bool need_horizontal = x_size != im_in.size().width ||
                                     (rect[0] != 0.0F) ||
                                     static_cast<int32_t>(rect[2]) != x_size;
        const bool need_vertical = y_size != im_in.size().height ||
                                   (rect[1] != 0.0F) ||
                                   static_cast<int32_t>(rect[3]) != y_size;

        // 计算水平滤波器系数。
        const int32_t ksize_horiz =
            _precomputeCoeffs(im_in.size().width, rect[0], rect[2], x_size,
                              filter_p, bounds_horiz, kk_horiz);

        // 计算垂直滤波器系数。
        const int32_t ksize_vert =
            _precomputeCoeffs(im_in.size().height, rect[1], rect[3], y_size,
                              filter_p, bounds_vert, kk_vert);

        // 源图像中第一个使用的行。
        const int32_t ybox_first = bounds_vert[0];
        // 源图像中最后一个使用的行。
        const int32_t ybox_last =
            bounds_vert[y_size * 2 - 2] + bounds_vert[y_size * 2 - 1];

        // 两次调整大小，水平通道。
        if (need_horizontal) {
            // 为垂直通道调整边界。
            for (int32_t i = 0; i < y_size; ++i) {
                bounds_vert[i * 2] -= ybox_first;
            }

            // 创建具有期望输出宽度和与输入像素类型相同的目标图像。
            im_temp.create(ybox_last - ybox_first, x_size, im_in.type());
            if (!im_temp.empty()) {
                _resampleHorizontal(im_temp, im_in, ybox_first, ksize_horiz,
                                    bounds_horiz, kk_horiz);
            }
            else {
                return cv::Mat();
            }
            im_out = im_temp;
        }

        // 垂直通道。
        if (need_vertical) {
            // 创建具有期望输出大小和与输入像素类型相同的目标图像。

            const auto new_w =
                (im_temp.size().width != 0) ? im_temp.size().width : x_size;
            im_out.create(y_size, new_w, im_in.type());
            if (!im_out.empty()) {
                if (im_temp.empty()) {
                    im_temp = im_in;
                }
                // 输入可以是原始图像或水平重采样的图像。
                _resampleVertical(im_out, im_temp, ksize_vert, bounds_vert,
                                  kk_vert);
            }
            else {
                return cv::Mat();
            }
        }

        // 没有执行任何前面的步骤，直接复制。
        if (im_out.empty()) {
            im_out = im_in;
        }

        return im_out;
    }

    /**
     * \brief _nearestResample 使用最近邻插值调整矩阵大小。
     * 
     * \param[in] im_in 输入矩阵。
     * \param[in] x_size 期望的输出宽度。
     * \param[in] y_size 期望的输出高度。
     * \param[in] rect 要调整大小的输入区域。
     *                 区域由四个点 x0, y0, x1, y1 定义的向量。
     * 
     * \return 调整大小的矩阵。矩阵的类型将与 im_in 相同。
     * 
     * \throws std::runtime_error 如果输入矩阵类型不受支持。
     */
    [[nodiscard]] static cv::Mat _nearestResample(const cv::Mat& im_in,
                                                  int32_t x_size,
                                                  int32_t y_size,
                                                  const cv::Vec4f& rect) {
        auto rx0 = static_cast<int32_t>(rect[0]);
        auto ry0 = static_cast<int32_t>(rect[1]);
        auto rx1 = static_cast<int32_t>(rect[2]);
        auto ry1 = static_cast<int32_t>(rect[3]);
        rx0 = std::max(rx0, 0);
        ry0 = std::max(ry0, 0);
        rx1 = std::min(rx1, im_in.size().width);
        ry1 = std::min(ry1, im_in.size().height);

        // 仿射变换矩阵。
        cv::Mat m = cv::Mat::zeros(2, 3, CV_64F);
        m.at<double>(0, 0) =
            static_cast<double>(rx1 - rx0) / static_cast<double>(x_size);
        m.at<double>(0, 2) = static_cast<double>(rx0);
        m.at<double>(1, 1) =
            static_cast<double>(ry1 - ry0) / static_cast<double>(y_size);
        m.at<double>(1, 2) = static_cast<double>(ry0);

        cv::Mat im_out = cv::Mat::zeros(y_size, x_size, im_in.type());

        // 检查像素类型并确定像素大小（元素大小 * 通道数）。
        size_t pixel_size = 0;
        switch (_getPixelType(im_in)) {
            case CV_8U:
                pixel_size = sizeof(uint8_t);
                break;
            case CV_8S:
                pixel_size = sizeof(int8_t);
                break;
            case CV_16U:
                pixel_size = sizeof(uint16_t);
                break;
            case CV_16S:
                pixel_size = sizeof(int16_t);
                break;
            case CV_32S:
                pixel_size = sizeof(int32_t);
                break;
            case CV_32F:
                pixel_size = sizeof(float);
                break;
            default:
                throw std::runtime_error("Pixel type not supported");
        }
        pixel_size *= im_in.channels();

        const int32_t x0 = 0;
        const int32_t y0 = 0;
        const int32_t x1 = x_size;
        const int32_t y1 = y_size;

        double xo = m.at<double>(0, 2) + m.at<double>(0, 0) * 0.5;
        double yo = m.at<double>(1, 2) + m.at<double>(1, 1) * 0.5;

        auto coord = [](double x) -> int32_t {
            return x < 0. ? -1 : static_cast<int32_t>(x);
        };

        std::vector<int> xintab;
        xintab.resize(im_out.size().width);

        /* 预先计算水平像素位置 */
        int32_t xmin = x1;
        int32_t xmax = x0;
        for (int32_t x = x0; x < x1; ++x) {
            int32_t xin = coord(xo);
            if (xin >= 0 && xin < im_in.size().width) {
                xmax = x + 1;
                if (x < xmin) {
                    xmin = x;
                }
                xintab[x] = xin;
            }
            xo += m.at<double>(0, 0);
        }

        for (int32_t y = y0; y < y1; ++y) {
            int32_t yi = coord(yo);
            if (yi >= 0 && yi < im_in.size().height) {
                for (int32_t x = xmin; x < xmax; ++x) {
                    memcpy(im_out.ptr(y, x), im_in.ptr(yi, xintab[x]), pixel_size);
                }
            }
            yo += m.at<double>(1, 1);
        }

        return im_out;
    }

public:
    /**
     * \brief InterpolationMethods 插值方法。
     *
     * \see https://pillow.readthedocs.io/en/stable/handbook/concepts.html#concept-filters.
     */
    enum InterpolationMethods {
        INTERPOLATION_NEAREST = 0,
        INTERPOLATION_BOX = 4,
        INTERPOLATION_BILINEAR = 2,
        INTERPOLATION_HAMMING = 5,
        INTERPOLATION_BICUBIC = 3,
        INTERPOLATION_LANCZOS = 1,
    };

    /**
     * \brief resize 移植 Pillow 的 resize 方法。
     * 
     * \param[in] src 要处理的输入矩阵。
     * \param[in] out_size 输出矩阵大小。
     * \param[in] filter 插值方法代码，请参阅 InterpolationMethods。
     * \param[in] box 输入 ROI。只有框内的元素会被调整大小。
     * 
     * \return 调整大小的矩阵。
     * 
     * \throw std::runtime_error 如果框无效、插值滤波器或输入矩阵类型不受支持。
     */
    [[nodiscard]] static cv::Mat resize(const cv::Mat& src,
                                        const cv::Size& out_size,
                                        int32_t filter,
                                        const cv::Rect2f& box) {
        // Box = x0,y0,w,h
        // Rect = x0,y0,x1,y1
        const cv::Vec4f rect(box.x, box.y, box.x + box.width, box.y + box.height);

        const int32_t x_size = out_size.width;
        const int32_t y_size = out_size.height;
        if (x_size < 1 || y_size < 1) {
            throw std::runtime_error("Height and width must be > 0");
        }

        if (rect[0] < 0.F || rect[1] < 0.F) {
            throw std::runtime_error("Box offset can't be negative");
        }

        if (static_cast<int32_t>(rect[2]) > src.size().width ||
            static_cast<int32_t>(rect[3]) > src.size().height) {
            throw std::runtime_error("Box can't exceed original image size");
        }

        if (box.width < 0 || box.height < 0) {
            throw std::runtime_error("Box can't be empty");
        }

        // 如果框的坐标是整数并且框大小与请求的大小匹配
        if (static_cast<int32_t>(box.width) == x_size &&
            static_cast<int32_t>(box.height) == y_size) {
            cv::Rect roi = box;
            return cv::Mat(src, roi);
        }
        if (filter == INTERPOLATION_NEAREST) {
            return _nearestResample(src, x_size, y_size, rect);
        }
        std::shared_ptr<Filter> filter_p;

        // 检查滤波器。
        switch (filter) {
            case INTERPOLATION_BOX:
                filter_p = std::make_shared<BoxFilter>(BoxFilter());
                break;
            case INTERPOLATION_BILINEAR:
                filter_p = std::make_shared<BilinearFilter>(BilinearFilter());
                break;
            case INTERPOLATION_HAMMING:
                filter_p = std::make_shared<HammingFilter>(HammingFilter());
                break;
            case INTERPOLATION_BICUBIC:
                filter_p = std::make_shared<BicubicFilter>(BicubicFilter());
                break;
            case INTERPOLATION_LANCZOS:
                filter_p = std::make_shared<LanczosFilter>(LanczosFilter());
                break;
            default:
                throw std::runtime_error("unsupported resampling filter");
        }

        return PillowResize::_resample(src, x_size, y_size, filter_p, rect);
    }

    /**
     * \brief resize 移植 Pillow 的 resize 方法。
     * 
     * \param[in] src 要处理的输入矩阵。
     * \param[in] out_size 输出矩阵大小。
     * \param[in] filter 插值方法代码，请参阅 InterpolationMethods。
     * 
     * \return 调整大小的矩阵。
     * 
     * \throw std::runtime_error 如果框无效、插值滤波器或输入矩阵类型不受支持。
     */
    [[nodiscard]] static cv::Mat resize(const cv::Mat& src,
                                        const cv::Size& out_size,
                                        int32_t filter) {
        cv::Rect2f box(0.F, 0.F, static_cast<float>(src.size().width),
                       static_cast<float>(src.size().height));
        return resize(src, out_size, filter, box);
    }
};

#endif