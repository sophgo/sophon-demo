#ifndef DPU_H
#define DPU_H

#include <memory>
#include <string>
#include <bmcv_api_ext.h>
#include <bmlib_runtime.h>
#include "utils.hpp"

// 如果FFALIGN已经定义，先取消定义
#ifdef FFALIGN
#undef FFALIGN
#endif
#define FFALIGN(x, a) (((x) + (a) - 1) & ~((a) - 1))

/**
 * @brief DPU处理模式
 */
enum class DPUMode {
    SGBM = 0,    // SGBM算法
    ONLINE = 1   // 在线处理
};

/**
 * @brief SGBM输出模式
 */
enum class SGBMMode {
    MUX0 = 1,    // 8位未后处理视差图
    MUX1 = 2,    // 16位后处理视差图
    MUX2 = 3     // 8位后处理视差图
};

/**
 * @brief Online输出模式
 */
enum class OnlineMode {
    MUX0 = 4,    // FGS处理，输出8位视差图
    MUX1 = 5,    // SGBM+FGS处理，输出16位深度图
    MUX2 = 6     // SGBM处理，输出16位深度图
};

/**
 * @brief DPU类 - 深度处理单元
 * 用于处理双目视觉的深度估计
 */
class DPU {
public:
    // 构造函数和析构函数
    DPU(int dev_id, bool debug, int width, int height);
    ~DPU();

    int align_to_width_ = 32; // 宽的stride需要对齐到32的倍数
    int align_to_height_ = 2; // 高的stride需要对齐到2的倍数

    // 预处理函数
    int pre_process(bm_image& input_image, bm_image& preprocessed_image);
    
    // 处理函数
    int process(bm_image& left_img, bm_image& right_img, 
                bm_image& depth_img, DPUMode mode = DPUMode::ONLINE,
                bmcv_dpu_sgbm_mode sgbm_mode = DPU_SGBM_MUX0,
                bmcv_dpu_online_mode online_mode = DPU_ONLINE_MUX0);

    // 保存图像
    bool save_image(bm_image& img, const std::string& output_path);

    TimeStamp* m_ts;

private:
    // 参数验证函数
    bool validateImageFormat(const bm_image& img, bool is_input = true) const;
    bool validateImageSize(const bm_image& left, const bm_image& right) const;
    bool validateDisparityRange(const bm_image& right_img) const;

    // 成员变量
    bm_handle_t handle_;
    bmcv_dpu_sgbm_attrs sgbm_params_;
    bmcv_dpu_fgs_attrs fgs_params_;
    bool initialized_;
    bool debug_;

    int width_;
    int height_;
    bm_image aligned_left_;
    bm_image aligned_right_;
    bm_image depth_img_;

    void release() noexcept;
};

#endif // DPU_H 