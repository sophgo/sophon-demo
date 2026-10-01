// clang-format off
// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Tencent is pleased to support the open source community by making WeChat QRCode available.
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.

//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#ifndef __OPENCV_WECHAT_QRCODE_BINARIZERMGR_HPP__
#define __OPENCV_WECHAT_QRCODE_BINARIZERMGR_HPP__

#include "zxing/binarizer.hpp"
#include "zxing/common/binarizer/adaptive_threshold_mean_binarizer.hpp"
#include "zxing/common/counted.hpp"
#include "zxing/common/binarizer/fast_window_binarizer.hpp"
#include "zxing/common/binarizer/hybrid_binarizer.hpp"
#include "zxing/common/binarizer/simple_adaptive_binarizer.hpp"
#include "zxing/zxing.hpp"

namespace wechat_qrcode {
class BinarizerMgr {
public:
    enum BINARIZER {
        Hybrid = 0,
        FastWindow = 1,
        SimpleAdaptive = 2,
        AdaptiveThreshold = 3
    };

public:
    BinarizerMgr();
    ~BinarizerMgr();

    zxing::Ref<zxing::Binarizer> Binarize(zxing::Ref<zxing::LuminanceSource> source);

    void SwitchBinarizer();

    int GetCurBinarizer();

    void SetNextOnceBinarizer(int iBinarizerIndex);

    void SetBinarizer(vector<BINARIZER> vecRotateBinarizer);

private:
    int m_iNowRotateIndex;
    int m_iNextOnceBinarizer;
    vector<BINARIZER> m_vecRotateBinarizer;
};
}  // namespace wechat_qrcode
#endif  // __OPENCV_WECHAT_QRCODE_BINARIZERMGR_HPP__
// clang-format on
