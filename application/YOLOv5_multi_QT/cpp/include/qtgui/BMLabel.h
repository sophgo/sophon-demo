// ===----------------------------------------------------------------------===
// 
//  Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
// 
//  SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
//  third-party components.
// 
// ===----------------------------------------------------------------------===

#ifndef BMIMG_LABEL_H_
#define BMIMG_LABEL_H_

#include <iostream>
#include <QApplication>
#include <QWidget>
#include <QLabel>
#include <QPainter>
#include <QRect>
#include <QGridLayout>
#include <QDebug>
#include <vector>
#include <memory>
#include <unordered_map>
#include <mutex>
#include <thread>
#include "bmcv_api.h"
#include "bmcv_api_ext.h"
#include "bmlib_runtime.h"
#include "datapipe.h"
#include "opencv2/opencv.hpp"
#include "common.h"

/*
    BmimgLabel类继承了QLabel,具备QLabel的基本显示功能，在此基础拓展了show_bmimg的显示接口，传入std::shared_ptr<bm_image>即可显示其中的图片 
    注意：智能指针要自定义删除器，删除器内完成bm_image_destroy操作，否则会存在内存泄露。
*/

class BMLabel : public QLabel{
    Q_OBJECT
public:

    explicit BMLabel(int dev_id=0, QWidget *parent=nullptr, Qt::WindowFlags f=Qt::WindowFlags()){
        bm_dev_request(&handle,dev_id);
        connect(this, &BMLabel::show_signals,this,&BMLabel::show_pixmap);
    }
    ~BMLabel() override {
        bm_dev_free(handle);
    }

    void show_img(std::shared_ptr<bm_image> bmimg_ptr);
    void show_img(std::shared_ptr<cv::Mat> cvmat_ptr);
    void show_img(std::shared_ptr<FrameInfoDetect> frameinfo);

protected:
    void paintEvent(QPaintEvent *event);

public slots:

    void show_pixmap();

signals:
    void show_signals();

private:
    QPixmap image_pixmap;
    std::vector<QRect> rects;
    bm_handle_t handle;
};

#endif 