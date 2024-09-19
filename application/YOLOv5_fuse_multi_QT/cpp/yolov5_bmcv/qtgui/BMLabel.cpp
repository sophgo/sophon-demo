// ===----------------------------------------------------------------------===
// 
//  Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
// 
//  SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
//  third-party components.
// 
// ===----------------------------------------------------------------------===

#include "BMLabel.h"

void BMLabel::show_img(std::shared_ptr<cv::Mat> cvmat_ptr){

    int label_width = this->width();
    int label_height = this->height();
    cv::Mat o_mat;
    cv::resize(*cvmat_ptr,o_mat,cv::Size(label_width,label_height));

    cv::Mat rgb_mat;
    cv::cvtColor(o_mat, rgb_mat, cv::COLOR_BGR2RGB);

    // 公版qt版本>5.14，qimage才能支持bgr显示，但rgb都支持
    QImage _image((uchar *)rgb_mat.data, label_width, label_height, rgb_mat.step, QImage::Format_RGB888);

    if (_image.isNull()) {
        std::cerr << "Error: QImage creation failed." << std::endl;
        exit(-1);
    }

    emit BMLabel::show_signals(QPixmap::fromImage(_image));//异步接口，注意线程安全
}

void BMLabel::paintEvent(QPaintEvent *event){
    QLabel::paintEvent(event);
    QPainter painter(this);
    painter.setPen(QPen(Qt::red,2));
    if(rects.size()>0){
        painter.drawRects(rects.data(),rects.size());
    }
    painter.end();
}