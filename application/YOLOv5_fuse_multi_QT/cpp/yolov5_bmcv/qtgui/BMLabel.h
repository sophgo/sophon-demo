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
#include "opencv2/opencv.hpp"

class BMLabel : public QLabel{
    Q_OBJECT
public:

    explicit BMLabel(int dev_id=0, QWidget *parent=nullptr, Qt::WindowFlags f=Qt::WindowFlags()){
        connect(this, &BMLabel::show_signals,this,&BMLabel::setPixmap); //信号传值或引用到槽函数是线程安全的
    }
    
    virtual ~BMLabel() override {
    }

    void show_img(std::shared_ptr<cv::Mat> cvmat_ptr);

protected:
    void paintEvent(QPaintEvent *event);

public slots:
    void setPixmap(const QPixmap &pixmap) {
        QLabel::setPixmap(pixmap); 
        this->update();
    }

signals:
    void show_signals(const QPixmap &pixmap);

private:
    std::vector<QRect> rects;
};

#endif 