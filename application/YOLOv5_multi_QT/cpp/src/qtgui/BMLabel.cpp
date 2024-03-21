// ===----------------------------------------------------------------------===
// 
//  Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
// 
//  SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
//  third-party components.
// 
// ===----------------------------------------------------------------------===

#include "BMLabel.h"

void BMLabel::show_img(std::shared_ptr<bm_image> bmimg_ptr){

    int label_width = this->width();
    int label_height = this->height();
    bm_image convert_bmimg;
    bm_image_create(handle,bmimg_ptr->height,bmimg_ptr->width,FORMAT_BGR_PACKED, DATA_TYPE_EXT_1N_BYTE,&convert_bmimg);
    
    int stride[3] = {label_width*3,0,0};
    bm_image resize_bmimg;
    bm_image_create(handle,label_height,label_width,FORMAT_BGR_PACKED, DATA_TYPE_EXT_1N_BYTE,&resize_bmimg,stride);
    
    bmcv_resize_image resize_attr[4];
    bmcv_resize_t resize_img_attr[1];
    resize_img_attr[0].start_x = 0;
    resize_img_attr[0].start_y = 0;
    resize_img_attr[0].in_width = bmimg_ptr->width;
    resize_img_attr[0].in_height = bmimg_ptr->height;
    resize_img_attr[0].out_width = label_width;
    resize_img_attr[0].out_height = label_height;
    resize_attr[0].resize_img_attr = &resize_img_attr[0];
    resize_attr[0].roi_num = 1;
    resize_attr[0].stretch_fit = 1;
    resize_attr[0].interpolation = BMCV_INTER_NEAREST;

    bmcv_image_storage_convert(handle,1,bmimg_ptr.get(),&convert_bmimg);
    bmcv_image_resize(handle,1,resize_attr,&convert_bmimg,&resize_bmimg);

    cv::Mat o_mat;
    int ret = cv::bmcv::toMAT((bm_image *) &resize_bmimg, o_mat, true);
    QImage _image((uchar *)o_mat.data, label_width, label_height, o_mat.step, QImage::Format_BGR888);
    image_pixmap = QPixmap::fromImage(_image);
    emit BMLabel::show_signals();
    bm_image_destroy(convert_bmimg);
    bm_image_destroy(resize_bmimg);

}

void BMLabel::show_img(std::shared_ptr<cv::Mat> cvmat_ptr){

    int label_width = this->width();
    int label_height = this->height();
    cv::Mat o_mat;
    cv::resize(*cvmat_ptr,o_mat,cv::Size(label_width,label_height));

    QImage _image((uchar *)o_mat.data, label_width, label_height, o_mat.step, QImage::Format_BGR888);
    image_pixmap = QPixmap::fromImage(_image);
    emit BMLabel::show_signals();

}


void BMLabel::show_img(std::shared_ptr<FrameInfoDetect> frameinfo_ptr){

    std::shared_ptr<bm_image> bmimg_ptr = frameinfo_ptr->image_ptr;
    int label_width = this->width();
    int label_height = this->height();
    bm_image convert_bmimg;
    bm_image_create(handle,bmimg_ptr->height,bmimg_ptr->width,FORMAT_BGR_PACKED, DATA_TYPE_EXT_1N_BYTE,&convert_bmimg);
    
    int stride[3] = {label_width*3,0,0};
    bm_image resize_bmimg;
    bm_image_create(handle,label_height,label_width,FORMAT_BGR_PACKED, DATA_TYPE_EXT_1N_BYTE,&resize_bmimg,stride);
    
    bmcv_resize_image resize_attr[4];
    bmcv_resize_t resize_img_attr[1];
    resize_img_attr[0].start_x = 0;
    resize_img_attr[0].start_y = 0;
    resize_img_attr[0].in_width = bmimg_ptr->width;
    resize_img_attr[0].in_height = bmimg_ptr->height;
    resize_img_attr[0].out_width = label_width;
    resize_img_attr[0].out_height = label_height;
    resize_attr[0].resize_img_attr = &resize_img_attr[0];
    resize_attr[0].roi_num = 1;
    resize_attr[0].stretch_fit = 1;
    resize_attr[0].interpolation = BMCV_INTER_NEAREST;

    bmcv_image_storage_convert(handle,1,bmimg_ptr.get(),&convert_bmimg);
    bmcv_image_resize(handle,1,resize_attr,&convert_bmimg,&resize_bmimg);

    cv::Mat o_mat;
    int ret = cv::bmcv::toMAT((bm_image *) &resize_bmimg, o_mat, true);
    QImage _image((uchar *)o_mat.data, label_width, label_height, o_mat.step, QImage::Format_BGR888);
    image_pixmap = QPixmap::fromImage(_image);
   
    float w_ratio = 1.0*label_width/bmimg_ptr->width;
    float h_ratio = 1.0*label_height/bmimg_ptr->height;
    rects.clear();
    for(int i=0;i<frameinfo_ptr->boxs_vec.size();i++){
        rects.emplace_back(ceil(frameinfo_ptr->boxs_vec[i].x*w_ratio),ceil(frameinfo_ptr->boxs_vec[i].y*h_ratio),
                            floor(frameinfo_ptr->boxs_vec[i].width*w_ratio),floor(frameinfo_ptr->boxs_vec[i].height*h_ratio));
    }
    emit BMLabel::show_signals();
    bm_image_destroy(convert_bmimg);
    bm_image_destroy(resize_bmimg);

}


void BMLabel::show_pixmap(){
    this->setPixmap(image_pixmap);
    this->update(); 
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