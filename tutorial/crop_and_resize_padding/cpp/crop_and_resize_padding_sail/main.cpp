//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include <fstream>
#include <iostream>

#include "cvwrapper.h"

#ifndef USE_VPP
#define USE_VPP 1
#endif

bool is_file_exists(const string& filename) {
    ifstream file(filename);
    return (file.good()); 
}

float get_aspect_scaled_ratio(int src_w, int src_h, int dst_w, int dst_h, bool *pIsAligWidth)
{
  float ratio;
  float r_w = (float)dst_w / src_w;
  float r_h = (float)dst_h / src_h;
  if (r_h > r_w){
    *pIsAligWidth = true;
    ratio = r_w;
  }
  else{
    *pIsAligWidth = false;
    ratio = r_h;
  }
  return ratio;
}

int main(int argc, char *argv[]){

    if (argc != 2) {
        std::cout << "USAGE:" << std::endl;
        std::cout << "  " << argv[0] << " <image_path>" << std::endl;
        exit(1);
    }
    std::string input_path = argv[1];
    int dev_id = 0;
    int output_w = 640;
    int output_h = 640;

    if (!is_file_exists(input_path)){
        std::cout << "[ERROR]" << input_path << " is not existed." << std::endl;
        exit(1);
    }

    // init params
    auto handle = sail::Handle(dev_id);
    sail::Bmcv bmcv(handle);

    // init bmimage
    sail::BMImage input_image;
    sail::Decoder decoder(input_path, true, dev_id);
    int ret = decoder.read(handle, input_image);
    if (ret != 0) {
        cout << "[ERROR]read failed" << std::endl;
        exit(1);
    }
    sail::BMImage resized_img(handle, input_image.height(), input_image.width(), FORMAT_RGB_PLANAR,
                               DATA_TYPE_EXT_1N_BYTE);

    bool isAlignWidth = false;
    float ratio = get_aspect_scaled_ratio(input_image.width(), input_image.height(), output_w, output_h, &isAlignWidth);
    sail::PaddingAtrr pad = sail::PaddingAtrr();
    pad.set_r(114);
    pad.set_g(114);
    pad.set_b(114);
    if (isAlignWidth) {
        unsigned int th = input_image.height() * ratio;
        pad.set_h(th);
        pad.set_w(output_w);
        int ty1 = (int)((output_h - th) / 2);
        pad.set_sty(ty1);
        pad.set_stx(0);
    } else {
        pad.set_h(output_h);
        unsigned int tw = input_image.width() * ratio;
        pad.set_w(tw);

        int tx1 = (int)((output_w - tw) / 2);
        pad.set_sty(0);
        pad.set_stx(tx1);
    }


    // crop
#if USE_VPP
    if (!input_image.check_align()){
        input_image.align();
    }
    if (!resized_img.check_align()){
        resized_img.align();
    }
    
    ret = bmcv.vpp_crop_and_resize_padding(input_image, resized_img, 0, 0, 
            input_image.width(), input_image.height(), output_w, output_h, pad, BMCV_INTER_NEAREST);

#else
    ret = bmcv.crop_and_resize_padding(input_image, resized_img, 0, 0, 
            input_image.width(), input_image.height(), output_w, output_h, pad, BMCV_INTER_NEAREST)

#endif
    if (ret != 0) {
        cout << "[ERROR]vpp_crop_and_resize_padding failed." << "\n";
        exit(1);
    }

    bmcv.imwrite("./crop_and_resize_padding.jpg", resized_img);

    std::cout << "[PASS]All done." << std::endl;

    return 0;
}