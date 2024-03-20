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

int main(int argc, char *argv[]){

    if (argc != 2) {
        std::cout << "USAGE:" << std::endl;
        std::cout << "  " << argv[0] << " <image_path>" << std::endl;
        exit(1);
    }
    std::string input_path = argv[1];
    int dev_id = 0;

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
    sail::BMImage crop_img(handle, input_image.height(), input_image.width(), FORMAT_RGB_PLANAR,
                               DATA_TYPE_EXT_1N_BYTE);

    int crop_x = input_image.width() / 4;
    int crop_y = input_image.height() / 4;
    int crop_w = input_image.width() / 2;
    int crop_h = input_image.height() / 2;
    // crop
#if USE_VPP
    if (!input_image.check_align()){
        input_image.align();
    }
    if (!crop_img.check_align()){
        crop_img.align();
    }
    ret = bmcv.vpp_crop(input_image, crop_img, crop_x, crop_y, crop_w, crop_h);
#else
    ret = bmcv.crop(input_image, crop_img, crop_x, crop_y, crop_w, crop_h);
#endif
    if (ret != 0) {
        cout << "[ERROR]crop failed." << "\n";
        exit(1);
    }

    bmcv.imwrite("./crop.jpg", crop_img);

    std::cout << "[PASS]All done." << std::endl;

    return 0;
}