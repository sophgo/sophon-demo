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
    int output_width = 234;
    int output_height = 234;

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

    // resize
#if USE_VPP
    if (!input_image.check_align()){
        input_image.align();
    }
    if (!resized_img.check_align()){
        resized_img.align();
    }
    ret = bmcv.vpp_resize(input_image, resized_img, output_width, output_height, BMCV_INTER_LINEAR);
#else
    ret = bmcv.resize(input_image, resized_img, output_width, output_height, BMCV_INTER_LINEAR);
#endif
    if (ret != 0) {
        cout << "[ERROR]resize failed." << "\n";
        exit(1);
    }

    bmcv.imwrite("./resized.jpg", resized_img);

    std::cout << "[PASS]All done." << std::endl;

    return 0;
}