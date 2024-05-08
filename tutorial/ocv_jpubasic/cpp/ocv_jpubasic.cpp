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
#include <streambuf>
#include <stdio.h>
#ifndef _WIN32
#include <sys/time.h>
#endif

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

using namespace std;
using namespace cv;

int dumpFile(char *filename, cv::Mat & image);
int dumpFile(char *filename, cv::Mat & image){

#if (defined HAVE_BMCV)

    if(!image.avOK()){
        if(image.channels() > 3) return -1;

        if (image.type() == CV_8UC3) bmcv::dumpMat(image, string(filename)+".BGR");
        else if (image.type() == CV_8UC1) bmcv::dumpMat(image, string(filename)+".GRAY");
    }else
        bmcv::dumpMat(image, string(filename) +".YUV");
#endif

    return 0;
}

int main(int argc, char *argv[])
{
#ifdef _WIN32
    clock_t start, end;
#else
    struct timeval start;
    struct timeval end;
#endif
    int card = 0;
    double t;
    int yuv_enable = 0;
    int dump_enable = 0;
    int flags = -1;

    if (argc < 5) {
        cout << "usage:" << endl;
        cout << "\t" << argv[0] << " <file> <loop> <yuv_enable> <dump_enable> [card]" << endl;
        cout << "params:" << endl;
        cout << "\t" << "<yuv_enable>: 0 decode output BGR; 1 decode output YUV." << endl;
        cout << "\t" << "<dump_enable>: 0 no dump file; 1 output dump file." << endl;
        exit(1);
    }

    yuv_enable = atoi(argv[3]);
    if(yuv_enable == 1) {
        flags = cv::IMREAD_AVFRAME;
    }else if(yuv_enable == 0) {
        flags = cv::IMREAD_COLOR;
    }else if(yuv_enable == 2) {
        flags = cv::IMREAD_UNCHANGED;
    }else{
        cout << "yuv_enable param err." << endl;
        return -1;
    }

    dump_enable = atoi(argv[4]);
    if((dump_enable != 1) && (dump_enable != 0)){
        cout << "dump_enable param err." << endl;
        return -1;
    }

    if (argc == 6) {
        card = atoi(argv[5]);
    }

    string file(argv[1]);
    ifstream in(argv[1], std::ifstream::binary);
    if (!in.good()) {
        cout << "file '" << argv[1] << "' doesn't exist" << endl;
        exit(2);
    }

    int loop = atoi(argv[2]);
    string s((istreambuf_iterator<char>(in)), istreambuf_iterator<char>());
    in.close();

    // Test case 1
    cout << "Test case 1" << endl;
    std::vector<char> pic(s.c_str(), s.c_str() + s.length());

    cv::Mat image;
#ifdef _WIN32
    start = clock();
#else
    gettimeofday(&start, NULL);
#endif

    try{
        for (int i = 0; i < loop; i++) {
            cv::imdecode(pic, flags, &image, card);
        }
    }
    catch (const cv::Exception& e){
        std::cerr << "imdecode failed: " << e.what() << std::endl;
        exit(-1);
    }

#ifdef _WIN32
    end = clock();
    t = end - start;
#else
    gettimeofday(&end, NULL);
    t = (end.tv_sec*1000.0 + end.tv_usec/1000.0) - (start.tv_sec*1000.0 + start.tv_usec/1000.0);
#endif
    cout << "decoder time(ms): " << t/loop << endl;
    try{
        cv::Mat image;
        cv::imdecode(pic, flags, &image, card);

        if (image.empty())
            cout << "Warning! decoded image is empty" << endl;
        else {
            if(dump_enable == 1){
                dumpFile("dump", image);
            }
            cv::imwrite("out1.jpg", image);
        }
    }
    catch (const cv::Exception& e){
        std::cerr << "imdecode failed: " << e.what() << std::endl;
        exit(-1);
    }

    // Test case 2
    cout << endl << "Test case 2" << endl;
    // save.cols is width; save.rows is height
    cv::Mat save = cv::imread(argv[1], flags, card);
    if (save.empty())
        cout << "Warning! decoded image is empty" << endl;
    else
        cv::imwrite("out2.jpg", save);

    // Test case 3
    cout << endl << "Test case 3" << endl;
    cv::Mat new_mat = cv::imread(argv[1], flags, card);
    if (new_mat.empty())
    {
        cout << "Warning! decoded image is empty" << endl;
        return 0;
    }

    vector<uchar> encoded;

#ifdef _WIN32
    start = clock();
#else
    gettimeofday(&start, NULL);
#endif

    try{
        for (int i = 0; i < loop; i++) {
            cv::imencode(".jpg", new_mat, encoded);
        }
    }
    catch (const cv::Exception& e){
        std::cerr << "imencode failed: " << e.what() << std::endl;
        exit(-1);
    }

#ifdef _WIN32
    end = clock();
    t = end - start;
#else
    gettimeofday(&end, NULL);
    t = (end.tv_sec*1000.0 + end.tv_usec/1000.0) - (start.tv_sec*1000.0 + start.tv_usec/1000.0);
#endif
    cout << "encoder time(ms): " << t/loop << endl;

    int bufLen = encoded.size();
    if (bufLen){
        unsigned char* pYuvBuf = encoded.data();
        FILE * fclr = fopen("out3.jpg", "wb");
        fwrite( pYuvBuf, 1, bufLen, fclr);
        fclose(fclr);
    }

    return 0;
}

