//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <iostream>
#include <stdio.h>
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <sys/time.h>
#endif
#include <unistd.h>

#define MAX_READ_TIMEOUT 1000*30*50 // 30s
using namespace cv;
using namespace std;

int main(int argc, char* argv[])
{
    FILE *dumpfile = NULL;
    int yuv_enable = 0;
    int card = 0;

    //--- INITIALIZE VIDEOCAPTURE
    VideoCapture cap;
#ifdef WIN32
    clock_t tv1;
    clock_t tv2;
#else
    struct timeval tv1, tv2;
#endif
#ifndef USING_SOC
    int max_argc = 8;
#else
    int max_argc = 7;
#endif
    int arg_idx = 5;
    if (argc < 5) {
        cout << "usage:" << endl;
        cout << "\t" << argv[0] << " <input_video> <output_name> <frame_num> <yuv_enable> ";
#ifndef USING_SOC
        cout << "[card] ";
#endif
        cout << "[WxH] [dump.BGR or dump.YUV]" <<endl;
        cout << "\t--> test video record as png or jpg(enable yuv). And dump BGR or YUV raw data if you enable the dump." <<endl;
        cout << "params:" << endl;
        cout << "\t" << "<input_video>:           input video path." <<endl;
        cout << "\t" << "<output_name>:           output image name." <<endl;
        cout << "\t" << "<frame_num>:             the number of frames that need to be decoded." <<endl;
        cout << "\t" << "<yuv_enable>:            0 decode output BGR; 1 decode output YUV." <<endl;
#ifndef USING_SOC
        cout << "\t" << "<card>:                  device id." <<endl;
#endif
        cout << "\t" << "<WxH>:                   decoded image width and height." <<endl;
        cout << "\t" << "<dump.BGR or dump.YUV>:  dump.BGR dump BGR file; dump.BGR dump YUV file, this parameter is optional." <<endl;
        return -1;
    }
#ifndef USING_SOC
    if (argc >= 6) {
       card = atoi(argv[5]);
       arg_idx += 1;
    }
#endif
    yuv_enable = atoi(argv[4]);
    if ((yuv_enable != 0) && (yuv_enable != 1)) {
        cout << "yuv_enable param err." << endl;
        return -1;
    }

    // open the default camera using default API
    cap.open(argv[1], CAP_FFMPEG, card);
    if (!cap.isOpened()) {
        cerr << "ERROR! Unable to open camera\n";
        return -1;
    }

    // Set Resampler
    cap.set(CAP_PROP_OUTPUT_SRC, 1.0);
    double out_src = cap.get(CAP_PROP_OUTPUT_SRC);
    cout << "CAP_PROP_OUTPUT_SRC: " << out_src << endl;

//    cap.set(CAP_PROP_FOURCC, VideoWriter::fourcc('M', 'J', 'P', 'G'));

    // Set scalar size
    cout << "orig CAP_PROP_FRAME_HEIGHT: " << (int) cap.get(CAP_PROP_FRAME_HEIGHT) << endl;
    cout << "orig CAP_PROP_FRAME_WIDTH: " << (int) cap.get(CAP_PROP_FRAME_WIDTH) << endl;

    cout << "orig CAP_PROP_FRAME_COUNT: " << (int) cap.get(CAP_PROP_FRAME_COUNT) << endl;
    cout << "orig CAP_PROP_FPS: " << (int) cap.get(CAP_PROP_FPS) << endl;

    if (yuv_enable == 1){
        cap.set(cv::CAP_PROP_OUTPUT_YUV, PROP_TRUE);
    }

    if (argc >= max_argc - 1) {
        int w, h, ret;
        ret = sscanf(argv[arg_idx], "%dx%d", &w, &h);
        arg_idx += 1;
        if (ret != 2 || w > 8192 || h > 8192) {
            cout << "ret: " << ret << ", width: " << w << ", height: " << h << endl;
            cout << "width or height wrong. please check!" << endl;
            cap.release();
            return -1;
        }

        cap.set(CAP_PROP_FRAME_HEIGHT, (double)h);
        cap.set(CAP_PROP_FRAME_WIDTH, (double)w);

        cout << "CAP_PROP_FRAME_HEIGHT: " << (int) cap.get(CAP_PROP_FRAME_HEIGHT) << endl;
        cout << "CAP_PROP_FRAME_WIDTH: " << (int) cap.get(CAP_PROP_FRAME_WIDTH) << endl;
    }

    if (argc == max_argc) {
        if (strstr(argv[arg_idx], ".BGR") != NULL || strstr(argv[arg_idx], ".YUV") != NULL)
            dumpfile = fopen(argv[arg_idx], "wb");
        arg_idx += 1;
    }

    //--- GRAB AND WRITE LOOP
#ifdef WIN32
    tv1 = clock();
#else
    gettimeofday(&tv1, NULL);
#endif
    Mat image;
    int read_times = MAX_READ_TIMEOUT;
    int i_frame_nums = 0;
    while(true)
    {
        if (i_frame_nums >= atoi(argv[3])) {
            break;
        }
        if (read_times <= 0) {
            break;
        }
        // wait for a new frame from camera and store it into 'frame'
        cap.read(image);

        // check if we succeeded
        if (image.empty()) {
            cerr << "ERROR! blank frame grabbed\n";
            if ((int)cap.get(CAP_PROP_STATUS) == 2) {     // eof
                cout << "file ends!" << endl;
                cap.release();

                cap.open(argv[1], CAP_FFMPEG, card);
                cout << "loop again " << endl;
            }
            read_times--;
            usleep(10);
            continue;
        }
        read_times = MAX_READ_TIMEOUT;

        // show live and wait for a key with timeout long enough to show images
        if (dumpfile && yuv_enable)    // YUV420P
        {
#ifdef HAVE_BMCV
            bmcv::downloadMat(image);
#endif
            for (int i = 0; i < image.avRows(); i++)
            {
                fwrite((char*)image.avAddr(0)+i*image.avStep(0),1,image.avCols(),dumpfile);
            }
            for (int i = 0; i < image.avRows()/2; i++)
            {
                fwrite((char*)image.avAddr(1)+i*image.avStep(1),1,image.avCols()/2,dumpfile);
            }
            for (int i = 0; i < image.avRows()/2; i++)
            {
                fwrite((char*)image.avAddr(2)+i*image.avStep(2),1,image.avCols()/2,dumpfile);
            }
        }
        else if (dumpfile && image.channels() <= 3)        // BGR
        {
            for (int i = 0; i < image.rows; i++)
            {
                fwrite(image.data+i*image.step[0],1,image.cols*image.channels(),dumpfile);
            }
        }

        if (yuv_enable)
            imwrite(argv[2] + to_string(i_frame_nums) + ".jpg", image);
        else
            imwrite(argv[2] + to_string(i_frame_nums) + ".png", image);

        if ((i_frame_nums+1) % 300 == 0)
        {
#ifdef WIN32
            tv2 = clock();
            double t = tv2 - tv1;
            printf("current process is %f fps!\n", i_frame_nums * 1000.0 / t);
#else
            double time;
            gettimeofday(&tv2, NULL);
            time = (tv2.tv_sec - tv1.tv_sec)*1000.0 + (tv2.tv_usec - tv1.tv_usec)/1000.0;
			printf("tv_sec is %f !\n", (tv2.tv_sec - tv1.tv_sec)*1.0);
			printf("tv_usec is %f !\n", (tv2.tv_usec - tv1.tv_usec)*1.0);
			printf("cost time is %f !\n", (float)time);
            printf("current process is %f fps!\n", i_frame_nums * 1000.0 / time);
#endif
        }
        i_frame_nums++;
    }

    if (dumpfile)
        fclose(dumpfile);
    // the camera will be deinitialized automatically in VideoCapture destructor

    cap.release();
    return 0;
}
