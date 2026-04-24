#include <fstream>
#include <string.h>
#include <dirent.h>
#include <unistd.h>
#include <sys/stat.h>
#include "json.hpp"
#include "opencv2/opencv.hpp"
#include "ff_decode.hpp"
#include "unet.hpp"

using json = nlohmann::json;
using namespace std;

int main(int argc, char * argv[])
{
    cout.setf(ios::fixed);
    // get params
    const char *keys="{bmodel | ../../models/BM1684/unet_fp32_1b.bmodel | bmodel file path}"
        "{dev_id | 0 | TPU device id}"
        "{help | 0 | print help information.}"
        "{input | ../../datasets/test | input path, images direction or video file path}"
        "{n_classes | 2 | the number of segmentation classes}"
        "{out_threshold | 0.5 | the threshold while converting output tensor to mask, only if n_classes == 1}";

    cv::CommandLineParser parser(argc, argv, keys);
    if (parser.get<bool>("help")) {
        parser.printMessage();
        return 0;
    }

    string bmodel_file = parser.get<string>("bmodel");
    string input = parser.get<string>("input");
    int n_classes = parser.get<int>("n_classes");
    float out_threshold = parser.get<float>("out_threshold");
    int dev_id = parser.get<int>("dev_id");

    struct stat info;
    if(stat(bmodel_file.c_str(), &info)!=0)
    {
        cout << "Cannot find valid model file." << endl;
        exit(1);
    }
    if(stat(input.c_str(), &info)!=0)
    {
        cout << "Cannot find input path." << endl;
        exit(1);
    }

    BMNNHandlePtr handle = make_shared<BMNNHandle>(dev_id);
    cout << "set device id: " << dev_id << endl;
    bm_handle_t h = handle->handle();

    shared_ptr<BMNNContext> bm_ctx = make_shared<BMNNContext>(handle, bmodel_file.c_str());

    UNet unet(bm_ctx);
    CV_Assert(0 == unet.Init(out_threshold, n_classes));

    TimeStamp unet_ts;
    TimeStamp *ts = &unet_ts;
    unet.enableProfile(&unet_ts);

    int batch_size = unet.batch_size();

    if (access("results", 0) != F_OK)
        mkdir("results", S_IRWXU);
    if (access("results/images", 0) != F_OK)
        mkdir("results/images", S_IRWXU);

    // test images
    if (info.st_mode & S_IFDIR)
    {
        vector<string> files_vector;
        DIR * pDir;
        struct dirent * ptr;
        pDir = opendir(input.c_str());
        if (pDir == nullptr) {
            std::cerr << "错误: 无法打开目录: " << input << std::endl;
            return -1;
        }
        while((ptr = readdir(pDir)) != 0)
        {
            if(strcmp(ptr->d_name, ".")!=0 && strcmp(ptr->d_name, "..")!=0)
            {
                files_vector.push_back(input + "/" + ptr->d_name);
            }
        }
        closedir(pDir);

        vector<bm_image> batch_imgs;
        vector<string> batch_names;
        vector<json> results_json;
        vector<bm_image> masks;
        int cn = files_vector.size();
        int id = 0;
        for (vector<string>::iterator iter = files_vector.begin(); iter != files_vector.end();++iter)
        {
            string img_file = *iter;
            id++;
            cout << id << "/" << cn << ", img_file: " << img_file << endl;
            ts->save("read image");
            bm_image bmimg;
            picDec(h, img_file.c_str(), bmimg);

            ts->save("read image");
            size_t index = img_file.rfind("/");
            string img_name = img_file.substr(index+1);
            batch_imgs.push_back(bmimg);
            batch_names.push_back(img_name);
            if((int)batch_imgs.size() == batch_size)
            {
                CV_Assert(0 == unet.Segment(batch_imgs, masks));
                for(int i = 0;i<batch_size;++i)
                {
                    void *jpeg_data = NULL;
                    size_t out_size = 0;
                    int ret = bmcv_image_jpeg_enc(h, 1, &masks[i], &jpeg_data, &out_size);
                    if(ret == BM_SUCCESS)
                    {
                        string img_file = "./results/images/" + batch_names[i];
                        std::cout << "save image" << std::endl;
                        std::cout << "path = " << img_file << std::endl;
                        FILE * fp = fopen(img_file.c_str(), "wb");
                        fwrite(jpeg_data, out_size, 1, fp);
                        fclose(fp);
                    }
                    free(jpeg_data);
                    bm_image_destroy(batch_imgs[i]);
                    bm_image_destroy(masks[i]);
                }
                batch_imgs.clear();
                batch_names.clear();
                masks.clear();
            }
        }
        if(!batch_imgs.empty())
        {
            CV_Assert(0 == unet.Segment(batch_imgs, masks));
            for(int i = 0;i<batch_size;++i)
            {
                void *jpeg_data = nullptr;
                size_t out_size = 0;
                int ret = bmcv_image_jpeg_enc(h, 1, &masks[i], &jpeg_data, &out_size);
                string img_file = "./results/images/" + batch_names[i];
                if(ret == BM_SUCCESS)
                {
                    FILE * fp = fopen(img_file.c_str(), "wb");
                    fwrite(jpeg_data, out_size, 1, fp);
                    fclose(fp);
                }
                free(jpeg_data);
                bm_image_destroy(batch_imgs[i]);
                bm_image_destroy(masks[i]);
            }
            batch_imgs.clear();
            batch_names.clear();
            masks.clear();
        }
    }

    // test video
    else
    {
        VideoDecFFM decoder;
        decoder.openDec(&h, input.c_str());
        int id = 0;
        vector<bm_image> batch_imgs;
        vector<bm_image> masks;
        while(true)
        {
            bm_image *img = decoder.grab();
            if (!img)
                break;
            bm_image BGR_img;
            bm_image_create(h, (*img).height, (*img).width, FORMAT_BGR_PLANAR, DATA_TYPE_EXT_1N_BYTE, &BGR_img);
            bmcv_image_yuv2bgr_ext(h, 1, img, &BGR_img);
            batch_imgs.push_back(BGR_img);
            if ((int)batch_imgs.size() == batch_size) 
            {
                CV_Assert(0 == unet.Segment(batch_imgs, masks));
                for(int i = 0;i<batch_size;++i)
                {
                    id ++;
                    void *jpeg_data = NULL;
                    size_t out_size = 0;
                    int ret = bmcv_image_jpeg_enc(h, 1, &masks[i], &jpeg_data, &out_size);
                    if(ret == BM_SUCCESS)
                    {
                        string img_file = "./results/images/" + to_string(id) + ".jpg";
                        std::cout << "save image" << std::endl;
                        std::cout << "path = " << img_file << std::endl;
                        FILE * fp = fopen(img_file.c_str(), "wb");
                        fwrite(jpeg_data, out_size, 1, fp);
                        fclose(fp);
                    }
                    free(jpeg_data);
                    bm_image_destroy(batch_imgs[i]);
                    bm_image_destroy(masks[i]);
                }
                batch_imgs.clear();
                masks.clear();
            }
        }
        if (!batch_imgs.empty()){
        CV_Assert(0 == unet.Segment(batch_imgs, masks));
                for(int i = 0;i<batch_size;++i)
                {
                    id ++;
                    void *jpeg_data = NULL;
                    size_t out_size = 0;
                    int ret = bmcv_image_jpeg_enc(h, 1, &masks[i], &jpeg_data, &out_size);
                    if(ret == BM_SUCCESS)
                    {
                        string img_file = "./results/images/" + to_string(id) + ".jpg";
                        std::cout << "save image" << std::endl;
                        std::cout << "path = " << img_file << std::endl;
                        FILE * fp = fopen(img_file.c_str(), "wb");
                        fwrite(jpeg_data, out_size, 1, fp);
                        fclose(fp);
                    }
                    free(jpeg_data);
                    bm_image_destroy(batch_imgs[i]);
                    bm_image_destroy(masks[i]);
                }
                batch_imgs.clear();
                masks.clear();
        }
    }
    time_stamp_t base_time = time_point_cast<microseconds>(steady_clock::now());
    unet_ts.calbr_basetime(base_time);
    unet_ts.build_timeline("unet test");
    unet_ts.show_summary("unet test");
    unet_ts.clear();

    return 0;
}