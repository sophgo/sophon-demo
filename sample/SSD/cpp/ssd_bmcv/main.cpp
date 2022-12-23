//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "ssd.hpp"

using json = nlohmann::json;
bool IS_DIR;
bool CONFIDENCE;
bool NMS;

int main(int argc, char **argv){
    /*
     * Custom configurations.
     */
    if(argc < 3){
        std::cout << "USAGE:" << std::endl;
        std::cout << "    " << argv[0] <<" <image directory or video path> <bmodel path> <device id(default: 0)> <conf_thre(default: unset)> <nms_thre(default: unset)>" << std::endl;
        exit(1);
    }
    struct stat info;
    std::string input_url = argv[1];
    if(stat(input_url.c_str(), &info) != 0){
        std::cout << "Cannot find input_path." << std::endl;
        exit(1);
    }
    auto index = input_url.rfind('.');
    std::string postfix = input_url.substr(index + 1);
    std::vector<std::string> video_postfixes = {"mp4", "avi"}; 
    
    auto compare_postfix = [&]()->bool{
        for(int i = 0; i < video_postfixes.size(); i++){
            if(postfix == video_postfixes[i]){
                std::cout << "read input path postfix: " << postfix << std::endl;
                return true;
            }
        }
        return false;
    };

    if(compare_postfix()){
        IS_DIR = false;
        std::cout << "input is video."<< std::endl;
    }else if (info.st_mode & S_IFDIR) {
        IS_DIR = true;
        std::cout << "input is image directory." << std::endl;
    }else{
        std::cout << "unrecognized input path!" << std::endl;
        exit(1);
    }
    std::string bmodel_file = argv[2];
    if(stat(bmodel_file.c_str(), &info) != 0){
        std::cout << "Cannot find valid model file." << std::endl;
        exit(1);
    }
    int dev_id = 0;
    if(argc >= 4){
        dev_id = std::stoi(argv[3]);
    }
    float conf_thre = 0;
    CONFIDENCE = 0;
    if(argc >= 5){
        std::cout << "using confidence threshold!" << std::endl;
        CONFIDENCE = 1;
        conf_thre = std::stof(argv[4]);
    }
    float nms_thre = 0.45;
    NMS = 0;
    if(argc >= 6){
        std::cout << "using NMS!" << std::endl;
        NMS = 1;
        nms_thre = std::stof(argv[5]);
    }
    
    /*------------------------------------------------------
     * Inference flow.
     *------------------------------------------------------
     */

    //1. Get device handle and load bmodel file.
    std::shared_ptr<BMNNHandle> handle = std::make_shared<BMNNHandle>(dev_id);
    std::shared_ptr<BMNNContext> bm_ctx = std::make_shared<BMNNContext>(handle, bmodel_file.c_str());
    std::cout << "Set device id: " << dev_id << " ." << std::endl;     

    //2. Initialize network.
    SSD ssd(bm_ctx, dev_id, conf_thre, nms_thre);
    ssd.Init();
    int batch_size = ssd.batch_size();
    
    //3. Profile
    TimeStamp ssd_ts;
    TimeStamp *ts = &ssd_ts;
    ssd.enableProfile(&ssd_ts);

    //4. Data structures for inference.
    std::vector<cv::Mat> batch_imgs;
    std::vector<std::string> batch_names;
    std::vector<std::vector<SSDObjRect> > results;
    
    //5. Results saving
    json result_json;
    int count = 0;
    ts->save("SSD overall");
    //6. Forward data to network, output detected object boxes.
    if(IS_DIR){
        //Get images in input directory.
        std::cout << "Get images in input directory." << std::endl;
        std::vector<std::string> files_vector;
        DIR *pDir;
        struct dirent* ptr;
        pDir = opendir(input_url.c_str());
        while((ptr = readdir(pDir))!=0) {
            if (strcmp(ptr->d_name, ".") != 0 && strcmp(ptr->d_name, "..") != 0){
                files_vector.push_back(input_url + "/" + ptr->d_name);
            }
        }
        closedir(pDir);
        std::sort(files_vector.begin(),files_vector.end());
        std::cout << "Start SSD inference, total image num: " << files_vector.size() << std::endl;
        #if DEBUG
            for(int i = 0; i < MIN(160, files_vector.size()); i++){
        #else
            for(int i = 0; i < files_vector.size(); i++){
        #endif
                std::cout << "Read image path: " << files_vector[i] << std::endl;
                cv::Mat img_bgr = cv::imread(files_vector[i], cv::IMREAD_COLOR, dev_id);
                cv::Mat img(img_bgr.rows, img_bgr.cols, CV_8UC3, cv::SophonDevice(dev_id));
                cv::cvtColor(img_bgr, img, cv::COLOR_BGR2RGB);
                size_t index = files_vector[i].rfind("/");
                std::string img_name = files_vector[i].substr(index + 1);
                batch_imgs.push_back(img);
                batch_names.push_back(img_name);
                results.push_back(std::vector<SSDObjRect>());
                if((int)batch_imgs.size() == batch_size){
                    CV_Assert(0 == ssd.detect(batch_imgs, batch_names, results));
                    #if DEBUG
                        for(int j = 0; j < results.size(); j++){
                            std::cout << "image " << (i + 1 - batch_size) + j << " :" << std::endl;
                            for(int k = 0; k < results[j].size(); k++){
                                std::cout<<"   ";results[j][k].printBox();
                            }
                        }
                    #endif
                    for(int j = 0; j < results.size(); j++){
                        int lIndex, rIndex;
                        lIndex = batch_names[j].find_first_not_of('0');
                        rIndex = batch_names[j].rfind('.');
                        if(lIndex == rIndex)
                            lIndex--; 
                        auto name = batch_names[j].substr(lIndex, rIndex - lIndex);
                        for(int k = 0; k < results[j].size(); k++){
                            result_json[count]["image_id"] = std::stoi(name);
                            result_json[count]["category_id"] = results[j][k].class_id;
                            result_json[count]["score"] = results[j][k].score;
                            result_json[count++]["bbox"] = {results[j][k].x1, results[j][k].y1, 
                                                            results[j][k].x2 - results[j][k].x1, results[j][k].y2 - results[j][k].y1};
                        }
                    }           
                    batch_imgs.clear();
                    batch_names.clear();
                    results.clear();
                }
            }
            if(!batch_imgs.empty()){
                CV_Assert(0 == ssd.detect(batch_imgs, batch_names, results));
                #if DEBUG
                    for(int j = 0; j < results.size(); j++){
                        std::cout << "image " << (15 + 1 - batch_size) + j << " :" << std::endl;
                        for(int k = 0; k < results[j].size(); k++){
                            std::cout<<"   ";
                            results[j][k].printBox();
                        }
                    }
                #endif    
                for(int j = 0; j < results.size(); j++){
                    int lIndex, rIndex;
                    lIndex = batch_names[j].find_first_not_of('0');
                    rIndex = batch_names[j].rfind('.');
                    if(lIndex == rIndex)
                        lIndex--;
                    auto name = batch_names[j].substr(lIndex, rIndex - lIndex);
                    for(int k = 0; k < results[j].size(); k++){
                        result_json[count]["image_id"] = std::stoi(name);
                        result_json[count]["category_id"] = results[j][k].class_id;
                        result_json[count]["score"] = results[j][k].score;
                        result_json[count++]["bbox"] = {results[j][k].x1, results[j][k].y1, 
                                                        results[j][k].x2 - results[j][k].x1, results[j][k].y2 - results[j][k].y1};
                    }
                }
                batch_imgs.clear();
                batch_names.clear();   
                results.clear();
            }
            std::string json_file = "./bmcv_cpp_result_b" + std::to_string(batch_size) + ".json";
            std::cout << "================================================" << std::endl;
            std::cout << "result saved in " << json_file << std::endl;
            std::cout << "================================================" << std::endl;
            std::ofstream(json_file)<<std::setw(4)<<result_json;
    }else{
        //Get frames in input video.
        cv::VideoCapture cap(input_url, cv::CAP_ANY, dev_id);
        if(!cap.isOpened()){
            std::cout << "open video stream failed!" << std::endl;
            exit(1);
        }
        int w = int(cap.get(cv::CAP_PROP_FRAME_WIDTH));
        int h = int(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
        int frame_num = cap.get(cv::CAP_PROP_FRAME_COUNT);
        std::cout << "Frame num: " << frame_num << std::endl;
        std::cout << "resolution of input stream: " << h << ", " << w << std::endl;
        cv::VideoWriter VideoWriter;
        int frameRate = cap.get(cv::CAP_PROP_FPS);

        //encode type depends on user's machine.
        VideoWriter.open("video.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), frameRate, cv::Size(w, h));
        
        int frame_count = 0;
        while(frame_count < frame_num){
            frame_count++;
            std::vector<cv::Mat> images;
            cv::Mat img_bgr, img;
            cap >> img_bgr;
            std::cout << "frame num: " << frame_count << std::endl;
            cv::cvtColor(img_bgr, img, cv::COLOR_BGR2RGB);
            batch_imgs.push_back(img);
            batch_names.push_back("tmp.jpg"); //useless
            results.push_back(std::vector<SSDObjRect>()); //useless
            if((int)batch_imgs.size() == batch_size){
                CV_Assert(0 == ssd.detect(batch_imgs, batch_names, results, &VideoWriter));
                #if DEBUG
                    for(int j = 0; j < results.size(); j++){
                        for(int k = 0; k < results[j].size(); k++){
                            std::cout<<"   ";results[j][k].printBox();
                        }
                    }
                #endif
                batch_imgs.clear();
                batch_names.clear();
                results.clear();
            }
        }
        cap.release();
        VideoWriter.release();
    }

    ts->save("SSD overall");
    time_stamp_t base_time = time_point_cast<microseconds>(steady_clock::now());
    ts->calbr_basetime(base_time);
    ts->build_timeline("SSD detect");
    ts->show_summary("SSD detect");
    ts->clear();
    return 0;
}