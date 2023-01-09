//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "c3d.hpp"

using json = nlohmann::json;

int main(int argc, char **argv){
    /*
     * Custom configurations.
     */
    if(argc < 3){
        std::cout << "USAGE:" << std::endl;
        std::cout << "    " << argv[0] <<" <dataset path> <bmodel path> <device id(default: 0)>" << std::endl;
        exit(1);
    }
    struct stat info;
    std::string input_url = argv[1];
    if(stat(input_url.c_str(), &info) != 0){
        std::cout << "Cannot find dataset path." << std::endl;
        exit(1);
    }
    if(!(info.st_mode & S_IFDIR)) {
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
    int step_len = 6;
    /*------------------------------------------------------
     * Inference flow.
     *------------------------------------------------------
     */

    //1. Get device handle and load bmodel file.
    std::shared_ptr<BMNNHandle> handle = std::make_shared<BMNNHandle>(dev_id);
    std::shared_ptr<BMNNContext> bm_ctx = std::make_shared<BMNNContext>(handle, bmodel_file.c_str());
    std::cout << "Set device id: " << dev_id << " ." << std::endl;     

    //2. Initialize network.
    C3D c3d(bm_ctx, step_len, dev_id);
    c3d.Init();
    int batch_size = c3d.batch_size();
    
    //3. Profile
    TimeStamp c3d_ts;
    TimeStamp *ts = &c3d_ts;
    c3d.enableProfile(&c3d_ts);

    //4. Data structures for inference.
    std::vector<std::string> batch_videos;
    int total = 0;
    int correct = 0;
    ts->save("C3D overall");
    //5. Forward data to network, output detected object boxes.
    //get classes in dataset.
    std::vector<std::string> class_folders, class_names;
    DIR *pDir;
    struct dirent* ptr;
    pDir = opendir(input_url.c_str());
    while((ptr = readdir(pDir))!=0) {
        if(strcmp(ptr->d_name, ".") == 0 || strcmp(ptr->d_name, "..") == 0)
            continue;
        std::string class_folder = input_url + "/" + ptr->d_name;
        if(stat(class_folder.c_str(), &info) != 0){
            std::cout << "Cannot find class path." << std::endl;
            exit(1);
        }
        if(!(info.st_mode & S_IFDIR)) {
            std::cout << "invalid dataset structure!" << std::endl;
            exit(1);
        }
        class_folders.push_back(class_folder);
        class_names.push_back(ptr->d_name);
    }
    std::sort(class_folders.begin(),class_folders.end());
    std::sort(class_names.begin(),class_names.end());
    closedir(pDir);
    for(int i = 0; i < class_folders.size(); i++){
        pDir = opendir(class_folders[i].c_str());
        std::vector<std::string> video_paths;
        while((ptr = readdir(pDir))!=0) {
            if(strcmp(ptr->d_name, ".") == 0 || strcmp(ptr->d_name, "..") == 0)
                continue;
            std::string video_path = class_folders[i] + "/" + ptr->d_name;
            if(stat(video_path.c_str(), &info) != 0){
                std::cout << "Cannot find video path." << std::endl;
                exit(1);
            }
            auto index = video_path.rfind('.');
            std::string postfix = video_path.substr(index + 1);
            std::vector<std::string> video_postfixes = {"mp4", "avi"}; 
            #if DEBUG
                std::cout<<video_path<<std::endl;
            #endif
            if(std::find(video_postfixes.begin(), video_postfixes.end(), postfix) 
                    != video_postfixes.end()){
                video_paths.push_back(video_path);
            }
            else{
                std::cout << "skipping video path, please check your dataset!" << std::endl;
            }
        }
        std::sort(video_paths.begin(),video_paths.end());

        for(int j = 0; j < video_paths.size(); j++){
            std::cout << "Read video path: " << video_paths[j] << std::endl;
            batch_videos.push_back(video_paths[j]);
            if(batch_videos.size() == batch_size){
                std::vector<int> predict_ids;
                c3d.detect(batch_videos, predict_ids);
                total += batch_videos.size();
                for(int k = 0; k < predict_ids.size(); k++){
                    std::cout << "Predict: " << class_names[predict_ids[k]] << std::endl;
                    if(predict_ids[k] == i){
                        correct += 1;
                    }
                }
                batch_videos.clear();
            }
        }
        if(batch_videos.size() > 0){
            std::vector<int> predict_ids;
            c3d.detect(batch_videos, predict_ids);
            total += batch_videos.size();
            for(int k = 0; k < predict_ids.size(); k++){
                std::cout << "Predict: " << class_names[predict_ids[k]] << std::endl;
                if(predict_ids[k] == i){
                    correct += 1;
                }
            }
            batch_videos.clear();
        }
        std::cout << "========================================" << std::endl;
        std::cout << "acc now: " << (float)correct / (float)total << std::endl; 
        std::cout << "========================================" << std::endl;
    }
    ts->save("C3D overall");
    time_stamp_t base_time = time_point_cast<microseconds>(steady_clock::now());
    ts->calbr_basetime(base_time);
    ts->build_timeline("C3D detect");
    ts->show_summary("C3D detect");
    ts->clear();
    return 0;
}