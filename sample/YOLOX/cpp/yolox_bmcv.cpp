//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include "yolox.h"
#include <stdio.h>
#include <sys/stat.h>
#include <sys/types.h>

int save_result(std::string save_file_name, std::map<std::string,std::vector<ObjRect>> dete_resu)
{
    FILE* fp = fopen(save_file_name.c_str(),"w+");
    if(!fp){
        printf("Can not open file: %s\n",save_file_name.c_str());
        return 1;
    }
    auto temp_iter = dete_resu.begin();
    while (temp_iter != dete_resu.end())    {
        for (int i=0;i<temp_iter->second.size();++i){
            fprintf(fp,"[%s]\n",temp_iter->first.c_str());
            fprintf(fp,"category=%d\n",temp_iter->second.at(i).class_id);
            fprintf(fp,"score=%.2f\n",temp_iter->second.at(i).score);
            fprintf(fp,"left=%.2f\n",temp_iter->second.at(i).left);
            fprintf(fp,"top=%.2f\n",temp_iter->second.at(i).top);
            fprintf(fp,"right=%.2f\n",temp_iter->second.at(i).right);
            fprintf(fp,"bottom=%.2f\n\n",temp_iter->second.at(i).bottom);
        }
        temp_iter++;
    }
    fclose(fp);
    printf("save detect result: %s\n",save_file_name.c_str());
    return 0;
}

int main(int argc, char** argv)
{
     if (argc != 9){
        printf("USAGE: \n");
        printf("      %s video <video url> <bmodel path> <test count> <detect threshold> <nms threshold> <save path> <device id>\n",argv[0]);
        exit(1);
    }
    bool is_video = false;
    if (strcmp(argv[1], "video") == 0)
        is_video = true;
    
    std::string file_path(argv[2]);
    std::string model_name(argv[3]);
    int loops = atoi(argv[4]);
    float threshold_dete = atof(argv[5]);
    float threshold_nms = atof(argv[6]);
    std::string save_path(argv[7]);
    int device_id = atoi(argv[8]);

    mkdir(save_path.c_str(),0755);

    if (file_path.c_str()[file_path.length()-1] == '/')    {
        file_path = file_path.substr(0,file_path.length()-1);
    }
    if (save_path.c_str()[save_path.length()-1] == '/')    {
        save_path = save_path.substr(0,save_path.length()-1);
    }
    std::string save_result_name = save_path+"/";
    if(is_video)
        save_result_name += file_path.substr(file_path.find_last_of('/')+1,file_path.find_last_of(".")-file_path.find_last_of("/")-1);
    else
        save_result_name += file_path.substr(file_path.find_last_of('/')+1);
    save_result_name += "_";
    save_result_name += model_name.substr(model_name.find_last_of("/")+1,model_name.find_last_of(".")-model_name.find_last_of("/")-1);
    save_result_name += "_cpp.txt";

    std::map<std::string,std::vector<ObjRect>> dete_result;

    YoloXDete pyolox(model_name,device_id);
    float scale = pyolox.get_input_scale();
    sail::Handle handle = pyolox.get_handle();
    int batch_size = pyolox.get_batchsize();
    int net_w = pyolox.get_net_input_w();
    int net_h = pyolox.get_net_input_h();
    bm_image_data_format_ext img_dtype = pyolox.get_img_dtype();
    sail::Bmcv bmcv(handle);

    float* output_data = (float*)pyolox.get_output_data_prt();
    std::vector<int> output_shape = pyolox.get_output_shape();

    printf("Input Scale: %f\n",scale);
    printf("Batch Size: %d\n",batch_size);
    printf("Input Width: %d\n",net_w);
    printf("Input Height: %d\n",net_h);

    std::tuple<std::pair<float, float>, std::pair<float, float>,std::pair<float, float>> alpha_beta(std::pair<float, float>(scale,0),std::pair<float, float>(scale,0),std::pair<float, float>(scale,0));

    std::vector<int> strides;
    strides.push_back(8);
    strides.push_back(16);
    strides.push_back(32);
    YoloX_PostForward postprocessor(net_w, net_h, strides);
   
    if(is_video){
        VideoProcess video_process(file_path, handle,batch_size,device_id,net_w,net_h,img_dtype);
        std::vector<float> resize_scale = video_process.get_resize_scalemin();
        for (int i=0;i<loops;++i){
            video_process.getTensor(handle, bmcv, *pyolox.input_tensor,alpha_beta);
            pyolox.Inference();
            std::vector<std::vector<ObjRect>> detections;
            postprocessor.process(output_data,output_shape,resize_scale,threshold_dete,threshold_nms,detections);
       
            if (batch_size == 1)            {
                char save_name[256] = {0};
                char frame_name[256] = {0};
                sprintf(frame_name,"frame_%d",i);
                sprintf(save_name,"%s/%s_device_%d.jpg",save_path.c_str(),frame_name,device_id);
                auto temp_iter = dete_result.find(std::string(frame_name));
                if(temp_iter == dete_result.end()){
                    dete_result.insert(std::pair<std::string,std::vector<ObjRect>>(std::string(frame_name),detections[0]));
                }
                for (size_t box_idx = 0; box_idx < detections[0].size(); box_idx++)
                {
                    bmcv.rectangle(video_process.input_image, 
                        int(detections[0][box_idx].left),
                        int(detections[0][box_idx].top),
                        int(detections[0][box_idx].width),
                        int(detections[0][box_idx].height),
                        std::make_tuple(0, 0, 255), 4);
                }
                
                // bmcv.imwrite(save_name,video_process.input_image);
                // sprintf(save_name,"%s/loop_%d_resize.jpg",save_path.c_str(),i);
                // bmcv.imwrite(save_name,video_process.resize_image);
            }else{
                char save_name[256] = {0};
                for (size_t j = 0; j < batch_size; ++j)           {
                    char frame_name[256] = {0};
                    sprintf(frame_name,"frame_%d",i*batch_size+j);
                    sprintf(save_name,"%s/%s_device_%d.jpg",save_path.c_str(), frame_name, device_id);
                    auto temp_iter = dete_result.find(std::string(frame_name));
                    if(temp_iter == dete_result.end()){
                        dete_result.insert(std::pair<std::string,std::vector<ObjRect>>(std::string(frame_name),detections[j]));
                    }
                    for (size_t box_idx = 0; box_idx < detections[j].size(); box_idx++)
                    {
                        bmcv.rectangle_(video_process.input_image_array[j], 
                            int(detections[j][box_idx].left),
                            int(detections[j][box_idx].top),
                            int(detections[j][box_idx].width),
                            int(detections[j][box_idx].height),
                            std::make_tuple(0, 0, 255), 4);
                        /* code */
                    }
                    //bmcv.imwrite_(save_name,video_process.input_image_array[j]);
                    // sprintf(save_name,"%s/loop_%d_%d_resize.jpg",save_path.c_str(),i, j);
                    // bmcv.imwrite_(save_name,video_process.resize_image_array[j]);
                }
                
            }
        }
    }else{
        PictureProcess picture_process(file_path, handle,batch_size,device_id,net_w,net_h,img_dtype);
        while (picture_process.getTensor(handle, bmcv, *pyolox.input_tensor,alpha_beta) == 0)
        {
            std::vector<float> resize_scale = picture_process.get_resize_scalemin();
            std::vector<string> name_list = picture_process.get_curr_imagename();
            pyolox.Inference();
            std::vector<std::vector<ObjRect>> detections;
            postprocessor.process(output_data,output_shape,resize_scale,threshold_dete,threshold_nms,detections);
       
            if (batch_size == 1)            {
                for (size_t box_idx = 0; box_idx < detections[0].size(); box_idx++)
                {
                    bmcv.rectangle(picture_process.input_image, 
                        int(detections[0][box_idx].left),
                        int(detections[0][box_idx].top),
                        int(detections[0][box_idx].width),
                        int(detections[0][box_idx].height),
                        std::make_tuple(0, 0, 255), 4);
                }
                int idx_start = name_list.at(0).find_last_of('/')+1;
                std::string image_name_temp = name_list.at(0).substr(idx_start,name_list.at(0).size());
                string save_name = save_path + "/" + image_name_temp;
                // printf("Save: %s\n",save_name.c_str());

                auto temp_iter = dete_result.find(image_name_temp);
                if(temp_iter == dete_result.end()){
                    dete_result.insert(std::pair<std::string,std::vector<ObjRect>>(image_name_temp,detections[0]));
                }
                //bmcv.imwrite(save_name,picture_process.input_image);
            }else{
                for (size_t j = 0; j < batch_size; ++j)           {
                    for (size_t box_idx = 0; box_idx < detections[j].size(); box_idx++)
                    {
                        bmcv.rectangle(picture_process.input_image_array[j], 
                            int(detections[j][box_idx].left),
                            int(detections[j][box_idx].top),
                            int(detections[j][box_idx].width),
                            int(detections[j][box_idx].height),
                            std::make_tuple(0, 0, 255), 4);                   
                    }

                    int idx_start = name_list.at(j).find_last_of('/')+1;
                    std::string image_name_temp = name_list.at(j).substr(idx_start,name_list.at(j).size());
                    string save_name = save_path + "/" + image_name_temp;
                    // printf("Save: %s\n",save_name.c_str());

                    auto temp_iter = dete_result.find(image_name_temp);
                    if(temp_iter == dete_result.end()){
                        dete_result.insert(std::pair<std::string,std::vector<ObjRect>>(image_name_temp,detections[j]));
                    }
                    //bmcv.imwrite(save_name,picture_process.input_image_array[j]);
                    // sprintf(save_name,"%s/loop_%d_%d_resize.jpg",save_path.c_str(),i, j);
                    // bmcv.imwrite_(save_name,video_process.resize_image_array[j]);
                }
            }
        }
        
    }
    save_result(save_result_name,dete_result);
    return 0;
}


//./yolox_sail.pcie video /workspace/test_sail/data/zhuheqiao.mp4 /workspace/test/YOLOX/models/yolox_s_fp32_batch1/compilation.bmodel 16 0.45 0.25 save_pic 0