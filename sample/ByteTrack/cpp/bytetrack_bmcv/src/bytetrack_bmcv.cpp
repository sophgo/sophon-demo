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
#include "BYTETracker.h"

int save_result(std::string save_file_name, std::vector<SaveResult> results)
{
    std::ofstream fp(save_file_name);

    if (!fp)
    {
        printf("Can not open file: %s\n", save_file_name.c_str());
        return 1;
    }
    std::string save_format = "{frame},{id},{x1},{y1},{w},{h},-1,-1,-1,-1\n";

    for (const SaveResult track : results)
    {
        std::string line = save_format;
        line.replace(line.find("{frame}"), 7, std::to_string(track.frame_id));
        line.replace(line.find("{id}"), 4, std::to_string(track.track_id));
        line.replace(line.find("{x1}"), 4, std::to_string(track.tlwh[0]));
        line.replace(line.find("{y1}"), 4, std::to_string(track.tlwh[1]));
        line.replace(line.find("{w}"), 3, std::to_string(track.tlwh[2]));
        line.replace(line.find("{h}"), 3, std::to_string(track.tlwh[3]));

        fp << line;
    }

    fp.close();
    printf("save detect result: %s\n", save_file_name.c_str());
    return 0;
}

void ObjRect2Object(std::vector<ObjRect> yolo_out, std::vector<Object> &objects)
{
    int count = yolo_out.size();
    objects.resize(count);
    for (int i = 0; i < count; i++)
    {
        objects[i].label = yolo_out[i].class_id;
        objects[i].prob = yolo_out[i].score;
        objects[i].rect.x = yolo_out[i].left;
        objects[i].rect.y = yolo_out[i].top;
        objects[i].rect.width = yolo_out[i].width;
        objects[i].rect.height = yolo_out[i].height;
    }
}

int main(int argc, char **argv)
{
    if (argc != 9)
    {
        printf("USAGE: \n");
        printf("      %s video <video url> <bmodel path> <test count> <detect threshold> <nms threshold> <save path> <device id>\n", argv[0]);
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

    mkdir(save_path.c_str(), 0755);

    if (file_path.c_str()[file_path.length() - 1] == '/')
    {
        file_path = file_path.substr(0, file_path.length() - 1);
    }
    if (save_path.c_str()[save_path.length() - 1] == '/')
    {
        save_path = save_path.substr(0, save_path.length() - 1);
    }
    std::string save_result_name = save_path + "/";
    if (is_video)
        save_result_name += file_path.substr(file_path.find_last_of('/') + 1, file_path.find_last_of(".") - file_path.find_last_of("/") - 1);
    else
        save_result_name += file_path.substr(file_path.find_last_of('/') + 1);
    save_result_name += "_";
    save_result_name += model_name.substr(model_name.find_last_of("/") + 1, model_name.find_last_of(".") - model_name.find_last_of("/") - 1);
    save_result_name += "_cpp.txt";

    std::map<std::string, std::vector<ObjRect>> dete_result;
    SaveResult result;

    std::vector<SaveResult> results;

    YoloXDete pyolox(model_name, device_id);
    float scale = pyolox.get_input_scale();
    sail::Handle handle = pyolox.get_handle();
    int batch_size = pyolox.get_batchsize();
    int net_h = pyolox.get_net_input_w();
    int net_w = pyolox.get_net_input_h();
    bm_image_data_format_ext img_dtype = pyolox.get_img_dtype();
    sail::Bmcv bmcv(handle);

    float *output_data = (float *)pyolox.get_output_data_prt();
    std::vector<int> output_shape = pyolox.get_output_shape();

    printf("Input Scale: %f\n", scale);
    printf("Batch Size: %d\n", batch_size);
    printf("Input Width: %d\n", net_w);
    printf("Input Height: %d\n", net_h);
    scale = 0.00392157;
    std::tuple<std::pair<float, float>, std::pair<float, float>, std::pair<float, float>> alpha_beta(std::pair<float, float>(scale, 0), std::pair<float, float>(scale, 0), std::pair<float, float>(scale, 0));

    std::vector<int> strides;
    strides.push_back(8);
    strides.push_back(16);
    strides.push_back(32);
    YoloX_PostForward postprocessor(net_w, net_h, strides);
    BYTETracker tracker(30, 30);
    if (is_video)
    {
        VideoProcess video_process(file_path, handle, batch_size, device_id, net_w, net_h, img_dtype);
        std::vector<float> resize_scale = video_process.get_resize_scalemin();
        for (int i = 0; i < loops; ++i)
        {
            printf("loops: %d\n", i);
            video_process.getTensor(handle, bmcv, *pyolox.input_tensor, alpha_beta);
            pyolox.Inference();
            std::vector<std::vector<ObjRect>> detections;
            postprocessor.process(output_data, output_shape, resize_scale, threshold_dete, threshold_nms, detections);

            char save_name[256] = {0};
            char frame_name[256] = {0};
            auto temp_iter = dete_result.find(std::string(frame_name));
            if (temp_iter == dete_result.end())
            {
                dete_result.insert(std::pair<std::string, std::vector<ObjRect>>(std::string(frame_name), detections[0]));
            }
            std::vector<ObjRect> output = detections[0];
            std::vector<Object> objects;
            ObjRect2Object(output, objects);
            vector<STrack> output_stracks = tracker.update(objects);
            for (const STrack output_track : output_stracks)
            {
                result.frame_id = output_track.frame_id;
                result.track_id = output_track.track_id;
                result.tlwh = output_track.tlwh;
                results.push_back(result);
            }
        }
    }
    else
    {
        PictureProcess picture_process(file_path, handle, batch_size, device_id, net_w, net_h, img_dtype);
        while (picture_process.getTensor(handle, bmcv, *pyolox.input_tensor, alpha_beta) == 0)
        {
            std::vector<float> resize_scale = picture_process.get_resize_scalemin();
            std::vector<string> name_list = picture_process.get_curr_imagename();
            pyolox.Inference();
            std::vector<std::vector<ObjRect>> detections;
            postprocessor.process(output_data, output_shape, resize_scale, threshold_dete, threshold_nms, detections);

            for (size_t box_idx = 0; box_idx < detections[0].size(); box_idx++)
            {
                bmcv.rectangle(picture_process.input_image,
                               int(detections[0][box_idx].left),
                               int(detections[0][box_idx].top),
                               int(detections[0][box_idx].width),
                               int(detections[0][box_idx].height),
                               std::make_tuple(0, 0, 255), 4);
            }
            int idx_start = name_list.at(0).find_last_of('/') + 1;
            std::string image_name_temp = name_list.at(0).substr(idx_start, name_list.at(0).size());
            string save_name = save_path + "/" + image_name_temp;

            auto temp_iter = dete_result.find(image_name_temp);
            if (temp_iter == dete_result.end())
            {
                dete_result.insert(std::pair<std::string, std::vector<ObjRect>>(image_name_temp, detections[0]));
            }
            std::vector<ObjRect> output = detections[0];
            std::vector<Object> objects;
            ObjRect2Object(output, objects);
            vector<STrack> output_stracks = tracker.update(objects);
            for (const STrack output_track : output_stracks)
            {
                result.frame_id = output_track.frame_id;
                result.track_id = output_track.track_id;
                result.tlwh = output_track.tlwh;
                results.push_back(result);
            }
        }
    }
    save_result(save_result_name, results);
    return 0;
}