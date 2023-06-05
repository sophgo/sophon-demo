#define USE_FFMPEG 1
#define USE_OPENCV 1
#define USE_BMCV 1
#include <iostream>
#include <sys/stat.h>
#include <fstream>
#include <dirent.h>
#include <unistd.h>
#include <string>
#include <map>

#include "ppocr_rec.hpp"
PPOCR_Rec::PPOCR_Rec(std::shared_ptr<BMNNContext> context):m_bmContext(context)
{
    std::cout << "PPOCR_Rec ..." << std::endl;
}

PPOCR_Rec::~PPOCR_Rec()
{   
    for(auto& stage_size : img_size){
        auto ret = bm_image_destroy_batch(resize_bmcv_map[stage_size.w].data(), max_batch);
        assert(BM_SUCCESS == ret);
        ret = bm_image_destroy_batch(linear_trans_bmcv_map[stage_size.w].data(), max_batch);
        assert(BM_SUCCESS == ret);
    }
}     

int PPOCR_Rec::Init(const std::string &label_path)
{
    //1. get network
    m_bmNetwork = m_bmContext->network(0);
    std::unordered_set<int> exists_stage_w_;
    for(int i = 0; i < m_bmNetwork->m_netinfo->stage_num; i++){
        auto tensor = m_bmNetwork->inputTensor(0, i);
        if(tensor->get_shape()->dims[0] == 1){
            exists_stage_w_.insert(tensor->get_shape()->dims[3]); //add new stage_w
        }
    }
    for(int i = 0; i < m_bmNetwork->m_netinfo->stage_num; i++){
        auto tensor = m_bmNetwork->inputTensor(0, i);
        if(tensor->get_shape()->dims[0] != 1){
            if(exists_stage_w_.count(tensor->get_shape()->dims[3]) <= 0){
                std::cerr << "Warning: A batch4 stage cannot exist alone without a batch1 stage with the same shape!" << std::endl;
                incomplete_stages.insert(tensor->get_shape()->dims[3]);
            } //check stage_w
        }
    }

    // max_batch = batch_size;
    max_batch = m_bmNetwork->maxBatch();

    // std::cout << "stage_num = " << m_bmNetwork->m_netinfo->stage_num << std::endl;
    // std::cout << "input_num = " << m_bmNetwork->m_netinfo->input_num << std::endl;
    // get all stages' ratios and sizes.
    int pre_net_h = -1;
    for(int i = 0; i < m_bmNetwork->m_netinfo->stage_num; i++){
        auto tensor = m_bmNetwork->inputTensor(0,i);
        net_h_ = tensor->get_shape()->dims[2];
        if(pre_net_h == -1){
            pre_net_h = net_h_;
        }else if(pre_net_h != net_h_){
            std::cerr << "Invalid model size! All Stage's height must be identical." << std::endl;
            exit(1);
        }
        net_w_ = tensor->get_shape()->dims[3];
        bool skip_flag = false;
        for(auto& tmp_size : img_size){
            if(tmp_size.w == net_w_){
                skip_flag = true;
                break;
            }
        }
        if(skip_flag == true){
            continue;
        }
        img_size.push_back({net_w_, net_h_});
    }
    std::sort(img_size.begin(), img_size.end(), [](const RecModelSize& a, const RecModelSize& b) {
                                                    return a.w < b.w;
                                                });
    for(auto& s : img_size){
        img_ratio.push_back((float)s.w / (float)s.h);
    }

    auto tensor = m_bmNetwork->inputTensor(0, 0);
    for(int i = 1; i < m_bmNetwork->m_netinfo->stage_num; i++){
        auto tensor_i = m_bmNetwork->inputTensor(0, i);
        if(tensor_i->get_dtype() != tensor->get_dtype()){
            std::cerr << "All stages' input_dtype must be identical." << std::endl;
            exit(1);
        }
    }

    if (tensor->get_dtype() == BM_FLOAT32)
        input_is_int8_ = false;
    else
        input_is_int8_ = true;

    if (input_is_int8_)
    { // INT8
        data_type = DATA_TYPE_EXT_1N_BYTE_SIGNED;
    }
    else
    { // FP32
        data_type = DATA_TYPE_EXT_FLOAT32;
    }
        
    // bm_image map for storing inference inputs.
    for(auto& stage_size : img_size){
        std::vector<bm_image> tmp_resize;
        tmp_resize.resize(max_batch);
        resize_bmcv_map[stage_size.w] = tmp_resize;
        auto ret = bm_image_create_batch(m_bmContext->handle(), 
                                        stage_size.h, stage_size.w, 
                                        FORMAT_BGR_PLANAR, 
                                        DATA_TYPE_EXT_1N_BYTE, 
                                        resize_bmcv_map[stage_size.w].data(), 
                                        max_batch);
        std::vector<bm_image> tmp_converto;
        tmp_converto.resize(max_batch);
        linear_trans_bmcv_map[stage_size.w] = tmp_converto;                                        
        ret = bm_image_create_batch(m_bmContext->handle(), 
                                        stage_size.h, stage_size.w, 
                                        FORMAT_BGR_PLANAR, 
                                        data_type, 
                                        linear_trans_bmcv_map[stage_size.w].data(), 
                                        max_batch);
        assert(BM_SUCCESS == ret);
    }
    linear_trans_param_.alpha_0 = 0.0078125;
    linear_trans_param_.alpha_1 = 0.0078125;
    linear_trans_param_.alpha_2 = 0.0078125;
    linear_trans_param_.beta_0 = -127.5 * 0.0078125;
    linear_trans_param_.beta_1 = -127.5 * 0.0078125;
    linear_trans_param_.beta_2 = -127.5 * 0.0078125;

    this->label_list_ = PPOCR_Rec::ReadDict(label_path);
    this->label_list_.insert(this->label_list_.begin(),
                            "#"); // blank char for ctc
    this->label_list_.push_back(" ");

    return 0;
}

int PPOCR_Rec::run(std::vector<bm_image> input_images, std::vector<OCRBoxVec> &boxes_vec, std::vector<std::pair<int, int>> ids){
    std::map<int, std::vector<bm_image>> rec_img_map;
    std::map<int, std::vector<std::pair<int, int>>> rec_id_map;
    int ret = 0;
    
    for(int i = 0; i < input_images.size(); i++){
        float ratio = (float)input_images[i].width / (float)input_images[i].height;
        bool assign_flag = false;
        for(int j = 0; j < img_ratio.size(); j++){
           if(ratio <= img_ratio[j]){
                rec_img_map[img_size[j].w].push_back(input_images[i]);
                rec_id_map[img_size[j].w].push_back(ids[i]);
                assign_flag = true;
                break;
           } 
        }
        if(assign_flag == false){
            rec_img_map[img_size.back().w].push_back(input_images[i]);
            rec_id_map[img_size.back().w].push_back(ids[i]);
        }
    }
    
    for (auto it = rec_img_map.begin(); it != rec_img_map.end(); ++it) {
        auto input_images_staged = it->second;
        auto stage_w = it->first;
        auto ids_staged = rec_id_map[stage_w];
        std::vector<bm_image> batch_images;
        std::vector<std::pair<int, int>> batch_ids;
        std::vector<std::pair<std::string, float>> results;
        for(int i = 0; i < input_images_staged.size(); i++){
            batch_images.push_back(input_images_staged[i]);
            batch_ids.push_back(ids_staged[i]);
            if(batch_images.size() == max_batch){
                m_ts->save("(per crop)Rec preprocess", batch_images.size());
                assert(0 == preprocess_bmcv(batch_images, stage_w));
                m_ts->save("(per crop)Rec preprocess", batch_images.size());

                m_ts->save("(per crop)Rec inference", batch_images.size());
                assert(0 == m_bmNetwork->forward());
                m_ts->save("(per crop)Rec inference", batch_images.size());

                m_ts->save("(per crop)Rec postprocess", batch_images.size());
                assert(0 == post_process(results));
                m_ts->save("(per crop)Rec postprocess", batch_images.size());
                assert(max_batch == results.size());
                for(int j = 0; j < results.size(); j++){
                    boxes_vec[batch_ids[j].first][batch_ids[j].second].rec_res = results[j].first;
                    boxes_vec[batch_ids[j].first][batch_ids[j].second].score = results[j].second;
                    
                    /*TODO: check this bug when batchsize == 9.*/
                    // if(boxes_vec[batch_ids[j].first][batch_ids[j].second].x1 == 916 && boxes_vec[batch_ids[j].first][batch_ids[j].second].y1 == 307){
                    //     cv::Mat img;
                    //     cv::bmcv::toMAT(&batch_images[j], img);
                    //     std::string fname = cv::format("test1.jpg");
                    //     cv::imwrite(fname, img);
                    // }
                }
                
                batch_images.clear();
                batch_ids.clear();
                results.clear();
            }
        }
        // if there is not 1 batch stage bmodel:
        if(incomplete_stages.count(stage_w)){
            int batch_size_tmp = batch_images.size();
            for(int i = batch_images.size(); i < max_batch; i++){
                batch_images.push_back(batch_images[0]);
            }
            m_ts->save("(per crop)Rec preprocess", batch_images.size());
            assert(0 == preprocess_bmcv(batch_images, stage_w));
            m_ts->save("(per crop)Rec preprocess", batch_images.size());

            m_ts->save("(per crop)Rec inference", batch_images.size());
            assert(0 == m_bmNetwork->forward());
            m_ts->save("(per crop)Rec inference", batch_images.size());

            m_ts->save("(per crop)Rec postprocess", batch_images.size());
            assert(0 == post_process(results));
            m_ts->save("(per crop)Rec postprocess", batch_images.size());
            for(int j = 0; j < batch_size_tmp; j++){
                boxes_vec[batch_ids[j].first][batch_ids[j].second].rec_res = results[j].first;
                boxes_vec[batch_ids[j].first][batch_ids[j].second].score = results[j].second;
            }
        }
        // Last incomplete batch, use single batch model stage.
        else for(int i = 0; i < batch_images.size(); i++){
            m_ts->save("(per crop)Rec preprocess", 1);
            assert(0 == preprocess_bmcv({batch_images[i]}, stage_w));
            m_ts->save("(per crop)Rec preprocess", 1);

            m_ts->save("(per crop)Rec inference", 1);
            assert(0 == m_bmNetwork->forward());
            m_ts->save("(per crop)Rec inference", 1);

            m_ts->save("(per crop)Rec postprocess", 1);
            assert(0 == post_process(results));
            m_ts->save("(per crop)Rec postprocess", 1);
            
            boxes_vec[batch_ids[i].first][batch_ids[i].second].rec_res = results[i].first;
            boxes_vec[batch_ids[i].first][batch_ids[i].second].score = results[i].second;
                    
                    /*DEBUG code, you can ignore this.*/
                    // int j = i;
                    // if(boxes_vec[batch_ids[j].first][batch_ids[j].second].x1 == 916 && boxes_vec[batch_ids[j].first][batch_ids[j].second].y1 == 307){ //batchsize == 9.
                    //     cv::Mat img;
                    //     cv::bmcv::toMAT(&batch_images[j], img);
                    //     std::string fname = cv::format("test1.jpg");
                    //     cv::imwrite(fname, img);
                    // }

        }
        batch_images.clear();
        batch_ids.clear();
        results.clear();
    }

    return 0;
}


int PPOCR_Rec::preprocess_bmcv(const std::vector<bm_image> batch_bmimgs, int stage_w){ //stage_w: a std::map key.
    for(int i = 0; i < batch_bmimgs.size(); i++){
        bm_image image_aligned;
        int stride1[3], stride2[3];
        bm_image_get_stride(batch_bmimgs[i], stride1);
        stride2[0] = FFALIGN(stride1[0], 64);
        stride2[1] = FFALIGN(stride1[1], 64);
        stride2[2] = FFALIGN(stride1[2], 64);
        bm_image_create(m_bmContext->handle(), batch_bmimgs[i].height, batch_bmimgs[i].width,
            batch_bmimgs[i].image_format, batch_bmimgs[i].data_type, &image_aligned, stride2);
        bm_image_alloc_dev_mem(image_aligned, BMCV_IMAGE_FOR_IN);
        bmcv_copy_to_atrr_t copyToAttr;
        memset(&copyToAttr, 0, sizeof(copyToAttr));
        copyToAttr.start_x = 0;
        copyToAttr.start_y = 0;
        copyToAttr.if_padding = 1;
        bmcv_image_copy_to(m_bmContext->handle(), copyToAttr, batch_bmimgs[i], image_aligned);
        int h = image_aligned.height;
        int w = image_aligned.width;
        float ratio = w / float(h);
        int resize_h;
        int resize_w;
        int padding_w;
        if(ratio > img_ratio.back()){
            resize_h = net_h_;
            resize_w = net_w_;
        }else{
            for(int i = 0; i < img_ratio.size(); i ++){
                if(ratio <= img_ratio[i]){
                    resize_h = img_size[i].h;
                    resize_w = (int)(resize_h * ratio);
                    padding_w = img_size[i].w;
                    break;
                }
            }
        }

        // resize + padding
        bmcv_padding_atrr_t padding_attr;
        memset(&padding_attr, 0, sizeof(padding_attr));
        padding_attr.dst_crop_sty = 0;
        padding_attr.dst_crop_stx = 0;
        padding_attr.dst_crop_w = resize_w;
        padding_attr.dst_crop_h = resize_h;
        padding_attr.padding_b = 0;
        padding_attr.padding_g = 0;
        padding_attr.padding_r = 0;
        padding_attr.if_memset = 1;
        bmcv_rect_t crop_rect{0, 0, image_aligned.width, image_aligned.height};
        bmcv_image_vpp_convert_padding(m_bmContext->handle(), 1, image_aligned, &resize_bmcv_map[stage_w][i], &padding_attr, &crop_rect);
        bm_image_destroy(image_aligned);
    }
    
    // converto
    bm_device_mem_t input_dev_mem;
    bmcv_image_convert_to(m_bmContext->handle(), batch_bmimgs.size(), linear_trans_param_, resize_bmcv_map[stage_w].data(), linear_trans_bmcv_map[stage_w].data());
    bm_image_get_contiguous_device_mem(batch_bmimgs.size(), linear_trans_bmcv_map[stage_w].data(), &input_dev_mem);
    
    // attach to responding input_tensor
    bool input_flag = false;
    for(int i = 0; i < m_bmNetwork->m_netinfo->stage_num; i++){
        std::shared_ptr<BMNNTensor> input_tensor = m_bmNetwork->inputTensor(0, i);
        if(input_tensor->get_shape()->dims[0] == batch_bmimgs.size() && input_tensor->get_shape()->dims[3] == stage_w){
            input_flag = true;
            input_tensor->set_device_mem(&input_dev_mem);
            input_tensor->set_shape_by_dim(0, batch_bmimgs.size());  // set real batch number
            stage = i;
            break;
        }
    }
    if(input_flag){
        return 0;
    }else{
        return -1;
    }
}

std::vector<std::string> PPOCR_Rec::ReadDict(const std::string &path){
  std::ifstream in(path);
  std::string line;
  std::vector<std::string> m_vec;
  if (in) {
    while (getline(in, line)) {
      m_vec.push_back(line);
    }
  } else {
    std::cout << "no such label file: " << path << ", exit the program..."
              << std::endl;
    exit(1);
  }
  return m_vec;
}

//get recognition results.
int PPOCR_Rec::post_process(std::vector<std::pair<std::string, float>>& results)
{
    int output_num = m_bmNetwork->outputTensorNum(); //Now only test for 1 output
    std::vector<std::shared_ptr<BMNNTensor>> outputTensors(output_num);
    for(int i=0; i < output_num; i++){
        outputTensors[i] = m_bmNetwork->outputTensor(i, stage);
        auto output_shape = outputTensors[i]->get_shape();
        auto output_dims = output_shape->num_dims;
        int batch_num = output_shape->dims[0];
        int outputdim_1 = output_shape->dims[1];
        int outputdim_2 = output_shape->dims[2];

        float* predict_batch = nullptr;
        predict_batch = (float*)outputTensors[i]->get_cpu_data();

        for(int m = 0; m < batch_num; m++)
        {
            std::string str_res;
            int* argmax_idx = new int[outputdim_1];
            float* max_value = new float[outputdim_1];
            for (int n = 0; n < outputdim_1; n++){
                argmax_idx[n] =
                    int(PPOCR_Rec::argmax(&predict_batch[(m * outputdim_1 + n) * outputdim_2],
                                        &predict_batch[(m * outputdim_1 + n + 1) * outputdim_2]));
                max_value[n] =
                    float(*std::max_element(&predict_batch[(m * outputdim_1 + n) * outputdim_2],
                                            &predict_batch[(m * outputdim_1 + n + 1) * outputdim_2]));
            }

            int last_index = 0;
            float score = 0.f;
            int count = 0;
            for (int n = 0; n < outputdim_1; n++) {
                // argmax_idx =
                //     int(PPOCR_Rec::argmax(&predict_batch[(m * outputdim_1 + n) * outputdim_2],
                //                         &predict_batch[(m * outputdim_1 + n + 1) * outputdim_2]));
                // max_value =
                //     float(*std::max_element(&predict_batch[(m * outputdim_1 + n) * outputdim_2],
                //                             &predict_batch[(m * outputdim_1 + n + 1) * outputdim_2]));
                
                if (argmax_idx[n] > 0 && (!(n > 0 && argmax_idx[n] == last_index))) {
                    score += max_value[n];
                    count += 1;
                    str_res += label_list_[argmax_idx[n]];
                }
                last_index = argmax_idx[n];
            }
            score /= count;
            if (isnan(score)){
                score = 0;
                str_res = "###";
            }
            results.push_back({str_res, score});
            free(argmax_idx);
            free(max_value);
        }
    }
    return 0;
}
