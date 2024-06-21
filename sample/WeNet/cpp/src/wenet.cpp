//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include "wenet.h"
#include "wrapper.h"
#include "ctcdecode.h"
#define IGNORE_ID -1
using namespace bmruntime;

static std::string shape_to_str(const bm_shape_t& shape) {
    std::string str ="[ ";
    for(int i=0; i<shape.num_dims; i++){
      str += std::to_string(shape.dims[i]) + " ";
    }
    str += "]";
    return str;
}

void showInfo(const bm_net_info_t* m_netinfo)
{
    const char* dtypeMap[] = {
    "FLOAT32",
    "FLOAT16",
    "INT8",
    "UINT8",
    "INT16",
    "UINT16",
    "INT32",
    "UINT32",
    };
    printf("\n########################\n");
    printf("NetName: %s\n", m_netinfo->name);
    for(int s=0; s<m_netinfo->stage_num; s++){
        printf("---- stage %d ----\n", s);
        for(int i=0; i<m_netinfo->input_num; i++){
            auto shapeStr = shape_to_str(m_netinfo->stages[s].input_shapes[i]);
            printf("  Input %d) '%s' shape=%s dtype=%s scale=%g\n",
                i,
                m_netinfo->input_names[i],
                shapeStr.c_str(),
                dtypeMap[m_netinfo->input_dtypes[i]],
                m_netinfo->input_scales[i]);
        }
        for(int i=0; i<m_netinfo->output_num; i++){
            auto shapeStr = shape_to_str(m_netinfo->stages[s].output_shapes[i]);
            printf("  Output %d) '%s' shape=%s dtype=%s scale=%g\n",
                i,
                m_netinfo->output_names[i],
                shapeStr.c_str(),
                dtypeMap[m_netinfo->output_dtypes[i]],
                m_netinfo->output_scales[i]);
        }
    }
    printf("########################\n\n");

}
    
int WeNet::Init(const std::vector<std::string>& dict, int sample_frequency, int num_mel_bins, int frame_shift, int frame_length, int decoding_chunk_size, int subsampling_rate, int context, const std::string& mode) {
    this->dict = dict;
    this->sample_frequency = sample_frequency;
    this->num_mel_bins = num_mel_bins;
    this->frame_shift = frame_shift;
    this->frame_length = frame_length;
    this->decoding_chunk_size = decoding_chunk_size;
    this->subsampling_rate = subsampling_rate;
    this->context = context;
    this->mode = mode;

    // create Network
    std::vector<const char *> network_names;
    encoder_ctx->get_network_names(&network_names);
    encoder_net = std::make_shared<Network>(*encoder_ctx, network_names[0], 0); // use stage[0]
    assert(encoder_net->info()->input_num == 6 || encoder_net->info()->input_num == 2);
    // showInfo(encoder_net->info());
    // Initialize the memory space required for the input and output tensors
    encoder_inputs = encoder_net->Inputs();
    encoder_outputs = encoder_net->Outputs();

    if(encoder_inputs.size() == 2){ // non streaming
        const bm_tensor_t * log_probs_bm_tensor_t = encoder_outputs[3]->tensor();
        batch_size = log_probs_bm_tensor_t->shape.dims[0];
        out_size = log_probs_bm_tensor_t->shape.dims[1];
        beam_size = log_probs_bm_tensor_t->shape.dims[2];
        out_length = encoder_outputs[0]->tensor()->shape.dims[2];
        assert(batch_size == 1 && "Streaming inference only supports batch size = 1!");
    } else {
        const bm_tensor_t * log_probs_bm_tensor_t = encoder_outputs[0]->tensor();
        batch_size = log_probs_bm_tensor_t->shape.dims[0];
        out_size = log_probs_bm_tensor_t->shape.dims[1];
        beam_size = log_probs_bm_tensor_t->shape.dims[2];
        out_length = encoder_outputs[2]->tensor()->shape.dims[2];
        assert(batch_size == 1 && "Streaming inference only supports batch size = 1!");
    }

    if(mode == "attention_rescoring") {
        decoder_ctx->get_network_names(&network_names);
        decoder_net = std::make_shared<Network>(*decoder_ctx, network_names[0], 0); // use stage[0]
        assert(decoder_net->info()->input_num == 6);

        // Initialize the memory space required for the input and output tensors
        decoder_inputs = decoder_net->Inputs();
        decoder_outputs = decoder_net->Outputs();

        max_len = decoder_inputs[0]->tensor()->shape.dims[1] - 2;  
    }

    return 0;
}

int WeNet::pre_process(const char* file_path) {
    LOG_TS(m_ts, "wenet load file");
    SF_INFO sfinfo;
    SNDFILE* sndfile = sf_open(file_path, SFM_READ, &sfinfo);

    if (sndfile == NULL) {
        std::cerr << "Error opening WAV file: " << sf_strerror(sndfile) << std::endl;
        return 1;
    }

    total_data_time += (float)sfinfo.frames / (float)sfinfo.samplerate;
    // std::cout << "Sample rate: " << sfinfo.samplerate << std::endl;
    // std::cout << "Channels: " << sfinfo.channels << std::endl;
    // std::cout << "Frames: " << sfinfo.frames << std::endl;

    // Read the samples into a buffer
    const int buffer_size = sfinfo.frames * sfinfo.channels;
    double buffer[buffer_size];
    sf_read_double(sndfile, buffer, buffer_size);
    // sf_count_t samples_read = sf_read_double(sndfile, buffer, buffer_size);
    // std::cout << "Samples read: " << samples_read << std::endl;

    // Close the file
    sf_close(sndfile);

    arma::fmat waveform = arma::fmat(sfinfo.channels, sfinfo.frames);
    if(sfinfo.channels == 1) {
        for (int i = 0; i < sfinfo.frames; i++) {
            waveform(0, i) = buffer[i] * (1 << 15);
        }
    }
    else if(sfinfo.channels == 2) {
        for (int i = 0; i < sfinfo.frames; i += 2) {
            waveform(0, i / 2) = buffer[i] * (1 << 15);
            waveform(0, i / 2) = buffer[i + 1] * (1 << 15);
        }       
    }
    else {
        std::cerr << "The number of channels in the wav file is not normal!" << std::endl;
        return 1;
    }
    LOG_TS(m_ts, "wenet load file");

    LOG_TS(m_ts, "wenet fbank");
    feats = fbank(waveform, num_mel_bins, frame_length, frame_shift, sample_frequency, 0.0, 0.0, true, true, false);
    // todo:spec_sub
    LOG_TS(m_ts, "wenet fbank");

    return 0;
}

int WeNet::inference() {
    result = "";
    arma::fmat encoder_out;
    void *chunk_out_lens_ptr=NULL;
    arma::fmat beam_log_probs;
    arma::Mat<int32_t> beam_log_probs_idx;
    if(encoder_inputs.size() == 2){ // non streaming
        int encoder_feat_len = encoder_inputs[0]->tensor()->shape.dims[1];
        int input_feat_len = feats.n_rows;
        if(encoder_feat_len < input_feat_len){
            std::cout << "Skipping this audio, input feat length exceed bmodel's input shape: feat_length " << input_feat_len << " > bmodel_input_shape " << encoder_feat_len << std::endl;
            return 0;
        } else{
            if(encoder_feat_len > input_feat_len){
                arma::fmat pad_zeros(encoder_feat_len - input_feat_len, feats.n_cols, arma::fill::zeros);
                feats = arma::join_cols(feats, pad_zeros);
            }
            void* speech = mat_to_sys_mem<float>(feats);
            encoder_inputs[0]->CopyFrom(speech);
            encoder_inputs[1]->CopyFrom((void*)&input_feat_len);
            LOG_TS(m_ts, "wenet encoder inference");
            bool status = encoder_net->Forward();
            LOG_TS(m_ts, "wenet encoder inference");
            assert(BM_SUCCESS == status);

            void* encoder_out_ptr = calloc(encoder_outputs[0]->num_elements(), sizeof(float));
            chunk_out_lens_ptr = calloc(encoder_outputs[1]->num_elements(), sizeof(float));
            void* beam_log_probs_ptr = calloc(encoder_outputs[3]->num_elements(), sizeof(float));
            void* beam_log_probs_idx_ptr = calloc(encoder_outputs[4]->num_elements(), sizeof(int32_t));

            encoder_outputs[0]->CopyTo(encoder_out_ptr);
            encoder_outputs[1]->CopyTo(chunk_out_lens_ptr);
            encoder_outputs[3]->CopyTo(beam_log_probs_ptr);
            encoder_outputs[4]->CopyTo(beam_log_probs_idx_ptr);
            
            std::vector<std::vector<std::pair<double, std::vector<int>>>> score_hyps;
            LOG_TS(m_ts, "wenet postprocess");
            std::vector<std::string> hyps = ctc_decoding(beam_log_probs_ptr, beam_log_probs_idx_ptr, chunk_out_lens_ptr, beam_size, batch_size, dict, score_hyps);
            LOG_TS(m_ts, "wenet postprocess");

            encoder_out = arma::join_cols(encoder_out, sys_mem_to_mat<float>(encoder_out_ptr, out_size, out_length));
            beam_log_probs = arma::join_cols(beam_log_probs, sys_mem_to_mat<float>(beam_log_probs_ptr, out_size, beam_size));
            beam_log_probs_idx = arma::join_cols(beam_log_probs_idx, sys_mem_to_mat<int32_t>(beam_log_probs_idx_ptr, out_size, beam_size));
            result += hyps[0];
            
            std::free(speech);
            std::free(encoder_out_ptr);
            std::free(beam_log_probs_ptr);
            std::free(beam_log_probs_idx_ptr);
        }
    }else{ // streaming
        void* offset = calloc(encoder_inputs[2]->num_elements(), sizeof(int32_t));
        void* att_cache = calloc(encoder_inputs[3]->num_elements(), sizeof(float));
        void* cnn_cache = calloc(encoder_inputs[4]->num_elements(), sizeof(float));
        void* cache_mask = calloc(encoder_inputs[5]->num_elements(), sizeof(float));

        void* log_probs = calloc(encoder_outputs[0]->num_elements(), sizeof(float));
        void* log_probs_idx = calloc(encoder_outputs[1]->num_elements(), sizeof(int32_t));
        void* chunk_out = calloc(encoder_outputs[2]->num_elements(), sizeof(float));
        void* chunk_out_lens = calloc(encoder_outputs[3]->num_elements(), sizeof(float));

        // inference
        int num_frames = feats.n_rows;
        int stride = subsampling_rate * decoding_chunk_size;
        int decoding_window = (decoding_chunk_size - 1) * subsampling_rate + context;
        
        for(int cur = 0; cur < num_frames - context + 1; cur += stride) {
            int end = std::min(cur + decoding_window, num_frames);
            arma::fmat chunk_xs = feats.submat(cur, 0, end - 1, feats.n_cols - 1);
            // pad if needed
            if((int)chunk_xs.n_rows < decoding_window) {
                arma::fmat pad_zeros(decoding_window - chunk_xs.n_rows, chunk_xs.n_cols, arma::fill::zeros);
                chunk_xs = arma::join_cols(chunk_xs, pad_zeros);
            }
            
            void* chunk_xs_ptr = mat_to_sys_mem<float>(chunk_xs);
            int chunk_lens = chunk_xs.n_rows;
            int* chunk_lens_ptr = (int*) malloc(sizeof(int));
            if (chunk_lens_ptr != nullptr) {
                *chunk_lens_ptr = chunk_lens;
            }
            else {
                std::cerr << "Failed to request memory space" << std::endl;
                exit(1);
            }
            void* void_chunk_lens_ptr = chunk_lens_ptr;

            //Ensure your bmodel's input names encount following order.
            encoder_inputs[0]->CopyFrom(chunk_xs_ptr);
            encoder_inputs[1]->CopyFrom(void_chunk_lens_ptr);
            encoder_inputs[2]->CopyFrom(offset);
            encoder_inputs[3]->CopyFrom(att_cache);
            encoder_inputs[4]->CopyFrom(cnn_cache);
            encoder_inputs[5]->CopyFrom(cache_mask);
            LOG_TS(m_ts, "wenet encoder inference");
            bool status = encoder_net->Forward();
            LOG_TS(m_ts, "wenet encoder inference");
            assert(BM_SUCCESS == status);
        
            //Ensure your bmodel's output names encount following order.
            encoder_outputs[0]->CopyTo(log_probs);
            encoder_outputs[1]->CopyTo(log_probs_idx);
            encoder_outputs[2]->CopyTo(chunk_out);
            encoder_outputs[3]->CopyTo(chunk_out_lens);
            encoder_outputs[4]->CopyTo(offset);
            encoder_outputs[5]->CopyTo(att_cache);
            encoder_outputs[6]->CopyTo(cnn_cache);
            encoder_outputs[7]->CopyTo(cache_mask);
            
            std::vector<std::vector<std::pair<double, std::vector<int>>>> score_hyps;
            LOG_TS(m_ts, "wenet postprocess");
            std::vector<std::string> hyps = ctc_decoding(log_probs, log_probs_idx, chunk_out_lens, beam_size, batch_size, dict, score_hyps);
            LOG_TS(m_ts, "wenet postprocess");
            // std::cout << hyps[0] << std::endl;
            result += hyps[0];
            encoder_out = arma::join_cols(encoder_out, sys_mem_to_mat<float>(chunk_out, out_size, out_length));
            beam_log_probs = arma::join_cols(beam_log_probs, sys_mem_to_mat<float>(log_probs, out_size, beam_size));
            beam_log_probs_idx = arma::join_cols(beam_log_probs_idx, sys_mem_to_mat<int32_t>(log_probs_idx, out_size, beam_size));

            std::free(chunk_lens_ptr);
            std::free(chunk_xs_ptr);
        }
    
        std::free(att_cache);
        std::free(cnn_cache);
        std::free(cache_mask);
        std::free(offset);

        std::free(log_probs);
        std::free(log_probs_idx);
        std::free(chunk_out);
        std::free(chunk_out_lens);
    }

    if(mode == "attention_rescoring" && max_len + 2 >= encoder_out.n_rows) {
        LOG_TS(m_ts, "wenet preprocess");
        int eos = dict.size() - 1;
        int sos = dict.size() - 1;
        std::vector<std::vector<std::pair<double, std::vector<int>>>> score_hyps;
        
        if(encoder_inputs.size() > 2){ //streaming
            chunk_out_lens_ptr = calloc(encoder_outputs[3]->num_elements(), sizeof(float));
            *(float*)chunk_out_lens_ptr = (float)beam_log_probs.n_rows;
        }
        void* beam_log_probs_ptr = mat_to_sys_mem(beam_log_probs);
        void* beam_log_probs_idx_ptr = mat_to_sys_mem<int32_t>(beam_log_probs_idx);
        ctc_decoding(beam_log_probs_ptr, beam_log_probs_idx_ptr, chunk_out_lens_ptr, beam_size, batch_size, dict, score_hyps, mode);
        
        std::free(chunk_out_lens_ptr);

        std::vector<std::vector<float>> ctc_score;
        std::vector<std::vector<int>> all_hyps;
        for(auto& hyps : score_hyps) {
            int cur_len = hyps.size();
            if(cur_len < beam_size) {
                for(int i = 0; i < beam_size - cur_len; i++) {
                    hyps.push_back(std::make_pair(-INFINITY, std::vector<int>{0}));
                }
            }
            std::vector<float> cur_ctc_score;
            for(auto& hyp : hyps) {
                cur_ctc_score.push_back((float)hyp.first);
                all_hyps.push_back(hyp.second);
            }
            ctc_score.push_back(cur_ctc_score);
        }

        arma::Mat<int> hyps_pad_sos_eos = arma::Mat<int>(beam_size, max_len + 2, arma::fill::ones) * IGNORE_ID;
        arma::Mat<int> r_hyps_pad_sos_eos = arma::Mat<int>(beam_size, max_len + 2, arma::fill::ones) * IGNORE_ID;
        arma::Row<int> hyps_lens_sos = arma::Row<int>(beam_size, arma::fill::ones);

        int k = 0;
        for(int j = 0; j < beam_size; j++) {
            std::vector<int> cand = all_hyps[k];
            arma::Row<int> cand_rowvec = arma::conv_to<arma::Row<int>>::from(cand);

            int l = cand.size() + 2;
            hyps_pad_sos_eos(j, 0) = sos;
            hyps_pad_sos_eos(j, l - 1) = eos;
            r_hyps_pad_sos_eos(j, 0) = sos;
            r_hyps_pad_sos_eos(j, l - 1) = eos;
            if(l > 2) {
                hyps_pad_sos_eos.row(j).subvec(1, l - 2) = cand_rowvec;
                r_hyps_pad_sos_eos.row(j).subvec(1, l - 2) = arma::reverse(cand_rowvec);
            }

            hyps_lens_sos(j) = cand.size() + 1;
            k += 1;
        }

        // std::cout<<"max_len:"<<max_len<<"; encoder_out.n_rows:"<<encoder_out.n_rows<<"; encoder_out.n_cols:"<<encoder_out.n_cols<<std::endl;
        if(max_len + 2 > encoder_out.n_rows){
            arma::fmat pad_matrix(max_len + 2 - encoder_out.n_rows, encoder_out.n_cols, arma::fill::zeros);
            encoder_out = arma::join_cols(encoder_out, pad_matrix);
        }
        void* encoder_out_ptr = mat_to_sys_mem<float>(encoder_out);
        int encoder_out_lens = max_len + 2;
        void* encoder_out_lens_ptr = static_cast<void*>(&encoder_out_lens);
        void* hyps_pad_sos_eos_ptr = mat_to_sys_mem<int>(hyps_pad_sos_eos);
        void* hyps_lens_sos_ptr = mat_to_sys_mem<int>(hyps_lens_sos);
        void* r_hyps_pad_sos_eos_ptr = mat_to_sys_mem<int>(r_hyps_pad_sos_eos);
        arma::fmat ctc_score_fmat(ctc_score.size(), ctc_score[0].size());
        for(arma::uword i = 0; i < ctc_score_fmat.n_rows; i++) {
            ctc_score_fmat.row(i) = arma::frowvec(ctc_score[i]);
        }
        void* ctc_score_ptr = mat_to_sys_mem<float>(ctc_score_fmat);

        decoder_inputs[0]->CopyFrom(encoder_out_ptr);
        decoder_inputs[1]->CopyFrom(encoder_out_lens_ptr);
        decoder_inputs[2]->CopyFrom(hyps_pad_sos_eos_ptr);
        decoder_inputs[3]->CopyFrom(hyps_lens_sos_ptr);
        decoder_inputs[4]->CopyFrom(r_hyps_pad_sos_eos_ptr);
        decoder_inputs[5]->CopyFrom(ctc_score_ptr);
        LOG_TS(m_ts, "wenet preprocess");
        LOG_TS(m_ts, "wenet decoder inference");
        auto status = decoder_net->Forward();
        LOG_TS(m_ts, "wenet decoder inference");
        assert(BM_SUCCESS == status);

        LOG_TS(m_ts, "wenet postprocess");
        void* best_idx = calloc(decoder_outputs[0]->num_elements(), sizeof(int));
        decoder_outputs[0]->CopyTo(best_idx);

        k = 0;
        std::vector<std::vector<int>> best_sents;
        int* best_idx_int_ptr = static_cast<int*>(best_idx);
        // std::cout<< decoder_outputs[0]->num_elements() << std::endl;
        
        for(uint64_t i = 0; i < decoder_outputs[0]->num_elements(); i++) {
            best_sents.push_back(all_hyps[k + *(best_idx_int_ptr + i)]);
            k += beam_size;
        }
        
        int num_cores = std::thread::hardware_concurrency();
        size_t num_processes = std::min(num_cores, batch_size);
        std::vector<std::string> hyps = map_batch(best_sents, dict, num_processes);
        result = hyps[0];

        std::free(encoder_out_ptr);
        std::free(hyps_pad_sos_eos_ptr);
        std::free(hyps_lens_sos_ptr);
        std::free(r_hyps_pad_sos_eos_ptr);
        std::free(ctc_score_ptr);
        std::free(beam_log_probs_ptr);
        std::free(beam_log_probs_idx_ptr);
        std::free(best_idx);
        LOG_TS(m_ts, "wenet postprocess");
    } else if(max_len + 2 >= encoder_out.n_rows){
        std::cout<<"encoder_output's length is too long for decoder, skip decoder part..."<<std::endl;
    }

    
    return 0;
}

std::string WeNet::Recognize(const char* file_path) {
    LOG_TS(m_ts, "wenet preprocess");
    pre_process(file_path);
    LOG_TS(m_ts, "wenet preprocess");

    inference();
    return result;
}

void WeNet::enableProfile(TimeStamp *ts) {
    m_ts = ts;
}
