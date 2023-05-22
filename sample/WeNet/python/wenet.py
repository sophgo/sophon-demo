#===----------------------------------------------------------------------===#
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
"""
It requires a python wrapped c++ ctc decoder.
Please install it by following:
https://github.com/Slyne/ctc_decoder.git
"""

from __future__ import print_function

import argparse
import copy
import logging
import os
import sys
sys.dont_write_bytecode = True
sys.path.append(os.getcwd()+"/swig_decoders")

import torch
import yaml
from torch.utils.data import DataLoader

from dataset.dataset import Dataset
from utils.common import IGNORE_ID
from utils.file_utils import read_symbol_table
from utils.file_utils import read_lists

import multiprocessing
import numpy as np

from swig_decoders import map_batch,ctc_beam_search_decoder_batch,TrieVector, PathTrie
from utils.sophon_inference import SophonInference

import contextlib
import wave
import time

import logging
logging.basicConfig(level=logging.INFO)

def ctc_decoding(beam_log_probs, beam_log_probs_idx, encoder_out_lens, vocabulary, mode='ctc_prefix_beam_search'):
    beam_size = beam_log_probs.shape[-1]
    batch_size = beam_log_probs.shape[0]
    num_processes = min(multiprocessing.cpu_count(), batch_size)
    hyps = []
    score_hyps = []
    
    if mode == 'ctc_greedy_search':
        if beam_size != 1:
            log_probs_idx = beam_log_probs_idx[:, :, 0]
        batch_sents = []
        for idx, seq in enumerate(log_probs_idx):
            batch_sents.append(seq[0:encoder_out_lens[idx]].tolist())
        hyps = map_batch(batch_sents, vocabulary, num_processes,
                         True, 0)
    elif mode in ('ctc_prefix_beam_search', "attention_rescoring"):
        batch_log_probs_seq_list = beam_log_probs.tolist()
        batch_log_probs_idx_list = beam_log_probs_idx.tolist()
        batch_len_list = encoder_out_lens.tolist()
        batch_log_probs_seq = []
        batch_log_probs_ids = []
        batch_start = []  # only effective in streaming deployment
        batch_root = TrieVector()
        root_dict = {}
        for i in range(len(batch_len_list)):
            num_sent = batch_len_list[i]
            batch_log_probs_seq.append(
                batch_log_probs_seq_list[i][0:num_sent])
            batch_log_probs_ids.append(
                batch_log_probs_idx_list[i][0:num_sent])
            root_dict[i] = PathTrie()
            batch_root.append(root_dict[i])
            batch_start.append(True)
        score_hyps = ctc_beam_search_decoder_batch(batch_log_probs_seq,
                                                   batch_log_probs_ids,
                                                   batch_root,
                                                   batch_start,
                                                   beam_size,
                                                   num_processes,
                                                   0, -2, 0.99999)
        if mode == 'ctc_prefix_beam_search': 
            for cand_hyps in score_hyps:
                hyps.append(cand_hyps[0][1])
            hyps = map_batch(hyps, vocabulary, num_processes, False, 0)
    return hyps, score_hyps
            
def adjust_feature_length(feats, length, padding_value=0):
    # Adjust the length of the feature to a uniform length
    # feats: B*T*L tensor, where B is the batch size, T is the length of the feature, and L is the size of the feature
    # length: Length of adjusted T
    if feats.shape[1] < length:
        B, T, L = feats.shape
        tmp = np.full([B, length-T, L], padding_value)
        feats = np.concatenate((feats, tmp), axis=1)
    elif feats.shape[1] > length:
        feats = feats[:, :length, :]
    return feats

def calculate_total_time(data_list):
    # Calculate the total duration of all wav data files to measure the inference speed of the model
    lists = read_lists(data_list)
    total_time = 0
    for _, list in enumerate(lists):
        list = eval(list)
        wav_file_path = list["wav"]
        with contextlib.closing(wave.open(wav_file_path, 'r')) as f:
            frames = f.getnframes()
            rate = f.getframerate()
            total_time += frames / float(rate)
    return total_time
    
def get_args():
    parser = argparse.ArgumentParser(description='recognize with your model')
    
    parser.add_argument('--input', default='../datasets/aishell_S0764/aishell_S0764.list', help='path of input')
    parser.add_argument('--encoder_bmodel', default='../models/BM1684/wenet_encoder_fp32.bmodel', help='path of encoder bmodel')
    parser.add_argument('--decoder_bmodel', default='', help='path of decoder bmodel')
    parser.add_argument('--dev_id', type=int, default=0, help='dev id')
    parser.add_argument('--result_file', default='./result.txt', help='asr result file')
    parser.add_argument('--mode',
                        choices=[
                            'ctc_greedy_search', 'ctc_prefix_beam_search', 'attention_rescoring'],
                        default='ctc_prefix_beam_search',
                        help='decoding mode')
    parser.add_argument('--data_type',
                        default='raw',
                        choices=['raw', 'shard'],
                        help='train and cv data type')
    parser.add_argument('--dict', default='../config/lang_char.txt', help='dict file')
    parser.add_argument('--config', default='../config/train_u2++_conformer.yaml', help='config file')
    parser.add_argument('--decoder_len',
                        type=int,
                        default=350,
                        help='maximum length supported by decoder')
    args = parser.parse_args()
    # print(args)
    return args

if __name__ == '__main__':
    batch_size = 1
    subsampling = 4
    context = 7
    decoding_chunk_size = 16
    num_decoding_left_chunks = 5

    args = get_args()
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s')

    with open(args.config, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)

    reverse_weight = configs["model_conf"].get("reverse_weight", 0.0)
    symbol_table = read_symbol_table(args.dict)
    test_conf = copy.deepcopy(configs['dataset_conf'])
    test_conf['filter_conf']['max_length'] = 102400
    test_conf['filter_conf']['min_length'] = 0
    test_conf['filter_conf']['token_max_length'] = 102400
    test_conf['filter_conf']['token_min_length'] = 0
    test_conf['filter_conf']['max_output_input_ratio'] = 102400
    test_conf['filter_conf']['min_output_input_ratio'] = 0
    test_conf['speed_perturb'] = False
    test_conf['spec_aug'] = False
    test_conf['spec_sub'] = False
    test_conf['spec_trim'] = False
    test_conf['shuffle'] = False
    test_conf['sort'] = False
    test_conf['fbank_conf']['dither'] = 0.0
    test_conf['batch_conf']['batch_type'] = "static"
    test_conf['batch_conf']['batch_size'] = batch_size
    
    start_time = time.time()
    test_dataset = Dataset(args.data_type,
                           args.input,
                           symbol_table,
                           test_conf,
                           bpe_model=None,
                           partition=False)
    preprocess_time = time.time() - start_time

    test_data_loader = DataLoader(test_dataset, batch_size=None, num_workers=0)

    # Init encoder and decoder
    encoder = SophonInference(model_path=args.encoder_bmodel, device_id=args.dev_id, input_mode=0)
    decoder = None
    if(args.mode == 'attention_rescoring'):
        decoder = SophonInference(model_path=args.decoder_bmodel, device_id=args.dev_id, input_mode=0)

    # Load dict
    vocabulary = []
    char_dict = {}
    with open(args.dict, 'r') as fin:
        for line in fin:
            arr = line.strip().split()
            assert len(arr) == 2
            char_dict[int(arr[1])] = arr[0]
            vocabulary.append(arr[0])
    eos = sos = len(char_dict) - 1
    
    stride = subsampling * decoding_chunk_size
    decoding_window = (decoding_chunk_size - 1) * subsampling + context        
    required_cache_size = decoding_chunk_size * num_decoding_left_chunks

    output_size = configs["encoder_conf"]["output_size"]
    num_layers = configs["encoder_conf"]["num_blocks"]
    cnn_module_kernel = configs["encoder_conf"].get("cnn_module_kernel", 1) - 1
    head = configs["encoder_conf"]["attention_heads"]
    d_k = configs["encoder_conf"]["output_size"] // head
    
    encoder_inference_time = 0.0
    decoder_inference_time = 0.0
    postprocess_time = 0.0
    # Start speech recognition
    with torch.no_grad(), open(args.result_file, 'w') as fout:
        for _, batch in enumerate(test_data_loader):
            keys, feats, _, feats_lengths, _ = batch
            feats, feats_lengths = feats.numpy(), feats_lengths.numpy()
            
            supplemental_batch_size = batch_size - feats.shape[0]
            
            att_cache = np.zeros((batch_size, num_layers, head, required_cache_size, d_k * 2), dtype=np.float32)
            cnn_cache = np.zeros((batch_size, num_layers, output_size, cnn_module_kernel), dtype=np.float32)
            cache_mask = np.zeros((batch_size, 1, required_cache_size), dtype=np.float32)
            offset = np.zeros((batch_size, 1), dtype=np.int32)
            
            encoder_out = []
            beam_log_probs = []
            beam_log_probs_idx = []
            
            num_frames = feats.shape[1]
            result = ""
            for cur in range(0, num_frames - context + 1, stride):
                end = min(cur + decoding_window, num_frames)
                chunk_xs = feats[:, cur:end, :]
                if chunk_xs.shape[1] < decoding_window:
                    chunk_xs = adjust_feature_length(chunk_xs, decoding_window, padding_value=0)
                    chunk_xs = chunk_xs.astype(np.float32)
                chunk_lens = np.full(batch_size, fill_value=chunk_xs.shape[1], dtype=np.int32)

                encoder_input = [chunk_lens, att_cache, cnn_cache, chunk_xs, cache_mask, offset]
                start_time = time.time()
                out_dict = encoder.infer_numpy(encoder_input)
                encoder_inference_time += time.time() - start_time
                
                chunk_out = out_dict["chunk_out"]
                chunk_log_probs = out_dict["log_probs"]
                chunk_log_probs_idx = out_dict["log_probs_idx"].astype(np.int32)
                att_cache = out_dict['r_att_cache']
                cnn_cache = out_dict['r_cnn_cache']
                offset = out_dict['r_offset'].astype(np.int32)
                cache_mask = out_dict['r_cache_mask']
                chunk_out_lens = out_dict['chunk_out_lens'].astype(np.int32)

                encoder_out.append(chunk_out)
                beam_log_probs.append(chunk_log_probs)
                beam_log_probs_idx.append(chunk_log_probs_idx)
                
                # ctc decode
                start_time = time.time()
                chunk_hyps, _ = ctc_decoding(chunk_log_probs, chunk_log_probs_idx, chunk_out_lens, vocabulary)
                postprocess_time += time.time() - start_time
                print(chunk_hyps)
                result += chunk_hyps[0]
           
            encoder_out = np.concatenate(encoder_out, axis=1)
            encoder_out_lens = np.full(batch_size, fill_value=encoder_out.shape[1], dtype=np.int32)
            beam_log_probs = np.concatenate(beam_log_probs, axis=1)
            beam_log_probs_idx = np.concatenate(beam_log_probs_idx, axis=1)
            
            if args.mode == 'attention_rescoring':
                start_time = time.time()
                beam_size = beam_log_probs.shape[-1]
                batch_size = beam_log_probs.shape[0]
                num_processes = min(multiprocessing.cpu_count(), batch_size)
                hyps, score_hyps = ctc_decoding(beam_log_probs, beam_log_probs_idx, encoder_out_lens, vocabulary, args.mode)
                ctc_score, all_hyps = [], []
                max_len = 0
                for hyps in score_hyps:
                    cur_len = len(hyps)
                    if len(hyps) < beam_size:
                        hyps += (beam_size - cur_len) * [(-float("INF"), (0,))]
                    cur_ctc_score = []
                    for hyp in hyps:
                        cur_ctc_score.append(hyp[0])
                        all_hyps.append(list(hyp[1]))
                        if len(hyp[1]) > max_len:
                            max_len = len(hyp[1])
                    ctc_score.append(cur_ctc_score)
                    ctc_score = np.array(ctc_score, dtype=np.float32)
                max_len = decoder_len - 2
                hyps_pad_sos_eos = np.ones(
                    (batch_size, beam_size, max_len + 2), dtype=np.int64) * IGNORE_ID
                r_hyps_pad_sos_eos = np.ones(
                    (batch_size, beam_size, max_len + 2), dtype=np.int64) * IGNORE_ID
                hyps_lens_sos = np.ones((batch_size, beam_size), dtype=np.int32)
                k = 0
                for i in range(batch_size):
                    for j in range(beam_size):
                        cand = all_hyps[k]
                        l = len(cand) + 2
                        hyps_pad_sos_eos[i][j][0:l] = [sos] + cand + [eos]
                        r_hyps_pad_sos_eos[i][j][0:l] = [sos] + cand[::-1] + [eos]
                        hyps_lens_sos[i][j] = len(cand) + 1
                        k += 1
                encoder_out = np.pad(encoder_out, [(0, 0),(0, decoder_len - encoder_out.shape[1]), (0, 0)], mode='constant', constant_values=0)
                hyps_pad_sos_eos = hyps_pad_sos_eos.astype(np.int32)
                r_hyps_pad_sos_eos = r_hyps_pad_sos_eos.astype(np.int32)
                encoder_out_lens = np.full(batch_size, fill_value=encoder_out.shape[1], dtype=np.int32)
                decoder_input = [encoder_out, encoder_out_lens, hyps_pad_sos_eos, hyps_lens_sos, r_hyps_pad_sos_eos, ctc_score]
                postprocess_time += time.time() - start_time

                start_time = time.time()
                out_dict = decoder.infer_numpy(decoder_input)
                decoder_inference_time += time.time() - start_time

                start_time = time.time()  
                best_index = out_dict["best_index"].astype(np.int32)
                best_sents = []
                k = 0
                for idx in best_index:
                    cur_best_sent = all_hyps[k: k + beam_size][idx]
                    best_sents.append(cur_best_sent)
                    k += beam_size
                hyps = map_batch(best_sents, vocabulary, num_processes)
                postprocess_time += time.time() - start_time
            
            for i, key in enumerate(keys):
                content = None
                if args.mode == 'attention_rescoring':
                    content = hyps[i]
                else:
                    content = result
                logging.info('{} {}'.format(key, content))
                fout.write('{} {}\n'.format(key, content))
                
    logging.info("------------------ Predict Time Info ----------------------")
    total_data_time = calculate_total_time(args.input)
    logging.info("preprocess_time(ms): {:.4f}".format((preprocess_time / total_data_time) * 1000))
    logging.info("encoder_inference_time(ms): {:.4f}".format((encoder_inference_time / total_data_time) * 1000))
    logging.info("decoder_inference_time(ms): {:.4f}".format((decoder_inference_time / total_data_time) * 1000))
    logging.info("postprocess_time(ms): {:.4f}".format((postprocess_time / total_data_time) * 1000))
