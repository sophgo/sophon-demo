#===----------------------------------------------------------------------===#
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#

import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union
import math
import os
import argparse
import time

import torch
import torch.nn as nn
from fairseq2.data import VocabularyInfo
from fairseq2.models.nllb.tokenizer import NllbTokenizer
from fairseq2.data import Collater, SequenceData, StringLike
from fairseq2.data.audio import AudioDecoder, WaveformToFbankConverter
from fairseq2.memory import MemoryBlock
from fairseq2.nn.padding import PaddingMask, get_seqs_and_padding_mask
from torch import Tensor
import sophon.sail as sail

from generator import (
    SequenceGeneratorOptions,
    UnitYGenerator,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s -- %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)

class Translator(nn.Module):
    def __init__(
        self,
        m4t_encoder_frontend_path: str,
        m4t_encoder_path: str,
        m4t_decoder_frontend_path: str,
        m4t_decoder_path: str,
        m4t_decoder_final_proj_path: str,
        tokenizer_model: str,
        dev_id: int,
        tgt_lang: str,
        max_output_seq_len: int,
        beam_size: int,
        text_generation_opts: SequenceGeneratorOptions
    ):
        super().__init__()
        self.handle = sail.Handle(dev_id)
        self.layer_num = 24
        self.max_input_len = 576 * 2
        self.beam_size = beam_size

        # NOTE: the process contains init seq input with len 1
        max_output_seq_len += 1
        device = torch.device("cpu")
        dtype = torch.float32

        print('m4t_unity_speech_encoder_frontend model loading...')
        # dynamic bmodel
        self.wav2Vec2Frontend_net = sail.Engine(m4t_encoder_frontend_path, dev_id, sail.IOMode.DEVIO)
        logging.info("load {} success!".format(m4t_encoder_frontend_path))
        print('m4t_unity_speech_encoder_frontend model loaded...')

        print('m4t_unity_speech_encoder model loading...')
        self.unitY_encoder_adaptor_net = sail.Engine(m4t_encoder_path, dev_id, sail.IOMode.DEVIO)
        logging.info("load {} success!".format(m4t_encoder_path))
        print('m4t_unity_speech_encoder model loaded...')

        print('m4t_decoder_frontend_beam_size_s2t model loading...')
        self.text_decoder_frontend_net = sail.Engine(m4t_decoder_frontend_path, dev_id, sail.IOMode.DEVIO)
        logging.info("load {} success!".format(m4t_decoder_frontend_path))
        print('m4t_decoder_frontend_beam_size_s2t model loaded...')

        print('m4t_decoder_beam_size_s2t model loading...')
        self.text_decoder_net = sail.Engine(m4t_decoder_path, dev_id, sail.IOMode.DEVIO)
        logging.info("load {} success!".format(m4t_decoder_path))
        print('m4t_decoder_beam_size_s2t model loaded...')

        print('m4t_decoder_final_proj_beam_size_s2t model loading...')
        self.final_proj_net = sail.Engine(m4t_decoder_final_proj_path, dev_id, sail.IOMode.DEVIO)
        logging.info("load {} success!".format(m4t_decoder_final_proj_path))
        print('m4t_decoder_final_proj_beam_size_s2t model loaded...')

        self.target_vocab_info = VocabularyInfo(size=256102, unk_idx=1, bos_idx=2, eos_idx=3, pad_idx=0)

        self.text_tokenizer = NllbTokenizer(tokenizer_model, ['afr', 'amh', 'arb', 'ary', 'arz', 'asm', 
                                                         'azj', 'bel', 'ben', 'bos', 'bul', 'cat', 
                                                         'ceb', 'ces', 'ckb', 'cmn', 'cmn_Hant', 
                                                         'cym', 'dan', 'deu', 'ell', 'eng', 'est', 
                                                         'eus', 'fin', 'fra', 'fuv', 'gaz', 'gle', 
                                                         'glg', 'guj', 'heb', 'hin', 'hrv', 'hun', 
                                                         'hye', 'ibo', 'ind', 'isl', 'ita', 'jav', 
                                                         'jpn', 'kan', 'kat', 'kaz', 'khk', 'khm', 
                                                         'kir', 'kor', 'lao', 'lit', 'lug', 'luo', 
                                                         'lvs', 'mai', 'mal', 'mar', 'mkd', 'mlt', 
                                                         'mni', 'mya', 'nld', 'nno', 'nob', 'npi', 
                                                         'nya', 'ory', 'pan', 'pbt', 'pes', 'pol', 
                                                         'por', 'ron', 'rus', 'sat', 'slk', 'slv', 
                                                         'sna', 'snd', 'som', 'spa', 'srp', 'swe', 
                                                         'swh', 'tam', 'tel', 'tgk', 'tgl', 'tha', 
                                                         'tur', 'ukr', 'urd', 'uzn', 'vie', 'yor', 
                                                         'yue', 'zsm', 'zul'], 'eng')
        
        self.tgt_lang = tgt_lang
        if text_generation_opts is None:
            text_generation_opts = SequenceGeneratorOptions(
                beam_size=self.beam_size, soft_max_seq_len=(1, 200)
            )
        self.text_generation_opts = text_generation_opts
        self.generator = UnitYGenerator(
            self.wav2Vec2Frontend_net,
            self.unitY_encoder_adaptor_net,
            self.text_decoder_frontend_net,
            self.text_decoder_net,
            self.final_proj_net,
            self.text_tokenizer,
            self.target_vocab_info,
            dev_id,
            tgt_lang,
            max_output_seq_len=max_output_seq_len,
            text_opts=text_generation_opts,
        )

        self.device = device
        self.decode_audio = AudioDecoder(dtype=torch.float32, device=device)
        self.convert_to_fbank = WaveformToFbankConverter(
            num_mel_bins=80,
            waveform_scale=2**15,
            channel_last=True,
            standardize=True,
            device=device,
            dtype=dtype,
        )
        self.collate = Collater(
            pad_value=self.text_tokenizer.vocab_info.pad_idx or 0, pad_to_multiple=2
        )

    def get_prediction(
        self,
        seqs: Tensor,
        padding_mask: Optional[PaddingMask],
        duration_factor: float = 1.0,
        prosody_encoder_input: Optional[SequenceData] = None,
    ) -> Tuple[List[StringLike], Optional[Tensor]]:

        return self.generator(
            seqs,
            padding_mask,
            duration_factor=duration_factor,
            prosody_encoder_input=prosody_encoder_input,
        )

    def predict(
        self,
        input: Union[str, dict],
        duration_factor: float = 1.0,
        prosody_encoder_input: Optional[SequenceData] = None,
    ) -> str:
        """
        The main method used to perform inference on s2t.

        :param input:
            path to audio or dict with normalized data.
        :param tgt_lang:
            Target language to decode into.

        :returns:
            - Translated text.
        """
        audio = input
        start_time = time.time()
        if isinstance(audio, str):
            with Path(audio).open("rb") as fb:
                block = MemoryBlock(fb.read())
            decoded_audio = self.decode_audio(block)
            src = self.collate(self.convert_to_fbank(decoded_audio))["fbank"]
            seqs, padding_mask = get_seqs_and_padding_mask(src)
        elif isinstance(audio, dict):
            src = self.collate(self.convert_to_fbank(audio))["fbank"]
            seqs, padding_mask = get_seqs_and_padding_mask(src)
        else:
            raise ValueError('input type {} is not str or torch.tensor type'.format(type(audio)))
        logging.info("load time cost(ms): {:.2f}".format((time.time() - start_time) * 1000))

        if seqs.shape[1] > self.max_input_len:
            logging.warning('offline input seqs len bigger than the max input len {} vs. {}, the input will be split to segments.'.format(seqs.shape[1], self.max_input_len))
            num_seg = math.ceil(seqs.shape[1] / self.max_input_len)
            new_seqs = []
            new_padding_mask = []
            for i in range(num_seg-1):
                new_seqs.append(seqs[:, i*self.max_input_len:(i+1)*self.max_input_len])
                new_padding_mask.append(None)
            diff = (num_seg-1)*self.max_input_len
            new_seqs.append(seqs[:, diff:])
            if padding_mask is None:
                new_padding_mask.append(None)
            else:
                new_padding_mask.append(PaddingMask(padding_mask.seq_lens-diff, padding_mask.batch_seq_len-diff))
            seqs = new_seqs
            padding_mask = new_padding_mask
        else:
            seqs = [seqs]
            padding_mask = [padding_mask]
        rec = ''
        for seq, pad in zip(seqs, padding_mask):
            texts, units = self.get_prediction(
                seq,
                pad,
                duration_factor=duration_factor,
                prosody_encoder_input=prosody_encoder_input,
            )
            rec += str(texts[0])
        return rec, None


def main(args):
    # check params
    if not os.path.exists(args.input):
        raise FileNotFoundError('{} is not existed.'.format(args.input))
    if not os.path.isdir(args.input):
        raise ValueError("{} is not a directory".format(args.input))
    
    bmodels_path = [args.encoder_frontend_bmodel, args.encoder_bmodel, args.tokenizer_model,
                    args.decoder_frontend_bmodel, 
                    args.decoder_bmodel,
                    args.decoder_final_proj_bmodel]
    for bmodel_path in bmodels_path:
        if not os.path.exists(bmodel_path):
            raise FileNotFoundError('{} is not existed.'.format(bmodel_path))
    
    # initialize net
    translator = Translator(
        args.encoder_frontend_bmodel,
        args.encoder_bmodel,
        args.decoder_frontend_bmodel,
        args.decoder_bmodel,
        args.decoder_final_proj_bmodel,
        args.tokenizer_model,
        args.dev_id,
        args.tgt_lang,
        args.max_output_seq_len,
        args.beam_size,
        None
    )
    
    cn = 0
    total_time = 0.0
    # test local audio
    if os.path.isdir(args.input): 
        results_list = []
        for root, dirs, filenames in os.walk(args.input):
            for filename in filenames:
                if os.path.splitext(filename)[-1].lower() not in ['.wav']:
                    continue
                audio_file = os.path.join(root, filename)
                cn += 1
                logging.info("{}, audio_file: {}".format(cn, audio_file))
                start_time = time.time()
                # predict
                text_output, _ = translator.predict(
                    input=audio_file,
                )
                total_time += time.time() - start_time
                logging.info("whole result: " + text_output)
                results_list.append({'filename' : filename, 'content' : text_output.strip()})

        # save results
        if not os.path.exists('results/'):
            os.makedirs('results/') 
        for result in results_list:
            with open(os.path.join('results/', result['filename'].split('/')[-1].split('.')[0]+'.txt'), 'w') as jf:
                jf.write(result['content'].strip())
        logging.info("result saved in {}".format('results dir'))
        
    # calculate speed  
    logging.info("------------------ Predict Time Info ----------------------")
    second_per_sample = total_time / (cn)
    logging.info("second_per_sample(ms): {:.2f}".format(second_per_sample * 1000))
    logging.info("total time cost(ms): {:.2f}".format(total_time * 1000))

def argsparser():
    parser = argparse.ArgumentParser(prog=__file__)
    parser.add_argument('--input', type=str, default='../datasets/aishell_S0764', help='path of input')
    parser.add_argument('--tgt_lang', type=str, default='cmn', help='output langauge')
    parser.add_argument('--encoder_frontend_bmodel', type=str, default="../models/BM1684X/m4t_encoder_frontend_fp16_s2t.bmodel", help='path of Wav2Vec2Frontend bmodel')
    parser.add_argument('--encoder_bmodel', type=str, default="../models/BM1684X/m4t_encoder_fp16_s2t.bmodel", help='path of UnitYEncoderAdaptor bmodel')
    parser.add_argument('--tokenizer_model', type=str, default='../models/tokenizer.model', help='path of tokenizer model')
    parser.add_argument('--decoder_frontend_bmodel', type=str, default="../models/BM1684X/m4t_decoder_frontend_beam_size_fp16_s2t.bmodel", help='path of text decoder frontend bmodel')
    parser.add_argument('--decoder_bmodel', type=str, default="../models/BM1684X/m4t_decoder_beam_size_fp16_s2t.bmodel", help='path of text decoder bmodel')
    parser.add_argument('--decoder_final_proj_bmodel', type=str, default="../models/BM1684X/m4t_decoder_final_proj_beam_size_fp16_s2t.bmodel", help='path of final proj bmodel')
    parser.add_argument('--max_output_seq_len', type=int, default=50, help='max seq output length')
    parser.add_argument('--beam_size', type=int, default=1, help='beam size')
    parser.add_argument('--dev_id', type=int, default=0, help='dev id')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = argsparser()
    main(args)
    print('all done.')