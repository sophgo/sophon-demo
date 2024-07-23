#===----------------------------------------------------------------------===#
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#   
from utils import sigmoid, get_phrases_from_posmap_np, create_positive_map_from_span
import numpy as np

class PostProcess():
    def __init__(self, caption, token_spans, tokenizer, box_threshold, text_threshold, with_logits=True):

        self.caption = caption
        self.token_spans = token_spans
        self.tokenizer = tokenizer
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.with_logits = with_logits
    
        # get phrase
        self.tokenized = self.tokenizer(self.caption)

        if self.token_spans is not None:
            self.positive_maps = create_positive_map_from_span(
                self.tokenized,
                token_span=self.token_spans
            ) # n_phrase, 256
        else:
            self.positive_maps = None

    def __call__(self, output):
        logits = sigmoid(output[0][0]) # (nq, 256)
        boxes = output[1][0]  # (nq, 4)
        # filter output
        if self.token_spans is None:
            logits_filt = logits
            boxes_filt = boxes

            filt_mask = np.max(logits_filt,axis=1) > self.box_threshold
            logits_filt = logits_filt[filt_mask]  # num_filt, 256
            boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
            
            # build pred
            pred_phrases = []
            for logit, _ in zip(logits_filt, boxes_filt):
                pred_phrase = get_phrases_from_posmap_np(logit > self.text_threshold, self.tokenized, self.tokenizer)
                if self.with_logits:
                    pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
                else:
                    pred_phrases.append(pred_phrase)
        else:
            # given-phrase mode
            logits_for_phrases = self.positive_maps @ logits.T # n_phrase, nq
            all_logits = []
            all_phrases = []
            all_boxes = []
            for (token_span, logit_phr) in zip(self.token_spans, logits_for_phrases):
                # get phrase
                phrase = ' '.join([self.caption[_s:_e] for (_s, _e) in token_span])
                # get mask
                filt_mask = logit_phr > self.box_threshold
                # filt box
                all_boxes.append(boxes[filt_mask])
                # filt logits
                all_logits.append(logit_phr[filt_mask])
                if self.with_logits:
                    logit_phr_num = logit_phr[filt_mask]
                    all_phrases.extend([phrase + f"({str(logit.item())[:4]})" for logit in logit_phr_num])
                else:
                    all_phrases.extend([phrase for _ in range(len(filt_mask))])
            boxes_filt = np.concatenate(all_boxes, axis=0)

            pred_phrases = all_phrases
        return boxes_filt, pred_phrases