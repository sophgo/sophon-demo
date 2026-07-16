"""Sliding-window no-repeat-n-gram logit processor.

Ported verbatim from the original Unlimited-OCR modeling code
(modeling_unlimitedocr.py: SlidingWindowNoRepeatNgramProcessor), which is
aligned with SGLang's DeepseekOCRNoRepeatNGramLogitProcessor.

This is a *logit* processor: it bans the next-token candidates that would
complete an already-seen n-gram within a sliding window. To use it, the
bmodel must expose logits (compile with `--do_sample` so lm_head returns
logits + a greedy_head does argmax); see unlimited_ocr_sail.py.
"""

import numpy as np


class SlidingWindowNoRepeatNgramProcessor:
    """Block n-gram repetitions within a sliding window."""

    def __init__(self, ngram_size, window, whitelist_token_ids=None):
        self.ngram_size = ngram_size
        self.window = window
        self.whitelist = set(whitelist_token_ids) if whitelist_token_ids else set()

    def __call__(self, input_ids, scores):
        """input_ids: [batch, seq] int (np.ndarray or list of list).
        scores: [batch, vocab] float logits (np.ndarray), mutated in place."""
        if isinstance(input_ids, list):
            input_ids = np.asarray(input_ids)
        if isinstance(scores, np.ndarray):
            scores = scores.astype(np.float32, copy=False)
        for batch_idx in range(input_ids.shape[0]):
            sequence = input_ids[batch_idx].tolist()
            if len(sequence) < self.ngram_size:
                continue
            search_start = max(0, len(sequence) - self.window)
            search_end = len(sequence) - self.ngram_size + 1
            if search_end <= search_start:
                continue
            if self.ngram_size > 1:
                current_prefix = tuple(sequence[-(self.ngram_size - 1):])
            else:
                current_prefix = tuple()
            banned = set()
            for idx in range(search_start, search_end):
                ngram = sequence[idx:idx + self.ngram_size]
                if self.ngram_size == 1 or tuple(ngram[:-1]) == current_prefix:
                    banned.add(ngram[-1])
            banned.difference_update(self.whitelist)
            for token_id in banned:
                scores[batch_idx, token_id] = float('-inf')
        return scores
