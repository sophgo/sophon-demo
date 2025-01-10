import torch
import torch.nn.functional as F
from transformers.generation import TopKLogitsWarper, TopPLogitsWarper

class CustomRepetitionPenaltyLogitsProcessorRepeat:

    def __init__(self, penalty: float, max_input_ids: int, past_window: int):
        if not isinstance(penalty, float) or not (penalty > 0):
            raise ValueError(
                f"`penalty` has to be a strictly positive float, but is {penalty}"
            )

        self.penalty = penalty
        self.max_input_ids = max_input_ids
        self.past_window = past_window

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        # Step 1: Restrict input_ids to the past_window size
        if input_ids.size(1) > self.past_window:
            input_ids = input_ids[:, -self.past_window:]  # Take the last `past_window` tokens

        # Step 2: Compute token frequencies
        # One-hot encode input_ids and sum along the sequence dimension
        freq = F.one_hot(input_ids, num_classes=scores.size(1)).sum(dim=1)

        # Step 3: Zero out frequencies beyond max_input_ids
        if freq.size(0) > self.max_input_ids:
            freq[self.max_input_ids:] = 0  # Ensure slicing is valid

        # Step 4: Compute alpha (penalty factor)
        alpha = torch.pow(self.penalty, freq)

        # Step 5: Apply penalty to scores
        # Use torch.where to handle positive and negative scores
        scores = torch.where(scores < 0, scores * alpha, scores / alpha)

        return scores
def gen_logits(
    num_code: int,
    top_P=0.7,
    top_K=20,
    repetition_penalty=1.0,
):
    logits_warpers = []
    if top_P is not None:
        logits_warpers.append(TopPLogitsWarper(top_P, min_tokens_to_keep=3))
    if top_K is not None:
        logits_warpers.append(TopKLogitsWarper(top_K, min_tokens_to_keep=3))

    logits_processors = []
    if repetition_penalty is not None and repetition_penalty != 1:
        logits_processors.append(
            CustomRepetitionPenaltyLogitsProcessorRepeat(
                repetition_penalty, num_code, 16
            )
        )

    return logits_warpers, logits_processors