#===----------------------------------------------------------------------===#
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#

from dataclasses import dataclass
from typing import List, Optional, Tuple

from fairseq2.data import SequenceData, StringLike
from fairseq2.data.text import TextTokenizer
from fairseq2.data import VocabularyInfo
from fairseq2.generation import (
    SequenceToTextConverter,
    StepProcessor,
)
from fairseq2.nn.padding import (
    PaddingMask,
)
from torch import Tensor

from model import (
    UnitYX2TModel,
)
from customized_fairseq2.beam_search import BeamSearchSeq2SeqGenerator


@dataclass
class SequenceGeneratorOptions:
    """Holds the options to pass to a sequence generator."""

    beam_size: int = 5
    """The beam size."""

    soft_max_seq_len: Tuple[int, int] = (1, 200)
    """The terms ``a`` and ``b`` of ``ax + b`` where ``x`` is the source
    sequence length. The generated sequences (including prefix sequence) will
    have the maximum length of ``min(hard_max_seq_len, ax + b)``. See also
    ``hard_max_seq_len``."""

    hard_max_seq_len: int = 1024
    """The hard limit on maximum length of generated sequences."""

    step_processor: Optional[StepProcessor] = None
    """The processor called at each generation step."""

    unk_penalty: float = 0.0
    """The UNK symbol penalty, where values less than 0 produce more UNKs;
    values greater than 0 produce fewer UNKs."""

    len_penalty: float = 1.0
    """The length penalty, where values less than 1.0 favor shorter
    sequences; values greater than 1.0 favor longer sequences."""


class UnitYGenerator:
    """Generates text translations from a UnitY model."""
    s2t_converter: SequenceToTextConverter

    def __init__(
        self,
        m4t_encoder_frontend,
        m4t_encoder,
        m4t_decoder_frontend,
        m4t_decoder,
        m4t_decoder_final_proj,
        text_tokenizer: TextTokenizer,
        target_vocab_info: VocabularyInfo,
        dev_id: int,
        target_lang: str,
        max_output_seq_len: int = 50,
        text_opts: Optional[SequenceGeneratorOptions] = None,
    ) -> None:
        """
        :param model:
            The UnitY model to use for generation.
        :param text_tokenizer:
            The text tokenizer to use.
        :param target_lang:
            The target language.
        :param text_generator_opts:
            The options to pass to the underlying text :class:`Seq2SeqGenerator`.
        """
        self.wav2Vec2Frontend_net = m4t_encoder_frontend
        self.unitY_encoder_adaptor_net = m4t_encoder
        self.text_decoder_frontend_net = m4t_decoder_frontend
        self.text_decoder_net = m4t_decoder
        self.final_proj_net = m4t_decoder_final_proj
        self.max_output_seq_len = max_output_seq_len

        if text_opts is None:
            text_opts = SequenceGeneratorOptions()

        self.s2t_model = UnitYX2TModel(
            encoder_frontend=self.wav2Vec2Frontend_net,
            encoder=self.unitY_encoder_adaptor_net,
            decoder_frontend=self.text_decoder_frontend_net,
            decoder=self.text_decoder_net,
            final_proj=self.final_proj_net,
            dev_id=dev_id,
            target_vocab_info=target_vocab_info,
        )

        step_processors = []
        if text_opts.step_processor is not None:
            step_processors.append(text_opts.step_processor)

        generator = BeamSearchSeq2SeqGenerator(
            self.s2t_model,
            beam_size=text_opts.beam_size,
            max_gen_len=text_opts.soft_max_seq_len,
            max_seq_len=self.max_output_seq_len, # text_opts.hard_max_seq_len,
            echo_prompt=True,
            step_processors=step_processors,
            unk_penalty=text_opts.unk_penalty,
            len_penalty=text_opts.len_penalty,
        )
        self.s2t_converter = SequenceToTextConverter(
            generator, text_tokenizer, "translation", target_lang
        )

        self.t2t_generator = None
        self.unit_generator = None
        self.unit_decoder = None

    def __call__(
        self,
        source_seqs: Tensor,
        source_padding_mask: Optional[PaddingMask],
        duration_factor: float = 1.0,
        prosody_encoder_input: Optional[SequenceData] = None,
    ) -> Tuple[List[StringLike], Optional[Tensor]]:
        """
        :param source_seqs:
            The source sequences to use for generation. *Shape:* :math:`(N,S,*)`,
            where :math:`N` is the batch size, :math:`S` is the sequence length,
            and :math:`*` is any number of sequence-specific dimensions
            including none.
        :param source_padding_mask:
            The padding mask of ``source_seqs``. *Shape:* :math:`(N,S)`, where
            :math:`N` is the batch size and :math:`S` is the sequence length.

        :returns:
            - The output of the text generator.
        """
        self.s2t_model.reset()

        texts, text_gen_output = self.s2t_converter.batch_convert(
            source_seqs, source_padding_mask
        )

        # We skip T2U when we only need to output text.
        return texts, None