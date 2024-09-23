import os
import time
from typing import List

import numpy as np
import pysbd
import torch
from torch import nn
import scipy

from xtts_pipeline import Xtts, load_config


def save_wav(*, wav: np.ndarray, path: str, sample_rate: int = None, **kwargs) -> None:
    """Save float waveform to a file using Scipy.

    Args:
        wav (np.ndarray): Waveform with float values in range [-1, 1] to save.
        path (str): Path to a output file.
        sr (int, optional): Sampling rate used for saving to the file. Defaults to None.
    """
    wav_norm = wav * (32767 / max(0.01, np.max(np.abs(wav))))
    wav_norm = wav_norm.astype(np.int16)
    scipy.io.wavfile.write(path, sample_rate, wav_norm)

class Synthesizer(nn.Module):
    def __init__(
        self,
        tts_checkpoint: str = "",
        tts_config_path: str = "",
        tpu_inference_config: dict = None,
    ) -> None:
        """General 🐸 TTS interface for inference. It takes a tts and a vocoder
        model and synthesize speech from the provided text.

        The text is divided into a list of sentences using `pysbd` and synthesize
        speech on each sentence separately.

        If you have certain special characters in your text, you need to handle
        them before providing the text to Synthesizer.

        TODO: set the segmenter based on the source language

        Args:
            tts_checkpoint (str, optional): path to the tts model file.
            tts_config_path (str, optional): path to the tts config file.
        """
        super().__init__()
        self.tts_checkpoint = tts_checkpoint
        self.tts_config_path = tts_config_path

        self.tts_model = None
        self.seg = self._get_segmenter("en")

        if tts_checkpoint:
            self._load_tts(tts_checkpoint, tts_config_path, tpu_inference_config)
            self.output_sample_rate = self.tts_config.audio["sample_rate"]

    @staticmethod
    def _get_segmenter(lang: str):
        """get the sentence segmenter for the given language.

        Args:
            lang (str): target language code.

        Returns:
            [type]: [description]
        """
        return pysbd.Segmenter(language=lang, clean=True)

    def _load_tts(self, tts_checkpoint: str, tts_config_path: str, tpu_inference_config: dict=None) -> None:
        """Load the TTS model.

        1. Load the model config.
        2. Init the model from the config.
        3. Load the model weights.
        4. Move the model to the GPU if CUDA is enabled.
        5. Init the speaker manager in the model.

        Args:
            tts_checkpoint (str): path to the model checkpoint.
            tts_config_path (str): path to the model config file.
            use_cuda (bool): enable/disable CUDA use.
        """
        # pylint: disable=global-statement
        self.tts_config = load_config(tts_config_path)
        if self.tts_config["use_phonemes"] and self.tts_config["phonemizer"] is None:
            raise ValueError("Phonemizer is not defined in the TTS config.")
        self.tts_model = Xtts(config=self.tts_config, tpu_inference_config=tpu_inference_config)

        self.tts_model.load_checkpoint(self.tts_config, tts_checkpoint, tpu_inference_config=tpu_inference_config)

    def split_into_sentences(self, text) -> List[str]:
        """Split give text into sentences.

        Args:
            text (str): input text in string format.

        Returns:
            List[str]: list of sentences.
        """
        return self.seg.segment(text)

    def save_wav(self, wav: List[int], path: str) -> None:
        """Save the waveform as a file.

        Args:
            wav (List[int]): waveform as a list of values.
            path (str): output path to save the waveform.
        """
        # if tensor convert to numpy
        if torch.is_tensor(wav):
            wav = wav.cpu().numpy()
        if isinstance(wav, list):
            wav = np.array(wav)
        save_wav(wav=wav, path=path, sample_rate=self.output_sample_rate)

    def tts(
        self,
        text: str = "",
        speaker_name: str = "",
        language_name: str = "",
        speaker_wav=None,
        style_wav=None,
        style_text=None,
        reference_wav=None,
        reference_speaker_name=None,
        split_sentences: bool = True,
        **kwargs,
    ) -> List[int]:
        """🐸 TTS magic. Run all the models and generate speech.

        Args:
            text (str): input text.
            speaker_name (str, optional): speaker id for multi-speaker models. Defaults to "".
            language_name (str, optional): language id for multi-language models. Defaults to "".
            speaker_wav (Union[str, List[str]], optional): path to the speaker wav for voice cloning. Defaults to None.
            style_wav ([type], optional): style waveform for GST. Defaults to None.
            style_text ([type], optional): transcription of style_wav for Capacitron. Defaults to None.
            reference_wav ([type], optional): reference waveform for voice conversion. Defaults to None.
            reference_speaker_name ([type], optional): speaker id of reference waveform. Defaults to None.
            split_sentences (bool, optional): split the input text into sentences. Defaults to True.
            **kwargs: additional arguments to pass to the TTS model.
        Returns:
            List[int]: [description]
        """
        start_time = time.time()
        wavs = []

        if not text and not reference_wav:
            raise ValueError(
                "You need to define either `text` (for sythesis) or a `reference_wav` (for voice conversion) to use the Coqui TTS API."
            )

        if text:
            sens = [text]
            if split_sentences:
                print(" > Text splitted to sentences.")
                sens = self.split_into_sentences(text)
            print(sens)

        speaker_embedding = None

        preprocess_profile = time.time() - start_time

        # wav generate
        tt = time.time()
        # import pdb; pdb.set_trace()
        for sen in sens:
            # TODO fix length
            outputs = self.tts_model.synthesize(
                text=sen,
                config=self.tts_config,
                speaker_id=speaker_name,
                d_vector=speaker_embedding,
                speaker_wav=speaker_wav,
                language=language_name,
                **kwargs,
            )
            waveform = outputs["wav"]
            waveform = waveform.squeeze()

            wavs += list(waveform)
            wavs += [0] * 10000
        wav_gen_profile = time.time() - tt

        # compute stats
        process_time = time.time() - start_time
        audio_time = len(wavs) / self.tts_config.audio["sample_rate"]
        tpu_time = self.tts_model.gpt.gpt_inference.gpt_inference_first_profile \
                    + self.tts_model.gpt.gpt_inference.gpt_inference_loop_profile
        print("{: <30}{: <10}".format(f" Real-time factor:", f"{process_time / audio_time:.2f}"))
        print("{: <30}{: <10}".format(f" process_time:", f"{process_time:.2f}s"))
        print("{: <40}{: >10}{: >20}{}".format(f" time tree", f"time(s)", f"percentage(%)", "=time/process_time"))
        print("{: <40}{: >10}{: >20}".format(f" process_time:", f"{process_time:.2f}", f"{process_time / process_time * 100 :.2f}"))
        print("{: <40}{: >10}{: >20}".format(f" ├── preprocess time:", f"{preprocess_profile:.2f}", f"{preprocess_profile / process_time * 100 :.2f}"))
        print("{: <40}{: >10}{: >20}".format(f" ├── wav generate time:", f"{wav_gen_profile:.2f}", f"{wav_gen_profile / process_time * 100 :.2f}"))
        print("{: <40}{: >10}{: >20}".format(f" │    ├── get_conditioning_latents time:", f"{self.tts_model.get_conditioning_latents_profile:.2f}", f"{self.tts_model.get_conditioning_latents_profile / process_time * 100 :.2f}"))
        print("{: <40}{: >10}{: >20}".format(f" │    └── gpt.generate time:", f"{self.tts_model.gpt_generate_profile:.2f}", f"{self.tts_model.gpt_generate_profile / process_time * 100 :.2f}"))
        print("{: <40}{: >10}{: >20}".format(f" │         ├── tpu inference time:", f"{self.tts_model.gpt.gpt_inference.gpt_inference_first_profile + self.tts_model.gpt.gpt_inference.gpt_inference_loop_profile:.2f}", f"{self.tts_model.gpt.gpt_inference.gpt_inference_first_profile + self.tts_model.gpt.gpt_inference.gpt_inference_loop_profile / process_time * 100 :.2f}"))
        print("{: <40}{: >10}{: >20}".format(f" │         └── speaker_encoder time:", f"{self.tts_model.speaker_encoder_profile:.2f}", f"{self.tts_model.speaker_encoder_profile / process_time * 100 :.2f}"))
        print("{: <40}{: >10}{: >20}".format(f" ├── gpt time:", f"{self.tts_model.gpt_profile:.2f}", f"{self.tts_model.gpt_profile / process_time * 100 :.2f}"))
        print("{: <40}{: >10}{: >20}".format(f" └── hifigan_decoder time:", f"{self.tts_model.hifigan_decoder_profile:.2f}", f"{self.tts_model.hifigan_decoder_profile / process_time * 100 :.2f}"))
        return wavs
