from pathlib import Path
from typing import Union
import os
import numpy as np
from torch import nn
from synthesizer import Synthesizer, save_wav


class TTS(nn.Module):
    def __init__(
        self,
        model_path: str = None,
        config_path: str = None,
        progress_bar: bool = True,
        tpu_inference_config: dict = None,
    ):
        super().__init__()
        self.synthesizer = None
        self.voice_converter = None
        self.model_name = ""
        self.sample_rate = 16000

        if model_path is not None:
            assert os.path.exists(model_path)
            self.load_tts_model_by_path(
                model_path, config_path, tpu_inference_config=tpu_inference_config
            )
        else:
            raise ValueError("model_path is None!")

    def load_tts_model_by_path(
        self, model_path: str, config_path: str, tpu_inference_config: dict = None
    ):
        """Load a model from a path.

        Args:
            model_path (str): Path to the model checkpoint.
            config_path (str): Path to the model config.
            vocoder_path (str, optional): Path to the vocoder checkpoint. Defaults to None.
            vocoder_config (str, optional): Path to the vocoder config. Defaults to None.
            gpu (bool, optional): Enable/disable GPU. Some models might be too slow on CPU. Defaults to False.
        """

        self.synthesizer = Synthesizer(
            tts_checkpoint=model_path,
            tts_config_path=config_path,
            tpu_inference_config=tpu_inference_config,
        )

    def _check_arguments(
        self,
        speaker: str = None,
        language: str = None,
        speaker_wav: str = None,
        **kwargs,
    ) -> None:
        """Check if the arguments are valid for the model."""
        # check for the coqui tts models
        if speaker is None and speaker_wav is None:
            raise ValueError("Model is multi-speaker but no `speaker` is provided.")
        if language is None:
            raise ValueError("Model is multi-lingual but no `language` is provided.")

    def tts(
        self,
        text: str,
        speaker: str = None,
        language: str = None,
        speaker_wav: str = None,
        split_sentences: bool = True,
        **kwargs,
    ):
        """Convert text to speech.

        Args:
            text (str):
                Input text to synthesize.
            speaker (str, optional):
                Speaker name for multi-speaker. You can check whether loaded model is multi-speaker by
                `tts.is_multi_speaker` and list speakers by `tts.speakers`. Defaults to None.
            language (str): Language of the text. If None, the default language of the speaker is used. Language is only
                supported by `XTTS` model.
            speaker_wav (str, optional):
                Path to a reference wav file to use for voice cloning with supporting models like YourTTS.
                Defaults to None.
            split_sentences (bool, optional):
                Split text into sentences, synthesize them separately and concatenate the file audio.
                Setting it False uses more VRAM and possibly hit model specific text length or VRAM limits. Only
                applicable to the 🐸TTS models. Defaults to True.
            kwargs (dict, optional):
                Additional arguments for the model.
        """
        self._check_arguments(
            speaker=speaker, language=language, speaker_wav=speaker_wav, **kwargs
        )
        wav = self.synthesizer.tts(
            text=text,
            speaker_name=speaker,
            language_name=language,
            speaker_wav=speaker_wav,
            reference_wav=None,
            style_wav=None,
            style_text=None,
            reference_speaker_name=None,
            split_sentences=split_sentences,
            **kwargs,
        )
        return wav

    def tts_to_file(
        self,
        text: str,
        speaker: str = None,
        language: str = None,
        speaker_wav: str = None,
        file_path: str = "output.wav",
        split_sentences: bool = True,
        **kwargs,
    ):
        """Convert text to speech.

        Args:
            text (str):
                Input text to synthesize.
            speaker (str, optional):
                Speaker name for multi-speaker. You can check whether loaded model is multi-speaker by
                `tts.is_multi_speaker` and list speakers by `tts.speakers`. Defaults to None.
            language (str, optional):
                Language code for multi-lingual models. You can check whether loaded model is multi-lingual
                `tts.is_multi_lingual` and list available languages by `tts.languages`. Defaults to None.
            speaker_wav (str, optional):
                Path to a reference wav file to use for voice cloning with supporting models like YourTTS.
                Defaults to None.
            file_path (str, optional):
                Output file path. Defaults to "output.wav".
            split_sentences (bool, optional):
                Split text into sentences, synthesize them separately and concatenate the file audio.
                Setting it False uses more VRAM and possibly hit model specific text length or VRAM limits. Only
                applicable to the 🐸TTS models. Defaults to True.
            kwargs (dict, optional):
                Additional arguments for the model.
        """
        self._check_arguments(speaker=speaker, language=language, speaker_wav=speaker_wav, **kwargs)

        wav = self.tts(
            text=text,
            speaker=speaker,
            language=language,
            speaker_wav=speaker_wav,
            split_sentences=split_sentences,
            **kwargs,
        )
        self.synthesizer.save_wav(wav=wav, path=file_path)
        return file_path
