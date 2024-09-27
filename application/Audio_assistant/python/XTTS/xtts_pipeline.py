import os
from dataclasses import dataclass, dataclass, field
import time
import sys
from typing import List

import librosa
import torch
import torch.nn.functional as F
import torchaudio
from coqpit import Coqpit
import yaml
import json
from trainer import TrainerConfig
import fsspec

from xtts.gpt import GPT
from xtts.hifigan_decoder import HifiDecoder
from xtts.stream_generator import init_stream_support
from xtts.tokenizer import VoiceBpeTokenizer, split_sentence
from xtts.xtts_manager import SpeakerManager, LanguageManager

init_stream_support()


def wav_to_mel_cloning(
    wav,
    mel_norms_file="../experiments/clips_mel_norms.pth",
    mel_norms=None,
    device=torch.device("cpu"),
    n_fft=4096,
    hop_length=1024,
    win_length=4096,
    power=2,
    normalized=False,
    sample_rate=22050,
    f_min=0,
    f_max=8000,
    n_mels=80,
):
    """
    Convert waveform to mel-spectrogram with hard-coded parameters for cloning.

    Args:
        wav (torch.Tensor): Input waveform tensor.
        mel_norms_file (str): Path to mel-spectrogram normalization file.
        mel_norms (torch.Tensor): Mel-spectrogram normalization tensor.
        device (torch.device): Device to use for computation.

    Returns:
        torch.Tensor: Mel-spectrogram tensor.
    """
    mel_stft = torchaudio.transforms.MelSpectrogram(
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        power=power,
        normalized=normalized,
        sample_rate=sample_rate,
        f_min=f_min,
        f_max=f_max,
        n_mels=n_mels,
        norm="slaney",
    ).to(device)
    wav = wav.to(device)
    mel = mel_stft(wav)
    mel = torch.log(torch.clamp(mel, min=1e-5))
    if mel_norms is None:
        mel_norms = torch.load(mel_norms_file, map_location=device)
    mel = mel / mel_norms.unsqueeze(0).unsqueeze(-1)
    return mel


def load_audio(audiopath, sampling_rate):
    # better load setting following: https://github.com/faroit/python_audio_loading_benchmark

    # torchaudio should chose proper backend to load audio depending on platform
    audio, lsr = torchaudio.load(audiopath)

    # stereo to mono if needed
    if audio.size(0) != 1:
        audio = torch.mean(audio, dim=0, keepdim=True)

    if lsr != sampling_rate:
        audio = torchaudio.functional.resample(audio, lsr, sampling_rate)

    # Check some assumptions about audio range. This should be automatically fixed in load_wav_to_torch, but might not be in some edge cases, where we should squawk.
    # '10' is arbitrarily chosen since it seems like audio will often "overdrive" the [-1,1] bounds.
    if torch.any(audio > 10) or not torch.any(audio < 0):
        print(f"Error with {audiopath}. Max={audio.max()} min={audio.min()}")
    # clip audio invalid values
    audio.clip_(-1, 1)
    return audio


def pad_or_truncate(t, length):
    """
    Ensure a given tensor t has a specified sequence length by either padding it with zeros or clipping it.

    Args:
        t (torch.Tensor): The input tensor to be padded or truncated.
        length (int): The desired length of the tensor.

    Returns:
        torch.Tensor: The padded or truncated tensor.
    """
    tp = t[..., :length]
    if t.shape[-1] == length:
        tp = t
    elif t.shape[-1] < length:
        tp = F.pad(t, (0, length - t.shape[-1]))
    return tp


@dataclass
class XttsAudioConfig(Coqpit):
    """
    Configuration class for audio-related parameters in the XTTS model.

    Args:
        sample_rate (int): The sample rate in which the GPT operates.
        output_sample_rate (int): The sample rate of the output audio waveform.
    """

    sample_rate: int = 22050
    output_sample_rate: int = 24000


@dataclass
class XttsArgs(Coqpit):
    """A dataclass to represent XTTS model arguments that define the model structure.

    Args:
        gpt_batch_size (int): The size of the auto-regressive batch.
        enable_redaction (bool, optional): Whether to enable redaction. Defaults to True.
        kv_cache (bool, optional): Whether to use the kv_cache. Defaults to True.
        gpt_checkpoint (str, optional): The checkpoint for the autoregressive model. Defaults to None.
        clvp_checkpoint (str, optional): The checkpoint for the ConditionalLatentVariablePerseq model. Defaults to None.
        decoder_checkpoint (str, optional): The checkpoint for the DiffTTS model. Defaults to None.
        num_chars (int, optional): The maximum number of characters to generate. Defaults to 255.

        For GPT model:
        gpt_max_audio_tokens (int, optional): The maximum mel tokens for the autoregressive model. Defaults to 604.
        gpt_max_text_tokens (int, optional): The maximum text tokens for the autoregressive model. Defaults to 402.
        gpt_max_prompt_tokens (int, optional): The maximum prompt tokens or the autoregressive model. Defaults to 70.
        gpt_layers (int, optional): The number of layers for the autoregressive model. Defaults to 30.
        gpt_n_model_channels (int, optional): The model dimension for the autoregressive model. Defaults to 1024.
        gpt_n_heads (int, optional): The number of heads for the autoregressive model. Defaults to 16.
        gpt_number_text_tokens (int, optional): The number of text tokens for the autoregressive model. Defaults to 255.
        gpt_start_text_token (int, optional): The start text token for the autoregressive model. Defaults to 255.
        gpt_checkpointing (bool, optional): Whether to use checkpointing for the autoregressive model. Defaults to False.
        gpt_train_solo_embeddings (bool, optional): Whether to train embeddings for the autoregressive model. Defaults to False.
        gpt_code_stride_len (int, optional): The hop_size of dvae and consequently of the gpt output. Defaults to 1024.
        gpt_use_masking_gt_prompt_approach (bool, optional):  If True, it will use ground truth as prompt and it will mask the loss to avoid repetition. Defaults to True.
        gpt_use_perceiver_resampler (bool, optional):  If True, it will use perceiver resampler from flamingo paper - https://arxiv.org/abs/2204.14198. Defaults to False.
    """

    gpt_batch_size: int = 1
    enable_redaction: bool = False
    kv_cache: bool = True
    gpt_checkpoint: str = None
    clvp_checkpoint: str = None
    decoder_checkpoint: str = None
    num_chars: int = 255

    # XTTS GPT Encoder params
    tokenizer_file: str = ""
    gpt_max_audio_tokens: int = 605
    gpt_max_text_tokens: int = 402
    gpt_max_prompt_tokens: int = 70
    gpt_layers: int = 30
    gpt_n_model_channels: int = 1024
    gpt_n_heads: int = 16
    gpt_number_text_tokens: int = None
    gpt_start_text_token: int = None
    gpt_stop_text_token: int = None
    gpt_num_audio_tokens: int = 8194
    gpt_start_audio_token: int = 8192
    gpt_stop_audio_token: int = 8193
    gpt_code_stride_len: int = 1024
    gpt_use_masking_gt_prompt_approach: bool = True
    gpt_use_perceiver_resampler: bool = False

    # HifiGAN Decoder params
    input_sample_rate: int = 22050
    output_sample_rate: int = 24000
    output_hop_length: int = 256
    decoder_input_dim: int = 1024
    d_vector_dim: int = 512
    cond_d_vector_in_each_upsampling_layer: bool = True

    # constants
    duration_const: int = 102400

@dataclass
class XttsConfig(TrainerConfig):
    model: str = "xtts"
    # model specific params
    model_args: XttsArgs = field(default_factory=XttsArgs)
    audio: XttsAudioConfig = field(default_factory=XttsAudioConfig)
    # phoneme settings
    use_phonemes: bool = False
    phonemizer: str = None
    model_dir: str = None
    languages: List[str] = field(
        default_factory=lambda: [
            "en",
            "es",
            "fr",
            "de",
            "it",
            "pt",
            "pl",
            "tr",
            "ru",
            "nl",
            "cs",
            "ar",
            "zh-cn",
            "hu",
            "ko",
            "ja",
            "hi",
        ]
    )

    # inference params
    temperature: float = 0.85
    length_penalty: float = 1.0
    repetition_penalty: float = 2.0
    top_k: int = 50
    top_p: float = 0.85
    num_gpt_outputs: int = 1

    # cloning
    gpt_cond_len: int = 12
    gpt_cond_chunk_len: int = 4
    max_ref_len: int = 10
    sound_norm_refs: bool = False

def read_json_with_comments(json_path):
    """for backward compat."""
    # fallback to json
    with fsspec.open(json_path, "r", encoding="utf-8") as f:
        input_str = f.read()
    # handle comments but not urls with //
    input_str = re.sub(r"(\"(?:[^\"\\]|\\.)*\")|(/\*(?:.|[\\n\\r])*?\*/)|(//.*)", lambda m: m.group(1) or m.group(2) or "", input_str)
    return json.loads(input_str)

def load_config(config_path: str) -> Coqpit:
    config_dict = {}
    ext = os.path.splitext(config_path)[1]
    if ext in (".yml", ".yaml"):
        with fsspec.open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    elif ext == ".json":
        try:
            with fsspec.open(config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.decoder.JSONDecodeError:
            # backwards compat.
            data = read_json_with_comments(config_path)
    else:
        raise TypeError(f" [!] Unknown config file type {ext}")
    config_dict.update(data)
    config = XttsConfig()
    config.from_dict(config_dict)
    return config

class Xtts(torch.nn.Module):
    """ⓍTTS model implementation.
    """

    def __init__(self, config: Coqpit, tpu_inference_config: dict = None):
        super().__init__()
        self.mel_stats_path = None
        self.config = config
        self.tokenizer = None
        self._set_model_args(config)
        self.models_dir = config.model_dir
        self.gpt_batch_size = self.args.gpt_batch_size

        self.tokenizer = VoiceBpeTokenizer()
        self.gpt = None
        self.init_models(tpu_inference_config)
        self.register_buffer("mel_stats", torch.ones(80))

        self.hifigan_decoder_profile = 0
        self.speaker_encoder_profile = 0
        self.gpt_profile = 0
        self.get_conditioning_latents_profile = 0
        self.gpt_generate_profile = 0
        self.hifigan_decoder = HifiDecoder(
            input_sample_rate=self.args.input_sample_rate,
            output_sample_rate=self.args.output_sample_rate,
            output_hop_length=self.args.output_hop_length,
            ar_mel_length_compression=self.args.gpt_code_stride_len,
            tpu_inference_config=tpu_inference_config,
        )

    def _set_model_args(self, config: Coqpit):
        # don't use isintance not to import recursively
        if "Config" in config.__class__.__name__:
            config_num_chars = (
                self.config.model_args.num_chars if hasattr(self.config, "model_args") else self.config.num_chars
            )
            num_chars = config_num_chars if self.tokenizer is None else self.tokenizer.characters.num_chars
            if "characters" in config:
                self.config.num_chars = num_chars
                if hasattr(self.config, "model_args"):
                    config.model_args.num_chars = num_chars
                    self.args = self.config.model_args
            else:
                self.config = config
                self.args = config.model_args
        elif "Args" in config.__class__.__name__:
            self.args = config
        else:
            raise ValueError("config must be either a *Config or *Args")

    def init_models(self, tpu_inference_config=None):
        """Initialize the models. We do it here since we need to load the tokenizer first."""
        if self.tokenizer.tokenizer is not None:
            self.args.gpt_number_text_tokens = self.tokenizer.get_number_tokens()
            self.args.gpt_start_text_token = self.tokenizer.tokenizer.token_to_id("[START]")
            self.args.gpt_stop_text_token = self.tokenizer.tokenizer.token_to_id("[STOP]")

        if self.args.gpt_number_text_tokens:
            self.gpt = GPT(
                model_dim=self.args.gpt_n_model_channels,
                start_text_token=self.args.gpt_start_text_token,
                stop_text_token=self.args.gpt_stop_text_token,
                heads=self.args.gpt_n_heads,
                max_text_tokens=self.args.gpt_max_text_tokens,
                max_mel_tokens=self.args.gpt_max_audio_tokens,
                number_text_tokens=self.args.gpt_number_text_tokens,
                num_audio_tokens=self.args.gpt_num_audio_tokens,
                start_audio_token=self.args.gpt_start_audio_token,
                stop_audio_token=self.args.gpt_stop_audio_token,
                code_stride_len=self.args.gpt_code_stride_len,
                tpu_inference_config=tpu_inference_config
            )

    @torch.inference_mode()
    def get_gpt_cond_latents(self, audio, sr, length: int = 30, chunk_length: int = 6):
        """Compute the conditioning latents for the GPT model from the given audio.

        Args:
            audio (tensor): audio tensor.
            sr (int): Sample rate of the audio.
            length (int): Length of the audio in seconds. If < 0, use the whole audio. Defaults to 30.
            chunk_length (int): Length of the audio chunks in seconds. When `length == chunk_length`, the whole audio
                is being used without chunking. It must be < `length`. Defaults to 6.
        """
        if sr != 22050:
            audio = torchaudio.functional.resample(audio, sr, 22050)
        if length > 0:
            audio = audio[:, : 22050 * length]
        style_embs = []
        for i in range(0, audio.shape[1], 22050 * chunk_length):
            audio_chunk = audio[:, i : i + 22050 * chunk_length]

            # if the chunk is too short ignore it 
            if audio_chunk.size(-1) < 22050 * 0.33:
                continue

            mel_chunk = wav_to_mel_cloning(
                audio_chunk,
                mel_norms=self.mel_stats.cpu(),
                n_fft=2048,
                hop_length=256,
                win_length=1024,
                power=2,
                normalized=False,
                sample_rate=22050,
                f_min=0,
                f_max=8000,
                n_mels=80,
            )
            style_emb = self.gpt.get_style_emb(mel_chunk.to("cpu"), None)
            style_embs.append(style_emb)

        # mean style embedding
        cond_latent = torch.stack(style_embs).mean(dim=0)
        return cond_latent.transpose(1, 2)

    @torch.inference_mode()
    def get_speaker_embedding(self, audio, sr):
        audio_16k = torchaudio.functional.resample(audio, sr, 16000)
        st = time.time()
        res = (
            self.hifigan_decoder.speaker_encoder.forward(audio_16k.to("cpu"), l2_norm=True)
            .unsqueeze(-1)
            .to("cpu")
        )
        self.speaker_encoder_profile += time.time() - st
        return res

    @torch.inference_mode()
    def get_conditioning_latents(
        self,
        audio_path,
        max_ref_length=30,
        gpt_cond_len=6,
        gpt_cond_chunk_len=6,
        librosa_trim_db=None,
        sound_norm_refs=False,
        load_sr=22050,
    ):
        """Get the conditioning latents for the GPT model from the given audio.

        Args:
            audio_path (str or List[str]): Path to reference audio file(s).
            max_ref_length (int): Maximum length of each reference audio in seconds. Defaults to 30.
            gpt_cond_len (int): Length of the audio used for gpt latents. Defaults to 6.
            gpt_cond_chunk_len (int): Chunk length used for gpt latents. It must be <= gpt_conf_len. Defaults to 6.
            librosa_trim_db (int, optional): Trim the audio using this value. If None, not trimming. Defaults to None.
            sound_norm_refs (bool, optional): Whether to normalize the audio. Defaults to False.
            load_sr (int, optional): Sample rate to load the audio. Defaults to 24000.
        """
        # deal with multiples references
        if not isinstance(audio_path, list):
            audio_paths = [audio_path]
        else:
            audio_paths = audio_path

        speaker_embeddings = []
        audios = []
        speaker_embedding = None
        for file_path in audio_paths:
            audio = load_audio(file_path, load_sr)
            audio = audio[:, : load_sr * max_ref_length].to("cpu")
            if sound_norm_refs:
                audio = (audio / torch.abs(audio).max()) * 0.75
            if librosa_trim_db is not None:
                audio = librosa.effects.trim(audio, top_db=librosa_trim_db)[0]

            # compute latents for the decoder
            speaker_embedding = self.get_speaker_embedding(audio, load_sr)
            speaker_embeddings.append(speaker_embedding)

            audios.append(audio)

        # merge all the audios and compute the latents for the gpt
        full_audio = torch.cat(audios, dim=-1)
        gpt_cond_latents = self.get_gpt_cond_latents(
            full_audio, load_sr, length=gpt_cond_len, chunk_length=gpt_cond_chunk_len
        )  # [1, 1024, T]

        if speaker_embeddings:
            speaker_embedding = torch.stack(speaker_embeddings)
            speaker_embedding = speaker_embedding.mean(dim=0)

        return gpt_cond_latents, speaker_embedding

    def synthesize(self, text, config, speaker_wav, language, speaker_id=None, **kwargs):
        """Synthesize speech with the given input text.
        """
        assert (
            "zh-cn" if language == "zh" else language in self.config.languages
        ), f" ❗ Language {language} is not supported. Supported languages are {self.config.languages}"
        # Use generally found best tuning knobs for generation.
        settings = {
            "temperature": config.temperature,
            "length_penalty": config.length_penalty,
            "repetition_penalty": config.repetition_penalty,
            "top_k": config.top_k,
            "top_p": config.top_p,
        }
        settings.update(kwargs)  # allow overriding of preset settings with kwargs
        if speaker_id is not None:
            gpt_cond_latent, speaker_embedding = self.speaker_manager.speakers[speaker_id].values()
            return self.inference(text, language, gpt_cond_latent, speaker_embedding, **settings)
        settings.update({
            "gpt_cond_len": config.gpt_cond_len,
            "gpt_cond_chunk_len": config.gpt_cond_chunk_len,
            "max_ref_len": config.max_ref_len,
            "sound_norm_refs": config.sound_norm_refs,
        })
        return self.full_inference(text, speaker_wav, language, **settings)

    @torch.inference_mode()
    def full_inference(
        self,
        text,
        ref_audio_path,
        language,
        # GPT inference
        temperature=0.75,
        length_penalty=1.0,
        repetition_penalty=10.0,
        top_k=50,
        top_p=0.85,
        do_sample=True,
        # Cloning
        gpt_cond_len=30,
        gpt_cond_chunk_len=6,
        max_ref_len=10,
        sound_norm_refs=False,
        **hf_generate_kwargs,
    ):
        """
        This function produces an audio clip of the given text being spoken with the given reference voice.
        """
        start_time = time.time()
        (gpt_cond_latent, speaker_embedding) = self.get_conditioning_latents(
            audio_path=ref_audio_path,
            gpt_cond_len=gpt_cond_len,
            gpt_cond_chunk_len=gpt_cond_chunk_len,
            max_ref_length=max_ref_len,
            sound_norm_refs=sound_norm_refs,
        )
        self.get_conditioning_latents_profile += time.time() - start_time

        return self.inference(
            text,
            language,
            gpt_cond_latent,
            speaker_embedding,
            temperature=temperature,
            length_penalty=length_penalty,
            repetition_penalty=repetition_penalty,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
            **hf_generate_kwargs,
        )

    @torch.inference_mode()
    def inference(
        self,
        text,
        language,
        gpt_cond_latent,
        speaker_embedding,
        # GPT inference
        temperature=0.75,
        length_penalty=1.0,
        repetition_penalty=10.0,
        top_k=50,
        top_p=0.85,
        do_sample=True,
        num_beams=1,
        speed=1.0,
        enable_text_splitting=False,
        **hf_generate_kwargs,
    ):
        language = language.split("-")[0]  # remove the country code
        length_scale = 1.0 / max(speed, 0.05)
        gpt_cond_latent = gpt_cond_latent.to("cpu")
        speaker_embedding = speaker_embedding.to("cpu")
        if enable_text_splitting:
            text = split_sentence(text, language, self.tokenizer.char_limits[language])
        else:
            text = [text]

        wavs = []
        gpt_latents_list = []
        for sent in text:
            sent = sent.strip().lower()
            text_tokens = torch.IntTensor(self.tokenizer.encode(sent, lang=language)).unsqueeze(0).to("cpu")

            assert (
                text_tokens.shape[-1] < self.args.gpt_max_text_tokens
            ), " ❗ XTTS can only generate text with a maximum of 400 tokens."

            with torch.no_grad():
                start_time = time.time()
                # import pdb
                # pdb.set_trace()
                gpt_codes = self.gpt.generate(
                    cond_latents=gpt_cond_latent,
                    text_inputs=text_tokens,
                    input_tokens=None,
                    do_sample=do_sample,
                    top_p=top_p,
                    top_k=top_k,
                    temperature=temperature,
                    num_return_sequences=self.gpt_batch_size,
                    num_beams=num_beams,
                    length_penalty=length_penalty,
                    repetition_penalty=repetition_penalty,
                    output_attentions=False,
                    **hf_generate_kwargs,
                )
                self.gpt_generate_profile += time.time() - start_time
                expected_output_len = torch.tensor(
                    [gpt_codes.shape[-1] * self.gpt.code_stride_len], device=text_tokens.device
                )

                text_len = torch.tensor([text_tokens.shape[-1]], device="cpu")
                start_time = time.time()
                gpt_latents = self.gpt(
                    text_tokens,
                    text_len,
                    gpt_codes,
                    expected_output_len,
                    cond_latents=gpt_cond_latent
                )
                self.gpt_profile += time.time() - start_time

                if length_scale != 1.0:
                    gpt_latents = F.interpolate(
                        gpt_latents.transpose(1, 2), scale_factor=length_scale, mode="linear"
                    ).transpose(1, 2)

                gpt_latents_list.append(gpt_latents.cpu())
                start_time = time.time()
                wavs.append(self.hifigan_decoder(gpt_latents, g=speaker_embedding).cpu().squeeze())
                self.hifigan_decoder_profile += time.time() - start_time

        return {
            "wav": torch.cat(wavs, dim=0).numpy(),
            "gpt_latents": torch.cat(gpt_latents_list, dim=1).numpy(),
            "speaker_embedding": speaker_embedding,
        }

    def get_compatible_checkpoint_state_dict(self, model_path):
        with fsspec.open(model_path, "rb") as f:
            checkpoint = torch.load(f, map_location=torch.device("cpu"))
        return checkpoint

    def load_checkpoint(
        self,
        config,
        checkpoint_dir=None,
        checkpoint_path=None,
        vocab_path=None,
        strict=True,
        speaker_file_path=None,
        tpu_inference_config=None,
    ):
        """
        Loads a checkpoint from disk and initializes the model's state and tokenizer.
        """

        model_path = checkpoint_path or os.path.join(checkpoint_dir, "mel_stats_gpt_part.pth")
        vocab_path = vocab_path or os.path.join(checkpoint_dir, "vocab.json")

        assert speaker_file_path is None and checkpoint_dir is not None
        speaker_file_path = os.path.join(checkpoint_dir, "speakers_xtts.pth")

        self.language_manager = LanguageManager(config)
        self.speaker_manager = None
        assert speaker_file_path is not None and os.path.exists(speaker_file_path)
        self.speaker_manager = SpeakerManager(speaker_file_path)

        assert vocab_path is not None and os.path.exists(vocab_path)
        self.tokenizer = VoiceBpeTokenizer(vocab_file=vocab_path)

        self.init_models()

        assert model_path is not None and os.path.exists(model_path)
        checkpoint = self.get_compatible_checkpoint_state_dict(model_path)

        # deal with v1 and v1.1. V1 has the init_gpt_for_inference keys, v1.1 do not
        try:
            self.load_state_dict(checkpoint, strict=strict)
        except Exception as e:
            print(e)
            sys.exit(-1)
            
        self.hifigan_decoder.eval()
        self.gpt.init_gpt_for_inference(kv_cache=self.args.kv_cache, tpu_inference_config=tpu_inference_config)
        self.gpt.eval()

