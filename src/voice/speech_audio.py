"""
src/killeraiagent/speech_audio.py

Speech/Audio adapter for KAIA that provides:
- Kokoro-based TTS (text -> audio)
- Sherpa-ONNX-based STT (audio -> text)

Requires 'kokoro', 'sherpa-onnx', 'sounddevice', 'soundfile', etc.
"""

import os
import numpy as np
import torch
import torchaudio
import logging

from kokoro import KModel, KPipeline
from sherpa_onnx import OnlineRecognizer
logger = logging.getLogger(__name__)

class KokoroTTS:
    """
    Simple text-to-speech using Kokoro.
    Basic usage:
        tts = KokoroTTS("af_heart")  # load voice
        wav_samples, sr = tts.generate("Hello world")
        # write to .wav or play in memory
    """
    def __init__(self, voice:str = "af_heart", device:str = "cuda"):
        """
        Args:
            voice: voice name (like 'af_heart', 'am_michael', etc.)
            device: "cuda" or "cpu", if cuda is available
        """
        # We assume a single KModel is enough for all voices.
        # Alternatively you can load multiple if needed.
        self.model = KModel().to(device).eval()
        self.voice = voice
        self.device = device
        # create pipeline (lang_code='a' or 'b' based on voice name)
        # For example if voice starts with 'a' => pipeline('a') else 'b'
        # but your example code used an approach like: 'af_heart' => pipeline 'a'
        if voice.startswith("af_") or voice.startswith("am_"):
            self.lang_code = "a"
        else:
            self.lang_code = "b"
        self.pipeline = KPipeline(lang_code=self.lang_code, model=False)
        # load voice data
        self.pack = self.pipeline.load_voice(voice)

    def generate(self, text:str, speed:float =1.0) -> tuple[np.ndarray, int]:
        """
        Convert text to audio. Returns (waveform, sample_rate).
        """
        # Tokenize text via pipeline
        # We produce partial phoneme sequences in a single pass
        # Then pass to the model
        # This is a naive approach. For large text, you'd chunk it.

        phoneme_seq = []
        for _, ps, _ in self.pipeline(text, self.voice, speed):
            phoneme_seq = ps  # last chunk
        if not phoneme_seq:
            # maybe empty or error
            logger.warning("No phoneme sequence generated.")
            return (np.zeros(1, dtype=np.float32), 24000)

        ref_s = self.pack[len(phoneme_seq) - 1] if (len(phoneme_seq) > 0) else 0
        # Generate audio
        try:
            with torch.no_grad():
                if self.device == "cuda" and torch.cuda.is_available():
                    audio_t = self.model(phoneme_seq, ref_s, speed)
                else:
                    audio_t = self.model(phoneme_seq, ref_s, speed)
            # Convert to numpy
            wav = audio_t.cpu().numpy()
            return (wav, 24000)
        except Exception as e:
            logger.error(f"TTS generation error: {e}")
            return (np.zeros(1, dtype=np.float32), 24000)


class KrokoSTT:
    """
    Basic speech-to-text using Sherpa-ONNX. 
    We load multiple language models if needed, or just one (English, etc.).
    Example usage:
        stt = KrokoSTT("en")
        text = stt.transcribe_file("audio.wav")
    """

    def __init__(
        self,
        language:str = "en",
        tokens_path:str = "en_tokens.txt",
        encoder_path:str = "en_encoder.onnx",
        decoder_path:str = "en_decoder.onnx",
        joiner_path:str = "en_joiner.onnx"
    ):
        """
        Args:
            language: 'en', 'fr', 'de', 'es', ...
            tokens_path, encoder_path, decoder_path, joiner_path:
                paths to the model files for your chosen language.
        """
        self.language = language
        self.recognizer = OnlineRecognizer.from_transducer(
            tokens=tokens_path,
            encoder=encoder_path,
            decoder=decoder_path,
            joiner=joiner_path,
            num_threads=1,
            decoding_method="modified_beam_search",
            debug=False
        )
        logger.info(f"Initialized Sherpa-ONNX for language={language}")

    def transcribe_file(self, filepath:str) -> str:
        """
        Transcribes a whole audio file (blocking) and returns the text.
        """
        if not os.path.isfile(filepath):
            raise FileNotFoundError(filepath)
        waveform, sr = torchaudio.load(filepath)
        # ensure 16k
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
            waveform = resampler(waveform)
            sr = 16000

        # convert to 1D if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        s = self.recognizer.create_stream()

        # chunk approach
        chunk_size = 3200  # ~0.2 seconds at 16k
        data = waveform.squeeze(0).numpy()
        offset = 0
        length = data.shape[0]
        while offset < length:
            end = offset + chunk_size
            chunk = data[offset:end]
            s.accept_waveform(sr, chunk.tolist())
            # decode partial
            while self.recognizer.is_ready(s):
                self.recognizer.decode_streams([s])
            offset = end

        # flush
        s.accept_waveform(sr, [])
        s.input_finished()
        while self.recognizer.is_ready(s):
            self.recognizer.decode_streams([s])

        result = self.recognizer.get_result(s)
        # handle type of result
        if isinstance(result, (list, np.ndarray)):
            result = " ".join(map(str, result))
        elif isinstance(result, bytes):
            result = result.decode("utf-8", errors="ignore")

        return result
