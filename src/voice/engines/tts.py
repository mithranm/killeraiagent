"""
Text-to-speech engine implementation using Kokoro.
"""
import logging
import numpy as np
import torch

from kokoro import KModel, KPipeline

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