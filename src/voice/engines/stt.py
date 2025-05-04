"""
Speech-to-text engine implementation using Sherpa-ONNX.
"""
import os
import logging
import numpy as np
import torch
import torchaudio

from sherpa_onnx import OnlineRecognizer

logger = logging.getLogger(__name__)

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