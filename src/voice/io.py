"""
Audio input/output utilities for voice functionality.
"""
import os
import logging
import tempfile
import numpy as np
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

def play_audio(waveform: np.ndarray, sample_rate: int) -> bool:
    """
    Play audio using available libraries.
    
    Args:
        waveform: Audio waveform as numpy array
        sample_rate: Sample rate of the audio
        
    Returns:
        True if audio was played, False otherwise
    """
    try:
        import sounddevice as sd
        
        # Play the audio
        sd.play(waveform, sample_rate)
        sd.wait()  # Wait until audio is finished playing
        return True
    except ImportError:
        # Try alternative playback method
        return save_audio_temp(waveform, sample_rate)

def save_audio_temp(waveform: np.ndarray, sample_rate: int) -> bool:
    """
    Save audio to a temporary file when direct playback isn't available.
    
    Args:
        waveform: Audio waveform as numpy array
        sample_rate: Sample rate of the audio
        
    Returns:
        True if audio was saved, False otherwise
    """
    try:
        import soundfile as sf
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            sf.write(tmp.name, waveform, sample_rate)
            logger.info(f"Audio saved to temporary file: {tmp.name}")
        return True
    except ImportError:
        logger.warning("Neither sounddevice nor soundfile available for audio playback")
        return False

def record_audio(duration: float = 5.0, sample_rate: int = 16000) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """
    Record audio to a temporary file.
    
    Args:
        duration: Recording duration in seconds
        sample_rate: Sample rate for recording
        
    Returns:
        Tuple of (recorded_audio, temp_file_path) or (None, None) on failure
    """
    try:
        import sounddevice as sd
        import soundfile as sf
        
        # Create a temporary file for the recording
        tmp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False).name
        
        # Inform user
        logger.info(f"Recording for {duration} seconds...")
        
        # Record audio
        recording = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype='float32'
        )
        sd.wait()  # Wait for recording to complete
        
        # Save to temporary file
        sf.write(tmp_file, recording, sample_rate)
        
        return recording, tmp_file
    except Exception as e:
        logger.warning(f"Error during recording: {e}")
        return None, None