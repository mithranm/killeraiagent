"""
High-level API for voice functionality.
"""
import os
import logging
from typing import Optional, Any, Tuple

from killeraiagent.features import features
from .io import play_audio, record_audio

logger = logging.getLogger(__name__)

def initialize_voice_support() -> Tuple[Optional[Any], Optional[Any]]:
    """
    Initialize voice input and output support.
    
    Returns:
        Tuple of (voice_output, voice_input) objects or (None, None) if initialization failed
    """
    voice_output = None
    voice_input = None
    
    # Check for voice feature
    if not features.require("voice"):
        return None, None
        
    # Check for STT feature if needed
    use_stt = features.has("stt")
    
    try:
        # Import modules only after feature check
        from .engines.tts import KokoroTTS
        voice_output = KokoroTTS()
        logger.info("Voice output initialized successfully")
        
        if use_stt:
            from .engines.stt import KrokoSTT
            voice_input = KrokoSTT()
            logger.info("Voice input initialized successfully")
            
    except Exception as e:
        logger.warning(f"Failed to initialize voice support: {e}")
    
    return voice_output, voice_input

def text_to_speech(voice_output: Any, text: str) -> bool:
    """
    Convert text to speech and play it using the initialized TTS engine.
    
    Args:
        voice_output: Initialized TTS object
        text: Text to convert to speech
        
    Returns:
        True if successful, False otherwise
    """
    if voice_output is None:
        logger.debug("Voice output not available")
        return False
    
    try:
        # Generate audio from text
        waveform, sample_rate = voice_output.generate(text)
        
        # Play audio
        return play_audio(waveform, sample_rate)
    except Exception as e:
        logger.warning(f"Text-to-speech failed: {e}")
        return False

def speech_to_text(voice_input: Any) -> Optional[str]:
    """
    Convert speech to text using the initialized STT engine.
    
    Args:
        voice_input: Initialized STT object
        
    Returns:
        Transcribed text if successful, None otherwise
    """
    if voice_input is None:
        logger.debug("Voice input not available")
        return None
    
    try:
        # Record audio
        _, tmp_file = record_audio()
        if not tmp_file:
            return None
        
        try:
            # Transcribe the temporary file
            return voice_input.transcribe_file(tmp_file)
        finally:
            # Clean up temporary file
            if tmp_file and os.path.exists(tmp_file):
                try:
                    os.unlink(tmp_file)
                except Exception as e:
                    logger.error(f"Failed to delete temporary file: {e}")
    
    except Exception as e:
        logger.warning(f"Speech-to-text failed: {e}")
        return None

def speak(text: str) -> bool:
    """
    Convenience function to speak text. Initializes voice support if needed.
    
    Args:
        text: Text to speak
        
    Returns:
        True if successful, False otherwise
    """
    # Initialize voice support if not already done
    voice_output, _ = initialize_voice_support()
    if voice_output is None:
        return False
    
    # Speak the text
    return text_to_speech(voice_output, text)

def listen() -> Optional[str]:
    """
    Convenience function to listen for speech. Initializes voice support if needed.
    
    Returns:
        Transcribed text if successful, None otherwise
    """
    # Initialize voice support if not already done
    _, voice_input = initialize_voice_support()
    if voice_input is None:
        return None
    
    # Listen and return transcribed text
    return speech_to_text(voice_input)