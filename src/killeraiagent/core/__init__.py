"""
Core functionality shared between CLI and GUI interfaces for KAIA.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple, Callable

from killeraiagent.models.base import LLM

from .model_management import load_model

__all__ = [
    'load_model'
]

logger = logging.getLogger(__name__)

def initialize_voice_support() -> Tuple[Optional[Any], Optional[Any]]:
    """
    Initialize voice input and output support.
    
    Returns:
        Tuple of (voice_output, voice_input) objects or (None, None) if initialization failed
    """
    voice_output = None
    voice_input = None
    
    try:
        from killeraiagent.speech_audio import KokoroTTS, KrokoSTT
        voice_output = KokoroTTS()
        voice_input = KrokoSTT()
        logger.info("Voice support initialized")
    except ImportError as e:
        logger.error(f"Failed to initialize voice support: {e}")
    
    return voice_output, voice_input
 
def format_chat_prompt(conversation: List[Dict[str, str]], system_prompt: Optional[str] = None) -> str:
    """
    Format a conversation history into a prompt suitable for the model.
    
    Args:
        conversation: List of message dictionaries with 'role' and 'content' keys
        system_prompt: Optional system prompt to include
        
    Returns:
        Formatted prompt string
    """
    if system_prompt:
        system = f"<s>[SYSTEM] {system_prompt} [/SYSTEM]\n\n"
    else:
        system = "<s>"
    
    prompt = system
    for msg in conversation:
        if msg["role"] == "user":
            prompt += f"[HUMAN] {msg['content']} [/HUMAN]\n\n"
        elif msg["role"] == "assistant":
            prompt += f"[AI] {msg['content']} [/AI]\n\n"
    
    prompt += "[AI]"
    return prompt

def no_op_callback(*args, **kwargs):
    """
    A no-operation callback function.
    
    This function does nothing and is used as a default for optional callbacks.
    """
    pass

def handle_chat_session(
    llm: LLM,
    on_message: Callable[[str, str], None],
    on_error: Callable[[str], None],
    on_thinking: Callable[[], None] = no_op_callback,
    system_prompt: Optional[str] = None,
    voice_output: Optional[Any] = None,
    temperature: float = 0.7,
    top_p: float = 0.9,
    repeat_penalty: float = 1.1,
    max_tokens: int = 1024
) -> List[Dict[str, str]]:
    """
    Run an interactive chat session with the given LLM.
    
    Args:
        llm: LLM instance to use
        on_message: Callback function for new messages (user, content)
        on_error: Callback function for errors
        on_thinking: Optional callback function for when the model is thinking
        system_prompt: Optional system prompt to use
        voice_output: Optional voice output module
        temperature: Temperature parameter for generation
        top_p: Top-p sampling parameter
        repeat_penalty: Repeat penalty parameter
        max_tokens: Maximum tokens to generate
        
    Returns:
        List of message dictionaries with 'role' and 'content' keys
    """
    conversation = []
    
    try:
        while True:
            # Get user input
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() in ("exit", "quit", "bye", "goodbye"):
                break
            
            if not user_input:
                continue
            
            # Add to conversation history
            conversation.append({"role": "user", "content": user_input})
            
            # Notify callback
            on_message("user", user_input)
            
            # Prepare prompt
            prompt = format_chat_prompt(conversation, system_prompt)
            
            # Notify thinking callback if provided
            if on_thinking:
                on_thinking()
            
            # Generate response
            try:
                response, _ = llm.generate(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    repeat_penalty=repeat_penalty
                )
                
                # Add to conversation history
                conversation.append({"role": "assistant", "content": response})
                
                # Notify callback
                on_message("assistant", response)
                
                # Voice output if enabled
                if voice_output:
                    try:
                        wav, sr = voice_output.generate(response)
                        # TODO: Implement audio playback
                        logger.info("Generated voice output")
                    except Exception as e:
                        logger.error(f"Failed to generate speech: {e}")
            
            except Exception as e:
                error_msg = f"Error generating response: {str(e)}"
                logger.error(error_msg)
                on_error(error_msg)
    
    except KeyboardInterrupt:
        logger.info("Chat session interrupted")
    
    return conversation