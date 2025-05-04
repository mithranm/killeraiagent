#!/usr/bin/env python
"""
Command-line interface for KillerAI Agent.
"""

import logging
import argparse

from killeraiagent.models import ModelInfo, create_llm_instance
from killeraiagent.paths import get_data_paths
from killeraiagent.hardware import detect_hardware_capabilities, get_optimal_model_config
from killeraiagent.core import (
    load_model,
    initialize_voice_support,
    handle_chat_session
)

logger = logging.getLogger(__name__)

def setup_logging():
    """Configure logging for the CLI."""
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_format)
    
    # Get log directory from data paths
    data_paths = get_data_paths()
    log_file = data_paths.logs / "kaia_cli.log"
    data_paths.logs.mkdir(parents=True, exist_ok=True)
    
    # Add file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(file_handler)
    
def chat_mode(args):
    """
    Start a chat session with the specified model.
    
    Args:
        args: Command-line arguments
    """
    # Load specified model or find a default
    model_info = None
    
    # Detect hardware capabilities
    hw = detect_hardware_capabilities()
    
    if args.model:
        model_info = load_model(args.model)
    
    if not model_info:
        # Try to find a suitable default model
        available_models = list_available_models()

        if not available_models:
            logger.error("No models available. Please download a model first.")
            return
        
        # For simplicity, just use the first model
        model_info = ModelInfo(**available_models[0])
        logger.info(f"Using default model: {model_info.model_id}")
    
    # Get optimal configuration based on hardware and model size
    model_size_gb = model_info.model_size_gb
    optimal_config = get_optimal_model_config(hw, model_size_gb)
    
    # Override with command line args if specified
    n_gpu_layers = args.n_gpu_layers if args.n_gpu_layers is not None else optimal_config["n_gpu_layers"]
    n_threads = args.threads if args.threads is not None else optimal_config["n_threads"]
    context_size = args.context_size if args.context_size is not None else optimal_config["context_size"]
    
    # Initialize model with optimal configuration
    llm = create_llm_instance(
        model_info=model_info,
        use_llama_cpp=args.use_llama_cpp,  # New flag
        n_gpu_layers=n_gpu_layers,
        n_threads=n_threads,
        context_size=context_size,
        use_mlock=optimal_config["use_mlock"],
        use_mmap=optimal_config["use_mmap"],
        batch_size=optimal_config["batch_size"]
    )
    
    if not llm:
        logger.error("Failed to create LLM instance")
        return
    
    # Load the model
    logger.info(f"Loading model {model_info.model_id}...")
    if not llm.load():
        logger.error("Failed to load model")
        return
    
    # Set up voice if requested
    voice_output, voice_input = None, None
    if args.voice:
        voice_output, voice_input = initialize_voice_support()
    
    # Start interactive chat
    logger.info("Starting chat session. Type 'exit' or 'quit' to end.")
    print(f"\nKAIA Chat with {model_info.model_id}")
    print("Type 'exit' or 'quit' to end the session.")
    
    # Define callback functions
    def on_message(role, content):
        if role == "assistant":
            print(f"\nAssistant: {content}")
    
    def on_error(error_msg):
        print(f"\nError: {error_msg}")
    
    def on_thinking():
        print("Thinking...", end="", flush=True)
    
    # Run the chat session
    try:
        handle_chat_session(
            llm=llm,
            on_message=on_message,
            on_error=on_error,
            on_thinking=on_thinking,
            system_prompt=args.system_prompt,
            voice_output=voice_output,
            temperature=args.temperature,
            top_p=args.top_p,
            repeat_penalty=args.repeat_penalty,
            max_tokens=args.max_tokens
        )
    finally:
        # Clean up
        llm.unload()
        print("\nChat session ended.")

def teapot_chat_mode(args):
    """
    Start a chat session with the Teapot model.
    
    Args:
        args: Command-line arguments
    """
    try:
        from killeraiagent.multiturn_teapot import MultiturnTeapot
        
        # Initialize teapot
        teapot = MultiturnTeapot()
        
        # Set up voice if requested
        voice_output, voice_input = None, None
        if args.voice:
            voice_output, voice_input = initialize_voice_support()
        
        print("\nKAIA Teapot Chat")
        print("Type 'exit' or 'quit' to end the session.")
        
        # Handle initial query if provided
        if args.query:
            response = teapot.chat(args.query)
            print(f"\nYou: {args.query}")
            print(f"Teapot: {response}")
            
            if voice_output:
                try:
                    wav, sr = voice_output.generate(response)
                    # TODO: Implement audio playback
                    logger.info("Generated voice output")
                except Exception as e:
                    logger.error(f"Failed to generate speech: {e}")
        
        # Interactive loop
        while True:
            user_input = input("\nYou: ")
            
            if user_input.lower() in ("exit", "quit", "bye", "goodbye"):
                break
            
            if not user_input.strip():
                continue
            
            response = teapot.chat(user_input)
            print(f"Teapot: {response}")
            
            if voice_output:
                try:
                    wav, sr = voice_output.generate(response)
                    # TODO: Implement audio playback
                    logger.info("Generated voice output")
                except Exception as e:
                    logger.error(f"Failed to generate speech: {e}")
        
        # Clean up
        teapot.close()
        print("\nChat session ended.")
        
    except ImportError as e:
        logger.error(f"Failed to import MultiturnTeapot: {e}")
        print("Teapot is not available. Make sure teapotai is installed.")

def main():
    """Main entry point for the KAIA CLI."""
    setup_logging()
    
    parser = argparse.ArgumentParser(description="KAIA - Killer AI Agent")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # List models command
    list_parser = subparsers.add_parser("list", help="List available models")
    list_parser.add_argument("--engine", choices=["all", "llamacpp", "hf"], 
                          default="all", help="Filter models by engine type")
    list_parser.add_argument("--sort", choices=["size", "name"], 
                          default="name", help="Sort models by size or name")
    list_parser.add_argument("--detailed", action="store_true", 
                          help="Show detailed model information")
    
    # Chat command
    chat_parser = subparsers.add_parser("chat", help="Start a chat session with a model")
    chat_parser.add_argument("--model", type=str, help="Model ID to use")
    chat_parser.add_argument("--use-llama-cpp", action="store_true", 
                            help="Use llama.cpp backend instead of transformers")
    chat_parser.add_argument("--n-gpu-layers", type=int, default=0, help="Number of GPU layers")
    chat_parser.add_argument("--threads", type=int, default=4, help="Number of CPU threads")
    chat_parser.add_argument("--context-size", type=int, default=4096, help="Context window size")
    chat_parser.add_argument("--temperature", type=float, default=0.7, help="Temperature")
    chat_parser.add_argument("--top-p", type=float, default=0.9, help="Top-p sampling")
    chat_parser.add_argument("--repeat-penalty", type=float, default=1.1, help="Repeat penalty")
    chat_parser.add_argument("--max-tokens", type=int, default=1024, help="Max tokens to generate")
    chat_parser.add_argument("--system-prompt", type=str, help="System prompt")
    chat_parser.add_argument("--voice", action="store_true", help="Enable voice mode")
    
    # Teapot chat command
    teapot_parser = subparsers.add_parser("teapot-chat", help="Start a chat session with Teapot")
    teapot_parser.add_argument("--query", type=str, help="Initial query")
    teapot_parser.add_argument("--voice", action="store_true", help="Enable voice mode")
    
    # GUI command
    gui_parser = subparsers.add_parser("gui", help="Start the GUI")
    gui_parser.add_argument("--voice", action="store_true", help="Enable voice mode")
    
    # Global arguments
    parser.add_argument("--gui", action="store_true", help="Start the GUI")
    parser.add_argument("--voice", action="store_true", help="Enable voice mode")
    
    args = parser.parse_args()
    
    # Handle the case where --gui is passed as a global argument
    if args.gui:
        raise NotImplementedError("GUI mode is not implemented yet.")
    
    if args.command == "list":
        models = list_available_models()
        
        # Filter by engine if specified
        if args.engine != "all":
            models = [m for m in models if m["model_engine"] == args.engine]
            
        # Sort models
        if args.sort == "size":
            models.sort(key=lambda x: x["model_size_gb"], reverse=True)
        else:  # sort by name
            models.sort(key=lambda x: x["model_id"].lower())
            
        print(f"\nFound {len(models)} models:")
        for model in models:
            print(f"\n- {model['model_id']}")
            print(f"  Engine: {model['model_engine']}")
            print(f"  Size: {model['model_size_gb']:.1f} GB")
            
            if args.detailed:
                print(f"  Path: {model['model_path']}")
                print(f"  Description: {model['description']}")
                if model.get("quantization"):
                    print(f"  Quantization: {model['quantization']}")
                if model.get("context_length"):
                    print(f"  Context Length: {model['context_length']}")
                print(f"  Requires GPU: {model.get('requires_gpu', False)}")
    
    elif args.command == "chat":
        chat_mode(args)
    
    elif args.command == "teapot-chat":
        teapot_chat_mode(args)
    else:
        # Default to showing help if no command specified
        parser.print_help()

if __name__ == "__main__":
    main()