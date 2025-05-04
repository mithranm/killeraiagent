"""
setup_wizard.py

Interactive wizard for KAIA setup that guides the user through installation steps.
At each prompt you may answer 'y' (yes), 'n' (no), or 'help' for additional details.
A comprehensive FAQ document (in JSON Lines format) is expected to reside in the user's
data directory (~/.kaia/faq/kaia_faq.jsonl); if not present, it is copied from the project 
resources directory.
"""

import logging
import shutil
import importlib.util
from pathlib import Path

from killeraiagent.hardware import detect_hardware_capabilities
from killeraiagent.setup.setup_core import (
    install_torch_with_acceleration,
    build_llama_cpp,
    get_data_paths,
)
from killeraiagent.multiturn_teapot import MultiturnTeapot

logger = logging.getLogger(__name__)

def is_module_installed(module_name: str) -> bool:
    spec = importlib.util.find_spec(module_name)
    return spec is not None

def prompt_with_help(prompt_text: str) -> str:
    """
    Prompt the user, accepting 'y', 'n', or 'help'. If the user answers 'help',
    an interactive help chat is launched for that topic.
    
    Returns "y" or "n".
    """
    while True:
        answer = input(f"{prompt_text} (y/n/help): ").strip().lower()
        if answer in ("y", "yes", "n", "no"):
            return "y" if answer in ("y", "yes") else "n"
        elif answer == "help":
            launch_help_chat(prompt_text)
            continue
        else:
            print("Please answer 'y', 'n', or 'help'.")

def launch_help_chat(topic: str) -> None:
    """
    Launch an interactive help chat about the given topic.
    """
    print(f"\n--- Help Session for: {topic} ---")
    assistant = MultiturnTeapot()
    assistant.rag_context = f"Help requested on topic: {topic}."
    print("Type your question about this step; type 'exit' to finish the help session.\n")
    while True:
        user_input = input("Help> ").strip()
        if user_input.lower() in ("exit", "quit"):
            print("Exiting help session.\n")
            break
        response = assistant.chat(user_input)
        print(f"Assistant: {response}\n")
    assistant.close()

def copy_faq_file_if_needed():
    """
    Copies the FAQ file from the project resources (resources/faq/kaia_faq.jsonl)
    to the user's data directory (~/.kaia/faq/kaia_faq.jsonl) if it does not exist.
    """
    paths = get_data_paths()
    faq_dir = paths.base / "faq"
    faq_dir.mkdir(parents=True, exist_ok=True)
    local_faq_path = faq_dir / "kaia_faq.jsonl"
    
    project_root = __file__.split("src/killeraiagent")[0]
    resource_faq_path = Path(project_root) / "resources" / "faq" / "kaia_faq.jsonl"
    
    if not local_faq_path.exists():
        try:
            shutil.copy2(str(resource_faq_path), str(local_faq_path))
            logger.info(f"Copied FAQ from {resource_faq_path} to {local_faq_path}")
        except Exception as e:
            logger.warning(f"Unable to copy FAQ from resources: {e}")

def is_llama_cpp_built() -> bool:
    """
    Check if local llama.cpp appears built by looking in the bin directory.
    If at least one binary (llama-cli or llama-server) exists, assume it is built.
    """
    paths = get_data_paths()
    bin_dir = paths.bin
    for binary in ["llama-cli", "llama-server"]:
        if any(bin_dir.glob(binary + "*")):
            return True
    return False

def run_wizard():
    print("--------------------------------------------------")
    print("Welcome to the KAIA Setup Wizard!")
    print("We will guide you step by step through the installation process.")
    print("At each prompt, answer 'y' (yes), 'n' (no), or 'help' for more details.\n")
    
    copy_faq_file_if_needed()
    
    hw = detect_hardware_capabilities()
    ram_gb = hw.memory.total_gb
    print(f"Detected {ram_gb:.1f} GB of RAM.\n")
    
    # Step 1: Check PyTorch installation.
    if is_module_installed("torch"):
        print("PyTorch is already installed with proper acceleration.\n")
    else:
        ans = prompt_with_help("Install PyTorch for hardware acceleration?")
        if ans == "y":
            if install_torch_with_acceleration():
                print("PyTorch installed successfully.\n")
            else:
                print("PyTorch installation failed. Check logs.\n")
        else:
            print("Skipping PyTorch installation.\n")
    
    # Step 2: Check teapot.ai installation.
    if is_module_installed("teapotai"):
        print("teapot.ai is already installed and will power on instantly.\n")
    else:
        ans = prompt_with_help("Install teapot.ai for easy QnA?")
        if ans == "y":
            from killeraiagent.setup.setup_core import install_teapot_ai
            if install_teapot_ai():
                print("teapot.ai installed successfully!\n")
            else:
                print("teapot.ai installation failed.\n")
        else:
            print("Skipping teapot.ai installation.\n")
    
    # Step 3: Offer to build local llama.cpp.
    if ram_gb >= 4:
        if is_llama_cpp_built():
            print("It appears that local llama.cpp is already built; skipping this step.\n")
            # Print the bin directory for user verification.
            bin_path = get_data_paths().bin.resolve()
            print(f"Built binaries are located here: {bin_path}\n")
        else:
            ans = prompt_with_help("Build llama.cpp for local inference?")
            if ans == "y":
                if build_llama_cpp():
                    print("llama.cpp built successfully.\n")
                    bin_path = get_data_paths().bin.resolve()
                    print(f"Binaries are located at: {bin_path}")
                    # Suggest opening the man page for further guidance.
                    project_root = Path(__file__).parent.parent.parent
                    doc_path = project_root / "docs" / "llamacpp_usage.md"
                    if doc_path.exists():
                        print(f"For detailed usage instructions, open: {doc_path.resolve()}\n")
                    else:
                        print("No detailed documentation found for llama.cpp usage.\n")
                    
                    ans2 = prompt_with_help("Download Gemma 3 1B (~800MB)?")
                    if ans2 == "y":
                        from killeraiagent.setup.setup_core import download_gemma_small
                        if download_gemma_small():
                            print("Gemma 3 1B downloaded successfully!\n")
                        else:
                            print("Gemma download failed.\n")
                else:
                    print("llama.cpp build failed.\n")
            else:
                print("Skipping local llama.cpp build.\n")
    else:
        print("System memory is less than 4GB; skipping local model build.\n")
    
    # Step 4: Offer to launch interactive help assistant.
    if is_module_installed("teapotai"):
        ans = prompt_with_help("Would you like to launch the interactive help assistant now?")
        if ans == "y":
            print("Launching interactive Teapot chat for help...\n")
            assistant = MultiturnTeapot()
            assistant.rag_context = ("Current status: PyTorch and teapot.ai are installed; "
                                     "check built binaries or documentation for details.")
            print("You can now ask questions about KAIA setup. For example, try 'doc llama' to get documentation on llama.cpp.")
            print("Type 'exit' to finish the help session.\n")
            while True:
                user_input = input("Help> ").strip()
                if user_input.lower() in ("exit", "quit"):
                    print("Exiting help session.\n")
                    break
                response = assistant.chat(user_input)
                print(f"Assistant: {response}\n")
            assistant.close()
    else:
        print("teapot.ai not installed. Skipping help session.\n")
    
    print("Wizard complete! You can now run KAIA or import the library as needed.\n")

if __name__ == "__main__":
    run_wizard()