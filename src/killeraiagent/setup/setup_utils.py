# killeraiagent/setup/setup_utils.py

import os
import sys
import logging
import datetime
import argparse
import shutil
from pathlib import Path

from killeraiagent.hardware import detect_hardware_capabilities, AcceleratorType
from killeraiagent.setup.setup_wizard import run_wizard
from killeraiagent.setup.setup_core import (
    build_llama_cpp,
    install_torch_with_acceleration,
    install_voice_dependencies,
    install_stt_dependencies,
    install_gui_dependencies
)
from killeraiagent.paths import get_data_paths

def is_dev_mode() -> bool:
    return os.environ.get("KAIA_DEV_MODE", "").lower() in ("1", "true", "yes")

def setup_logger() -> None:
    paths = get_data_paths()
    log_dir = paths.logs
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"kaia_{timestamp}.log"
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.handlers = []
    fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    logging.info(f"Debug logs will be saved to: {log_file}")
    logging.info(f"Running in {'development' if is_dev_mode() else 'production'} mode")

def parse_args():
    parser = argparse.ArgumentParser(description="KAIA Setup Utilities")
    parser.add_argument("--all", action="store_true", help="Install all components")
    parser.add_argument("--torch", action="store_true", help="Install PyTorch with appropriate acceleration")
    parser.add_argument("--llama-cpp", action="store_true", help="Build llama.cpp with acceleration")
    parser.add_argument("--voice", action="store_true", help="Install voice support dependencies")
    parser.add_argument("--stt", action="store_true", help="Install STT support dependencies")
    parser.add_argument("--gui", action="store_true", help="Install GUI (PySide6) support")
    parser.add_argument("--acceleration", choices=["auto"] + [a.value for a in AcceleratorType],
                        default="auto", help="Acceleration type to use")
    parser.add_argument("--llama-cpp-path", type=str, help="Path to existing llama.cpp executable")
    return parser.parse_args()

def main():
    setup_logger()
    args = parse_args()
    logging.info("KAIA Setup Utilities")
    
    # If no flags provided, run the interactive wizard.
    if not any([
        args.all, args.torch, args.llama_cpp,
        args.voice, args.stt, args.gui, args.llama_cpp_path
    ]):
        logging.info("No flags detected, entering wizard mode.")
        run_wizard()
        return 0

    # Otherwise proceed with the legacy CLI installation process.
    if args.acceleration != "auto":
        os.environ["KAIA_ACCELERATION"] = args.acceleration

    hw_caps = detect_hardware_capabilities()
    logging.info(f"Detected primary accelerator: {hw_caps.primary_acceleration.value}")
    available = ", ".join(a.value for a in hw_caps.available_accelerators)
    logging.info(f"Available accelerators: {available}")

    if args.llama_cpp_path:
        cpp_path = Path(args.llama_cpp_path)
        if not cpp_path.exists():
            logging.error(f"Specified llama.cpp executable not found: {cpp_path}")
            return 1
        paths = get_data_paths()
        target_path = paths.bin / cpp_path.name
        shutil.copy2(cpp_path, target_path)
        logging.info(f"Copied {cpp_path.name} to {target_path}")

    success = True
    if args.all or args.torch:
        success = install_torch_with_acceleration() and success
    if args.all or args.llama_cpp and not args.llama_cpp_path:
        success = build_llama_cpp() and success
    if args.all or args.voice:
        success = install_voice_dependencies() and success
    if args.all or args.stt:
        success = install_stt_dependencies() and success
    if args.all or args.gui:
        success = install_gui_dependencies() and success

    if success:
        logging.info("Setup completed successfully!")
    else:
        logging.warning("Setup completed with some failures. Check the logs for details.")

    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())