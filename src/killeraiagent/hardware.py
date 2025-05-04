"""
Modern hardware detection system for AI acceleration.

This module provides comprehensive detection of hardware capabilities for optimal
AI model inference across CPU, GPU, and specialized accelerators.
"""

import os
import platform
import subprocess
import logging
import shutil
from enum import Enum
from typing import List, Optional, Dict, Any, Set
from pathlib import Path

import psutil
import torch
from pydantic import BaseModel, Field, validator

from killeraiagent.paths import DataPaths

# Conditionally import winreg with ignores for Pyright
if platform.system() == "Windows":
    import winreg

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class AcceleratorType(str, Enum):
    """Supported hardware acceleration backends."""
    CPU = "cpu"
    CUDA = "cuda"  # NVIDIA CUDA
    ROCM = "rocm"  # AMD ROCm/HIP
    METAL = "metal"  # Apple Metal
    VULKAN = "vulkan"  # Vulkan (cross-platform)
    OPENCL = "opencl"  # OpenCL (cross-platform)
    SYCL = "sycl"  # Intel SYCL
    MUSA = "musa"  # Moore Threads MUSA
    CANN = "cann"  # Huawei Ascend CANN
    KLEIDIAI = "kleidiai"  # Arm KleidiAI

class CUDAInfo(BaseModel):
    """Information about NVIDIA CUDA installation."""
    available: bool = False
    version: Optional[str] = None
    compute_capability: Optional[str] = None
    device_count: int = 0
    devices: List[Dict[str, Any]] = Field(default_factory=list)
    driver_version: Optional[str] = None
    nvcc_path: Optional[Path] = None
    cuda_path: Optional[Path] = None
    
    @property
    def cuda_version_major(self) -> Optional[int]:
        """Get major CUDA version number."""
        if not self.version:
            return None
        try:
            return int(self.version.split('.')[0])
        except (ValueError, IndexError):
            return None
    
    @property
    def cuda_version_minor(self) -> Optional[int]:
        """Get minor CUDA version number."""
        if not self.version:
            return None
        try:
            return int(self.version.split('.')[1])
        except (ValueError, IndexError):
            return None
    
    @property
    def cuda_wheel_version(self) -> Optional[str]:
        """Get the PyTorch CUDA wheel version string."""
        if not self.version:
            return None
        
        major = self.cuda_version_major
        minor = self.cuda_version_minor
        
        if major is None or minor is None:
            return None
            
        if major >= 12:
            return "cu121"  # Latest
        elif major == 11:
            if minor >= 8:
                return "cu118"
            elif minor >= 7:
                return "cu117"
            else:
                return "cu116"
        return None


class ROCmInfo(BaseModel):
    """Information about AMD ROCm/HIP installation."""
    available: bool = False
    version: Optional[str] = None
    device_count: int = 0
    devices: List[Dict[str, Any]] = Field(default_factory=list)
    driver_version: Optional[str] = None
    rocm_path: Optional[Path] = None
    architectures: List[str] = Field(default_factory=list)


class MetalInfo(BaseModel):
    """Information about Apple Metal GPU."""
    available: bool = False
    is_apple_silicon: bool = False
    device_count: int = 0
    devices: List[Dict[str, Any]] = Field(default_factory=list)


class VulkanInfo(BaseModel):
    """Information about Vulkan support."""
    available: bool = False
    version: Optional[str] = None
    device_count: int = 0
    devices: List[Dict[str, Any]] = Field(default_factory=list)
    icd_loader_available: bool = False


class OpenCLInfo(BaseModel):
    """Information about OpenCL support."""
    available: bool = False
    version: Optional[str] = None
    device_count: int = 0
    devices: List[Dict[str, Any]] = Field(default_factory=list)
    platforms: List[Dict[str, Any]] = Field(default_factory=list)


class CPUInfo(BaseModel):
    """Detailed CPU information."""
    count: int = Field(gt=0)
    physical_cores: int = Field(gt=0)
    logical_cores: int = Field(gt=0)
    architecture: str
    model_name: str
    frequency_mhz: float
    supports_avx: bool = False
    supports_avx2: bool = False
    supports_avx512: bool = False
    supports_vnni: bool = False
    supports_arm_sve: bool = False
    supports_arm_neon: bool = False
    supports_kleidiai: bool = False


class MemoryInfo(BaseModel):
    """System memory information."""
    total_gb: float = Field(gt=0)
    available_gb: float = Field(ge=0)
    used_gb: float
    percent_used: float = Field(ge=0, le=100)


class GPUInfo(BaseModel):
    """Unified GPU information across different backends."""
    acceleration_type: AcceleratorType
    name: str
    vendor: str
    driver_version: Optional[str] = None
    total_memory_mb: float = 0
    free_memory_mb: float = 0
    compute_capability: Optional[str] = None
    device_id: int = 0
    pci_bus_id: Optional[str] = None


class LlamaCppConfig(BaseModel):
    """Configuration for llama.cpp."""
    build_type: str = "Release"
    n_gpu_layers: int = 0
    n_threads: int = 4
    acceleration: AcceleratorType = AcceleratorType.CPU
    main_gpu: int = 0
    tensor_split: Optional[List[float]] = None
    executable_path: Optional[Path] = None
    cmake_args: List[str] = Field(default_factory=list)
    env_vars: Dict[str, str] = Field(default_factory=dict)
    
    @validator('executable_path')
    def check_executable_exists(cls, v):
        """Verify the specified executable exists."""
        if v is not None and not v.exists():
            raise ValueError(f"Executable not found: {v}")
        return v


class HardwareCapabilities(BaseModel):
    """Complete hardware capabilities of the system."""
    cpu: CPUInfo
    memory: MemoryInfo
    primary_acceleration: AcceleratorType = AcceleratorType.CPU
    gpus: List[GPUInfo] = Field(default_factory=list)
    cuda: CUDAInfo = Field(default_factory=CUDAInfo)
    rocm: ROCmInfo = Field(default_factory=ROCmInfo)
    metal: MetalInfo = Field(default_factory=MetalInfo)
    vulkan: VulkanInfo = Field(default_factory=VulkanInfo)
    opencl: OpenCLInfo = Field(default_factory=OpenCLInfo)
    
    @property
    def available_accelerators(self) -> Set[AcceleratorType]:
        """Get the set of available acceleration backends."""
        result = {AcceleratorType.CPU}
        
        if self.cuda.available:
            result.add(AcceleratorType.CUDA)
        
        if self.rocm.available:
            result.add(AcceleratorType.ROCM)
            
        if self.metal.available:
            result.add(AcceleratorType.METAL)
            
        if self.vulkan.available:
            result.add(AcceleratorType.VULKAN)
            
        if self.opencl.available:
            result.add(AcceleratorType.OPENCL)
            
        if self.cpu.supports_kleidiai:
            result.add(AcceleratorType.KLEIDIAI)
            
        return result


def detect_cpu_capabilities() -> CPUInfo:
    """
    Detect CPU capabilities including architecture, cores, and supported instruction sets.
    
    Returns:
        CPUInfo: Detailed CPU information
    """
    cpu_count = os.cpu_count() or 4
    physical_cores = psutil.cpu_count(logical=False) or 2
    logical_cores = psutil.cpu_count(logical=True) or 4
    
    # Get CPU model information
    if platform.system() == "Windows":
        try:
            # Use winreg if available
            if 'winreg' in globals():
                key_path = r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
                with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path) as _key:  # type: ignore[attr-defined]
                    model_name = winreg.QueryValueEx(_key, "ProcessorNameString")[0]  # type: ignore[attr-defined]
            else:
                model_name = platform.processor()
        except Exception:
            model_name = platform.processor()
    elif platform.system() == "Darwin":
        try:
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"], 
                capture_output=True, text=True, check=True
            )
            model_name = result.stdout.strip()
        except Exception:
            model_name = platform.processor()
    else:  # Linux and others
        try:
            with open("/proc/cpuinfo", "r") as f:
                model_name = platform.processor()
                for line in f:
                    if "model name" in line:
                        model_name = line.split(":", 1)[1].strip()
                        break
        except Exception:
            model_name = platform.processor()
    
    # Get CPU frequency
    try:
        frequency = psutil.cpu_freq().max
        if not frequency:
            frequency = psutil.cpu_freq().current
    except Exception:
        frequency = 0
    
    # Detect architecture
    arch = platform.machine().lower()
    
    # Detect instruction set support
    supports_avx = False
    supports_avx2 = False
    supports_avx512 = False
    supports_vnni = False
    supports_arm_sve = False
    supports_arm_neon = False
    supports_kleidiai = False
    
    if arch in ("x86_64", "amd64", "i386", "i686"):
        try:
            # Try using cpuinfo library if available
            try:
                import py_cpuinfo  # type: ignore[import]
                info = py_cpuinfo.get_cpu_info()
                flags = info.get("flags", [])
                supports_avx = "avx" in flags
                supports_avx2 = "avx2" in flags
                supports_avx512 = any(flag.startswith("avx512") for flag in flags)
                supports_vnni = "avx512_vnni" in flags or "avx_vnni" in flags
            except ImportError:
                logger.debug("py_cpuinfo not available, using fallback detection")
                # Fallback to manual detection
                if platform.system() == "Linux":
                    try:
                        with open("/proc/cpuinfo", "r") as f:
                            content = f.read().lower()
                            supports_avx = " avx " in content
                            supports_avx2 = " avx2 " in content
                            supports_avx512 = " avx512" in content
                            supports_vnni = (" avx512_vnni" in content 
                                             or " avx_vnni" in content)
                    except Exception:
                        pass
                elif platform.system() == "Darwin":
                    try:
                        result = subprocess.run(
                            ["sysctl", "-n", "machdep.cpu.features"], 
                            capture_output=True, text=True, check=True
                        )
                        features = result.stdout.lower()
                        supports_avx = "avx" in features
                        supports_avx2 = "avx2" in features
                        supports_avx512 = "avx512" in features
                    except Exception:
                        pass
                elif platform.system() == "Windows":
                    # Minimal fallback for Windows if py_cpuinfo isn't installed
                    pass
        except Exception as e:
            logger.warning(f"Failed to detect CPU instruction set support: {e}")
    
    elif arch in ("arm64", "aarch64", "armv8"):
        supports_arm_neon = True  # Most ARM64 processors support NEON
        
        # Check for SVE support
        if platform.system() == "Linux":
            try:
                with open("/proc/cpuinfo", "r") as f:
                    content = f.read().lower()
                    supports_arm_sve = "sve" in content
            except Exception:
                pass
            
        # On Apple Silicon, we know NEON is available
        if platform.system() == "Darwin" and platform.machine() == "arm64":
            supports_arm_neon = True
            
        # Check for ARM KleidiAI support (simplified assumption)
        try:
            result = subprocess.run(["uname", "-a"], capture_output=True, text=True, check=False)
            supports_kleidiai = supports_arm_neon  # Simplified placeholder
        except Exception:
            pass
    
    return CPUInfo(
        count=cpu_count,
        physical_cores=physical_cores,
        logical_cores=logical_cores,
        architecture=arch,
        model_name=model_name,
        frequency_mhz=frequency,
        supports_avx=supports_avx,
        supports_avx2=supports_avx2,
        supports_avx512=supports_avx512,
        supports_vnni=supports_vnni,
        supports_arm_sve=supports_arm_sve,
        supports_arm_neon=supports_arm_neon,
        supports_kleidiai=supports_kleidiai
    )


def detect_memory() -> MemoryInfo:
    """
    Detect system memory information.
    
    Returns:
        MemoryInfo: System memory details
    """
    mem = psutil.virtual_memory()
    return MemoryInfo(
        total_gb=mem.total / (1024**3),
        available_gb=mem.available / (1024**3),
        used_gb=(mem.total - mem.available) / (1024**3),
        percent_used=mem.percent
    )


def detect_cuda() -> CUDAInfo:
    """
    Detect NVIDIA CUDA capabilities.
    
    Returns:
        CUDAInfo: CUDA hardware and software information
    """
    result = CUDAInfo(available=False)
    
    # Check if CUDA is available through PyTorch
    if not torch.cuda.is_available():
        return result
    
    # CUDA is available
    result.available = True
    result.device_count = torch.cuda.device_count()
    
    # Get CUDA version
    try:
        # Access CUDA version through PyTorch
        result.version = torch.version.cuda  # type: ignore[attr-defined]
    except Exception:
        # Fallback: parse with nvcc
        try:
            nvcc_path = shutil.which("nvcc")
            if nvcc_path:
                result.nvcc_path = Path(nvcc_path)
                output = subprocess.run(
                    [nvcc_path, "--version"], 
                    capture_output=True, text=True, check=True
                )
                import re
                version_match = re.search(r"release (\d+\.\d+)", output.stdout)
                if version_match:
                    result.version = version_match.group(1)
        except Exception as e:
            logger.warning(f"Failed to detect CUDA version: {e}")
    
    # Get CUDA driver version
    try:
        # Access driver version
        result.driver_version = torch.version.cuda  # type: ignore[attr-defined]
    except Exception:
        # Try nvidia-smi
        try:
            nvidia_smi = shutil.which("nvidia-smi")
            if nvidia_smi:
                output = subprocess.run(
                    [nvidia_smi, "--query-gpu=driver_version", "--format=csv,noheader"], 
                    capture_output=True, text=True, check=True
                )
                result.driver_version = output.stdout.strip()
        except Exception as e:
            logger.warning(f"Failed to detect NVIDIA driver version: {e}")
    
    # Get CUDA path
    cuda_path = os.environ.get("CUDA_PATH")
    if cuda_path:
        result.cuda_path = Path(cuda_path)
    else:
        # Try to find CUDA path from nvcc
        if result.nvcc_path:
            # nvcc is typically in CUDA_PATH/bin/nvcc
            result.cuda_path = result.nvcc_path.parent.parent
    
    # Get device information
    for i in range(result.device_count):
        try:
            device_props = torch.cuda.get_device_properties(i)
            
            # Get compute capability
            if i == 0:
                result.compute_capability = f"{device_props.major}.{device_props.minor}"
            
            # Calculate memory
            total_memory = device_props.total_memory
            try:
                free_memory = (
                    torch.cuda.memory_reserved(i) - torch.cuda.memory_allocated(i)
                )
            except Exception:
                free_memory = total_memory * 0.8  # Estimate
                
            device_info = {
                "name": device_props.name,
                "compute_capability": f"{device_props.major}.{device_props.minor}",
                "total_memory_mb": total_memory / (1024**2),
                "free_memory_mb": free_memory / (1024**2),
                "multi_processor_count": device_props.multi_processor_count,
                "device_id": i
            }
            result.devices.append(device_info)
        except Exception as e:
            logger.warning(f"Error getting properties for CUDA device {i}: {e}")
            
    return result


def detect_rocm() -> ROCmInfo:
    """
    Detect AMD ROCm/HIP capabilities.
    
    Returns:
        ROCmInfo: ROCm hardware and software information
    """
    result = ROCmInfo(available=False)
    
    # Check if rocm-smi is available
    rocm_smi = shutil.which("rocm-smi")
    if not rocm_smi:
        return result
    
    # Try to get ROCm version
    try:
        output = subprocess.run(
            [rocm_smi, "--showversion"], 
            capture_output=True, text=True, check=False
        )
        if output.returncode == 0:
            result.available = True
            import re
            version_match = re.search(r"ROCm-SMI Version: ([\d\.]+)", output.stdout)
            if version_match:
                result.version = version_match.group(1)
    except Exception as e:
        logger.warning(f"Failed to detect ROCm version: {e}")
        return result
    
    # Try to get GPU information
    try:
        output = subprocess.run(
            [rocm_smi, "--showallinfo"], 
            capture_output=True, text=True, check=False
        )
        if output.returncode == 0:
            import re
            gpu_matches = re.findall(r"GPU\[(\d+)\]", output.stdout)
            unique_gpus = set(gpu_matches)
            result.device_count = len(unique_gpus)
            
            for gpu_id in unique_gpus:
                mem_total_match = re.search(
                    rf"GPU\[{gpu_id}\].*?Total Memory.*?: (\d+) (\w+)", 
                    output.stdout, re.DOTALL
                )
                mem_used_match = re.search(
                    rf"GPU\[{gpu_id}\].*?Used Memory.*?: (\d+) (\w+)", 
                    output.stdout, re.DOTALL
                )
                
                total_memory_mb = 0
                free_memory_mb = 0
                
                if mem_total_match and mem_used_match:
                    total_val = int(mem_total_match.group(1))
                    total_unit = mem_total_match.group(2)
                    used_val = int(mem_used_match.group(1))
                    used_unit = mem_used_match.group(2)
                    
                    unit_multipliers = {"MB": 1, "GB": 1024, "KB": 1/1024}
                    total_memory_mb = total_val * unit_multipliers.get(total_unit, 1)
                    used_memory_mb = used_val * unit_multipliers.get(used_unit, 1)
                    free_memory_mb = total_memory_mb - used_memory_mb
                
                name_match = re.search(
                    rf"GPU\[{gpu_id}\].*?Card series.*?: (.*?)$", 
                    output.stdout, re.MULTILINE
                )
                name = name_match.group(1) if name_match else f"AMD GPU {gpu_id}"
                
                arch_match = re.search(
                    rf"GPU\[{gpu_id}\].*?gfx\s*:\s*(gfx\d+)", 
                    output.stdout, re.DOTALL
                )
                arch = arch_match.group(1) if arch_match else None
                
                if arch and arch not in result.architectures:
                    result.architectures.append(arch)
                
                device_info = {
                    "name": name,
                    "device_id": int(gpu_id),
                    "total_memory_mb": total_memory_mb,
                    "free_memory_mb": free_memory_mb,
                    "architecture": arch
                }
                result.devices.append(device_info)
    except Exception as e:
        logger.warning(f"Failed to detect ROCm GPU information: {e}")
    
    # Try to get ROCm path
    result.rocm_path = Path("/opt/rocm") if os.path.exists("/opt/rocm") else None
    
    return result


def detect_metal() -> MetalInfo:
    """
    Detect Apple Metal GPU capabilities.
    
    Returns:
        MetalInfo: Metal hardware information
    """
    result = MetalInfo(available=False)
    
    # Only check on macOS
    if platform.system() != "Darwin":
        return result
    
    # Check if we're on Apple Silicon
    result.is_apple_silicon = platform.machine() == "arm64"
    
    # Check if Metal is available through PyTorch
    has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    result.available = has_mps
    
    if not result.available:
        # Try checking with system_profiler
        try:
            output = subprocess.run(
                ["system_profiler", "SPDisplaysDataType"], 
                capture_output=True, text=True, check=True
            )
            result.available = "Metal" in output.stdout
        except Exception:
            pass
    
    # For Apple Silicon, we know there's typically one integrated GPU
    if result.is_apple_silicon and result.available:
        result.device_count = 1
        
        try:
            output = subprocess.run(
                ["system_profiler", "SPDisplaysDataType"], 
                capture_output=True, text=True, check=True
            )
            import re
            chip_match = re.search(r"Chipset Model: (.*?)$", output.stdout, re.MULTILINE)
            gpu_name = chip_match.group(1).strip() if chip_match else "Apple GPU"
            
            vram_match = re.search(r"VRAM \(.*?\): (\d+) (\w+)", output.stdout)
            total_memory_mb = 0
            if vram_match:
                vram_val = int(vram_match.group(1))
                vram_unit = vram_match.group(2)
                if vram_unit.upper() == "GB":
                    total_memory_mb = vram_val * 1024
                elif vram_unit.upper() == "MB":
                    total_memory_mb = vram_val
            else:
                system_memory = psutil.virtual_memory().total / (1024**3)
                total_memory_mb = int(system_memory * 0.4 * 1024)
            
            device_info = {
                "name": gpu_name,
                "device_id": 0,
                "total_memory_mb": total_memory_mb,
                "free_memory_mb": total_memory_mb * 0.8,
                "metal_family": "Apple"
            }
            result.devices.append(device_info)
        except Exception as e:
            logger.warning(f"Failed to detect Metal GPU information: {e}")
            # Provide a fallback device
            device_info = {
                "name": "Apple GPU",
                "device_id": 0,
                "total_memory_mb": 4096,
                "free_memory_mb": 3276,
                "metal_family": "Apple"
            }
            result.devices.append(device_info)
    
    return result


def detect_vulkan() -> VulkanInfo:
    """
    Detect Vulkan support.
    
    Returns:
        VulkanInfo: Vulkan hardware and driver information
    """
    result = VulkanInfo(available=False)
    
    vulkaninfo = shutil.which("vulkaninfo")
    if not vulkaninfo:
        # Check Windows registry for Vulkan
        if platform.system() == "Windows" and 'winreg' in globals():
            try:
                with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Khronos\Vulkan") as _key:  # type: ignore[attr-defined]
                    result.available = True
                    result.icd_loader_available = True
            except Exception:
                pass
        return result
    
    try:
        output = subprocess.run(
            [vulkaninfo], capture_output=True, text=True, check=False
        )
        if output.returncode == 0:
            result.available = True
            result.icd_loader_available = True
            
            import re
            version_match = re.search(r"Vulkan Instance Version: (\d+\.\d+\.\d+)", output.stdout)
            if version_match:
                result.version = version_match.group(1)
            
            device_matches = re.findall(r"GPU id : (\d+)", output.stdout)
            unique_devices = set(device_matches)
            result.device_count = len(unique_devices)
            
            for section in output.stdout.split("VkPhysicalDeviceProperties:"):
                if not section.strip():
                    continue
                
                name_match = re.search(r"deviceName\s*=\s*(\S.*?)\s*$", section, re.MULTILINE)
                if not name_match:
                    continue
                device_name = name_match.group(1)
                
                device_id_match = re.search(r"deviceID\s*=\s*(\S+)", section)
                device_id = int(device_id_match.group(1), 16) if device_id_match else 0
                
                vendor_match = re.search(r"vendorID\s*=\s*(\S+)", section)
                vendor_id = vendor_match.group(1) if vendor_match else "0x0000"
                
                vendor_map = {
                    "0x1002": "AMD",
                    "0x10DE": "NVIDIA",
                    "0x8086": "Intel",
                    "0x106B": "Apple"
                }
                vendor_name = vendor_map.get(vendor_id.upper(), "Unknown")
                
                memory_match = re.search(
                    r"VkPhysicalDeviceMemoryProperties:\s*memoryHeapCount\s*=\s*(\d+)",
                    section
                )
                total_memory_mb = 0
                
                if memory_match:
                    heap_count = int(memory_match.group(1))
                    for i in range(heap_count):
                        size_match = re.search(
                            rf"memoryHeaps\[{i}\]\.size\s*=\s*(\d+)", 
                            section
                        )
                        if size_match:
                            size_bytes = int(size_match.group(1))
                            size_mb = size_bytes / (1024**2)
                            total_memory_mb = max(total_memory_mb, size_mb)
                
                device_info = {
                    "name": device_name,
                    "device_id": device_id,
                    "vendor": vendor_name,
                    "total_memory_mb": int(total_memory_mb),
                    "free_memory_mb": int(total_memory_mb * 0.8)
                }
                
                result.devices.append(device_info)
    except Exception as e:
        logger.warning(f"Failed to detect Vulkan information: {e}")
    
    return result


def detect_opencl() -> OpenCLInfo:
    """
    Detect OpenCL support.
    
    Returns:
        OpenCLInfo: OpenCL hardware and software information
    """
    result = OpenCLInfo(available=False)
    
    # Attempt to import pyopencl
    try:
        import pyopencl as cl  # type: ignore[import]
        result.available = True
    except ImportError:
        # Fallback to clinfo
        clinfo = shutil.which("clinfo")
        if clinfo:
            try:
                output = subprocess.run(
                    [clinfo],
                    capture_output=True,
                    text=True,
                    check=False
                )
                if output.returncode == 0:
                    result.available = True
                    import re
                    # We no longer store platform_count
                    device_matches = re.findall(r"Device #(\d+):", output.stdout)
                    result.device_count = len(device_matches)
                    
                    version_match = re.search(
                        r"Platform Version\s*:\s*OpenCL (\d+\.\d+)", output.stdout
                    )
                    if version_match:
                        result.version = version_match.group(1)
            except Exception:
                pass
        return result
    
    # If pyopencl is installed, gather info
    try:
        import pyopencl as cl  # type: ignore[import]
        platforms = cl.get_platforms()
        result.device_count = 0
        
        for platform_id, platform_ in enumerate(platforms):
            platform_info = {
                "name": platform_.name,
                "vendor": platform_.vendor,
                "version": platform_.version,
                "profile": platform_.profile,
                "platform_id": platform_id,
                "devices": []
            }
            
            if not result.version:
                import re
                version_match = re.search(r"OpenCL (\d+\.\d+)", platform_.version)
                if version_match:
                    result.version = version_match.group(1)
            
            devices = platform_.get_devices()
            result.device_count += len(devices)
            
            for device_id, device in enumerate(devices):
                total_memory_mb = device.global_mem_size / (1024**2)
                
                device_info = {
                    "name": device.name,
                    "vendor": device.vendor,
                    "device_type": str(device.type).replace("cl.device_type.", ""),
                    "version": device.version,
                    "driver_version": device.driver_version,
                    "max_compute_units": device.max_compute_units,
                    "max_work_group_size": device.max_work_group_size,
                    "total_memory_mb": total_memory_mb,
                    "free_memory_mb": total_memory_mb * 0.8,
                    "platform_id": platform_id,
                    "device_id": device_id
                }
                
                platform_info["devices"].append(device_info)
                result.devices.append({
                    "name": device.name,
                    "vendor": device.vendor,
                    "device_type": str(device.type).replace("cl.device_type.", ""),
                    "total_memory_mb": total_memory_mb,
                    "free_memory_mb": total_memory_mb * 0.8,
                    "platform_name": platform_.name,
                    "platform_id": platform_id,
                    "device_id": device_id
                })
            
            result.platforms.append(platform_info)
    except Exception as e:
        logger.warning(f"Failed to detect OpenCL information: {e}")
    
    return result


def detect_hardware_capabilities() -> HardwareCapabilities:
    """
    Detect all hardware capabilities of the system.
    
    Returns:
        HardwareCapabilities: Complete hardware information
    """
    # Detect CPU capabilities
    cpu_info = detect_cpu_capabilities()
    
    # Detect memory
    memory_info = detect_memory()
    
    # Detect GPU backends
    cuda_info = detect_cuda()
    rocm_info = detect_rocm()
    metal_info = detect_metal()
    vulkan_info = detect_vulkan()
    opencl_info = detect_opencl()
    
    # Compile unified GPU list
    gpus = []
    
    # Add CUDA GPUs
    for device in cuda_info.devices:
        gpus.append(GPUInfo(
            acceleration_type=AcceleratorType.CUDA,
            name=device["name"],
            vendor="NVIDIA",
            driver_version=cuda_info.driver_version,
            total_memory_mb=device["total_memory_mb"],
            free_memory_mb=device["free_memory_mb"],
            compute_capability=device["compute_capability"],
            device_id=device["device_id"]
        ))
    
    # Add ROCm GPUs
    for device in rocm_info.devices:
        gpus.append(GPUInfo(
            acceleration_type=AcceleratorType.ROCM,
            name=device["name"],
            vendor="AMD",
            driver_version=rocm_info.version,
            total_memory_mb=device["total_memory_mb"],
            free_memory_mb=device["free_memory_mb"],
            compute_capability=device.get("architecture"),
            device_id=device["device_id"]
        ))
    
    # Add Metal GPUs
    for device in metal_info.devices:
        gpus.append(GPUInfo(
            acceleration_type=AcceleratorType.METAL,
            name=device["name"],
            vendor="Apple",
            total_memory_mb=device["total_memory_mb"],
            free_memory_mb=device["free_memory_mb"],
            device_id=device["device_id"]
        ))
    
    # Add Vulkan GPUs
    for device in vulkan_info.devices:
        gpus.append(GPUInfo(
            acceleration_type=AcceleratorType.VULKAN,
            name=device["name"],
            vendor=device["vendor"],
            total_memory_mb=device["total_memory_mb"],
            free_memory_mb=device["free_memory_mb"],
            device_id=device["device_id"]
        ))
    
    # Add OpenCL GPUs (only if not already added through another backend)
    opencl_device_names = set()
    for device in opencl_info.devices:
        # Skip CPU devices
        if device["device_type"].lower() == "cpu":
            continue
            
        if device["name"] in opencl_device_names:
            continue
        
        opencl_device_names.add(device["name"])
        
        gpus.append(GPUInfo(
            acceleration_type=AcceleratorType.OPENCL,
            name=device["name"],
            vendor=device["vendor"],
            total_memory_mb=device["total_memory_mb"],
            free_memory_mb=device["free_memory_mb"],
            device_id=device["device_id"]
        ))
    
    # Determine primary acceleration type
    primary_acceleration = AcceleratorType.CPU
    
    if cuda_info.available and cuda_info.device_count > 0:
        primary_acceleration = AcceleratorType.CUDA
    elif rocm_info.available and rocm_info.device_count > 0:
        primary_acceleration = AcceleratorType.ROCM
    elif metal_info.available:
        primary_acceleration = AcceleratorType.METAL
    elif vulkan_info.available and vulkan_info.device_count > 0:
        primary_acceleration = AcceleratorType.VULKAN
    elif opencl_info.available and opencl_info.device_count > 0:
        primary_acceleration = AcceleratorType.OPENCL
    elif cpu_info.supports_kleidiai:
        primary_acceleration = AcceleratorType.KLEIDIAI
    
    return HardwareCapabilities(
        cpu=cpu_info,
        memory=memory_info,
        primary_acceleration=primary_acceleration,
        gpus=gpus,
        cuda=cuda_info,
        rocm=rocm_info,
        metal=metal_info,
        vulkan=vulkan_info,
        opencl=opencl_info
    )


def get_optimal_llama_cpp_config(hw: HardwareCapabilities) -> LlamaCppConfig:
    """
    Determine optimal configuration for llama.cpp based on hardware.
    
    Args:
        hw: HardwareCapabilities object containing hardware information
        
    Returns:
        LlamaCppConfig: Optimal configuration for llama.cpp
    """
    config = LlamaCppConfig(
        n_threads=max(1, min(hw.cpu.physical_cores - 1, 8)),
        acceleration=hw.primary_acceleration
    )
    
    # Choose acceleration backend
    if hw.primary_acceleration == AcceleratorType.CUDA and hw.cuda.available:
        config.acceleration = AcceleratorType.CUDA
        
        if hw.cuda.device_count > 0:
            main_gpu = 0
            max_memory = 0
            
            for i, device in enumerate(hw.cuda.devices):
                if device["total_memory_mb"] > max_memory:
                    max_memory = device["total_memory_mb"]
                    main_gpu = i
                
            config.main_gpu = main_gpu
            free_memory_gb = hw.cuda.devices[main_gpu]["free_memory_mb"] / 1024
            
            if free_memory_gb > 16:
                config.n_gpu_layers = -1
            elif free_memory_gb > 8:
                config.n_gpu_layers = 35
            elif free_memory_gb > 4:
                config.n_gpu_layers = 20
            else:
                config.n_gpu_layers = 10
            
            if hw.cuda.device_count > 1:
                tensor_split = []
                total_memory = sum(d["free_memory_mb"] for d in hw.cuda.devices)
                for d in hw.cuda.devices:
                    split_ratio = d["free_memory_mb"] / total_memory
                    tensor_split.append(split_ratio)
                config.tensor_split = tensor_split
            
            config.cmake_args = ["-DGGML_CUDA=ON"]
            config.env_vars = {
                "CUDA_VISIBLE_DEVICES": ",".join(str(i) for i in range(hw.cuda.device_count)),
                "GGML_CUDA_ENABLE_UNIFIED_MEMORY": "1"
            }
    
    elif hw.primary_acceleration == AcceleratorType.ROCM and hw.rocm.available:
        config.acceleration = AcceleratorType.ROCM
        
        if hw.rocm.device_count > 0:
            config.n_gpu_layers = -1
            config.cmake_args = ["-DGGML_HIP=ON"]
            
            if hw.rocm.architectures:
                config.cmake_args.append(
                    f"-DAMDGPU_TARGETS={';'.join(hw.rocm.architectures)}"
                )
            if hw.rocm.rocm_path:
                config.env_vars["HIP_PATH"] = str(hw.rocm.rocm_path)
            
            config.env_vars["HIP_VISIBLE_DEVICES"] = ",".join(
                str(i) for i in range(hw.rocm.device_count)
            )
    
    elif hw.primary_acceleration == AcceleratorType.METAL and hw.metal.available:
        config.acceleration = AcceleratorType.METAL
        config.n_gpu_layers = -1
        config.cmake_args = ["-DGGML_METAL=ON"]
    
    elif hw.primary_acceleration == AcceleratorType.VULKAN and hw.vulkan.available:
        config.acceleration = AcceleratorType.VULKAN
        config.n_gpu_layers = -1
        config.cmake_args = ["-DGGML_VULKAN=ON"]
    
    elif hw.primary_acceleration == AcceleratorType.OPENCL and hw.opencl.available:
        config.acceleration = AcceleratorType.OPENCL
        config.n_gpu_layers = -1
        config.cmake_args = ["-DGGML_OPENCL=ON"]
    
    elif hw.primary_acceleration == AcceleratorType.KLEIDIAI and hw.cpu.supports_kleidiai:
        config.acceleration = AcceleratorType.KLEIDIAI
        config.n_gpu_layers = 0
        config.cmake_args = ["-DGGML_CPU_KLEIDIAI=ON"]
        
        if hw.cpu.supports_arm_sve:
            config.env_vars["GGML_KLEIDIAI_SME"] = "1"
    
    else:
        config.acceleration = AcceleratorType.CPU
        config.n_gpu_layers = 0
    
    return config


def find_llama_cpp_executable(paths: DataPaths, acceleration: AcceleratorType) -> Optional[Path]:
    """
    Find existing llama.cpp executable in the specified paths.
    
    Args:
        paths: DataPaths object with directory structure
        acceleration: Type of acceleration to look for
        
    Returns:
        Path to executable if found, None otherwise
    """
    bin_dir = paths.bin
    
    if platform.system() == "Windows":
        extensions = [".exe"]
    else:
        extensions = [""]

    base_names = ["llama-cli", "llama-server", "main"]
    
    # Check for acceleration-specific binaries
    if acceleration != AcceleratorType.CPU:
        for base in base_names:
            for ext in extensions:
                exe_path = bin_dir / f"{base}-{acceleration.value}{ext}"
                if exe_path.exists():
                    return exe_path
    
    # Check for generic binaries
    for base in base_names:
        for ext in extensions:
            exe_path = bin_dir / f"{base}{ext}"
            if exe_path.exists():
                return exe_path
    
    # Check llama.cpp/build directories
    llama_cpp_dir = paths.base / "llama.cpp"
    if llama_cpp_dir.exists():
        build_dir = llama_cpp_dir / "build"
        if build_dir.exists():
            # Check bin subdirectory
            bin_subdir = build_dir / "bin"
            if bin_subdir.exists():
                for base in base_names:
                    for ext in extensions:
                        exe_path = bin_subdir / f"{base}{ext}"
                        if exe_path.exists():
                            return exe_path
            
            # Check Release subdirectory on Windows
            if platform.system() == "Windows":
                release_dir = build_dir / "Release"
                if release_dir.exists():
                    for base in base_names:
                        for ext in extensions:
                            exe_path = release_dir / f"{base}{ext}"
                            if exe_path.exists():
                                return exe_path
            
            # Check root of build dir
            for base in base_names:
                for ext in extensions:
                    exe_path = build_dir / f"{base}{ext}"
                    if exe_path.exists():
                        return exe_path
    
    return None

def get_optimal_model_config(hw: HardwareCapabilities, model_size_gb: float) -> Dict[str, Any]:
    """
    Get optimal configuration for loading and running a model.
    
    Args:
        hw: HardwareCapabilities object with hardware information
        model_size_gb: Size of the model in GB
        
    Returns:
        Dict with optimal configuration parameters
    """
    config = {
        "n_threads": max(1, min(hw.cpu.physical_cores, 8)),
        "n_gpu_layers": 0,
        "use_mlock": True,
        "context_size": 4096,
        "batch_size": 512,
        "use_mmap": True,
        "use_gpu": False
    }
    
    available_memory_gb = hw.memory.available_gb
    
    # If the model is large relative to RAM, try GPU offload if available
    if model_size_gb > available_memory_gb * 0.7:
        if hw.cuda.available and hw.cuda.device_count > 0:
            config["use_gpu"] = True
            largest_gpu = max(hw.cuda.devices, key=lambda d: d["free_memory_mb"])
            free_gpu_gb = largest_gpu["free_memory_mb"] / 1024
            
            if free_gpu_gb > model_size_gb * 0.9:
                config["n_gpu_layers"] = -1
            else:
                ratio = free_gpu_gb / model_size_gb
                config["n_gpu_layers"] = max(1, int(ratio * 64))
        
        elif hw.rocm.available and hw.rocm.device_count > 0:
            config["use_gpu"] = True
            config["n_gpu_layers"] = -1
        
        elif hw.metal.available:
            config["use_gpu"] = True
            config["n_gpu_layers"] = -1
        
        elif hw.vulkan.available and hw.vulkan.device_count > 0:
            config["use_gpu"] = True
            config["n_gpu_layers"] = -1
    
    # Adjust context size for memory
    if available_memory_gb < 8:
        config["context_size"] = 2048
    elif available_memory_gb >= 16:
        config["context_size"] = 8192
    
    # Adjust batch size
    if config["use_gpu"]:
        config["batch_size"] = min(1024, config["context_size"] // 4)
    else:
        config["batch_size"] = min(512, config["context_size"] // 8)
    
    config["use_mmap"] = (model_size_gb > 2)
    config["use_mlock"] = (available_memory_gb > model_size_gb * 1.2)
    
    return config
