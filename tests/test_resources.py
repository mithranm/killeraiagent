"""
Tests for the resource management module.
"""

import pytest
from killeraiagent.resources import get_resource_manager


def test_resource_manager_singleton():
    """Test that get_resource_manager returns the same instance each time."""
    rm1 = get_resource_manager()
    rm2 = get_resource_manager()
    assert rm1 is rm2


def test_hardware_profile():
    """Test that hardware profile contains expected fields."""
    rm = get_resource_manager()
    hw = rm.hardware
    
    # Basic CPU info
    assert hw.physical_cores > 0
    assert hw.logical_cores > 0
    assert hw.total_memory_gb > 0
    
    # GPU info might not be available on all systems
    assert hasattr(hw, 'has_cuda')
    assert hasattr(hw, 'has_mps')
    
    # Test optimal worker count calculation
    workers = hw.get_optimal_worker_count(memory_per_worker_gb=0.5, cpu_intensive=True)
    assert workers > 0


def test_embedding_config():
    """Test that embedding config contains expected fields."""
    rm = get_resource_manager()
    config = rm.get_embedding_config()
    
    assert 'device' in config
    assert 'batch_size' in config
    assert config['batch_size'] > 0


def test_llm_config():
    """Test that LLM config contains expected fields."""
    rm = get_resource_manager()
    config = rm.get_llm_config()
    
    assert 'n_threads' in config
    assert 'n_gpu_layers' in config
    assert config['n_threads'] > 0