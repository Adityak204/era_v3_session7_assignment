import pytest
import torch
import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import MiniCNN_3


def test_parameter_count():
    """Test if model has less than 8k parameters"""
    model = MiniCNN_3()
    param_count = sum(p.numel() for p in model.parameters())
    assert param_count < 8000, f"Model has {param_count} parameters, which exceeds 8000"
