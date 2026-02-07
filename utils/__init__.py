"""
LLM Package
===========

This package provides tools for training, tokenizing, and running inference
on language models. It includes modules for data loading, tokenization,
model definition, training utilities, and experiments.
"""

# Expose key submodules at the package level
from . import data
from . import tokenizer
from . import model
from . import training
from . import utils
from . import inference

__all__ = [
    "data",
    "tokenizer",
    "model",
    "training",
    "utils",
    "inference",
]

# Optional: version info
__version__ = "0.1.0"
