
"""
LLM: Lightweight Language Model Library

This package provides tools for:
- Data preprocessing and dataset handling
- Tokenization (BPE & SentencePiece)
- Model definition and training
- Inference
- Utilities for experiments and evaluation
"""

# Version
__version__ = "0.1.0"

# Dataset
from .data.dataloader import DataLoader
from .data.dataset import Dataset

# Tokenizers
from .tokenizer.bpe import BPE
from .tokenizer.sentencepiece import SentencePieceTokenizer

# Model
from .model import Model

# Training
from .training import Trainer

# Inference
from .inference import infer_text

# Utilities
from .utils import (
    save_model,
    load_model,
    set_seed,
    get_logger
)

# Optional: helper function to list available datasets
def list_datasets():
    """Returns available raw datasets."""
    import os
    raw_dir = os.path.join(os.path.dirname(__file__), "data", "raw")
    return [f for f in os.listdir(raw_dir) if f.endswith(".txt")]
