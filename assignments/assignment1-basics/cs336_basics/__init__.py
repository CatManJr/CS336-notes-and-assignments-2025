import importlib.metadata

__version__ = importlib.metadata.version("cs336_basics")

from .data import get_batch
from .transformer import MultiHeadSelfAttention, TransformerBlock, TransformerLM
from .nn_utils import Linear, Embedding, RMSNorm, SwiGLU, RotaryPositionalEmbedding, softmax, scaled_dot_product_attention
from .tokenizer import BPETokenizer
from .train_lm import train
from .optim import AdamW, gradient_clipping, get_lr_cosine_schedule
from .losses import cross_entropy
from .experiment import ExperimentTracker
from .generate import generate_text
from .decoding import sample

__all__ = [
    'get_batch',
    'MultiHeadSelfAttention',
    'TransformerBlock',
    'TransformerLM',
    'Linear',
    'Embedding',
    'RMSNorm',
    'SwiGLU',
    'RotaryPositionalEmbedding',
    'softmax',
    'scaled_dot_product_attention',
    'BPETokenizer',
    'train',
    'AdamW',
    'gradient_clipping',
    'get_lr_cosine_schedule',
    'cross_entropy',
    'ExperimentTracker',
    'generate_text',
    'sample',
]
