"""
This script is used to generate text from a trained language model.

Usage example:
    python generate.py --model_path ./model/checkpoints/tinystories_best.pt --prompt "Once upon a time" --temperature 0.8 --top_p 0.9 --max_tokens 100
"""

import os
import argparse
import json
from pathlib import Path

import torch

from cs336_basics.tokenizer import BPETokenizer as Tokenizer
from cs336_basics.transformer import TransformerLM
from cs336_basics.decoding import generate_text


def load_model_and_tokenizer(model_path, vocab_path=None, merges_path=None):
    """
    Load model and tokenizer
    
    Args:
        model_path: Model checkpoint path
        vocab_path: Vocabulary path (optional)
        merges_path: Merges rules path (optional)
        
    Returns:
        model: Loaded model
        tokenizer: Loaded tokenizer
        config: Model configuration
    """
    # Load model checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Try to find configuration file
    model_dir = Path(model_path).parent.parent
    config_path = model_dir / "config.json"
    config = {}
    
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        print(f"Warning: Configuration file {config_path} not found, default parameters will be used")
    
    # Extract model hyperparameters, use defaults if not present
    vocab_size = config.get("vocab_size", 50257)  # GPT-2 default vocabulary size
    context_length = config.get("context_length", 1024)
    d_model = config.get("d_model", 512)
    num_heads = config.get("num_heads", 8)
    num_layers = config.get("num_layers", 6)
    d_ff = config.get("d_ff", 2048)
    dropout = config.get("dropout", 0.1)
    
    # Initialize model
    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        dropout=dropout,
        device="cpu",  # Use CPU initially
        dtype=torch.float32
    )
    
    # Load model weights
    model.load_state_dict(checkpoint['model'])
    model.eval()
    
    # Find tokenizer files
    if vocab_path is None or merges_path is None:
        # Search in model directory
        potential_vocab_paths = list(model_dir.glob("*vocab*.json")) + list(model_dir.glob("*.vocab"))
        potential_merges_paths = list(model_dir.glob("*merges*.txt")) + list(model_dir.glob("*.merges"))
        
        # If not found, try data directory
        data_dir = model_dir.parent / "data"
        if data_dir.exists():
            potential_vocab_paths += list(data_dir.glob("*vocab*.json")) + list(data_dir.glob("*.vocab"))
            potential_merges_paths += list(data_dir.glob("*merges*.txt")) + list(data_dir.glob("*.merges"))
        
        # If only one is found, use it
        if len(potential_vocab_paths) == 1 and vocab_path is None:
            vocab_path = str(potential_vocab_paths[0])
            print(f"Found vocabulary file: {vocab_path}")
        
        if len(potential_merges_paths) == 1 and merges_path is None:
            merges_path = str(potential_merges_paths[0])
            print(f"Found merges file: {merges_path}")
    
    # If tokenizer files not found, exit
    if vocab_path is None or merges_path is None:
        raise ValueError(
            "Tokenizer files not found. Please specify --vocab_path and --merges_path parameters, "
            "or place these files in the model or data directory."
        )
    
    # Load tokenizer
    tokenizer = Tokenizer.from_file(vocab_path=vocab_path, merges_path=merges_path)
    
    return model, tokenizer, config


def main():
    parser = argparse.ArgumentParser(description="Generate text from a trained language model")
    parser.add_argument("--model_path", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--vocab_path", type=str, help="Vocabulary path")
    parser.add_argument("--merges_path", type=str, help="Merges rules path")
    parser.add_argument("--prompt", type=str, default="", help="Prompt for text generation")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=1.0, help="Nucleus sampling threshold")
    parser.add_argument("--max_tokens", type=int, default=100, help="Maximum number of tokens to generate")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use (cpu, mps, cuda, etc.)")
    parser.add_argument("--seed", type=int, help="Random seed for reproducible generation")
    
    args = parser.parse_args()
    
    # Set random seed
    if args.seed is not None:
        torch.manual_seed(args.seed)
    
    # Load model and tokenizer
    model, tokenizer, _ = load_model_and_tokenizer(args.model_path, args.vocab_path, args.merges_path)
    
    # Move model to specified device
    device = args.device
    if device == "auto":
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    
    model.to(device)
    
    # Generate text
    print("\n" + "="*50)
    print(f"Generating text with the following parameters:")
    print(f"  - Prompt: '{args.prompt}'")
    print(f"  - Temperature: {args.temperature}")
    print(f"  - Top-p: {args.top_p}")
    print(f"  - Maximum tokens: {args.max_tokens}")
    print(f"  - Device: {device}")
    print("="*50 + "\n")
    
    print("Generating...\n")
    
    # Use verbose mode to display generation process in real-time
    generated_text = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt_text=args.prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        verbose=True,
        device=device
    )
    
    print("\n" + "="*50)
    print("Generation complete!")
    print("="*50)


if __name__ == "__main__":
    main()