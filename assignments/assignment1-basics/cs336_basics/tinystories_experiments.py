"""
Script to run TinyStories language model training experiments.
This script implements the experiments required in the assignment:
1. Learning rate tuning
2. Batch size variation
"""
import os
import time
import argparse
import logging
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import subprocess, sys, pathlib

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from cs336_basics.train_lm import train
from cs336_basics.tokenizer import BPETokenizer as Tokenizer
from cs336_basics.experiment import ExperimentTracker
from cs336_basics.generate import generate_text
from cs336_basics.decoding import sample

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

def run_learning_rate_experiment(
    train_dataset,
    val_dataset,
    vocab_size,
    checkpoint_dir,
    device,
    use_wandb=False,
):
    """
    Run learning rate hyperparameter sweep experiment.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        vocab_size: Vocabulary size
        checkpoint_dir: Directory to save checkpoints
        device: Device to use for training
        use_wandb: Whether to use W&B for logging
    """
    # Common hyperparameters based on assignment specifications
    common_params = {
        'train_dataset': train_dataset,
        'val_dataset': val_dataset,
        'vocab_size': vocab_size,
        'context_length': 256,
        'd_model': 512,
        'num_heads': 16,
        'num_layers': 4,
        'd_ff': 1344,
        'dropout': 0.1,
        'device': device,
        'batch_size': 64,  # Can be tuned based on GPU memory
        'eval_interval': 100,
        'eval_iters': 10,
        'log_interval': 10,
        'beta1': 0.9,
        'beta2': 0.999,
        'weight_decay': 0.01,
        'grad_clip': 1.0,
        'checkpoint_interval': 500,
        'use_wandb': use_wandb,
        'wandb_project': 'tinystories-lr-experiment',
    }
    
    # Calculate total number of tokens to process based on assignment spec
    if device == 'cpu' or device == 'mps':
        # Low-resource configuration: 40M tokens
        total_tokens = 40_000_000
        target_val_loss = 2.0
    else:
        # H100 configuration: 327.68M tokens
        total_tokens = 327_680_000
        target_val_loss = 1.45
    
    # Learning rates to try (log scale)
    learning_rates = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3]
    
    # Calculate max_iters based on total tokens and batch_size * context_length
    tokens_per_iter = common_params['batch_size'] * common_params['context_length']
    max_iters = total_tokens // tokens_per_iter
    
    logger.info(f"Running learning rate experiment with max_iters={max_iters} to process {total_tokens} tokens")
    
    # Store results for analysis
    results = []
    
    # Run experiments with different learning rates
    for lr in learning_rates:
        logger.info(f"Training with learning rate: {lr}")
        
        # Create checkpoint directory for this run
        lr_checkpoint_dir = os.path.join(checkpoint_dir, f"lr_{lr}")
        os.makedirs(lr_checkpoint_dir, exist_ok=True)
        
        # Set up experiment parameters
        params = common_params.copy()
        params.update({
            'learning_rate': lr,
            'min_learning_rate': lr / 10,  # Min LR is 1/10 of max LR
            'warmup_iters': int(0.1 * max_iters),  # 10% warmup
            'max_iters': max_iters,
            'checkpoint_dir': lr_checkpoint_dir,
            'checkpoint_prefix': f'tinystories_lr_{lr}',
            'wandb_run_name': f'lr_{lr}',
        })
        
        # Initialize experiment tracker for this LR run
        tracker = ExperimentTracker(
            experiment_name=f"lr_{lr}",
            base_dir=checkpoint_dir,
            use_wandb=use_wandb,
            config=params,
        )
        try:
            # Train via subprocess
            script = pathlib.Path(__file__).parent / "train_tiny_stories.py"
            cmd = [
                sys.executable, str(script),
                "--data_dir", args.data_dir,
                "--output_dir", lr_checkpoint_dir,
                "--batch_size", str(params['batch_size']),
                "--max_iters", str(params['max_iters']),
                "--learning_rate", str(lr),
                "--device", params['device'],
            ]
            if params.get('use_wandb'):
                cmd.append("--use_wandb")
            subprocess.run(cmd, check=True)
            
            # Evaluate final validation loss
            model = train(**params)
            with torch.no_grad():
                val_loss = model.estimate_loss(
                    val_dataset, 
                    common_params['batch_size'], 
                    common_params['context_length'], 
                    device, 
                    common_params['eval_iters']
                )
            # Log final val_loss
            tracker.log({'val_loss': val_loss}, step=params['max_iters'])
            
            results.append({
                'learning_rate': lr,
                'val_loss': val_loss,
                'converged': True,
            })
            
            logger.info(f"Training with lr={lr} completed. Final val loss: {val_loss:.4f}")
            
            # Save artifact if target met
            if val_loss <= target_val_loss:
                logger.info(f"Model achieved target validation loss of {target_val_loss}")
                model_path = os.path.join(lr_checkpoint_dir, f'final_model_lr_{lr}.pt')
                torch.save(model.state_dict(), model_path)
                tracker.save_artifact(model_path, name=f"model_lr_{lr}")
        except Exception as e:
            logger.error(f"Training diverged with lr={lr}: {str(e)}")
            results.append({
                'learning_rate': lr,
                'val_loss': float('inf'),
                'converged': False,
            })
        finally:
            tracker.finish()
    
    # Save experiment results
    with open(os.path.join(checkpoint_dir, 'lr_experiment_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Plot learning rate vs. validation loss
    plt.figure(figsize=(10, 6))
    
    # Extract converged runs
    converged_runs = [(r['learning_rate'], r['val_loss']) for r in results if r['converged']]
    if converged_runs:
        converged_lr, converged_loss = zip(*converged_runs)
        plt.plot(converged_lr, converged_loss, 'o-', label='Converged runs')
    
    # Extract diverged runs
    diverged_runs = [(r['learning_rate'], 10.0) for r in results if not r['converged']]
    if diverged_runs:
        diverged_lr, _ = zip(*diverged_runs)
        plt.scatter(diverged_lr, [10.0] * len(diverged_lr), marker='x', color='red', label='Diverged runs')
    
    plt.xscale('log')
    plt.xlabel('Learning Rate')
    plt.ylabel('Validation Loss')
    plt.title('Learning Rate vs. Validation Loss')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(checkpoint_dir, 'lr_experiment_results.png'))
    
    return results


def run_batch_size_experiment(
    train_dataset,
    val_dataset,
    vocab_size,
    checkpoint_dir,
    device,
    use_wandb=False,
    best_lr=None,
):
    """
    Run batch size variation experiment.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        vocab_size: Vocabulary size
        checkpoint_dir: Directory to save checkpoints
        device: Device to use for training
        use_wandb: Whether to use W&B for logging
        best_lr: Best learning rate from the previous experiment
    """
    # Common hyperparameters based on assignment specifications
    common_params = {
        'train_dataset': train_dataset,
        'val_dataset': val_dataset,
        'vocab_size': vocab_size,
        'context_length': 256,
        'd_model': 512,
        'num_heads': 16,
        'num_layers': 4,
        'd_ff': 1344,
        'dropout': 0.1,
        'device': device,
        'eval_interval': 100,
        'eval_iters': 10,
        'log_interval': 10,
        'beta1': 0.9,
        'beta2': 0.999,
        'weight_decay': 0.01,
        'grad_clip': 1.0,
        'checkpoint_interval': 500,
        'use_wandb': use_wandb,
        'wandb_project': 'tinystories-batch-experiment',
        'learning_rate': best_lr if best_lr else 3e-4,  # Default to 3e-4 if best_lr not provided
    }
    
    # Calculate total number of tokens to process based on assignment spec
    if device == 'cpu' or device == 'mps':
        # Low-resource configuration: 40M tokens
        total_tokens = 40_000_000
    else:
        # H100 configuration: 327.68M tokens
        total_tokens = 327_680_000
    
    # Batch sizes to try (vary from small to memory limit)
    if device == 'cpu':
        batch_sizes = [1, 4, 8, 16, 32, 64]
    elif device == 'mps':
        batch_sizes = [1, 8, 16, 32, 64, 128]
    else:
        # For H100, try larger batch sizes
        batch_sizes = [1, 32, 64, 128, 256, 512]
    
    # Store results for analysis
    results = []
    
    for batch_size in batch_sizes:
        logger.info(f"Training with batch size: {batch_size}")
        
        # Create checkpoint directory for this run
        bs_checkpoint_dir = os.path.join(checkpoint_dir, f"bs_{batch_size}")
        os.makedirs(bs_checkpoint_dir, exist_ok=True)
        
        # Calculate max_iters based on total tokens and batch_size * context_length
        tokens_per_iter = batch_size * common_params['context_length']
        max_iters = total_tokens // tokens_per_iter
        
        # Set up experiment parameters
        params = common_params.copy()
        params.update({
            'batch_size': batch_size,
            'max_iters': max_iters,
            'min_learning_rate': params['learning_rate'] / 10,  # Min LR is 1/10 of max LR
            'warmup_iters': int(0.1 * max_iters),  # 10% warmup
            'checkpoint_dir': bs_checkpoint_dir,
            'checkpoint_prefix': f'tinystories_bs_{batch_size}',
            'wandb_run_name': f'bs_{batch_size}',
        })
        
        # Initialize experiment tracker for this batch size run
        tracker = ExperimentTracker(
            experiment_name=f"bs_{batch_size}",
            base_dir=checkpoint_dir,
            use_wandb=use_wandb,
            config=params,
        )
        
        logger.info(f"Running batch size={batch_size} experiment with max_iters={max_iters} to process {total_tokens} tokens")
        
        # Train the model
        try:
            start_time = time.time()
            model = train(**params)
            total_time = time.time() - start_time
            
            # Evaluate final validation loss
            with torch.no_grad():
                val_loss = model.estimate_loss(
                    val_dataset, 
                    batch_size, 
                    common_params['context_length'], 
                    device, 
                    common_params['eval_iters']
                )
            # Log metrics for this run
            tracker.log({
                'val_loss': val_loss,
                'training_time': total_time,
                'tokens_per_second': total_tokens / total_time
            }, step=max_iters)
        
            results.append({
                'batch_size': batch_size,
                'val_loss': val_loss,
                'training_time': total_time,
                'tokens_per_second': total_tokens / total_time,
                'converged': True,
            })
            
            logger.info(f"Training with batch_size={batch_size} completed in {total_time:.2f}s. Final val loss: {val_loss:.4f}")
            logger.info(f"Processing speed: {total_tokens / total_time:.2f} tokens/sec")
        
        except Exception as e:
            logger.error(f"Training failed with batch_size={batch_size}: {str(e)}")
            results.append({
                'batch_size': batch_size,
                'val_loss': float('inf'),
                'training_time': float('inf'),
                'tokens_per_second': 0,
                'converged': False,
            })
        finally:
            tracker.finish()
    
    # Save experiment results
    with open(os.path.join(checkpoint_dir, 'batch_size_experiment_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Plot batch size vs. validation loss
    plt.figure(figsize=(10, 6))
    
    # Extract converged runs
    converged_runs = [(r['batch_size'], r['val_loss']) for r in results if r['converged']]
    if converged_runs:
        converged_bs, converged_loss = zip(*converged_runs)
        plt.plot(converged_bs, converged_loss, 'o-', label='Validation Loss')
    
    plt.xscale('log')
    plt.xlabel('Batch Size')
    plt.ylabel('Validation Loss')
    plt.title('Batch Size vs. Validation Loss')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(checkpoint_dir, 'bs_loss_results.png'))
    
    # Plot batch size vs. tokens per second (throughput)
    plt.figure(figsize=(10, 6))
    
    # Extract converged runs
    converged_runs = [(r['batch_size'], r['tokens_per_second']) for r in results if r['converged']]
    if converged_runs:
        converged_bs, converged_tps = zip(*converged_runs)
        plt.plot(converged_bs, converged_tps, 'o-', label='Tokens per second')
    
    plt.xscale('log')
    plt.xlabel('Batch Size')
    plt.ylabel('Tokens per second')
    plt.title('Batch Size vs. Training Throughput')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(checkpoint_dir, 'bs_throughput_results.png'))
    
    return results


def generate_from_model(
    model_path,
    tokenizer_path,
    context_length=256,
    device='cpu',
    temperature=0.8,
    top_p=0.9,
):
    """
    Generate text from a trained model.
    
    Args:
        model_path: Path to model checkpoint
        tokenizer_path: Path to tokenizer files (directory with vocab.json and merges.txt)
        context_length: Model context length
        device: Device to use for generation
        temperature: Sampling temperature
        top_p: Nucleus sampling threshold
    """
    # Load tokenizer
    tokenizer = Tokenizer.from_file(
        vocab_path=os.path.join(tokenizer_path, "vocab.json"),
        merges_path=os.path.join(tokenizer_path, "merges.txt")
    )
    
    # Load model
    model_state = torch.load(model_path, map_location=device)
    
    # Get model parameters from filename or state dict
    if hasattr(model_state, 'get') and model_state.get('config'):
        config = model_state['config']
        vocab_size = config.get('vocab_size', tokenizer.vocab_size)
        d_model = config.get('d_model', 512)
        num_heads = config.get('num_heads', 16)
        num_layers = config.get('num_layers', 4)
        d_ff = config.get('d_ff', 1344)
    else:
        # Use default parameters
        vocab_size = tokenizer.vocab_size
        d_model = 512
        num_heads = 16
        num_layers = 4
        d_ff = 1344
    
    # Create model instance
    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        dropout=0.0,  # No dropout during generation
        device=device,
    )
    
    # Load model weights
    if hasattr(model_state, 'get') and model_state.get('model'):
        model.load_state_dict(model_state['model'])
    else:
        model.load_state_dict(model_state)
    
    model.eval()
    
    # Generate text
    prompt = "Once upon a time"
    tokens = tokenizer.encode(prompt, bos=False, eos=False)
    
    # Generate at least 256 tokens or until end of text
    max_new_tokens = 300  # Generate a bit more than needed
    generated = generate_text(
        model=model,
        prompt_tokens=tokens,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        sample_fn=sample,
        eos_token_id=tokenizer.special_tokens.get("<|endoftext|>"),
    )
    
    # Decode generated tokens
    output_text = tokenizer.decode(generated)
    
    # If end of text token was generated, truncate there
    if "<|endoftext|>" in output_text:
        output_text = output_text.split("<|endoftext|>")[0]
    
    return output_text


def main():
    parser = argparse.ArgumentParser(description='Run TinyStories LM training experiments')
    
    parser.add_argument('--experiment', choices=['learning_rate', 'batch_size', 'generate'], required=True,
                       help='Which experiment to run')
    
    parser.add_argument('--train-data', default='../data/TinyStoriesV2-GPT4-train.txt',
                       help='Path to TinyStories training data')
    parser.add_argument('--val-data', default='../data/TinyStoriesV2-GPT4-valid.txt',
                       help='Path to TinyStories validation data')
    
    parser.add_argument('--tokenizer-dir', default='./output',
                       help='Directory with tokenizer files (vocab.json, merges.txt)')
    parser.add_argument('--vocab-size', type=int, default=10000,
                       help='Vocabulary size for tokenizer')
    
    parser.add_argument('--output-dir', default='./experiments',
                       help='Directory to save experiment results')
    
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu',
                       help='Device to use for training')
    
    parser.add_argument('--use-wandb', action='store_true', 
                       help='Use Weights & Biases for logging')
    
    parser.add_argument('--best-lr', type=float, default=None,
                       help='Best learning rate from learning rate experiment')
    
    parser.add_argument('--model-path', 
                       help='Path to model checkpoint for generation')
    
    parser.add_argument('--temperature', type=float, default=0.8,
                       help='Sampling temperature for generation')
    
    parser.add_argument('--top-p', type=float, default=0.9,
                       help='Nucleus sampling threshold for generation')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Write args to file for reproducibility
    with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA is not available. Falling back to CPU.")
        args.device = 'cpu'
    elif args.device == 'mps' and not torch.backends.mps.is_available():
        logger.warning("MPS is not available. Falling back to CPU.")
        args.device = 'cpu'
    
    logger.info(f"Using device: {args.device}")
    
    # Configure optimization based on device
    if args.device == 'cuda':
        # Enable TF32 for better performance on A100/H100 GPUs
        torch.set_float32_matmul_precision('high')
    elif args.device == 'mps':
        # MPS doesn't support TF32 kernels
        pass
    
    if args.experiment == 'generate':
        # Generate text from a trained model
        if not args.model_path:
            raise ValueError("--model-path is required for text generation")
        
        logger.info(f"Generating text from model: {args.model_path}")
        output_text = generate_from_model(
            model_path=args.model_path,
            tokenizer_path=args.tokenizer_dir,
            device=args.device,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        
        logger.info(f"Generated text:\n{output_text}")
        
        # Save generated text to file
        output_path = os.path.join(args.output_dir, 'generated_text.txt')
        with open(output_path, 'w') as f:
            f.write(output_text)
        logger.info(f"Generated text saved to: {output_path}")
        
    else:
        # Load or train tokenizer
        vocab_path = os.path.join(args.tokenizer_dir, 'vocab.json')
        merges_path = os.path.join(args.tokenizer_dir, 'merges.txt')
        
        if not (os.path.exists(vocab_path) and os.path.exists(merges_path)):
            logger.info(f"Tokenizer files not found. Training BPE tokenizer with vocab size {args.vocab_size}...")
            from train_bpe_tinystories import main as train_tokenizer
            import sys
            
            # Create args for tokenizer training
            tokenizer_args = [
                '--input', args.train_data,
                '--vocab-size', str(args.vocab_size),
                '--output-dir', args.tokenizer_dir
            ]
            
            # Run tokenizer training
            sys.argv = ['train_bpe_tinystories.py'] + tokenizer_args
            train_tokenizer()
        else:
            logger.info(f"Using existing tokenizer from {args.tokenizer_dir}")
        
        # Load tokenizer
        tokenizer = Tokenizer.from_file(vocab_path, merges_path)
        logger.info(f"Loaded tokenizer with vocabulary size: {tokenizer.vocab_size}")
        
        # Load tokenized datasets directly via numpy memmap
        logger.info("Loading tokenized datasets via numpy.memmap...")
        train_dataset = np.memmap(args.train_data, dtype=np.int32, mode='r')
        val_dataset = np.memmap(args.val_data, dtype=np.int32, mode='r')
        logger.info(f"Train dataset size: {train_dataset.shape[0]}, Val dataset size: {val_dataset.shape[0]}")
        
        if args.experiment == 'learning_rate':
            # Create experiment directory
            experiment_dir = os.path.join(args.output_dir, 'learning_rate')
            os.makedirs(experiment_dir, exist_ok=True)
            
            logger.info("Running learning rate experiment...")
            results = run_learning_rate_experiment(
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                vocab_size=tokenizer.vocab_size,
                checkpoint_dir=experiment_dir,
                device=args.device,
                use_wandb=args.use_wandb,
            )
            
            # Find best learning rate
            converged_results = [r for r in results if r['converged']]
            if converged_results:
                best_result = min(converged_results, key=lambda x: x['val_loss'])
                best_lr = best_result['learning_rate']
                logger.info(f"Best learning rate: {best_lr} with validation loss: {best_result['val_loss']:.4f}")
            else:
                logger.warning("No converged runs found. Unable to determine best learning rate.")
        
        elif args.experiment == 'batch_size':
            # Create experiment directory
            experiment_dir = os.path.join(args.output_dir, 'batch_size')
            os.makedirs(experiment_dir, exist_ok=True)
            
            logger.info("Running batch size experiment...")
            results = run_batch_size_experiment(
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                vocab_size=tokenizer.vocab_size,
                checkpoint_dir=experiment_dir,
                device=args.device,
                use_wandb=args.use_wandb,
                best_lr=args.best_lr,
            )
            
            # Find optimal batch size
            converged_results = [r for r in results if r['converged']]
            if converged_results:
                # Find batch size with best throughput
                best_throughput = max(converged_results, key=lambda x: x['tokens_per_second'])
                logger.info(f"Best throughput with batch size: {best_throughput['batch_size']} "
                           f"({best_throughput['tokens_per_second']:.2f} tokens/sec)")
                
                # Find batch size with best validation loss
                best_loss = min(converged_results, key=lambda x: x['val_loss'])
                logger.info(f"Best validation loss with batch size: {best_loss['batch_size']} "
                           f"(loss: {best_loss['val_loss']:.4f})")
            else:
                logger.warning("No converged runs found. Unable to determine optimal batch size.")

if __name__ == "__main__":
    main()