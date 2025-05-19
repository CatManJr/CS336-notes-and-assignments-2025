import os
import time
import math
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, Union
from collections.abc import Iterable

import numpy as np
import torch
from torch.nn import Module

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    from rich.console import Console
    from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn
    from rich.table import Table
    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

from cs336_basics.optim import AdamW, gradient_clipping, get_lr_cosine_schedule
from cs336_basics.transformer import TransformerLM
from cs336_basics.losses import cross_entropy
from cs336_basics.data import get_batch
from cs336_basics.tokenizer import BPETokenizer

# setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

def save_checkpoint(
    model: Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    loss: float,
    output_path: Union[str, os.PathLike],
    extra_data: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save a model checkpoint.

    Args:
        model: The model to save.
        optimizer: The optimizer state.
        iteration: Current iteration number.
        loss: Current loss value.
        output_path: Path to save the checkpoint.
        extra_data: Any additional data to save.
    """
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'iteration': iteration,
        'loss': loss,
    }
    
    if extra_data:
        checkpoint.update(extra_data)
    
    # transform string paths to Path objects
    if isinstance(output_path, str):
        output_path = Path(output_path)
    
    # ensure the directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # save to a temporary file first, then rename to prevent corruption
    temp_path = output_path.with_suffix('.tmp')
    torch.save(checkpoint, temp_path)
    temp_path.rename(output_path)
    
    logger.info(f"Saved checkpoint at iteration {iteration} to {output_path}")


def load_checkpoint(
    model: Module,
    optimizer: torch.optim.Optimizer,
    checkpoint_path: Union[str, os.PathLike],
) -> Tuple[int, float, Dict[str, Any]]:
    """
    Load a model checkpoint.
    
    Args:
        model: The model to restore.
        optimizer: The optimizer to restore.
        checkpoint_path: Checkpoint path
        
    Returns:
        (iteration, loss value, dictionary of extra data)
    """
    # If it's a file path string, convert to Path object
    if isinstance(checkpoint_path, str):
        checkpoint_path = Path(checkpoint_path)
        
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    
    # Load checkpoint
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Restore model and optimizer state
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    
    # Extract iteration and loss value
    iteration = checkpoint.get('iteration', 0)
    loss = checkpoint.get('loss', float('inf'))
    
    # Remove processed keys, remaining are extra data
    extra_data = {k: v for k, v in checkpoint.items() 
                 if k not in ('model', 'optimizer', 'iteration', 'loss')}
    
    logger.info(f"Restored checkpoint from iteration {iteration}")
    return iteration, loss, extra_data


def estimate_loss(
    model: Module,
    dataset: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str,
    eval_iters: int = 10,
) -> float:
    """
    Evaluate the average loss of the model on the given dataset.
    
    Args:
        model: The model to evaluate
        dataset: Evaluation dataset
        batch_size: Batch size
        context_length: Context length
        device: Device string
        eval_iters: Number of evaluation iterations
        
    Returns:
        Average loss over the evaluation iterations
    """
    model.eval()  
    losses = []
    
    with torch.no_grad():  
        for _ in range(eval_iters):
            # 获取数据
            x_batch, y_batch = get_batch(dataset, batch_size, context_length, device)
            
            # 确保数据和模型在同一设备上
            model_device = next(model.parameters()).device
            x_batch = x_batch.to(model_device)
            y_batch = y_batch.to(model_device)
            
            # 前向传播
            logits = model(x_batch)
            
            # Calculate loss
            # Reshape to match the expected input of cross-entropy loss
            B, T, C = logits.shape  # batch, time, channels
            logits_flat = logits.view(B*T, C)
            targets_flat = y_batch.reshape(-1)
            
            loss = cross_entropy(logits_flat, targets_flat)
            losses.append(loss.item())
    
    model.train()  
    return sum(losses) / len(losses)


def train(
    # Todo: implement Resume from checkpoint
    # dataset config
    train_dataset: np.ndarray,
    val_dataset: Optional[np.ndarray] = None,
    # model config
    vocab_size: int = 50257,
    context_length: int = 1024,
    d_model: int = 512,
    num_heads: int = 8,
    num_layers: int = 6,
    d_ff: int = 2048,
    # training config
    device: str = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu',
    batch_size: int = 16,
    max_iters: int = 10000,
    eval_interval: int = 100,
    eval_iters: int = 10,
    log_interval: int = 10,
    # optimizer params
    learning_rate: float = 1e-4,
    min_learning_rate: float = 1e-5,
    warmup_iters: int = 100,
    weight_decay: float = 0.01,
    beta1: float = 0.9,
    beta2: float = 0.999,
    grad_clip: float = 1.0,
    # checkpoint params
    checkpoint_dir: str = './checkpoints',
    checkpoint_interval: int = 500,
    checkpoint_prefix: str = 'model',
    resume_from: Optional[str] = None,
    # wandb params
    use_wandb: bool = False,
    wandb_project: str = "cs336-lm-training",
    wandb_run_name: Optional[str] = None,
    wandb_entity: Optional[str] = None,
    # visualization and sample generation
    visualize: bool = True,
    generate_samples: bool = True,
    sample_length: int = 100,
    sample_interval: int = 500,
    # tokenizer for text generation
    tokenizer: Optional[BPETokenizer] = None,
    # early stopping
    early_stopping_patience: Optional[int] = None,  # 新增参数
) -> Module:
    """
    Train a Transformer language model.

    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset (optional)
        vocab_size: Vocabulary size
        context_length: Context window size
        d_model: Model dimension
        num_heads: Number of attention heads
        num_layers: Number of Transformer layers
        d_ff: Feedforward layer dimension
        device: Training device ('cpu', 'mps', or 'cuda')
        batch_size: Batch size
        max_iters: Maximum training iterations
        eval_interval: Evaluate every N iterations
        eval_iters: Number of iterations for validation evaluation
        log_interval: Log every N iterations
        learning_rate: Max learning rate
        min_learning_rate: Min learning rate
        warmup_iters: Warmup iterations
        weight_decay: Weight decay
        beta1: Adam beta1
        beta2: Adam beta2
        grad_clip: Gradient clipping threshold
        checkpoint_dir: Directory to save checkpoints
        checkpoint_interval: Save checkpoint every N iterations
        checkpoint_prefix: Prefix for checkpoint files
        resume_from: Resume from checkpoint (optional)
        use_wandb: Use Weights & Biases logging
        wandb_project: W&B project name
        wandb_run_name: W&B run name (optional)
        wandb_entity: W&B entity (optional)
        visualize: Whether to use visual progress tracking (requires tqdm or rich)
        generate_samples: Whether to generate text samples during training
        sample_length: Length of generated samples
        sample_interval: Generate samples every N iterations
        tokenizer: Tokenizer for text generation
        early_stopping_patience: Number of evaluations without improvement before stopping (optional)

    Returns:
        Trained model
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(42)

    if use_wandb:
        if not WANDB_AVAILABLE:
            logger.warning("Weights & Biases (wandb) not installed. Logging to wandb will be disabled.")
            use_wandb = False
        else:
            config = {
                "vocab_size": vocab_size,
                "context_length": context_length,
                "d_model": d_model,
                "num_heads": num_heads,
                "num_layers": num_layers,
                "d_ff": d_ff,
                "device": device,
                "batch_size": batch_size,
                "max_iters": max_iters,
                "learning_rate": learning_rate,
                "min_learning_rate": min_learning_rate,
                "warmup_iters": warmup_iters,
                "weight_decay": weight_decay,
                "beta1": beta1,
                "beta2": beta2,
                "grad_clip": grad_clip,
                "dataset_size": len(train_dataset) if train_dataset is not None else None,
                "val_dataset_size": len(val_dataset) if val_dataset is not None else None,
                "resuming": resume_from is not None,
            }
            wandb.init(
                project=wandb_project,
                name=wandb_run_name,
                entity=wandb_entity,
                config=config,
                resume="allow" if resume_from else None,
            )
            with open(checkpoint_dir / "wandb_id.txt", "w") as f:
                f.write(wandb.run.id)

    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        theta=10000,
    ).to(device)  # 确保模型到指定的设备上

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model has {param_count / 1e6:.2f}M parameters")

    bytes_per_param = 4  # FP32
    model_size_gb = param_count * bytes_per_param / (1024**3)
    forward_backward_gb = model_size_gb * 6  # Forward and backward passes use approximately 6x parameter memory
    optimizer_size_gb = model_size_gb * 2  # Adam optimizer states (momentum and variance)
    batch_memory_gb = (batch_size * context_length * d_model * 4) / (1024**3)
    total_memory_gb = model_size_gb + forward_backward_gb + optimizer_size_gb + batch_memory_gb
        
    
    # Calculate FLOPS based on transformer operations
    # Attention operations: 4 * d_model^2 * seq_len + 2 * seq_len^2 * d_model
    # FFN operations: 2 * d_model * d_ff * seq_len
    flops_per_seq_per_layer = context_length * (4 * d_model**2 + 2 * context_length * d_model + 2 * d_model * d_ff)
    flops_forward = batch_size * num_layers * flops_per_seq_per_layer
    flops_training = 3 * flops_forward  # Forward + backward (roughly 3x)
    flops_training_tflops = flops_training / 1e12  # Convert to TFLOPS

    logger.info(f"Estimated memory usage:")
    logger.info(f"  - Model parameters: {model_size_gb:.2f} GB")
    logger.info(f"  - Forward/backward passes: {forward_backward_gb:.2f} GB")
    logger.info(f"  - Optimizer states: {optimizer_size_gb:.2f} GB")
    logger.info(f"  - Batch processing: {batch_memory_gb:.2f} GB")
    logger.info(f"  - Total estimated: {total_memory_gb:.2f} GB")
    logger.info(f"  - Training FLOPS per iteration: {flops_training_tflops:.4f} TFLOPS")

    if use_wandb:
        wandb.log({
            "model/parameters_count": param_count,
            "model/parameters_gb": model_size_gb,
            "memory/optimizer_gb": optimizer_size_gb,
            "memory/batch_gb": batch_memory_gb,
            "memory/total_gb": total_memory_gb
        }, step=0)

    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(beta1, beta2)
    )

    start_iter = 0
    best_val_loss = float('inf')
    patience_counter = 0  # 新增变量
    if resume_from:
        start_iter, val_loss, extra_data = load_checkpoint(
            model=model,
            optimizer=optimizer,
            checkpoint_path=resume_from
        )
        best_val_loss = val_loss
        logger.info(f"Resuming from iteration {start_iter} with validation loss {val_loss:.4f}")

    # Setup visualization components
    if visualize:
        if RICH_AVAILABLE:
            progress_display = Progress(
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                console=console
            )
            training_task = progress_display.add_task("[green]Training", total=max_iters - start_iter)
        elif TQDM_AVAILABLE:
            progress_bar = tqdm(total=max_iters - start_iter, desc="Training", 
                              initial=0, position=0, leave=True)
        else:
            logger.warning("Neither rich nor tqdm is installed. Basic progress reporting will be used.")
            visualize = False

    # Create a function for generating text samples
    def generate_text_sample(
        model: Module,
        tokenizer: Optional[BPETokenizer] = None,
        starter_tokens: Optional[list[int]] = None,
        max_length: int = 100,
        temp: float = 0.8,
        top_k: Optional[int] = None,
        device: str = 'cpu'
    ) -> Tuple[torch.Tensor, Optional[str]]:
        model.eval()
        with torch.no_grad():
            # 获取模型实际 device
            model_device = next(model.parameters()).device
            # context 初始化在模型 device 上
            if starter_tokens is None:
                context = torch.tensor([0], dtype=torch.long, device=model_device)
            else:
                context = torch.tensor(starter_tokens, dtype=torch.long, device=model_device)

            for _ in range(max_length):
                logits = model(context)
                if logits.dim() == 3:
                    logits = logits[0, -1, :]
                elif logits.dim() == 2:
                    logits = logits[-1, :]
                if temp > 0:
                    logits = logits / temp
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[-1]] = -float('Inf')
                probs = torch.nn.functional.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                # 保证 next_token 也在 model_device 上
                next_token = next_token.to(model_device)
                context = torch.cat([context, next_token], dim=0)

        tokens = context.unsqueeze(0)
        decoded_text = tokenizer.decode(tokens[0].tolist()) if tokenizer else None
        return tokens, decoded_text

    # Training stats tracking
    training_stats = {
        "train_loss": [],
        "val_loss": [],
        "learning_rate": [],
        "iterations": []
    }

    # Start the progress tracker if using rich
    if visualize and RICH_AVAILABLE:
        progress_display.start()

    model.train()
    start_time = time.time()

    try:
        for it in range(start_iter, max_iters):
            lr = get_lr_cosine_schedule(
                it=it,
                max_learning_rate=learning_rate,
                min_learning_rate=min_learning_rate,
                warmup_iters=warmup_iters,
                cosine_cycle_iters=max_iters
            )
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            x_batch, y_batch = get_batch(train_dataset, batch_size, context_length, device)
            
            # 二次确认数据和模型在同一设备上
            model_device = next(model.parameters()).device
            x_batch = x_batch.to(model_device)
            y_batch = y_batch.to(model_device)

            logits = model(x_batch)
            B, T, C = logits.shape
            logits_flat = logits.view(B*T, C)
            targets_flat = y_batch.reshape(-1)
            loss = cross_entropy(logits_flat, targets_flat)

            optimizer.zero_grad()
            loss.backward()
            gradient_clipping(model.parameters(), grad_clip)
            optimizer.step()

            # Update progress visualization
            if visualize:
                if RICH_AVAILABLE:
                    progress_display.update(training_task, advance=1)
                    progress_display.update(training_task, description=f"[green]Training - Loss: {loss.item():.4f}, LR: {lr:.6f}")
                elif TQDM_AVAILABLE:
                    progress_bar.set_description(f"Loss: {loss.item():.4f}, LR: {lr:.6f}")
                    progress_bar.update(1)

            # Track training stats
            if it % log_interval == 0:
                training_stats["train_loss"].append(loss.item())
                training_stats["learning_rate"].append(lr)
                training_stats["iterations"].append(it)

                elapsed = time.time() - start_time
                steps_per_sec = (it - start_iter + 1) / elapsed if elapsed > 0 else 0
                logger.info(f"Iter {it}: loss {loss.item():.4f}, lr {lr:.6f}, {steps_per_sec:.2f} it/s")
                if use_wandb:
                    wandb.log({
                        "train/loss": loss.item(),
                        "train/learning_rate": lr,
                        "perf/steps_per_sec": steps_per_sec,
                        "perf/elapsed_seconds": elapsed,
                    }, step=it)

                # Display a nicely formatted stats table periodically
                if RICH_AVAILABLE and it > 0 and it % (log_interval * 5) == 0:
                    stats_table = Table(title=f"Training Stats (Iteration {it})")
                    stats_table.add_column("Metric", style="cyan")
                    stats_table.add_column("Value", justify="right")
                    
                    stats_table.add_row("Current Loss", f"{loss.item():.4f}")
                    stats_table.add_row("Learning Rate", f"{lr:.6f}")
                    stats_table.add_row("Steps/Sec", f"{steps_per_sec:.2f}")
                    if len(training_stats['train_loss']) > 1:
                        avg_loss = sum(training_stats['train_loss'][-5:]) / min(5, len(training_stats['train_loss']))
                        stats_table.add_row("Avg Loss (last 5)", f"{avg_loss:.4f}")
                    if val_dataset is not None and len(training_stats['val_loss']) > 0:
                        stats_table.add_row("Best Val Loss", f"{best_val_loss:.4f}")
                    
                    console.print(stats_table)

            if val_dataset is not None and it > 0 and it % eval_interval == 0:
                val_loss = estimate_loss(
                    model=model,
                    dataset=val_dataset,
                    batch_size=batch_size,
                    context_length=context_length,
                    device=device,
                    eval_iters=eval_iters
                )
                logger.info(f"Iter {it}: val_loss {val_loss:.4f}")
                if use_wandb:
                    wandb.log({"val/loss": val_loss}, step=it)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0  # reset patience
                    save_path = checkpoint_dir / f"{checkpoint_prefix}_best.pt"
                    extra_data = {
                        'is_best': True,
                        'val_loss': val_loss,
                    }
                    if use_wandb:
                        extra_data['wandb_run_id'] = wandb.run.id
                    save_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        iteration=it,
                        loss=val_loss,
                        output_path=save_path,
                        extra_data=extra_data
                    )
                    if use_wandb:
                        wandb.run.summary["best_val_loss"] = val_loss
                        wandb.run.summary["best_val_loss_step"] = it
                else:
                    patience_counter += 1
                    if early_stopping_patience is not None and patience_counter >= early_stopping_patience:
                        logger.info(f"Early stopping at iteration {it} (no val improvement for {early_stopping_patience} evals)")
                        break

                training_stats["val_loss"].append(val_loss)

            # Generate and display sample text periodically
            if generate_samples and it > 0 and it % sample_interval == 0:
                sample_tokens, decoded_text = generate_text_sample(
                    model=model,
                    tokenizer=tokenizer,  # Pass the tokenizer
                    temp=0.8,
                    max_length=sample_length
                )
                
                if RICH_AVAILABLE:
                    console.print("\n[bold yellow]Generated Sample Text:[/bold yellow]")
                    if decoded_text:
                        console.print(decoded_text)
                    else:
                        console.print(f"Token IDs: {sample_tokens[:20]}...{sample_tokens[-20:]}")
                    console.print("\n")
                else:
                    logger.info("Generated Sample:")
                    if decoded_text:
                        logger.info(decoded_text)
                    else:
                        logger.info(f"Token IDs: {sample_tokens[:20]}...{sample_tokens[-20:]}")
                
                # Log sample to wandb if enabled
                if use_wandb:
                    if decoded_text:
                        wandb.log({"samples/text": decoded_text}, step=it)
                    else:
                        wandb.log({"samples/token_sequence": wandb.Html(str(sample_tokens))}, step=it)

            if it > 0 and it % checkpoint_interval == 0:
                save_path = checkpoint_dir / f"{checkpoint_prefix}_{it:06d}.pt"
                extra_data = {}
                if use_wandb:
                    extra_data['wandb_run_id'] = wandb.run.id
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    iteration=it,
                    loss=loss.item(),
                    output_path=save_path,
                    extra_data=extra_data
                )

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        # Save a checkpoint when interrupted
        interrupted_checkpoint_path = checkpoint_dir / f"{checkpoint_prefix}_interrupted.pt"
        save_checkpoint(model=model, optimizer=optimizer, iteration=it, 
                       loss=loss.item(), output_path=interrupted_checkpoint_path)
        logger.info(f"Saved interrupt checkpoint to {interrupted_checkpoint_path}")
        
    finally:
        # Cleanup visualization
        if visualize:
            if RICH_AVAILABLE:
                progress_display.stop()
            elif TQDM_AVAILABLE:
                progress_bar.close()

    final_checkpoint_path = checkpoint_dir / f"{checkpoint_prefix}_final.pt"
    extra_data = {
        'total_time': time.time() - start_time
    }
    if use_wandb:
        extra_data['wandb_run_id'] = wandb.run.id
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        iteration=max_iters,
        loss=loss.item(),
        output_path=final_checkpoint_path,
        extra_data=extra_data
    )

    logger.info(f"Training completed! Final model saved to {final_checkpoint_path}")

    if use_wandb:
        wandb.run.summary["final_loss"] = loss.item()
        wandb.run.summary["total_training_time"] = time.time() - start_time
        wandb.finish()

    # Training summary with rich if available 
    if RICH_AVAILABLE:
        console.print("[bold green]Training Complete![/bold green]")
        summary_table = Table(title="Training Summary")
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", justify="right")
        
        total_time = time.time() - start_time
        summary_table.add_row("Total Training Time", f"{total_time:.2f} seconds")
        summary_table.add_row("Final Train Loss", f"{loss.item():.4f}")
        if val_dataset is not None:
            summary_table.add_row("Best Val Loss", f"{best_val_loss:.4f}")
        summary_table.add_row("Model Parameters", f"{param_count / 1e6:.2f}M")
        summary_table.add_row("Total Iterations", str(max_iters))
        
        console.print(summary_table)

    return model


def train_tiny_stories(
    data_dir: str = './data',
    output_dir: str = './tiny_stories_model',
    context_length: int = 256,
    d_model: int = 384,
    num_heads: int = 6,
    num_layers: int = 6,
    d_ff: int = 1536,
    batch_size: int = 32,
    max_iters: int = 5000,
    learning_rate: float = 1e-4,
    device: str = 'mps',
    use_wandb: bool = False,
    wandb_project: str = "cs336-tinystories",
    wandb_entity: Optional[str] = None,
    visualize: bool = True,
    generate_samples: bool = True,
    vocab_size: Optional[int] = 10000,
    tokenizer: Optional[BPETokenizer] = None,
    early_stopping_patience: Optional[int] = None,  # early stopping patience
    resume_from: Optional[str] = None,
) -> None:
    """
    Quick training function for language modeling on the TinyStories dataset.
    Suitable for training on Apple Silicon M-series chips.

    Args:
        data_dir: Directory containing TinyStories tokenized data
        output_dir: Output directory for checkpoints and logs
        context_length: Context window size
        d_model: Model dimension
        num_heads: Number of attention heads
        num_layers: Number of transformer layers
        d_ff: Feedforward layer dimension
        batch_size: Batch size
        max_iters: Maximum training iterations
        learning_rate: Learning rate
        device: Training device
        use_wandb: Whether to use W&B for logging
        wandb_project: W&B project name
        wandb_entity: W&B entity/username (optional)
        visualize: Whether to use visual progress tracking (requires tqdm or rich)
        generate_samples: Whether to generate text samples during training
        vocab_size: Vocabulary size
        tokenizer: Tokenizer for text generation
    """
    # Prepare directories
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load tokenized data
    train_path = data_dir / "tinystories_train_tokens.npy"
    val_path = data_dir / "tinystories_val_tokens.npy"

    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError(
            f"Tokenized data files not found. Expected at {train_path} and {val_path}. "
            "Please run tokenization first."
        )

    # Use memory mapping for loading
    logger.info("Loading tokenized data...")
    train_data = np.memmap(train_path, dtype=np.int32, mode='r')
    val_data = np.memmap(val_path, dtype=np.int32, mode='r')

    # Estimate vocabulary size
    try:
        # Method 1: Random sampling from all training data
        sample_size = min(100000, len(train_data))
        random_indices = np.random.choice(len(train_data), sample_size, replace=False)
        train_sample = train_data[random_indices]


        # Ensure vocabulary size is within a reasonable range
        if vocab_size > 100000:
            logger.warning(f"Detected unusually large vocabulary size: {vocab_size}")
            logger.warning("Using default BPE vocabulary size of 50257 instead")
            vocab_size = 50257  # GPT-2 default vocabulary size
    except Exception as e:
        logger.warning(f"Error detecting vocabulary size: {e}")
        logger.warning("Using default vocabulary size of 50257")
        vocab_size = 50257

    logger.info(f"Using vocabulary size: {vocab_size}")

    # Save training config
    config = {
        "vocab_size": int(vocab_size),
        "context_length": context_length,
        "d_model": d_model,
        "num_heads": num_heads,
        "num_layers": num_layers,
        "d_ff": d_ff,
        "batch_size": batch_size,
        "max_iters": max_iters,
        "learning_rate": learning_rate,
        "device": device,
        "train_data_path": str(train_path),
        "val_data_path": str(val_path),
        "timestamp": time.strftime("%Y-%m-%d_%H-%M-%S")
    }

    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Generate wandb run name
    wandb_run_name = f"tinystories-{num_layers}l-{d_model}d-{num_heads}h-{time.strftime('%Y%m%d-%H%M')}"

    # Train the model
    train(
        train_dataset=train_data,
        val_dataset=val_data,
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        batch_size=batch_size,
        max_iters=max_iters,
        learning_rate=learning_rate,
        device=device,
        checkpoint_dir=str(output_dir / "checkpoints"),
        checkpoint_prefix="tinystories",
        use_wandb=use_wandb,
        wandb_project=wandb_project,
        wandb_run_name=wandb_run_name,
        wandb_entity=wandb_entity,
        visualize=visualize,
        generate_samples=generate_samples,
        sample_interval=200,  # Generate samples more frequently for TinyStories
        tokenizer=tokenizer,  # Pass the tokenizer
        early_stopping_patience=early_stopping_patience, # Early stopping patience
        resume_from=resume_from,  # Resume from checkpoint if provided
    )

    logger.info(f"Training completed! Model saved to {output_dir}/checkpoints")

def train_owt(
    data_dir: str = './data',
    output_dir: str = './owt_model',
    context_length: int = 256,
    d_model: int = 768,
    num_heads: int = 12,
    num_layers: int = 12,
    d_ff: int = 3072,
    batch_size: int = 16,
    max_iters: int = 500,
    learning_rate: float = 2e-4,
    device: str = 'mps',
    use_wandb: bool = False,
    wandb_project: str = "cs336-owt",
    wandb_entity: Optional[str] = None,
    visualize: bool = True,
    generate_samples: bool = True,
    sample_interval: int = 500,
    ) -> None:
        """
        Quick training function for language modeling on the OpenWebText dataset.

        Args:
            data_dir: Directory containing OWT tokenized data
            output_dir: Output directory for checkpoints and logs
            context_length: Context window size
            d_model: Model dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            d_ff: Feedforward layer dimension
            batch_size: Batch size
            max_iters: Maximum training iterations
            learning_rate: Learning rate
            device: Training device
            use_wandb: Whether to use W&B for logging
            wandb_project: W&B project name
            wandb_entity: W&B username or team name (optional)
            visualize: Whether to use visual progress tracking (requires tqdm or rich)
            generate_samples: Whether to generate text samples during training
            sample_interval: Generate samples every N iterations
        """
        data_dir = Path(data_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        train_path = data_dir / "owt_train_tokens.npy"
        val_path = data_dir / "owt_val_tokens.npy"

        if not train_path.exists() or not val_path.exists():
            raise FileNotFoundError(
                f"Tokenized OWT data files not found. Expected at {train_path} and {val_path}."
            )

        logger.info("Loading OWT tokenized data...")
        train_data = np.memmap(train_path, dtype=np.int32, mode='r')
        val_data = np.memmap(val_path, dtype=np.int32, mode='r')

        vocab_size = max(
            np.max(np.memmap(train_path, dtype=np.int32, mode='r', shape=(100,))),
            np.max(np.memmap(val_path, dtype=np.int32, mode='r', shape=(100,)))
        ) + 1

        logger.info(f"OWT training data size: {len(train_data)} tokens")
        logger.info(f"OWT validation data size: {len(val_data)} tokens")
        logger.info(f"Detected OWT vocabulary size: {vocab_size}")

        config = {
            "vocab_size": int(vocab_size),
            "context_length": context_length,
            "d_model": d_model,
            "num_heads": num_heads,
            "num_layers": num_layers,
            "d_ff": d_ff,
            "batch_size": batch_size,
            "max_iters": max_iters,
            "learning_rate": learning_rate,
            "device": device,
            "train_data_path": str(train_path),
            "val_data_path": str(val_path),
            "timestamp": time.strftime("%Y-%m-%d_%H-%M-%S")
        }

        with open(output_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        wandb_run_name = f"owt-{num_layers}l-{d_model}d-{num_heads}h-{time.strftime('%Y%m%d-%H%M')}"

        train(
            train_dataset=train_data,
            val_dataset=val_data,
            vocab_size=vocab_size,
            context_length=context_length,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            d_ff=d_ff,
            batch_size=batch_size,
            max_iters=max_iters,
            learning_rate=learning_rate,
            device=device,
            checkpoint_dir=str(output_dir / "checkpoints"),
            checkpoint_prefix="owt",
            use_wandb=use_wandb,
            wandb_project=wandb_project,
            wandb_run_name=wandb_run_name,
            wandb_entity=wandb_entity,
            visualize=visualize,
            generate_samples=generate_samples,
            sample_interval=sample_interval,
        )

        logger.info(f"OWT training completed! Model saved to {output_dir}/checkpoints")

# Example usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train a Transformer language model")
    parser.add_argument("--data_dir", type=str, default="./data", help="Directory with training data")
    parser.add_argument("--output_dir", type=str, default="./model", help="Output directory")
    parser.add_argument("--dataset", type=str, choices=["tinystories", "owt"], default="tinystories",
                      help="Dataset to train on")
    parser.add_argument("--device", type=str, default="mps", help="Device to train on (cpu, mps, cuda)")
    # wandb related arguments
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="cs336-lm", help="W&B project name")
    parser.add_argument("--wandb_entity", type=str, default=None, help="W&B username or team name")
    
    # Add new visualization arguments
    parser.add_argument("--visualize", action="store_true", help="Use visual progress tracking", default=True)
    parser.add_argument("--no-visualize", action="store_false", dest="visualize", help="Disable visual progress tracking")
    parser.add_argument("--generate-samples", action="store_true", help="Generate text samples during training", default=True)
    parser.add_argument("--no-generate-samples", action="store_false", dest="generate_samples", help="Disable text sample generation")
    parser.add_argument("--sample-interval", type=int, default=500, help="Generate samples every N iterations")
    
    args = parser.parse_args()

    # Select training function based on dataset
    if args.dataset == "tinystories":
        train_tiny_stories(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            device=args.device,
            use_wandb=args.use_wandb,
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
            visualize=args.visualize,
            generate_samples=args.generate_samples,
        )
    elif args.dataset == "owt":
        train_owt(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            device=args.device,
            use_wandb=args.use_wandb,
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
            visualize=args.visualize,
            generate_samples=args.generate_samples,
            sample_interval=args.sample_interval,
        )
    else:
        raise ValueError("Invalid dataset. Choose 'tinystories' or 'owt'.")
    # Note: The above code assumes that the TinyStories and OWT datasets are already tokenized and saved as .npy files.
    # The tokenization process is not included in this script.