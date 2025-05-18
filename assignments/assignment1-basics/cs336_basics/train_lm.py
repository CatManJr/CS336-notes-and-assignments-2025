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
from torch import nn, Tensor
import torch.nn.functional as F

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from cs336_basics.optim import AdamW, gradient_clipping, get_lr_cosine_schedule
from cs336_basics.transformer import TransformerLM
from cs336_basics.losses import cross_entropy
from cs336_basics.data import get_batch

# 配置日志
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    loss: float,
    output_path: Union[str, os.PathLike],
    extra_data: Optional[Dict[str, Any]] = None,
) -> None:
    """
    保存模型检查点。
    
    Args:
        model: 要保存的模型
        optimizer: 优化器状态
        iteration: 当前迭代数
        loss: 当前损失值
        output_path: 保存路径
        extra_data: 要保存的其他数据
    """
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'iteration': iteration,
        'loss': loss,
    }
    
    if extra_data:
        checkpoint.update(extra_data)
    
    # 如果是文件路径字符串，转换为Path对象
    if isinstance(output_path, str):
        output_path = Path(output_path)
    
    # 确保目录存在
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 首先保存到临时文件，然后重命名，防止异常中断导致损坏的检查点文件
    temp_path = output_path.with_suffix('.tmp')
    torch.save(checkpoint, temp_path)
    temp_path.rename(output_path)
    
    logger.info(f"Saved checkpoint at iteration {iteration} to {output_path}")


def load_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    checkpoint_path: Union[str, os.PathLike],
) -> Tuple[int, float, Dict[str, Any]]:
    """
    加载模型检查点。
    
    Args:
        model: 要恢复的模型
        optimizer: 要恢复的优化器
        checkpoint_path: 检查点路径
        
    Returns:
        (迭代数, 损失值, 额外数据的字典)
    """
    # 如果是文件路径字符串，转换为Path对象
    if isinstance(checkpoint_path, str):
        checkpoint_path = Path(checkpoint_path)
        
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    
    # 加载检查点
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # 恢复模型和优化器状态
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    
    # 提取迭代数和损失值
    iteration = checkpoint.get('iteration', 0)
    loss = checkpoint.get('loss', float('inf'))
    
    # 删除已处理的键，剩余的就是额外数据
    extra_data = {k: v for k, v in checkpoint.items() 
                 if k not in ('model', 'optimizer', 'iteration', 'loss')}
    
    logger.info(f"Restored checkpoint from iteration {iteration}")
    return iteration, loss, extra_data


def estimate_loss(
    model: nn.Module,
    dataset: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str,
    eval_iters: int = 10,
) -> float:
    """
    在给定的数据集上评估模型的平均损失。
    
    Args:
        model: 要评估的模型
        dataset: 评估数据集
        batch_size: 批次大小
        context_length: 上下文长度
        device: 设备字符串
        eval_iters: 评估迭代数
        
    Returns:
        平均损失值
    """
    model.eval()  # 设置为评估模式
    losses = []
    
    with torch.no_grad():  # 关闭梯度计算
        for _ in range(eval_iters):
            # 使用数据模块中的get_batch函数获取批次数据
            x_batch, y_batch = get_batch(dataset, batch_size, context_length, device)
            
            # 前向传播
            logits = model(x_batch)
            
            # 计算损失
            # 重整形状以匹配交叉熵损失的期望输入
            B, T, C = logits.shape  # batch, time, channels
            logits_flat = logits.view(B*T, C)
            targets_flat = y_batch.reshape(-1)
            
            loss = cross_entropy(logits_flat, targets_flat)
            losses.append(loss.item())
    
    model.train()  # 恢复训练模式
    return sum(losses) / len(losses)


def train(
    # 数据参数
    train_dataset: np.ndarray,
    val_dataset: Optional[np.ndarray] = None,
    # 模型参数
    vocab_size: int = 50257,  # GPT-2 词汇表大小
    context_length: int = 1024,  # 上下文窗口大小
    d_model: int = 512,       # 嵌入维度
    num_heads: int = 8,       # 注意力头数
    num_layers: int = 6,      # Transformer层数
    d_ff: int = 2048,         # 前馈层维度
    dropout: float = 0.1,     # Dropout概率
    # 训练参数
    device: str = 'mps',      # Apple Silicon 默认使用MPS
    batch_size: int = 16,     # 批次大小 (适应内存限制)
    max_iters: int = 10000,   # 最大训练迭代数
    eval_interval: int = 100,  # 多少迭代后评估一次
    eval_iters: int = 10,     # 验证集评估时的迭代数
    log_interval: int = 10,   # 多少迭代后记录一次日志
    # 优化器参数
    learning_rate: float = 1e-4,
    min_learning_rate: float = 1e-5,
    warmup_iters: int = 100,
    weight_decay: float = 0.01,
    beta1: float = 0.9,
    beta2: float = 0.999,
    grad_clip: float = 1.0,   # 梯度裁剪阈值
    # 检查点参数
    checkpoint_dir: str = './checkpoints',
    checkpoint_interval: int = 500,
    checkpoint_prefix: str = 'model',
    resume_from: Optional[str] = None,
    # wandb参数
    use_wandb: bool = False,
    wandb_project: str = "cs336-lm-training",
    wandb_run_name: Optional[str] = None,
    wandb_entity: Optional[str] = None,
) -> nn.Module:
    """
    训练Transformer语言模型。
    
    Args:
        train_dataset: 训练数据集
        val_dataset: 验证数据集（可选）
        
        vocab_size: 词汇表大小
        context_length: 上下文长度
        d_model: 模型维度
        num_heads: 注意力头数量
        num_layers: Transformer层数
        d_ff: 前馈网络隐藏维度
        dropout: Dropout比率
        
        device: 训练设备 ('cpu'、'mps'或'cuda:0')
        batch_size: 批次大小
        max_iters: 最大训练迭代数
        eval_interval: 评估间隔
        eval_iters: 每次评估的迭代数
        log_interval: 日志记录间隔
        
        learning_rate: 最大学习率
        min_learning_rate: 最小学习率
        warmup_iters: 预热迭代数
        weight_decay: 权重衰减率
        beta1: Adam优化器beta1
        beta2: Adam优化器beta2
        grad_clip: 梯度裁剪阈值
        
        checkpoint_dir: 检查点保存目录
        checkpoint_interval: 检查点保存间隔
        checkpoint_prefix: 检查点文件名前缀
        resume_from: 从检查点恢复（可选）
        
        use_wandb: 是否使用Weights & Biases记录训练过程
        wandb_project: W&B项目名称
        wandb_run_name: W&B运行名称（可选）
        wandb_entity: W&B组织/用户名（可选）
        
    Returns:
        训练好的模型
    """
    # 确保检查点目录存在
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # 配置环境
    torch.manual_seed(42)  # 设置随机种子以确保可重现性
    
    # 初始化wandb
    if use_wandb:
        if not WANDB_AVAILABLE:
            logger.warning("Weights & Biases (wandb) not installed. Logging to wandb will be disabled.")
            use_wandb = False
        else:
            # 创建配置字典，包含所有超参数
            config = {
                "vocab_size": vocab_size,
                "context_length": context_length,
                "d_model": d_model,
                "num_heads": num_heads,
                "num_layers": num_layers,
                "d_ff": d_ff,
                "dropout": dropout,
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
            
            # 初始化wandb
            wandb.init(
                project=wandb_project,
                name=wandb_run_name,
                entity=wandb_entity,
                config=config,
                resume="allow" if resume_from else None,
            )
            
            # 在检查点目录中创建wandb id文件，用于恢复运行
            with open(checkpoint_dir / "wandb_id.txt", "w") as f:
                f.write(wandb.run.id)
    
    # 初始化模型
    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        dropout=dropout,
        theta=10000.0,  # RoPE theta参数，使用默认值
        device=device,
        dtype=torch.float32  # 对于Apple Silicon，float32通常较好
    )
    
    # 参数数量估计
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model has {param_count / 1e6:.2f}M parameters")
    
    # 内存使用估计（粗略计算，假设float32）
    bytes_per_param = 4  # float32占4字节
    model_size_gb = param_count * bytes_per_param / (1024**3)
    optimizer_size_gb = model_size_gb * 2  # 优化器状态通常是模型的2倍
    
    # 批次大小和梯度的内存估计
    batch_memory_gb = (batch_size * context_length * d_model * 4) / (1024**3) * 4  # 乘4因为有激活值、梯度等
    
    total_memory_gb = model_size_gb + optimizer_size_gb + batch_memory_gb
    
    logger.info(f"Estimated memory usage:")
    logger.info(f"  - Model parameters: {model_size_gb:.2f} GB")
    logger.info(f"  - Optimizer states: {optimizer_size_gb:.2f} GB")
    logger.info(f"  - Batch processing: {batch_memory_gb:.2f} GB")
    logger.info(f"  - Total estimated: {total_memory_gb:.2f} GB")
    
    if use_wandb:
        wandb.log({
            "model/parameters_count": param_count,
            "model/parameters_gb": model_size_gb,
            "memory/optimizer_gb": optimizer_size_gb,
            "memory/batch_gb": batch_memory_gb,
            "memory/total_gb": total_memory_gb
        }, step=0)
    
    # 初始化优化器
    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(beta1, beta2)
    )
    
    # 从检查点恢复（如果提供）
    start_iter = 0
    best_val_loss = float('inf')
    if resume_from:
        start_iter, val_loss, extra_data = load_checkpoint(
            model=model,
            optimizer=optimizer,
            checkpoint_path=resume_from
        )
        best_val_loss = val_loss
        logger.info(f"Resuming from iteration {start_iter} with validation loss {val_loss:.4f}")
        
        if use_wandb and "wandb_run_id" in extra_data:
            # 尝试恢复相同的wandb运行
            wandb.run.id = extra_data["wandb_run_id"]
            logger.info(f"Resuming wandb run with ID: {wandb.run.id}")
    
    # 训练循环
    model.train()
    start_time = time.time()
    
    for it in range(start_iter, max_iters):
        # 计算当前学习率
        lr = get_lr_cosine_schedule(
            it=it,
            max_learning_rate=learning_rate,
            min_learning_rate=min_learning_rate,
            warmup_iters=warmup_iters,
            cosine_cycle_iters=max_iters
        )
        
        # 更新优化器学习率
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # 使用数据模块中的get_batch函数采样批次数据
        x_batch, y_batch = get_batch(train_dataset, batch_size, context_length, device)
        
        # 前向传播
        logits = model(x_batch)
        
        # 计算损失
        B, T, C = logits.shape  # batch, time, channels
        logits_flat = logits.view(B*T, C)
        targets_flat = y_batch.reshape(-1)
        
        loss = cross_entropy(logits_flat, targets_flat)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        gradient_clipping(model.parameters(), grad_clip)
        
        # 更新参数
        optimizer.step()
        
        # 定期记录训练信息
        if it % log_interval == 0:
            elapsed = time.time() - start_time
            steps_per_sec = (it - start_iter + 1) / elapsed if elapsed > 0 else 0
            logger.info(f"Iter {it}: loss {loss.item():.4f}, lr {lr:.6f}, {steps_per_sec:.2f} it/s")
            
            # 记录到wandb
            if use_wandb:
                wandb.log({
                    "train/loss": loss.item(),
                    "train/learning_rate": lr,
                    "perf/steps_per_sec": steps_per_sec,
                    "perf/elapsed_seconds": elapsed,
                }, step=it)
        
        # 定期验证评估
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
            
            # 记录到wandb
            if use_wandb:
                wandb.log({"val/loss": val_loss}, step=it)
            
            # 如果是最佳模型，保存一个特殊检查点
            if val_loss < best_val_loss:
                best_val_loss = val_loss
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
                
                # 记录模型到wandb
                if use_wandb:
                    wandb.run.summary["best_val_loss"] = val_loss
                    wandb.run.summary["best_val_loss_step"] = it
        
        # 定期保存检查点
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
    
    # 训练结束，保存最终模型
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
    
    # 完成wandb记录
    if use_wandb:
        wandb.run.summary["final_loss"] = loss.item()
        wandb.run.summary["total_training_time"] = time.time() - start_time
        wandb.finish()
    
    return model


def train_tiny_stories(
    data_dir: str = './data',
    output_dir: str = './tiny_stories_model',
    context_length: int = 512,
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
) -> None:
    """
    在TinyStories数据集上训练语言模型的快速训练函数。
    适合在Apple Silicon M系列芯片上训练。
    
    Args:
        data_dir: 存储TinyStories tokenized数据的目录
        output_dir: 输出目录，用于保存检查点和日志
        context_length: 上下文窗口大小
        d_model: 模型维度
        num_heads: 注意力头数
        num_layers: transformer层数
        d_ff: 前馈层维度
        batch_size: 批次大小
        max_iters: 最大训练迭代数
        learning_rate: 学习率
        device: 训练设备
        use_wandb: 是否使用W&B记录训练过程
        wandb_project: W&B项目名称
        wandb_entity: W&B组织/用户名（可选）
    """
    # 准备目录
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载tokenized数据
    train_path = data_dir / "tinystories_train_tokens.npy"
    val_path = data_dir / "tinystories_val_tokens.npy"
    
    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError(
            f"Tokenized data files not found. Expected at {train_path} and {val_path}. "
            "Please run tokenization first."
        )
    
    # 使用内存映射加载
    logger.info("Loading tokenized data...")
    train_data = np.memmap(train_path, dtype=np.int32, mode='r')
    val_data = np.memmap(val_path, dtype=np.int32, mode='r')
    
    # 获取词汇表大小
    vocab_size = max(
        np.max(np.memmap(train_path, dtype=np.int32, mode='r', shape=(100,))),
        np.max(np.memmap(val_path, dtype=np.int32, mode='r', shape=(100,)))
    ) + 1
    
    logger.info(f"Training data size: {len(train_data)} tokens")
    logger.info(f"Validation data size: {len(val_data)} tokens")
    logger.info(f"Detected vocabulary size: {vocab_size}")
    
    # 保存训练配置
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
    
    # 生成wandb运行名称
    wandb_run_name = f"tinystories-{num_layers}l-{d_model}d-{num_heads}h-{time.strftime('%Y%m%d-%H%M')}"
    
    # 训练模型
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
    )
    
    logger.info(f"Training completed! Model saved to {output_dir}/checkpoints")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train a Transformer language model")
    parser.add_argument("--data_dir", type=str, default="./data", help="Directory with training data")
    parser.add_argument("--output_dir", type=str, default="./model", help="Output directory")
    parser.add_argument("--dataset", type=str, choices=["tinystories", "owt"], default="tinystories", 
                      help="Dataset to train on")
    parser.add_argument("--device", type=str, default="mps", help="Device to train on (cpu, mps, cuda)")
    # 添加wandb相关参数
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="cs336-lm", help="W&B project name")
    parser.add_argument("--wandb_entity", type=str, default=None, help="W&B username or team name")
    
    args = parser.parse_args()
    
    # 根据数据集选择训练函数
    if args.dataset == "tinystories":
        train_tiny_stories(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            device=args.device,
            use_wandb=args.use_wandb,
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity
        )
    else:
        # 为OpenWebText数据集实现类似的训练函数
        logger.info("OpenWebText training not implemented yet")