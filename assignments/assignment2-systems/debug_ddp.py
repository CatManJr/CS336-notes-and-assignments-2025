#!/usr/bin/env python3
"""
Debug script to understand the DDPBucketed gradient synchronization issue
"""

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy
import os


def debug_ddp_worker(rank, world_size):
    """Debug worker to test DDPBucketed implementation"""
    # Setup distributed training
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356'
    dist.init_process_group('gloo', rank=rank, world_size=world_size)
    
    # Set seed for reproducibility
    torch.manual_seed(42)  # Same seed for all ranks
    
    # Simple model
    model = nn.Sequential(
        nn.Linear(2, 3),
        nn.Linear(3, 1)
    )
    
    # Make a copy for DDP
    ddp_model_base = deepcopy(model)
    
    # Import and wrap with our DDPBucketed
    from cs336_systems.naive_ddp import DDPBucketed
    ddp_model = DDPBucketed(ddp_model_base, bucket_size_mb=0.001)
    
    # Simple data
    torch.manual_seed(rank)  # Different data per rank
    x = torch.randn(4, 2)
    y = torch.randn(4, 1)
    
    # Optimizers
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    ddp_optimizer = optim.SGD(ddp_model.parameters(), lr=0.1)
    
    # Forward pass on non-parallel model (all data)
    optimizer.zero_grad()
    out = model(x)
    loss = nn.MSELoss()(out, y)
    loss.backward()
    
    print(f"Rank {rank}: Before DDP sync")
    for i, param in enumerate(model.parameters()):
        print(f"  Non-parallel param {i} grad: {param.grad.flatten()[:3]}")
    
    optimizer.step()
    
    # Forward pass on DDP model (rank-specific data)
    ddp_optimizer.zero_grad()
    ddp_out = ddp_model(x)
    ddp_loss = nn.MSELoss()(ddp_out, y)
    ddp_loss.backward()
    
    print(f"Rank {rank}: Before finish_gradient_synchronization")
    for i, param in enumerate(ddp_model.parameters()):
        print(f"  DDP param {i} grad: {param.grad.flatten()[:3]}")
    
    # Synchronize gradients
    ddp_model.finish_gradient_synchronization()
    
    print(f"Rank {rank}: After finish_gradient_synchronization")
    for i, param in enumerate(ddp_model.parameters()):
        print(f"  DDP param {i} grad: {param.grad.flatten()[:3]}")
    
    ddp_optimizer.step()
    
    # Compare final parameters
    print(f"Rank {rank}: Final parameter comparison")
    for i, (non_par, ddp_par) in enumerate(zip(model.parameters(), ddp_model.parameters())):
        close = torch.allclose(non_par, ddp_par, atol=1e-5)
        print(f"  Param {i} close: {close}")
        if not close:
            print(f"    Non-parallel: {non_par.flatten()[:3]}")
            print(f"    DDP: {ddp_par.flatten()[:3]}")
            print(f"    Diff: {(non_par - ddp_par).flatten()[:3]}")
    
    dist.destroy_process_group()


if __name__ == "__main__":
    world_size = 2
    mp.spawn(debug_ddp_worker, args=(world_size,), nprocs=world_size, join=True)