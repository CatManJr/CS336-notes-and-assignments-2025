#!/usr/bin/env python3
"""
Naive Distributed Data Parallel (DDP) Implementation

This module implements a simple version of distributed data parallel training
that all-reduces individual parameter gradients after the backward pass.
"""

import torch
import torch.distributed as dist
import torch.nn as nn
from typing import Iterator, Optional, List, Dict
import threading
import queue


class DDPIndividualParameters(nn.Module):
    """
    A naive DDP implementation that all-reduces individual parameter gradients.
    
    This implementation follows the basic DDP algorithm:
    1. Broadcast parameters from rank 0 to all ranks during initialization
    2. During backward pass, accumulate gradients locally
    3. After backward pass, all-reduce each parameter's gradient individually
    4. Apply optimizer step with synchronized gradients
    """
    
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        
        # Broadcast parameters from rank 0 to all other ranks
        self._broadcast_parameters()
        
        # Track gradients that need to be synchronized
        self._grad_sync_pending = False  # Start as False since no gradients yet
    
    def _broadcast_parameters(self):
        """Broadcast all parameters from rank 0 to all other ranks"""
        for param in self.module.parameters():
            # Broadcast parameter values from rank 0
            dist.broadcast(param.data, src=0)
    
    def forward(self, *args, **kwargs):
        """Forward pass through the wrapped module"""
        # Mark that we'll need gradient sync after backward
        self._grad_sync_pending = True
        return self.module(*args, **kwargs)
    
    def finish_gradient_synchronization(self):
        """
        All-reduce gradients across all ranks.
        This should be called after backward() but before optimizer.step()
        """
        if not self._grad_sync_pending:
            return
            
        for param in self.module.parameters():
            if param.requires_grad and param.grad is not None:
                # All-reduce the gradient across all ranks
                # This averages the gradients across all processes
                dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                # Average the gradients (since all_reduce sums them)
                param.grad.data /= self.world_size
        
        self._grad_sync_pending = False
    
    def zero_grad(self, set_to_none: bool = False):
        """Zero gradients and reset sync state"""
        self.module.zero_grad(set_to_none=set_to_none)
        self._grad_sync_pending = False  # No gradients to sync after zeroing
    
    def parameters(self, recurse: bool = True) -> Iterator[nn.Parameter]:
        """Return parameters of the wrapped module"""
        return self.module.parameters(recurse=recurse)
    
    def named_parameters(self, prefix: str = '', recurse: bool = True):
        """Return named parameters of the wrapped module"""
        return self.module.named_parameters(prefix=prefix, recurse=recurse)
    
    def state_dict(self, *args, **kwargs):
        """Return state dict of the wrapped module"""
        return self.module.state_dict(*args, **kwargs)
    
    def load_state_dict(self, *args, **kwargs):
        """Load state dict into the wrapped module"""
        return self.module.load_state_dict(*args, **kwargs)


def naive_ddp_train_step(model: DDPIndividualParameters, 
                        data: torch.Tensor, 
                        targets: torch.Tensor,
                        loss_fn: nn.Module,
                        optimizer: torch.optim.Optimizer) -> torch.Tensor:
    """
    Perform a single training step with naive DDP.
    
    Args:
        model: DDP-wrapped model
        data: Input data for this rank
        targets: Target labels for this rank  
        loss_fn: Loss function
        optimizer: Optimizer
        
    Returns:
        Loss value
    """
    # Zero gradients
    optimizer.zero_grad()
    
    # Forward pass
    outputs = model(data)
    loss = loss_fn(outputs, targets)
    
    # Backward pass
    loss.backward()
    
    # Synchronize gradients across ranks
    model.finish_gradient_synchronization()
    
    # Optimizer step
    optimizer.step()
    
    return loss


def create_naive_ddp_training_script():
    """
    Create a complete training script demonstrating naive DDP usage.
    This function shows how to use the DDP implementation in practice.
    """
    import os
    import torch.multiprocessing as mp
    from torch.utils.data import DataLoader, TensorDataset
    
    def train_worker(rank: int, world_size: int, data: torch.Tensor, targets: torch.Tensor):
        """Worker function for each process in distributed training"""
        # Setup distributed training
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group('gloo', rank=rank, world_size=world_size)
        
        # Set random seed for reproducibility (different per rank initially)
        torch.manual_seed(rank)
        
        # Create model
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 1)
        )
        
        # Wrap with DDP
        ddp_model = DDPIndividualParameters(model)
        
        # Create optimizer
        optimizer = torch.optim.SGD(ddp_model.parameters(), lr=0.01)
        loss_fn = nn.MSELoss()
        
        # Split data across ranks
        batch_size = data.size(0) // world_size
        start_idx = rank * batch_size
        end_idx = start_idx + batch_size
        local_data = data[start_idx:end_idx]
        local_targets = targets[start_idx:end_idx]
        
        # Training loop
        for epoch in range(5):
            loss = naive_ddp_train_step(ddp_model, local_data, local_targets, loss_fn, optimizer)
            
            if rank == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
        
        # Cleanup
        dist.destroy_process_group()
    
    # Generate synthetic data
    torch.manual_seed(42)
    data = torch.randn(100, 10)
    targets = torch.randn(100, 1)
    
    # Launch distributed training
    world_size = 2
    mp.spawn(train_worker, args=(world_size, data, targets), nprocs=world_size, join=True)


class DDPOverlapIndividualParameters(nn.Module):
    """
    DDP implementation that overlaps computation with communication.
    
    This implementation uses autograd hooks to trigger all-reduce operations
    as soon as gradients are computed for each parameter, allowing communication
    to overlap with the backward pass computation.
    """
    
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        
        # Track async communication handles
        self._async_handles: List[dist.Work] = []
        self._grad_ready_count = 0
        self._total_params = sum(1 for p in self.module.parameters() if p.requires_grad)
        self._sync_finished = False
        
        # Register backward hooks for overlapping communication
        self._register_backward_hooks()
        
        # Broadcast parameters from rank 0 to all other ranks
        self._broadcast_parameters()
    
    def _broadcast_parameters(self):
        """Broadcast all parameters from rank 0 to all other ranks"""
        for param in self.module.parameters():
            dist.broadcast(param.data, src=0)
    
    def _register_backward_hooks(self):
        """Register backward hooks to trigger async all-reduce as gradients are ready"""
        def make_hook(param):
            def hook(grad):
                if grad is not None:
                    # Clone gradient to avoid issues with in-place operations
                    grad_copy = grad.clone()
                    
                    # Launch async all-reduce
                    handle = dist.all_reduce(grad_copy, op=dist.ReduceOp.SUM, async_op=True)
                    self._async_handles.append((handle, param, grad_copy))
                    
                return grad
            return hook
        
        # Register hooks for all parameters that require gradients
        for param in self.module.parameters():
            if param.requires_grad:
                param.register_hook(make_hook(param))
    
    def forward(self, *args, **kwargs):
        """Forward pass through the wrapped module"""
        # Reset state for new forward pass
        self._async_handles.clear()
        self._grad_ready_count = 0
        self._sync_finished = False
        
        return self.module(*args, **kwargs)
    
    def finish_gradient_synchronization(self):
        """
        Wait for all async all-reduce operations to complete and apply results.
        This should be called after backward() but before optimizer.step()
        """
        if self._sync_finished:
            return
        
        # Wait for all async operations to complete and apply results
        for handle, param, grad_copy in self._async_handles:
            # Wait for the all-reduce to complete
            handle.wait()
            
            # Average the gradient and update the parameter's gradient
            grad_copy /= self.world_size
            param.grad.copy_(grad_copy)
        
        self._sync_finished = True
        self._async_handles.clear()
    
    def zero_grad(self, set_to_none: bool = False):
        """Zero gradients and reset sync state"""
        self.module.zero_grad(set_to_none=set_to_none)
        self._async_handles.clear()
        self._grad_ready_count = 0
        self._sync_finished = False
    
    def parameters(self, recurse: bool = True) -> Iterator[nn.Parameter]:
        """Return parameters of the wrapped module"""
        return self.module.parameters(recurse=recurse)
    
    def named_parameters(self, prefix: str = '', recurse: bool = True):
        """Return named parameters of the wrapped module"""
        return self.module.named_parameters(prefix=prefix, recurse=recurse)
    
    def state_dict(self, *args, **kwargs):
        """Return state dict of the wrapped module"""
        return self.module.state_dict(*args, **kwargs)
    
    def load_state_dict(self, *args, **kwargs):
        """Load state dict into the wrapped module"""
        return self.module.load_state_dict(*args, **kwargs)


class DDPBucketed(nn.Module):
    """
    Distributed Data Parallel with gradient bucketing and overlapping communication.
    
    This implementation organizes parameters into buckets based on size and launches
    all-reduce operations as soon as all gradients in a bucket are ready, enabling
    overlapping of communication with computation.
    """
    
    def __init__(self, module: nn.Module, bucket_size_mb: float = 25.0):
        super().__init__()
        self.module = module
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        
        # Calculate bucket size in number of parameters
        param_dtype = next(module.parameters()).dtype
        bytes_per_param = param_dtype.itemsize
        self.bucket_size_params = int(bucket_size_mb * 1024 * 1024 / bytes_per_param)
        
        # Initialize buckets
        self._buckets: List[Dict] = []
        self._init_buckets()
        
        # Track async handles
        self._async_handles: List[tuple] = []
        
        # Broadcast parameters from rank 0 to all other ranks
        self._broadcast_parameters()
        
        # Register hooks
        self._register_hooks()
    
    def _init_buckets(self):
        """Assign parameters to buckets, respecting the maximum bucket size."""
        current_bucket = {
            'params': [],
            'num_params': 0,
            'num_params_ready': 0
        }
        bucket_idx = 0
        
        # Process parameters in reverse order (gradients become ready in reverse order)
        for param in reversed(list(self.module.parameters())):
            if not param.requires_grad:
                continue
            
            # If adding this parameter would exceed bucket size and we have params in bucket,
            # finalize the current bucket and start a new one
            if (current_bucket['num_params'] + param.numel() > self.bucket_size_params and 
                current_bucket['num_params'] > 0):
                self._buckets.append(current_bucket)
                current_bucket = {
                    'params': [],
                    'num_params': 0,
                    'num_params_ready': 0
                }
                bucket_idx += 1
            
            # Store bucket index on parameter for hook lookup
            param.bucket_idx = bucket_idx
            current_bucket['params'].append(param)
            current_bucket['num_params'] += param.numel()
        
        # Add the last bucket if it has any parameters
        if current_bucket['num_params'] > 0:
            self._buckets.append(current_bucket)
    
    def _broadcast_parameters(self):
        """Broadcast all parameters from rank 0 to all other ranks"""
        for param in self.module.parameters():
            dist.broadcast(param.data, src=0)
    
    def _register_hooks(self):
        """Register post-accumulate gradient hooks for each parameter"""
        def make_hook(param: nn.Parameter):
            def hook(*args):
                bucket_idx = param.bucket_idx
                bucket = self._buckets[bucket_idx]
                bucket['num_params_ready'] += param.numel()
                
                # If all parameters in the bucket have accumulated gradients, all-reduce them
                if bucket['num_params_ready'] == bucket['num_params']:
                    params_with_grads = [p for p in bucket['params'] if p.grad is not None]
                    if not params_with_grads:
                        bucket['num_params_ready'] = 0
                        return
                    
                    grads = [p.grad for p in params_with_grads]
                    # Use PyTorch's built-in tensor flattening utilities
                    flattened_grads = torch._utils._flatten_dense_tensors(grads)
                    handle = dist.all_reduce(flattened_grads, op=dist.ReduceOp.SUM, async_op=True)
                    self._async_handles.append((handle, flattened_grads, grads, params_with_grads))
                    bucket['num_params_ready'] = 0
            
            return hook
        
        # Register post-accumulate gradient hooks
        for param in self.module.parameters():
            if param.requires_grad:
                param.register_post_accumulate_grad_hook(make_hook(param))
    
    def forward(self, *args, **kwargs):
        """Forward pass through the wrapped module"""
        return self.module(*args, **kwargs)
    
    def finish_gradient_synchronization(self):
        """
        Wait for all async all-reduce operations to complete and apply averaged gradients.
        This should be called after backward() but before optimizer.step()
        """
        # Wait for all async operations to complete and apply results
        for handle, flattened_grads, grads, params_with_grads in self._async_handles:
            # Wait for the all-reduce to complete
            handle.wait()
            
            # Average the gradients (all-reduce sums, so we divide by world size)
            flattened_grads.div_(self.world_size)
            
            # Unflatten the gradients back to original shapes
            unflattened_grads = torch._utils._unflatten_dense_tensors(flattened_grads, grads)
            
            # Copy the averaged gradients back to parameters
            for param, updated_grad in zip(params_with_grads, unflattened_grads):
                param.grad.copy_(updated_grad)
        
        # Clear handles for next iteration
        self._async_handles.clear()
    
    def zero_grad(self, set_to_none: bool = False):
        """Zero gradients and reset synchronization state"""
        self.module.zero_grad(set_to_none=set_to_none)
        # Reset bucket ready counts
        for bucket in self._buckets:
            bucket['num_params_ready'] = 0
        self._async_handles.clear()

    def parameters(self, recurse: bool = True):
        return self.module.parameters(recurse)

    def named_parameters(self, prefix: str = '', recurse: bool = True):
        return self.module.named_parameters(prefix=prefix, recurse=recurse)

    def state_dict(self, *args, **kwargs):
        return self.module.state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        return self.module.load_state_dict(*args, **kwargs)


if __name__ == "__main__":
    print("Testing naive DDP implementation...")
    create_naive_ddp_training_script()
    print("Naive DDP test completed!")