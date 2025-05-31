#!/usr/bin/env python3
"""
Optimizer State Sharding Implementation

This module implements a sharded optimizer that reduces memory consumption by
partitioning optimizer states across ranks in distributed training.
"""

import torch
import torch.distributed as dist
from torch.optim import Optimizer
from typing import Type, Any, Dict, List, Union
import copy


class ShardedOptimizer(Optimizer):
    """
    Sharded optimizer that distributes optimizer states across ranks to reduce memory usage.
    
    This implementation follows a simplified version of ZeRO stage 1, where optimizer states
    are partitioned across ranks but parameters remain replicated.
    """
    
    def __init__(self, params, optimizer_cls: Type[Optimizer], **kwargs: Any):
        """
        Initialize the sharded optimizer.
        
        Args:
            params: Collection of parameters or parameter groups to optimize
            optimizer_cls: Type of optimizer to wrap (e.g., torch.optim.AdamW)
            **kwargs: Additional arguments passed to the optimizer constructor
        """
        # Store optimizer class and kwargs for later use
        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = kwargs
        
        # Get distributed info
        if dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1
        
        # Process parameters - handle both single params and param groups
        if isinstance(params, torch.nn.Parameter):
            params = [params]
        elif hasattr(params, '__iter__'):
            # Could be a generator or list
            params = list(params)
        
        # Track all parameters across all groups
        self.all_params = []
        processed_param_groups = []
        
        for param_or_group in params:
            if isinstance(param_or_group, dict):
                # It's a parameter group
                group_params = param_or_group['params']
                if isinstance(group_params, torch.nn.Parameter):
                    group_params = [group_params]
                else:
                    group_params = list(group_params)
                
                self.all_params.extend(group_params)
                processed_param_groups.append(param_or_group)
            else:
                # It's a parameter
                self.all_params.append(param_or_group)
                processed_param_groups.append({'params': [param_or_group]})
        
        # Initialize the base Optimizer class with processed param groups
        super().__init__(processed_param_groups, {})
        
        # Shard parameters across ranks
        self.param_shards = self._shard_parameters()
        
        # Create the underlying optimizer with only this rank's parameters
        self.optimizer = self._create_optimizer()
        
    def _shard_parameters(self) -> List[torch.nn.Parameter]:
        """
        Distribute parameters across ranks.
        
        Returns:
            List of parameters assigned to this rank
        """
        total_params = len(self.all_params)
        
        if total_params == 0:
            return []
        
        # Calculate shard boundaries
        params_per_rank = total_params // self.world_size
        remainder = total_params % self.world_size
        
        # Handle remainder by giving extra params to first few ranks
        if self.rank < remainder:
            start_idx = self.rank * (params_per_rank + 1)
            end_idx = start_idx + params_per_rank + 1
        else:
            start_idx = remainder * (params_per_rank + 1) + (self.rank - remainder) * params_per_rank
            end_idx = start_idx + params_per_rank
        
        # Return this rank's shard
        return self.all_params[start_idx:end_idx]
    
    def _create_optimizer(self) -> Optimizer:
        """
        Create the underlying optimizer with only this rank's parameters.
        
        Returns:
            Optimizer instance for this rank's parameter shard
        """
        if not self.param_shards:
            # If no parameters assigned to this rank, create dummy parameter
            dummy_param = torch.nn.Parameter(torch.tensor(0.0))
            return self.optimizer_cls([dummy_param], **self.optimizer_kwargs)
        
        return self.optimizer_cls(self.param_shards, **self.optimizer_kwargs)
    
    def step(self, closure=None, **kwargs):
        """
        Perform optimizer step and synchronize parameters across ranks.
        
        Args:
            closure: Optional closure function
            **kwargs: Additional arguments passed to the optimizer step
            
        Returns:
            Loss value if closure is provided
        """
        # Perform optimizer step on this rank's parameters
        loss = None
        if closure is not None:
            loss = self.optimizer.step(closure, **kwargs)
        else:
            self.optimizer.step(**kwargs)
        
        # Synchronize updated parameters across all ranks
        self._synchronize_parameters()
        
        return loss
    
    def _synchronize_parameters(self):
        """
        Broadcast updated parameters from each rank to all other ranks.
        """
        if self.world_size <= 1:
            return
        
        # Each rank broadcasts its updated parameters to all other ranks
        for rank in range(self.world_size):
            # Get the parameter shard for this rank
            rank_params = self._get_rank_parameters(rank)
            
            for param in rank_params:
                if param.data.numel() > 0:  # Skip empty parameters
                    # Broadcast from the rank that owns this parameter
                    dist.broadcast(param.data, src=rank)
    
    def _get_rank_parameters(self, rank: int) -> List[torch.nn.Parameter]:
        """
        Get the parameters assigned to a specific rank.
        
        Args:
            rank: Rank to get parameters for
            
        Returns:
            List of parameters assigned to the specified rank
        """
        total_params = len(self.all_params)
        
        if total_params == 0:
            return []
        
        params_per_rank = total_params // self.world_size
        remainder = total_params % self.world_size
        
        if rank < remainder:
            start_idx = rank * (params_per_rank + 1)
            end_idx = start_idx + params_per_rank + 1
        else:
            start_idx = remainder * (params_per_rank + 1) + (rank - remainder) * params_per_rank
            end_idx = start_idx + params_per_rank
        
        return self.all_params[start_idx:end_idx]
    
    def add_param_group(self, param_group: Dict[str, Any]):
        """
        Add a parameter group to the sharded optimizer.
        
        Args:
            param_group: Dictionary containing 'params' key and other optimizer settings
        """
        # Add to base class param_groups
        super().add_param_group(param_group)
        
        # Update our all_params list
        new_params = param_group['params']
        if isinstance(new_params, torch.nn.Parameter):
            new_params = [new_params]
        else:
            new_params = list(new_params)
        
        self.all_params.extend(new_params)
        
        # Re-shard parameters and recreate optimizer
        self.param_shards = self._shard_parameters()
        self.optimizer = self._create_optimizer()
    
    def zero_grad(self, set_to_none: bool = True):
        """
        Zero gradients for this rank's parameters.
        
        Args:
            set_to_none: Whether to set gradients to None instead of zero
        """
        self.optimizer.zero_grad(set_to_none=set_to_none)
    
    def state_dict(self):
        """
        Return the state dictionary of the underlying optimizer.
        
        Returns:
            State dictionary
        """
        return self.optimizer.state_dict()
    
    def load_state_dict(self, state_dict):
        """
        Load state dictionary into the underlying optimizer.
        
        Args:
            state_dict: State dictionary to load
        """
        self.optimizer.load_state_dict(state_dict)
    
    def __getattr__(self, name):
        """
        Delegate attribute access to the underlying optimizer if not found in this class.
        
        Args:
            name: Attribute name
            
        Returns:
            Attribute value from underlying optimizer
        """
        if name in ['optimizer_cls', 'optimizer_kwargs', 'rank', 'world_size', 
                    'all_params', 'param_shards', 'optimizer']:
            return object.__getattribute__(self, name)
        
        if hasattr(self, 'optimizer'):
            return getattr(self.optimizer, name)
        
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")