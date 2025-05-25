# -*- coding: utf-8 -*-
"""
Created on 5-18-2025
Author: Catman Jr.
"""

import torch
from collections.abc import Iterable
import math


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """
    Given a set of parameters, clip their combined gradient so that the L2 norm is at most max_l2_norm.
    
    Args:
        parameters: Collection of trainable parameters
        max_l2_norm: Maximum allowed value for the gradient L2 norm
        
    The gradients (parameter.grad) will be modified in-place.
    """
    # Filter out parameters without gradients
    params_with_grads = [p for p in parameters if p.grad is not None]
    
    if len(params_with_grads) == 0:
        # No gradients to clip
        return
    
    # Calculate the L2 norm of all combined gradients
    total_norm = torch.norm(
        torch.stack([torch.norm(p.grad.detach(), 2) for p in params_with_grads]), 2
    )
    
    # Apply clipping if the norm exceeds the threshold
    if total_norm > max_l2_norm:
        # Calculate the scaling factor
        clip_factor = max_l2_norm / (total_norm + 1e-6)  # Avoid division by zero
        
        # Scale gradients in-place
        for p in params_with_grads:
            p.grad.detach().mul_(clip_factor)


class AdamW(torch.optim.Optimizer):
    """
    Implements the AdamW optimizer, a variant of Adam that correctly applies weight decay.
    
    Parameters:
        params: Iterator of parameters to optimize or iterator of parameter groups
        lr: Learning rate (default: 1e-3)
        betas: Coefficients for computing running averages of gradient and its square (default: (0.9, 0.999))
        eps: Term added to the denominator for numerical stability (default: 1e-8)
        weight_decay: Weight decay coefficient (default: 1e-2)
    """
    def __init__(
        self, 
        params, 
        lr=1e-3, 
        betas=(0.9, 0.999), 
        eps=1e-8, 
        weight_decay=1e-2
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Learning rate must be non-negative, but got {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Epsilon must be non-negative, but got {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"beta1 must be in the interval [0, 1), but got {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"beta2 must be in the interval [0, 1), but got {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Weight decay must be non-negative, but got {weight_decay}")
            
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        
    def step(self, closure=None):
        """Performs a single optimization step.
        
        Args:
            closure (callable, optional): A closure that reevaluates the model and returns the loss
        """
        loss = None
        if closure is not None:
            loss = closure()
            
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                    
                # Extract gradient
                grad = p.grad.data
                
                # Extract parameters
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    # Step count starts from 1
                    state['step'] = 1
                    # Initialize first moment estimate (m)
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Initialize second moment estimate (v)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                else:
                    state['step'] += 1
                    
                # Extract hyperparameters
                beta1, beta2 = group['betas']
                step = state['step']
                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']
                lr = group['lr']
                eps = group['eps']
                weight_decay = group['weight_decay']
                
                # Update momentum estimates
                # m <- β₁m + (1-β₁)g
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                # v <- β₂v + (1-β₂)g²
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # Calculate bias correction
                # α_t <- α * √(1-β₂ᵗ) / (1-β₁ᵗ)
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step
                step_size = lr * math.sqrt(bias_correction2) / bias_correction1
                
                # Update parameters
                # θ <- θ - α_t * m / (√v + ε)
                denom = exp_avg_sq.sqrt().add_(eps)
                p.data.addcdiv_(exp_avg, denom, value=-step_size)
                
                # Apply weight decay
                # θ <- θ - αλθ
                p.data.mul_(1 - lr * weight_decay)
                
        return loss


def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    """
    Calculate learning rate with warmup and cosine decay schedule.
    
    Args:
        it: Current iteration number
        max_learning_rate: α_max, maximum value of the learning rate
        min_learning_rate: α_min, minimum/final value of the learning rate
        warmup_iters: T_w, number of iterations for linear warmup
        cosine_cycle_iters: T_c, number of iterations for the cosine decay cycle
    
    Returns:
        Learning rate for the current iteration
    """
    # Warmup phase: linear increase of learning rate
    if it < warmup_iters:
        # If it's the 0th iteration, learning rate is 0
        if warmup_iters == 0:
            return max_learning_rate
        # t/T_w * α_max
        return (it / warmup_iters) * max_learning_rate
    
    # Cosine annealing phase
    elif it <= cosine_cycle_iters:
        # α_min + 0.5 * (1 + cos((t-T_w)/(T_c-T_w) * π)) * (α_max - α_min)
        cosine_decay = 0.5 * (1 + math.cos(
            math.pi * (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        ))
        return min_learning_rate + cosine_decay * (max_learning_rate - min_learning_rate)
    
    # After annealing phase: maintain minimum learning rate
    else:
        return min_learning_rate