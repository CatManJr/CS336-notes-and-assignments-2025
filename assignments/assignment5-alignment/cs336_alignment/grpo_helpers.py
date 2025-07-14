"""
GRPO (Group Relative Policy Optimization) helper functions.
"""

import torch
from typing import List, Dict, Any, Callable, Tuple, Optional
import numpy as np


def compute_group_normalized_rewards(
    reward_fn: Callable[[str, str], Dict[str, float]],
    rollout_responses: List[str],
    repeated_ground_truths: List[str],
    group_size: int,
    advantage_eps: float = 1e-8,
    normalize_by_std: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    """
    Compute rewards for each group of rollout responses, normalized by the group size.
    
    This implements the group normalization from Equation 28 and the simplified version
    from Dr. GRPO (Equation 31).
    
    Args:
        reward_fn: Callable that scores rollout responses against ground truths
        rollout_responses: List of rollout responses from the policy
        repeated_ground_truths: List of ground truths (repeated group_size times)
        group_size: Number of responses per question (group)
        advantage_eps: Small constant to avoid division by zero
        normalize_by_std: If True, normalize by std; otherwise just subtract mean
        
    Returns:
        advantages: Group-normalized rewards (advantages) for each response
        raw_rewards: Unnormalized rewards for each response
        metadata: Statistics for logging
    """
    
    # Compute raw rewards for all responses
    raw_rewards = []
    format_rewards = []
    answer_rewards = []
    
    for response, ground_truth in zip(rollout_responses, repeated_ground_truths):
        try:
            reward_dict = reward_fn(response, ground_truth)
            raw_rewards.append(reward_dict.get('reward', 0.0))
            format_rewards.append(reward_dict.get('format_reward', 0.0))
            answer_rewards.append(reward_dict.get('answer_reward', 0.0))
        except Exception as e:
            # Handle cases where reward function fails
            raw_rewards.append(0.0)
            format_rewards.append(0.0)
            answer_rewards.append(0.0)
    
    # Convert to tensors
    raw_rewards_tensor = torch.tensor(raw_rewards, dtype=torch.float32)
    
    # Reshape into groups: (num_groups, group_size)
    num_groups = len(rollout_responses) // group_size
    raw_rewards_grouped = raw_rewards_tensor.view(num_groups, group_size)
    
    # Compute group statistics
    group_means = raw_rewards_grouped.mean(dim=1, keepdim=True)  # (num_groups, 1)
    
    # Compute advantages (group-normalized rewards)
    advantages_grouped = raw_rewards_grouped - group_means  # (num_groups, group_size)
    
    if normalize_by_std:
        # Standard GRPO normalization (Equation 28)
        group_stds = raw_rewards_grouped.std(dim=1, keepdim=True)  # (num_groups, 1)
        advantages_grouped = advantages_grouped / (group_stds + advantage_eps)
    # else: Dr. GRPO simplification (Equation 31) - just subtract mean
    
    # Flatten back to original shape
    advantages = advantages_grouped.view(-1)  # (rollout_batch_size,)
    
    # Compute metadata for logging
    metadata = {
        'mean_reward': float(raw_rewards_tensor.mean()),
        'std_reward': float(raw_rewards_tensor.std()),
        'max_reward': float(raw_rewards_tensor.max()),
        'min_reward': float(raw_rewards_tensor.min()),
        'mean_format_reward': float(np.mean(format_rewards)),
        'mean_answer_reward': float(np.mean(answer_rewards)),
        'num_groups': num_groups,
        'group_size': group_size,
        'normalize_by_std': normalize_by_std
    }
    
    return advantages, raw_rewards_tensor, metadata


def masked_mean(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: Optional[int] = None
) -> torch.Tensor:
    """
    Compute masked mean of a tensor.
    
    Args:
        tensor: Input tensor
        mask: Boolean mask tensor (same shape as tensor)
        dim: Dimension to compute mean over (None for all dimensions)
        
    Returns:
        Masked mean tensor
    """
    # Apply mask
    masked_tensor = tensor * mask.float()
    
    if dim is None:
        # Mean over all dimensions
        num_masked = mask.float().sum()
        if num_masked == 0:
            return torch.tensor(float('nan'), dtype=tensor.dtype, device=tensor.device)
        return masked_tensor.sum() / num_masked
    else:
        # Mean over specific dimension
        numerator = masked_tensor.sum(dim=dim)
        denominator = mask.float().sum(dim=dim)
        
        # Return NaN where denominator is 0, otherwise compute mean
        result = numerator / denominator
        result = torch.where(denominator == 0, torch.tensor(float('nan'), dtype=tensor.dtype, device=tensor.device), result)
        return result


def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
) -> torch.Tensor:
    """
    Compute naive policy gradient loss (REINFORCE) per token.
    
    Implements Equation 32: -A_t * log p_θ(o_t | q, o_<t)
    
    Args:
        raw_rewards_or_advantages: Rewards or advantages, shape (batch_size, 1)
        policy_log_probs: Log probabilities from the policy, shape (batch_size, sequence_length)
        
    Returns:
        Per-token policy gradient loss, shape (batch_size, sequence_length)
    """
    # raw_rewards_or_advantages shape: (batch_size, 1)
    # policy_log_probs shape: (batch_size, sequence_length)
    
    # Broadcast rewards/advantages over sequence length dimension
    # The shape (batch_size, 1) will automatically broadcast to (batch_size, sequence_length)
    per_token_loss = -raw_rewards_or_advantages * policy_log_probs
    
    return per_token_loss


def compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float = 0.2,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Compute GRPO clipped loss (Equation 33) per token.
    
    Implements: -min(ratio * A, clip(ratio, 1-ε, 1+ε) * A)
    where ratio = π_θ(o_t|q,o_<t) / π_θ_old(o_t|q,o_<t)
    
    Args:
        advantages: Group-normalized advantages, shape (batch_size, 1)
        policy_log_probs: Log probabilities from current policy, shape (batch_size, sequence_length)
        old_log_probs: Log probabilities from old policy, shape (batch_size, sequence_length)
        cliprange: Clipping parameter ε
        
    Returns:
        loss: Per-token GRPO clipped loss, shape (batch_size, sequence_length)
        metadata: Additional statistics
    """
    # Compute probability ratios: π_θ / π_θ_old
    log_ratios = policy_log_probs - old_log_probs
    ratios = torch.exp(log_ratios)
    
    # advantages shape: (batch_size, 1) - already correct for broadcasting
    # ratios shape: (batch_size, sequence_length)
    # Broadcasting will work automatically: (batch_size, 1) * (batch_size, sequence_length) -> (batch_size, sequence_length)
    
    # Compute unclipped objective: ratio * advantage
    unclipped_objective = ratios * advantages
    
    # Compute clipped objective: clip(ratio, 1-ε, 1+ε) * advantage  
    clipped_ratios = torch.clamp(ratios, 1 - cliprange, 1 + cliprange)
    clipped_objective = clipped_ratios * advantages
    
    # Take minimum (GRPO-Clip mechanism)
    per_token_objective = torch.min(unclipped_objective, clipped_objective)
    
    # Compute per-token loss (negative because we want to maximize the objective)
    per_token_loss = -per_token_objective
    
    # Compute metadata
    # Check which tokens were clipped
    clipped_upper = (ratios > (1 + cliprange)).float()
    clipped_lower = (ratios < (1 - cliprange)).float()
    clipped_total = clipped_upper + clipped_lower
    
    metadata = {
        'mean_ratio': ratios.mean(),
        'max_ratio': ratios.max(),
        'min_ratio': ratios.min(),
        'clipped_fraction': clipped_total.mean(),
        'clipped_upper_fraction': clipped_upper.mean(),
        'clipped_lower_fraction': clipped_lower.mean(),
        'mean_advantage': advantages.mean(),
        'std_advantage': advantages.std(),
    }
    
    return per_token_loss, metadata


def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: str,
    raw_rewards: torch.Tensor = None,
    advantages: torch.Tensor = None,
    old_log_probs: torch.Tensor = None,
    cliprange: float = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Wrapper function that delegates to appropriate policy gradient loss function.
    
    Args:
        policy_log_probs: Log probabilities from current policy, shape (batch_size, sequence_length)
        loss_type: Type of loss ("no_baseline", "reinforce_with_baseline", "grpo_clip")
        raw_rewards: Raw rewards, shape (batch_size, 1) - required for "no_baseline"
        advantages: Group-normalized advantages, shape (batch_size, 1) - required for others
        old_log_probs: Log probabilities from old policy - required for "grpo_clip"
        cliprange: Clipping parameter - required for "grpo_clip"
        
    Returns:
        loss: Per-token loss, shape (batch_size, sequence_length)
        metadata: Additional statistics
    """
    if loss_type == "no_baseline":
        # Argument checks
        assert raw_rewards is not None, "raw_rewards required for no_baseline"
        
        # Use raw rewards without baseline
        per_token_loss = compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs)
        metadata = {"loss_type": loss_type}
        
    elif loss_type == "reinforce_with_baseline":
        # Argument checks  
        assert advantages is not None, "advantages required for reinforce_with_baseline"
        
        # Use advantages (which are rewards minus baseline)
        per_token_loss = compute_naive_policy_gradient_loss(advantages, policy_log_probs)
        metadata = {"loss_type": loss_type}
        
    elif loss_type == "grpo_clip":
        # Argument checks
        assert advantages is not None, "advantages required for grpo_clip"
        assert old_log_probs is not None, "old_log_probs required for grpo_clip"
        assert cliprange is not None, "cliprange required for grpo_clip"
        
        # Use GRPO clipped loss
        per_token_loss, clip_metadata = compute_grpo_clip_loss(
            advantages, policy_log_probs, old_log_probs, cliprange
        )
        metadata = {"loss_type": loss_type, **clip_metadata}
        
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")
    
    return per_token_loss, metadata


def grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: str,
    raw_rewards: torch.Tensor = None,
    advantages: torch.Tensor = None,
    old_log_probs: torch.Tensor = None,
    cliprange: float = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Execute a single GRPO microbatch training step.
    
    Args:
        policy_log_probs: Log probabilities from current policy, shape (batch_size, sequence_length)
        response_mask: Mask for response tokens, shape (batch_size, sequence_length)
        gradient_accumulation_steps: Number of gradient accumulation steps
        loss_type: Type of loss to compute
        raw_rewards: Raw rewards, shape (batch_size, 1) - needed for "no_baseline"
        advantages: Group-normalized advantages, shape (batch_size, 1) - needed for others
        old_log_probs: Log probabilities from old policy - needed for "grpo_clip"
        cliprange: Clipping parameter - needed for "grpo_clip"
        
    Returns:
        loss: Scaled loss for gradient accumulation
        metadata: Additional statistics
    """
    
    # Compute per-token policy gradient loss
    per_token_loss, loss_metadata = compute_policy_gradient_loss(
        policy_log_probs=policy_log_probs,
        loss_type=loss_type,
        raw_rewards=raw_rewards,
        advantages=advantages,
        old_log_probs=old_log_probs,
        cliprange=cliprange
    )
    
    # Use masked_mean to aggregate per-token losses to per-example losses
    # then average over the batch dimension
    per_example_loss = masked_mean(per_token_loss, response_mask, dim=1)  # Average over sequence length
    batch_loss = per_example_loss.mean()  # Average over batch dimension
    
    # Scale loss for gradient accumulation
    scaled_loss = batch_loss / gradient_accumulation_steps
    
    # Call backward
    scaled_loss.backward()
    
    # Prepare metadata
    metadata = {
        "loss": scaled_loss.item(),
        "batch_loss": batch_loss.item(),
        **loss_metadata
    }
    
    return scaled_loss, metadata