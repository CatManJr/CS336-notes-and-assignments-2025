from __future__ import annotations

import os
from typing import Any, Callable, Dict, Literal

import torch
from torch import Tensor
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

# Import our implemented SFT functions
from cs336_alignment.sft_helpers import (
    compute_entropy,
    get_response_log_probs,
    log_generations,
    masked_normalize,
    sft_microbatch_train_step,
    tokenize_prompt_and_output,
)

# Import our implemented GRPO functions
from cs336_alignment.grpo_helpers import (
    compute_group_normalized_rewards,
    compute_grpo_clip_loss,
    compute_naive_policy_gradient_loss,
    compute_policy_gradient_loss,
    grpo_microbatch_train_step,
    masked_mean,
)


def run_tokenize_prompt_and_output(
    prompt_strs, output_strs, tokenizer
) -> Dict[str, Any]:
    """
    This is an adapter function that calls the student's `tokenize_prompt_and_output` implementation.
    """
    return tokenize_prompt_and_output(prompt_strs, output_strs, tokenizer)
    


def run_compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """
    This is an adapter function that calls the student's `compute_entropy` implementation.
    """
    return compute_entropy(logits)
    


def run_get_response_log_probs(
    model, input_ids, labels, return_token_entropy
) -> Dict[str, Any]:
    """
    This is an adapter function that calls the student's `get_response_log_probs` implementation.
    """
    return get_response_log_probs(model, input_ids, labels, return_token_entropy)
    


def run_masked_normalize(tensor, mask, normalize_constant, dim=None) -> torch.Tensor:
    """
    This is an adapter function that calls the student's `masked_normalize` implementation.
    """
    return masked_normalize(tensor, mask, normalize_constant, dim)
    


def run_sft_microbatch_train_step(
    policy_log_probs, response_mask, gradient_accumulation_steps, normalize_constant=1.0
):
    """
    This is an adapter function that calls the student's `sft_microbatch_train_step` implementation.
    """
    return sft_microbatch_train_step(
        policy_log_probs, response_mask, gradient_accumulation_steps, normalize_constant
    )
    


def run_compute_group_normalized_rewards(
    reward_fn,
    rollout_responses,
    repeated_ground_truths,
    group_size,
    advantage_eps,
    normalize_by_std,
):
    """Compute group normalized rewards."""
    return compute_group_normalized_rewards(
        reward_fn=reward_fn,
        rollout_responses=rollout_responses,
        repeated_ground_truths=repeated_ground_truths,
        group_size=group_size,
        advantage_eps=advantage_eps,
        normalize_by_std=normalize_by_std,
    )


def run_grpo_microbatch_train_step(
    policy_log_probs,
    response_mask,
    gradient_accumulation_steps,
    loss_type,
    raw_rewards,
    advantages,
    old_log_probs,
    cliprange,
):
    """GRPO microbatch train step."""
    return grpo_microbatch_train_step(
        policy_log_probs=policy_log_probs,
        response_mask=response_mask,
        gradient_accumulation_steps=gradient_accumulation_steps,
        loss_type=loss_type,
        raw_rewards=raw_rewards,
        advantages=advantages,
        old_log_probs=old_log_probs,
        cliprange=cliprange,
    )


def run_masked_mean(tensor, mask, dim=None):
    """Masked mean computation."""
    return masked_mean(tensor=tensor, mask=mask, dim=dim)


# TODO: #5.1 Metrics
def run_get_rewards(
    responses: list[str],
    repeated_ground_truths: list[str],
    reward_fn: Callable,
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Compute rewards for each group of rollout responses, normalized by the group size."""
    return compute_group_normalized_rewards(
        reward_fn=reward_fn,
        rollout_responses=responses,
        repeated_ground_truths=repeated_ground_truths,
        group_size=group_size,
        advantage_eps=advantage_eps,
        normalize_by_std=normalize_by_std,
    )


def run_compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
) -> torch.Tensor:
    """Compute policy gradient loss using either raw rewards or advantages."""
    return compute_naive_policy_gradient_loss(
        raw_rewards_or_advantages=raw_rewards_or_advantages,
        policy_log_probs=policy_log_probs,
    )


def run_compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the GRPO-Clip loss."""
    return compute_grpo_clip_loss(
        advantages=advantages,
        policy_log_probs=policy_log_probs,
        old_log_probs=old_log_probs,
        cliprange=cliprange,
    )


def run_compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: str,
    raw_rewards: torch.Tensor,
    advantages: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Wrapper that delegates to the appropriate policy gradient loss function above."""
    return compute_policy_gradient_loss(
        policy_log_probs=policy_log_probs,
        loss_type=loss_type,
        raw_rewards=raw_rewards,
        advantages=advantages,
        old_log_probs=old_log_probs,
        cliprange=cliprange,
    )


"""
The below adapters are used in the optional 
RLHF / safety part of the Alignment assignment.
"""


def get_packed_sft_dataset(
    tokenizer: PreTrainedTokenizerBase,
    dataset_path: str | os.PathLike,
    seq_length: int,
    shuffle: bool,
) -> Dataset:
    """Construct a PyTorch Dataset for language modeling with packed sequences."""
    raise NotImplementedError("Safety/RLHF functions are optional")


def run_iterate_batches(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
):
    """Return an iterable over batches of size `batch_size`."""
    raise NotImplementedError("Safety/RLHF functions are optional")


def run_parse_mmlu_response(
    mmlu_example: dict[str, Any],
    model_output: str,
) -> str | None:
    """Parse MMLU model output into a predicted option letter."""
    # Import the parsing function from our baseline script
    from cs336_alignment.benchmark.evaluate_mmlu_baseline import parse_mmlu_response
    return parse_mmlu_response(model_output)


def run_parse_gsm8k_response(
    model_output: str,
) -> str | None:
    """Parse GSM8K model output into a predicted numeric answer."""
    # Import the parsing function from our baseline script
    from cs336_alignment.benchmark.evaluate_gsm8k_baseline import parse_gsm8k_response
    return parse_gsm8k_response(model_output)


def run_compute_per_instance_dpo_loss(
    lm: torch.nn.Module,
    lm_ref: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    beta: float,
    prompt: str,
    response_chosen: str,
    response_rejected: str,
) -> torch.Tensor:
    """Compute the value of the DPO loss for this example."""
    raise NotImplementedError("Safety/RLHF functions are optional")
