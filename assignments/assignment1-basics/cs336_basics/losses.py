import torch

def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    计算交叉熵损失，具有数值稳定性保证。
    
    Args:
        logits: 预测的对数 (logits)，形状为 (..., vocab_size)
        targets: 目标类别索引，形状为 (...)
    
    Returns:
        标量损失值，所有样本的平均交叉熵
    """
    # 保存原始形状
    input_shape = logits.shape
    # 重塑输入为二维，将所有批次维度合并为第一维
    if len(input_shape) > 2:
        logits = logits.reshape(-1, input_shape[-1])
        targets = targets.reshape(-1)
    
    # 为数值稳定性，减去每个样本的最大logit值
    logits_max = torch.max(logits, dim=-1, keepdim=True)[0]
    logits_shifted = logits - logits_max
    
    # 计算 log_softmax = log(exp(logits) / sum(exp(logits)))
    # 简化为: logits - log(sum(exp(logits)))
    exp_logits = torch.exp(logits_shifted)
    log_sum_exp = torch.log(torch.sum(exp_logits, dim=-1, keepdim=True))
    log_softmax = logits_shifted - log_sum_exp
    
    # 提取每个样本对应目标类别的log_softmax值
    batch_indices = torch.arange(log_softmax.shape[0], device=logits.device)
    target_log_probs = log_softmax[batch_indices, targets]
    
    # 交叉熵 = -log(p(target))，取均值为批次损失
    return -target_log_probs.mean()