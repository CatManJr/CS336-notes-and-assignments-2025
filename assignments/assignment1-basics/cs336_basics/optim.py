import torch
from collections.abc import Iterable
import math


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """
    给定一组参数，裁剪它们的组合梯度，使L2范数最多为 max_l2_norm。
    
    Args:
        parameters: 可训练参数的集合
        max_l2_norm: 梯度 L2 范数的最大允许值
        
    梯度 (parameter.grad) 将被原地修改。
    """
    # 过滤掉没有梯度的参数
    params_with_grads = [p for p in parameters if p.grad is not None]
    
    if len(params_with_grads) == 0:
        # 没有需要裁剪的梯度
        return
    
    # 计算所有梯度组合的 L2 范数
    total_norm = torch.norm(
        torch.stack([torch.norm(p.grad.detach(), 2) for p in params_with_grads]), 2
    )
    
    # 如果范数超过阈值，则应用裁剪
    if total_norm > max_l2_norm:
        # 计算缩放因子
        clip_factor = max_l2_norm / (total_norm + 1e-6)  # 避免除零
        
        # 原地缩放梯度
        for p in params_with_grads:
            p.grad.detach().mul_(clip_factor)


class AdamW(torch.optim.Optimizer):
    """
    实现 AdamW 优化器，这是 Adam 的一个变体，正确应用权重衰减。
    
    参数:
        params: 要优化的参数迭代器或参数组的迭代器
        lr: 学习率 (默认: 1e-3)
        betas: 用于计算梯度及其平方的运行平均值的系数 (默认: (0.9, 0.999))
        eps: 添加到分母以提高数值稳定性的项 (默认: 1e-8)
        weight_decay: 权重衰减系数 (默认: 1e-2)
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
            raise ValueError(f"学习率必须非负，但得到了 {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"epsilon 必须非负，但得到了 {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"beta1 必须在 [0, 1) 区间内，但得到了 {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"beta2 必须在 [0, 1) 区间内，但得到了 {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"权重衰减必须非负，但得到了 {weight_decay}")
            
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        
    def step(self, closure=None):
        """执行单个优化步骤。
        
        参数:
            closure (callable, 可选): 重新评估模型并返回损失的闭包
        """
        loss = None
        if closure is not None:
            loss = closure()
            
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                    
                # 提取梯度
                grad = p.grad.data
                
                # 提取参数
                state = self.state[p]
                
                # 状态初始化
                if len(state) == 0:
                    # 步数从1开始
                    state['step'] = 1
                    # 初始化一阶动量估计 (m)
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # 初始化二阶动量估计 (v)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                else:
                    state['step'] += 1
                    
                # 提取超参数
                beta1, beta2 = group['betas']
                step = state['step']
                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']
                lr = group['lr']
                eps = group['eps']
                weight_decay = group['weight_decay']
                
                # 更新动量估计
                # m <- β₁m + (1-β₁)g
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                # v <- β₂v + (1-β₂)g²
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # 计算偏差修正
                # α_t <- α * √(1-β₂ᵗ) / (1-β₁ᵗ)
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step
                step_size = lr * math.sqrt(bias_correction2) / bias_correction1
                
                # 更新参数
                # θ <- θ - α_t * m / (√v + ε)
                denom = exp_avg_sq.sqrt().add_(eps)
                p.data.addcdiv_(exp_avg, denom, value=-step_size)
                
                # 应用权重衰减
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
    计算带预热的余弦学习率衰减调度。
    
    Args:
        it: 当前迭代次数
        max_learning_rate: α_max, 学习率的最大值
        min_learning_rate: α_min, 学习率的最小/最终值
        warmup_iters: T_w, 线性预热的迭代次数
        cosine_cycle_iters: T_c, 余弦衰减周期的迭代次数
    
    Returns:
        当前迭代的学习率
    """
    # 预热阶段：线性增加学习率
    if it < warmup_iters:
        # 如果是第0次迭代，学习率为0
        if warmup_iters == 0:
            return max_learning_rate
        # t/T_w * α_max
        return (it / warmup_iters) * max_learning_rate
    
    # 余弦退火阶段
    elif it <= cosine_cycle_iters:
        # α_min + 0.5 * (1 + cos((t-T_w)/(T_c-T_w) * π)) * (α_max - α_min)
        cosine_decay = 0.5 * (1 + math.cos(
            math.pi * (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        ))
        return min_learning_rate + cosine_decay * (max_learning_rate - min_learning_rate)
    
    # 退火后阶段：保持最小学习率
    else:
        return min_learning_rate