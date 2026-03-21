import torch
import torch.nn as nn
import torch.nn.functional as F


class IntraADR(nn.Module):
    """Intra-Model Attention Diversification
    无参数注意力机制，基于 Eq.(2) of 'Attention Diversification for Domain Generalization'
    """

    def __init__(self, temperature: float = 1.0):
        """Args:
            temperature (float): softmax 温度参数，越大越平滑，越小越锐化。
        """
        super().__init__()
        self.temperature = temperature

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播：对每个通道执行空间 softmax 并加权原特征。"""
        B, C, H, W = x.shape
        x_flat = x.view(B, C, -1)
        attn = F.softmax(x_flat / self.temperature, dim=-1)
        attn_map = attn.view(B, C, H, W)
        return x * attn_map
