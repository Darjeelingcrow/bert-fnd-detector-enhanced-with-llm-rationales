import torch
import torch.nn.functional as F
from torch import nn


class _LightweightConvBlock(nn.Module):
    """
    轻量级深度可分离卷积块（支持GLU门控）
    输入/输出均为 (seq_len, batch, embed_dim) 以兼容现有调用。
    结构: 1x1扩展 -> GLU -> 深度可分离卷积 -> GELU -> 1x1回投 -> Dropout -> 残差 + LayerNorm
    """
    def __init__(self, embed_dim: int, expansion: int = 2, kernel_size: int = 3, dropout: float = 0.1, use_glu: bool = True):
        super().__init__()
        self.embed_dim = embed_dim
        self.dropout = nn.Dropout(dropout)
        self.use_glu = use_glu

        hidden = expansion * embed_dim
        in_channels = embed_dim
        # 先做1x1扩展；若启用GLU则扩展为2倍通道，随后切分为门控
        self.pre = nn.Conv1d(in_channels, hidden * (2 if use_glu else 1), kernel_size=1, bias=False)
        # 深度可分离卷积（按通道分组）
        self.dw = nn.Conv1d(hidden, hidden, kernel_size=kernel_size, padding=kernel_size // 2, groups=hidden, bias=False)
        # 1x1回投至原维度
        self.post = nn.Conv1d(hidden, embed_dim, kernel_size=1, bias=False)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: (seq_len, batch, embed_dim)
        residual = x
        x = x.transpose(0, 1)  # (batch, seq_len, embed_dim)
        x = x.transpose(1, 2)  # (batch, embed_dim, seq_len)

        y = self.pre(x)
        if self.use_glu:
            a, b = torch.chunk(y, 2, dim=1)
            y = a * torch.sigmoid(b)
        else:
            y = y

        y = self.dw(y)
        y = F.gelu(y)
        y = self.post(y)  # (batch, embed_dim, seq_len)
        y = self.dropout(y)

        y = y.transpose(1, 2)  # (batch, seq_len, embed_dim)
        y = y + residual.transpose(0, 1)  # 残差 (batch, seq_len, embed_dim)
        y = self.norm(y)

        # 返回到原始接口形状: (seq_len, batch, embed_dim)
        y = y.transpose(0, 1)
        return y


class LightweightConvEncoder(nn.Module):
    """
    轻量级CNN序列编码器（替代TransformerEncoder）
    - 维持输入/输出: (seq_len, batch, embed_dim)
    - 不使用自注意力/掩码，显著降低参数与计算量，减轻过拟合风险
    """
    def __init__(
        self,
        embed_dim: int,
        layers: int = 2,
        expansion: int = 2,
        kernel_size: int = 3,
        dropout: float = 0.1,
        use_glu: bool = True,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            _LightweightConvBlock(
                embed_dim=embed_dim,
                expansion=expansion,
                kernel_size=kernel_size,
                dropout=dropout,
                use_glu=use_glu,
            )
            for _ in range(layers)
        ])

    def forward(self, x):
        # x: (seq_len, batch, embed_dim)
        for blk in self.layers:
            x = blk(x)
        return x
