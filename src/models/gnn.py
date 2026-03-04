import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleGAT(nn.Module):
    """A lightweight Graph Attention layer."""
    def __init__(self, in_dim, hidden_dim=None, heads=1, leaky_relu_neg_slope=0.2):
        super(SimpleGAT, self).__init__()
        hidden_dim = hidden_dim or in_dim
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.W = nn.Linear(in_dim, hidden_dim, bias=False)
        self.a = nn.Linear(hidden_dim * 2, 1, bias=False)
        self.leaky_relu = nn.LeakyReLU(leaky_relu_neg_slope)
        self.out_proj = nn.Linear(hidden_dim, in_dim)

    def forward(self, x, mask=None, top_k=None):
        # x: (B, L, D)
        B, L, D = x.size()
        h = self.W(x)  # (B, L, H)

        h_i = h.unsqueeze(2).expand(-1, -1, L, -1)
        h_j = h.unsqueeze(1).expand(-1, L, -1, -1)
        a_input = torch.cat([h_i, h_j], dim=-1)
        
        e = self.leaky_relu(self.a(a_input).squeeze(-1))

        # === 修复点：FP16 下掩码值不能太小，改用 -1e4 ===
        MASK_VAL = -1e4 

        # Mask self-loops
        diag_mask = torch.eye(L, device=e.device, dtype=torch.bool).unsqueeze(0)
        e = e.masked_fill(diag_mask, MASK_VAL)

        if mask is not None:
            mask_expanded = mask.unsqueeze(1).unsqueeze(2).expand(-1, -1, L)
            mask_key = mask.unsqueeze(1).unsqueeze(1).expand(-1, L, -1)
            combined_mask = mask_expanded & mask_key
            e = e.masked_fill(~combined_mask, MASK_VAL)

        # Top-k sparsity (Logic check: if L <= top_k, we keep all)
        if top_k is not None and top_k < L:
            topk_vals, topk_idx = torch.topk(e, min(top_k, L), dim=-1, largest=True)
            mask_topk = torch.zeros_like(e)
            mask_topk.scatter_(-1, topk_idx, 1.0)
            e = e.masked_fill(mask_topk == 0, MASK_VAL)

        alpha = torch.softmax(e, dim=-1)
        h_prime = torch.matmul(alpha, h)
        refined = self.out_proj(h_prime)
        redundancy_scores = alpha.mean(dim=1)
        redundancy_scores = torch.sigmoid(redundancy_scores)

        return redundancy_scores, refined


class GatedResidualFusion(nn.Module):
    """
    自适应门控残差融合：用于将 GAT 提取的上下文信息融合回原始视图特征。
    """
    def __init__(self, hidden_size):
        super().__init__()
        self.gate_fc = nn.Linear(hidden_size * 2, 1)
        self.sigmoid = nn.Sigmoid()
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, original_feature, refined_feature):
        # 1. 拼接 -> 计算门控系数
        concat = torch.cat([original_feature, refined_feature], dim=-1)
        gate = self.sigmoid(self.gate_fc(concat))
        
        # 2. 门控机制：决定吸收多少 GAT 信息
        gated_refined = gate * refined_feature
        
        # 3. 残差连接 + LayerNorm
        fused = self.norm(original_feature + gated_refined)
        return fused