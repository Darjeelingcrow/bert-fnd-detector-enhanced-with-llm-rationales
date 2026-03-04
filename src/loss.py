import torch
import torch.nn as nn
import torch.nn.functional as F


# Supervised Contrastive Loss for (bs, dim) embeddings
class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # features: [B, D], labels: [B]
        device = features.device
        B = features.size(0)
        if B <= 1:
            return torch.tensor(0.0, device=device)

        feats = F.normalize(features, p=2, dim=1)
        sim = torch.matmul(feats, feats.t())  # [B, B]

        labels = labels.view(-1, 1)
        pos_mask = (labels == labels.t()).float().to(device)
        # remove self-comparisons
        logits_mask = torch.ones_like(pos_mask) - torch.eye(B, device=device)
        pos_mask = pos_mask * logits_mask

        logits = sim / self.temperature
        exp_logits = torch.exp(logits) * logits_mask  # zero out diagonal

        # log_prob for all pairs
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

        # mean over positives per anchor
        pos_counts = pos_mask.sum(dim=1)
        # avoid div by zero: anchors without positives contribute 0
        mean_log_pos = (pos_mask * log_prob).sum(dim=1) / (pos_counts + 1e-12)

        loss = -mean_log_pos.mean()
        return loss
    

def dirichlet_kl(alpha, beta):
    """
    KL( Dir(alpha) || Dir(beta) )
    alpha, beta: [B, K]
    """
    S_alpha = alpha.sum(dim=-1, keepdim=True)    # [B,1]
    S_beta  = beta.sum(dim=-1, keepdim=True)     # [B,1]

    # log Γ(·) 用 torch.lgamma，ψ 用 torch.digamma
    lgamma_sum_alpha = torch.lgamma(S_alpha)
    lgamma_sum_beta  = torch.lgamma(S_beta)
    lgamma_alpha = torch.lgamma(alpha)
    lgamma_beta  = torch.lgamma(beta)

    # 第一块：log Γ(Σα) - Σ log Γ(α_i)
    term1 = lgamma_sum_alpha - lgamma_alpha.sum(dim=-1, keepdim=True)

    # 第二块：-log Γ(Σβ) + Σ log Γ(β_i)
    term2 = -lgamma_sum_beta + lgamma_beta.sum(dim=-1, keepdim=True)

    # 第三块：Σ (α_i - β_i) [ψ(α_i) - ψ(Σα)]
    digamma_alpha = torch.digamma(alpha)
    digamma_sum_alpha = torch.digamma(S_alpha)
    term3 = ((alpha - beta) * (digamma_alpha - digamma_sum_alpha)).sum(dim=-1, keepdim=True)

    kl = term1 + term2 + term3           # [B,1]
    return kl.squeeze(-1) 