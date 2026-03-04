import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import math
# from timm.models.vision_transformer import Block

class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, input_, alpha):
        ctx.alpha = alpha
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class MLP(torch.nn.Module):

    def __init__(self, input_dim, embed_dims, dropout, output_dim=1, output_layer=True):
        super().__init__()
        layers = list()
        for embed_dim in embed_dims:
            layers.append(torch.nn.Linear(input_dim, embed_dim))
            #layers.append(torch.nn.BatchNorm1d(embed_dim))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(p=dropout))
            input_dim = embed_dim
        if output_layer:
            layers.append(torch.nn.Linear(input_dim, output_dim))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, embed_dim)``
        """
        return self.mlp(x)

class cnn_extractor(nn.Module):
    def __init__(self, feature_kernel, input_size):
        super(cnn_extractor, self).__init__()
        self.convs = torch.nn.ModuleList(
            [torch.nn.Conv1d(input_size, feature_num, kernel)
             for kernel, feature_num in feature_kernel.items()])
        input_shape = sum([feature_kernel[kernel] for kernel in feature_kernel])

    def forward(self, input_data):
        share_input_data = input_data.permute(0, 2, 1)
        feature = [conv(share_input_data) for conv in self.convs]
        feature = [torch.max_pool1d(f, f.shape[-1]) for f in feature]
        feature = torch.cat(feature, dim=1)
        feature = feature.view([-1, feature.shape[1]])
        return feature

class MaskAttention(torch.nn.Module):
    """
    Compute attention layer
    """
    def __init__(self, input_shape):
        super(MaskAttention, self).__init__()
        self.attention_layer = torch.nn.Linear(input_shape, 1)

    def forward(self, inputs, mask=None):
        scores = self.attention_layer(inputs).view(-1, inputs.size(1))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        scores = torch.softmax(scores, dim=-1).unsqueeze(1)
        outputs = torch.matmul(scores, inputs).squeeze(1)

        return outputs, scores


class LightweightAttentionPooling(nn.Module):
    """
    轻量注意力池化 (LightweightAttentionPooling)

    用法示例:

        # x: (L, B, D), mask: (B, L), 1 表示有效位置, 0 表示 padding
        pool = LightweightAttentionPooling(input_dim=D, hidden_dim=None, dropout=0.1, temperature=1.0)
        pooled, attn = pool(x, mask)  # pooled: (B, D), attn: (B, 1, L)

    替换建议:
        - 将原先的均值池化/取 CLS 向量替换为本模块, 可在小数据场景下降低过拟合, 同时保留轻量的可学习权重。
        - 与已有 MaskAttention 类似, 但额外支持温度缩放、可选两层轻量打分网络、可选 LayerNorm, 更稳健。

    参数说明:
        - input_dim: 输入最后维度 D
        - hidden_dim: 若为 None 则使用线性到 1 的最轻量打分; 否则使用 D->hidden->1 的两层网络
        - dropout: 对注意力权重的 dropout (训练期生效)
        - temperature: 温度系数, >1 更平滑, <1 更尖锐
        - use_layernorm: 是否在打分前对 token 向量做 LayerNorm
    """
    """
    轻量注意力池化: 将 (B, L, D) 池化为 (B, D), 同时返回注意力权重 (B, 1, L)。
    设计目标: 极少参数、稳定、可控 (温度缩放 / LayerNorm / 两层打分)。
    """
    def __init__(self, input_dim: int, hidden_dim: int | None = None, dropout: float = 0.0,
                 temperature: float = 1.0, use_layernorm: bool = True):
        super().__init__()
        self.temperature = float(temperature) if temperature is not None else 1.0
        self.use_ln = use_layernorm
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()
        self.ln = nn.LayerNorm(input_dim) if self.use_ln else nn.Identity()

        if hidden_dim is None:
            # 最轻量: 线性到标量分数
            self.score = nn.Linear(input_dim, 1)
        else:
            # 两层轻量MLP, 更表达但仍很小
            self.score = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        """
        x: (L, B, D)
        mask: (B, L), 1 有效, 0 为 padding
        return: pooled (B, D), attn (B, 1, L)
        """
        L, B, D = x.shape
        x_bld = x.transpose(0, 1)  # (B, L, D)
        h = self.ln(x_bld)
        scores = self.score(h).squeeze(-1)  # (B, L)
        if self.temperature != 1.0:
            scores = scores / self.temperature

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = torch.softmax(scores, dim=-1)  # (B, L)
        # 对注意力权重进行 dropout 更偏正则化
        attn = self.dropout(attn)
        attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-12)  # 重新归一化
        attn_exp = attn.unsqueeze(1)  # (B, 1, L)

        pooled = torch.bmm(attn_exp, x_bld).squeeze(1)  # (B, D)
        return pooled, attn_exp

class Attention(torch.nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """

    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(query.size(-1))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(torch.nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0

        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h

        self.linear_layers = torch.nn.ModuleList([torch.nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = torch.nn.Linear(d_model, d_model)
        self.attention = Attention()

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        if mask is not None:
            mask = mask.repeat(1, self.h, 1, 1)
        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)
        # print('x shape after self attention: {}'.format(x.shape))

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.output_linear(x), attn

class SelfAttentionFeatureExtract(torch.nn.Module):
    def __init__(self, multi_head_num, input_size, output_size=None):
        super(SelfAttentionFeatureExtract, self).__init__()
        self.attention = MultiHeadedAttention(multi_head_num, input_size)
    def forward(self, inputs, query, mask=None):
        if mask is not None:
            mask = mask.view(mask.size(0), 1, 1, mask.size(-1))

        feature, attn = self.attention(query=query,
                                 value=inputs,
                                 key=inputs,
                                 mask=mask
                                 )
        return feature, attn

def masked_softmax(scores, mask):
    """Apply source length masking then softmax.
    Input and output have shape bsz x src_len"""

    # Fill pad positions with -inf
    scores = scores.masked_fill(mask == 0, -np.inf)
 
    # Cast to float and then back again to prevent loss explosion under fp16.
    return F.softmax(scores.float(), dim=-1).type_as(scores)
 
class ParallelCoAttentionNetwork(nn.Module):
 
    def __init__(self, hidden_dim, co_attention_dim, mask_in=False):
        super(ParallelCoAttentionNetwork, self).__init__()
 
        self.hidden_dim = hidden_dim
        self.co_attention_dim = co_attention_dim
        self.mask_in = mask_in
        # self.src_length_masking = src_length_masking
 
        # [hid_dim, hid_dim]
        self.W_b = nn.Parameter(torch.randn(self.hidden_dim, self.hidden_dim))
        # [co_dim, hid_dim]
        self.W_v = nn.Parameter(torch.randn(self.co_attention_dim, self.hidden_dim))
        # [co_dim, hid_dim]
        self.W_q = nn.Parameter(torch.randn(self.co_attention_dim, self.hidden_dim))
        # [co_dim, 1]
        self.w_hv = nn.Parameter(torch.randn(self.co_attention_dim, 1))
        # [co_dim, 1]
        self.w_hq = nn.Parameter(torch.randn(self.co_attention_dim, 1))
 
    def forward(self, V, Q, V_mask=None, Q_mask=None):
        """ ori_setting
        :param V: batch_size * hidden_dim * region_num, eg B x 512 x 196
        :param Q: batch_size * seq_len * hidden_dim, eg B x L x 512
        :param Q_lengths: batch_size
        :return:batch_size * 1 * region_num, batch_size * 1 * seq_len,
        batch_size * hidden_dim, batch_size * hidden_dim
        """
        """ new_setting
        :param V: news content, batch_size * hidden_dim * content_length , eg B x 768 x 170
        :param Q: FTR info, batch_size * FTR_length * hidden_dim, eg B x 512 x 768
        :param batch_size: batch_size
        :return:batch_size * 1 * region_num, batch_size * 1 * seq_len,
        batch_size * hidden_dim, batch_size * hidden_dim
        """

        C = torch.matmul(Q, torch.matmul(self.W_b, V))
        # (batch_size, co_attention_dim, region_num)
        H_v = nn.Tanh()(torch.matmul(self.W_v, V) + torch.matmul(torch.matmul(self.W_q, Q.permute(0, 2, 1)), C))
        # (batch_size, co_attention_dim, seq_len)
        H_q = nn.Tanh()(
            torch.matmul(self.W_q, Q.permute(0, 2, 1)) + torch.matmul(torch.matmul(self.W_v, V), C.permute(0, 2, 1)))
 
        # (batch_size, 1, region_num)
        a_v = F.softmax(torch.matmul(torch.t(self.w_hv), H_v), dim=2)
        # (batch_size, 1, seq_len)
        a_q = F.softmax(torch.matmul(torch.t(self.w_hq), H_q), dim=2)

        if self.mask_in:
            # # (batch_size, 1, region_num)
            masked_a_v = masked_softmax(
                a_v.squeeze(1), V_mask
            ).unsqueeze(1)
    
            # # (batch_size, 1, seq_len)
            masked_a_q = masked_softmax(
                a_q.squeeze(1), Q_mask
            ).unsqueeze(1)
 
            # (batch_size, hidden_dim)
            v = torch.squeeze(torch.matmul(masked_a_v, V.permute(0, 2, 1)))
            # (batch_size, hidden_dim)
            q = torch.squeeze(torch.matmul(masked_a_q, Q))
    
            return masked_a_v, masked_a_q, v, q
        else:
            # (batch_size, hidden_dim)
            v = torch.squeeze(torch.matmul(a_v, V.permute(0, 2, 1)))
            # (batch_size, hidden_dim)
            q = torch.squeeze(torch.matmul(a_q, Q))
    
            return a_v, a_q, v, q
        

class EvidentialUsefulnessHead(nn.Module):
    def __init__(self, in_dim, hidden_dim=256, num_classes=2):
        super().__init__()
        self.layer_norm = nn.LayerNorm(in_dim)   # 可选
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, z):
        """
        z: [batch_size, in_dim]  这里就是 f_{x→t}
        return: α, ŷ, u
        """
        x = self.layer_norm(z)          # 可选
        x = self.fc1(x)
        x = F.gelu(x)                   # 或 F.relu
        x = self.dropout(x)
        alpha_raw = self.fc2(x)         # shape: [B, K]

        # 关键：softplus+1，保证 alpha_k > 1
        alpha = F.softplus(alpha_raw) + 1.0   # [B, K]

        S = torch.sum(alpha, dim=-1, keepdim=True)  # [B,1]
        y_hat = alpha / S                           # 期望概率 ŷ
        u = alpha.size(-1) / S                     # 不确定度 u = K/S

        return alpha, y_hat, u
    
class multimodal_attention(nn.Module):
    """
    dot-product attention mechanism
    """

    def __init__(self, attention_dropout=0.5):
        super(multimodal_attention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, scale=None, attn_mask=None):

        attention = torch.matmul(q, k.transpose(-2, -1))
        # print('attention.shape:{}'.format(attention.shape))
        if scale:
            attention = attention * scale

        if attn_mask:
            attention = attention.masked_fill_(attn_mask, -np.inf)
            
        attention = self.softmax(attention)
        # print('attention.shftmax:{}'.format(attention))
        attention = self.dropout(attention)
        v_result = torch.matmul(attention, v)
        # print('attn_final.shape:{}'.format(attention.shape))

        return v_result
    
class CrossAttention(nn.Module):
    """
    Multi-Head Cross Attention mechanism
    """

    def __init__(self, model_dim=768, num_heads=8, dropout=0.5):
        super(CrossAttention, self).__init__()

        self.model_dim = model_dim
        self.dim_per_head = model_dim // num_heads
        self.num_heads = num_heads

        self.linear_q = nn.Linear(model_dim, self.dim_per_head * num_heads, bias=False)
        self.linear_k = nn.Linear(model_dim, self.dim_per_head * num_heads, bias=False)
        self.linear_v = nn.Linear(model_dim, self.dim_per_head * num_heads, bias=False)

        self.dot_product_attention = multimodal_attention(dropout)
        self.linear_final = nn.Linear(model_dim, model_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, query, key, value, attn_mask=None):
        residual = query

        # Linear projection
        query = self.linear_q(query)
        key = self.linear_k(key)
        value = self.linear_v(value)

        # Split by heads
        batch_size = query.size(0)
        query = query.view(batch_size, -1, self.num_heads, self.dim_per_head).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_heads, self.dim_per_head).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, self.dim_per_head).transpose(1, 2)

        # Scaled dot product attention
        scale = (self.dim_per_head) ** -0.5
        attention = self.dot_product_attention(query, key, value, scale, attn_mask)

        # Concatenate heads
        attention = attention.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.dim_per_head)

        # Final linear projection
        output = self.linear_final(attention)

        # Dropout
        output = self.dropout(output)

        # Add residual and norm layer
        output = self.layer_norm(residual + output)

        return output
    

class MultiHeadCrossAttention(nn.Module):
    def __init__(self, model_dim=768, num_heads=8, dropout=0.5):
        super(MultiHeadCrossAttention, self).__init__()

        self.model_dim = model_dim
        self.dim_per_head = model_dim // num_heads
        self.num_heads = num_heads

        self.cross_attention = CrossAttention(model_dim, num_heads, dropout)
        self.layer_norm = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x1, x2, attn_mask=None):
        # Cross attention from x1 to x2
        cross_attn_output_1 = self.cross_attention(x1, x2, x2, attn_mask)
        # Cross attention from x2 to x1
        cross_attn_output_2 = self.cross_attention(x2, x1, x1, attn_mask)

        # Combine the outputs
        output_1 = self.layer_norm(x1 + cross_attn_output_1)
        output_2 = self.layer_norm(x2 + cross_attn_output_2)

        return output_1, output_2
    

class MultiHeadAttention(nn.Module):
    def __init__(self, model_dim=768, num_heads=8, dropout=0.5):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.dim_per_head = model_dim // num_heads
        
        self.linear_q = nn.Linear(model_dim, model_dim)
        self.linear_k = nn.Linear(model_dim, model_dim)
        self.linear_v = nn.Linear(model_dim, model_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)
        self.linear_out = nn.Linear(model_dim, model_dim)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        query = self.linear_q(query).view(batch_size, -1, self.num_heads, self.dim_per_head).transpose(1, 2)
        key = self.linear_k(key).view(batch_size, -1, self.num_heads, self.dim_per_head).transpose(1, 2)
        value = self.linear_v(value).view(batch_size, -1, self.num_heads, self.dim_per_head).transpose(1, 2)
        
        scores = torch.matmul(query, key.transpose(-2, -1)) / (self.dim_per_head ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn = self.softmax(scores)
        attn = self.dropout(attn)
        
        context = torch.matmul(attn, value).transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.dim_per_head)
        output = self.linear_out(context)
        
        return output


def adaptive_resize(tensor, target_len):
    return F.adaptive_avg_pool2d(tensor.transpose(1, 2), (target_len, tensor.size(2)))

class CoAttention(nn.Module):
    def __init__(self, model_dim=768, num_heads=8, dropout=0.5):
        super(CoAttention, self).__init__()
        self.attention1 = MultiHeadAttention(model_dim, num_heads, dropout)
        self.attention2 = MultiHeadAttention(model_dim, num_heads, dropout)
        self.linear_out = nn.Linear(2 * model_dim, model_dim)
        self.layer_norm = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x1, x2, mask1=None, mask2=None):
        attn_output1 = self.attention1(x1, x2, x2, mask2)
        attn_output2 = self.attention2(x2, x1, x1, mask1)
        
        combined_1 = torch.cat([attn_output1.mean(dim=1), attn_output2.mean(dim=1)], dim=-1)
        output_1 = self.dropout(self.linear_out(combined_1))
        output_1 = self.layer_norm(output_1)
        
        attn_output2_new = adaptive_resize(attn_output2, x1.size(1))

        combined_2 = torch.cat([attn_output1, attn_output2_new], dim=-1)
        
        output_2 = self.dropout(self.linear_out(combined_2))
        output_2 = self.layer_norm(output_2)
        
        return output_1, output_2
    

class SimpleSeqEncoder(nn.Module):
    """
    简化版序列 encoder：
    输入 (B, L, D)，先做 pooling 得到 (B, D)，再映射到 z, w
    """
    def __init__(self, in_dim, z_dim, w_dim, hidden_dim=256, pooling="mean"):
        super().__init__()
        assert pooling in ["mean", "cls"], "pooling 只能是 mean 或 cls"
        self.pooling = pooling

        # backbone：把 pooled 向量映射到一个隐藏表示
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        # 输出 shared z 和 private w
        self.proj_z = nn.Linear(hidden_dim, z_dim)
        self.proj_w = nn.Linear(hidden_dim, w_dim)

    def forward(self, x):
        """
        x: (B, L, D)
        """
        if self.pooling == "mean":
            # 对序列做 mean pooling：平均所有 token 的向量
            # mask 的话可以以后再加
            h_seq = x.mean(dim=1)    # (B, D)
        else:
            # cls pooling：取第一个位置的 token 向量
            h_seq = x[:, 0, :]       # (B, D)

        h = self.backbone(h_seq)     # (B, hidden)
        z = self.proj_z(h)           # (B, z_dim)
        w = self.proj_w(h)           # (B, w_dim)
        return z, w
    
class SelfDecoder(nn.Module):
    """
    自重建 decoder：根据 [z, w] 重建 pooled 之后的特征向量
    """
    def __init__(self, z_dim, w_dim, out_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim + w_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, z, w):
        x = torch.cat([z, w], dim=-1)
        return self.net(x)


class CrossDecoder(nn.Module):
    """
    跨模态 decoder：只用 z 重建另一模态的 pooled 特征
    """
    def __init__(self, z_dim, out_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, z):
        return self.net(z)
    

