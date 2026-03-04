import os
import torch
import tqdm
import time
from .layers import *
from .gnn import SimpleGAT, GatedResidualFusion
from sklearn.metrics import *
from transformers import BertModel, ChineseCLIPModel, AutoModel
from .transformers_encoder.transformer import TransformerEncoder
from .transformers_encoder.lightweight_cnn_encoder import LightweightConvEncoder
import pdb

class MModel(torch.nn.Module):
    def __init__(self, config):
        super(MModel, self).__init__()

        self.bert_content = BertModel.from_pretrained(config['bert_path']).requires_grad_(False)
        self.bert_FTR = BertModel.from_pretrained(config['bert_path']).requires_grad_(False)
        self.clip = ChineseCLIPModel.from_pretrained(config['img_encoder_path']).requires_grad_(False)
        # Configurable: unfreeze last N BERT encoder layers (default 1)
        unfreeze_n = config.get('unfreeze_bert_layers', 1)
        def _set_unfreeze(model, n):
            total = getattr(model.config, 'num_hidden_layers', None)
            for name, param in model.named_parameters():
                if name.startswith('encoder.layer.'):
                    try:
                        idx = int(name.split('.')[2])
                    except Exception:
                        idx = None
                    if total is not None and idx is not None:
                        param.requires_grad = (idx >= total - n)
                    else:
                        # fallback to previous behavior: unfreeze last layer if unknown
                        param.requires_grad = ('encoder.layer.11' in name)
                else:
                    param.requires_grad = False

        _set_unfreeze(self.bert_content, unfreeze_n)
        _set_unfreeze(self.bert_FTR, unfreeze_n)

        for name, param in self.clip.named_parameters():
            # === 视觉部分 (Vision) ===
            # 解冻视觉编码器的最后一层 (假设是 ViT-Base, 共有12层, 索引为11)
            if "vision_model.encoder.layers.11" in name:
                param.requires_grad = True
            # 强烈建议同时解冻视觉投影层，因为它直接决定了特征的最终表示
            # elif "visual_projection" in name:
            #     param.requires_grad = True

        self.aggregator = MaskAttention(config['emb_dim'])
        self.mlp = MLP(config['emb_dim'], config['model']['mlp']['dims'], config['model']['mlp']['dropout'])

        self.hard_ftr_2_attention = MaskAttention(config['emb_dim'])
        self.hard_mlp_ftr_2 = nn.Sequential(nn.Linear(config['emb_dim'], config['model']['mlp']['dims'][-1]),
                        nn.ReLU(),
                        nn.Linear(config['model']['mlp']['dims'][-1], 1),
                        nn.Sigmoid()
                        )
        self.score_mapper_ftr_2 = nn.Sequential(nn.Linear(config['emb_dim'], config['model']['mlp']['dims'][-1]),
                        nn.BatchNorm1d(config['model']['mlp']['dims'][-1]),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(config['model']['mlp']['dims'][-1], 64),
                        nn.BatchNorm1d(64),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(64, 1),
                        nn.Sigmoid()
                        )

        self.hard_ftr_3_attention = MaskAttention(config['emb_dim'])
        self.hard_mlp_ftr_3 = nn.Sequential(nn.Linear(config['emb_dim'], config['model']['mlp']['dims'][-1]),
                        nn.ReLU(),
                        nn.Linear(config['model']['mlp']['dims'][-1], 1),
                        nn.Sigmoid()
                        )
        self.score_mapper_ftr_3 = nn.Sequential(nn.Linear(config['emb_dim'], config['model']['mlp']['dims'][-1]),
                        nn.BatchNorm1d(config['model']['mlp']['dims'][-1]),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(config['model']['mlp']['dims'][-1], 64),
                        nn.BatchNorm1d(64),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(64, 1),
                        nn.Sigmoid()
                        )

        self.simple_ftr_2_attention = MaskAttention(config['emb_dim'])
        self.simple_mlp_ftr_2 = nn.Sequential(nn.Linear(config['emb_dim'], config['model']['mlp']['dims'][-1]),
                        nn.ReLU(),
                        nn.Linear(config['model']['mlp']['dims'][-1], 3))
        self.simple_ftr_3_attention = MaskAttention(config['emb_dim'])
        self.simple_mlp_ftr_3 = nn.Sequential(nn.Linear(config['emb_dim'], config['model']['mlp']['dims'][-1]),
                        nn.ReLU(),
                        nn.Linear(config['model']['mlp']['dims'][-1], 3))

        self.content_attention = MaskAttention(config['emb_dim'])    

        self.co_attention_2 = ParallelCoAttentionNetwork(config['emb_dim'], config['co_attention_dim'], mask_in=True)
        self.co_attention_3 = ParallelCoAttentionNetwork(config['emb_dim'], config['co_attention_dim'], mask_in=True)

        self.cross_attention_content_2 = SelfAttentionFeatureExtract(1, config['emb_dim'])
        self.cross_attention_content_3 = SelfAttentionFeatureExtract(1, config['emb_dim'])

        self.cross_attention_ftr_2 = SelfAttentionFeatureExtract(1, config['emb_dim'])
        self.cross_attention_ftr_3 = SelfAttentionFeatureExtract(1, config['emb_dim'])

        # GNN modules for rationale redundancy scoring / refinement
        gnn_cfg = config.get('gnn', {}) if isinstance(config, dict) else {}
        gnn_hidden = gnn_cfg.get('hidden_dim', max(64, config['emb_dim']//2))
        self.gnn_top_k = gnn_cfg.get('top_k', 8)
        self.gnn_ftr_2 = SimpleGAT(config['emb_dim'], hidden_dim=gnn_hidden)
        self.gnn_ftr_3 = SimpleGAT(config['emb_dim'], hidden_dim=gnn_hidden)
        
        # Gated residual fusion modules for improved feature integration
        self.fusion_ftr_2 = GatedResidualFusion(config['emb_dim'])
        self.fusion_ftr_3 = GatedResidualFusion(config['emb_dim'])

        self.bert_dim = config['emb_dim']  # 假设 config['emb_dim'] 是 BERT 的维度 (768)
        self.clip_dim = self.clip.config.projection_dim # 获取 CLIP 投影后的维度 (通常是 512 或 768)

        # 2. 图像特征投影层: 将 CLIP 维度映射到 BERT 维度
        self.image_projection = nn.Sequential(
            nn.Linear(self.clip_dim, self.bert_dim),
            nn.LayerNorm(self.bert_dim),
            nn.ReLU()
        )
        # Text 查询 Image (Text-Guided Image Attention)
        # Query: Text, Key/Value: Image
        self.cross_attn_text_query_image = SelfAttentionFeatureExtract(1, config['emb_dim'])
        
        # Image 查询 Text (Image-Guided Text Attention)
        # Query: Image, Key/Value: Text
        self.cross_attn_image_query_text = SelfAttentionFeatureExtract(1, config['emb_dim'])

        # 4. 融合后的聚合层 (可选)
        self.fusion_gate = nn.Sequential(
            nn.Linear(self.bert_dim * 2, self.bert_dim),
            nn.Sigmoid()
        )


    def forward(self, **kwargs):
        content, content_masks = kwargs['content'], kwargs['content_masks']
        image = kwargs['img_tensors']

        FTR_2, FTR_2_masks = kwargs['FTR_2'], kwargs['FTR_2_masks']
        FTR_3, FTR_3_masks = kwargs['FTR_3'], kwargs['FTR_3_masks']

        content_feature = self.bert_content(content, attention_mask = content_masks)[0]
        image_feature = self.clip.visual_projection(self.clip.vision_model(image).last_hidden_state)
        image_feature = self.image_projection(image_feature) # [batch, seq_len_img, bert_dim]

        text_img, _ = self.cross_attn_text_query_image(image_feature, content_feature)
        img_text, _ = self.cross_attn_image_query_text(content_feature, image_feature, content_masks)
        
        # fusion plan A
        # 1. 聚合 img_text -> [batch, 1, dim]
        # 可以用 Mean Pooling 或 Max Pooling
        img_text_global = torch.mean(img_text, dim=1, keepdim=True)

        # 2. 广播并拼接到 text_img
        # [batch, 1, dim] -> [batch, seq_len_text, dim]
        img_text_expanded = img_text_global.expand(-1, content_feature.size(1), -1)

        # 3. 拼接
        concat_feature = torch.cat([text_img, img_text_expanded], dim=-1)

        # 4. 融合门
        fusion_feature = self.fusion_gate(concat_feature)

        # # fusion plan B
        # # 1. 文本分支 (保留序列结构)
        # # 融合了图片信息的文本序列特征
        # # 建议加上残差连接: text_img = text_img + content_feature
        # text_branch = text_img 

        # # 2. 图片分支 (提取全局视觉上下文)
        # # 将图片分支 Pooling 成全局向量 [batch, 1, dim]
        # img_branch_global = torch.mean(img_text, dim=1, keepdim=True)
        
        # # 3. 融合
        # # 将全局视觉特征扩展到文本长度 [batch, seq_len_text, dim]
        # img_branch_expanded = img_branch_global.expand(-1, content_feature.size(1), -1)
        
        # # 拼接 [batch, seq_len_text, dim * 2]
        # concat_feature = torch.cat([text_branch, img_branch_expanded], dim=-1)
        
        # # 经过门控网络降维 [batch, seq_len_text, dim]
        # fusion_feature = self.fusion_gate(concat_feature)

        # 4. 再次 Mask (重要)
        if content_masks is not None:
            fusion_feature = fusion_feature * content_masks.unsqueeze(-1)


        content_feature_1, content_feature_2 = fusion_feature, fusion_feature

        FTR_2_feature = self.bert_FTR(FTR_2, attention_mask = FTR_2_masks)[0]
        FTR_3_feature = self.bert_FTR(FTR_3, attention_mask = FTR_3_masks)[0]

        # === Graph Attention Network with Gated Residual Fusion ===
        # FTR_*_feature: (B, L, D)
        # Apply vectorized GAT for redundancy scoring and feature refinement
        if hasattr(self, 'gnn_ftr_2'):
            try:
                scores2, refined2 = self.gnn_ftr_2(FTR_2_feature, mask=FTR_2_masks, top_k=self.gnn_top_k)
                # Use gated residual fusion for better feature integration
                # This adaptively combines original and refined features
                FTR_2_feature = self.fusion_ftr_2(FTR_2_feature, refined2)
            except Exception:
                # If GNN fails, gracefully fall back to original features
                pass

        if hasattr(self, 'gnn_ftr_3'):
            try:
                scores3, refined3 = self.gnn_ftr_3(FTR_3_feature, mask=FTR_3_masks, top_k=self.gnn_top_k)
                # Use gated residual fusion for better feature integration
                FTR_3_feature = self.fusion_ftr_3(FTR_3_feature, refined3)
            except Exception:
                pass

        mutual_content_FTR_2, _ = self.cross_attention_content_2( \
            content_feature_2, FTR_2_feature, content_masks)
        expert_2 = torch.mean(mutual_content_FTR_2, dim=1)
    
        mutual_content_FTR_3, _ = self.cross_attention_content_3( \
            content_feature_2, FTR_3_feature, content_masks)
        expert_3 = torch.mean(mutual_content_FTR_3, dim=1)

        mutual_FTR_content_2, _ = self.cross_attention_ftr_2( \
            FTR_2_feature, content_feature_2, FTR_2_masks)
        mutual_FTR_content_2 = torch.mean(mutual_FTR_content_2, dim=1)

        mutual_FTR_content_3, _ = self.cross_attention_ftr_3( \
            FTR_3_feature, content_feature_2, FTR_3_masks)
        mutual_FTR_content_3 = torch.mean(mutual_FTR_content_3, dim=1)

        hard_ftr_2_pred = self.hard_mlp_ftr_2(mutual_FTR_content_2).squeeze(1)
        hard_ftr_3_pred = self.hard_mlp_ftr_3(mutual_FTR_content_3).squeeze(1)

        simple_ftr_2_pred = self.simple_mlp_ftr_2(self.simple_ftr_2_attention(FTR_2_feature)[0]).squeeze(1)
        simple_ftr_3_pred = self.simple_mlp_ftr_3(self.simple_ftr_3_attention(FTR_3_feature)[0]).squeeze(1)    

        attn_content, _ = self.content_attention(content_feature_1, mask=content_masks)

        reweight_score_ftr_2 = self.score_mapper_ftr_2(mutual_FTR_content_2)
        reweight_score_ftr_3 = self.score_mapper_ftr_3(mutual_FTR_content_3)

        reweight_expert_2 = reweight_score_ftr_2 * expert_2
        reweight_expert_3 = reweight_score_ftr_3 * expert_3


        all_feature = torch.cat(
            (attn_content.unsqueeze(1), reweight_expert_2.unsqueeze(1), reweight_expert_3.unsqueeze(1)), 
            dim = 1
        )
        final_feature, _ = self.aggregator(all_feature)

        label_pred = self.mlp(final_feature)
        gate_value = torch.concat([
            reweight_score_ftr_2,
            reweight_score_ftr_3
        ], dim=1)
        # pdb.set_trace()
        res = {
            'classify_pred': torch.sigmoid(label_pred.squeeze(1)),
            'gate_value': gate_value,
            'final_feature': final_feature,
            'content_feature': attn_content,
            'ftr_2_feature': reweight_expert_2,
            'ftr_3_feature': reweight_expert_3,
        }

        res['hard_ftr_2_pred'] = hard_ftr_2_pred
        res['hard_ftr_3_pred'] = hard_ftr_3_pred

        res['simple_ftr_2_pred'] = simple_ftr_2_pred
        res['simple_ftr_3_pred'] = simple_ftr_3_pred

        return res

class MModel_CE(torch.nn.Module):
    def __init__(self, config):
        super(MModel_CE, self).__init__()
        self.config = config
        self.bert_content = AutoModel.from_pretrained(config['text_encoder_path']).requires_grad_(False)
        self.bert_FTR = AutoModel.from_pretrained(config['rational_encoder_path']).requires_grad_(False)
        # self.clip = AutoModel.from_pretrained(config['img_encoder_path']).requires_grad_(False)
        if config['img_encoder'] == "swinT":
            self.img_encoder = AutoModel.from_pretrained(config['img_encoder_path']).requires_grad_(False)
            self.visual_dim = self.img_encoder.config.hidden_size
        else:
            self.img_encoder = AutoModel.from_pretrained(config['img_encoder_path']).requires_grad_(False)
            self.visual_dim = self.img_encoder.config.projection_dim
        # for name, param in self.bert_content.named_parameters():
        #     if name.startswith("encoder.layer.11"):
        #         param.requires_grad = True
        #     else:
        #         param.requires_grad = False
        # for name, param in self.bert_FTR.named_parameters():
        #     if name.startswith("encoder.layer.11"):
        #         param.requires_grad = True
        #     else:
        #         param.requires_grad = False

        # for name, param in self.clip.named_parameters():
        #     # === 视觉部分 (Vision) ===
        #     # 解冻视觉编码器的最后一层 (假设是 ViT-Base, 共有12层, 索引为11)
        #     if "vision_model.encoder.layers.11" in name:
        #         param.requires_grad = True
        #     # 强烈建议同时解冻视觉投影层，因为它直接决定了特征的最终表示
        #     # elif "visual_projection" in name:
        #     #     param.requires_grad = True

        self.aggregator = MaskAttention(config['emb_dim'])
        self.mlp = MLP(config['emb_dim'], config['model']['mlp']['dims'], config['model']['mlp']['dropout'], output_dim=config['num_classes'])

        # self.hard_ftr_2_attention = MaskAttention(config['emb_dim'])
        self.mlp_usefulpred_r1 = nn.Sequential(nn.Linear(config['emb_dim'], config['model']['mlp']['dims'][-1]),
                        nn.ReLU(),
                        nn.Linear(config['model']['mlp']['dims'][-1], 1),
                        nn.Sigmoid()
                        )
        self.useful_mapper_r1 = nn.Sequential(nn.Linear(config['emb_dim'], config['model']['mlp']['dims'][-1]),
                        nn.LayerNorm(config['model']['mlp']['dims'][-1]),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(config['model']['mlp']['dims'][-1], 64),
                        nn.LayerNorm(64),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(64, 1),
                        nn.Sigmoid()
                        )

        # self.hard_ftr_3_attention = MaskAttention(config['emb_dim'])
        self.mlp_usefulpred_r2 = nn.Sequential(nn.Linear(config['emb_dim'], config['model']['mlp']['dims'][-1]),
                        nn.ReLU(),
                        nn.Linear(config['model']['mlp']['dims'][-1], 1),
                        nn.Sigmoid()
                        )
        self.useful_mapper_r2 = nn.Sequential(nn.Linear(config['emb_dim'], config['model']['mlp']['dims'][-1]),
                        nn.LayerNorm(config['model']['mlp']['dims'][-1]),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(config['model']['mlp']['dims'][-1], 64),
                        nn.LayerNorm(64),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(64, 1),
                        nn.Sigmoid()
                        )

        self.r1_attention = MaskAttention(config['emb_dim'])
        self.mlp_pred_r1 = nn.Sequential(nn.Linear(config['emb_dim'], config['model']['mlp']['dims'][-1]),
                        nn.ReLU(),
                        nn.Linear(config['model']['mlp']['dims'][-1], 3))
        self.r2_attention = MaskAttention(config['emb_dim'])
        self.mlp_pred_r2 = nn.Sequential(nn.Linear(config['emb_dim'], config['model']['mlp']['dims'][-1]),
                        nn.ReLU(),
                        nn.Linear(config['model']['mlp']['dims'][-1], 3))

        self.content_attention = MaskAttention(config['emb_dim'])    

        self.co_attention_2 = ParallelCoAttentionNetwork(config['emb_dim'], config['co_attention_dim'], mask_in=True)
        self.co_attention_3 = ParallelCoAttentionNetwork(config['emb_dim'], config['co_attention_dim'], mask_in=True)

        self.cross_attention_content_2 = SelfAttentionFeatureExtract(1, config['emb_dim'])
        self.cross_attention_content_3 = SelfAttentionFeatureExtract(1, config['emb_dim'])

        self.cross_attention_ftr_2 = SelfAttentionFeatureExtract(1, config['emb_dim'])
        self.cross_attention_ftr_3 = SelfAttentionFeatureExtract(1, config['emb_dim'])


        self.bert_dim = config['emb_dim']  # 假设 config['emb_dim'] 是 BERT 的维度 (768)
        # self.clip_dim = self.clip.config.projection_dim # 获取 CLIP 投影后的维度 (通常是 512 或 768)

        # 2. 图像特征投影层: 将 CLIP 维度映射到 BERT 维度
        self.image_projection = nn.Sequential(
            nn.Linear(self.visual_dim, self.bert_dim),
            nn.LayerNorm(self.bert_dim),
            nn.ReLU()
        )
        # Text 查询 Image (Text-Guided Image Attention)
        # Query: Text, Key/Value: Image
        self.cross_attn_text_query_image = SelfAttentionFeatureExtract(1, config['emb_dim'])
        
        # Image 查询 Text (Image-Guided Text Attention)
        # Query: Image, Key/Value: Text
        self.cross_attn_image_query_text = SelfAttentionFeatureExtract(1, config['emb_dim'])

        # 4. 融合后的聚合层 (可选)
        self.fusion_gate = nn.Sequential(
            nn.Linear(self.bert_dim * 2, self.bert_dim),
            nn.Sigmoid()
        )

        self.cross_layer_1 = MultiHeadCrossAttention(model_dim=config['emb_dim'], num_heads=8, dropout=0.5)
        self.co_layer_1 = CoAttention(model_dim=config['emb_dim'], num_heads=8, dropout=0.5)


    def forward(self, **kwargs):
        # pdb.set_trace()
        content, content_masks = kwargs['content'], kwargs['content_masks']
        image = kwargs['img']

        r1, r1_masks = kwargs['r1'], kwargs['r1_masks']
        r2, r2_masks = kwargs['r2'], kwargs['r2_masks']

        content_feature = self.bert_content(content, attention_mask = content_masks)[0]
        if self.config['img_encoder'] == "clip":
            image_feature = self.img_encoder.visual_projection(self.img_encoder.vision_model(image).last_hidden_state) # [batch, seq_len_img, clip_dim]
        elif self.config['img_encoder'] == "swinT":
            image_feature = self.img_encoder(image).last_hidden_state

        image_feature = self.image_projection(image_feature) # [batch, seq_len_img, bert_dim]

        # text_img, _ = self.cross_attn_text_query_image(image_feature, content_feature) # [batch, seq_len_text, bert_dim]
        # img_text, _ = self.cross_attn_image_query_text(content_feature, image_feature, content_masks) # [batch, seq_len_img, bert_dim]
        
        # # fusion plan A
        # # 1. 聚合 img_text -> [batch, 1, dim]
        # # 可以用 Mean Pooling 或 Max Pooling
        # img_text_global = torch.mean(img_text, dim=1, keepdim=True)

        # # 2. 广播并拼接到 text_img
        # # [batch, 1, dim] -> [batch, seq_len_text, dim]
        # img_text_expanded = img_text_global.expand(-1, content_feature.size(1), -1)

        # # 3. 拼接
        # concat_feature = torch.cat([text_img, img_text_expanded], dim=-1)

        # # 4. 融合门
        # fusion_feature = self.fusion_gate(concat_feature)

        # # 4. 再次 Mask (重要)
        # if content_masks is not None:
        #     fusion_feature = fusion_feature * content_masks.unsqueeze(-1)

        # content_feature_1, content_feature_2 = fusion_feature, fusion_feature

        # fusion plan B
        content_feature, image_feature = self.cross_layer_1(content_feature, image_feature) 

        fusion_1,fusion_high_dim = self.co_layer_1(content_feature, image_feature)

        # pdb.set_trace()
        content_feature_1, content_feature_2 = fusion_high_dim , fusion_high_dim 


        r1_feature = self.bert_FTR(r1, attention_mask = r1_masks)[0]
        r2_feature = self.bert_FTR(r2, attention_mask = r2_masks)[0]

        # Apply Graph Attention Network with Gated Residual Fusion
        if hasattr(self, 'gnn_r1'):
            try:
                scores_r1, refined_r1 = self.gnn_r1(r1_feature, mask=r1_masks, top_k=self.gnn_top_k)
                r1_feature = self.fusion_r1(r1_feature, refined_r1)
            except Exception:
                pass
        
        if hasattr(self, 'gnn_r2'):
            try:
                scores_r2, refined_r2 = self.gnn_r2(r2_feature, mask=r2_masks, top_k=self.gnn_top_k)
                r2_feature = self.fusion_r2(r2_feature, refined_r2)
            except Exception:
                pass

        # === 注意力对照：用新闻读理由（content->rationale） & 用理由读新闻（rationale->content）
        # 返回注意力权重，便于后续可视化和解释
        mutual_content_r1, attn_news_read_r1 = self.cross_attention_content_2(
            content_feature_2, r1_feature, content_masks)
        mutual_content_r2, attn_news_read_r2 = self.cross_attention_content_3(
            content_feature_2, r2_feature, content_masks)
        expert_2 = torch.mean(mutual_content_r1, dim=1)
        expert_3 = torch.mean(mutual_content_r2, dim=1)

        mutual_r1_content, attn_r1_read_news = self.cross_attention_ftr_2(
            r1_feature, content_feature_2, r1_masks)
        mutual_r2_content, attn_r2_read_news = self.cross_attention_ftr_3(
            r2_feature, content_feature_2, r2_masks)
        mutual_r1_content = torch.mean(mutual_r1_content, dim=1)
        mutual_r2_content = torch.mean(mutual_r2_content, dim=1)

        r1_useful_pred = self.mlp_usefulpred_r1(mutual_r1_content).squeeze(1)
        r2_useful_pred = self.mlp_usefulpred_r2(mutual_r2_content).squeeze(1)

        r1_pred = self.mlp_pred_r1(self.r1_attention(r1_feature)[0]).squeeze(1)
        r2_pred = self.mlp_pred_r2(self.r2_attention(r2_feature)[0]).squeeze(1)    

        # attn_content, _ = self.content_attention(content_feature_1, mask=content_masks)
        attn_content = fusion_1

        reweight_score_r1 = self.useful_mapper_r1(mutual_r1_content)
        reweight_score_r2 = self.useful_mapper_r2(mutual_r2_content)

        reweight_expert_2 = reweight_score_r1 * expert_2
        reweight_expert_3 = reweight_score_r2 * expert_3


        all_feature = torch.cat(
            (attn_content.unsqueeze(1), reweight_expert_2.unsqueeze(1), reweight_expert_3.unsqueeze(1)), 
            dim = 1
        )
        final_feature, _ = self.aggregator(all_feature)

        label_pred = self.mlp(final_feature)
        gate_value = torch.concat([
            reweight_score_r1,
            reweight_score_r2
        ], dim=1)
        # pdb.set_trace()
        res = {
            'classify_pred': label_pred,
            'r1_useful': r1_useful_pred,
            'r2_useful': r2_useful_pred,
            'r1_pred': r1_pred,
            'r2_pred': r2_pred,
            # 记录注意力权重用于可视化（news->rationale 与 rationale->news）
            'attn_news_read_r1': attn_news_read_r1,
            'attn_news_read_r2': attn_news_read_r2,
            'attn_r1_read_news': attn_r1_read_news,
            'attn_r2_read_news': attn_r2_read_news,
        }

        debug_info = {
            'gate_value': gate_value,
            'final_feature': final_feature,
            'content_feature': attn_content,
            'r1_feature': reweight_expert_2,
            'r2_feature': reweight_expert_3,
        }
        return res, debug_info
    
class MModel_uncertainty(torch.nn.Module):
    def __init__(self, config):
        super(MModel_uncertainty, self).__init__()

        self.bert_content = BertModel.from_pretrained(config['bert_path']).requires_grad_(False)
        self.bert_FTR = BertModel.from_pretrained(config['bert_path']).requires_grad_(False)
        self.clip = ChineseCLIPModel.from_pretrained(config['img_encoder_path']).requires_grad_(False)
        # Configurable: unfreeze last N BERT encoder layers (default 1)
        unfreeze_n = config.get('unfreeze_bert_layers', 1)
        def _set_unfreeze(model, n):
            total = getattr(model.config, 'num_hidden_layers', None)
            for name, param in model.named_parameters():
                if name.startswith('encoder.layer.'):
                    try:
                        idx = int(name.split('.')[2])
                    except Exception:
                        idx = None
                    if total is not None and idx is not None:
                        param.requires_grad = (idx >= total - n)
                    else:
                        # fallback to previous behavior: unfreeze last layer if unknown
                        param.requires_grad = ('encoder.layer.11' in name)
                else:
                    param.requires_grad = False

        _set_unfreeze(self.bert_content, unfreeze_n)
        _set_unfreeze(self.bert_FTR, unfreeze_n)

        for name, param in self.clip.named_parameters():
            # === 视觉部分 (Vision) ===
            # 解冻视觉编码器的最后一层 (假设是 ViT-Base, 共有12层, 索引为11)
            if "vision_model.encoder.layers.11" in name:
                param.requires_grad = True
            # 强烈建议同时解冻视觉投影层，因为它直接决定了特征的最终表示
            # elif "visual_projection" in name:
            #     param.requires_grad = True

        self.aggregator = MaskAttention(config['emb_dim'])
        self.mlp = MLP(config['emb_dim'], config['model']['mlp']['dims'], config['model']['mlp']['dropout'], config['num_classes'])

        self.hard_ftr_2_attention = MaskAttention(config['emb_dim'])
        self.hard_mlp_ftr_2 = nn.Sequential(nn.Linear(config['emb_dim'], config['model']['mlp']['dims'][-1]),
                        nn.ReLU(),
                        nn.Linear(config['model']['mlp']['dims'][-1], 1),
                        nn.Sigmoid()
                        )
        self.score_mapper_ftr_2 = nn.Sequential(nn.Linear(config['emb_dim'], config['model']['mlp']['dims'][-1]),
                        nn.LayerNorm(config['model']['mlp']['dims'][-1]),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(config['model']['mlp']['dims'][-1], 64),
                        nn.LayerNorm(64),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(64, 1),
                        nn.Sigmoid()
                        )

        self.hard_ftr_3_attention = MaskAttention(config['emb_dim'])
        self.hard_mlp_ftr_3 = nn.Sequential(nn.Linear(config['emb_dim'], config['model']['mlp']['dims'][-1]),
                        nn.ReLU(),
                        nn.Linear(config['model']['mlp']['dims'][-1], 1),
                        nn.Sigmoid()
                        )
        self.score_mapper_ftr_3 = nn.Sequential(nn.Linear(config['emb_dim'], config['model']['mlp']['dims'][-1]),
                        nn.LayerNorm(config['model']['mlp']['dims'][-1]),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(config['model']['mlp']['dims'][-1], 64),
                        nn.LayerNorm(64),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(64, 1),
                        nn.Sigmoid()
                        )

        self.simple_ftr_2_attention = MaskAttention(config['emb_dim'])
        self.simple_mlp_ftr_2 = nn.Sequential(nn.Linear(config['emb_dim'], config['model']['mlp']['dims'][-1]),
                        nn.ReLU(),
                        nn.Linear(config['model']['mlp']['dims'][-1], 3))
        self.simple_ftr_3_attention = MaskAttention(config['emb_dim'])
        self.simple_mlp_ftr_3 = nn.Sequential(nn.Linear(config['emb_dim'], config['model']['mlp']['dims'][-1]),
                        nn.ReLU(),
                        nn.Linear(config['model']['mlp']['dims'][-1], 3))

        self.content_attention = MaskAttention(config['emb_dim'])    

        self.co_attention_2 = ParallelCoAttentionNetwork(config['emb_dim'], config['co_attention_dim'], mask_in=True)
        self.co_attention_3 = ParallelCoAttentionNetwork(config['emb_dim'], config['co_attention_dim'], mask_in=True)

        self.cross_attention_content_2 = SelfAttentionFeatureExtract(1, config['emb_dim'])
        self.cross_attention_content_3 = SelfAttentionFeatureExtract(1, config['emb_dim'])

        self.cross_attention_ftr_2 = SelfAttentionFeatureExtract(1, config['emb_dim'])
        self.cross_attention_ftr_3 = SelfAttentionFeatureExtract(1, config['emb_dim'])


        self.bert_dim = config['emb_dim']  # 假设 config['emb_dim'] 是 BERT 的维度 (768)
        self.clip_dim = self.clip.config.projection_dim # 获取 CLIP 投影后的维度 (通常是 512 或 768)

        # 2. 图像特征投影层: 将 CLIP 维度映射到 BERT 维度
        self.image_projection = nn.Sequential(
            nn.Linear(self.clip_dim, self.bert_dim),
            nn.LayerNorm(self.bert_dim),
            nn.ReLU()
        )
        # Text 查询 Image (Text-Guided Image Attention)
        # Query: Text, Key/Value: Image
        self.cross_attn_text_query_image = SelfAttentionFeatureExtract(1, config['emb_dim'])
        
        # Image 查询 Text (Image-Guided Text Attention)
        # Query: Image, Key/Value: Text
        self.cross_attn_image_query_text = SelfAttentionFeatureExtract(1, config['emb_dim'])

        # 4. 融合后的聚合层 (可选)
        self.fusion_gate = nn.Sequential(
            nn.Linear(self.bert_dim * 2, self.bert_dim),
            nn.Sigmoid()
        )

        self.fusion_pred_mlp = MLP(config['emb_dim'], config['model']['mlp']['dims'], config['model']['mlp']['dropout'], config['num_classes'])

        self.rational_projector = nn.Sequential(
            nn.Linear(config['emb_dim'], config['model']['mlp']['dims'][-1]),
            nn.LayerNorm(config['model']['mlp']['dims'][-1]),
            nn.ReLU()
        )

        # === 新增：不确定度转权重模块 ===
        self.uncertainty_to_alpha = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid() # 输出范围 [0,1]，作为 rational 融合比重
        )

        # GNN config
        gnn_cfg = config.get('gnn', {}) if isinstance(config, dict) else {}
        gnn_hidden = gnn_cfg.get('hidden_dim', max(64, config['emb_dim']//2))
        self.gnn_top_k = gnn_cfg.get('top_k', 8)
        self.gnn_ftr_2 = SimpleGAT(config['emb_dim'], hidden_dim=gnn_hidden)
        self.gnn_ftr_3 = SimpleGAT(config['emb_dim'], hidden_dim=gnn_hidden)
        
        # Gated residual fusion modules for improved feature integration
        self.fusion_ftr_2 = GatedResidualFusion(config['emb_dim'])
        self.fusion_ftr_3 = GatedResidualFusion(config['emb_dim'])


    def forward(self, **kwargs):
        content, content_masks = kwargs['content'], kwargs['content_masks']
        image = kwargs['img_tensors']

        FTR_2, FTR_2_masks = kwargs['FTR_2'], kwargs['FTR_2_masks']
        FTR_3, FTR_3_masks = kwargs['FTR_3'], kwargs['FTR_3_masks']

        content_feature = self.bert_content(content, attention_mask = content_masks)[0]
        image_feature = self.clip.visual_projection(self.clip.vision_model(image).last_hidden_state)
        image_feature = self.image_projection(image_feature) # [batch, seq_len_img, bert_dim]

        text_img, _ = self.cross_attn_text_query_image(image_feature, content_feature)
        img_text, _ = self.cross_attn_image_query_text(content_feature, image_feature, content_masks)
        
        # fusion plan A
        # 1. 聚合 img_text -> [batch, 1, dim]
        # 可以用 Mean Pooling 或 Max Pooling
        img_text_global = torch.mean(img_text, dim=1, keepdim=True)

        # 2. 广播并拼接到 text_img
        # [batch, 1, dim] -> [batch, seq_len_text, dim]
        img_text_expanded = img_text_global.expand(-1, content_feature.size(1), -1)

        # 3. 拼接
        concat_feature = torch.cat([text_img, img_text_expanded], dim=-1)

        # 4. 融合门
        fusion_feature = self.fusion_gate(concat_feature) # (bs, seq_len, bert_dim)

        # 4. 再次 Mask (重要)
        if content_masks is not None:
            fusion_feature = fusion_feature * content_masks.unsqueeze(-1)

        content_feature_1, content_feature_2 = fusion_feature, fusion_feature

        FTR_2_feature = self.bert_FTR(FTR_2, attention_mask = FTR_2_masks)[0]
        FTR_3_feature = self.bert_FTR(FTR_3, attention_mask = FTR_3_masks)[0]
        
        # Apply Graph Attention Network with Gated Residual Fusion
        if hasattr(self, 'gnn_ftr_2'):
            try:
                scores2, refined2 = self.gnn_ftr_2(FTR_2_feature, mask=FTR_2_masks, top_k=self.gnn_top_k)
                FTR_2_feature = self.fusion_ftr_2(FTR_2_feature, refined2)
            except Exception:
                pass

        if hasattr(self, 'gnn_ftr_3'):
            try:
                scores3, refined3 = self.gnn_ftr_3(FTR_3_feature, mask=FTR_3_masks, top_k=self.gnn_top_k)
                FTR_3_feature = self.fusion_ftr_3(FTR_3_feature, refined3)
            except Exception:
                pass

        mutual_content_FTR_2, _ = self.cross_attention_content_2( \
            content_feature_2, FTR_2_feature, content_masks)
        expert_2 = torch.mean(mutual_content_FTR_2, dim=1)
    
        mutual_content_FTR_3, _ = self.cross_attention_content_3( \
            content_feature_2, FTR_3_feature, content_masks)
        expert_3 = torch.mean(mutual_content_FTR_3, dim=1)

        mutual_FTR_content_2, _ = self.cross_attention_ftr_2( \
            FTR_2_feature, content_feature_2, FTR_2_masks)
        mutual_FTR_content_2 = torch.mean(mutual_FTR_content_2, dim=1)

        mutual_FTR_content_3, _ = self.cross_attention_ftr_3( \
            FTR_3_feature, content_feature_2, FTR_3_masks)
        mutual_FTR_content_3 = torch.mean(mutual_FTR_content_3, dim=1)

        hard_ftr_2_pred = self.hard_mlp_ftr_2(mutual_FTR_content_2).squeeze(1)
        hard_ftr_3_pred = self.hard_mlp_ftr_3(mutual_FTR_content_3).squeeze(1)

        simple_ftr_2_pred = self.simple_mlp_ftr_2(self.simple_ftr_2_attention(FTR_2_feature)[0]).squeeze(1)
        simple_ftr_3_pred = self.simple_mlp_ftr_3(self.simple_ftr_3_attention(FTR_3_feature)[0]).squeeze(1)    

        attn_content, _ = self.content_attention(content_feature_1, mask=content_masks)

        reweight_score_ftr_2 = self.score_mapper_ftr_2(mutual_FTR_content_2)
        reweight_score_ftr_3 = self.score_mapper_ftr_3(mutual_FTR_content_3)

        reweight_expert_2 = reweight_score_ftr_2 * expert_2
        reweight_expert_3 = reweight_score_ftr_3 * expert_3

        fusion_label_pred = self.fusion_pred_mlp(attn_content)
        with torch.no_grad():
            # 1. 使用 Softmax 将 Logits 转为概率分布 (和为1)
            prob = torch.softmax(fusion_label_pred, dim=-1) # [Batch, 2]
            
            # 2. 计算香农熵: H(x) = - sum(p * log(p))
            # dim=-1 确保我们在类别维度求和，结果变为 [Batch, 1]
            entropy = -(prob * torch.log(prob + 1e-8)).sum(dim=-1, keepdim=True) 
            
            # 3. 归一化: 二分类最大熵为 log(2) ≈ 0.693
            # 这样 entropy_score 范围就在 [0, 1] 之间，便于后续网络处理
            entropy_score = entropy / 0.693147 
        pdb.set_trace()
        alpha = self.uncertainty_to_alpha(entropy_score)  # [B, 1]

        content_weight = 1 - alpha  # [B, 1]
        expert_weight = alpha / 2   # [B, 1] 平分给两个 expert

        # 加权特征
        weighted_content = content_weight * attn_content  # [B, dim]
        weighted_expert_2 = expert_weight * reweight_expert_2  # [B, dim]
        weighted_expert_3 = expert_weight * reweight_expert_3  # [B, dim]

        all_feature = torch.cat(
            (weighted_content.unsqueeze(1), weighted_expert_2.unsqueeze(1), weighted_expert_3.unsqueeze(1)), 
            dim = 1
        )

        final_feature, _ = self.aggregator(all_feature)

        label_pred = self.mlp(final_feature)
        gate_value = torch.concat([
            reweight_score_ftr_2,
            reweight_score_ftr_3
        ], dim=1)

        res = {
            'classify_pred': label_pred,
            'gate_value': gate_value,
            'final_feature': final_feature,
            'content_feature': attn_content,
            'ftr_2_feature': reweight_expert_2,
            'ftr_3_feature': reweight_expert_3,
            'multimodal_pred': fusion_label_pred,
        }

        res['hard_ftr_2_pred'] = hard_ftr_2_pred
        res['hard_ftr_3_pred'] = hard_ftr_3_pred

        res['simple_ftr_2_pred'] = simple_ftr_2_pred
        res['simple_ftr_3_pred'] = simple_ftr_3_pred

        return res

class MModel_evidential_learning(torch.nn.Module):
    def __init__(self, config):
        super(MModel_evidential_learning, self).__init__()
        self.config = config
        self.bert_content = AutoModel.from_pretrained(config['text_encoder_path']).requires_grad_(False)
        self.bert_FTR = AutoModel.from_pretrained(config['rational_encoder_path']).requires_grad_(False)
        if config['img_encoder'] == "swinT":
            self.img_encoder = AutoModel.from_pretrained(config['img_encoder_path']).requires_grad_(False)
            self.visual_dim = self.img_encoder.config.hidden_size
        else:
            self.img_encoder = AutoModel.from_pretrained(config['img_encoder_path']).requires_grad_(False)
            self.visual_dim = self.img_encoder.config.projection_dim # 获取 CLIP 投影后的维度 (通常是 512 或 768)
        # for name, param in self.bert_content.named_parameters():
        #     if name.startswith("encoder.layer.11"):
        #         param.requires_grad = True
        #     else:
        #         param.requires_grad = False
        # for name, param in self.bert_FTR.named_parameters():
        #     if name.startswith("encoder.layer.11"):
        #         param.requires_grad = True
        #     else:
        #         param.requires_grad = False

        # for name, param in self.clip.named_parameters():
        #     # === 视觉部分 (Vision) ===
        #     # 解冻视觉编码器的最后一层 (假设是 ViT-Base, 共有12层, 索引为11)
        #     if "vision_model.encoder.layers.11" in name:
        #         param.requires_grad = True
        #     # 强烈建议同时解冻视觉投影层，因为它直接决定了特征的最终表示
        #     # elif "visual_projection" in name:
        #     #     param.requires_grad = True

        self.aggregator = MaskAttention(config['emb_dim'])
        self.mlp = MLP(config['emb_dim'], config['model']['mlp']['dims'], config['model']['mlp']['dropout'], output_dim=config['num_classes'])

        # self.hard_ftr_2_attention = MaskAttention(config['emb_dim'])
        self.mlp_usefulpred_r1 = nn.Sequential(nn.Linear(config['emb_dim'], config['model']['mlp']['dims'][-1]),
                        nn.ReLU(),
                        nn.Linear(config['model']['mlp']['dims'][-1], 1),
                        nn.Sigmoid()
                        )
        self.useful_mapper_r1 = nn.Sequential(nn.Linear(config['emb_dim'], config['model']['mlp']['dims'][-1]),
                        nn.LayerNorm(config['model']['mlp']['dims'][-1]),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(config['model']['mlp']['dims'][-1], 64),
                        nn.LayerNorm(64),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(64, 1),
                        nn.Sigmoid()
                        )

        # self.hard_ftr_3_attention = MaskAttention(config['emb_dim'])
        self.mlp_usefulpred_r2 = nn.Sequential(nn.Linear(config['emb_dim'], config['model']['mlp']['dims'][-1]),
                        nn.ReLU(),
                        nn.Linear(config['model']['mlp']['dims'][-1], 1),
                        nn.Sigmoid()
                        )
        self.useful_mapper_r2 = nn.Sequential(nn.Linear(config['emb_dim'], config['model']['mlp']['dims'][-1]),
                        nn.LayerNorm(config['model']['mlp']['dims'][-1]),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(config['model']['mlp']['dims'][-1], 64),
                        nn.LayerNorm(64),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(64, 1),
                        nn.Sigmoid()
                        )

        self.r1_attention = MaskAttention(config['emb_dim'])
        self.mlp_pred_r1 = nn.Sequential(nn.Linear(config['emb_dim'], config['model']['mlp']['dims'][-1]),
                        nn.ReLU(),
                        nn.Linear(config['model']['mlp']['dims'][-1], 3))
        self.r2_attention = MaskAttention(config['emb_dim'])
        self.mlp_pred_r2 = nn.Sequential(nn.Linear(config['emb_dim'], config['model']['mlp']['dims'][-1]),
                        nn.ReLU(),
                        nn.Linear(config['model']['mlp']['dims'][-1], 3))

        self.content_attention = MaskAttention(config['emb_dim'])    

        self.co_attention_2 = ParallelCoAttentionNetwork(config['emb_dim'], config['co_attention_dim'], mask_in=True)
        self.co_attention_3 = ParallelCoAttentionNetwork(config['emb_dim'], config['co_attention_dim'], mask_in=True)

        self.cross_attention_content_2 = SelfAttentionFeatureExtract(1, config['emb_dim'])
        self.cross_attention_content_3 = SelfAttentionFeatureExtract(1, config['emb_dim'])

        self.cross_attention_ftr_2 = SelfAttentionFeatureExtract(1, config['emb_dim'])
        self.cross_attention_ftr_3 = SelfAttentionFeatureExtract(1, config['emb_dim'])


        self.bert_dim = config['emb_dim']  # 假设 config['emb_dim'] 是 BERT 的维度 (768)

        # 2. 图像特征投影层: 将 CLIP 维度映射到 BERT 维度
        self.image_projection = nn.Sequential(
            nn.Linear(self.visual_dim, self.bert_dim),
            nn.LayerNorm(self.bert_dim),
            nn.ReLU()
        )
        # Text 查询 Image (Text-Guided Image Attention)
        # Query: Text, Key/Value: Image
        self.cross_attn_text_query_image = SelfAttentionFeatureExtract(1, config['emb_dim'])
        
        # Image 查询 Text (Image-Guided Text Attention)
        # Query: Image, Key/Value: Text
        self.cross_attn_image_query_text = SelfAttentionFeatureExtract(1, config['emb_dim'])

        # 4. 融合后的聚合层 (可选)
        self.fusion_gate = nn.Sequential(
            nn.Linear(self.bert_dim * 2, self.bert_dim),
            nn.Sigmoid()
        )

        # GNN config
        gnn_cfg = config.get('gnn', {}) if isinstance(config, dict) else {}
        gnn_hidden = gnn_cfg.get('hidden_dim', max(64, config['emb_dim']//2))
        self.gnn_top_k = gnn_cfg.get('top_k', 8)
        self.gnn_r1 = SimpleGAT(config['emb_dim'], hidden_dim=gnn_hidden)
        self.gnn_r2 = SimpleGAT(config['emb_dim'], hidden_dim=gnn_hidden)
        
        # Gated residual fusion modules for improved feature integration
        self.fusion_r1 = GatedResidualFusion(config['emb_dim'])
        self.fusion_r2 = GatedResidualFusion(config['emb_dim'])

        # evidential learning
        self.envidential_mlp_r1 = EvidentialUsefulnessHead(in_dim=config['emb_dim'])
        self.envidential_mlp_r2 = EvidentialUsefulnessHead(in_dim=config['emb_dim'])

        # self.weight_mlp_r1 = nn.Sequential(nn.Linear(config['emb_dim'] + 1, config['model']['mlp']['dims'][-1]),
        #                 nn.BatchNorm1d(config['model']['mlp']['dims'][-1]),
        #                 nn.ReLU(),
        #                 nn.Dropout(0.2),
        #                 nn.Linear(config['model']['mlp']['dims'][-1], 64),
        #                 nn.BatchNorm1d(64),
        #                 nn.ReLU(),
        #                 nn.Dropout(0.2),
        #                 nn.Linear(64, 1),
        #                 nn.Sigmoid()
        #                 )

        # self.weight_mlp_r2 = nn.Sequential(nn.Linear(config['emb_dim'] + 1, config['model']['mlp']['dims'][-1]),
        #                 nn.BatchNorm1d(config['model']['mlp']['dims'][-1]),
        #                 nn.ReLU(),
        #                 nn.Dropout(0.2),
        #                 nn.Linear(config['model']['mlp']['dims'][-1], 64),
        #                 nn.BatchNorm1d(64),
        #                 nn.ReLU(),
        #                 nn.Dropout(0.2),
        #                 nn.Linear(64, 1),
        #                 nn.Sigmoid()
        #                 )

        self.a_param = nn.Parameter(torch.tensor(1.0))
        self.b_param = nn.Parameter(torch.tensor(1.0))
        self.c_param = nn.Parameter(torch.tensor(1.0))
        self.d_param = nn.Parameter(torch.tensor(1.0))

        self.cross_layer_1 = MultiHeadCrossAttention(model_dim=config['emb_dim'], num_heads=8, dropout=0.5)
        self.co_layer_1 = CoAttention(model_dim=config['emb_dim'], num_heads=8, dropout=0.5)

    def forward(self, **kwargs):
        # mode = kwargs.get('mode', 'train')
        # ablation_type = kwargs.get('type', 4)
        content, content_masks = kwargs['content'], kwargs['content_masks']
        image = kwargs['img']

        r1, r1_masks = kwargs['r1'], kwargs['r1_masks']
        r2, r2_masks = kwargs['r2'], kwargs['r2_masks']

        content_feature = self.bert_content(content, attention_mask = content_masks)[0]

        if self.config['img_encoder'] == "clip":
            image_feature = self.img_encoder.visual_projection(self.img_encoder.vision_model(image).last_hidden_state)
        elif self.config['img_encoder'] == "swinT":
            image_feature = self.img_encoder(image).last_hidden_state
            
        image_feature = self.image_projection(image_feature) # [batch, seq_len_img, bert_dim]

        # text_img, _ = self.cross_attn_text_query_image(image_feature, content_feature)
        # img_text, _ = self.cross_attn_image_query_text(content_feature, image_feature, content_masks)
        
        # # fusion plan A
        # # 1. 聚合 img_text -> [batch, 1, dim]
        # # 可以用 Mean Pooling 或 Max Pooling
        # img_text_global = torch.mean(img_text, dim=1, keepdim=True)

        # # 2. 广播并拼接到 text_img
        # # [batch, 1, dim] -> [batch, seq_len_text, dim]
        # img_text_expanded = img_text_global.expand(-1, content_feature.size(1), -1)

        # # 3. 拼接
        # concat_feature = torch.cat([text_img, img_text_expanded], dim=-1)

        # # 4. 融合门
        # fusion_feature = self.fusion_gate(concat_feature)

        # # 4. 再次 Mask (重要)
        # if content_masks is not None:
        #     fusion_feature = fusion_feature * content_masks.unsqueeze(-1)


        # content_feature_1, content_feature_2 = fusion_feature, fusion_feature

        # fusion plan B
        content_feature, image_feature = self.cross_layer_1(content_feature, image_feature) 

        fusion_1,fusion_high_dim = self.co_layer_1(content_feature, image_feature)


        content_feature_1, content_feature_2 = fusion_high_dim , fusion_high_dim 

        r1_feature = self.bert_FTR(r1, attention_mask = r1_masks)[0]
        r2_feature = self.bert_FTR(r2, attention_mask = r2_masks)[0]

        # Apply Graph Attention Network with Gated Residual Fusion
        if hasattr(self, 'gnn_r1'):
            try:
                scores_r1, refined_r1 = self.gnn_r1(r1_feature, mask=r1_masks, top_k=self.gnn_top_k)
                r1_feature = self.fusion_r1(r1_feature, refined_r1)
            except Exception:
                pass
        
        if hasattr(self, 'gnn_r2'):
            try:
                scores_r2, refined_r2 = self.gnn_r2(r2_feature, mask=r2_masks, top_k=self.gnn_top_k)
                r2_feature = self.fusion_r2(r2_feature, refined_r2)
            except Exception:
                pass

        mutual_content_r1, _ = self.cross_attention_content_2( \
            content_feature_2, r1_feature, content_masks)
        expert_2 = torch.mean(mutual_content_r1, dim=1)
    
        mutual_content_r2, _ = self.cross_attention_content_3( \
            content_feature_2, r2_feature, content_masks)
        expert_3 = torch.mean(mutual_content_r2, dim=1)

        mutual_r1_content, _ = self.cross_attention_ftr_2( \
            r1_feature, content_feature_2, r1_masks)
        mutual_r1_content = torch.mean(mutual_r1_content, dim=1)

        mutual_r2_content, _ = self.cross_attention_ftr_3( \
            r2_feature, content_feature_2, r2_masks)
        mutual_r2_content = torch.mean(mutual_r2_content, dim=1)

        # r1_useful_pred = self.mlp_usefulpred_r1(mutual_r1_content).squeeze(1)
        # r2_useful_pred = self.mlp_usefulpred_r2(mutual_r2_content).squeeze(1)

        r1_pred = self.mlp_pred_r1(self.r1_attention(r1_feature)[0]).squeeze(1)
        r2_pred = self.mlp_pred_r2(self.r2_attention(r2_feature)[0]).squeeze(1)    

        # attn_content, _ = self.content_attention(content_feature_1, mask=content_masks)
        attn_content = fusion_1

        reweight_score_r1 = self.useful_mapper_r1(mutual_r1_content)
        reweight_score_r2 = self.useful_mapper_r2(mutual_r2_content)

        # reweight_expert_2 = reweight_score_ftr_2 * expert_2
        # reweight_expert_3 = reweight_score_ftr_3 * expert_3

        
        alpha_r1, pred_r1, uncertainty_r1 = self.envidential_mlp_r1(mutual_r1_content)
        alpha_r2, pred_r2, uncertainty_r2 = self.envidential_mlp_r1(mutual_r2_content)
             
        # r1_usefulness = pred_r1[:, 1].unsqueeze(-1) * (1 - uncertainty_r1)
        # r2_usefulness = pred_r2[:, 1].unsqueeze(-1) * (1 - uncertainty_r2)
        
        # version 1
        # reweight_expert_2 = r1_usefulness * expert_2
        # reweight_expert_3 = r2_usefulness * expert_3

        # version 2
        # r1_gate_i = torch.cat([mutual_FTR_content_2, r1_usefulness], dim=-1)
        # r2_gate_i = torch.cat([mutual_FTR_content_3, r2_usefulness], dim=-1)

        # r1_weight = self.weight_mlp_r1(r1_gate_i)
        # r2_weight = self.weight_mlp_r2(r2_gate_i)

        # reweight_expert_2 = r1_weight * expert_2
        # reweight_expert_3 = r2_weight * expert_3

        # version 3
        # r1_weight = 0.5 + (1 - uncertainty_r1) * (reweight_score_ftr_2 - 0.5)
        # r2_weight = 0.5 + (1 - uncertainty_r2) * (reweight_score_ftr_3 - 0.5)
        # uncertain_gate_value = torch.concat([r1_weight, r2_weight], dim=1)

        # version 4
        eps = 1e-6

        a = torch.relu(self.a_param)
        b = torch.relu(self.b_param)
        c = torch.relu(self.c_param)
        d = torch.relu(self.d_param)

        logit_g_r1 = torch.log(reweight_score_r1 + eps) - torch.log(1 - reweight_score_r1 + eps)
        logit_p_r1 = torch.log(pred_r1[:, 1] + eps) - torch.log(1 - pred_r1[:, 1] + eps)
        # logit_n_r1 = torch.log(pred_r1[:, 0] + eps) - torch.log(1 - pred_r1[:, 0] + eps)

        logit_g_r2 = torch.log(reweight_score_r2 + eps) - torch.log(1 - reweight_score_r2 + eps)
        logit_p_r2 = torch.log(pred_r2[:, 1] + eps) - torch.log(1 - pred_r2[:, 1] + eps)
        # logit_n_r2 = torch.log(pred_r2[:, 0] + eps) - torch.log(1 - pred_r2[:, 0] + eps)

        w_r1_logit = a * logit_g_r1 + b * logit_p_r1.unsqueeze(-1) - c * uncertainty_r1
        w_r2_logit = a * logit_g_r2 + b * logit_p_r2.unsqueeze(-1) - c * uncertainty_r2
        # if mode == 'train':
        #     w_r1_logit = a * logit_g_r1 + b * logit_p_r1.unsqueeze(-1) - c * uncertainty_r1
        #     w_r2_logit = a * logit_g_r2 + b * logit_p_r2.unsqueeze(-1) - c * uncertainty_r2
        # else:
        #     if ablation_type == 1:
        #         w_r1_logit = a * logit_g_r1 + b * logit_p_r1.unsqueeze(-1)
        #         w_r2_logit = a * logit_g_r2 + b * logit_p_r2.unsqueeze(-1)
        #     elif ablation_type == 2:
        #         w_r1_logit = a * logit_g_r1 - c * uncertainty_r1
        #         w_r2_logit = a * logit_g_r2 - c * uncertainty_r2
        #     elif ablation_type == 3:
        #         w_r1_logit = a * logit_g_r1
        #         w_r2_logit = a * logit_g_r2 
        #     else:
        #         w_r1_logit = a * logit_g_r1 + b * logit_p_r1.unsqueeze(-1) - c * uncertainty_r1
        #         w_r2_logit = a * logit_g_r2 + b * logit_p_r2.unsqueeze(-1) - c * uncertainty_r2

        r1_weight = torch.sigmoid(w_r1_logit)
        r2_weight = torch.sigmoid(w_r2_logit)

        # r1_weight = torch.tanh(w_r1_logit)
        # r2_weight = torch.tanh(w_r2_logit)
        # pdb.set_trace()
        reweight_expert_2 = r1_weight * expert_2
        reweight_expert_3 = r2_weight * expert_3

        uncertain_gate_value = torch.concat([r1_weight, r2_weight], dim=1)

        # all_feature = torch.cat(
        #     ((1 - r1_weight * r2_weight) * attn_content.unsqueeze(1), reweight_expert_2.unsqueeze(1), reweight_expert_3.unsqueeze(1)), 
        #     dim = 1
        # )

        all_feature = torch.cat(
            (attn_content.unsqueeze(1), reweight_expert_2.unsqueeze(1), reweight_expert_3.unsqueeze(1)), 
            dim = 1
        )
        final_feature, _ = self.aggregator(all_feature)

        label_pred = self.mlp(final_feature)
        gate_value = torch.concat([
            reweight_score_r1,
            reweight_score_r2
        ], dim=1)
        # pdb.set_trace()
        res = {
            'classify_pred': label_pred,
            'r1_pred': r1_pred,
            'r2_pred': r2_pred,
        }

        res['evidential_r1'] = dict(alpha=alpha_r1, pred=pred_r1, uncertainty=uncertainty_r1)
        res['evidential_r2'] = dict(alpha=alpha_r2, pred=pred_r2, uncertainty=uncertainty_r2)

        debug_info = {
            'gate_value': gate_value,
            'final_feature': final_feature,
            'content_feature': attn_content,
            'ftr_2_feature': reweight_expert_2,
            'ftr_3_feature': reweight_expert_3,
            'uncertain_gate_value': uncertain_gate_value
        }
        return res, debug_info
    
class MModel_distangle(torch.nn.Module):
    def __init__(self, config):
        super(MModel_distangle, self).__init__()
        self.config = config
        self.bert_content = AutoModel.from_pretrained(config['text_encoder_path']).requires_grad_(False)
        self.bert_FTR = AutoModel.from_pretrained(config['rational_encoder_path']).requires_grad_(False)
        # self.clip = AutoModel.from_pretrained(config['img_encoder_path']).requires_grad_(False)
        if config['img_encoder'] == "swinT":
            self.img_encoder = AutoModel.from_pretrained(config['img_encoder_path']).requires_grad_(False)
            self.visual_dim = self.img_encoder.config.hidden_size
        else:
            self.img_encoder = AutoModel.from_pretrained(config['img_encoder_path']).requires_grad_(False)
            self.visual_dim = self.img_encoder.config.projection_dim
        # for name, param in self.bert_content.named_parameters():
        #     if name.startswith("encoder.layer.11"):
        #         param.requires_grad = True
        #     else:
        #         param.requires_grad = False
        # for name, param in self.bert_FTR.named_parameters():
        #     if name.startswith("encoder.layer.11"):
        #         param.requires_grad = True
        #     else:
        #         param.requires_grad = False

        # for name, param in self.clip.named_parameters():
        #     # === 视觉部分 (Vision) ===
        #     # 解冻视觉编码器的最后一层 (假设是 ViT-Base, 共有12层, 索引为11)
        #     if "vision_model.encoder.layers.11" in name:
        #         param.requires_grad = True
        #     # 强烈建议同时解冻视觉投影层，因为它直接决定了特征的最终表示
        #     # elif "visual_projection" in name:
        #     #     param.requires_grad = True

        self.aggregator = MaskAttention(config['model']['z_dim'])
        self.mlp = MLP(config['model']['z_dim'], config['model']['mlp']['dims'], config['model']['mlp']['dropout'], output_dim=config['num_classes'])

        # self.hard_ftr_2_attention = MaskAttention(config['emb_dim'])
        self.mlp_usefulpred_r1 = nn.Sequential(nn.Linear(config['emb_dim'], config['model']['mlp']['dims'][-1]),
                        nn.ReLU(),
                        nn.Linear(config['model']['mlp']['dims'][-1], 1),
                        nn.Sigmoid()
                        )
        self.useful_mapper_r1 = nn.Sequential(nn.Linear(config['emb_dim'], config['model']['mlp']['dims'][-1]),
                        nn.BatchNorm1d(config['model']['mlp']['dims'][-1]),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(config['model']['mlp']['dims'][-1], 64),
                        nn.BatchNorm1d(64),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(64, 1),
                        nn.Sigmoid()
                        )

        # self.hard_ftr_3_attention = MaskAttention(config['emb_dim'])
        self.mlp_usefulpred_r2 = nn.Sequential(nn.Linear(config['emb_dim'], config['model']['mlp']['dims'][-1]),
                        nn.ReLU(),
                        nn.Linear(config['model']['mlp']['dims'][-1], 1),
                        nn.Sigmoid()
                        )
        self.useful_mapper_r2 = nn.Sequential(nn.Linear(config['emb_dim'], config['model']['mlp']['dims'][-1]),
                        nn.BatchNorm1d(config['model']['mlp']['dims'][-1]),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(config['model']['mlp']['dims'][-1], 64),
                        nn.BatchNorm1d(64),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(64, 1),
                        nn.Sigmoid()
                        )

        self.r1_attention = MaskAttention(config['emb_dim'])
        self.mlp_pred_r1 = nn.Sequential(nn.Linear(config['emb_dim'], config['model']['mlp']['dims'][-1]),
                        nn.ReLU(),
                        nn.Linear(config['model']['mlp']['dims'][-1], 3))
        self.r2_attention = MaskAttention(config['emb_dim'])
        self.mlp_pred_r2 = nn.Sequential(nn.Linear(config['emb_dim'], config['model']['mlp']['dims'][-1]),
                        nn.ReLU(),
                        nn.Linear(config['model']['mlp']['dims'][-1], 3))

        self.content_attention = MaskAttention(config['emb_dim'])    

        self.co_attention_2 = ParallelCoAttentionNetwork(config['emb_dim'], config['co_attention_dim'], mask_in=True)
        self.co_attention_3 = ParallelCoAttentionNetwork(config['emb_dim'], config['co_attention_dim'], mask_in=True)

        self.cross_attention_content_2 = SelfAttentionFeatureExtract(1, config['emb_dim'])
        self.cross_attention_content_3 = SelfAttentionFeatureExtract(1, config['emb_dim'])

        self.cross_attention_ftr_2 = SelfAttentionFeatureExtract(1, config['emb_dim'])
        self.cross_attention_ftr_3 = SelfAttentionFeatureExtract(1, config['emb_dim'])


        self.bert_dim = config['emb_dim']  # 假设 config['emb_dim'] 是 BERT 的维度 (768)
        # self.clip_dim = self.clip.config.projection_dim # 获取 CLIP 投影后的维度 (通常是 512 或 768)

        # 2. 图像特征投影层: 将 CLIP 维度映射到 BERT 维度
        self.image_projection = nn.Sequential(
            nn.Linear(self.visual_dim, self.bert_dim),
            nn.LayerNorm(self.bert_dim),
            nn.ReLU()
        )
        # Text 查询 Image (Text-Guided Image Attention)
        # Query: Text, Key/Value: Image
        self.cross_attn_text_query_image = SelfAttentionFeatureExtract(1, config['emb_dim'])
        
        # Image 查询 Text (Image-Guided Text Attention)
        # Query: Image, Key/Value: Text
        self.cross_attn_image_query_text = SelfAttentionFeatureExtract(1, config['emb_dim'])

        # 4. 融合后的聚合层 (可选)
        self.fusion_gate = nn.Sequential(
            nn.Linear(self.bert_dim * 2, self.bert_dim),
            nn.Sigmoid()
        )

        self.cross_layer_1 = MultiHeadCrossAttention(model_dim=config['emb_dim'], num_heads=8, dropout=0.5)
        self.co_layer_1 = CoAttention(model_dim=config['emb_dim'], num_heads=8, dropout=0.5)

        self.encoder_m = SimpleSeqEncoder(config['emb_dim'], config['model']['z_dim'], config['model']['w_dim'])
        self.encoder_r = SimpleSeqEncoder(config['emb_dim'], config['model']['z_dim'], config['model']['w_dim'])

        self.decoder_m_self = SelfDecoder(config['model']['z_dim'], config['model']['w_dim'], config['emb_dim'])
        self.decoder_r_self = SelfDecoder(config['model']['z_dim'], config['model']['w_dim'], config['emb_dim'])

        self.decoder_m2r = CrossDecoder(config['model']['z_dim'], config['emb_dim'])
        self.decoder_r2m = CrossDecoder(config['model']['z_dim'], config['emb_dim'])

    def forward(self, **kwargs):
        # pdb.set_trace()
        content, content_masks = kwargs['content'], kwargs['content_masks']
        image = kwargs['img']

        r1, r1_masks = kwargs['r1'], kwargs['r1_masks']
        r2, r2_masks = kwargs['r2'], kwargs['r2_masks']

        content_feature = self.bert_content(content, attention_mask = content_masks)[0]
        if self.config['img_encoder'] == "clip":
            image_feature = self.img_encoder.visual_projection(self.img_encoder.vision_model(image).last_hidden_state) # [batch, seq_len_img, clip_dim]
        elif self.config['img_encoder'] == "swinT":
            image_feature = self.img_encoder(image).last_hidden_state

        image_feature = self.image_projection(image_feature) # [batch, seq_len_img, bert_dim]

        # fusion plan B
        content_feature, image_feature = self.cross_layer_1(content_feature, image_feature) 

        fusion_1,fusion_high_dim = self.co_layer_1(content_feature, image_feature)        
        
        content_feature_1, content_feature_2 = fusion_high_dim , fusion_high_dim 

        r1_feature = self.bert_FTR(r1, attention_mask = r1_masks)[0]
        r2_feature = self.bert_FTR(r2, attention_mask = r2_masks)[0]

        # pdb.set_trace()
        z_m, w_m = self.encoder_m(fusion_high_dim)
        z_r, w_r = self.encoder_r(r1_feature)

        m_rec_self = self.decoder_m_self(z_m, w_m)
        r_rec_self = self.decoder_r_self(z_r, w_r)

        r_from_m = self.decoder_m2r(z_m)
        m_from_r = self.decoder_r2m(z_r)

        r_pooled = r1_feature.mean(dim=1)
        m_pooled = fusion_high_dim.mean(dim=1)

        all_feature = torch.cat(
            (z_m.unsqueeze(1), z_r.unsqueeze(1)), 
            dim = 1
        )
        final_feature, _ = self.aggregator(all_feature)

        label_pred = self.mlp(final_feature)
        # pdb.set_trace()
        res = {
            'classify_pred': label_pred,
            'r_target': r_pooled,
            'm_target': m_pooled,
            'm_rec_self': m_rec_self,
            'r_rec_self': r_rec_self,
            'r_from_m': r_from_m,
            'm_from_r': m_from_r,
            'z_m': z_m,
            'z_r': z_r,            
        }

        debug_info = {
            'final_feature': final_feature,
            
        }
        return res, debug_info
    
class MModel_distangle_1(torch.nn.Module):
    def __init__(self, config):
        super(MModel_distangle_1, self).__init__()
        self.config = config
        self.bert_content = AutoModel.from_pretrained(config['text_encoder_path']).requires_grad_(False)
        self.bert_FTR = AutoModel.from_pretrained(config['rational_encoder_path']).requires_grad_(False)
        
        # Image Encoder
        if config['img_encoder'] == "swinT":
            self.img_encoder = AutoModel.from_pretrained(config['img_encoder_path']).requires_grad_(False)
            self.visual_dim = self.img_encoder.config.hidden_size
        else:
            self.img_encoder = AutoModel.from_pretrained(config['img_encoder_path']).requires_grad_(False)
            self.visual_dim = self.img_encoder.config.projection_dim

        self.bert_dim = config['emb_dim']
        
        # Image Projection
        self.image_projection = nn.Sequential(
            nn.Linear(self.visual_dim, self.bert_dim),
            nn.LayerNorm(self.bert_dim),
            nn.ReLU()
        )

        self.fusion_gate = nn.Sequential(
            nn.Linear(self.bert_dim * 2, self.bert_dim),
            nn.Sigmoid()
        )

        self.len_mm, self.len_r = self.config['text_max_len'], self.config['rational_max_len']
        self.cross_layer_1 = MultiHeadCrossAttention(model_dim=config['emb_dim'], num_heads=8, dropout=0.5)
        self.co_layer_1 = CoAttention(model_dim=config['emb_dim'], num_heads=8, dropout=0.5)

        self.origin_dim = self.config['emb_dim']
        self.dim = self.config['model']['embed_dim']

        # View-Level GAT Fusion
        self.view_gat = SimpleGAT(in_dim=self.dim, hidden_dim=self.dim, heads=4)
        self.view_fusion = GatedResidualFusion(self.dim)

        self.proj_m = nn.Conv1d(self.origin_dim, self.dim, kernel_size=self.config['model']['conv1d_kernel_size_m'], padding=0, bias=False)
        self.proj_r1 = nn.Conv1d(self.origin_dim, self.dim, kernel_size=self.config['model']['conv1d_kernel_size_r'], padding=0, bias=False)
        self.proj_r2 = nn.Conv1d(self.origin_dim, self.dim, kernel_size=self.config['model']['conv1d_kernel_size_r'], padding=0, bias=False)

        self.encoder_p_m = self.get_network(network_type=self.config['model']['network_type']['encoder'], self_type='m')
        self.encoder_p_r1 = self.get_network(network_type=self.config['model']['network_type']['encoder'], self_type='r')
        self.encoder_p_r2 = self.get_network(network_type=self.config['model']['network_type']['encoder'], self_type='r')

        self.encoder_s = self.get_network(network_type=self.config['model']['network_type']['encoder'], self_type='shared')

        self.decoder_m = nn.Conv1d(2*self.dim, self.dim, kernel_size=1, padding=0, bias=False)
        self.decoder_r1 = nn.Conv1d(2*self.dim, self.dim, kernel_size=1, padding=0, bias=False)
        self.decoder_r2 = nn.Conv1d(2*self.dim, self.dim, kernel_size=1, padding=0, bias=False)

        self.proj_contra_m = nn.Linear(self.dim * (self.len_mm - self.config['model']['conv1d_kernel_size_m'] + 1), self.dim)
        self.proj_contra_r1 = nn.Linear(self.dim * (self.len_r- self.config['model']['conv1d_kernel_size_r'] + 1), self.dim)
        self.proj_contra_r2 = nn.Linear(self.dim * (self.len_r - self.config['model']['conv1d_kernel_size_r'] + 1), self.dim)

        self.self_attn_m_s = self.get_network(network_type=self.config['model']['network_type']['self_attn_shared'], self_type='m_self')
        self.self_attn_r1_s = self.get_network(network_type=self.config['model']['network_type']['self_attn_shared'], self_type='r_self')
        self.self_attn_r2_s = self.get_network(network_type=self.config['model']['network_type']['self_attn_shared'], self_type='r_self')

        self.proj1_s = nn.Linear(self.dim * 3, self.dim * 3)
        self.proj2_s = nn.Linear(self.dim * 3, self.dim * 3)
        self.out_layer_s = nn.Linear(self.dim * 3, self.config['num_classes'])

        self.m_p_cross = self.get_network(network_type=self.config['model']['network_type']['cross_attn'], self_type='mm')
        
        self.m_query_r1 = self.get_network(network_type=self.config['model']['network_type']['cross_query'], self_type='mr')
        self.r1_p_cross = self.get_network(network_type=self.config['model']['network_type']['cross_attn'], self_type='r_p')

        self.m_query_r2 = self.get_network(network_type=self.config['model']['network_type']['cross_query'], self_type='mr')
        self.r2_p_cross = self.get_network(network_type=self.config['model']['network_type']['cross_attn'], self_type='r_p')

        self.proj1_m_p = nn.Linear(self.dim, self.dim)
        self.proj2_m_p = nn.Linear(self.dim, self.dim)
        self.out_layer_m_p = nn.Linear(self.dim, self.config['num_classes'])
        self.proj1_r1_p = nn.Linear(self.dim, self.dim)
        self.proj2_r1_p = nn.Linear(self.dim, self.dim)
        self.out_layer_r1_p = nn.Linear(self.dim, self.config['num_classes'])
        self.proj1_r2_p = nn.Linear(self.dim, self.dim)
        self.proj2_r2_p = nn.Linear(self.dim, self.dim)
        self.out_layer_r2_p = nn.Linear(self.dim, self.config['num_classes'])

        self.projector_m = nn.Linear(self.dim, self.dim)
        self.projector_r1 = nn.Linear(self.dim, self.dim)
        self.projector_r2 = nn.Linear(self.dim, self.dim)
        self.projector_s = nn.Linear(3*self.dim, 3*self.dim)

        self.proj1 = nn.Linear(6*self.dim, 6*self.dim)
        self.proj2 = nn.Linear(6*self.dim, 6*self.dim)
        self.out_layer = nn.Linear(6*self.dim, self.config['num_classes'])

        self.output_dropout = self.config['model']['drop_out']['output']
        
        self.m_s_pooling = LightweightAttentionPooling(input_dim=self.dim, dropout=self.config['model']['drop_out']['pooling'])
        self.r1_s_pooling = LightweightAttentionPooling(input_dim=self.dim, dropout=self.config['model']['drop_out']['pooling'])
        self.r2_s_pooling = LightweightAttentionPooling(input_dim=self.dim, dropout=self.config['model']['drop_out']['pooling'])

        self.m_p_pooling = LightweightAttentionPooling(input_dim=self.dim, dropout=self.config['model']['drop_out']['pooling'])
        self.r1_p_pooling = LightweightAttentionPooling(input_dim=self.dim, dropout=self.config['model']['drop_out']['pooling'])
        self.r2_p_pooling = LightweightAttentionPooling(input_dim=self.dim, dropout=self.config['model']['drop_out']['pooling'])

    def forward(self, **kwargs):
        content, content_masks = kwargs['content'], kwargs['content_masks']
        image = kwargs['img']
        r1, r1_masks = kwargs['r1'], kwargs['r1_masks']
        r2, r2_masks = kwargs['r2'], kwargs['r2_masks']

        # 1. Encoders
        r1_feature = self.bert_FTR(r1, attention_mask = r1_masks)[0]
        r2_feature = self.bert_FTR(r2, attention_mask = r2_masks)[0]
        content_feature = self.bert_content(content, attention_mask = content_masks)[0]
        
        if self.config['img_encoder'] == "clip":
            image_feature = self.img_encoder.visual_projection(self.img_encoder.vision_model(image).last_hidden_state)
        elif self.config['img_encoder'] == "swinT":
            image_feature = self.img_encoder(image).last_hidden_state
        image_feature = self.image_projection(image_feature)

        content_feature, image_feature = self.cross_layer_1(content_feature, image_feature) 
        _, mm_feature = self.co_layer_1(content_feature, image_feature)  
        
        # 2. Projection & Disentanglement
        x_m = mm_feature.transpose(1, 2)
        x_r1 = r1_feature.transpose(1, 2)
        x_r2 = r2_feature.transpose(1, 2)

        proj_x_m = self.proj_m(x_m) # (Batch, Dim, Seq)
        proj_x_r1 = self.proj_r1(x_r1)
        proj_x_r2 = self.proj_r2(x_r2)

        # Prepare for encoders (Seq, BS, Dim)
        proj_x_m_t = proj_x_m.permute(2, 0, 1)
        proj_x_r1_t = proj_x_r1.permute(2, 0, 1)
        proj_x_r2_t = proj_x_r2.permute(2, 0, 1)

        # Private / Shared Encoding
        m_p = self.encoder_p_m(proj_x_m_t) # (Seq, BS, Dim)
        r1_p = self.encoder_p_r1(proj_x_r1_t)
        r2_p = self.encoder_p_r2(proj_x_r2_t)

        m_s = self.encoder_s(proj_x_m_t) # (Seq, BS, Dim)
        r1_s = self.encoder_s(proj_x_r1_t)
        r2_s = self.encoder_s(proj_x_r2_t)

        # 恢复 (BS, Dim, Seq) 用于解码器
        m_p_dec = m_p.permute(1, 2, 0)
        r1_p_dec = r1_p.permute(1, 2, 0)
        r2_p_dec = r2_p.permute(1, 2, 0)
        m_s_dec = m_s.permute(1, 2, 0)
        r1_s_dec = r1_s.permute(1, 2, 0)
        r2_s_dec = r2_s.permute(1, 2, 0)

        # 3. Contrastive Sim Features
        m_s_sim = self.proj_contra_m(m_s_dec.contiguous().view(x_m.size(0), -1))
        r1_s_sim = self.proj_contra_r1(r1_s_dec.contiguous().view(x_r1.size(0), -1))
        r2_s_sim = self.proj_contra_r2(r2_s_dec.contiguous().view(x_r2.size(0), -1))

        # 4. Reconstruction
        shared_list = [m_s_dec, r1_s_dec, r2_s_dec]
        m_recon = self.decoder_m(torch.cat([m_p_dec, shared_list[0]], dim=1)).permute(2, 0, 1)
        r1_recon = self.decoder_r1(torch.cat([r1_p_dec, shared_list[1]], dim=1)).permute(2, 0, 1)
        r2_recon = self.decoder_r2(torch.cat([r2_p_dec, shared_list[2]], dim=1)).permute(2, 0, 1)

        # Re-encode reconstructed features
        m_recon_p = self.encoder_p_m(m_recon).transpose(0, 1) # (BS, Seq, Dim)
        r1_recon_p = self.encoder_p_r1(r1_recon).transpose(0, 1)
        r2_recon_p = self.encoder_p_r2(r2_recon).transpose(0, 1)
        
        m_recon = m_recon.transpose(0, 1) # (BS, Seq, Dim)
        r1_recon = r1_recon.transpose(0, 1)
        r2_recon = r2_recon.transpose(0, 1)

        # 5. Pooling & Fusion
        m_s_pool, _ = self.m_s_pooling(m_s)
        r1_s_pool, _ = self.r1_s_pooling(r1_s)
        r2_s_pool, _ = self.r2_s_pooling(r2_s)

        s_fusion = torch.cat([m_s_pool, r1_s_pool, r2_s_pool], dim=1)
        s_proj = self.proj2_s(F.dropout(F.relu(self.proj1_s(s_fusion), inplace=True), p=self.output_dropout, training=self.training))
        s_proj += s_fusion
        logits_s = self.out_layer_s(s_proj)

        # Private pooling & interaction
        h_m_p = self.m_p_cross(m_p) 
        last_h_m, _ = self.m_p_pooling(h_m_p)

        h_r1_p = self.r1_p_cross(self.m_query_r1(m_p, r1_p, r1_p))
        last_h_r1, _ = self.r1_p_pooling(h_r1_p)

        h_r2_p = self.r2_p_cross(self.m_query_r2(m_p, r2_p, r2_p))
        last_h_r2, _ = self.r2_p_pooling(h_r2_p)

        # Aux Classifiers
        h_proj_m_p = self.proj2_m_p(F.dropout(F.relu(self.proj1_m_p(last_h_m), inplace=True), p=self.output_dropout, training=self.training)) + last_h_m
        logits_m_p = self.out_layer_m_p(h_proj_m_p)

        h_proj_r1_p = self.proj2_r1_p(F.dropout(F.relu(self.proj1_r1_p(last_h_r1), inplace=True), p=self.output_dropout, training=self.training)) + last_h_r1
        logits_r1_p = self.out_layer_r1_p(h_proj_r1_p)

        h_proj_r2_p = self.proj2_r2_p(F.dropout(F.relu(self.proj1_r2_p(last_h_r2), inplace=True), p=self.output_dropout, training=self.training)) + last_h_r2
        logits_r2_p = self.out_layer_r2_p(h_proj_r2_p)
        
        # 6. View-Level GAT Fusion
        last_m_p = torch.sigmoid(self.projector_m(h_proj_m_p))
        last_r1_p = torch.sigmoid(self.projector_r1(h_proj_r1_p))
        last_r2_p = torch.sigmoid(self.projector_r2(h_proj_r2_p))
        
        view_graph_input = torch.stack([last_m_p, last_r1_p, last_r2_p], dim=1) # (BS, 3, Dim)
        _, refined_graph = self.view_gat(view_graph_input, top_k=3)
        fused_view_graph = self.view_fusion(view_graph_input, refined_graph)
        
        final_m, final_r1, final_r2 = fused_view_graph.unbind(1)
        s_fusion = torch.sigmoid(self.projector_s(s_fusion))

        last_private = torch.cat([final_m, final_r1, final_r2, s_fusion], dim=1)
        
        last_pri_proj = self.proj2(
            F.dropout(F.relu(self.proj1(last_private), inplace=True), p=self.output_dropout, training=self.training))
        last_pri_proj += last_private

        output = self.out_layer(last_pri_proj)

        # 7. Return Dictionary (Fixed proj_x_* dims)
        res = {
            'proj_x_m': proj_x_m.transpose(1, 2), # Fixed: (Batch, Seq, Dim)
            'proj_x_r1': proj_x_r1.transpose(1, 2),
            'proj_x_r2': proj_x_r2.transpose(1, 2),
            'm_p': m_p,   # (Seq, BS, Dim)
            'r1_p': r1_p,
            'r2_p': r2_p,
            'm_s': m_s,
            'r1_s': r1_s,
            'r2_s': r2_s,
            'm_s_sim': m_s_sim,
            'r1_s_sim': r1_s_sim,
            'r2_s_sim': r2_s_sim,
            'm_recon_p': m_recon_p, # (BS, Seq, Dim)
            'r1_recon_p': r1_recon_p,
            'r2_recon_p': r2_recon_p,
            'm_recon': m_recon, # (BS, Seq, Dim)
            'r1_recon': r1_recon,
            'r2_recon': r2_recon,
            'logits_m_p': logits_m_p,
            'logits_r1_p': logits_r1_p,
            'logits_r2_p': logits_r2_p,
            'logits_s': logits_s,
            'classify_pred': output,
        }

        return res, None

    # get_network 方法保持不变 ...
    def get_network(self, network_type='transformer', self_type='m', layers=2):
        if self_type in ['m', 'r']:
            embed_dim, attn_dropout, layers, use_ffn = self.dim, self.config['model']['drop_out']['encoder'], self.config['model']['layer']['encoder'], self.config['model']['use_ffn']['encoder']
        else:
            embed_dim, attn_dropout, layers, use_ffn = self.dim, self.config['model']['drop_out'][self_type], self.config['model']['layer'][self_type], self.config['model']['use_ffn'][self_type]

        if network_type == 'conv':
            kernel_size = 3
            if 'conv1d_kernel_size' in self.config['model']:
                kernel_size = self.config['model']['conv1d_kernel_size']
            expansion = self.config['model'].get('cnn_expansion', 2)
            use_glu = self.config['model'].get('cnn_use_glu', True)

            return LightweightConvEncoder(
                embed_dim=embed_dim,
                layers=layers,
                expansion=expansion,
                kernel_size=kernel_size,
                dropout=attn_dropout,
                use_glu=use_glu,
            )
        elif network_type == 'transformer':
            return TransformerEncoder(
                embed_dim=embed_dim,
                num_heads=self.config['model']['num_heads'],
                layers=layers,
                attn_dropout=attn_dropout,
                relu_dropout=self.config['model']['drop_out']["attn"]['relu'],
                res_dropout=self.config['model']['drop_out']["attn"]['res'],
                embed_dropout=self.config['model']['drop_out']["attn"]['embed'],
                use_ffn=use_ffn
            )