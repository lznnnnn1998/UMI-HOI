"""
Implementations of transformers and variants

Implementation of encoder and decoder are adapted from DETR codebase by Facebook Research
https://github.com/facebookresearch/detr/blob/main/models/transformer.py

Implementation of Swin Transformers are adapted from microsoft/Swin-Transformer and torchvision
https://github.com/microsoft/Swin-Transformer/tree/main/models
https://github.com/pytorch/vision/blob/main/torchvision/models/swin_transformer.py

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Microsoft Research Asia

===================================================

The semantic MHA and unified token are composed in TransformerDecoderLayer

Zhinan Li <zhinan.li@tum.de>

Technical University of Munich


"""
import copy
import math
import torch
import torch.nn.functional as F

from torch import nn, Tensor
from attention import MultiheadAttention, MultiheadAttentionSigmoid
from typing import List, Optional, Callable

class TransformerEncoderLayer(nn.Module):
    def __init__(self, dim, num_heads, ffn_interm_dim, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.ffn_interm_dim = ffn_interm_dim
        # Linear projections on qkv have been removed in this custom layer.
        self.attn = MultiheadAttention(dim, num_heads, dropout=dropout)
        # Add the missing linear projections.
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        # The positional embeddings include box centres, widths and heights,
        # which will be twice the representation size.
        self.qpos_proj = nn.Linear(2 * dim, dim)
        self.kpos_proj = nn.Linear(2 * dim, dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_interm_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(ffn_interm_dim, dim)
        )
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.dp1 = nn.Dropout(dropout)
        self.dp2 = nn.Dropout(dropout)

    def forward(self, x, pos,
            attn_mask: Optional[Tensor] = None,
            key_padding_mask: Optional[Tensor] = None,
        ):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        q_pos = self.qpos_proj(pos)
        k_pos = self.kpos_proj(pos)
        q = q + q_pos
        k = k + k_pos
        attn, attn_weights = self.attn(
            query=q, key=k, value=v,
            attn_mask=attn_mask, key_padding_mask=key_padding_mask
        )
        x = self.ln1(x + self.dp1(attn))
        x = self.ln2(x + self.dp2(self.ffn(x)))
        return x, attn_weights

class TransformerEncoder(nn.Module):
    def __init__(self, hidden_size=256, num_heads=8, num_layers=2, dropout=.1):
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList([TransformerEncoderLayer(
            dim=hidden_size, num_heads=num_heads,
            ffn_interm_dim=hidden_size * 4, dropout=dropout
        ) for _ in range(num_layers)])

    def forward(self, x, pos):
        attn_weights = []
        for layer in self.layers:
            x, attn = layer(x, pos)
            attn_weights.append(attn)
        return x, attn_weights

class SelfAttnBlock(nn.Module):
    def __init__(self, q_dim, ffn_interm_dim, num_heads, dropout):
        super().__init__()
        self.ffn_interm_dim = ffn_interm_dim
        self.q_dim = q_dim
        self.num_heads = num_heads
        self.head_dim = q_dim // num_heads
        self.self_attn = MultiheadAttention(q_dim, self.num_heads, dropout=dropout)
        self.self_attn_q_proj = nn.ModuleList([nn.Linear(q_dim // num_heads, q_dim // num_heads) for _ in range(num_heads)])
        self.self_attn_k_proj = nn.ModuleList([nn.Linear(q_dim // num_heads, q_dim // num_heads) for _ in range(num_heads)])
        self.self_attn_v_proj = nn.ModuleList([nn.Linear(q_dim // num_heads, q_dim // num_heads) for _ in range(num_heads)])
        self.self_attn_ffn = nn.Sequential(
            nn.Linear(q_dim, ffn_interm_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(ffn_interm_dim, q_dim)
        )
        # self.self_attn_ffn = TopkMoE(q_dim, ffn_interm_dim, 4, 2, 0.1)
        self.self_attn_ln1 = nn.GroupNorm(num_heads, q_dim)
        self.self_attn_ln2 = nn.GroupNorm(num_heads, q_dim)
        self.self_attn_dp1 = nn.Dropout(dropout)
        self.self_attn_dp2 = nn.Dropout(dropout)
    def forward(self, input_tokens, input_masks):
        # perform self-attention
        q = torch.cat([self.self_attn_q_proj[i](input_tokens[..., i * self.head_dim: (i+1) * self.head_dim]) for i in range(self.num_heads)], dim=-1)
        k = torch.cat([self.self_attn_k_proj[i](input_tokens[..., i * self.head_dim: (i+1) * self.head_dim]) for i in range(self.num_heads)], dim=-1)
        v = torch.cat([self.self_attn_v_proj[i](input_tokens[..., i * self.head_dim: (i+1) * self.head_dim]) for i in range(self.num_heads)], dim=-1)
        
        attn, attn_weights = self.self_attn(
            query=q, key=k, value=v,
            key_padding_mask=input_masks
        )
        x = self.self_attn_ln1((input_tokens + self.self_attn_dp1(attn)).permute(0,2,1)).permute(0,2,1)
        x = self.self_attn_ln2((x + self.self_attn_dp2(self.self_attn_ffn(x))).permute(0,2,1)).permute(0,2,1)
        return x
class CrossAttnBlock(nn.Module):
    def __init__(self, q_dim, ffn_interm_dim, num_heads, dropout):
        super().__init__()
        self.ffn_interm_dim = ffn_interm_dim
        self.q_dim = q_dim
        self.num_heads = num_heads
        self.head_dim = q_dim // num_heads
        self.self_attn = MultiheadAttention(q_dim, self.num_heads, dropout=dropout)
        self.self_attn_q_proj = nn.ModuleList([nn.Linear(q_dim // num_heads, q_dim // num_heads) for _ in range(num_heads)])
        self.self_attn_k_proj = nn.ModuleList([nn.Linear(q_dim // num_heads, q_dim // num_heads) for _ in range(num_heads)])
        self.self_attn_v_proj = nn.ModuleList([nn.Linear(q_dim // num_heads, q_dim // num_heads) for _ in range(num_heads)])
        self.self_attn_ffn = nn.Sequential(
            nn.Linear(q_dim, ffn_interm_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(ffn_interm_dim, q_dim)
        )
        # self.self_attn_ffn = TopkMoE(q_dim, ffn_interm_dim, 4, 2, 0.1)
        self.self_attn_ln1 = nn.GroupNorm(num_heads, q_dim)
        self.self_attn_ln2 = nn.GroupNorm(num_heads, q_dim)
        self.self_attn_dp1 = nn.Dropout(dropout)
        self.self_attn_dp2 = nn.Dropout(dropout)
    def forward(self, cls_tokens, input_tokens, input_masks):
        # perform cross-attention
        q = torch.cat([self.self_attn_q_proj[i](cls_tokens[..., i * self.head_dim: (i+1) * self.head_dim]) for i in range(self.num_heads)], dim=-1)
        k = torch.cat([self.self_attn_k_proj[i](input_tokens[..., i * self.head_dim: (i+1) * self.head_dim]) for i in range(self.num_heads)], dim=-1)
        v = torch.cat([self.self_attn_v_proj[i](input_tokens[..., i * self.head_dim: (i+1) * self.head_dim]) for i in range(self.num_heads)], dim=-1)
        
        attn, attn_weights = self.self_attn(
            query=q, key=k, value=v,
            key_padding_mask=input_masks
        )
        x = self.self_attn_ln1((cls_tokens + self.self_attn_dp1(attn)).permute(0,2,1)).permute(0,2,1)
        x = self.self_attn_ln2((x + self.self_attn_dp2(self.self_attn_ffn(x))).permute(0,2,1)).permute(0,2,1)
        return x
class Expert(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))

class TopkMoE(nn.Module):
    def __init__(self, d_model, d_ff, num_experts, top_k=2, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        
        self.experts = nn.ModuleList([Expert(d_model, d_ff, dropout) for _ in range(num_experts)])
        self.gate = nn.Linear(d_model, num_experts)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # x shape: [batch_size, seq_len, d_model]
        seq_len, batch_size, d_model = x.shape
        x_flat = x.reshape(-1, d_model)  # [batch_size * seq_len, d_model]
        
        # Calculate gate scores
        gate_logits = self.gate(x_flat)  # [batch_size * seq_len, num_experts]
        gate_scores = self.softmax(gate_logits)
        
        # Top-k selection
        topk_weights, topk_indices = torch.topk(gate_scores, self.top_k, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)  # Normalize
        
        
        # Expert routing and computation
        expert_inputs = []
        expert_batch_indices = []
        
        for expert_idx in range(self.num_experts):
            # Find tokens assigned to this expert
            mask = (topk_indices == expert_idx).any(dim=-1)
            if mask.any():
                # Get tokens for this expert
                expert_tokens = x_flat[mask]
                expert_inputs.append(expert_tokens)
                expert_batch_indices.append(mask.nonzero(as_tuple=True)[0])
        
        # Process through experts
        expert_outputs = []
        for idx, (expert, tokens) in enumerate(zip(self.experts, expert_inputs)):
            expert_outputs.append(expert(tokens))
        
        # Combine outputs
        output_flat = torch.zeros_like(x_flat)
        for batch_indices, expert_out in zip(expert_batch_indices, expert_outputs):
            output_flat[batch_indices] = expert_out
        
        # Reshape back to original dimensions
        output = output_flat.reshape(seq_len, batch_size, d_model)
        
        # Calculate auxiliary loss for load balancing
        # aux_loss = self._load_balancing_loss(gate_scores, topk_indices)
        
        return output

    def _load_balancing_loss(self, gate_scores, topk_indices):
        # Calculate fraction of tokens handled by each expert
        expert_mask = torch.zeros_like(gate_scores).scatter_(
            1, topk_indices, 1.0
        )
        expert_load = expert_mask.sum(dim=0)  # [num_experts]
        
        # Calculate gate score sums for each expert
        gate_score_sums = (gate_scores * expert_mask).sum(dim=0)
        
        # Load balancing loss
        load_balance_loss = (expert_load.float() * gate_score_sums).sum()
        load_balance_loss = self.num_experts * load_balance_loss / (gate_scores.size(0) ** 2)
        
        return load_balance_loss

class TransformerDecoderLayer(nn.Module):

    def __init__(self, q_dim, kv_dim, num_heads, ffn_interm_dim, dropout=0.1, layer_id=0, sub_headnum=2, obj_headnum=6):
        """
        Parameters:
        -----------
        q_dim: int
            Dimension of the interaction queries.
        kv_dim: int
            Dimension of the image features.
        num_heads: int
            Number of heads used in multihead attention.
        ffn_interm_dim: int
            Dimension of the intermediate representation in the feedforward network.
        dropout: float, default: 0.1
            Dropout percentage used during training.
        """
        super().__init__()
        self.q_dim = q_dim
        self.head_dim = q_dim // num_heads
        self.kv_dim = kv_dim
        self.num_heads = num_heads
        self.inter_head_dim = 32
        self.h_dim = q_dim // 8 * sub_headnum
        self.o_dim = q_dim // 8 * obj_headnum
        self.h_head_num = sub_headnum
        self.o_head_num = obj_headnum
        self.dropout = dropout
        self.ffn_interm_dim = ffn_interm_dim
        self.layer_id = layer_id
        # ho_queries, all features, codes projections
        self.reg_num = 256
        self.llava_cali_embedding = nn.Embedding(256, q_dim)
        self.register_embedding = nn.Embedding(self.reg_num, q_dim)
        if layer_id == 0:
            h_query_dim = self.h_dim
            o_query_dim = self.o_dim
        else:
            h_query_dim = q_dim
            o_query_dim = q_dim
        self.h_query_adapter = nn.Sequential(
            nn.Linear(h_query_dim, ffn_interm_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_interm_dim, q_dim),
            nn.ReLU(),
            nn.Linear(q_dim, self.h_dim // 2)
        )
        self.o_query_adapter = nn.Sequential(
            nn.Linear(o_query_dim, ffn_interm_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_interm_dim, q_dim),
            nn.ReLU(),
            nn.Linear(q_dim, self.o_dim // 2)
        )
        

        self.cls_pos_proj = nn.Linear(kv_dim * 2 + kv_dim * 4, q_dim // num_heads // 2)
        self.h_feature_proj = nn.Sequential(
            nn.Linear(self.h_dim, ffn_interm_dim),
            nn.ReLU(),
            nn.Linear(ffn_interm_dim, self.h_dim//2)
        )
        self.o_feature_proj = nn.Sequential(
            nn.Linear(self.o_dim, ffn_interm_dim),
            nn.ReLU(),
            nn.Linear(ffn_interm_dim, self.o_dim//2)
        )
        self.backbone_pos_proj_h = nn.Sequential(nn.Linear(q_dim, ffn_interm_dim), nn.ReLU(), nn.Linear(ffn_interm_dim, self.h_dim // 2))
        self.backbone_pos_proj_o = nn.Sequential(nn.Linear(q_dim, ffn_interm_dim), nn.ReLU(), nn.Linear(ffn_interm_dim, self.o_dim // 2))

        self.answer_proj = nn.Sequential(nn.Linear(q_dim, ffn_interm_dim), nn.ReLU(), nn.Linear(ffn_interm_dim, q_dim))
        self.clip_token_proj_h = nn.Sequential(
            nn.Linear(1536, ffn_interm_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_interm_dim, self.h_dim))
        self.clip_token_proj_o = nn.Sequential(
            nn.Linear(1536, ffn_interm_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_interm_dim, self.o_dim))
        self.clip_cls_token_proj = nn.Sequential(
            nn.Linear(1536, ffn_interm_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_interm_dim, q_dim))
        self.input_token_ln = nn.LayerNorm(q_dim)

        
        # self-attn
        
        self.self_attn = MultiheadAttention(q_dim, self.num_heads, dropout=dropout)
        self.self_attn_q_proj_h = nn.Linear(self.h_dim, self.h_dim)
        self.self_attn_k_proj_h = nn.Linear(self.h_dim, self.h_dim)
        self.self_attn_v_proj_h = nn.Linear(self.h_dim, self.h_dim)
        self.self_attn_q_proj_o = nn.Linear(self.o_dim, self.o_dim)
        self.self_attn_k_proj_o = nn.Linear(self.o_dim, self.o_dim)
        self.self_attn_v_proj_o = nn.Linear(self.o_dim, self.o_dim)
        # self.self_attn_q_proj = nn.ModuleList([nn.Linear(q_dim // num_heads, q_dim // num_heads) for _ in range(num_heads)])
        # self.self_attn_k_proj = nn.ModuleList([nn.Linear(q_dim // num_heads, q_dim // num_heads) for _ in range(num_heads)])
        # self.self_attn_v_proj = nn.ModuleList([nn.Linear(q_dim // num_heads, q_dim // num_heads) for _ in range(num_heads)])
        self.self_attn_ffn = nn.Sequential(
            nn.Linear(q_dim, ffn_interm_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(ffn_interm_dim, q_dim)
        )
        # self.self_attn_ffn = TopkMoE(q_dim, ffn_interm_dim, 4, 2, 0.1)
        self.self_attn_ln1 = nn.LayerNorm(q_dim)
        self.self_attn_ln2 = nn.LayerNorm(q_dim)
        self.self_attn_dp1 = nn.Dropout(dropout)
        self.self_attn_dp2 = nn.Dropout(dropout)
    def forward(self,
            queries: Tensor, layer_id:int, features: Tensor,
            q_pos: Tensor, k_pos: Tensor,
            q_attn_mask: Optional[Tensor] = None,
            qk_attn_mask: Optional[Tensor] = None,
            q_padding_mask: Optional[Tensor] = None,
            kv_padding_mask: Optional[Tensor] = None,
            llava_answer_idx:Optional[Tensor] = None,
            llava_feature: Optional[Tensor] = None
        ):
        """
        Parameters:
        -----------
        queries: Tensor
            Interaction queries of size (N, B, K).
        features: Tensor
            Image features of size (HW, B, C).
        q_attn_mask: Tensor, default: None
            Attention mask to be applied during the self attention of queries.
        qk_attn_mask: Tensor, default: None
            Attention mask to be applied during the cross attention from image
            features to interaction queries.
        q_padding_mask: Tensor, default: None
            Padding mask for interaction queries of size (B, N). Values of `True`
            indicate the corresponding query was padded and to be ignored.
        kv_padding_mask: Tensor, default: None
            Padding mask for image features of size (B, HW).
        q_pos: Tensor, default: None
            Positional encodings for the interaction queries.
        k_pos: Tensor, default: None
            Positional encodings for the image features.

        Returns:
        --------
        queries: Tensor
        """
        # process cls token
        if layer_id == 0:
            q_h = self.h_query_adapter(queries[..., :self.h_dim])
            q_o = self.o_query_adapter(queries[..., self.h_dim:])
        else:
            q_h = self.h_query_adapter(queries)
            q_o = self.o_query_adapter(queries)
        q_p = self.cls_pos_proj(torch.cat((q_pos["centre"], q_pos["box"]), dim=-1))
        n_q, bs, _ = q_p.shape
        q_h = q_h.view(n_q, bs, self.h_head_num, self.h_dim // 2 // self.h_head_num)
        q_o = q_o.view(n_q, bs, self.o_head_num, self.o_dim // 2 // self.o_head_num)
        q_p = q_p.unsqueeze(-2)
        q_h = torch.cat((q_h, q_p.repeat(1,1,self.h_head_num, 1)), dim=-1).view(n_q, bs, self.h_dim)
        q_o = torch.cat((q_o, q_p.repeat(1,1,self.o_head_num, 1)), dim=-1).view(n_q, bs, self.o_dim)
        cls_token = torch.cat([q_h, q_o], dim=-1)
        num_cls_token = len(cls_token)

        # process vision token
        hw, bs, feature_dim = features.shape
        h_feature_pos = self.backbone_pos_proj_h(k_pos).view(hw, bs, self.h_head_num, self.h_dim // self.h_head_num // 2)
        o_feature_pos = self.backbone_pos_proj_o(k_pos).view(hw, bs, self.o_head_num, self.o_dim // self.o_head_num // 2)
        h_feature = self.h_feature_proj(features[..., :self.h_dim]).view(hw, bs, self.h_head_num, self.h_dim // self.h_head_num // 2)
        o_feature = self.o_feature_proj(features[..., self.h_dim:]).view(hw, bs, self.o_head_num, self.o_dim // self.o_head_num // 2)
        
        backbone_token = torch.cat((
            torch.cat((h_feature, h_feature_pos), dim=-1).view(hw, bs, self.h_dim), 
            torch.cat((o_feature, o_feature_pos), dim=-1).view(hw, bs, self.o_dim)
        ), dim=-1)
        clip_token = torch.cat((
            self.clip_token_proj_h(llava_feature[:-1]), self.clip_token_proj_o(llava_feature[:-1])
        ), dim=-1)
        clip_token = torch.cat((
            clip_token, self.clip_cls_token_proj(llava_feature[-1]).unsqueeze(0)
        ), dim=0).unsqueeze(1)
        num_backbone_token = len(backbone_token)
        num_clip_token = len(clip_token)

        # process answer token
        answer_token = self.llava_cali_embedding.weight[llava_answer_idx.unique()].unsqueeze(1)
        answer_token = self.answer_proj(answer_token)
        num_answer_token = len(answer_token)
        # register token
        reg_token = self.register_embedding.weight.unsqueeze(1)

        # concatenate all tokens
        input_tokens = torch.cat((cls_token, backbone_token, clip_token, answer_token, reg_token), dim=0)
        input_tokens = self.input_token_ln(input_tokens)

        input_masks = torch.zeros_like(input_tokens[:, :, 0]).to(torch.bool).transpose(0, 1)
        input_masks[:, num_cls_token:num_backbone_token+num_cls_token] = kv_padding_mask
        # perform self-attention
        q = torch.cat((
            self.self_attn_q_proj_h(cls_token[..., :self.h_dim]), self.self_attn_q_proj_o(cls_token[..., self.h_dim:])
        ), dim=-1)
        k = torch.cat((
            self.self_attn_k_proj_h(input_tokens[..., :self.h_dim]), self.self_attn_k_proj_o(input_tokens[..., self.h_dim:])
        ), dim=-1)
        v = torch.cat((
            self.self_attn_v_proj_h(input_tokens[..., :self.h_dim]), self.self_attn_v_proj_o(input_tokens[..., self.h_dim:])
        ), dim=-1)
        # q = torch.cat([self.self_attn_q_proj[i](cls_token[..., i * self.head_dim: (i+1) * self.head_dim]) for i in range(self.num_heads)], dim=-1)
        # k = torch.cat([self.self_attn_k_proj[i](input_tokens[..., i * self.head_dim: (i+1) * self.head_dim]) for i in range(self.num_heads)], dim=-1)
        # v = torch.cat([self.self_attn_v_proj[i](input_tokens[..., i * self.head_dim: (i+1) * self.head_dim]) for i in range(self.num_heads)], dim=-1)
        
        attn, attn_weights = self.self_attn(
            query=q, key=k, value=v,
            key_padding_mask=input_masks
        )
        x = self.self_attn_ln1((cls_token + self.self_attn_dp1(attn)))
        x = self.self_attn_ln2((x + self.self_attn_dp2(self.self_attn_ffn(x))))

        return x

class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer:TransformerDecoderLayer, num_layers, return_intermediate=True):
        super().__init__()
        
        layers = []
        for i in range(num_layers):
            layers.append(
                TransformerDecoderLayer(
                    decoder_layer.q_dim, decoder_layer.kv_dim, 
                    decoder_layer.num_heads, decoder_layer.ffn_interm_dim, layer_id=i, 
                    sub_headnum=decoder_layer.h_head_num, obj_headnum=decoder_layer.o_head_num))
        self.layers = nn.ModuleList(layers)
        self.num_layers = num_layers
        self.norm = nn.LayerNorm(decoder_layer.q_dim)
        self.return_intermediate = return_intermediate

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, queries, features,
            q_attn_mask: Optional[Tensor] = None,
            qk_attn_mask: Optional[Tensor] = None,
            q_padding_mask: Optional[Tensor] = None,
            kv_padding_mask: Optional[Tensor] = None,
            q_pos: Optional[Tensor] = None,
            k_pos: Optional[Tensor] = None,
            llava_answer_idx:Optional[Tensor] = None,
            llava_feature:Optional[Tensor] = None
        ):
        # Add support for zero layers
        if self.num_layers == 0:
            return queries.unsqueeze(0)
        # Explicitly handle zero-size queries
        if queries.numel() == 0:
            rp = self.num_layers if self.return_intermediate else 1
            return queries.unsqueeze(0).repeat(rp, 1, 1, 1)

        output = queries
        intermediate = []

        for idx, layer in enumerate(self.layers):
            # history = [queries]
            # history.extend(intermediate)
            # history.pop(-1)
            output = layer(
                output, idx, features,
                q_attn_mask=q_attn_mask,
                qk_attn_mask=qk_attn_mask,
                q_padding_mask=q_padding_mask,
                kv_padding_mask=kv_padding_mask,
                q_pos=q_pos, k_pos=k_pos,llava_answer_idx=llava_answer_idx, llava_feature=llava_feature
            )
            if self.return_intermediate:
                intermediate.append(self.norm(output))
                # intermediate.append(output)

        if self.return_intermediate:
            output = torch.stack(intermediate)
        else:
            output = self.norm(output).unsqueeze(0)
            # output = output.unsqueeze(0)
        return output

class TransformerDecoderLayerNoTaskSpecified(nn.Module):

    def __init__(self, q_dim, kv_dim, num_heads, ffn_interm_dim, dropout=0.1):
        """
        Parameters:
        -----------
        q_dim: int
            Dimension of the interaction queries.
        kv_dim: int
            Dimension of the image features.
        num_heads: int
            Number of heads used in multihead attention.
        ffn_interm_dim: int
            Dimension of the intermediate representation in the feedforward network.
        dropout: float, default: 0.1
            Dropout percentage used during training.
        """
        super().__init__()
        self.q_dim = q_dim
        self.kv_dim = kv_dim
        self.num_heads = num_heads
        self.inter_head_dim = 32
        self.dropout = dropout
        self.ffn_interm_dim = ffn_interm_dim

        # ho_queries, all features, codes projections
        self.reg_num = 256
        self.llava_cali_embedding = nn.Embedding(256, q_dim)
        self.register_embedding = nn.Embedding(self.reg_num, q_dim)
        self.query_adapter = nn.Sequential(
            nn.Linear(q_dim, ffn_interm_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_interm_dim, q_dim),
            nn.ReLU(),
            nn.Linear(q_dim, q_dim // 2)
        )
        self.cls_pos_proj = nn.Linear(kv_dim * 2 + kv_dim * 4, q_dim // 2)
        self.backbone_proj = nn.Sequential(nn.Linear(q_dim, ffn_interm_dim), nn.ReLU(), nn.Linear(ffn_interm_dim, q_dim // 2))
        self.backbone_pos_proj = nn.Sequential(nn.Linear(q_dim, ffn_interm_dim), nn.ReLU(), nn.Linear(ffn_interm_dim, q_dim // 2))
        self.answer_proj = nn.Sequential(nn.Linear(q_dim, ffn_interm_dim), nn.ReLU(), nn.Linear(ffn_interm_dim, q_dim))
        self.clip_token_proj = nn.Sequential(
            nn.Linear(1536, ffn_interm_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_interm_dim, q_dim))
        self.input_token_ln = nn.LayerNorm(q_dim)

        
        # self-attn
        
        self.self_attn = MultiheadAttention(q_dim, self.num_heads, dropout=dropout)
        self.self_attn_q_proj = nn.Linear(q_dim, q_dim)
        self.self_attn_k_proj = nn.Linear(q_dim, q_dim)
        self.self_attn_v_proj = nn.Linear(q_dim, q_dim)
        self.self_attn_ffn = nn.Sequential(
            nn.Linear(q_dim, ffn_interm_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(ffn_interm_dim, q_dim)
        )
        self.self_attn_ln1 = nn.LayerNorm(q_dim)
        self.self_attn_ln2 = nn.LayerNorm(q_dim)
        self.self_attn_dp1 = nn.Dropout(dropout)
        self.self_attn_dp2 = nn.Dropout(dropout)
    def forward(self,
            queries: Tensor, intermediate:List[Tensor], features: Tensor,
            q_pos: Tensor, k_pos: Tensor,
            q_attn_mask: Optional[Tensor] = None,
            qk_attn_mask: Optional[Tensor] = None,
            q_padding_mask: Optional[Tensor] = None,
            kv_padding_mask: Optional[Tensor] = None,
            llava_answer_idx:Optional[Tensor] = None,
            llava_feature: Optional[Tensor] = None
        ):
        """
        Parameters:
        -----------
        queries: Tensor
            Interaction queries of size (N, B, K).
        features: Tensor
            Image features of size (HW, B, C).
        q_attn_mask: Tensor, default: None
            Attention mask to be applied during the self attention of queries.
        qk_attn_mask: Tensor, default: None
            Attention mask to be applied during the cross attention from image
            features to interaction queries.
        q_padding_mask: Tensor, default: None
            Padding mask for interaction queries of size (B, N). Values of `True`
            indicate the corresponding query was padded and to be ignored.
        kv_padding_mask: Tensor, default: None
            Padding mask for image features of size (B, HW).
        q_pos: Tensor, default: None
            Positional encodings for the interaction queries.
        k_pos: Tensor, default: None
            Positional encodings for the image features.

        Returns:
        --------
        queries: Tensor
        """
        # process cls token
        q = self.query_adapter(queries)
        q_p = self.cls_pos_proj(torch.cat((q_pos["centre"], q_pos["box"]), dim=-1))
        n_q, bs, _ = q.shape
        q = q.view(n_q, bs, self.inter_head_dim, self.q_dim // self.inter_head_dim // 2)
        q_p = q_p.view(n_q, bs, self.inter_head_dim, self.q_dim // self.inter_head_dim // 2)
        cls_token = torch.cat([q, q_p], dim=3).view(n_q, bs, self.q_dim)
        num_cls_token = len(cls_token)

        # process vision token
        backbone_token = self.backbone_proj(features)
        backbone_token_pos = self.backbone_pos_proj(k_pos)
        hw, _, _ = backbone_token.shape
        backbone_token = backbone_token.view(hw, bs, self.inter_head_dim, self.q_dim // self.inter_head_dim // 2)
        backbone_token_pos = backbone_token_pos.view(hw, bs, self.inter_head_dim, self.q_dim // self.inter_head_dim // 2)
        backbone_token = torch.cat([backbone_token, backbone_token_pos], dim=3).view(hw, bs, self.q_dim)
        clip_token = self.clip_token_proj(llava_feature).unsqueeze(1)
        vision_token = torch.cat((backbone_token, clip_token), dim=0)
        num_backbone_token = len(backbone_token)
        num_clip_token = len(clip_token)

        # process answer token
        answer_token = self.llava_cali_embedding.weight[llava_answer_idx.unique()].unsqueeze(1)
        answer_token = self.answer_proj(answer_token)

        # register token
        reg_token = self.register_embedding.weight.unsqueeze(1)

        # concatenate all tokens
        if len(intermediate) == 0:
            input_tokens = self.input_token_ln(torch.cat((cls_token, vision_token, answer_token, reg_token), dim=0))
        else: # check history memory
            input_tokens = self.input_token_ln(torch.cat((cls_token, vision_token, answer_token, reg_token, torch.cat(intermediate, dim=0)), dim=0))
        input_masks = torch.zeros_like(input_tokens[:, :, 0]).to(torch.bool).transpose(0, 1)
        input_masks[:, num_cls_token:num_backbone_token+num_cls_token] = kv_padding_mask
        # perform self-attention
        q = self.self_attn_q_proj(cls_token)
        k = self.self_attn_k_proj(input_tokens)
        v = self.self_attn_v_proj(input_tokens)
        attn, attn_weights = self.self_attn(
            query=q, key=k, value=v,
            key_padding_mask=input_masks
        )
        x = self.self_attn_ln1(cls_token + self.self_attn_dp1(attn))
        x = self.self_attn_ln2(x + self.self_attn_dp2(self.self_attn_ffn(x)))
        # DETR=base CUDA_VISIBLE_DEVICES=0 python main.py --world-size 1 --pretrained /home/pvic/detr-r50-hicodet.pth --output-dir /home/MasterThesis/pvic/log/hico_r50_encoder_only_answer9_clipselfattn --port 1234 --epoch 15 --llava-answer-path /home/MasterThesis/hico_llava_answer_9 --llava-token-path /home/MasterThesis/hico_siglip_feature --repr-dim 512 --lr-drop 1
        return x

class TransformerDecoderNoTaskSpecified(nn.Module):

    def __init__(self, decoder_layer, num_layers, return_intermediate=True):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for i in range(num_layers)])
        self.num_layers = num_layers
        self.norm = nn.LayerNorm(decoder_layer.q_dim)
        self.return_intermediate = return_intermediate

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, queries, features,
            q_attn_mask: Optional[Tensor] = None,
            qk_attn_mask: Optional[Tensor] = None,
            q_padding_mask: Optional[Tensor] = None,
            kv_padding_mask: Optional[Tensor] = None,
            q_pos: Optional[Tensor] = None,
            k_pos: Optional[Tensor] = None,
            llava_answer_idx:Optional[Tensor] = None,
            llava_feature:Optional[Tensor] = None
        ):
        # Add support for zero layers
        if self.num_layers == 0:
            return queries.unsqueeze(0)
        # Explicitly handle zero-size queries
        if queries.numel() == 0:
            rp = self.num_layers if self.return_intermediate else 1
            return queries.unsqueeze(0).repeat(rp, 1, 1, 1)

        output = queries
        intermediate = []

        for layer in self.layers:
            history = [queries]
            history.extend(intermediate)
            history.pop(-1)
            output = layer(
                output, [], features,
                q_attn_mask=q_attn_mask,
                qk_attn_mask=qk_attn_mask,
                q_padding_mask=q_padding_mask,
                kv_padding_mask=kv_padding_mask,
                q_pos=q_pos, k_pos=k_pos,llava_answer_idx=llava_answer_idx, llava_feature=llava_feature
            )
            if self.return_intermediate:
                intermediate.append(self.norm(output))
                # intermediate.append(output)

        if self.return_intermediate:
            output = torch.stack(intermediate)
        else:
            output = self.norm(output).unsqueeze(0)
            # output = output.unsqueeze(0)
        return output

def _get_relative_position_bias(
    relative_position_bias_table: torch.Tensor, relative_position_index: torch.Tensor, window_size: List[int]
) -> torch.Tensor:
    N = window_size[0] * window_size[1]
    relative_position_bias = relative_position_bias_table[relative_position_index]  # type: ignore[index]
    relative_position_bias = relative_position_bias.view(N, N, -1)
    relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous().unsqueeze(0)
    return relative_position_bias

def stochastic_depth(input: Tensor, p: float, mode: str, training: bool = True) -> Tensor:
    """
    Implements the Stochastic Depth from `"Deep Networks with Stochastic Depth"
    <https://arxiv.org/abs/1603.09382>`_ used for randomly dropping residual
    branches of residual architectures.
    Args:
        input (Tensor[N, ...]): The input tensor or arbitrary dimensions with the first one
                    being its batch i.e. a batch with ``N`` rows.
        p (float): probability of the input to be zeroed.
        mode (str): ``"batch"`` or ``"row"``.
                    ``"batch"`` randomly zeroes the entire input, ``"row"`` zeroes
                    randomly selected rows from the batch.
        training: apply stochastic depth if is ``True``. Default: ``True``
    Returns:
        Tensor[N, ...]: The randomly zeroed tensor.
    """
    if p < 0.0 or p > 1.0:
        raise ValueError(f"drop probability has to be between 0 and 1, but got {p}")
    if mode not in ["batch", "row"]:
        raise ValueError(f"mode has to be either 'batch' or 'row', but got {mode}")
    if not training or p == 0.0:
        return input

    survival_rate = 1.0 - p
    if mode == "row":
        size = [input.shape[0]] + [1] * (input.ndim - 1)
    else:
        size = [1] * input.ndim
    noise = torch.empty(size, dtype=input.dtype, device=input.device)
    noise = noise.bernoulli_(survival_rate)
    if survival_rate > 0.0:
        noise.div_(survival_rate)
    return input * noise

class StochasticDepth(nn.Module):
    def __init__(self, p: float, mode: str) -> None:
        super().__init__()
        self.p = p
        self.mode = mode

    def forward(self, input: Tensor) -> Tensor:
        return stochastic_depth(input, self.p, self.mode, self.training)

    def __repr__(self) -> str:
        s = f"{self.__class__.__name__}(p={self.p}, mode={self.mode})"
        return s

def shifted_window_attention(
    input: Tensor,
    qkv_weight: Tensor,
    proj_weight: Tensor,
    relative_position_bias: Tensor,
    window_size: List[int],
    num_heads: int,
    shift_size: List[int],
    attention_dropout: float = 0.0,
    dropout: float = 0.0,
    qkv_bias: Optional[Tensor] = None,
    proj_bias: Optional[Tensor] = None,
    logit_scale: Optional[torch.Tensor] = None,
):
    """
    Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        input (Tensor[N, H, W, C]): The input tensor or 4-dimensions.
        qkv_weight (Tensor[in_dim, out_dim]): The weight tensor of query, key, value.
        proj_weight (Tensor[out_dim, out_dim]): The weight tensor of projection.
        relative_position_bias (Tensor): The learned relative position bias added to attention.
        window_size (List[int]): Window size.
        num_heads (int): Number of attention heads.
        shift_size (List[int]): Shift size for shifted window attention.
        attention_dropout (float): Dropout ratio of attention weight. Default: 0.0.
        dropout (float): Dropout ratio of output. Default: 0.0.
        qkv_bias (Tensor[out_dim], optional): The bias tensor of query, key, value. Default: None.
        proj_bias (Tensor[out_dim], optional): The bias tensor of projection. Default: None.
        logit_scale (Tensor[out_dim], optional): Logit scale of cosine attention for Swin Transformer V2. Default: None.
    Returns:
        Tensor[N, H, W, C]: The output tensor after shifted window attention.
    """
    B, H, W, C = input.shape
    # pad feature maps to multiples of window size
    pad_r = (window_size[1] - W % window_size[1]) % window_size[1]
    pad_b = (window_size[0] - H % window_size[0]) % window_size[0]
    x = F.pad(input, (0, 0, 0, pad_r, 0, pad_b))
    _, pad_H, pad_W, _ = x.shape

    shift_size = shift_size.copy()
    # If window size is larger than feature size, there is no need to shift window
    if window_size[0] >= pad_H:
        shift_size[0] = 0
    if window_size[1] >= pad_W:
        shift_size[1] = 0

    # cyclic shift
    if sum(shift_size) > 0:
        x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1]), dims=(1, 2))

    # partition windows
    num_windows = (pad_H // window_size[0]) * (pad_W // window_size[1])
    x = x.view(B, pad_H // window_size[0], window_size[0], pad_W // window_size[1], window_size[1], C)
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(B * num_windows, window_size[0] * window_size[1], C)  # B*nW, Ws*Ws, C

    # multi-head attention
    if logit_scale is not None and qkv_bias is not None:
        qkv_bias = qkv_bias.clone()
        length = qkv_bias.numel() // 3
        qkv_bias[length : 2 * length].zero_()
    qkv = F.linear(x, qkv_weight, qkv_bias)
    qkv = qkv.reshape(x.size(0), x.size(1), 3, num_heads, C // num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]
    if logit_scale is not None:
        # cosine attention
        attn = F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1)
        logit_scale = torch.clamp(logit_scale, max=math.log(100.0)).exp()
        attn = attn * logit_scale
    else:
        q = q * (C // num_heads) ** -0.5
        attn = q.matmul(k.transpose(-2, -1))
    # add relative position bias
    attn = attn + relative_position_bias

    if sum(shift_size) > 0:
        # generate attention mask
        attn_mask = x.new_zeros((pad_H, pad_W))
        h_slices = ((0, -window_size[0]), (-window_size[0], -shift_size[0]), (-shift_size[0], None))
        w_slices = ((0, -window_size[1]), (-window_size[1], -shift_size[1]), (-shift_size[1], None))
        count = 0
        for h in h_slices:
            for w in w_slices:
                attn_mask[h[0] : h[1], w[0] : w[1]] = count
                count += 1
        attn_mask = attn_mask.view(pad_H // window_size[0], window_size[0], pad_W // window_size[1], window_size[1])
        attn_mask = attn_mask.permute(0, 2, 1, 3).reshape(num_windows, window_size[0] * window_size[1])
        attn_mask = attn_mask.unsqueeze(1) - attn_mask.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        attn = attn.view(x.size(0) // num_windows, num_windows, num_heads, x.size(1), x.size(1))
        attn = attn + attn_mask.unsqueeze(1).unsqueeze(0)
        attn = attn.view(-1, num_heads, x.size(1), x.size(1))

    attn = F.softmax(attn, dim=-1)
    attn = F.dropout(attn, p=attention_dropout)

    x = attn.matmul(v).transpose(1, 2).reshape(x.size(0), x.size(1), C)
    x = F.linear(x, proj_weight, proj_bias)
    x = F.dropout(x, p=dropout)

    # reverse windows
    x = x.view(B, pad_H // window_size[0], pad_W // window_size[1], window_size[0], window_size[1], C)
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, pad_H, pad_W, C)

    # reverse cyclic shift
    if sum(shift_size) > 0:
        x = torch.roll(x, shifts=(shift_size[0], shift_size[1]), dims=(1, 2))

    # unpad features
    x = x[:, :H, :W, :].contiguous()
    return x

class ShiftedWindowAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        window_size: List[int],
        shift_size: List[int],
        num_heads: int,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attention_dropout: float = 0.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        if len(window_size) != 2 or len(shift_size) != 2:
            raise ValueError("window_size and shift_size must be of length 2")
        self.window_size = window_size
        self.shift_size = shift_size
        self.num_heads = num_heads
        self.attention_dropout = attention_dropout
        self.dropout = dropout

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)

        self.define_relative_position_bias_table()
        self.define_relative_position_index()

    def define_relative_position_bias_table(self):
        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1), self.num_heads)
        )  # 2*Wh-1 * 2*Ww-1, nH
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def define_relative_position_index(self):
        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid(coords_h, coords_w))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1).flatten()  # Wh*Ww*Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

    def get_relative_position_bias(self) -> torch.Tensor:
        return _get_relative_position_bias(
            self.relative_position_bias_table, self.relative_position_index, self.window_size  # type: ignore[arg-type]
        )

    def forward(self, x: Tensor):
        """
        Args:
            x (Tensor): Tensor with layout of [B, H, W, C]
        Returns:
            Tensor with same layout as input, i.e. [B, H, W, C]
        """
        relative_position_bias = self.get_relative_position_bias()
        return shifted_window_attention(
            x,
            self.qkv.weight,
            self.proj.weight,
            relative_position_bias,
            self.window_size,
            self.num_heads,
            shift_size=self.shift_size,
            attention_dropout=self.attention_dropout,
            dropout=self.dropout,
            qkv_bias=self.qkv.bias,
            proj_bias=self.proj.bias,
        )

class ShiftedWindowAttentionV2(ShiftedWindowAttention):
    def __init__(
        self,
        dim: int,
        window_size: List[int],
        shift_size: List[int],
        num_heads: int,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attention_dropout: float = 0.0,
        dropout: float = 0.0,
    ):
        super().__init__(
            dim,
            window_size,
            shift_size,
            num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attention_dropout=attention_dropout,
            dropout=dropout,
        )

        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))))
        # mlp to generate continuous relative position bias
        self.cpb_mlp = nn.Sequential(
            nn.Linear(2, 512, bias=True), nn.ReLU(inplace=True), nn.Linear(512, num_heads, bias=False)
        )
        if qkv_bias:
            length = self.qkv.bias.numel() // 3
            self.qkv.bias[length : 2 * length].data.zero_()

    def define_relative_position_bias_table(self):
        # get relative_coords_table
        relative_coords_h = torch.arange(-(self.window_size[0] - 1), self.window_size[0], dtype=torch.float32)
        relative_coords_w = torch.arange(-(self.window_size[1] - 1), self.window_size[1], dtype=torch.float32)
        relative_coords_table = torch.stack(torch.meshgrid(relative_coords_h, relative_coords_w))
        relative_coords_table = relative_coords_table.permute(1, 2, 0).contiguous().unsqueeze(0)  # 1, 2*Wh-1, 2*Ww-1, 2

        relative_coords_table[:, :, :, 0] /= self.window_size[0] - 1
        relative_coords_table[:, :, :, 1] /= self.window_size[1] - 1

        relative_coords_table *= 8  # normalize to -8, 8
        relative_coords_table = (
            torch.sign(relative_coords_table) * torch.log2(torch.abs(relative_coords_table) + 1.0) / 3.0
        )
        self.register_buffer("relative_coords_table", relative_coords_table)

    def get_relative_position_bias(self) -> torch.Tensor:
        relative_position_bias = _get_relative_position_bias(
            self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads),
            self.relative_position_index,  # type: ignore[arg-type]
            self.window_size,
        )
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        return relative_position_bias

    def forward(self, x: Tensor):
        """
        Args:
            x (Tensor): Tensor with layout of [B, H, W, C]
        Returns:
            Tensor with same layout as input, i.e. [B, H, W, C]
        """
        relative_position_bias = self.get_relative_position_bias()
        return shifted_window_attention(
            x,
            self.qkv.weight,
            self.proj.weight,
            relative_position_bias,
            self.window_size,
            self.num_heads,
            shift_size=self.shift_size,
            attention_dropout=self.attention_dropout,
            dropout=self.dropout,
            qkv_bias=self.qkv.bias,
            proj_bias=self.proj.bias,
            logit_scale=self.logit_scale,
        )

class MLP(torch.nn.Sequential):
    """This block implements the multi-layer perceptron (MLP) module.
    Args:
        in_channels (int): Number of channels of the input
        hidden_channels (List[int]): List of the hidden channel dimensions
        norm_layer (Callable[..., torch.nn.Module], optional): Norm layer that will be stacked on top of the linear layer. If ``None`` this layer won't be used. Default: ``None``
        activation_layer (Callable[..., torch.nn.Module], optional): Activation function which will be stacked on top of the normalization layer (if not None), otherwise on top of the linear layer. If ``None`` this layer won't be used. Default: ``torch.nn.ReLU``
        inplace (bool, optional): Parameter for the activation layer, which can optionally do the operation in-place.
            Default is ``None``, which uses the respective default values of the ``activation_layer`` and Dropout layer.
        bias (bool): Whether to use bias in the linear layer. Default ``True``
        dropout (float): The probability for the dropout layer. Default: 0.0
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: List[int],
        norm_layer: Optional[Callable[..., torch.nn.Module]] = None,
        activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
        inplace: Optional[bool] = None,
        bias: bool = True,
        dropout: float = 0.0,
    ):
        # The addition of `norm_layer` is inspired from the implementation of TorchMultimodal:
        # https://github.com/facebookresearch/multimodal/blob/5dec8a/torchmultimodal/modules/layers/mlp.py
        params = {} if inplace is None else {"inplace": inplace}

        layers = []
        in_dim = in_channels
        for hidden_dim in hidden_channels[:-1]:
            layers.append(torch.nn.Linear(in_dim, hidden_dim, bias=bias))
            if norm_layer is not None:
                layers.append(norm_layer(hidden_dim))
            layers.append(activation_layer(**params))
            layers.append(torch.nn.Dropout(dropout, **params))
            in_dim = hidden_dim

        layers.append(torch.nn.Linear(in_dim, hidden_channels[-1], bias=bias))
        layers.append(torch.nn.Dropout(dropout, **params))

        super().__init__(*layers)

class SwinTransformerBlock(nn.Module):
    """
    Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (List[int]): Window size.
        shift_size (List[int]): Shift size for shifted window attention.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.0.
        dropout (float): Dropout rate. Default: 0.0.
        attention_dropout (float): Attention dropout rate. Default: 0.0.
        stochastic_depth_prob: (float): Stochastic depth rate. Default: 0.0.
        norm_layer (nn.Module): Normalization layer.  Default: nn.LayerNorm.
        attn_layer (nn.Module): Attention layer. Default: ShiftedWindowAttention
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: List[int],
        shift_size: List[int],
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        stochastic_depth_prob: float = 0.0,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        attn_layer: Callable[..., nn.Module] = ShiftedWindowAttention,
    ):
        super().__init__()

        self.norm1 = norm_layer(dim)
        self.attn = attn_layer(
            dim,
            window_size,
            shift_size,
            num_heads,
            attention_dropout=attention_dropout,
            dropout=dropout,
        )
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")
        self.norm2 = norm_layer(dim)
        self.mlp = MLP(dim, [int(dim * mlp_ratio), dim], activation_layer=nn.GELU, inplace=None, dropout=dropout)

        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)

    def forward(self, x: Tensor):
        x = x + self.stochastic_depth(self.attn(self.norm1(x)))
        x = x + self.stochastic_depth(self.mlp(self.norm2(x)))
        return x


class SwinTransformerBlockV2(SwinTransformerBlock):
    """
    Swin Transformer V2 Block.
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (List[int]): Window size.
        shift_size (List[int]): Shift size for shifted window attention.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.0.
        dropout (float): Dropout rate. Default: 0.0.
        attention_dropout (float): Attention dropout rate. Default: 0.0.
        stochastic_depth_prob: (float): Stochastic depth rate. Default: 0.0.
        norm_layer (nn.Module): Normalization layer.  Default: nn.LayerNorm.
        attn_layer (nn.Module): Attention layer. Default: ShiftedWindowAttentionV2.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: List[int],
        shift_size: List[int],
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        stochastic_depth_prob: float = 0.0,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        attn_layer: Callable[..., nn.Module] = ShiftedWindowAttentionV2,
    ):
        super().__init__(
            dim,
            num_heads,
            window_size,
            shift_size,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            attention_dropout=attention_dropout,
            stochastic_depth_prob=stochastic_depth_prob,
            norm_layer=norm_layer,
            attn_layer=attn_layer,
        )

    def forward(self, x: Tensor):
        # Here is the difference, we apply norm after the attention in V2.
        # In V1 we applied norm before the attention.
        x = x + self.stochastic_depth(self.norm1(self.attn(x)))
        x = x + self.stochastic_depth(self.norm2(self.mlp(x)))
        return x

class SwinTransformer(nn.Module):

    def __init__(self, dim, num_layers, num_heads):
        """
        A feature stage consisting of a series of Swin Transformer V2 blocks.

        Parameters:
        -----------
        dim: int
            Dimension of the input features.
        """
        super().__init__()
        self.dim = dim

        self.depth = num_layers
        self.num_heads = num_heads
        self.window_size = 8
        self.base_sd_prob = 0.2

        shift_size = [
            [self.window_size // 2] * 2 if i % 2
            else [0, 0] for i in range(self.depth)
        ]
        # TODO fix this hack
        # Use stochastic depth parameters from the third stage of Swin-T variant.
        # In practice, varying this value does not make a significant difference.
        sd_prob = (torch.linspace(0, 1, 12)[10-num_layers:10] * self.base_sd_prob).tolist()

        blocks: List[nn.Module] = []
        for i in range(self.depth):
            blocks.append(SwinTransformerBlockV2(
                dim=dim, num_heads=self.num_heads,
                window_size=[self.window_size] * 2,
                shift_size=shift_size[i],
                stochastic_depth_prob=sd_prob[i]
            ))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        """
        Parameters:
        -----------
        x: Tensor
            Input features maps of shape (B, H, W, C).

        Returns:
        --------
        Tensor
            Output feature maps of shape (B, H, W, C).
        """
        return self.blocks(x)