# file: model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import EsmModel

from models.hypergraph_encoder import HypergraphDrugEncoder


def _pad_node_sequences(node_embs, batch_indices, num_graphs):
    """
    node_embs     : [Total_Nodes, D]
    batch_indices : [Total_Nodes]  LongTensor — graph id per node
    num_graphs    : int

    Returns
    -------
    padded  : [B, N_max, D]   — zero-padded per-graph node sequences
    mask    : [B, N_max]  bool — True where a real node exists
    """
    D       = node_embs.size(1)
    counts  = torch.bincount(batch_indices, minlength=num_graphs)
    max_n   = int(counts.max().item())

    padded  = node_embs.new_zeros(num_graphs, max_n, D)
    mask    = torch.zeros(num_graphs, max_n, dtype=torch.bool, device=node_embs.device)

    for g in range(num_graphs):
        idx = (batch_indices == g).nonzero(as_tuple=True)[0]
        n   = idx.size(0)
        padded[g, :n] = node_embs[idx]
        mask[g, :n]   = True

    return padded, mask


# Student Protein Encoder (configurable ESM-2 variant)
class StudentProteinEncoder(nn.Module):
    """
    Pretrained ESM-2 student encoder.
    esm_model_name : HuggingFace model ID  (default: ESM-2 35M)
    esm_out_dim    : hidden size of that model (480 for 35M, 640 for 150M)
    Projects esm_out_dim → hidden_dim.
    """

    def __init__(self, hidden_dim=512,
                 esm_model_name="facebook/esm2_t12_35M_UR50D",
                 esm_out_dim=480):
        super().__init__()
        self.encoder    = EsmModel.from_pretrained(esm_model_name)
        # Freeze backbone, then unfreeze the last 2 layers (10, 11) for fine-tuning.
        self.encoder.requires_grad_(False)
        for name, param in self.encoder.named_parameters():
             if "encoder.layer.10" in name or "encoder.layer.11" in name or "contact_head" in name:
                 param.requires_grad = True

        self.projection = nn.Linear(esm_out_dim, hidden_dim)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        seq = self.projection(out.last_hidden_state)   # [B, L, H]
        cls = seq[:, 0, :]                             # [B, H]
        return seq, cls


# Full DTI Model
class DTIModel(nn.Module):
    def __init__(self, hidden_dim=512,
                 esm_model_name="facebook/esm2_t12_35M_UR50D",
                 esm_out_dim=480):
        super().__init__()

        self.student_encoder = StudentProteinEncoder(
            hidden_dim=hidden_dim,
            esm_model_name=esm_model_name,
            esm_out_dim=esm_out_dim,
        )

        self.drug_encoder = HypergraphDrugEncoder(
            node_feat_dim=49,
            hidden_dim=hidden_dim,
        )

        self.align_student = nn.Linear(hidden_dim, hidden_dim)
        self.align_teacher = nn.Linear(1280, hidden_dim)   # ESM-2 650M dim

        # Learnable fusion & hyperparameters
        self.lambda_fusion = nn.Parameter(torch.tensor(0.0))
        self.alpha_raw = nn.Parameter(torch.tensor(-2.95))  # logit space for ~0.05
        self.beta_raw  = nn.Parameter(torch.tensor(0.0))    # logit space for 0.5

        # Bidirectional Cross-Attention
        self.cross_attn_d2p = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=8, batch_first=True,
        )
        self.cross_attn_p2d = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=8, batch_first=True,
        )

        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 3),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        node_features, hyperedge_indices, hyperedge_types, batch_indices,
        protein_input_ids,
        protein_attention_mask,
        teacher_cls,
    ):
        B = teacher_cls.size(0)

        drug_global, drug_nodes, drug_batch_idx = self.drug_encoder(
            node_features, hyperedge_indices, hyperedge_types, batch_indices,
        )

        drug_seq, drug_key_mask = _pad_node_sequences(drug_nodes, drug_batch_idx, B)
        drug_padding_mask = ~drug_key_mask   # [B, N_max]

        protein_seq, student_cls = self.student_encoder(
            protein_input_ids, protein_attention_mask,
        )

        student_aligned = self.align_student(student_cls)   # [B, H]
        teacher_aligned = self.align_teacher(teacher_cls)   # [B, H]

        # Learnable fusion weight (clamped to min 0.2 so student always has a voice)
        wt = torch.sigmoid(self.lambda_fusion).clamp(min=0.2)
        protein_fused_cls = wt * student_aligned + (1.0 - wt) * teacher_aligned

        protein_seq = protein_seq.clone()
        protein_seq[:, 0, :] = protein_fused_cls

        drug_query = drug_global.unsqueeze(1)               # [B, 1, D]

        d2p_out, _ = self.cross_attn_d2p(
            query=drug_query,
            key=protein_seq,
            value=protein_seq,
            key_padding_mask=~protein_attention_mask.bool(),
        )
        d2p_out = d2p_out.squeeze(1)                        # [B, D]

        p2d_out, _ = self.cross_attn_p2d(
            query=protein_seq,                              # [B, L, D]
            key=drug_seq,                                   # [B, N_max, D]
            value=drug_seq,
            key_padding_mask=drug_padding_mask,             # [B, N_max]
        )                                                   # → [B, L, D]

        prot_mask  = protein_attention_mask.unsqueeze(-1).float()
        p2d_global = (p2d_out * prot_mask).sum(dim=1) / prot_mask.sum(dim=1).clamp(min=1e-9)

        combined      = torch.cat([drug_global, d2p_out, p2d_global], dim=-1)
        pred_affinity = self.predictor(combined).squeeze(-1)

        return {
            "pred_affinity": pred_affinity,
            "student_emb":   student_aligned,
            "teacher_emb":   teacher_aligned,
            "lambda":        wt.detach(),
            "alpha":         torch.sigmoid(self.alpha_raw).clamp(min=0.01),
            "beta":          torch.sigmoid(self.beta_raw).clamp(min=0.02),
        }
