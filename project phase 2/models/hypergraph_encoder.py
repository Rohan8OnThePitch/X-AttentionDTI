import torch
import torch.nn as nn
import torch.nn.functional as F


class HypergraphConv(nn.Module):
    """
    Single Hypergraph Convolution Layer:
    Node -> Hyperedge -> Node  (with node-degree normalization)
    """
    def __init__(self, in_dim, out_dim, dropout=0.1):
        super().__init__()

        self.node_to_hedge = nn.Linear(in_dim, out_dim)
        self.hedge_to_node = nn.Linear(out_dim, out_dim)

        self.dropout = nn.Dropout(dropout)
        self.norm    = nn.LayerNorm(out_dim)
        self.act     = nn.GELU()

    def forward(self, x, hyperedge_indices, hyperedge_types, hyperedge_type_emb):
        """
        x                  : [Total_Nodes, In_Dim]
        hyperedge_indices  : [2, E]   — (node_idx, hedge_idx) pairs
        hyperedge_types    : [Total_Hyperedges]
        hyperedge_type_emb : nn.Embedding(3, D)
        """
        node_idx, hedge_idx = hyperedge_indices[0], hyperedge_indices[1]
        num_hyperedges = hyperedge_types.size(0)

        # ── 1. Node → Hyperedge (degree-normalised aggregation) ──────────
        x_trans   = self.node_to_hedge(x)                        # [N, D]

        hedge_aggr = torch.zeros(
            num_hyperedges, x_trans.size(1),
            device=x.device, dtype=x.dtype,
        )
        hedge_aggr = hedge_aggr.index_add(
            0, hedge_idx, x_trans[node_idx].to(dtype=x.dtype)
        )

        # Hyperedge degree normalisation (avg over member nodes)
        hedge_degree = torch.zeros(num_hyperedges, 1, device=x.device, dtype=x.dtype)
        hedge_degree = hedge_degree.index_add(
            0, hedge_idx,
            torch.ones(node_idx.size(0), 1, device=x.device, dtype=x.dtype),
        ).clamp(min=1.0)

        hedge_msg = hedge_aggr / hedge_degree            # [H, D]

        # Inject hyperedge-type bias
        hedge_msg = hedge_msg + hyperedge_type_emb(hyperedge_types)

        # ── 2. Hyperedge → Node (node-degree-normalised distribution) ────
        messages = hedge_msg[hedge_idx]                  # [E, D]

        node_aggr = torch.zeros_like(x_trans)
        node_aggr = node_aggr.index_add(0, node_idx, messages.to(dtype=node_aggr.dtype))

        # FIX: divide by number of hyperedges each node belongs to
        node_degree = torch.zeros(x.size(0), 1, device=x.device, dtype=x.dtype)
        node_degree = node_degree.index_add(
            0, node_idx,
            torch.ones(node_idx.size(0), 1, device=x.device, dtype=x.dtype),
        ).clamp(min=1.0)

        node_aggr = node_aggr / node_degree              # key fix — equal signal per atom

        out = self.hedge_to_node(node_aggr)
        out = self.dropout(out)

        # Residual + Norm + Activation
        if x.shape[-1] == out.shape[-1]:
            out = x + out

        out = self.norm(out)
        out = self.act(out)

        return out


class HypergraphDrugEncoder(nn.Module):
    """
    Deep Hypergraph Encoder with Residuals.
    Returns both the global graph embedding AND per-node embeddings
    (needed for the P2D cross-attention in the main model).
    """

    def __init__(self, node_feat_dim=49, hidden_dim=512, num_layers=3, dropout=0.1):
        super().__init__()

        self.input_proj = nn.Linear(node_feat_dim, hidden_dim)

        self.hyperedge_type_emb = nn.Embedding(3, hidden_dim)

        self.layers = nn.ModuleList([
            HypergraphConv(hidden_dim, hidden_dim, dropout)
            for _ in range(num_layers)
        ])

        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, node_features, hyperedge_indices, hyperedge_types, batch_indices):
        """
        Returns
        -------
        graph_emb   : [B, D]           — pooled global drug embedding
        node_embs   : [Total_Nodes, D] — per-atom embeddings (before output_proj)
        batch_indices : [Total_Nodes]  — which graph each node belongs to
        """
        # 1. Initial projection
        x = self.input_proj(node_features)          # [N, D]

        # 2. Message-passing layers
        for layer in self.layers:
            x = layer(x, hyperedge_indices, hyperedge_types, self.hyperedge_type_emb)

        # node_embs before output_proj — richer, higher-dim features for cross-attention
        node_embs = x                               # [N, D]

        # 3. Global readout (mean pooling)
        num_graphs  = batch_indices.max().item() + 1
        graph_emb   = torch.zeros(num_graphs, x.size(1), device=x.device, dtype=x.dtype)
        graph_emb   = graph_emb.index_add(0, batch_indices, x)

        batch_counts = (
            torch.bincount(batch_indices, minlength=num_graphs)
            .unsqueeze(-1).to(dtype=x.dtype)
            .clamp(min=1.0)
        )
        graph_emb = graph_emb / batch_counts        # [B, D]

        return self.output_proj(graph_emb), node_embs, batch_indices
