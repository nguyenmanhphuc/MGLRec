import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np

class NormAttentionGATLayer(nn.Module):
    def __init__(self, in_dim_u, in_dim_i, out_dim):
        super().__init__()
        self.proj_u = nn.Linear(in_dim_u, out_dim, bias=False)
        self.proj_i = nn.Linear(in_dim_i, out_dim, bias=False)
        self.attn_vec = nn.Parameter(torch.randn(2 * out_dim))

    def forward(self, user_feats, item_feats, edge_index):
        src, dst = edge_index
        h_u = self.proj_u(user_feats[src])
        h_i = self.proj_i(item_feats[dst])
        h_cat = torch.cat([h_u, h_i], dim=1)
        e = F.leaky_relu((h_cat * self.attn_vec).sum(dim=1))
        alpha = F.softmax(e, dim=0)
        agg = torch.zeros_like(user_feats)
        agg.index_add_(0, src, alpha.unsqueeze(1) * h_i)
        return F.elu(agg)

class NormAttentionGATEncoder(nn.Module):
    def __init__(self, in_dim_u, in_dim_i, hidden_dim, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([
            NormAttentionGATLayer(
                in_dim_u if i == 0 else hidden_dim,
                in_dim_i if i == 0 else hidden_dim,
                hidden_dim
            ) for i in range(num_layers)
        ])
        self.query = nn.Parameter(torch.randn(hidden_dim))

    def forward(self, u_feats, i_feats, edge_index):
        h_list = []
        u_h = u_feats
        for layer in self.layers:
            u_h = layer(u_h, i_feats, edge_index)
            h_list.append(u_h)
        h_stack = torch.stack(h_list, dim=0)
        scores = torch.einsum('lnd,d->ln', h_stack, self.query)
        w = F.softmax(scores, dim=0).unsqueeze(-1)
        h = (w * h_stack).sum(dim=0)
        return F.normalize(h, dim=1)