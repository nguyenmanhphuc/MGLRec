import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np

from models.layers import NormAttentionGATEncoder


class GRRecModel(nn.Module):
    def __init__(self, num_users, num_items, dim_u, dim_i, hidden_dim, num_layers):
        super().__init__()
        # embeddings
        self.user_emb = nn.Embedding(num_users, dim_u)
        self.item_emb = nn.Embedding(num_items, dim_i)
        # encoder
        self.encoder = NormAttentionGATEncoder(dim_u, dim_i, hidden_dim, num_layers)
        # reconstruction parameters
        self.W_e = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        self.user_mask = nn.Parameter(torch.randn(dim_u))
        self.item_mask = nn.Parameter(torch.randn(dim_i))
        self.dec_u = nn.Linear(hidden_dim, dim_u)
        self.dec_i = nn.Linear(hidden_dim, dim_i)

    def mask_edges(self, edge_index, mask_rate):
        E = edge_index.size(1)
        mask = torch.rand(E, device=edge_index.device) < mask_rate
        return edge_index[:, mask]

    def mask_features(self, feats, mask_vec, mask_rate):
        N = feats.size(0)
        mask = torch.rand(N, device=feats.device) < mask_rate
        feats_masked = feats.clone()
        feats_masked[mask] = mask_vec
        return feats_masked, mask

    def encode(self, edge_index, u_feats, i_feats):
        u_h = self.encoder(u_feats, i_feats, edge_index)
        i_h = self.encoder(i_feats, u_feats, edge_index[[1, 0]])
        return u_h, i_h

    def compute_er_loss(self, u_h, i_h, pos_edge, neg_edge):
        src_p, dst_p = pos_edge
        src_n, dst_n = neg_edge
        pos = (u_h[src_p] * (i_h[dst_p] @ self.W_e.t())).sum(dim=1)
        neg = (u_h[src_n] * (i_h[dst_n] @ self.W_e.t())).sum(dim=1)
        logits = torch.cat([pos, neg], dim=0)
        labels = torch.cat([torch.ones_like(pos), torch.zeros_like(neg)], dim=0)
        return F.binary_cross_entropy_with_logits(logits, labels)

    def compute_fr_loss(self, orig_feats, rec_feats, mask, gamma=1.0):
        orig = orig_feats[mask]
        rec = rec_feats[mask]
        cos = F.cosine_similarity(orig, rec, dim=1)
        return ((1 - cos) ** gamma).mean()

    def compute_bpr_loss(self, u_h, i_h, users, pos_items, neg_items):
        u_vec = u_h[users]
        pos_vec = i_h[pos_items]
        neg_vec = i_h[neg_items]
        pos_scores = (u_vec * pos_vec).sum(dim=1)
        neg_scores = (u_vec * neg_vec).sum(dim=1)
        return -F.logsigmoid(pos_scores - neg_scores).mean()

    def forward(self, edge_index, p_e, p_a_u, p_a_i,
                pos_edge, neg_edge, users, pos_items, neg_items,
                lambda_er, mu_fr):
        # Edge Reconstruction
        edge_masked = self.mask_edges(edge_index, p_e)
        u_h_er, i_h_er = self.encode(edge_masked,
                                       self.user_emb.weight,
                                       self.item_emb.weight)
        er_loss = self.compute_er_loss(u_h_er, i_h_er, pos_edge, neg_edge)
        # Feature Recovery
        u_feats, u_mask = self.mask_features(self.user_emb.weight,
                                             self.user_mask, p_a_u)
        i_feats, i_mask = self.mask_features(self.item_emb.weight,
                                             self.item_mask, p_a_i)
        u_h_fr, i_h_fr = self.encode(edge_index, u_feats, i_feats)
        u_rec = self.dec_u(u_h_fr)
        i_rec = self.dec_i(i_h_fr)
        fr_loss = self.compute_fr_loss(self.user_emb.weight, u_rec, u_mask)
        fr_loss += self.compute_fr_loss(self.item_emb.weight, i_rec, i_mask)
        # Recommendation
        u_h_rec, i_h_rec = self.encode(edge_index,
                                       self.user_emb.weight,
                                       self.item_emb.weight)
        bpr_loss = self.compute_bpr_loss(u_h_rec, i_h_rec,
                                         users, pos_items, neg_items)
        # Joint Loss
        return bpr_loss + lambda_er * er_loss + mu_fr * fr_loss