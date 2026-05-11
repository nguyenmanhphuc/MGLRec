import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.layers import NormAttentionGATEncoder


class GRRecModel(nn.Module):
    def __init__(
        self,
        num_users,
        num_items,
        dim_u,
        dim_i,
        hidden_dim,
        num_layers,
        dropout=0.0,
        gamma_fr=1.0,
        delta_er=1.0
    ):
        super().__init__()

        self.num_users = num_users
        self.num_items = num_items
        self.dim_u = dim_u
        self.dim_i = dim_i
        self.hidden_dim = hidden_dim
        self.gamma_fr = gamma_fr
        self.delta_er = delta_er

        # ============================================================
        # Initial learnable user/item features
        # ============================================================
        self.user_emb = nn.Embedding(num_users, dim_u)
        self.item_emb = nn.Embedding(num_items, dim_i)

        # ============================================================
        self.encoder = NormAttentionGATEncoder(
            in_dim_u=dim_u,
            in_dim_i=dim_i,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout
        )

        # ============================================================
        self.decoder = NormAttentionGATEncoder(
            in_dim_u=hidden_dim,
            in_dim_i=hidden_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout
        )

        self.W_e = nn.Parameter(torch.empty(hidden_dim, hidden_dim))

        self.user_mask = nn.Parameter(torch.empty(dim_u))
        self.item_mask = nn.Parameter(torch.empty(dim_i))

        self.user_remask = nn.Parameter(torch.empty(hidden_dim))
        self.item_remask = nn.Parameter(torch.empty(hidden_dim))

        self.dec_u = nn.Linear(hidden_dim, dim_u)
        self.dec_i = nn.Linear(hidden_dim, dim_i)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)

        nn.init.xavier_uniform_(self.W_e)

        nn.init.normal_(self.user_mask, mean=0.0, std=0.02)
        nn.init.normal_(self.item_mask, mean=0.0, std=0.02)
        nn.init.normal_(self.user_remask, mean=0.0, std=0.02)
        nn.init.normal_(self.item_remask, mean=0.0, std=0.02)

        nn.init.xavier_uniform_(self.dec_u.weight)
        nn.init.zeros_(self.dec_u.bias)
        nn.init.xavier_uniform_(self.dec_i.weight)
        nn.init.zeros_(self.dec_i.bias)


    def mask_edges(self, edge_index, mask_rate):

        if mask_r   ate <= 0.0:
            return edge_index

        E = edge_index.size(1)
        if E == 0:
            return edge_index

        keep = torch.rand(E, device=edge_index.device) >= mask_rate

        # Avoid empty graph in extreme cases.
        if keep.sum() == 0:
            rand_idx = torch.randint(0, E, (1,), device=edge_index.device)
            keep[rand_idx] = True

        return edge_index[:, keep]

    def mask_features(self, feats, mask_vec, mask_rate):

        N = feats.size(0)
        mask = torch.rand(N, device=feats.device) < mask_rate

        feats_masked = feats.clone()
        if mask.any():
            feats_masked[mask] = mask_vec

        return feats_masked, mask

    def remask_hidden(self, h, mask, remask_vec):
        h_masked = h.clone()
        if mask.any():
            h_masked[mask] = remask_vec
        return h_masked

    def _unpack_edge_pair(self, edge_pair, device):

        if isinstance(edge_pair, (tuple, list)):
            src, dst = edge_pair
        else:
            src, dst = edge_pair[0], edge_pair[1]

        src = src.to(device).long()
        dst = dst.to(device).long()
        return src, dst

    # ================================================================
    # Encoder / decoder
    # ================================================================

    def encode(self, edge_index, u_feats, i_feats):

        return self.encoder(u_feats, i_feats, edge_index)

    def decode_graph(self, edge_index, u_h, i_h):
        return self.decoder(u_h, i_h, edge_index)

    # ================================================================
    # Loss functions
    # ================================================================

    def edge_logits(self, u_h, i_h, edge_pair):

        device = u_h.device
        src, dst = self._unpack_edge_pair(edge_pair, device)

        u_vec = u_h[src]                       # [B, hidden_dim]
        i_vec = F.linear(i_h[dst], self.W_e)   # i_h @ W_e.T

        logits = (u_vec * i_vec).sum(dim=-1)
        return logits

    def compute_er_loss(self, u_h_dec, i_h_dec, pos_edge, neg_edge):

        pos_logits = self.edge_logits(u_h_dec, i_h_dec, pos_edge)
        neg_logits = self.edge_logits(u_h_dec, i_h_dec, neg_edge)

        logits = torch.cat([pos_logits, neg_logits], dim=0)

        labels = torch.cat([
            torch.ones_like(pos_logits),
            torch.zeros_like(neg_logits)
        ], dim=0)

        # Optional positive weighting, similar in spirit to delta in paper.
        if self.delta_er != 1.0:
            weights = torch.cat([
                torch.full_like(pos_logits, self.delta_er),
                torch.ones_like(neg_logits)
            ], dim=0)

            return F.binary_cross_entropy_with_logits(
                logits,
                labels,
                weight=weights
            )

        return F.binary_cross_entropy_with_logits(logits, labels)

    def compute_fr_loss(self, orig_feats, rec_feats, mask):

        if mask.sum() == 0:
            return rec_feats.new_tensor(0.0)

        orig = orig_feats[mask].detach()
        rec = rec_feats[mask]

        cos = F.cosine_similarity(orig, rec, dim=-1, eps=1e-12)
        loss = (1.0 - cos).clamp_min(0.0).pow(self.gamma_fr).mean()

        return loss

    def compute_bpr_loss(self, u_h, i_h, users, pos_items, neg_items):
        users = users.to(u_h.device).long()
        pos_items = pos_items.to(i_h.device).long()
        neg_items = neg_items.to(i_h.device).long()

        u_vec = u_h[users]
        pos_vec = i_h[pos_items]
        neg_vec = i_h[neg_items]

        pos_scores = (u_vec * pos_vec).sum(dim=-1)
        neg_scores = (u_vec * neg_vec).sum(dim=-1)

        # Equivalent to -logsigmoid(pos - neg), but numerically stable.
        return F.softplus(neg_scores - pos_scores).mean()

    # ================================================================
    # Forward
    # ================================================================

    def forward(
        self,
        edge_index,
        p_e,
        p_a_u,
        p_a_i,
        pos_edge,
        neg_edge,
        users,
        pos_items,
        neg_items,
        lambda_er,
        mu_fr,
        return_dict=False
    ):
        """
        Total objective:

            L = L_REC + lambda_er * L_ER + mu_fr * L_FR
        """

        raw_u = self.user_emb.weight
        raw_i = self.item_emb.weight

        # ============================================================
        # 1) Edge Reconstruction Branch
        # ============================================================
        edge_masked = self.mask_edges(edge_index, p_e)

        # H1 = f_E(A_tilde, X)
        u_h_er, i_h_er = self.encode(
            edge_masked,
            raw_u,
            raw_i
        )

        # H2 = f_D(A_tilde, H1)
        u_h_er_dec, i_h_er_dec = self.decode_graph(
            edge_masked,
            u_h_er,
            i_h_er
        )

        # L_ER
        er_loss = self.compute_er_loss(
            u_h_er_dec,
            i_h_er_dec,
            pos_edge,
            neg_edge
        )

        # ============================================================
        # 2) Feature Recovery Branch
        # ============================================================
        # Mask raw node features
        u_feats_masked, u_mask = self.mask_features(
            raw_u,
            self.user_mask,
            p_a_u
        )

        i_feats_masked, i_mask = self.mask_features(
            raw_i,
            self.item_mask,
            p_a_i
        )

        # H3 = f_E(A, X_tilde)
        u_h_fr, i_h_fr = self.encode(
            edge_index,
            u_feats_masked,
            i_feats_masked
        )

        # Re-masking on encoded hidden representations
        u_h_fr_remask = self.remask_hidden(
            u_h_fr,
            u_mask,
            self.user_remask
        )

        i_h_fr_remask = self.remask_hidden(
            i_h_fr,
            i_mask,
            self.item_remask
        )

        # Z_hidden = f_D(A, H3_tilde)
        u_z_hidden, i_z_hidden = self.decode_graph(
            edge_index,
            u_h_fr_remask,
            i_h_fr_remask
        )

        # Project hidden decoded representations back to raw feature space
        u_rec = self.dec_u(u_z_hidden)
        i_rec = self.dec_i(i_z_hidden)

        # L_FR
        fr_loss_u = self.compute_fr_loss(raw_u, u_rec, u_mask)
        fr_loss_i = self.compute_fr_loss(raw_i, i_rec, i_mask)
        fr_loss = fr_loss_u + fr_loss_i


        u_h_rec = 0.5 * (u_h_er + u_h_fr)
        i_h_rec = 0.5 * (i_h_er + i_h_fr)

        # Optional normalize after fusion.
        u_h_rec = F.normalize(u_h_rec, p=2, dim=-1, eps=1e-12)
        i_h_rec = F.normalize(i_h_rec, p=2, dim=-1, eps=1e-12)

        bpr_loss = self.compute_bpr_loss(
            u_h_rec,
            i_h_rec,
            users,
            pos_items,
            neg_items
        )

        total_loss = bpr_loss + lambda_er * er_loss + mu_fr * fr_loss

        if return_dict:
            return {
                "loss": total_loss,
                "bpr_loss": bpr_loss.detach(),
                "er_loss": er_loss.detach(),
                "fr_loss": fr_loss.detach(),
                "fr_loss_u": fr_loss_u.detach(),
                "fr_loss_i": fr_loss_i.detach(),
            }

        return total_loss

    # ================================================================
    # Inference helper
    # ================================================================

    @torch.no_grad()
    def get_user_item_embeddings(self, edge_index):

        self.eval()

        u_h, i_h = self.encode(
            edge_index,
            self.user_emb.weight,
            self.item_emb.weight
        )

        u_h = F.normalize(u_h, p=2, dim=-1, eps=1e-12)
        i_h = F.normalize(i_h, p=2, dim=-1, eps=1e-12)

        return u_h, i_h