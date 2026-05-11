import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def grouped_softmax(scores, group_index, num_groups, eps=1e-12):

    if scores.numel() == 0:
        return scores

    # max per group for numerical stability
    max_per_group = torch.full(
        (num_groups,),
        torch.finfo(scores.dtype).min,
        dtype=scores.dtype,
        device=scores.device
    )

    # Requires PyTorch version with scatter_reduce_.
    max_per_group.scatter_reduce_(
        dim=0,
        index=group_index,
        src=scores,
        reduce="amax",
        include_self=True
    )

    exp_scores = torch.exp(scores - max_per_group[group_index])

    denom = torch.zeros(
        num_groups,
        dtype=scores.dtype,
        device=scores.device
    )
    denom.index_add_(0, group_index, exp_scores)

    return exp_scores / (denom[group_index] + eps)


class NormAttentionGATLayer(nn.Module):

    def __init__(
        self,
        in_dim_u,
        in_dim_i,
        out_dim,
        dropout=0.0,
        negative_slope=0.2
    ):
        super().__init__()

        self.out_dim = out_dim
        self.negative_slope = negative_slope
        self.dropout = nn.Dropout(dropout)

        self.proj_u = nn.Linear(in_dim_u, out_dim, bias=False)
        self.proj_i = nn.Linear(in_dim_i, out_dim, bias=False)

        # user-side attention: target=user, source=item
        # e_ui = a_u^T [W_u h_u || W_i h_i]
        self.attn_u = nn.Parameter(torch.empty(2 * out_dim))

        # item-side attention: target=item, source=user
        # e_iu = a_i^T [W_i h_i || W_u h_u]
        self.attn_i = nn.Parameter(torch.empty(2 * out_dim))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.proj_u.weight)
        nn.init.xavier_uniform_(self.proj_i.weight)

        bound = math.sqrt(6.0 / (2 * self.out_dim + 1))
        nn.init.uniform_(self.attn_u, -bound, bound)
        nn.init.uniform_(self.attn_i, -bound, bound)

    def forward(self, user_feats, item_feats, edge_index):
        """
        user_feats: [num_users, in_dim_u]
        item_feats: [num_items, in_dim_i]
        edge_index: [2, E], user-item edges

        Return:
            user_out: [num_users, out_dim]
            item_out: [num_items, out_dim]
        """
        src = edge_index[0].long()  # users
        dst = edge_index[1].long()  # items

        num_users = user_feats.size(0)
        num_items = item_feats.size(0)

        # Project all users/items once.
        u_proj = self.proj_u(user_feats)  # [num_users, out_dim]
        i_proj = self.proj_i(item_feats)  # [num_items, out_dim]

        if src.numel() == 0:
            user_out = u_proj.new_zeros(num_users, self.out_dim)
            item_out = i_proj.new_zeros(num_items, self.out_dim)
            return F.elu(user_out), F.elu(item_out)

        u_e = u_proj[src]  # [E, out_dim]
        i_e = i_proj[dst]  # [E, out_dim]

        # ============================================================
        # 1) User update: users aggregate messages from item neighbors
        # ============================================================
        e_u = torch.cat([u_e, i_e], dim=-1)  # [E, 2*out_dim]
        e_u = F.leaky_relu(
            (e_u * self.attn_u).sum(dim=-1),
            negative_slope=self.negative_slope
        )

        # Correct: softmax over each user's item-neighborhood.
        alpha_u = grouped_softmax(e_u, src, num_users)
        alpha_u = self.dropout(alpha_u)

        user_out = u_proj.new_zeros(num_users, self.out_dim)
        user_out.index_add_(0, src, alpha_u.unsqueeze(-1) * i_e)

        # ============================================================
        # 2) Item update: items aggregate messages from user neighbors
        # ============================================================
        e_i = torch.cat([i_e, u_e], dim=-1)  # [E, 2*out_dim]
        e_i = F.leaky_relu(
            (e_i * self.attn_i).sum(dim=-1),
            negative_slope=self.negative_slope
        )

        # Correct: softmax over each item's user-neighborhood.
        alpha_i = grouped_softmax(e_i, dst, num_items)
        alpha_i = self.dropout(alpha_i)

        item_out = i_proj.new_zeros(num_items, self.out_dim)
        item_out.index_add_(0, dst, alpha_i.unsqueeze(-1) * u_e)

        return F.elu(user_out), F.elu(item_out)


class NormAttentionGATEncoder(nn.Module):

    def __init__(
        self,
        in_dim_u,
        in_dim_i,
        hidden_dim,
        num_layers,
        dropout=0.0
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Project h^(0) into hidden_dim so h^(0), h^(1), ..., h^(K)
        # can be stacked and combined by inter-layer attention.
        self.init_proj_u = nn.Linear(in_dim_u, hidden_dim, bias=False)
        self.init_proj_i = nn.Linear(in_dim_i, hidden_dim, bias=False)

        self.layers = nn.ModuleList([
            NormAttentionGATLayer(
                hidden_dim,
                hidden_dim,
                hidden_dim,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])

        # Inter-layer attention:
        # beta_k = q^T W_k h^(k) / sqrt(d)
        self.layer_proj_u = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim, bias=False)
            for _ in range(num_layers + 1)
        ])
        self.layer_proj_i = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim, bias=False)
            for _ in range(num_layers + 1)
        ])

        self.query_u = nn.Parameter(torch.empty(hidden_dim))
        self.query_i = nn.Parameter(torch.empty(hidden_dim))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.init_proj_u.weight)
        nn.init.xavier_uniform_(self.init_proj_i.weight)

        for layer in self.layers:
            layer.reset_parameters()

        for proj in self.layer_proj_u:
            nn.init.xavier_uniform_(proj.weight)

        for proj in self.layer_proj_i:
            nn.init.xavier_uniform_(proj.weight)

        bound = math.sqrt(6.0 / (self.hidden_dim + 1))
        nn.init.uniform_(self.query_u, -bound, bound)
        nn.init.uniform_(self.query_i, -bound, bound)

    def _inter_layer_attention(self, h_list, proj_list, query):
        """
        h_list: list of [N, hidden_dim], including h^(0)
        proj_list: W_k for each layer k
        query: q vector

        Return:
            h_final: [N, hidden_dim]
        """
        scores = []

        scale = math.sqrt(self.hidden_dim)

        for h_k, W_k in zip(h_list, proj_list):
            # [N, hidden_dim]
            z_k = W_k(h_k)

            # [N]
            beta_k = (z_k * query).sum(dim=-1) / scale
            scores.append(beta_k)

        # [K+1, N]
        scores = torch.stack(scores, dim=0)

        # softmax over layers for each node
        weights = F.softmax(scores, dim=0).unsqueeze(-1)  # [K+1, N, 1]

        h_stack = torch.stack(h_list, dim=0)  # [K+1, N, hidden_dim]
        h_final = (weights * h_stack).sum(dim=0)

        return F.normalize(h_final, p=2, dim=-1, eps=1e-12)

    def forward(self, u_feats, i_feats, edge_index):
        """
        u_feats: [num_users, in_dim_u]
        i_feats: [num_items, in_dim_i]
        edge_index: [2, E], edge_index[0] users, edge_index[1] items

        Return:
            u_final: [num_users, hidden_dim]
            i_final: [num_items, hidden_dim]
        """
        # h^(0), projected to hidden_dim.
        u_h = self.init_proj_u(u_feats)
        i_h = self.init_proj_i(i_feats)

        # Initial embeddings participate in layer combination.
        u_layers = [u_h]
        i_layers = [i_h]

        for layer in self.layers:
            u_h, i_h = layer(u_h, i_h, edge_index)
            u_layers.append(u_h)
            i_layers.append(i_h)

        u_final = self._inter_layer_attention(
            u_layers,
            self.layer_proj_u,
            self.query_u
        )

        i_final = self._inter_layer_attention(
            i_layers,
            self.layer_proj_i,
            self.query_i
        )

        return u_final, i_final