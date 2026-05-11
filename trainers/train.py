import torch

def train(model, train_loader, edge_index, device,
          p_e, p_a_u, p_a_i, lambda_er, mu_fr, lr):
    model.train()
    bat = 1
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for users, pos, neg, pos_edge, neg_edge in train_loader:
        print(f"Running batch: {bat}/{len(train_loader)}")
        bat = bat + 1

        users = torch.tensor(users, device=device)
        pos = torch.tensor(pos, device=device)
        neg = torch.tensor(neg, device=device)
        # pos_edge = torch.stack(pos_edge).to(device).t()
        # neg_edge = torch.stack(neg_edge).to(device).t()
        # pos_edge = torch.tensor(pos_edge, dtype=torch.long, device=device).t()
        # neg_edge = torch.tensor(neg_edge, dtype=torch.long, device=device).t()
        loss = model(edge_index, p_e, p_a_u, p_a_i,
                     pos_edge, neg_edge,
                     users, pos, neg,
                     lambda_er, mu_fr)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()