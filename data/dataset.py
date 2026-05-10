import numpy as np
from torch.utils.data import Dataset


class InteractionDataset(Dataset):
    def __init__(self, edge_index, num_users, num_items, user2items):
        self.edges = edge_index.t().tolist()
        self.num_users = num_users
        self.num_items = num_items
        self.user2items = user2items

    def __len__(self):
        return len(self.edges)

    def __getitem__(self, idx):
        u, i = self.edges[idx]
        # negative sampling
        neg = np.random.randint(self.num_items)
        while neg in self.user2items[u]:
            neg = np.random.randint(self.num_items)
        return u, i, neg, (u, i), (u, neg)