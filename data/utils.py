import os
import torch

def load_id_mapping(file_path):
    """
    Reads a mapping file where each line is: original_id mapped_id
    Returns a dict: {int(mapped_id): str(original_id)} and inverse dict {str: int}
    """
    orig2mapped = {}
    mapped2orig = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            orig, mapped = line.strip().split()
            mid = int(mapped)
            orig2mapped[orig] = mid
            mapped2orig[mid] = orig
    return orig2mapped, mapped2orig


def load_interactions(file_path):
    """
    Reads interaction file where each line: user_id item_id1 item_id2 ...
    Returns a dict: {int(user_id): [int(item_id), ...]}
    """
    user2items = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            tokens = line.strip().split()
            if not tokens:
                continue
            u = int(tokens[0])
            items = list(map(int, tokens[1:]))
            user2items[u] = items
    return user2items


def build_edge_index_and_gt(data_dir):
    """
    Loads train/val/test sets and returns:
      edge_index (LongTensor [2, num_edges]) for training graph,
      train_gt, val_gt, test_gt: dicts {user: [items]}
    """
    # Load mappings (not used directly here but available)
    u2orig, _ = load_id_mapping(os.path.join(data_dir, 'user_list.txt'))
    i2orig, _ = load_id_mapping(os.path.join(data_dir, 'item_list.txt'))

    # Load interactions
    train = load_interactions(os.path.join(data_dir, 'train.txt'))
    val = load_interactions(os.path.join(data_dir, 'validation.txt'))
    test = load_interactions(os.path.join(data_dir, 'test.txt'))

    # Build edge_index for train graph
    edges = []
    for u, items in train.items():
        for i in items:
            edges.append((u, i))
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    return edge_index, train, val, test


def build_edge_index_and_gt_test(data_dir):
    """
    Loads train/val/test sets and returns:
      edge_index (LongTensor [2, num_edges]) for training graph,
      train_gt, val_gt, test_gt: dicts {user: [items]}
    """
    # Load mappings (not used directly here but available)
    u2orig, _ = load_id_mapping(os.path.join(data_dir, 'user_list.txt'))
    i2orig, _ = load_id_mapping(os.path.join(data_dir, 'item_list.txt'))

    # Load interactions
    train_val = load_interactions(os.path.join(data_dir, 'train_val.txt'))
    test = load_interactions(os.path.join(data_dir, 'test.txt'))

    # Build edge_index for train graph
    edges = []
    for u, items in train_val.items():
        for i in items:
            edges.append((u, i))
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    return edge_index, train_val, test
