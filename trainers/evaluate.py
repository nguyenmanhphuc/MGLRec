import torch
import numpy as np

def recall_ndcg_at_k(scores, train_gt, ground_truth, k):
    # scores: [num_users, num_items]
    recall_list, ndcg_list = [], []
    for u, true_items in ground_truth.items():
        if u in train_gt:
            scores[u, list(train_gt[u])] = -1e9
        topk = torch.topk(scores[u], k).indices.tolist()
        hits = set(topk) & set(true_items)
        recall = len(hits) / len(true_items)
        dcg = sum([1 / np.log2(idx+2) for idx, item in enumerate(topk) if item in hits])
        idcg = sum([1 / np.log2(i+2) for i in range(min(len(true_items), k))])
        ndcg = dcg / idcg if idcg > 0 else 0.0
        recall_list.append(recall)
        ndcg_list.append(ndcg)
    return np.mean(recall_list), np.mean(ndcg_list)

def evaluate(model, edge_index, train_gt, eval_gt, device, k=20):
    model.eval()
    with torch.no_grad():
        u_h, i_h = model.get_user_item_embeddings(edge_index)
        scores = u_h @ i_h.t()
        scores = scores.cpu()
    return recall_ndcg_at_k(scores, train_gt, eval_gt, k)
