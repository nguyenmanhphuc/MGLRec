import torch
from configs.default import (
    DATA_DIR, NUM_USERS, NUM_ITEMS,
    DIM_U, DIM_I, HIDDEN_DIM, NUM_LAYERS,
    DEVICE, CHECKPOINT_PATH
)
from data.utils import build_edge_index_and_gt
from models.grrec import GRRecModel
from trainers.evaluate import evaluate

def main():
    device = torch.device(DEVICE)
    # Load data and model
    edge_index, train_gt, val_gt, test_gt = build_edge_index_and_gt(DATA_DIR)
    edge_index = edge_index.to(device)

    model = GRRecModel(
        NUM_USERS, NUM_ITEMS,
        DIM_U, DIM_I,
        HIDDEN_DIM, NUM_LAYERS
    ).to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))

    # Evaluation
    r_val, n_val = evaluate(model, edge_index, train_gt, val_gt, device, k=20)
    r_test, n_test = evaluate(model, edge_index, train_gt, test_gt, device, k=20)
    print(f"Validation Recall@20={r_val:.4f}, NDCG@20={n_val:.4f}")
    print(f"Test       Recall@20={r_test:.4f}, NDCG@20={n_test:.4f}")

if __name__ == '__main__':
    main()