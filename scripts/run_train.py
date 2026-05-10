import torch
from configs.default import (
    DATA_DIR, NUM_USERS, NUM_ITEMS,
    DIM_U, DIM_I, HIDDEN_DIM, NUM_LAYERS,
    P_E, P_A_U, P_A_I, LAMBDA_ER, MU_FR,
    LR, BATCH_SIZE, EPOCHS, EVAL_FREQ,
    DEVICE, CHECKPOINT_PATH
)
from data.utils import build_edge_index_and_gt
from data.dataset import InteractionDataset
from torch.utils.data import DataLoader
from models.grrec import GRRecModel
from trainers.train import train
from trainers.evaluate import evaluate

def main():
    # Prepare data
    edge_index, train_gt, val_gt, test_gt = build_edge_index_and_gt(DATA_DIR)
    train_dataset = InteractionDataset(edge_index, NUM_USERS, NUM_ITEMS, train_gt)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    device = torch.device(DEVICE)
    edge_index = edge_index.to(device)

    # Build model
    model = GRRecModel(
        NUM_USERS, NUM_ITEMS,
        DIM_U, DIM_I,
        HIDDEN_DIM, NUM_LAYERS
    ).to(device)

    best_ndcg = 0.0
    for epoch in range(1, EPOCHS + 1):
        train(
            model, train_loader, edge_index, device,
            P_E, P_A_U, P_A_I,
            LAMBDA_ER, MU_FR, LR
        )
        if epoch % EVAL_FREQ == 0 or epoch == EPOCHS:
            r_val, n_val = evaluate(
                model, edge_index, train_gt, val_gt, device, k=20
            )
            print(f"Epoch {epoch}: Val Recall@20={r_val:.4f}, NDCG@20={n_val:.4f}")
            if n_val > best_ndcg:
                best_ndcg = n_val
                torch.save(model.state_dict(), CHECKPOINT_PATH)
                print(f"  Best model saved (epoch {epoch}, NDCG={n_val:.4f})")

if __name__ == '__main__':
    main()