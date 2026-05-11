
import os
import torch
from data.utils import load_id_mapping

# --- Paths ---
# Root directory for datasets
DATA_DIR = os.getenv('DATA_DIR', 'data/Data/yelp2018')
USER_LIST_PATH = os.path.join(DATA_DIR, 'user_list.txt')
ITEM_LIST_PATH = os.path.join(DATA_DIR, 'item_list.txt')
TRAIN_FILE = os.path.join(DATA_DIR, 'train.txt')
VALID_FILE = os.path.join(DATA_DIR, 'validation.txt')
TEST_FILE = os.path.join(DATA_DIR, 'test.txt')

# Model checkpoint path
CHECKPOINT_DIR = os.getenv('CHECKPOINT_DIR', './checkpoints')
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, 'grrec_best.pt')

# --- Dataset Info ---
# Dynamically infer number of users and items from mapping files
_orig2u, _ = load_id_mapping(USER_LIST_PATH)
_orig2i, _ = load_id_mapping(ITEM_LIST_PATH)
NUM_USERS = len(_orig2u)
NUM_ITEMS = len(_orig2i)

# --- Model Hyperparameters ---
# Input embedding dimensions for users and items
DIM_U = 128
DIM_I = 128
# Hidden dimension for GAT encoder
HIDDEN_DIM = 128
# Number of GAT layers
NUM_LAYERS = 2

# --- Mask Rates ---
P_E = 0.3      # Edge mask rate
P_A_U = 0.3    # User feature mask rate
P_A_I = 0.3    # Item feature mask rate

# --- Loss Weights ---
LAMBDA_ER = 1.0    # Weight for edge reconstruction loss
MU_FR = 1.0        # Weight for feature recovery loss

# --- Training Config ---
LR = 1e-3          # Learning rate
BATCH_SIZE = 4096  # Training batch size
EPOCHS = 50        # Number of training epochs
EVAL_FREQ = 5      # Evaluate every EVAL_FREQ epochs

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEVICE = os.getenv('DEVICE', str(device))
