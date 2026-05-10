# MGLRec: A Generative Self-Supervised Graph Recommendation Model

This repository contains the official implementation of **MGLRec**, a generative self-supervised learning framework for graph-based recommendation. MGLRec addresses the limitations of purely supervised and contrastive learning approaches by introducing two auxiliary tasks: edge structure reconstruction and feature recovery, which are jointly optimized with the recommendation objective.

## 🔍 Motivation
Existing GNN-based recommendation models either rely solely on supervision signals (which suffer from label sparsity and noise), or adopt contrastive learning (which heavily depends on augmentation quality and negative sampling). GRRec alleviates these issues by incorporating **generative self-supervision**, enhancing representation robustness without handcrafted contrastive views.

## 🚀 Highlights
- **Multi-task Learning Framework**: Combines supervised BPR loss with two generative auxiliary tasks.
- **Edge Reconstruction**: Learns graph structure via masked link prediction.
- **Feature Recovery**: Learns node semantics via masked embedding restoration.
- **Norm-Attention GAT Encoder**: Applies inter-layer attention and L2 normalization for stable representation.

## ⚙️ Running GRRec
### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Set data path
In `configs/default.py`, set:
```python
DATA_DIR = '/path/to/your/dataset'
```

### 3. Train the model
```bash
python scripts/run_train.py
```

### 4. Evaluate the model
```bash
python scripts/run_eval.py
```

## 📈 Performance

## 📄 Citation

## 🧑‍💻 Contact
For questions or feedback, feel free to open an issue or contact the author.
