# 🧠 Mini Shakespeare GPT

A high-performance, *from-scratch* Transformer implemented in **PyTorch**, trained on over **1 million characters** of Shakespearean text. Guided by Andrej Karpathy’s practical walkthrough of *“Attention Is All You Need”*, this model demonstrates advanced neural-language-model design, end‑to‑end training, and expressive generation capabilities.

---

## 🚀 Key Highlights

| Metric                         | Value                                    |
|-------------------------------|-------------------------------------------|
| Corpus Size                   | ~1,000,000 characters (Shakespeare text) |
| Model Architecture            | 6 Transformer layers × 6 attention heads |
| Embedding Dimension           | 384 (total parameters ≈ 1.1 million)     |
| Maximum Context / Block Size  | 256 previous characters                   |
| Dropout Rate                  | 0.2                                       |
| Optimizer & Learning Rate     | AdamW @ 3e‑4                              |
| Training Iterations           | 2,500 mini-batch steps                    |
| Training Time (GPU)           | ~3 minutes per epoch (medium GPU)         |

---

## 🔍 Features & Academic Credibility

- Deep dive implementation based on the original **Vaswani et al.* “Attention Is All You Need”** attention mechanisms.
- Heavily influenced and guided by **Andrej Karpathy’s nanoGPT implementation**, but built manually in PyTorch for clarity and learning.
- Implements **positional embeddings**, **masked multi-head self-attention**, **feed-forward networks**, and **LayerNorm + residuals**.
- Achieves **autoregressive character-level generation**—generates Shakespeare-style text one character at a time with causal masking.

---

## 🧑‍💻 CLI Interface & Usage Guide

All functionality is accessed via command-line flags:

```bash
# Training mode (GPU recommended, but CPU still works)
python mini_shakespeare.py --train

# Generation mode with CPU or GPU using pretrained weights
python mini_shakespeare.py --generate
```

## 📂 Repo Structure
```graphql
.
├── mini_shakespeare.py       # Full Transformer + CLI implementation
├── input.txt                 # Raw Shakespeare corpus (~1 MB text)
├── shakespeare_model.pth     # Pretrained weights (~50 MB)
├── README.md                 # This file
└── Mini_Shakespheare.ipynb   # Notebook version (optional)
```

## 📖 References
- Vaswani, A. et al. “Attention Is All You Need” (2017)
- Karpathy, A. “nanoGPT” tutorial implementation
