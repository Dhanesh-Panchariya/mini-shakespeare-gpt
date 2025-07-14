# mini-shakespeare-gpt

A tiny transformer-based character-level language model trained on Shakespeare's works using PyTorch.

This project implements a minimal version of a GPT-like architecture, combining:
- Token + position embeddings
- Multi-head self-attention
- Feedforward layers
- Layer normalization
- Autoregressive text generation

Inspired by Andrej Karpathy's `nanoGPT`, with a focus on simplicity and educational value.

## 🚀 Features

- Character-level tokenization (no external tokenizer)
- Configurable transformer architecture (layers, heads, embedding size)
- Trains on plain text (`input.txt`)
- Efficient batching with context windows
- CUDA support (if available)
- Text generation using sampling

## 🧠 Model Architecture

- Transformer with:
  - `n_layers = 6`
  - `n_heads = 6`
  - `n_embd = 384`
  - `dropout = 0.2`
- Trained on sequences of length `block_size = 256`
- ~1.1 million parameters

## 📝 Usage

### Training

```bash
python train.py
