# 🧠 Mini Shakespeare GPT
(Guided Implementation of the research paper: __Attention Is All You Need__
A compact, transformer-based character-level language model trained from scratch on Shakespeare's text — implemented in pure PyTorch. Inspired by [Andrej Karpathy's](https://github.com/karpathy) `nanoGPT`, this model demonstrates the core ideas of attention, token embeddings, and autoregressive generation — all in under 300 lines.

---

## 📜 What It Does

This model learns to speak like Shakespeare! It’s trained on raw Shakespeare text using a GPT-style architecture with:
- Transformer blocks
- Multi-head self-attention
- Positional encoding
- Layer normalization
- Causal masking

---

## 🚀 Quick Features

- Full transformer architecture with `n_layers=6` and `n_heads=6`
- Character-level modeling (no tokenizer needed)
- Trainable on consumer GPUs
- Fully written in PyTorch
- Generates text *one character at a time*

---

## 🧰 Requirements

- Python 3.8+
- PyTorch (with CUDA if possible)
- `input.txt` (a text corpus — Shakespeare works well)

Install PyTorch from [https://pytorch.org/](https://pytorch.org/).

---

##🧠 Architecture Overview

| Component             | Description                         |
| --------------------- | ----------------------------------- |
| `Embedding`           | Maps characters to `n_embd=384` dim |
| `MultiHeadAttention`  | 6 heads, causal masked              |
| `FeedForward`         | 2-layer MLP with ReLU + Dropout     |
| `Block`               | (LN → Attention → LN → FFN)         |
| `BigramLanguageModel` | Stacks transformer blocks           |

---

## ⚙️ Usage

### 1. Prepare Input
Download or paste your corpus into a file named:

input.txt

### 2. Run Training
```bash
python mini_shakespeare.py
```

The script will:
Read and encode the corpus
Train for 5000 iterations on mini-batches
Print training and validation loss


###✍️ Text Generation
Once training finishes, the model can generate Shakespearean text like this:

```python
context = torch.zeros((1, 1), dtype=torch.long, device='cuda')
generated = model.generate(context, max_new_tokens=300)
print(decode(generated[0].tolist()))
```
Example output:

```css
But wherefore scorns he heard me of such grace?
And stand I thence for whom I shame to be,
To give him every stone and gentle say...
```

---


📂 Files
__mini_shakespeare.py__ – the entire model + training + inference script
__input.txt__ – Shakespeare text or your own corpus
__Mini_Shakespheare.ipynb__ – optional notebook version for exploration

💡 Inspiration
Inspired by Andrej Karpathy’s "let’s build GPT from scratch" initiative.

📖 License
MIT License.
Feel free to fork, experiment, and build upon it!
