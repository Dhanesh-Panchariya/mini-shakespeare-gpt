# Mini Shakespeare GPT

A minimal character-level GPT model trained on Shakespeare's text using PyTorch and transformer blocks.

---

## 📚 Description

This project builds a simple GPT-style transformer from scratch, trained on a corpus of Shakespeare text. It learns to generate character-by-character text in Shakespearean style.

Inspired by [Andrej Karpathy’s nanoGPT](https://github.com/karpathy/nanoGPT).

---

## 🧠 Model Architecture

- 6 Transformer blocks  
- 6 attention heads  
- 384-dimensional embeddings  
- Sequence/block size: 256 tokens  
- ~1.1 million parameters  

---

## 📁 Files

- `mini_shakespeare.py` — Full PyTorch training script  
- `Mini_Shakespheare.ipynb` — Notebook version for experimentation  
- `input.txt` — Raw training data (Shakespeare text)  

---

## 🚀 How to Train

```bash
python mini_shakespeare.py
```
Training is GPU-accelerated (CUDA required). Do not run this on CPU.

---

## ✍️ Text Generation

Once training completes, the model can generate text:

'''python
context = torch.zeros((1, 1), dtype=torch.long, device='cuda')
print(decode(model.generate(context, max_new_tokens=300)[0].tolist()))
'''

---

##🧪 Requirements

Python 3.8+
PyTorch with CUDA support
Basic GPU (4GB+ VRAM recommended)

📖 License
MIT License
