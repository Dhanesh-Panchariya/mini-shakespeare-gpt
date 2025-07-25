import torch
import torch.nn as nn
from torch.nn import functional as F
import argparse
import os

# ------------ Hyperparameters ------------
torch.manual_seed(1337)  # for reproducibility
batch_size = 64
block_size = 256  # context size (sequence length)
max_iters = 2500  # number of training iterations
learning_rate = 3e-4
if torch.cuda.is_available():
    device = 'cuda'
else:
    print("no gpu, shouldn't run")
    exit() # Not recommended to run without GPU
n_embd = 384  # embedding dimension
n_layers = 6  # number of transformer blocks
num_heads = 6  # number of attention heads
dropout = 0.2  # dropout rate to avoid overfitting

# Select device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ------------ Data Preparation ------------
# Load dataset
with open('input.txt','r', encoding='utf-8') as f:
    text = f.read()

# Create vocabulary
chars = sorted(list(set(text)))
vocab_size = len(chars)

# Mappings from char to int and vice versa
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]  # string to list of ints
decode = lambda l: ''.join([itos[i] for i in l])  # list of ints to string

# Encode entire dataset
data = torch.tensor(encode(text), dtype=torch.long)

# Split into train and validation
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# Create mini-batches
def build_dataset(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

# ------------ Model Definition ------------
class Head(nn.Module):
    # Single attention head
    def __init__(self, head_size, n_embd_total):
        super().__init__()
        self.key = nn.Linear(n_embd_total, head_size, bias=False)
        self.query = nn.Linear(n_embd_total, head_size, bias=False)
        self.value = nn.Linear(n_embd_total, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))  # causal mask
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        # Compute attention weights
        wei = q @ k.transpose(-2, -1) * (k.size(-1) ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        return wei @ v  # weighted sum of values

class MultiHeadAttention(nn.Module):
    # Multiple attention heads
    def __init__(self, num_heads, n_embd):
        super().__init__()
        head_size = n_embd // num_heads
        self.heads = nn.ModuleList([Head(head_size, n_embd) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)  # projection layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))

class FeedForward(nn.Module):
    # Simple 2-layer feedforward network
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    # Transformer block: attention + feedforward
    def __init__(self, n_embd, n_head):

        super().__init__()
        self.sa = MultiHeadAttention(n_head, n_embd)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))  # residual connection
        return x + self.ffwd(self.ln2(x))  # residual connection

class BigramLanguageModel(nn.Module):
    # Full language model
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=num_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(n_embd)  # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)  # projection to vocab size

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)  # token embeddings
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # positional embeddings
        x = tok_emb + pos_emb  # combine
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        # Loss computation (if targets provided)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1)) if targets is not None else None
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]  # crop context
            logits, _ = self(idx_cond)
            probs = F.softmax(logits[:, -1, :], dim=-1)  # next token probabilities
            idx_next = torch.multinomial(probs, num_samples=1)  # sample
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# ------------ Main Script ------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--generate', action='store_true', help='Generate text using the trained model')
    args = parser.parse_args()

    model = BigramLanguageModel().to(device)

    if args.train:
        # Training loop
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        for step in range(max_iters):
            xb, yb = build_dataset('train')
            logits, loss = model(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            # if step % 500 == 0:
            print(f"Step {step} | Loss: {loss.item():.4f}")

        # Save model
        torch.save(model.state_dict(), '')

        # Evaluate on validation set
        xb_val, yb_val = build_dataset('val')
        with torch.no_grad():
            _, val_loss = model(xb_val, yb_val)
        print(f"Validation Loss: {val_loss.item():.4f}")

    elif args.generate:
        # Load pretrained model
        if not os.path.exists('shakespeare_model.pth'):
            raise FileNotFoundError("Trained model not found. Please run with --train first or download the model.")

        model.load_state_dict(torch.load('shakespeare_model.pth', map_location=device, weights_only=True))
        model.eval()
        context = torch.zeros((1, 1), dtype=torch.long, device=device)  # initial token
        generated = model.generate(context, max_new_tokens=300)
        print(decode(generated[0].tolist()))
    else:
        print("Please use --train or --generate flag.")
