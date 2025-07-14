import torch
import torch.nn as nn
from torch.nn import functional as F


# setting the random seed for reproducibility
torch.manual_seed(1337)
batch_size = 64
block_size = 256  # context length, how many characters to predict
# this is the maximum length of the input sequence
# the model will look at the previous `block_size` characters to predict the next character
max_iters = 5000
learning_rate = 3e-4
if torch.cuda.is_available():
    device = 'cuda'
else:
    print("no gpu, shouldn't run")
    exit() # Not recommended to run without GPU
n_embd = 384  # embedding dimension
n_layers = 6  # number of transformer blocks
num_heads = 6  # number of attention heads
dropout = 0.2  # dropout rate

# -------------------------------------

# reading the text file
with open('input.txt','r', encoding='utf-8') as f:
    text = f.read()
print(len(text))

# making a vocabulary with each possible character
chars = sorted(list(set(text)))
vocab_size = len(chars)

# developing a strategy to tokenize the text from the vocabulary
# create a mapping from charecters to intergers
stoi = {ch : i for i,ch in enumerate(chars)}
itos = {i : ch for i,ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda s: ''.join(itos[c] for c in s)

# encoding the whole datset to store it in a torch.tensor
data = torch.tensor(encode(text), dtype =torch.long)
data.shape

# splitting the dataset into train and validatin sets
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]



# creating blocks of data from the datset for computational efficiency
train_data[:block_size+1]
x = train_data[:block_size]
y = train_data[1: block_size+1]
for i in range(block_size):
    context = x[:i+1]
    target = y[i]
    # print(f'for input {context}, the output is: {target}')

def build_dataset(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data)-block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x = x.to(device)
    y = y.to(device)
    return x, y

xb, yb = build_dataset('train')

class Head(nn.Module):
    def __init__(self, head_size, n_embd_total):
        super().__init__()
        self.key = nn.Linear(n_embd_total, head_size, bias=False) # key projection
        self.query = nn.Linear(n_embd_total, head_size, bias=False) # query projection
        self.value = nn.Linear(n_embd_total, head_size, bias=False) # value projection
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size))) # lower triangular mask
        self.dropout = nn.Dropout(dropout) # dropout layer to prevent overfitting

    def forward(self, x):
        B, T, C = x.shape # B is the batch size, T is the block size, C is the embedding dimension
        k = self.key(x) # k.shape = (b, t, head_size)
        q = self.query(x) # q.shape = (b, t, head_size)
        v = self.value(x) # v.shape = (b, t, head_size)

        head_size = k.size(-1)
        wei = q @ k.transpose(-2,-1) * head_size**-0.5 # wei.shape = (b, t, t)
        wei = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf')) # apply the mask
        wei = F.softmax(wei, dim=-1) # normalize the weights
        wei = self.dropout(wei) # apply dropout to the weights

        out = wei @ v # out.shape = (b, t, head_size)
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, n_embd): # number of attention heads and embedding dimension
        super().__init__()
        head_size = n_embd // num_heads
        self.heads = nn.ModuleList([Head(head_size, n_embd) for _ in range(num_heads)]) # create multiple attention heads
        self.proj = nn.Linear(num_heads * head_size, n_embd) # linear layer to project the concatenated output back to the embedding dimension
        self.dropout = nn.Dropout(dropout) # dropout layer to prevent overfitting

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1) # concatenate the outputs of all heads
        out = self.proj(out) # project the concatenated output back to the embedding dimension
        return out # return the output of the multi-head attention
    
class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), # linear layer to project the input to a higher dimension
            nn.ReLU(), # activation function
            nn.Linear(4 * n_embd, n_embd), # linear layer to project the input back to the embedding dimension
            nn.Dropout(dropout) # dropout layer to prevent overfitting
        )

    def forward(self, x):
        return self.net(x) # apply the feedforward network
class Block(nn.Module):
    def __init__(self, n_embd, n_head):

        super().__init__()
        head_size = n_embd // n_head # size of each attention head
        self.sa = MultiHeadAttention(n_head, n_embd) # self-attention head to capture dependencies in the input sequence
        self.ffwd = FeedForward(n_embd) # feedforward network to process the input
        self.ln1 = nn.LayerNorm(n_embd) # layer normalization to stabilize the training process
        self.ln2 = nn.LayerNorm(n_embd) # another layer normalization for the feedforward network
    def forward(self, x):
        x = x + self.sa(self.ln1(x)) # apply self-attention
        x = x + self.ffwd(self.ln2(x)) # apply feedforward network
        return x # return the output
    

# #### Bigram Model
class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd) # the token embedding table maps each character to an embedding vector
        self.position_embedding_table = nn.Embedding(block_size, n_embd) # the position embedding table maps each position in the block to an embedding vector
        
        # Uncomment the following lines if you want to use self-attention and feedforward network separately
        # self.lm_head = nn.Linear(n_embd, vocab_size) # the linear layer maps the embedding vector to a vocabulary size vector for prediction
        # self.ffwd = FeedForward(n_embd) # feedforward network to process the input
        # self.sa_head = MultiHeadAttention(4, n_embd//4) # self-attention head to capture dependencies in the input sequence
        
        # Uncomment the following lines if you want to use multiple blocks of self-attention and feedforward networks
        # self.blocks = nn.Sequential(
        #     Block(n_embd, n_head=4), # multiple blocks of self-attention and feedforward networks
        #     Block(n_embd, n_head=4),
        #     Block(n_embd, n_head=4),
        #     nn.LayerNorm(n_embd) # final layer normalization to stabilize the output
        # )
        
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=num_heads) for _ in range(n_layers)]) # multiple blocks of self-attention and feedforward networks
        self.ln_f = nn.LayerNorm(n_embd) # final layer normalization to stabilize the output
        self.lm_head = nn.Linear(n_embd, vocab_size) # the linear layer maps the embedding vector to a vocabulary size vector for prediction
        
    def forward(self, idx, targets):
        B, T = idx.shape # B is the batch size, T is the block size
        tok_emb = self.token_embedding_table(idx) # tok_emb.shape = (b, t, c) where b=batch size, t=block size, c=embedding dimension
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # pos_emb.shape = (t, c) where t=block size, c=embedding dimension
        x = tok_emb + pos_emb # x.shape = (b, t, c) where b=batch size, t=block size, c=embedding dimension
        x = self.blocks(x) # x.shape = (b, t, c) after passing through multiple blocks of self-attention and feedforward networks
        x = self.ln_f(x) # x.shape = (b, t, c) after final layer normalization
        # Uncomment the following lines if you want to use self-attention and feedforward network separately
        # x = self.sa_head(x) # x.shape = (b, t, c) after self-attention
        # x = self.ffwd(x) # x.shape = (b, t, c) after feedforward network
        logits = self.lm_head(x) # logits.shape = (b, t, c) where b=batch size, t=block size, c=vocab size
        
        if targets is None:
            loss = None
        else:
            b, t, c = logits.shape
            loss = F.cross_entropy(logits.view(b*t, c), targets.view(-1))
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            # crop idx to the last block size tokens
            idx_cond = idx[:, -block_size:] if idx.shape[1] >= block_size else idx
            # get the predictions
            logits, loss = self(idx_cond, targets=None)
            logits = logits[:,-1,:]
            probs = F.softmax(logits, dim=1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
    

m = BigramLanguageModel().to(device)
out, loss = m(xb, yb)
print(out.shape)
print(loss)

# create an optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

# update the parameters
for step in range(max_iters):
    xb, yb = build_dataset('train')
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    # print every 500 steps
    if step % 500 == 0 or step == max_iters - 1:
        print(f"Step {step:5d} | train loss: {loss.item():.4f}")

# validation loss
xb_val, yb_val = build_dataset('val')
with torch.no_grad():
    logits, val_loss = m(xb_val, yb_val)
print(val_loss.item()) 

# generating text
context = torch.zeros((1, 1), dtype=torch.long, device=device)
# Generate tokens
generated = m.generate(context, max_new_tokens=300)
# Convert to list of token IDs
token_ids = generated[0].tolist()  # take batch item 0
# Decode to text
text = decode(token_ids)
# Print
print(text)



