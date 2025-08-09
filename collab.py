import os
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import LambdaLR
import math
import time

device = 'cuda' if torch.cuda.is_available() else 'cpu'

data_path = '/kaggle/recipes.txt'
max_lines = 100000

with open(data_path, 'r', encoding='utf-8') as f:
    lines = []
    for i, line in enumerate(f):
        if i >= max_lines:
            break
        lines.append(line)
    text = ''.join(lines)

cleaned_text = re.sub(r'[()\[\]{}"]', '', text)
final_text = re.sub(r'\\u00b0', 'F', cleaned_text)

chars = sorted(list(set(final_text)))
vocab_size = len(chars)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(final_text), dtype=torch.long)
n = int(0.8*len(data))
train_data = data[:n]
val_data = data[n:]

class SeqDataset(Dataset):
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size
    def __len__(self):
        return len(self.data) - self.block_size
    def __getitem__(self, idx):
        x = self.data[idx:idx+self.block_size]
        y = self.data[idx+1:idx+self.block_size+1]
        return x, y

block_size = 64
batch_size = 32

train_dataset = SeqDataset(train_data, block_size)
val_dataset = SeqDataset(val_data, block_size)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)


n_embd = 256
n_head = 4
n_layer = 4
dropout = 0.1

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
    def forward(self, seq_len, device):
        t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        freqs = torch.einsum('i , j -> i j', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb

def apply_rotary_pos_emb(q, k, sinusoid):
    sin = sinusoid.sin()[None, :, :]
    cos = sinusoid.cos()[None, :, :]
    q1, q2 = q[..., :q.shape[-1]//2], q[..., q.shape[-1]//2:]
    k1, k2 = k[..., :k.shape[-1]//2], k[..., k.shape[-1]//2:]
    q_rot = torch.cat((q1 * cos - q2 * sin, q1 * sin + q2 * cos), dim=-1)
    k_rot = torch.cat((k1 * cos - k2 * sin, k1 * sin + k2 * cos), dim=-1)
    return q_rot, k_rot

class Head(nn.Module):
    def __init__(self, n_embd, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
    def forward(self, x, sinusoid):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        q, k = apply_rotary_pos_emb(q, k, sinusoid[:T, :])
        wei = q @ k.transpose(-2, -1) * (1.0 / math.sqrt(q.size(-1)))
        mask = self.tril[:T, :T]
        wei = wei.masked_fill(mask == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        assert n_embd % n_head == 0
        self.head_size = n_embd // n_head
        self.heads = nn.ModuleList([Head(n_embd, self.head_size) for _ in range(n_head)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, sinusoid):
        out = torch.cat([h(x, sinusoid) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.sa = MultiHeadAttention(n_embd, n_head)
        self.ff = FeedForward(n_embd)
    def forward(self, x, sinusoid):
        x = x + self.sa(self.ln1(x), sinusoid)
        x = x + self.ff(self.ln2(x))
        return x

class GPTModel(nn.Module):
    def __init__(self, vocab_size, n_embd, n_layer, n_head, block_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.head_size = n_embd // n_head
        self.rotary_emb = RotaryPositionalEmbedding(self.head_size // 2)
        self.blocks = nn.ModuleList([Block(n_embd, n_head) for _ in range(n_layer)])  //v imp.
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.block_size = block_size
        self.tie_weights()
    def tie_weights(self):
        self.lm_head.weight = self.token_embedding_table.weight
    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        sinusoid = self.rotary_emb(self.block_size, tok_emb.device)
        x = tok_emb
        for block in self.blocks:
            x = block(x, sinusoid)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            logits_view = logits.view(B*T, -1)
            targets_view = targets.view(B*T)
            loss = F.cross_entropy(logits_view, targets_view)
        return logits, loss
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=0, top_p=0.0):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-8)
            if top_k > 0:
                values, _ = torch.topk(logits, top_k)
                min_values = values[:, -1].unsqueeze(1)
                logits = torch.where(logits < min_values, torch.full_like(logits, -1e10), logits)
            if top_p > 0.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = False
                for b in range(logits.size(0)):
                    indices_to_remove = sorted_indices[b][sorted_indices_to_remove[b]]
                    logits[b, indices_to_remove] = -1e10
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)
        return idx


class LSTMModel(nn.Module):
    def __init__(self, vocab_size, n_embd, n_layer):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, n_embd)
        self.lstm = nn.LSTM(n_embd, n_embd, num_layers=n_layer, batch_first=True)
        self.fc = nn.Linear(n_embd, vocab_size)
    def forward(self, x, targets=None):
        x = self.embedding(x)
        out, _ = self.lstm(x)
        logits = self.fc(out)
        loss = None
        if targets is not None:
            B, T = targets.shape
            loss = F.cross_entropy(logits.view(B*T, -1), targets.view(B*T))
        return logits, loss
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -block_size:])
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_token], dim=1)
        return idx

def evaluate(model, val_loader, max_batches=100):
    model.eval()
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for i, (xb, yb) in enumerate(val_loader):
            if i >= max_batches:
                break
            xb, yb = xb.to(device), yb.to(device)
            _, loss = model(xb, yb)
            total_loss += loss.item() * xb.size(0)
            count += xb.size(0)
    model.train()
    return total_loss / count

def train(model, max_iters=200, model_name="Model"):
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda step: min(1.0, step / 100))
    train_loss_accum = 0.0
    iteration = 0
    start = time.time()
    while iteration < max_iters:
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            _, loss = model(xb, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss_accum += loss.item()
            iteration += 1
            if iteration % 10 == 0:
                print(f"{model_name} iter {iteration} train_loss {train_loss_accum / 10:.4f}")
                train_loss_accum = 0.0
            if iteration % 50 == 0:
                val_loss = evaluate(model, val_loader)
                print(f"{model_name} iter {iteration} val_loss {val_loss:.4f}")
            if iteration >= max_iters:
                break
    print(f"{model_name} training complete in {time.time() - start:.2f}s\n")


print("Training LSTM")
lstm_model = LSTMModel(vocab_size, n_embd, n_layer).to(device)
train(lstm_model, max_iters=200, model_name="LSTM")

print("Training GPT")
gpt_model = GPTModel(vocab_size, n_embd, n_layer, n_head, block_size).to(device)
train(gpt_model, max_iters=200, model_name="GPT")


context = torch.zeros((1, 1), dtype=torch.long, device=device)

print("\n--- LSTM Sample Output---")
lstm_output = lstm_model.generate(context.clone(), max_new_tokens=200)
print(decode(lstm_output[0].tolist()))

print("\n--- GPT Sample Output---")
gpt_output = gpt_model.generate(context.clone(), max_new_tokens=200, temperature=0.8, top_k=50, top_p=0.9)
print(decode(gpt_output[0].tolist()))
