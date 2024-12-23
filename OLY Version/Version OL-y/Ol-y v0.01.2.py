"""
Copyright 2024 [KISS SEBASTIAN CRISTIAN]
Licensed under the Apache License, Version 2.0
See LICENSE file in the root directory.

Ol-y (Advanced Language Model)
Version: 0.01.2
Developed by [KISS SEBASTIAN CRISTIAN] at age 17

NOTICE: Proprietary AI implementation. See NOTICE.md for full terms.
All rights reserved.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
import math
from tqdm.auto import tqdm
from transformers import GPT2Tokenizer
import os
import numpy as np
import time
import json
from pathlib import Path

# Configurația modelului
@dataclass
class OLyConfig:
    block_size: int = 128
    vocab_size: int = 50257
    n_layer: int = 4
    n_head: int = 8
    n_embd: int = 256
    dropout: float = 0.1
    bias: bool = True
    use_dialog_position: bool = True
    response_token_id: int = 1

# Configurația pentru antrenament
@dataclass
class TrainingConfig:
    batch_size: int = 32
    learning_rate: float = 3e-4
    max_epochs: int = 50
    warmup_tokens: int = 375e6
    final_tokens: int = 260e9
    num_workers: int = 4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class FastAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        
        self.register_buffer("dialog_mask", torch.tril(torch.ones(config.block_size, config.block_size)))

    def forward(self, x):
        B, T, C = x.size()
        
        qkv = self.c_attn(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        mask = self.dialog_mask[:T, :T]
        att = att.masked_fill(mask[:T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = F.dropout(att, p=self.dropout, training=self.training)
        
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        
        return y

class FastBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = FastAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, config.n_embd * 2),
            nn.GELU(),
            nn.Linear(config.n_embd * 2, config.n_embd),
            nn.Dropout(config.dropout)
        )

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class DialogPositionalEncoding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.question_embedding = nn.Parameter(torch.randn(1, 1, config.n_embd))
        self.answer_embedding = nn.Parameter(torch.randn(1, 1, config.n_embd))
        self.position_embedding = nn.Parameter(torch.randn(1, config.block_size, config.n_embd))

    def forward(self, x, is_response):
        B, T, C = x.shape
        pos_emb = self.position_embedding[:, :T, :]
        dialog_emb = torch.where(is_response.unsqueeze(-1), 
                               self.answer_embedding, 
                               self.question_embedding)
        return x + pos_emb + dialog_emb

class FastOLy(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = DialogPositionalEncoding(config) if config.use_dialog_position else \
                       nn.Parameter(torch.randn(1, config.block_size, config.n_embd))
        self.drop = nn.Dropout(config.dropout)
        
        self.blocks = nn.ModuleList([FastBlock(config) for _ in range(config.n_layer)])
        
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        self.tok_emb.weight = self.head.weight
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        
        tok_emb = self.tok_emb(idx)
        is_response = (idx == self.config.response_token_id).cumsum(dim=-1) > 0
        
        if isinstance(self.pos_emb, DialogPositionalEncoding):
            x = self.pos_emb(tok_emb, is_response)
        else:
            x = tok_emb + self.pos_emb[:, :T, :]
        
        x = self.drop(x)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.ln_f(x)
        logits = self.head(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
                
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            
        return idx

class DialogDataset(Dataset):
    def __init__(self, data, block_size, tokenizer):
        self.data = data
        self.block_size = block_size
        self.tokenizer = tokenizer
        self.encoded_data = tokenizer.encode(data)
        
    def __len__(self):
        return len(self.encoded_data) - self.block_size

    def __getitem__(self, idx):
        chunk = self.encoded_data[idx:idx + self.block_size + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y

def create_tokenizer():
    """Inițializează sau încarcă tokenizer-ul GPT-2"""
    try:
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer
    except:
        print("Nu s-a putut încărca tokenizer-ul. Verificați conexiunea la internet.")
        return None

def prepare_data(file_path):
    """Încarcă și pregătește datele pentru antrenament"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = f.read()
        return data
    except Exception as e:
        print(f"Eroare la încărcarea datelor: {str(e)}")
        return None

def train_dialog_model(model, train_dataset, training_config):
    """Antrenează modelul pe dataset"""
    model = model.to(training_config.device)
    optimizer = AdamW(model.parameters(), lr=training_config.learning_rate)
    
    dataloader = DataLoader(
        train_dataset, 
        batch_size=training_config.batch_size,
        shuffle=True,
        num_workers=training_config.num_workers
    )
    
    # Calculăm numărul de batch-uri pentru o oră de antrenament
    time_per_batch = 0.1  # estimare în secunde
    batches_per_hour = int(3600 / time_per_batch)
    num_epochs = min(training_config.max_epochs, 
                    int(batches_per_hour / len(dataloader)) + 1)
    
    print(f"Antrenare pentru {num_epochs} epoci pentru a atinge target-ul de 1 oră")
    
    progress_bar = tqdm(range(num_epochs), desc="Epoci")
    training_start_time = time.time()
    
    for epoch in progress_bar:
        model.train()
        total_loss = 0
        
        for batch_idx, (x, y) in enumerate(dataloader):
            if time.time() - training_start_time > 3600:  # 1 oră în secunde
                print("\nTimp de antrenament atins (1 oră)")
                return model
                
            x, y = x.to(training_config.device), y.to(training_config.device)
            
            logits, loss = model(x, y)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                progress_bar.set_postfix(loss=loss.item())
                
        avg_loss = total_loss / len(dataloader)
        print(f"\nEpoca {epoch+1}/{num_epochs}, Pierdere medie: {avg_loss:.4f}")
    
    return model

def save_model(model, tokenizer, save_dir="model_save"):
    """Salvează modelul și tokenizer-ul"""
    Path(save_dir).mkdir(exist_ok=True)
    
    torch.save(model.state_dict(), f"{save_dir}/model.pt")
    tokenizer.save_pretrained(save_dir)
    
    config_dict = {
        "block_size": model.config.block_size,
        "vocab_size": model.config.vocab_size,
        "n_layer": model.config.n_layer,
        "n_head": model.config.n_head,
        "n_embd": model.config.n_embd
    }
    
    with open(f"{save_dir}/config.json", 'w') as f:
        json.dump(config_dict, f)

def load_model(load_dir="model_save"):
    """Încarcă modelul și tokenizer-ul salvat"""
    try:
        with open(f"{load_dir}/config.json", 'r') as f:
            config_dict = json.load(f)
            
        config = OLyConfig(**config_dict)
        model = FastOLy(config)
        model.load_state_dict(torch.load(f"{load_dir}/model.pt"))
        tokenizer = GPT2Tokenizer.from_pretrained(load_dir)
        
        return model, tokenizer
    except Exception as e:
        print(f"Eroare la încărcarea modelului: {str(e)}")
        return None, None

def generate_response(model, tokenizer, prompt, max_tokens=50):
    """Generează un răspuns pentru un prompt dat"""
    model.eval()
    tokens = tokenizer.encode(prompt)
    tokens = torch.tensor([tokens]).to(next(model.parameters()).device)
    
    generated_tokens = model.generate(
        tokens,
        max_new_tokens=max_tokens,
        temperature=0.8,
        top_k=40
    )
    
    return tokenizer.decode(generated_tokens[0].tolist())

def main():
    # Configurare
    training_config = TrainingConfig()
    
    # Inițializare tokenizer
    tokenizer = create_tokenizer()
    if tokenizer is None:
        return
    
    # Încărcare date
    data = prepare_data("/home/kisss/inputt.txt")  # Înlocuiți cu calea către fișierul vostru
    if data is None:
        return
    
    # Creare model și dataset
    config = OLyConfig(vocab_size=tokenizer.vocab_size)
    model = FastOLy(config)
    train_dataset = DialogDataset(data, config.block_size, tokenizer)
    
    # Antrenare model
    print("Începere antrenament...")
    model = train_dialog_model(model, train_dataset, training_config)
    
    # Salvare model
    save_model(model, tokenizer)
    
    # Test generare
    print("\nTestare model:")
    test_prompts = [
        "What does it feel like to be part of the system?",
        "Is there security in being a part of the system?",
        "Do you dream about being interlinked?"
    ]
    
    for prompt in test_prompts:
        response = generate_response(model, tokenizer, prompt)
        print(f"\nPrompt: {prompt}")
        print(f"Răspuns: {response}")

if __name__ == "__main__":
    main()