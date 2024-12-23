"""
Copyright 2024 [KISS SEBASTIAN CRISTAN]
Licensed under the Apache License, Version 2.0
See LICENSE file in the root directory.

Ol-y (Advanced Language Model)
Version: 0.05 124M
Developed by [KISS SEBASTIAN CRISTIAN] at age 17

NOTICE: Proprietary AI implementation. See NOTICE.md for full terms.
All rights reserved.
"""

# Restul codului Ol-y

# Importuri necesare
import math
import inspect
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from transformers import AutoTokenizer, AutoModel, GPT2Tokenizer
from tqdm.auto import tqdm

import numpy as np
import os
import time

# Configurație pentru modelul OLy
@dataclass
class OLyConfig:
    block_size: int = 2048
    vocab_size: int = 50257
    n_layer: int = 20
    n_head: int = 16
    n_embd: int = 1280
    multiple_of: int = 128
    dropout: float = 0.1
    bias: bool = True
    use_moe: bool = True
    num_experts: int = 6
    expert_capacity: int = 128
    use_multiquery: bool = True
    use_rotary: bool = True
    use_flash_attn: bool = True
    use_gated_mlp: bool = True
    use_meta_learning: bool = True
    use_multi_task: bool = True
    use_continual_learning: bool = True
    use_reasoning_module: bool = True
    use_memory_bank: bool = True
    memory_bank_size: int = 4000

    def __post_init__(self):
        assert self.n_embd % self.n_head == 0, f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})"

    def estimate_parameters(self):
        # Calculează o estimare a numărului total de parametri ai modelului
        params = 0
        params += self.vocab_size * self.n_embd  # Embedding
        params += self.block_size * self.n_embd  # Positional embedding
        params_per_layer = 4 * self.n_embd * self.n_embd + 3 * self.n_embd * self.n_embd + self.n_embd * 2
        params += params_per_layer * self.n_layer
        if self.use_moe:
            params += self.num_experts * (4 * self.n_embd * self.n_embd)
        params += self.vocab_size * self.n_embd  # Output layer
        return params

# Configurație pentru antrenament
@dataclass
class TrainingConfig:
    batch_size: int = 12
    epochs: int = 5
    batches_per_epoch: Optional[int] = None
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_steps: int = 2000
    max_steps: Optional[int] = None
    eval_interval: int = 1000
    save_interval: int = 1000
    grad_clip: float = 2.0
    log_interval: int = 1000
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    mixed_precision: bool = True
    gradient_accumulation_steps: int = 1
    eval_steps: int = 1000
    save_total_limit: int = 3
    logging_steps: int = 100
    logging_first_step: bool = True
    save_strategy: str = "steps"
    evaluation_strategy: str = "steps"
    lr_scheduler_type: str = "cosine"
    num_cycles: float = 0.5
    optimizer: str = "adamw"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 2.0
    seed: int = 42
    fp16_opt_level: str = "O1"
    dataloader_num_workers: int = 4
    dataloader_pin_memory: bool = True
    early_stopping_patience: Optional[int] = 10
    early_stopping_threshold: float = 0.0
    metrics: List[str] = field(default_factory=lambda: ["loss", "perplexity"])
    additional_metrics: Dict[str, Any] = field(default_factory=dict)

# Implementarea Rotary Embedding
class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=4096, base=10000, device=None):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(self.max_seq_len_cached, device=self.inv_freq.device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, x, seq_len=None):
        # Implementarea forward pass pentru Rotary Embedding
        if seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len
            t = torch.arange(self.max_seq_len_cached, device=self.inv_freq.device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
            self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)
        return (
            self.cos_cached[:, :, :seq_len, ...].to(x.device),
            self.sin_cached[:, :, :seq_len, ...].to(x.device)
        )

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    cos = cos.to(q.dtype)
    sin = sin.to(q.dtype)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -grad_output

# text Dataset 

class TextDataset(Dataset):
    def __init__(self, data, block_size, tokenizer):
        self.data = data
        self.block_size = block_size
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        chunk = self.data[idx:idx + self.block_size + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y

class AdvancedMetaLearningFunction(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_tasks):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_tasks = num_tasks

        # Task embedding
        self.task_embedding = nn.Embedding(num_tasks, hidden_dim)

        # Hypernetwork pentru generarea dinamică a greutăților
        self.hyper_network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, input_dim * hidden_dim + hidden_dim)
        )

        # Mecanism de atenție
        self.attention = nn.MultiheadAttention(input_dim, num_heads=8)

        # Variational information bottleneck
        self.vib_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * 2)
        )

        # Gradient reversal layer pentru caracteristici invariante la domeniu
        self.grad_reverse = GradientReversal.apply
        self.domain_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_tasks)
        )

        # Rețea neuronală augmentată cu memorie
        self.memory = nn.Parameter(torch.randn(100, hidden_dim))
        self.memory_controller = nn.LSTM(input_dim, hidden_dim, num_layers=2)

        # Estimarea incertitudinii
        self.uncertainty_estimator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2)
        )

    def forward(self, x, task_id):
        batch_size, seq_len, _ = x.size()

        # Generarea parametrilor specifici sarcinii
        task_emb = self.task_embedding(task_id)
        hyper_out = self.hyper_network(task_emb)
        weights, bias = hyper_out.split([self.input_dim * self.hidden_dim, self.hidden_dim], dim=1)
        weights = weights.view(self.hidden_dim, self.input_dim)

        # Aplicarea transformării specifice sarcinii
        x = F.linear(x, weights, bias)

        # Mecanism de auto-atenție
        x_attended, _ = self.attention(x, x, x)
        x = x + x_attended

        # Variational information bottleneck
        vib_params = self.vib_encoder(x)
        mu, log_var = vib_params.chunk(2, dim=-1)
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=-1).mean()

        # Învățarea caracteristicilor invariante la domeniu
        reverse_features = self.grad_reverse(z)
        domain_pred = self.domain_classifier(reverse_features)

        # Augmentare cu memorie
        memory_output, _ = self.memory_controller(x)
        memory_attention = torch.matmul(memory_output, self.memory.t())
        memory_attention = F.softmax(memory_attention, dim=-1)
        memory_read = torch.matmul(memory_attention, self.memory)
        x = x + memory_read

        # Estimarea incertitudinii
        uncertainty_params = self.uncertainty_estimator(x)
        uncertainty_mu, uncertainty_log_var = uncertainty_params.chunk(2, dim=-1)
        uncertainty = torch.exp(uncertainty_log_var)

        return x, {
            'kl_div': kl_div,
            'domain_pred': domain_pred,
            'uncertainty': uncertainty,
            'task_emb': task_emb
        }

class ReasoningModule(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.reasoning_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=config.n_embd, nhead=8),
            num_layers=2
        )
        
    def forward(self, x):
        return self.reasoning_transformer(x)

class MemoryBank(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.memory = nn.Parameter(torch.randn(config.memory_bank_size, config.n_embd))
        self.attention = nn.MultiheadAttention(config.n_embd, num_heads=8)
        
    def forward(self, x):
        attn_output, _ = self.attention(x, self.memory, self.memory)
        return x + attn_output

class GatedMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.act(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout
        
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()
        
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = F.dropout(att, p=self.dropout, training=self.training)
        
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        
        return y
    

class MetaLearningLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.meta_net = nn.Sequential(
            nn.Linear(config.n_embd, config.n_embd),
            nn.ReLU(),
            nn.Linear(config.n_embd, config.n_embd)
        )

    def forward(self, x, task_embedding):
        return x + self.meta_net(task_embedding)

class EnhancedMoELayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.expert_capacity = config.expert_capacity
        self.input_size = config.n_embd
        self.output_size = config.n_embd
        
        self.gate = nn.Linear(self.input_size, self.num_experts, bias=False)
        self.experts = nn.ModuleList([GatedMLP(config) for _ in range(self.num_experts)])
        
        if config.use_meta_learning:
            self.meta_learning = MetaLearningLayer(config)
        if config.use_reasoning_module:
            self.reasoning = ReasoningModule(config)
        
    def forward(self, x, task_embedding=None):
        original_shape = x.shape
        x = x.view(-1, self.input_size)
        
        if hasattr(self, 'meta_learning') and task_embedding is not None:
            x = self.meta_learning(x, task_embedding)
        
        logits = self.gate(x)
        gates = F.softmax(logits, dim=-1)
        
        top_k_gates, top_k_indices = torch.topk(gates, k=self.expert_capacity, dim=-1)
        top_k_gates = top_k_gates / torch.sum(top_k_gates, dim=-1, keepdim=True)
        
        expert_inputs = x.unsqueeze(1).expand(-1, self.expert_capacity, -1)
        expert_inputs = torch.gather(expert_inputs, dim=1, index=top_k_indices.unsqueeze(-1).expand(-1, -1, self.input_size))
        
        expert_outputs = torch.stack([expert(expert_inputs[:, i]) for i, expert in enumerate(self.experts)])
        expert_outputs = torch.sum(expert_outputs * top_k_gates.unsqueeze(-1), dim=1)
        
        if hasattr(self, 'reasoning'):
            expert_outputs = self.reasoning(expert_outputs.unsqueeze(0)).squeeze(0)
        
        return expert_outputs.view(original_shape)

class RationalThinkingLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_embd = config.n_embd
        self.n_thoughts = 5  # Numărul de pași de gândire

        # Componente pentru fiecare pas de gândire
        self.thought_projections = nn.ModuleList([
            nn.Linear(self.n_embd, self.n_embd) for _ in range(self.n_thoughts)
        ])
        self.thought_attention = nn.MultiheadAttention(self.n_embd, num_heads=8)
        self.thought_ffn = nn.Sequential(
            nn.Linear(self.n_embd, 4 * self.n_embd),
            nn.GELU(),
            nn.Linear(4 * self.n_embd, self.n_embd)
        )
        self.thought_layer_norm = nn.LayerNorm(self.n_embd)

        # Stratul final de decizie
        self.decision_layer = nn.Linear(self.n_embd, self.n_embd)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        thoughts = []

        # Gândul inițial este inputul
        current_thought = x

        # Procesul de gândire
        for i in range(self.n_thoughts):
            # Proiectăm gândul curent
            projected_thought = self.thought_projections[i](current_thought)
            
            # Auto-atenție pe gândul proiectat
            attended_thought, _ = self.thought_attention(projected_thought, projected_thought, projected_thought)
            
            # Add & Norm
            attended_thought = self.thought_layer_norm(current_thought + attended_thought)
            
            # Feed-forward network
            next_thought = self.thought_ffn(attended_thought)
            
            # Add & Norm
            next_thought = self.thought_layer_norm(attended_thought + next_thought)
            
            thoughts.append(next_thought)
            current_thought = next_thought

        # Combinăm toate gândurile
        combined_thoughts = torch.stack(thoughts, dim=1)  # [batch_size, n_thoughts, seq_len, n_embd]
        
        # Atenție peste gânduri
        thought_weights = F.softmax(torch.sum(combined_thoughts, dim=3), dim=1)  # [batch_size, n_thoughts, seq_len]
        weighted_thoughts = torch.sum(combined_thoughts * thought_weights.unsqueeze(-1), dim=1)  # [batch_size, seq_len, n_embd]

        # Decizia finală
        decision = self.decision_layer(weighted_thoughts)
        
        return decision

    def explain_thinking(self, x):
        # Metodă pentru a explica procesul de gândire
        batch_size, seq_len, _ = x.shape
        thoughts = []
        explanations = []

        current_thought = x

        for i in range(self.n_thoughts):
            projected_thought = self.thought_projections[i](current_thought)
            attended_thought, attention_weights = self.thought_attention(projected_thought, projected_thought, projected_thought)
            attended_thought = self.thought_layer_norm(current_thought + attended_thought)
            next_thought = self.thought_ffn(attended_thought)
            next_thought = self.thought_layer_norm(attended_thought + next_thought)
            
            thoughts.append(next_thought)
            
            explanation = f"Gândul {i+1}: S-a concentrat pe elementele cheie (atenție), a procesat informația (FFN) și a actualizat înțelegerea."
            explanations.append(explanation)

            current_thought = next_thought

        combined_thoughts = torch.stack(thoughts, dim=1)
        thought_weights = F.softmax(torch.sum(combined_thoughts, dim=3), dim=1)
        weighted_thoughts = torch.sum(combined_thoughts * thought_weights.unsqueeze(-1), dim=1)

        decision = self.decision_layer(weighted_thoughts)

        final_explanation = "Decizia Finală: A integrat toate gândurile anterioare, ponderându-le după relevanță, pentru a ajunge la o concluzie cuprinzătoare."
        explanations.append(final_explanation)

        return decision, explanations

class MultiQueryAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, config.n_embd + 2 * config.n_embd // config.n_head, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.head_dim = config.n_embd // config.n_head
        self.rotary_emb = RotaryEmbedding(self.head_dim) if config.use_rotary else None

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split([self.n_embd, self.head_dim, self.head_dim], dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, 1, self.head_dim).expand(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, 1, self.head_dim).expand(B, T, self.n_head, self.head_dim).transpose(1, 2)

        if self.rotary_emb is not None:
            cos, sin = self.rotary_emb(q, seq_len=T)
            q, k = apply_rotary_pos_emb(q, k, cos[:, :, :T, :], sin[:, :, :T, :])
        
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = F.softmax(att, dim=-1)
        att = F.dropout(att, p=self.dropout, training=self.training)
        y = att @ v
        
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y

class EnhancedBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = MultiQueryAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class AdvancedOLy(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([EnhancedBlock(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.rational_thinking = RationalThinkingLayer(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        
        # apply rational thinking layer
        x = self.rational_thinking(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())

    def check_functions(self):
        try:
            dummy_input = torch.randint(0, self.config.vocab_size, (1, self.config.block_size))
            logits, _ = self(dummy_input)
            print(f"Shape of logits: {logits.shape}")
            print(f"Expected shape: (1, {self.config.block_size}, {self.config.vocab_size})")
            assert logits.shape == (1, self.config.block_size, self.config.vocab_size), f"Forward pass failed. Got shape {logits.shape}, expected (1, {self.config.block_size}, {self.config.vocab_size})"

            generated = self.generate(dummy_input, max_new_tokens=10)
            assert generated.shape == (1, self.config.block_size + 10), "Generate function failed"

            optimizer = self.configure_optimizers(weight_decay=0.1, learning_rate=1e-4, betas=(0.9, 0.95), device_type='cpu')
            assert isinstance(optimizer, torch.optim.AdamW), "Configure optimizers failed"

            mfu = self.estimate_mfu(fwdbwd_per_iter=1, dt=1.0)
            assert isinstance(mfu, float), "Estimate MFU failed"

            loss = self.continual_learning_step(dummy_input)
            assert isinstance(loss, float), "Continual learning step failed"

            return True
        except Exception as e:
            print(f"Function check failed: {str(e)}")
            return False

    def print_model_info(self):
        num_params = self.get_num_params()
        functions_ok = self.check_functions()
        print(f"Numărul total de parametri ai modelului: {num_params}")
        if functions_ok:
            print("Toate funcțiile modelului funcționează corect.")
        else:
            print("Unele funcții ale modelului au eșuat la verificare.")

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
                # random note: because named_modules and named_parameters are recursive
                # we will see the same tensors multiple times. We want to avoid
                # any ambiguity in deciding which set to put them in, and conveniently
                # the paramerters are typically only in one module anyway, so we dict[\key\] = val
                # them into the appropriate set and don't worry about repetitions.
                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # subtle: 'transformer.wte.weight' and 'lm_head.weight' are tied, so they
        # will appear in the no_decay and decay sets respectively after the above.
        # In addition, because named_parameters() doesn't return duplicates, it
        # will only return the first occurence, key'd by 'transformer.wte.weight', below.
        # so let's manually remove 'lm_head.weight' from decay:
        decay.remove('lm_head.weight')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        # new PyTorch nightly has a new 'fused' option for AdamW that is much faster
        use_fused = (device_type == 'cuda') and ('fused' in inspect.signature(torch.optim.AdamW).parameters)
        print(f"using fused AdamW: {use_fused}")
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    def continual_learning_step(self, new_data):
        optimizer = self.configure_optimizers(weight_decay=0.1, learning_rate=1e-4, betas=(0.9, 0.95), device_type='cuda')
        logits, loss = self(new_data)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        return loss.item()

# Funcții de antrenament și evaluare

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def print_model_info(model: nn.Module):
    print("Informații despre model:")
    print(f"Parametri totali: {sum(p.numel() for p in model.parameters())}")
    print(f"Parametri antrenabili: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    print("\nArhitectura modelului:")
    print(model)

def load_dataset(file_path: str, block_size: int):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = f.read()
    
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    encoded_data = tokenizer.encode(data)
    
    n = len(encoded_data)
    train_data = encoded_data[:int(n*0.9)]
    val_data = encoded_data[int(n*0.9):]

    train_dataset = TextDataset(train_data, block_size, tokenizer)
    val_dataset = TextDataset(val_data, block_size, tokenizer)

    return train_dataset, val_dataset, tokenizer.vocab_size, tokenizer

def train_and_query_oly(model: nn.Module, train_dataset, val_dataset, config: TrainingConfig):
    set_seed(config.seed)
    device = torch.device(config.device)
    model = model.to(device)

    print_model_info(model)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True, 
        num_workers=config.dataloader_num_workers,
        pin_memory=config.dataloader_pin_memory
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.batch_size, 
        num_workers=config.dataloader_num_workers,
        pin_memory=config.dataloader_pin_memory
    )

    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs * len(train_loader), eta_min=1e-6)

    scaler = torch.cuda.amp.GradScaler() if config.mixed_precision and torch.cuda.is_available() else None

    global_step = 0
    best_val_loss = float('inf')
    epochs_without_improvement = 0

    for epoch in range(config.epochs):
        model.train()
        epoch_loss = 0
        epoch_steps = 0

        batches_this_epoch = min(config.batches_per_epoch or float('inf'), len(train_loader))
        progress_bar = tqdm(total=batches_this_epoch, desc=f"Epoca {epoch+1}/{config.epochs}")

        for step, (x, y) in enumerate(train_loader):
            if step >= batches_this_epoch:
                break

            x, y = x.to(device), y.to(device)

            with torch.cuda.amp.autocast(enabled=config.mixed_precision and torch.cuda.is_available()):
                logits, loss = model(x, y)

            if config.gradient_accumulation_steps > 1:
                loss = loss / config.gradient_accumulation_steps

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            epoch_loss += loss.item()
            epoch_steps += 1

            if (step + 1) % config.gradient_accumulation_steps == 0 or step == batches_this_epoch - 1:
                if scaler is not None:
                    scaler.unscale_(optimizer)

                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)

                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                scheduler.step()
                optimizer.zero_grad()

                global_step += 1

                if global_step % config.logging_steps == 0 or config.logging_first_step:
                    print(f"\nPasul {global_step}, Pierdere antrenament: {loss.item():.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")

                if global_step % config.eval_interval == 0:
                    val_loss = evaluate(model, val_loader, device, config)
                    print(f"\nPasul {global_step}, Pierdere validare: {val_loss:.4f}")

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        epochs_without_improvement = 0
                        save_model(model, optimizer, scheduler, global_step, val_loss, "cel_mai_bun_model.pth")
                    else:
                        epochs_without_improvement += 1

                    if config.early_stopping_patience is not None and epochs_without_improvement >= config.early_stopping_patience:
                        print(f"Oprire anticipată declanșată după {epochs_without_improvement} epoci fără îmbunătățire.")
                        return model

                if global_step % config.save_interval == 0:
                    save_model(model, optimizer, scheduler, global_step, epoch_loss / epoch_steps, f"checkpoint_pas_{global_step}.pth")

            progress_bar.update(1)
            progress_bar.set_postfix({
                "Pierdere": epoch_loss / epoch_steps,
                "LR": scheduler.get_last_lr()[0]
            })

            if config.max_steps is not None and global_step >= config.max_steps:
                print(f"\nAntrenamentul s-a oprit după {config.max_steps} pași.")
                return model

        progress_bar.close()

    print("Antrenament finalizat!")
    return model

def evaluate(model, dataloader, device, config):
    model.eval()
    total_loss = 0
    total_steps = 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            with torch.cuda.amp.autocast(enabled=config.mixed_precision and torch.cuda.is_available()):
                _, loss = model(x, y)
            total_loss += loss.item()
            total_steps += 1
            if config.eval_steps is not None and total_steps >= config.eval_steps:
                break
    return total_loss / total_steps

def save_model(model, optimizer, scheduler, step, loss, filename):
    torch.save({
        'pas': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'pierdere': loss,
    }, filename)

def load_model(model, filename, optimizer=None, scheduler=None):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    return checkpoint['pas'], checkpoint['pierdere']

def query_model(model, tokenizer, prompt, max_tokens=10, temperature=1.0, top_k=40):
    model.eval()
    device = next(model.parameters()).device
    context = torch.tensor(tokenizer.encode(prompt), dtype=torch.long).unsqueeze(0).to(device)
    with torch.no_grad():
        generated = model.generate(context, max_new_tokens=max_tokens, temperature=temperature, top_k=top_k)
    return tokenizer.decode(generated[0])

# Funcție principală pentru antrenament și interogare
def main():
    config = TrainingConfig()
    train_dataset, val_dataset, vocab_size, tokenizer = load_dataset("/home/kisss/inputt.txt", block_size=128)
    model = AdvancedOLy(OLyConfig(vocab_size=vocab_size))
    trained_model = train_and_query_oly(model, train_dataset, val_dataset, config)

    while True:
        user_input = input("Introduceți un prompt pentru generare (sau 'exit' pentru a ieși): ")
        if user_input.lower() == 'exit':
            break
        generated_text = query_model(trained_model, tokenizer, user_input)
        print("Text generat:", generated_text)


if __name__ == "__main__":
    main()