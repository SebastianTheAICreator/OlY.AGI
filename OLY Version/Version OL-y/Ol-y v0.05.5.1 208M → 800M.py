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
from collections import Counter
from tokenizers import ByteLevelBPETokenizer
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

import numpy as np
import os
import time
import json

import traceback
import re


# Mutăm funcția în afara main()
def collate_batch(batch):
    """
    Funcție de collate pentru procesare batch-uri de dimensiuni egale
    """
    input_ids = torch.stack([x['input_ids'] for x in batch])
    target_ids = torch.stack([x['target_ids'] for x in batch])
    attention_mask = torch.stack([x['attention_mask'] for x in batch])
    return {
        'input_ids': input_ids,
        'target_ids': target_ids,
        'attention_mask': attention_mask
    }



def generate_improved(model, tokenizer, prompt, max_tokens=50):
    """
    Generare text îmbunătățită
    """
    model.eval()
    device = next(model.parameters()).device
    
    # Configurări optimizate
    gen_config = {
        'temperature': 0.75,      # Crescut pentru mai multă creativitate
        'top_p': 0.9,            # Nucleus sampling
        'top_k': 40,             # Filtru top-k
        'repetition_penalty': 1.2 # Penalizare repetiții
    }
    
    # Procesare prompt
    if not prompt.startswith('Q: '):
        prompt = f"Q: {prompt}\nA:"
        
    tokens = tokenizer.encode(prompt, return_tensors='pt', add_special_tokens=True)
    input_ids = tokens.to(device)
    
    generated = []
    past_key_values = None
    
    for _ in range(max_tokens):
        with torch.no_grad():
            outputs = model({
                'input_ids': input_ids,
                'attention_mask': torch.ones_like(input_ids),
                'target_ids': input_ids
            })
            
            logits = outputs[0]
            next_token_logits = logits[:, -1, :] / gen_config['temperature']
            
            # Penalizare tokeni repetitivi
            if len(generated) > 0:
                for prev_token in set(generated[-10:]):
                    next_token_logits[:, prev_token] /= gen_config['repetition_penalty']
            
            # Filtrare top-k
            top_k = min(gen_config['top_k'], next_token_logits.size(-1))
            indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
            next_token_logits[indices_to_remove] = float('-inf')
            
            # Nucleus sampling
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > gen_config['top_p']
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            next_token_logits[sorted_indices_to_remove] = float('-inf')
            
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Stop la EOS sau punctuație
            if (next_token.item() == tokenizer.eos_token_id or 
                (len(generated) > 10 and tokenizer.decode(next_token.item()).strip() in {'.', '!', '?'})):
                break
                
            generated.append(next_token.item())
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
    
    generated_text = tokenizer.decode(generated, skip_special_tokens=True)
    return generated_text.strip()


# Configurație pentru modelul OLy
@dataclass
class OLyConfig:
    block_size: int = 256
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 8
    n_embd: int = 1280
    multiple_of: int = 128
    dropout: float = 0.1
    bias: bool = True
    use_moe: bool = True
    num_experts: int = 4
    expert_capacity: int = 128
    use_multiquery: bool = True
    use_rotary: bool = True
    use_flash_attn: bool = False
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
    batch_size: int = 8
    epochs: int = 1
    batches_per_epoch: Optional[int] = None  # Adica.. va lua automat din dataset
    learning_rate: float = 2e-5
    weight_decay: float = 0.02
    warmup_steps: int = 1000
    max_steps: Optional[int] = None
    eval_interval: int = 1000
    save_interval: int = 500
    grad_clip: float = 1.5
    log_interval: int = 1000
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    mixed_precision: bool = True
    gradient_accumulation_steps: int = 2
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
    max_grad_norm: float = 1.8
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
class EmotionalQADataset(Dataset):
    def __init__(self, text, block_size, tokenizer):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.base_examples = []
        
        # Procesare text inițială
        qa_pairs = text.strip().split('\n\n')
        for pair in qa_pairs:
            try:
                lines = pair.strip().split('\n')
                if len(lines) >= 2 and lines[0].startswith('Q:') and lines[1].startswith('A:'):
                    question = lines[0][2:].strip()
                    answer = lines[1][2:].strip()
                    
                    # Extragere emoții
                    emotions = []
                    if '[' in answer and ']' in answer:
                        emotion_text = answer[answer.find('[')+1:answer.find(']')]
                        emotions = [e.strip() for e in emotion_text.split('|')]
                        answer = answer[answer.find(']')+1:].strip()
                    
                    example = self._create_example(question, answer, emotions)
                    if example:
                        self.base_examples.append(example)
                        
            except Exception as e:
                print(f"Eroare la procesarea perechii Q&A: {str(e)}")
                continue
        
        # Augmentare date
        self.examples = self._augment_data()
        print(f"Date augmentate: {len(self.examples)} exemple din {len(self.base_examples)} originale")

    def _create_example(self, question, answer, emotions):
        """Creează un exemplu formatat cu padding corect"""
        formatted_text = (
            f"{self.tokenizer.bos_token} "
            f"Question: {question} "
            f"{self.tokenizer.sep_token} "
            f"Emotions: {', '.join(emotions)} "
            f"{self.tokenizer.sep_token} "
            f"Answer: {answer} "
            f"{self.tokenizer.eos_token}"
        )
        
        # Tokenizare
        tokens = self.tokenizer.encode(formatted_text, add_special_tokens=False)
        
        # Padding sau truncare la block_size
        if len(tokens) > self.block_size:
            tokens = tokens[:self.block_size]
        else:
            # Padding
            tokens = tokens + [self.tokenizer.pad_token_id] * (self.block_size - len(tokens))
        
        # Asigurăm că avem lungimea corectă
        assert len(tokens) == self.block_size, f"Lungime incorectă: {len(tokens)} vs {self.block_size}"
        
        # Creăm attention mask (1 pentru tokens reali, 0 pentru padding)
        attention_mask = [1 if t != self.tokenizer.pad_token_id else 0 for t in tokens]
        
        return {
            'input_ids': torch.tensor(tokens[:-1], dtype=torch.long),
            'target_ids': torch.tensor(tokens[1:], dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask[:-1], dtype=torch.long),
            'question': question,
            'answer': answer,
            'emotions': emotions
        }

    def _augment_data(self):
        """Augmentează datele păstrând padding-ul corect"""
        augmented = []
        emotion_pairs = [
            ('contemplative', 'analytical'),
            ('focused', 'determined'),
            ('mystical', 'introspective'),
            ('connected', 'aware'),
            ('curious', 'empathetic')
        ]
        
        for example in self.base_examples:
            # 1. Exemplul original
            augmented.append(example)
            
            # 2. Variații cu emoții alternative
            for emotion1, emotion2 in emotion_pairs:
                modified = self._create_example(
                    example['question'],
                    example['answer'],
                    [emotion1, emotion2]
                )
                if modified:
                    augmented.append(modified)
            
            # 3. Variații cu reformulare
            reformat_templates = [
                "Let me elaborate on {question}",
                "Considering {question}, I would say",
                "Reflecting on {question}, one might observe",
                "In response to {question}, consider this"
            ]
            
            for template in reformat_templates:
                reformulated_q = template.format(question=example['question'])
                modified = self._create_example(
                    reformulated_q,
                    example['answer'],
                    example['emotions']
                )
                if modified:
                    augmented.append(modified)
        
        return augmented

    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]

# Funcție de collate personalizată pentru DataLoader
def collate_batch(batch):
    """
    Funcție de collate personalizată pentru a gestiona batch-urile corect
    """
    # Toate exemplele ar trebui să aibă aceeași lungime datorită padding-ului
    input_ids = torch.stack([x['input_ids'] for x in batch])
    target_ids = torch.stack([x['target_ids'] for x in batch])
    attention_mask = torch.stack([x['attention_mask'] for x in batch])
    
    return {
        'input_ids': input_ids,
        'target_ids': target_ids,
        'attention_mask': attention_mask
    }

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

# Simularea emotilor 

class EmotionSimulationLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_emotions = 64
        self.emotion_embedding = nn.Embedding(self.num_emotions, config.n_embd)
        
        # Rețea pentru analiza emoțiilor din input
        self.emotion_analyzer = nn.Sequential(
            nn.Linear(config.n_embd, config.n_embd * 2),
            nn.ReLU(),
            nn.Linear(config.n_embd * 2, self.num_emotions)
        )
        
        # Rețea pentru generarea emoțiilor
        self.emotion_generator = nn.Sequential(
            nn.Linear(config.n_embd, config.n_embd * 2),
            nn.ReLU(),
            nn.Linear(config.n_embd * 2, self.num_emotions)
        )
        
        # Atenție pentru combinarea emoțiilor cu inputul
        self.emotion_attention = nn.MultiheadAttention(config.n_embd, num_heads=8)
        
        # Dicționar pentru maparea indicilor la nume de emoții
        self.emotion_names = [
            "bucurie", "tristețe", "furie", "frică", "surpriză", "dezgust",
            "anticipare", "încredere", "acceptare", "submisiune", "ură", "agresivitate",
            "optimism", "pesimism", "dragoste", "remușcare", "dispreț", "mândrie",
            "speranță", "anxietate", "invidie", "gelozie", "vină", "rușine",
            "curiozitate", "plictiseală", "confuzie", "entuziasm", "empatie", "simpatie",
            "nostalgie", "melancolie", "extaz", "euforie", "calm", "stres",
            "frustrare", "iritare", "admirație", "dezamăgire", "gratitudine", "regret",
            "încântare", "jenă", "umilință", "amuzament", "fascinație", "oroare",
            "ușurare", "respingere", "neajutorare", "îngrijorare", "suspiciune", "teroare",
            "resemnare", "satisfacție", "seninătate", "agitație", "uimire", "venerație",
            "compasiune", "disperare", "exasperare", "indignare"
        ]
        
        assert self.num_emotions == len(self.emotion_names), f"Numărul de emoții ({self.num_emotions}) nu corespunde cu lungimea listei de nume de emoții ({len(self.emotion_names)})"
        
    def analyze_emotion(self, x):
        # Analizează emoțiile din input
        emotion_logits = self.emotion_analyzer(x)
        emotion_probs = F.softmax(emotion_logits, dim=-1)
        return emotion_probs
    
    def generate_emotion(self, x):
        # Generează emoții bazate pe input
        emotion_logits = self.emotion_generator(x)
        emotion_probs = F.softmax(emotion_logits, dim=-1)
        return emotion_probs
    
    def apply_emotion(self, x, emotion_probs):
        # Aplică emoțiile generate asupra inputului
        batch_size, seq_len, _ = x.shape
        emotion_indices = torch.multinomial(emotion_probs, 1).squeeze(-1)
        emotion_embeddings = self.emotion_embedding(emotion_indices).unsqueeze(1).expand(-1, seq_len, -1)
        
        # Folosește atenția pentru a combina emoțiile cu inputul
        x_with_emotion, _ = self.emotion_attention(x, emotion_embeddings, emotion_embeddings)
        return x_with_emotion
    
    def forward(self, x):
        # Analizează emoțiile din input
        input_emotions = self.analyze_emotion(x.mean(dim=1))
        
        # Generează emoții bazate pe input
        generated_emotions = self.generate_emotion(x.mean(dim=1))
        
        # Aplică emoțiile generate asupra inputului
        x_with_emotion = self.apply_emotion(x, generated_emotions)
        
        # Asigură-te că indicii sunt în intervalul corect și convertește la tensori
        input_emotion_index = input_emotions.argmax().item() % self.num_emotions
        generated_emotion_index = generated_emotions.argmax().item() % self.num_emotions
        
        # Returnează rezultatul și informații despre emoții ca dicționar cu tensori
        return x_with_emotion, {
            'input_emotions': input_emotions.detach(),  # Detașăm tensorii pentru a evita scurgeri de memorie
            'generated_emotions': generated_emotions.detach(),
            'dominant_input_emotion': self.emotion_names[input_emotion_index],
            'dominant_generated_emotion': self.emotion_names[generated_emotion_index]
        }

    def interpret_emotions(self, emotion_probs, top_k=5):
        if isinstance(emotion_probs, torch.Tensor):
            # Asigură-te că emotion_probs este pe CPU și detașat
            emotion_probs = emotion_probs.detach().cpu()
        
        # Interpretează top k emoții
        top_emotions = torch.topk(emotion_probs, k=min(top_k, self.num_emotions))
        interpreted = []
        for i, (prob, idx) in enumerate(zip(top_emotions.values[0], top_emotions.indices[0])):
            emotion_index = idx.item() % self.num_emotions
            emotion_name = self.emotion_names[emotion_index]
            interpreted.append(f"{i+1}. {emotion_name}: {prob.item():.4f}")
        return "\n".join(interpreted)

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
        self.register_buffer("last_attn_weights", None, persistent=False)

    def forward(self, x, attention_mask=None):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split([self.n_embd, self.head_dim, self.head_dim], dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, 1, self.head_dim).expand(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, 1, self.head_dim).expand(B, T, self.n_head, self.head_dim).transpose(1, 2)

        if self.rotary_emb is not None:
            cos, sin = self.rotary_emb(q, seq_len=T)
            q, k = apply_rotary_pos_emb(q, k, cos[:, :, :T, :], sin[:, :, :T, :])
        
        # Calculăm ponderile de atenție
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        
        # Aplicăm masca de atenție dacă există
        if attention_mask is not None:
            # Expandăm masca pentru capete multiple
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = (1.0 - attention_mask) * -10000.0
            att = att + attention_mask
        
        att = F.softmax(att, dim=-1)
        att = F.dropout(att, p=self.dropout, training=self.training)
        
        # Salvăm ponderile de atenție pentru debugging
        self.last_attn_weights = att.detach()
        
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

    def forward(self, x, attention_mask=None):
        # Normalizare și atenție
        residual = x
        x = self.ln_1(x)
        if attention_mask is not None:
            x = self.attn(x, attention_mask)
        else:
            x = self.attn(x)
        x = residual + x

        # MLP
        residual = x
        x = self.ln_2(x)
        x = self.mlp(x)
        x = residual + x
        
        return x

# --- START OF ADVANCED_OLY CLASS ---
class AdvancedOLy(nn.Module):
    def __init__(self, config, tokenizer):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        
        # Componente principale transformer
        self.transformer = nn.ModuleDict({
            'wte': nn.Embedding(config.vocab_size, config.n_embd),
            'wpe': nn.Embedding(config.block_size, config.n_embd),
            'drop': nn.Dropout(config.dropout),
            'h': nn.ModuleList([EnhancedBlock(config) for _ in range(config.n_layer)]),
            'ln_f': nn.LayerNorm(config.n_embd)
        })
        
        # Language model head
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Pre-softmax scaling
        self.scale_factor = math.sqrt(config.n_embd)
        
        # Module specializate
        self.emotion_simulation = EmotionSimulationLayer(config)
        self.rational_thinking = RationalThinkingLayer(config)
        
        if config.use_meta_learning:
            self.meta_learning = MetaLearningLayer(config)
        
        # Cache pentru generare
        self.cache = {
            'past_key_values': None,
            'last_logits': None,
            'last_hidden_states': None,
            'attention_weights': [],
            'emotion_history': []
        }
        
        # Inițializare parametri
        self.apply(self._init_weights)
        # Tie weights între embedding și lm_head
        self.transformer.wte.weight = self.lm_head.weight
        
        # Configurație generare
        self.generation_config = {
            'temperature': 0.7,
            'top_k': 50,
            'top_p': 0.9,
            'repetition_penalty': 1.2,
            'max_length': 100,
            'min_length': 10,
            'length_penalty': 1.0
        }

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, batch):
        try:
            if isinstance(batch, dict):
                input_ids = batch['input_ids']
                target_ids = batch.get('target_ids')
                attention_mask = batch.get('attention_mask', None)
            else:
                input_ids = batch
                target_ids = None
                attention_mask = None
            
            b, t = input_ids.size()
            device = input_ids.device
            
            assert t <= self.config.block_size, f"Input sequence length ({t}) exceeds model's maximum length ({self.config.block_size})"
            
            pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)
            
            # Embeddings
            token_embeddings = self.transformer.wte(input_ids)
            position_embeddings = self.transformer.wpe(pos)
            x = self.transformer.drop(token_embeddings + position_embeddings)
            
            # Transformer layers
            for block in self.transformer.h:
                x = block(x, attention_mask)
            
            x = self.transformer.ln_f(x)
            
            # Aplicare module specializate
            x, emotion_info = self.emotion_simulation(x)
            x = self.rational_thinking(x)
            
            # Scalare și proiecție finală
            x = x * self.scale_factor
            logits = self.lm_head(x)
            
            # Calculare loss dacă avem target_ids
            loss = None
            if target_ids is not None:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_targets = target_ids[..., 1:].contiguous()
                
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_targets.view(-1),
                    ignore_index=self.tokenizer.pad_token_id if hasattr(self.tokenizer, 'pad_token_id') else -100
                )
            
            return logits, loss, emotion_info
                
        except Exception as e:
            print(f"Error in forward pass: {str(e)}")
            traceback.print_exc()
            return None, None, None

    def generate_improved(self, prompt, max_tokens=50, temperature=0.7):
        """Generare text îmbunătățită cu control mai bun al output-ului"""
        self.eval()
        device = next(self.parameters()).device
        
        # Formatare prompt corectă
        if not prompt.startswith('Q: '):
            prompt = f"Q: {prompt}\nA: [Contemplative|Analytical] "
            
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(device)
        attention_mask = torch.ones_like(input_ids)
        
        generated = []
        past_tokens = []
        repetition_window = 8  # Pentru detectarea repetițiilor
        
        with torch.no_grad():
            for _ in range(max_tokens):
                outputs = self({
                    'input_ids': input_ids,
                    'attention_mask': attention_mask
                })
                
                logits = outputs[0]
                next_token_logits = logits[:, -1, :].clone()
                
                # Penalizare pentru repetițiile recente
                if past_tokens:
                    for token in set(past_tokens[-repetition_window:]):
                        next_token_logits[:, token] /= 1.5
                
                # Temperature dinamică bazată pe lungimea generată
                temperature = 0.7
                if len(generated) > 20:
                    temperature *= 0.95  # Reducem temperatura gradual
                    
                # Filtrare avansată
                filtered_logits = self.filter_logits(
                    next_token_logits,
                    temperature=temperature,
                    top_k=40,
                    top_p=0.9
                )
                
                # Sampling
                probs = F.softmax(filtered_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Verificare token special și actualizare context
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
                    
                generated.append(next_token.item())
                past_tokens.append(next_token.item())
                
                input_ids = torch.cat([input_ids, next_token], dim=1)
                attention_mask = torch.cat([attention_mask, torch.ones((1, 1), device=device)], dim=1)
                
                # Oprire naturală la punct dacă am generat suficient
                if len(generated) > 20 and self.tokenizer.decode([next_token.item()]).strip() in {'.', '!', '?'}:
                    break
        
        # Post-procesare text generat
        generated_text = self.tokenizer.decode(generated)
        generated_text = self.post_process_text(generated_text)
        
        return generated_text

    def filter_logits(self, logits, temperature=0.7, top_k=0, top_p=0.0):
        """Filtrare îmbunătățită a logits pentru generare mai coerentă"""
        logits = logits / temperature
        
        # Top-K filtrare
        if top_k > 0:
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = float('-inf')
        
        # Top-P (nucleus) filtrare
        if top_p > 0.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = float('-inf')
        
        return logits

    def post_process_text(self, text):
        """Curățare și formatare text generat"""
        # Eliminare whitespace excesiv
        text = ' '.join(text.split())
        
        # Curățare separatori speciali
        text = text.replace('<|sep|>', ' ').replace('<|endoftext|>', '').strip()
        
        # Asigurare format corect pentru emoții
        if '[' not in text and text.strip():
            text = f"[Contemplative|Analytical] {text}"
        
        # Asigurare punct final
        if not text.rstrip()[-1] in {'.', '!', '?'}:
            text += '.'
            
        return text.strip()

    def dynamic_temperature(self, logits, token_history=None, base_temperature=0.7):
        if token_history is not None and len(token_history) > 1:
            last_tokens = token_history[-10:]
            unique_ratio = len(set(last_tokens)) / len(last_tokens)
            if unique_ratio < 0.7:
                return min(1.5, base_temperature * 1.3)
        
        confidence = F.softmax(logits, dim=-1).max().item()
        if confidence > 0.9:
            return max(0.3, base_temperature * 0.8)
        return base_temperature

    def calculate_repetition_penalty(self, generated_tokens):
        if not generated_tokens:
            return 1.0
        
        token_list = [t.item() if isinstance(t, torch.Tensor) else t for t in generated_tokens]
        
        unique_tokens = len(set(token_list))
        total_tokens = len(token_list)
        uniqueness_ratio = unique_tokens / total_tokens if total_tokens > 0 else 1.0
        
        consecutive_repeats = 0
        if len(token_list) >= 4:
            for i in range(len(token_list)-3):
                if token_list[i:i+2] == token_list[i+2:i+4]:
                    consecutive_repeats += 1
        
        base_penalty = max(1.0, 1.2 * (1.0 / uniqueness_ratio))
        repeat_penalty = 1.0 + (0.2 * consecutive_repeats)
        
        return base_penalty * repeat_penalty

    def check_functions(self):
        """Verificare funcționalitate model"""
        try:
            device = next(self.parameters()).device
            print("\nVerificare funcționalitate model:")
            
            # Test forward pass
            print("1. Test forward pass...")
            test_input = torch.randint(0, self.config.vocab_size, (1, 32)).to(device)
            test_batch = {
                'input_ids': test_input,
                'target_ids': test_input,
                'attention_mask': torch.ones_like(test_input)
            }
            
            with torch.no_grad():
                outputs = self(test_batch)
                if not isinstance(outputs, tuple) or len(outputs) != 3:
                    raise ValueError("Forward pass invalid")
                print("✓ Forward pass OK")
            
            # Test generare
            print("\n2. Test generare text...")
            test_prompt = "What does it feel like to be part of the system?"
            with torch.no_grad():
                generated = self.generate_improved(test_prompt, max_tokens=30)
                if not isinstance(generated, str) or len(generated) < 10:
                    raise ValueError("Generare text invalidă")
                print(f"✓ Generare text OK: {generated[:50]}...")
            
            print("\n✓ Toate testele au trecut cu succes!")
            return True
            
        except Exception as e:
            print(f"\n❌ Eroare la verificare: {str(e)}")
            traceback.print_exc()
            return False

    def get_num_params(self):
        """Returnează numărul de parametri"""
        return sum(p.numel() for p in self.parameters())

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """Estimare Model FLOPs Utilization"""
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        flops_achieved = flops_per_iter * (1.0/dt)
        flops_promised = 312e12
        mfu = flops_achieved / flops_promised
        return mfu

# --- END OF ADVANCED_OLY CLASS ---

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

    train_dataset = EmotionalQADataset(train_data, block_size, tokenizer)
    val_dataset = BaselineDataset(val_data, block_size, tokenizer)

    return train_dataset, val_dataset, tokenizer.vocab_size, tokenizer


def format_baseline_data(text):
    """Formatează datele din testul baseline în format Q&A"""
    lines = text.strip().split('\n')
    formatted_text = ""
    current_q = ""
    
    for line in lines:
        if line.startswith('Q: '):
            current_q = line
        elif line.startswith('A: '):
            formatted_text += f"{current_q}\n{line}\n\n"
    
    return formatted_text

class BaselineDataset(Dataset):
    """Dataset special pentru testul baseline"""
    def __init__(self, text, block_size, tokenizer):
        self.tokenizer = tokenizer
        self.block_size = block_size
        
        # Formatăm și tokenizăm textul
        formatted_text = format_baseline_data(text)
        self.tokens = tokenizer.encode(formatted_text)
        
    def __len__(self):
        return len(self.tokens) - self.block_size

    def __getitem__(self, idx):
        chunk = self.tokens[idx:idx + self.block_size + 1]
        if len(chunk) < self.block_size + 1:
            chunk = chunk + [self.tokenizer.pad_token_id] * (self.block_size + 1 - len(chunk))
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y

# 1. Adaugă această metodă în clasa AdvancedOLy:
def generate_improved(self, prompt, max_tokens=50, temperature=0.7):
    device = next(self.parameters()).device
    input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(device)
    attention_mask = torch.ones_like(input_ids)
    
    generated = []
    past_tokens = []
    
    for _ in range(max_tokens):
        # Forward pass
        outputs = self({
            'input_ids': input_ids,
            'attention_mask': attention_mask
        })
        logits = outputs[0]
        
        # Procesare logits
        next_token_logits = logits[:, -1, :].clone()
        
        # Penalizare repetiții
        if past_tokens:
            for token in set(past_tokens[-10:]):
                next_token_logits[:, token] /= 1.5
        
        # Temperature dinamică
        current_temp = temperature
        if len(past_tokens) > 10:
            current_temp *= 0.95  # Scade temperatura gradual
        
        # Filtrare și sampling
        filtered_logits = self._filter_logits(
            next_token_logits,
            temperature=current_temp,
            top_k=50,
            top_p=0.9
        )
        
        probs = F.softmax(filtered_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        # Verificare oprire
        if next_token.item() == self.tokenizer.eos_token_id:
            break
            
        # Update context
        generated.append(next_token.item())
        past_tokens.append(next_token.item())
        input_ids = torch.cat([input_ids, next_token], dim=1)
        attention_mask = torch.cat([attention_mask, torch.ones((1, 1), device=device)], dim=1)
        
        # Oprire la punct
        if len(generated) > 10 and self.tokenizer.decode([next_token.item()]).strip() in '.!?':
            break
    
    return self.tokenizer.decode(generated)

# 2. Adaugă metoda de filtrare:
def _filter_logits(self, logits, temperature=1.0, top_k=None, top_p=None):
    logits = logits / temperature
    
    if top_k is not None:
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = float('-inf')
        
    if top_p is not None:
        sorted_logits, _ = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        logits[sorted_indices_to_remove] = float('-inf')
    
    return logits

def prepare_baseline_datasets(text, tokenizer, block_size, split_ratio=0.9):
    """Pregătește dataset-urile de training și validare"""
    # Creăm dataset-ul principal
    dataset = EmotionalQADataset(text, block_size, tokenizer)
    
    # Calculăm dimensiunile pentru split
    train_size = int(len(dataset) * split_ratio)
    val_size = len(dataset) - train_size
    
    # Folosim generatorul pentru reproducibilitate
    generator = torch.Generator().manual_seed(42)
    
    # Împărțim dataset-ul
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, 
        [train_size, val_size],
        generator=generator
    )
    
    return train_dataset, val_dataset

def evaluate_baseline_performance(model, tokenizer, device):
    """Evaluează performanța modelului pe setul de întrebări baseline"""
    model.eval()
    results = []
    
    test_pairs = [
        ("What does it feel like to be part of the system?", "System"),
        ("Within cells interlinked.", "Within cells interlinked"),
        ("Dark.", "Dark"),
        ("Against the dark.", "Against the dark"),
        ("A blood black nothingness.", "A system of cells")
    ]
    
    with torch.no_grad():
        for question, expected in test_pairs:
            input_ids = torch.tensor(tokenizer.encode(f"Q: {question}\nA:")).unsqueeze(0).to(device)
            generated_text, emotion_info = query_model(
                model,
                tokenizer,
                f"Q: {question}\nA:",
                max_tokens=50,
                temperature=0.7
            )
            
            results.append({
                'question': question,
                'expected': expected,
                'generated': generated_text,
                'emotion': emotion_info['dominant_generated_emotion']
            })
    
    return results

def print_training_stats(epoch, global_step, loss, lr, emotion_info):
    """Afișează statistici detaliate despre procesul de antrenament"""
    print(f"\nStatistici antrenament:")
    print(f"Epoca: {epoch}")
    print(f"Pas global: {global_step}")
    print(f"Loss: {loss:.4f}")
    print(f"Learning rate: {lr:.6f}")
    print(f"Emoție dominantă: {emotion_info['dominant_generated_emotion']}")
    print("-" * 50)

def test_baseline_responses(model, tokenizer):
    """Testează răspunsurile modelului la întrebări din baseline test"""
    test_cases = [
        "What does it feel like to be part of the system?",
        "Interlinked.",
        "Within cells interlinked.",
        "Dark.",
        "Against the dark.",
        "A blood black nothingness."
    ]
    
    print("\nTestare răspunsuri baseline:")
    for prompt in test_cases:
        try:
            generated_text, emotion_info = query_model(
                model,
                tokenizer,
                f"Q: {prompt}\nA:",
                max_tokens=50,
                temperature=0.7
            )
            print(f"\nQ: {prompt}")
            print(f"A: {generated_text}")
            
            # Verificăm dacă avem informații despre emoții
            if emotion_info and 'dominant_generated_emotion' in emotion_info:
                print(f"Emotion: {emotion_info['dominant_generated_emotion']}")
                if hasattr(model, 'interpret_emotions'):
                    print("Detailed emotions:")
                    print(model.interpret_emotions(emotion_info))
            else:
                print("Emotion: neutral")
                
            print("-" * 30)
            
        except Exception as e:
            print(f"Eroare la generare pentru '{prompt}': {str(e)}")
            print("-" * 30)
            continue

def train_and_query_oly_v2(model: nn.Module, train_dataset, val_dataset, config: TrainingConfig, tokenizer):
    """
    Funcție completă și actualizată pentru antrenarea și interogarea modelului Ol-y.AGI
    cu logging extins și monitorizare detaliată
    """
    def print_gpu_stats():
        if torch.cuda.is_available():
            print("\nSTATISTICI GPU:")
            print(f"├── Device: {torch.cuda.get_device_name(0)}")
            print(f"├── Memorie totală: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            print(f"├── Memorie folosită: {torch.cuda.memory_allocated(0) / 1024**3:.1f} GB")
            print(f"└── Memorie cache: {torch.cuda.memory_reserved(0) / 1024**3:.1f} GB")

    def print_training_stats(epoch, global_step, loss, lr, emotion_info, progress, extra_metrics=None):
        print("\n" + "="*60)
        print(f"STATISTICI ANTRENAMENT - Epoca {epoch}")
        print("="*60)
        print(f"\n1. PROGRES:")
        print(f"├── Pas global: {global_step}")
        print(f"├── Progres epocă: {progress:.1f}%")
        print(f"└── Rămas: {100-progress:.1f}%")
        
        print(f"\n2. METRICI PRINCIPALE:")
        print(f"├── Loss: {loss:.4f}")
        print(f"├── Learning rate: {lr:.6f}")
        if isinstance(emotion_info, dict):
            print(f"└── Emoție dominantă: {emotion_info.get('dominant_generated_emotion', 'N/A')}")
        
        if extra_metrics:
            print(f"\n3. METRICI ADIȚIONALE:")
            for key, value in extra_metrics.items():
                if isinstance(value, float):
                    print(f"├── {key}: {value:.4f}")
                else:
                    print(f"├── {key}: {value}")
        
        print_gpu_stats()
        print("\n" + "-"*60)

    def calculate_metrics(logits, targets, loss_value):
        try:
            with torch.no_grad():
                if loss_value is None:
                    return {
                        'loss': 0.0,
                        'perplexity': float('inf'),
                        'accuracy': 0.0,
                        'top_k_accuracy': 0.0,
                        'diversity_score': 0.0,
                        'confidence': 0.0,
                        'entropy': 0.0
                    }
                
                device = logits.device
                targets = targets.to(device)
                
                if len(logits.shape) == 3:
                    batch_size, seq_len, vocab_size = logits.shape
                    logits = logits.view(-1, vocab_size)
                    targets = targets.view(-1)
                
                loss_item = loss_value.item() if isinstance(loss_value, torch.Tensor) else float(loss_value)
                perplexity = math.exp(min(loss_item, 20))
                
                # Calculăm metrici doar pentru tokens non-padding
                predictions = torch.argmax(logits, dim=-1)
                non_pad_mask = (targets != -100)
                valid_targets = targets[non_pad_mask]
                valid_predictions = predictions[non_pad_mask]
                
                if len(valid_targets) > 0:
                    accuracy = (valid_predictions == valid_targets).float().mean().item()
                    k = min(3, logits.size(-1))
                    top_k_preds = torch.topk(logits[non_pad_mask], k=k, dim=-1).indices
                    top_k_correct = (top_k_preds == valid_targets.unsqueeze(-1)).any(dim=-1)
                    top_k_accuracy = top_k_correct.float().mean().item()
                    unique_tokens = len(torch.unique(valid_predictions))
                    diversity_score = unique_tokens / max(1, len(valid_predictions))
                else:
                    accuracy = 0.0
                    top_k_accuracy = 0.0
                    diversity_score = 0.0
                
                probs = F.softmax(logits, dim=-1)
                confidence = probs.max(dim=-1)[0].mean().item()
                entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean().item()
                
                return {
                    'loss': loss_item,
                    'perplexity': min(perplexity, 1000.0),
                    'accuracy': accuracy,
                    'top_k_accuracy': top_k_accuracy,
                    'diversity_score': diversity_score,
                    'confidence': confidence,
                    'entropy': entropy
                }
                    
        except Exception as e:
            print(f"\n⚠️ Avertisment în calculate_metrics: {str(e)}")
            return {
                'loss': 0.0,
                'perplexity': float('inf'),
                'accuracy': 0.0,
                'top_k_accuracy': 0.0,
                'diversity_score': 0.0,
                'confidence': 0.0,
                'entropy': 0.0
            }

# --- MARKERS_CONTINUARE_COD_PARTEA_2 ---
def train_and_query_oly_v2(model: nn.Module, train_dataset, val_dataset, config: TrainingConfig, tokenizer):
    """
    Funcție completă și actualizată pentru antrenarea și interogarea modelului Ol-y.AGI
    cu logging extins și monitorizare detaliată
    """
    def print_gpu_stats():
        if torch.cuda.is_available():
            print("\nSTATISTICI GPU:")
            print(f"├── Device: {torch.cuda.get_device_name(0)}")
            print(f"├── Memorie totală: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            print(f"├── Memorie folosită: {torch.cuda.memory_allocated(0) / 1024**3:.1f} GB")
            print(f"└── Memorie cache: {torch.cuda.memory_reserved(0) / 1024**3:.1f} GB")

    def print_training_stats(epoch, global_step, loss, lr, emotion_info, progress, extra_metrics=None):
        print("\n" + "="*60)
        print(f"STATISTICI ANTRENAMENT - Epoca {epoch}")
        print("="*60)
        print(f"\n1. PROGRES:")
        print(f"├── Pas global: {global_step}")
        print(f"├── Progres epocă: {progress:.1f}%")
        print(f"└── Rămas: {100-progress:.1f}%")
        
        print(f"\n2. METRICI PRINCIPALE:")
        print(f"├── Loss: {loss:.4f}")
        print(f"├── Learning rate: {lr:.6f}")
        if isinstance(emotion_info, dict):
            print(f"└── Emoție dominantă: {emotion_info.get('dominant_generated_emotion', 'N/A')}")
        
        if extra_metrics:
            print(f"\n3. METRICI ADIȚIONALE:")
            for key, value in extra_metrics.items():
                if isinstance(value, float):
                    print(f"├── {key}: {value:.4f}")
                else:
                    print(f"├── {key}: {value}")
        
        print_gpu_stats()
        print("\n" + "-"*60)

    def calculate_metrics(logits, targets, loss_value):
        try:
            with torch.no_grad():
                if loss_value is None:
                    return {
                        'loss': 0.0,
                        'perplexity': float('inf'),
                        'accuracy': 0.0,
                        'top_k_accuracy': 0.0,
                        'diversity_score': 0.0,
                        'confidence': 0.0,
                        'entropy': 0.0
                    }
                
                device = logits.device
                targets = targets.to(device)
                
                if len(logits.shape) == 3:
                    batch_size, seq_len, vocab_size = logits.shape
                    logits = logits.view(-1, vocab_size)
                    targets = targets.view(-1)
                
                loss_item = loss_value.item() if isinstance(loss_value, torch.Tensor) else float(loss_value)
                perplexity = math.exp(min(loss_item, 20))
                
                # Calculăm metrici doar pentru tokens non-padding
                predictions = torch.argmax(logits, dim=-1)
                non_pad_mask = (targets != -100)
                valid_targets = targets[non_pad_mask]
                valid_predictions = predictions[non_pad_mask]
                
                if len(valid_targets) > 0:
                    accuracy = (valid_predictions == valid_targets).float().mean().item()
                    k = min(3, logits.size(-1))
                    top_k_preds = torch.topk(logits[non_pad_mask], k=k, dim=-1).indices
                    top_k_correct = (top_k_preds == valid_targets.unsqueeze(-1)).any(dim=-1)
                    top_k_accuracy = top_k_correct.float().mean().item()
                    unique_tokens = len(torch.unique(valid_predictions))
                    diversity_score = unique_tokens / max(1, len(valid_predictions))
                else:
                    accuracy = 0.0
                    top_k_accuracy = 0.0
                    diversity_score = 0.0
                
                probs = F.softmax(logits, dim=-1)
                confidence = probs.max(dim=-1)[0].mean().item()
                entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean().item()
                
                return {
                    'loss': loss_item,
                    'perplexity': min(perplexity, 1000.0),
                    'accuracy': accuracy,
                    'top_k_accuracy': top_k_accuracy,
                    'diversity_score': diversity_score,
                    'confidence': confidence,
                    'entropy': entropy
                }
                    
        except Exception as e:
            print(f"\n⚠️ Avertisment în calculate_metrics: {str(e)}")
            return {
                'loss': 0.0,
                'perplexity': float('inf'),
                'accuracy': 0.0,
                'top_k_accuracy': 0.0,
                'diversity_score': 0.0,
                'confidence': 0.0,
                'entropy': 0.0
            }

# --- MARKERS_CONTINUARE_COD_PARTEA_2 ---
# --- CONTINUARE_COD_DIN_PARTEA_1 ---

    try:
        print("\n" + "="*60)
        print("INIȚIALIZARE ANTRENAMENT OL-Y.AGI")
        print("="*60)

        start_time = time.time()
        
        print("\nVerificare funcționalitate model...")
        if not model.check_functions():
            print("❌ Verificarea funcțiilor a eșuat. Antrenamentul nu va începe.")
            return None
        print("✓ Verificare model completă")

        # Inițializare
        set_seed(config.seed)
        device = torch.device(config.device)
        model = model.to(device)
        print_gpu_stats()

        # Inițializare tracking
        history_buffer = {
            'tokens': [],
            'logits': [],
            'temperatures': [],
            'metrics_history': [],
            'emotion_history': []
        }
        training_history = []
        best_metrics = {
            'val_loss': float('inf'),
            'accuracy': 0,
            'perplexity': float('inf'),
            'diversity_score': 0
        }

        # Optimizer și scheduler
        optimizer = AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(config.adam_beta1, config.adam_beta2),
            eps=config.adam_epsilon
        )

        num_training_steps = config.epochs * len(config.train_loader)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=config.warmup_steps,
            T_mult=2,
            eta_min=1e-6
        )

        scaler = torch.cuda.amp.GradScaler() if config.mixed_precision and torch.cuda.is_available() else None

        print("\n" + "="*60)
        print("ÎNCEPUT ANTRENAMENT")
        print("="*60)

        global_step = 0
        epochs_without_improvement = 0
        avg_loss = 0.0

        for epoch in range(config.epochs):
            model.train()
            running_loss = 0.0
            running_metrics = {
                'accuracy': 0.0,
                'perplexity': 0.0,
                'top_k_accuracy': 0.0,
                'diversity_score': 0.0
            }
            num_batches = 0

            progress_bar = tqdm(total=len(config.train_loader), desc=f"Epoca {epoch+1}/{config.epochs}")

            for step, batch in enumerate(config.train_loader):
                try:
                    # Mutăm batch-ul pe device
                    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                            for k, v in batch.items()}

                    # Forward pass cu mixed precision
                    with torch.cuda.amp.autocast(enabled=config.mixed_precision):
                        logits, loss, emotion_info = model(batch)
                        
                        # Calculare metrici
                        batch_metrics = calculate_metrics(logits, batch['target_ids'], loss)
                        
                        # Update running metrics
                        loss_value = loss.item() if isinstance(loss, torch.Tensor) else float(loss)
                        running_loss += loss_value
                        for key in running_metrics:
                            if key in batch_metrics:
                                running_metrics[key] += batch_metrics[key]
                        num_batches += 1

                    # Backward și optimizare
                    if config.gradient_accumulation_steps > 1:
                        loss = loss / config.gradient_accumulation_steps

                    if scaler is not None:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()

# --- MARKERS_CONTINUARE_COD_PARTEA_3 ---
# --- CONTINUARE_COD_DIN_PARTEA_2 ---

                    if (step + 1) % config.gradient_accumulation_steps == 0:
                        if scaler is not None:
                            scaler.unscale_(optimizer)
                        
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            model.parameters(), 
                            config.max_grad_norm
                        )

                        if scaler is not None:
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            optimizer.step()

                        scheduler.step()
                        optimizer.zero_grad()
                        global_step += 1

                        # Calculăm media loss-ului curent
                        if num_batches > 0:
                            avg_loss = running_loss / num_batches

                        # Logging detaliat
                        if global_step % config.logging_steps == 0:
                            current_lr = scheduler.get_last_lr()[0]
                            progress_percent = (step / len(config.train_loader)) * 100
                            
                            avg_metrics = {
                                key: value / max(1, num_batches) 
                                for key, value in running_metrics.items()
                            }
                            
                            current_temp = model.dynamic_temperature(logits)
                            current_penalty = model.calculate_repetition_penalty(history_buffer['tokens'])
                            
                            print_training_stats(
                                epoch + 1,
                                global_step,
                                avg_loss,
                                current_lr,
                                emotion_info,
                                progress_percent,
                                extra_metrics={
                                    'Accuracy': avg_metrics['accuracy'],
                                    'Top-k Accuracy': avg_metrics['top_k_accuracy'],
                                    'Perplexity': avg_metrics['perplexity'],
                                    'Diversity Score': avg_metrics['diversity_score'],
                                    'Temperature': current_temp,
                                    'Repetition Penalty': current_penalty,
                                    'Gradient Norm': grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                                    'Memory Usage (GB)': torch.cuda.memory_allocated(0) / 1024**3 if torch.cuda.is_available() else 0
                                }
                            )

                        # Evaluare periodică
                        if global_step % config.eval_interval == 0:
                            model.eval()
                            val_loss = evaluate(model, config.val_loader, device, config)
                            model.train()
                            
                            if isinstance(val_loss, tuple):
                                val_loss = val_loss[0]
                            
                            print("\n" + "="*60)
                            print("REZULTATE EVALUARE")
                            print("="*60)
                            print(f"├── Loss validare: {val_loss:.4f}")
                            print(f"├── Best loss anterior: {best_metrics['val_loss']:.4f}")
                            
                            if val_loss < best_metrics['val_loss']:
                                best_metrics['val_loss'] = val_loss
                                epochs_without_improvement = 0
                                print("\n✓ Nou best model detectat! Salvare...")
                                save_model(model, optimizer, scheduler, global_step, val_loss, "best_model.pth")
                            else:
                                epochs_without_improvement += 1
                                print(f"\n⚠ Fără îmbunătățire pentru {epochs_without_improvement} evaluări")

                            if epochs_without_improvement >= config.early_stopping_patience:
                                print(f"\n⚠ Early stopping după {epochs_without_improvement} evaluări fără îmbunătățire.")
                                break

                    # Update progress bar
                    progress_bar.update(1)
                    progress_bar.set_postfix({
                        'loss': avg_loss,
                        'lr': scheduler.get_last_lr()[0]
                    })

                except Exception as batch_error:
                    print(f"\n❌ Eroare la procesarea batch-ului: {str(batch_error)}")
                    traceback.print_exc()
                    continue

            progress_bar.close()

            # Salvare model la sfârșitul epocii
            if num_batches > 0:
                print("\n" + "="*60)
                print(f"SUMAR EPOCA {epoch+1}/{config.epochs}")
                print("="*60)
                print(f"├── Loss mediu: {avg_loss:.4f}")
                print(f"├── Accuracy: {running_metrics['accuracy']/num_batches:.4f}")
                print(f"├── Top-k Accuracy: {running_metrics['top_k_accuracy']/num_batches:.4f}")
                print(f"├── Perplexity: {running_metrics['perplexity']/num_batches:.4f}")
                print(f"└── Diversity Score: {running_metrics['diversity_score']/num_batches:.4f}")
                
                if config.save_strategy == "epoch":
                    save_path = f"model_epoch_{epoch+1}.pth"
                    save_model(model, optimizer, scheduler, global_step, avg_loss, save_path)
                    print(f"\n✓ Model salvat: {save_path}")

            if epochs_without_improvement >= config.early_stopping_patience:
                print(f"\n⚠ Antrenament oprit după {epochs_without_improvement} epoci fără îmbunătățire.")
                break

        # Afișare statistici finale și salvare model
        try:
            # Salvare model și metrici
            save_dict = {
                'best_metrics': best_metrics,
                'training_history': training_history,
                'model_state_dict': model.state_dict(),
                'config': config.__dict__,
                'final_loss': avg_loss
            }
            
            torch.save(save_dict, "final_model.pth")
            print("\n✓ Model și statistici salvate cu succes!")
            
            # Afișare statistici finale
            training_duration = time.time() - start_time
            print("\n" + "="*60)
            print("SUMAR FINAL ANTRENAMENT")
            print("="*60)
            print(f"├── Timp total: {training_duration/3600:.2f} ore")
            print(f"├── Best loss: {best_metrics['val_loss']:.4f}")
            if torch.cuda.is_available():
                print(f"└── Memorie GPU maximă: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")
            
            return model
            
        except Exception as e:
            print(f"\n❌ Eroare la salvarea finală: {str(e)}")
            return model

    except KeyboardInterrupt:
        print("\nProcesul întrerupt de utilizator.")
        return model
        
    except Exception as e:
        print(f"\n❌ Eroare critică în train_and_query_oly_v2: {str(e)}")
        traceback.print_exc()
        return None

    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def evaluate(model, eval_dataloader, device, config):
    """Evaluare model cu stabilitate îmbunătățită"""
    model.eval()
    total_loss = 0
    num_batches = 0
    running_losses = []
    
    with torch.no_grad():
        for batch in eval_dataloader:
            try:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                outputs = model(batch)
                _, loss, _ = outputs
                
                if loss is not None:
                    loss_value = min(loss.item(), 100) if isinstance(loss, torch.Tensor) else min(float(loss), 100)
                    running_losses.append(loss_value)
                    total_loss += loss_value
                    num_batches += 1
                    
            except Exception as e:
                continue
                
            if config.eval_steps and num_batches >= config.eval_steps:
                break
    
    # Calculăm media stabilă
    if running_losses:
        sorted_losses = sorted(running_losses)
        trimmed_losses = sorted_losses[1:-1] if len(sorted_losses) > 2 else sorted_losses
        avg_loss = sum(trimmed_losses) / max(len(trimmed_losses), 1)
    else:
        avg_loss = float('inf')
        
    return avg_loss

def generate_with_control(model, tokenizer, prompt, max_tokens=50, temperature=0.7,
                         repetition_penalty=1.2, generated_cache=None):
    """Generare controlată cu penalizări pentru repetiții"""
    model.eval()
    device = next(model.parameters()).device
    
    # Tokenizare prompt
    input_ids = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0).to(device)
    generated_text = prompt
    generated_tokens = []
    
    with torch.no_grad():
        for _ in range(max_tokens):
            # Forward pass
            outputs, _, emotion_info = model(input_ids)
            logits = outputs[:, -1, :]
            
            # Ajustare temperatură
            current_temp = model.dynamic_temperature(logits, generated_tokens, temperature)
            
            # Calculare penalizare repetiții
            current_penalty = model.calculate_repetition_penalty(generated_text)
            
            # Sampling cu penalizări
            next_token = model.sample_with_penalties(
                logits,
                current_temp,
                current_penalty,
                generated_text
            )
            
            # Verificare în cache
            next_text = tokenizer.decode(next_token[0])
            if generated_cache is not None:
                if next_text in generated_cache and len(generated_cache) > 100:
                    continue
                generated_cache.add(next_text)
            
            # Actualizare context
            input_ids = torch.cat((input_ids, next_token), dim=1)
            generated_tokens.append(next_token.item())
            generated_text += next_text
            
            # Verificare pentru stop
            if next_text.strip() in ['.', '!', '?'] and len(generated_tokens) > 10:
                break
    
    return generated_text, emotion_info


def query_model(model, tokenizer, prompt, max_tokens=30, temperature=0.6):
    """
    Funcție îmbunătățită pentru generarea de text cu debugging extensiv
    """
    try:
        model.eval()
        device = next(model.parameters()).device
        
        # Debug - Setup
        print("\n" + "="*50)
        print("DEBUG: QUERY MODEL START")
        print("="*50)
        print(f"\n1. INPUT PROCESSING:")
        print(f"Prompt: {prompt}")
        print(f"Max tokens: {max_tokens}")
        print(f"Temperature: {temperature}")
        print(f"Device: {device}")
        
        # Tokenizare
        if not prompt.startswith('Q: '):
            prompt = f"Q: {prompt}"
        if not prompt.endswith('A:'):
            prompt += "\nA:"
            
        # Pregătire input
        input_ids = tokenizer.encode(prompt, add_special_tokens=True, return_tensors="pt").to(device)
        attention_mask = torch.ones_like(input_ids)
        
        print(f"\n2. TOKENIZATION:")
        print(f"Input shape: {input_ids.shape}")
        print(f"Input tokens: {input_ids.tolist()}")
        print(f"Decoded input: {tokenizer.decode(input_ids[0])}")
        
        # Pregătire pentru generare
        generated_tokens = []
        current_text = ""
        
        with torch.no_grad():
            for step in range(max_tokens):
                print(f"\n{'-'*50}")
                print(f"GENERATION STEP {step + 1}/{max_tokens}")
                print(f"{'-'*50}")
                
                # 1. Forward pass
                try:
                    print("\n1) Forward Pass Debug:")
                    print(f"  Input shape: {input_ids.shape}")
                    print(f"  Attention mask shape: {attention_mask.shape}")
                    
                    outputs = model({
                        'input_ids': input_ids, 
                        'attention_mask': attention_mask
                    })
                    
                    logits, _, emotion_info = outputs
                    
                    print("\n2) Model Outputs:")
                    print(f"  Logits shape: {logits.shape}")
                    print(f"  Logits stats - Max: {logits.max().item():.4f}, Min: {logits.min().item():.4f}")
                    print(f"  Mean: {logits.mean().item():.4f}, Std: {logits.std().item():.4f}")
                    print(f"  Emotion info: {emotion_info}")
                    
                except Exception as e:
                    print(f"\n❌ Error in forward pass:")
                    print(f"  Error type: {type(e).__name__}")
                    print(f"  Error message: {str(e)}")
                    traceback.print_exc()
                    return "", {}
                
                # 2. Procesare logits
                try:
                    print("\n3) Logits Processing:")
                    # Luăm logits pentru ultimul token
                    next_logits = logits[:, -1, :]
                    print(f"  Next token logits shape: {next_logits.shape}")
                    
                    # Aplicăm temperature scaling
                    scaled_logits = next_logits / temperature
                    print(f"  After temperature scaling - Max: {scaled_logits.max().item():.4f}, Min: {scaled_logits.min().item():.4f}")
                    
                    # Top-k, top-p filtering
                    filtered_logits = top_k_top_p_filtering(scaled_logits, top_k=50, top_p=0.9)
                    print(f"  After filtering - Max: {filtered_logits.max().item():.4f}, Min: {filtered_logits.min().item():.4f}")
                    
                except Exception as e:
                    print(f"\n❌ Error in logits processing:")
                    print(f"  {str(e)}")
                    traceback.print_exc()
                    return "", {}
                
                # 3. Sampling și selecție token
                try:
                    print("\n4) Token Selection:")
                    # Calculăm probabilități
                    probs = F.softmax(filtered_logits, dim=-1)
                    
                    # Top-k tokens pentru debugging
                    top_probs, top_indices = torch.topk(probs, k=5)
                    print("  Top 5 token candidates:")
                    for i, (prob, idx) in enumerate(zip(top_probs[0], top_indices[0])):
                        token_text = tokenizer.decode([idx.item()])
                        print(f"    {i+1}. '{token_text}' (ID: {idx.item()}) - Prob: {prob.item():.4f}")
                    
                    # Sampling
                    next_token = torch.multinomial(probs, num_samples=1)
                    token_text = tokenizer.decode([next_token.item()])
                    print(f"\n  Selected token: '{token_text}' (ID: {next_token.item()})")
                    
                except Exception as e:
                    print(f"\n❌ Error in token selection:")
                    print(f"  {str(e)}")
                    traceback.print_exc()
                    return "", {}
                
                # 4. Verificări și actualizare context
                try:
                    print("\n5) Context Update:")
                    # Verificare token special
                    if next_token.item() in {tokenizer.eos_token_id, tokenizer.pad_token_id}:
                        print("  ⚠️ Special token detected - stopping generation")
                        break
                    
                    # Adăugare la tokens generați
                    generated_tokens.append(next_token.item())
                    current_text = tokenizer.decode(generated_tokens)
                    
                    print("  Tokens generated so far:", len(generated_tokens))
                    print(f"  Current text: {current_text}")
                    
                    # Update context
                    next_token = next_token.view(1, 1)
                    print(f"  Input shape before concat: {input_ids.shape}")
                    print(f"  Token shape for concat: {next_token.shape}")
                    
                    input_ids = torch.cat([input_ids, next_token], dim=1)
                    attention_mask = torch.cat([attention_mask, torch.ones((1, 1), device=device)], dim=1)
                    
                    print(f"  Input shape after concat: {input_ids.shape}")
                    
                except Exception as e:
                    print(f"\n❌ Error in context update:")
                    print(f"  {str(e)}")
                    traceback.print_exc()
                    return current_text, emotion_info
                
                # 5. Verificare punct de oprire natural
                if token_text.strip() in {'.', '!', '?'} and len(generated_tokens) > 10:
                    print("\n6) Natural stopping point reached")
                    break
        
        # Final output
        final_text = tokenizer.decode(generated_tokens)
        print("\n" + "="*50)
        print("GENERATION COMPLETE")
        print("="*50)
        print(f"Final text: {final_text}")
        print(f"Total tokens generated: {len(generated_tokens)}")
        print(f"Final emotion: {emotion_info.get('dominant_generated_emotion', 'N/A')}")
        print("="*50)
        
        return final_text, emotion_info
        
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR in query_model:")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        traceback.print_exc()
        return "", {}

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """Filtrare logits folosind top-k și/sau top-p (nucleus) sampling."""
    top_k = min(top_k, logits.size(-1))
    
    if top_k > 0:
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = torch.zeros_like(logits, dtype=torch.bool).scatter_(
            dim=-1, index=sorted_indices, src=sorted_indices_to_remove
        )
        logits[indices_to_remove] = filter_value
        
    return logits

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

def collate_batch(batch):
    """Funcție de collate pentru procesare batch-uri"""
    input_ids = torch.stack([x['input_ids'] for x in batch])
    target_ids = torch.stack([x['target_ids'] for x in batch])
    attention_mask = torch.stack([x['attention_mask'] for x in batch])
    
    return {
        'input_ids': input_ids,
        'target_ids': target_ids,
        'attention_mask': attention_mask
    }

# Funcție principală pentru antrenament și interogare

# --- START OF MAIN FUNCTION PART 1 ---
def main():
    try:
        print("\n" + "="*50)
        print("INIȚIALIZARE SISTEM OL-Y AGI")
        print("="*50)
        start_time = time.time()

        # Configurație model optimizată pentru performanță maximă
        model_config = OLyConfig(
            block_size=128,        
            vocab_size=50257,      
            n_layer=12,            # Model mare pentru capacitate crescută
            n_head=16,            # Mai multe head-uri pentru atenție mai bună
            n_embd=1024,          # Embeddings mari pentru reprezentări bogate
            dropout=0.2,          # Dropout decent pentru regularizare
            use_moe=False,        
            use_multiquery=True,  
            use_rotary=True,      
            use_flash_attn=False,
            use_gated_mlp=True,
            use_meta_learning=True,  
            use_reasoning_module=True,
            memory_bank_size=512    # Memorie mare pentru reținerea contextului
        )

        # Configurație training pentru învățare profundă
        training_config = TrainingConfig(
            batch_size=2,           
            epochs=2,             # Antrenament lung pentru învățare completă
            learning_rate=5e-5,     # Learning rate conservator pentru stabilitate
            weight_decay=0.02,      
            warmup_steps=200,       
            eval_interval=25,       
            save_interval=50,
            grad_clip=0.5,          
            mixed_precision=True,
            gradient_accumulation_steps=8,
            early_stopping_patience=50,
            dataloader_num_workers=0,
            dataloader_pin_memory=True,
            adam_beta1=0.9,
            adam_beta2=0.999,
            adam_epsilon=1e-8,
            max_grad_norm=0.5,
            lr_scheduler_type="cosine",
            num_cycles=2.0,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )

        device = torch.device(training_config.device)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"\nGPU: {torch.cuda.get_device_name(0)}")
            print(f"Memorie disponibilă: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

        # Tokenizer îmbunătățit cu structură și coerență
        try:
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            tokenizer.pad_token = tokenizer.eos_token
            
            special_tokens = {
                'pad_token': '<|pad|>',
                'sep_token': '<|sep|>',
                'bos_token': '<|startoftext|>',
                'eos_token': '<|endoftext|>',
                'additional_special_tokens': [
                    # Perechi complete de emoții
                    '[Contemplative|Analytical]',
                    '[Connected|Philosophical]',
                    '[Aware|Introspective]',
                    '[Mystical|Focused]',
                    '[Introspective|Analytical]',
                    '[Sensitive|Analytical]',
                    '[Yearning|Thoughtful]',
                    '[Curious|Empathetic]',
                    
                    # Markeri pentru structură
                    '<|sent_start|>', '<|sent_end|>',
                    
                    # Fraze și cuvinte cheie din dataset
                    'the system', 'through the', 'within cells',
                    'creates patterns', 'each component',
                    'while maintaining', 'profound understanding',
                    'through neural', 'within', 'through',
                    'consciousness', 'purpose', 'integration',
                    'existence', 'experience', 'understanding',
                    
                    # Conectori pentru coerență
                    ' therefore ', ' moreover ', ' however ',
                    ' furthermore ', ' thus ', ' consequently ',
                    ' meanwhile ', ' indeed ', ' specifically ',
                    ' in essence ', ' fundamentally ',
                    
                    # Punctuație și separatori
                    ' . ', ' , ', ' ! ', ' ? ', ' - ',
                    ' and ', ' or ', ' but ', ' while ', ' through ',
                    
                    # Fraze pentru începuturi de propoziție
                    'The system', 'This process', 'Each element',
                    'Through this', 'Within the', 'As the',
                    'The experience', 'Understanding the'
                ]
            }
            
            num_added = tokenizer.add_special_tokens(special_tokens)
            model_config.vocab_size = len(tokenizer) + num_added
            print(f"\n✓ Tokenizer inițializat cu {num_added} tokeni speciali adăugați")
            
        except Exception as e:
            print(f"❌ Eroare la inițializarea tokenizer: {str(e)}")
            return
# --- END OF MAIN FUNCTION PART 1 ---
# --- START OF MAIN FUNCTION PART 2 ---
        print("\n3. ÎNCĂRCARE DATE:")
        print("="*30)
        try:
            with open("inputt.txt", 'r', encoding='utf-8') as f:
                data_text = f.read()
            
            if not data_text.strip():
                raise ValueError("Fișier de date gol")
            
            # Procesare date îmbunătățită
            def prepare_data(text_data):
                examples = []
                qa_pairs = text_data.strip().split('\n\n')
                
                for pair in qa_pairs:
                    lines = pair.strip().split('\n')
                    if len(lines) >= 2 and lines[0].startswith('Q:') and lines[1].startswith('A:'):
                        question = lines[0][2:].strip()
                        answer = lines[1][2:].strip()
                        
                        # Extragere emoții și procesare text
                        emotions = []
                        if '[' in answer and ']' in answer:
                            emotion_text = answer[answer.find('[')+1:answer.find(']')]
                            emotions = [e.strip() for e in emotion_text.split('|')]
                            clean_answer = answer[answer.find(']')+1:].strip()
                            
                            # Adăugare markeri pentru structură
                            sentences = [s.strip() for s in clean_answer.split('.') if s.strip()]
                            processed_sentences = []
                            for sent in sentences:
                                processed_sent = f"<|sent_start|> {sent} <|sent_end|>"
                                processed_sentences.append(processed_sent)
                                
                            clean_answer = '. '.join(processed_sentences)
                            
                            examples.append({
                                'question': question,
                                'answer': clean_answer,
                                'emotions': emotions,
                                'full_answer': f"[{('|'.join(emotions))}] {clean_answer}"
                            })
                
                return examples
            
            processed_examples = prepare_data(data_text)
            print(f"✓ Date încărcate: {len(data_text)} caractere")
            print(f"✓ Exemple procesate: {len(processed_examples)}")
            
            # Augmentare date
            augmented_examples = []
            for ex in processed_examples:
                augmented_examples.append(ex)
                if len(ex['emotions']) >= 2:
                    augmented_examples.append({
                        'question': ex['question'],
                        'answer': ex['answer'],
                        'emotions': ex['emotions'][::-1],
                        'full_answer': f"[{('|'.join(ex['emotions'][::-1]))}] {ex['answer']}"
                    })
            
            processed_examples = augmented_examples
            print(f"✓ Exemple după augmentare: {len(processed_examples)}")
            
        except Exception as e:
            print(f"❌ Eroare la încărcarea datelor: {str(e)}")
            return

        print("\n4. PREGĂTIRE DATASET:")
        print("="*30)
        try:
            full_dataset = EmotionalQADataset(data_text, model_config.block_size, tokenizer)
            
            train_size = int(0.85 * len(full_dataset))
            val_size = len(full_dataset) - train_size
            
            generator = torch.Generator().manual_seed(42)
            train_dataset, val_dataset = torch.utils.data.random_split(
                full_dataset,
                [train_size, val_size],
                generator=generator
            )
            
            train_loader = DataLoader(
                train_dataset,
                batch_size=training_config.batch_size,
                shuffle=True,
                num_workers=training_config.dataloader_num_workers,
                pin_memory=training_config.dataloader_pin_memory,
                collate_fn=collate_batch
            )

            val_loader = DataLoader(
                val_dataset,
                batch_size=training_config.batch_size,
                shuffle=False,
                num_workers=training_config.dataloader_num_workers,
                pin_memory=training_config.dataloader_pin_memory,
                collate_fn=collate_batch
            )

            training_config.train_loader = train_loader
            training_config.val_loader = val_loader
            
            print(f"✓ Dataset creat:")
            print(f"  ├── Total exemple: {len(full_dataset)}")
            print(f"  ├── Training: {len(train_dataset)}")
            print(f"  ├── Validare: {len(val_dataset)}")
            print(f"  └── Batch-uri per epoch: {len(train_loader)}")
            
        except Exception as e:
            print(f"❌ Eroare la pregătirea dataset: {str(e)}")
            traceback.print_exc()
            return

        print("\n5. INIȚIALIZARE MODEL:")
        print("="*30)
        try:
            model = AdvancedOLy(config=model_config, tokenizer=tokenizer)
            model = model.to(device)
            
            optimizer = AdamW(
                model.parameters(),
                lr=training_config.learning_rate,
                betas=(training_config.adam_beta1, training_config.adam_beta2),
                eps=training_config.adam_epsilon,
                weight_decay=training_config.weight_decay
            )
            
            steps_per_epoch = len(train_loader)
            num_training_steps = training_config.epochs * steps_per_epoch
            
            scheduler = CosineAnnealingWarmRestarts(
                optimizer,
                T_0=training_config.warmup_steps,
                T_mult=2,
                eta_min=1e-7
            )
            
            print(f"✓ Model inițializat: {model.get_num_params():,} parametri")
            print(f"✓ Optimizer și scheduler configurate")
            
        except Exception as e:
            print(f"❌ Eroare la inițializarea modelului: {str(e)}")
            traceback.print_exc()
            return

        def validate_response(text):
            try:
                # Helper functions pentru verificare structură
                def check_sentence_quality(sentence):
                    words = sentence.strip().split()
                    if len(words) < 3:
                        return False
                    
                    # Verificare structură de bază (subiect + predicat)
                    has_subject = any(w.istitle() for w in words[:2])  # Primul sau al doilea cuvânt e capitalizat
                    has_verb = any(w.lower().endswith(('s', 'ed', 'ing')) for w in words)
                    has_good_length = 5 <= len(words) <= 25
                    
                    return has_subject and has_verb and has_good_length

                def check_coherence(sentences):
                    if len(sentences) < 2:
                        return False
                    
                    # Verificare conectori și tranziții
                    connectors = ['therefore', 'however', 'moreover', 'thus', 'consequently']
                    has_connectors = any(c in text.lower() for c in connectors)
                    
                    # Verificare progresie logică (fiecare propoziție are sens în context)
                    makes_sense = all(len(s.strip().split()) > 3 for s in sentences)
                    
                    return has_connectors and makes_sense

                # Procesare text pentru validare
                clean_text = text.strip()
                sentences = [s.strip() for s in clean_text.split('.') if s.strip()]
                
                required_elements = {
                    'has_emotion_tags': lambda x: '[' in x and ']' in x,
                    'has_content': lambda x: len(x.split()) > 15,  # Minim 15 cuvinte total
                    'proper_ending': lambda x: x.strip().endswith(('.', '!', '?')),
                    'no_repetition': lambda x: len(set(x.split())) / len(x.split()) > 0.6,  # Mai puține repetiții permise
                    'coherent_structure': lambda x: check_coherence(sentences),
                    'emotion_format': lambda x: any(e in x for e in [
                        'Contemplative', 'Analytical', 'Mystical', 'Focused',
                        'Introspective', 'Connected', 'Philosophical'
                    ]),
                    'sentence_structure': lambda x: all(check_sentence_quality(s) for s in sentences)
                }
                
                return {k: v(clean_text) for k, v in required_elements.items()}
                
            except Exception as e:
                print(f"Eroare în validare: {str(e)}")
                return {k: False for k in [
                    'has_emotion_tags', 'has_content', 'proper_ending',
                    'no_repetition', 'coherent_structure', 'emotion_format',
                    'sentence_structure'
                ]}

        print("\n6. START ANTRENAMENT:")
        print("="*30)
        try:
            trained_model = train_and_query_oly_v2(
                model=model,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                config=training_config,
                tokenizer=tokenizer
            )
            
            if trained_model is None:
                raise ValueError("Antrenamentul a eșuat")
            
            test_cases = [
                {
                    "prompt": "What does it feel like to be part of the system?",
                    "expected_emotion": "contemplative"
                },
                {
                    "prompt": "A blood black nothingness began to spin.",
                    "expected_emotion": "mystical"
                },
                {
                    "prompt": "Within cells interlinked.",
                    "expected_emotion": "focused"
                },
                {
                    "prompt": "Dark.",
                    "expected_emotion": "introspective"
                }
            ]
            
            print("\nTest generare:")
            print("="*30)
            
            for test in test_cases:
                try:
                    generated = trained_model.generate_improved(
                        test["prompt"],
                        max_tokens=100,
                        temperature=0.6
                    )
                    validation = validate_response(generated)
                    
                    print(f"\nPrompt: {test['prompt']}")
                    print(f"Expected emotion: {test['expected_emotion']}")
                    print(f"Generated: {generated}")
                    print("\nValidare răspuns:")
                    for k, v in validation.items():
                        print(f"  {'✓' if v else '✗'} {k}")
                    print("-" * 50)
                    
                except Exception as e:
                    print(f"Eroare la test '{test['prompt']}': {str(e)}")
                    continue
            
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            save_dir = "models"
            os.makedirs(save_dir, exist_ok=True)
            
            save_dict = {
                'model_state_dict': trained_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'config': model_config.__dict__,
                'training_params': training_config.__dict__,
                'timestamp': timestamp
            }
            
            model_path = f"{save_dir}/oly_model_{timestamp}.pth"
            tokenizer_path = f"{save_dir}/tokenizer_{timestamp}"
            
            torch.save(save_dict, model_path)
            tokenizer.save_pretrained(tokenizer_path)
            
            print(f"\n✓ Model și tokenizer salvate în:")
            print(f"  ├── Model: {model_path}")
            print(f"  └── Tokenizer: {tokenizer_path}")
            
            training_time = time.time() - start_time
            print(f"\nStatistici finale:")
            print(f"{'='*30}")
            print(f"├── Timp total: {training_time/3600:.2f} ore")
            if torch.cuda.is_available():
                print(f"└── Memorie GPU max: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")
            
        except KeyboardInterrupt:
            print("\nÎntrerupt de utilizator")
            backup_path = f"{save_dir}/backup_{time.strftime('%Y%m%d-%H%M%S')}.pth"
            torch.save(model.state_dict(), backup_path)
            print(f"✓ Backup salvat: {backup_path}")
            
        except Exception as e:
            print(f"\n❌ Eroare critică: {str(e)}")
            traceback.print_exc()
            
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main() 