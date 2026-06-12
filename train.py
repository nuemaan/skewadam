#!/usr/bin/env python3
"""
Mixture-of-Experts (MoE) Training and Optimizer Ablation Framework.
Evaluates memory-efficient optimizers (SkewAdam, Adam, Lion, Muon, GaLore)
on a 6.7B sparse SwiGLU topology.
"""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import os
import random
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import GPT2TokenizerFast
from transformers.optimization import Adafactor
from torch.utils.checkpoint import checkpoint  

from skewadam import SkewAdam

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def clear_vram() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    gc.collect()

# -------------------------------------------------------------------------
# Baseline Optimizers
# -------------------------------------------------------------------------

class StochasticAdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                grad = p.grad
                if grad.dtype != torch.float32: grad = grad.float()
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p, dtype=torch.float32)
                    state['exp_avg_sq'] = torch.zeros_like(p, dtype=torch.float32)
                state['step'] += 1
                
                if group['weight_decay'] != 0:
                    if p.dtype == torch.bfloat16:
                        p_fp32 = p.float()
                        p_fp32.mul_(1 - group['lr'] * group['weight_decay'])
                        p.copy_(p_fp32)
                    else:
                        p.mul_(1 - group['lr'] * group['weight_decay'])
                        
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] / bias_correction1
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                update = exp_avg / denom
                
                if p.dtype == torch.bfloat16:
                    p_fp32 = p.float()
                    p_fp32.add_(update, alpha=-step_size)
                    ulp = torch.abs(p_fp32) * 0.0078125
                    noise = (torch.rand_like(p_fp32) - 0.5) * ulp
                    p.copy_(p_fp32 + noise)
                else:
                    p.add_(update, alpha=-step_size)

class Lion(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)
        
    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            lr, beta1, beta2, wd = group["lr"], group["betas"][0], group["betas"][1], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None: continue
                grad = p.grad
                
                if grad.dtype != torch.float32: grad = grad.float()
                
                state = self.state[p]
                if len(state) == 0: state["exp_avg"] = torch.zeros_like(p, dtype=torch.float32)
                exp_avg = state["exp_avg"]
                
                if wd != 0: 
                    if p.dtype == torch.bfloat16:
                        p_fp32 = p.float()
                        p_fp32.mul_(1 - lr * wd)
                        p.copy_(p_fp32)
                    else:
                        p.mul_(1 - lr * wd)
                        
                update = exp_avg.mul(beta1).add(grad, alpha=1 - beta1)
                update_sign = torch.sign(update)
                
                if p.dtype == torch.bfloat16:
                    p_fp32 = p.float()
                    p_fp32.add_(update_sign, alpha=-lr)
                    ulp = torch.abs(p_fp32) * 0.0078125
                    noise = (torch.rand_like(p_fp32) - 0.5) * ulp
                    p.copy_(p_fp32 + noise)
                else:
                    p.add_(update_sign, alpha=-lr)
                    
                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)

def _newton_schulz(G, steps=3, eps=1e-7):
    """Memory-efficient Newton-Schulz orthogonalization using bfloat16 primitives."""
    a, b, c = 3.4445, -4.7750,  2.0315
    X = G.bfloat16()
    X = X / (X.norm() + eps)
    
    transpose = False
    if X.size(0) > X.size(1):
        X = X.T
        transpose = True
        
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
        
    if transpose:
        X = X.T
        
    return X.float()

class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, adam_lr=1e-3):
        defaults = dict(lr=lr, momentum=momentum, adam_lr=adam_lr, initial_muon_lr=lr)
        super().__init__(params, defaults)
        
    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            sched_ratio = group['lr'] / max(group.get('initial_muon_lr', 0.02), 1e-9)
            current_adam_lr = group['adam_lr'] * sched_ratio
            
            for p in group['params']:
                if p.grad is None: continue
                grad = p.grad
                
                if grad.dtype != torch.float32: grad = grad.float()
                
                state = self.state[p]
                is_1d = grad.ndim < 2
                is_vocab = (not is_1d) and (p.shape[0] > 20000 or p.shape[1] > 20000)
                is_router = (not is_1d) and (p.shape[0] == 128 and p.shape[1] == 4096)
                is_expert = (not is_1d) and (p.shape[0] == 2048 or p.shape[1] == 2048)
                is_pos_emb = (not is_1d) and (p.shape[0] == 256 and p.shape[1] == 4096)
                
                if is_1d or is_vocab or is_router or is_expert or is_pos_emb:
                    if len(state) == 0:
                        state['step'] = 0
                        state['m'] = torch.zeros_like(p, dtype=torch.float32)
                        state['v'] = torch.zeros_like(p, dtype=torch.float32)
                    state['step'] += 1
                    state['m'].mul_(0.9).add_(grad, alpha=0.1)
                    state['v'].mul_(0.999).addcmul_(grad, grad, value=0.001)
                    
                    bias_correction1 = 1 - 0.9 ** state['step']
                    bias_correction2 = 1 - 0.999 ** state['step']
                    m_hat = state['m'] / bias_correction1
                    v_hat = state['v'] / bias_correction2
                    
                    update = m_hat / (v_hat.sqrt() + 1e-8)
                    
                    if p.dtype == torch.bfloat16:
                        p_fp32 = p.float()
                        p_fp32.add_(update, alpha=-current_adam_lr)
                        ulp = torch.abs(p_fp32) * 0.0078125
                        noise = (torch.rand_like(p_fp32) - 0.5) * ulp
                        p.copy_(p_fp32 + noise)
                    else:
                        p.add_(update, alpha=-current_adam_lr)
                    continue
                
                if 'momentum_buffer' not in state: state['momentum_buffer'] = torch.zeros_like(grad, dtype=torch.float32)
                buf = state['momentum_buffer']
                buf.mul_(group['momentum']).add_(grad)
                
                X_update = _newton_schulz(buf, steps=3, eps=1e-7)
                
                if p.dtype == torch.bfloat16:
                    p_fp32 = p.float()
                    p_fp32.add_(X_update, alpha=-group['lr'])
                    ulp = torch.abs(p_fp32) * 0.0078125
                    noise = (torch.rand_like(p_fp32) - 0.5) * ulp
                    p.copy_(p_fp32 + noise)
                else:
                    p.add_(X_update, alpha=-group['lr'])

class GaLoreAdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, rank=128):
        defaults = dict(lr=lr, rank=rank)
        super().__init__(params, defaults)
        
    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                grad = p.grad
                
                if grad.dtype != torch.float32: grad = grad.float()
                
                if grad.ndim == 3 or grad.ndim < 2:
                    state = self.state[p]
                    if len(state) == 0:
                        state['m'] = torch.zeros_like(p, dtype=torch.float32)
                        state['v'] = torch.zeros_like(p, dtype=torch.float32)
                    state['m'].mul_(0.9).add_(grad, alpha=0.1)
                    state['v'].mul_(0.999).addcmul_(grad, grad, value=0.001)
                    update = state['m'] / (state['v'].sqrt() + 1e-8)
                    
                    if p.dtype == torch.bfloat16:
                        p_fp32 = p.float()
                        p_fp32.add_(update, alpha=-group['lr'])
                        ulp = torch.abs(p_fp32) * 0.0078125
                        noise = (torch.rand_like(p_fp32) - 0.5) * ulp
                        p.copy_(p_fp32 + noise)
                    else:
                        p.add_(update, alpha=-group['lr'])
                    continue
                    
                state = self.state[p]
                if 'U' not in state:
                    U, _, V = torch.svd_lowrank(grad, q=group['rank'])
                    state['U'], state['V'] = U, V.T
                    state['m'] = torch.zeros((group['rank'], group['rank']), device=p.device, dtype=torch.float32)
                    state['v'] = torch.zeros((group['rank'], group['rank']), device=p.device, dtype=torch.float32)
                    
                low_rank_grad = state['U'].T @ grad @ state['V'].T
                state['m'].mul_(0.9).add_(low_rank_grad, alpha=0.1)
                state['v'].mul_(0.999).addcmul_(low_rank_grad, low_rank_grad, value=0.001)
                update_lr = state['m'] / (state['v'].sqrt() + 1e-8)
                update = (state['U'] @ update_lr @ state['V']).view(grad.shape)
                
                if p.dtype == torch.bfloat16:
                    p_fp32 = p.float()
                    p_fp32.add_(update, alpha=-group['lr'])
                    ulp = torch.abs(p_fp32) * 0.0078125
                    noise = (torch.rand_like(p_fp32) - 0.5) * ulp
                    p.copy_(p_fp32 + noise)
                else:
                    p.add_(update, alpha=-group['lr'])

# -------------------------------------------------------------------------
# Configuration and Data Loaders
# -------------------------------------------------------------------------

@dataclass
class Config:
    seed: int = 42
    optimizers: Tuple[str, ...] = ("skewadam", "adam", "lion", "galore", "muon") 
    steps: int = 10000
    eval_interval: int = 1000
    batch_size: int = 64          
    micro_batch_size: int = 8      
    seq_len: int = 128
    train_batches: int = 11000
    val_batches: int = 64
    vocab_dim: int = 4096
    num_layers: int = 2
    num_heads: int = 32
    n_kv_heads: int = 8           
    ff_dim: int = 4096
    num_experts: int = 128
    expert_ff_dim: int = 2048
    top_k: int = 2
    max_positions: int = 256
    dropout: float = 0.0          
    load_balancing_coef: float = 0.05
    lr_adam: float = 3e-4         
    lr_adafactor: float = 3e-4
    lr_skewadam: float = 3e-4     
    lr_lion: float = 1e-4         
    lr_muon: float = 0.02
    lr_galore: float = 3e-4       
    weight_decay: float = 0.05
    grad_clip: float = 1.0
    warmup_ratio: float = 0.03
    dataset_name: str = "openwebtext"
    train_ratio: float = 0.95
    out_dir: str = "runs"

def get_lr(cfg: Config, optimizer_name: str) -> float:
    opt = optimizer_name.lower()
    if opt == "adam": return cfg.lr_adam
    if opt == "adafactor": return cfg.lr_adafactor
    if opt == "skewadam": return cfg.lr_skewadam
    if opt == "lion": return cfg.lr_lion
    if opt == "muon": return cfg.lr_muon
    if opt == "galore": return cfg.lr_galore
    raise ValueError(f"Unknown optimizer: {optimizer_name}")

def stable_split(text: str, train_ratio: float) -> str:
    h = hashlib.md5(text[:1024].encode("utf-8", errors="ignore")).hexdigest()
    bucket = int(h[:8], 16) % 1000
    return "train" if bucket < int(train_ratio * 1000) else "val"

def build_batch_caches(
    tokenizer, dataset_name: str, batch_size: int, seq_len: int,
    train_target_batches: int, val_target_batches: int, train_ratio: float,
) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor]], List[Tuple[torch.Tensor, torch.Tensor]]]:
    ds = load_dataset(dataset_name, split="train", streaming=True)
    train_batches, val_batches = [], []
    train_buffer, val_buffer = [], []
    chunk_size = batch_size * seq_len + 1

    def flush(buffer: List[int], out: List[Tuple[torch.Tensor, torch.Tensor]], target: int) -> None:
        while len(buffer) >= chunk_size and len(out) < target:
            chunk = buffer[:chunk_size]
            del buffer[: batch_size * seq_len]
            chunk_tensor = torch.tensor(chunk, dtype=torch.long)
            out.append((chunk_tensor[:-1].view(batch_size, seq_len).clone(), chunk_tensor[1:].view(batch_size, seq_len).clone()))

    for example in ds:
        text = example.get("text", "")
        if not text or not text.strip(): continue
        split = stable_split(text, train_ratio)
        ids = tokenizer.encode(text)
        if split == "train":
            train_buffer.extend(ids)
            flush(train_buffer, train_batches, train_target_batches)
        else:
            val_buffer.extend(ids)
            flush(val_buffer, val_batches, val_target_batches)
        if len(train_batches) >= train_target_batches and len(val_batches) >= val_target_batches:
            break
    return train_batches, val_batches

# -------------------------------------------------------------------------
# Model Architecture
# -------------------------------------------------------------------------

class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, dropout: float):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads
        self.n_rep = n_heads // n_kv_heads
        self.q_proj = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(n_heads * self.head_dim, d_model, bias=False)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        q = self.q_proj(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)
        k = torch.repeat_interleave(k, repeats=self.n_rep, dim=1)
        v = torch.repeat_interleave(v, repeats=self.n_rep, dim=1)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=True)
        return self.out_proj(out.transpose(1, 2).contiguous().view(b, t, -1))

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, ff_dim: int, dropout: float):
        super().__init__()
        self.w1 = nn.Linear(d_model, ff_dim, bias=False)
        self.w2 = nn.Linear(d_model, ff_dim, bias=False)
        self.w3 = nn.Linear(ff_dim, d_model, bias=False)
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        return self.drop(self.w3(F.silu(self.w1(x)) * self.w2(x)))

class MoEFFN(nn.Module):
    def __init__(self, d_model: int, expert_ff_dim: int, num_experts: int, top_k: int, dropout: float, load_balancing_coef: float):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        self.load_balancing_coef = load_balancing_coef
        self.router = nn.Linear(d_model, num_experts, bias=False)
        self.experts = nn.ModuleList([SwiGLU(d_model, expert_ff_dim, dropout) for _ in range(num_experts)])

    def forward(self, x: torch.Tensor):
        b, t, d = x.shape
        n = b * t
        x_flat = x.reshape(n, d)
        logits_fp32 = self.router(x_flat).float() 
        z_loss = 1e-4 * torch.logsumexp(logits_fp32, dim=-1).pow(2).mean()

        if self.training:
            logits_fp32 = logits_fp32 + torch.randn_like(logits_fp32) * 0.5 
            
        probs = torch.softmax(logits_fp32, dim=-1).to(x_flat.dtype) 
        topk_probs, topk_idx = torch.topk(probs, self.top_k, dim=-1)
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True).clamp_min(1e-9)

        expert_prob_mean = probs.mean(dim=0).float()
        tokens_per_expert = torch.bincount(topk_idx.reshape(-1), minlength=self.num_experts).float()
        tokens_per_expert = tokens_per_expert / tokens_per_expert.sum().clamp_min(1.0)
        aux_loss = self.load_balancing_coef * self.num_experts * torch.sum(expert_prob_mean * tokens_per_expert)

        out_flat = torch.zeros_like(x_flat)
        token_ids = torch.arange(n, device=x.device).repeat_interleave(self.top_k)
        flat_probs = topk_probs.reshape(-1, 1)
        flat_idx = topk_idx.reshape(-1)

        for e in range(self.num_experts):
            sel = flat_idx == e
            if not torch.any(sel): continue
            token_sel = token_ids[sel]
            x_sel = x_flat[token_sel]
            y_sel = self.experts[e](x_sel) * flat_probs[sel]
            out_flat.index_add_(0, token_sel, y_sel)

        return out_flat.view(b, t, d), aux_loss + z_loss

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, ff_dim: int, dropout: float, moe: bool = False,
                 num_experts: int = 0, top_k: int = 0, load_balancing_coef: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = GroupedQueryAttention(d_model, n_heads, n_kv_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        if moe:
            self.ff = MoEFFN(d_model, ff_dim, num_experts, top_k, dropout, load_balancing_coef)
            self.is_moe = True
        else:
            self.ff = SwiGLU(d_model, ff_dim, dropout)
            self.is_moe = False

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        ff_out = self.ff(self.norm2(x))
        if self.is_moe:
            ff_out, aux_loss = ff_out
            x = x + ff_out
            return x, aux_loss
        else:
            return x + ff_out, torch.tensor(0.0, device=x.device)

class CausalMoETransformerLM(nn.Module):
    def __init__(self, vocab_size: int, max_positions: int, d_model: int, n_heads: int, n_kv_heads: int, ff_dim: int, num_experts: int, expert_ff_dim: int, top_k: int, num_layers: int, dropout: float, load_balancing_coef: float):
        super().__init__()
        self.vocab_size = vocab_size
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_positions, d_model)
        self.drop = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=d_model, n_heads=n_heads, n_kv_heads=n_kv_heads, ff_dim=ff_dim, dropout=dropout, 
                moe=(i % 2 != 0), num_experts=num_experts, top_k=top_k, load_balancing_coef=load_balancing_coef
            ) for i in range(num_layers)
        ])
        self.norm_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('out_proj.weight') or pn.endswith('.w3.weight'): 
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * num_layers))
        for layer in self.layers:
            if layer.is_moe:
                torch.nn.init.zeros_(layer.ff.router.weight)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, x):
        b, t = x.shape
        pos = torch.arange(t, device=x.device).unsqueeze(0).expand(b, t)
        h = self.drop(self.tok_emb(x) + self.pos_emb(pos))
        total_aux_loss = 0.0
        for layer in self.layers:
            if self.training:
                # Apply activation checkpointing to bound memory usage
                h, aux_loss = checkpoint(layer, h, use_reentrant=False)
                total_aux_loss = total_aux_loss + aux_loss
            else:
                h, aux_loss = layer(h)
                total_aux_loss = total_aux_loss + aux_loss
        return self.lm_head(self.norm_f(h)), total_aux_loss

# -------------------------------------------------------------------------
# Training Utilities
# -------------------------------------------------------------------------

def build_optimizer(name: str, model: nn.Module, lr: float, wd: float):
    name = name.lower()
    expert_params, router_params, dense_params = [], [], []
    for n, p in model.named_parameters():
        if "experts" in n: expert_params.append(p)
        elif "router" in n: router_params.append(p)
        else: dense_params.append(p)
            
    universal_groups = [
        {'params': dense_params, 'weight_decay': wd},
        {'params': expert_params, 'weight_decay': wd},
        {'params': router_params, 'weight_decay': 0.0}
    ]

    if name == "adam": return StochasticAdamW(universal_groups, lr=lr)
    if name == "lion": return Lion(universal_groups, lr=lr)
    if name == "adafactor": return Adafactor(universal_groups, scale_parameter=False, relative_step=False, warmup_init=False, lr=lr)
    if name == "muon": return Muon(model.parameters(), lr=lr) 
    if name == "galore": return GaLoreAdamW(universal_groups, lr=lr)
    if name == "skewadam":
        return SkewAdam([
            {'params': dense_params, 'use_momentum': True, 'use_factored': True, 'weight_decay': wd},
            {'params': expert_params, 'use_momentum': False, 'use_factored': True, 'weight_decay': wd}, 
            {'params': router_params, 'use_momentum': False, 'use_factored': False, 'weight_decay': 0.0} 
        ], lr=lr)
    raise ValueError(f"Unknown optimizer: {name}")

def theoretical_state_bytes(model: nn.Module, optimizer_name: str) -> Dict[str, float]:
    """Calculates the physical memory requirement of the optimizer states based on structural components."""
    total_params = sum(p.numel() for p in model.parameters())
    actual_weights_gb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**3)
    result = {"total_params_m": total_params / 1e6, "weights_gb_actual": actual_weights_gb}
    opt = optimizer_name.lower()
    
    if opt == "adam": 
        result["optimizer_state_gb"] = total_params * 8 / (1024**3)
    elif opt in ("lion", "muon"): 
        result["optimizer_state_gb"] = (total_params * 4) / (1024**3) 
    elif opt == "skewadam":
        skew_bytes = 0
        for n, p in model.named_parameters():
            use_mom = "router" not in n and "experts" not in n
            use_fac = "router" not in n
            
            if use_mom:
                skew_bytes += p.numel() * 4  
                
            if p.dim() >= 2 and use_fac:
                if p.dim() == 2:
                    skew_bytes += (p.shape[0] + p.shape[1]) * 4
                elif p.dim() == 3:
                    skew_bytes += (p.shape[0]*p.shape[1] + p.shape[0]*p.shape[2]) * 4
            else:
                skew_bytes += p.numel() * 4
                
        result["optimizer_state_gb"] = skew_bytes / (1024**3)
    else: 
        result["optimizer_state_gb"] = float("nan")
        
    return result

@torch.no_grad()
def evaluate(model: nn.Module, criterion, val_batches, device):
    model.eval()
    total_loss, total_count = 0.0, 0
    for x, y in val_batches:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits, _ = model(x.to(device, non_blocking=True)) 
            loss = criterion(logits.reshape(-1, logits.size(-1)), y.to(device, non_blocking=True).reshape(-1))
        total_loss += float(loss.item())
        total_count += 1
    model.train()
    avg_loss = total_loss / max(total_count, 1)
    try: ppl = math.exp(avg_loss)
    except OverflowError: ppl = float("inf")
    return avg_loss, ppl

def cosine_lr(step: int, total_steps: int, base_lr: float, warmup_steps: int):
    if step <= warmup_steps: return base_lr * step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * base_lr * (1.0 + math.cos(math.pi * progress))

def train_one_optimizer(cfg: Config, optimizer_name: str, tokenizer, train_batches, val_batches, initial_state, device, vocab_size: int):
    print(f"\n{'=' * 80}\nOptimizer: {optimizer_name}\n{'=' * 80}")

    model = CausalMoETransformerLM(
        vocab_size=vocab_size, max_positions=cfg.max_positions, d_model=cfg.vocab_dim,
        n_heads=cfg.num_heads, n_kv_heads=cfg.n_kv_heads, ff_dim=cfg.ff_dim, num_experts=cfg.num_experts,
        expert_ff_dim=cfg.expert_ff_dim, top_k=cfg.top_k, num_layers=cfg.num_layers,
        dropout=cfg.dropout, load_balancing_coef=cfg.load_balancing_coef,
    ).to(device, dtype=torch.bfloat16)  
    
    for name, module in model.named_modules():
        if isinstance(module, nn.LayerNorm) or "router" in name:
            module.to(torch.float32)

    model.load_state_dict(initial_state, strict=False)
    model = torch.compile(model)

    opt_name = optimizer_name.lower()
    lr = get_lr(cfg, opt_name)
    optimizer = build_optimizer(opt_name, model, lr, cfg.weight_decay)
    crit = nn.CrossEntropyLoss()
    warmup_steps = max(1, int(cfg.steps * cfg.warmup_ratio))

    mem_info = theoretical_state_bytes(model, opt_name)
    print(f"BF16/Mixed Master Weights: {mem_info['weights_gb_actual']:.2f} GB | Optimizer state: {mem_info['optimizer_state_gb']:.2f} GB")

    best_val_ppl, best_step = float("inf"), -1
    best_path = os.path.join(cfg.out_dir, f"best_{opt_name}.pt")
    history, training_time = [], 0.0
    torch.cuda.reset_peak_memory_stats()

    accum_steps = max(1, cfg.batch_size // cfg.micro_batch_size)
    print(f"Gradient Accumulation: {accum_steps} micro-steps | Global Batch Size: {cfg.batch_size} sequences")

    for step in range(1, cfg.steps + 1):
        x_cpu, y_cpu = train_batches[(step - 1) % len(train_batches)]
        x, y = x_cpu.to(device, non_blocking=True), y_cpu.to(device, non_blocking=True)

        current_lr = cosine_lr(step, cfg.steps, lr, warmup_steps)
        if hasattr(optimizer, "param_groups"):
            for group in optimizer.param_groups: group["lr"] = current_lr

        start = time.time()
        optimizer.zero_grad(set_to_none=True)

        micro_x_chunks = x.chunk(accum_steps, dim=0)
        micro_y_chunks = y.chunk(accum_steps, dim=0)
        
        step_lm_loss, step_aux_loss, step_total_loss = 0.0, 0.0, 0.0

        for micro_x, micro_y in zip(micro_x_chunks, micro_y_chunks):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits, raw_aux_loss = model(micro_x)
                lm_loss = crit(logits.reshape(-1, vocab_size), micro_y.reshape(-1))
                loss = (lm_loss + raw_aux_loss) / accum_steps

            loss.backward()
            step_lm_loss += lm_loss.detach().item() / accum_steps
            step_aux_loss += raw_aux_loss.detach().item() / accum_steps
            step_total_loss += loss.detach().item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        if torch.cuda.is_available(): torch.cuda.synchronize()
        training_time += time.time() - start

        if step % cfg.eval_interval == 0 or step == cfg.steps:
            val_loss, val_ppl = evaluate(model, crit, val_batches, device)
            tok_sec = (step * cfg.batch_size * cfg.seq_len) / max(training_time, 1e-9)
            peak_vram = torch.cuda.max_memory_allocated() / (1024**3)

            record = {
                "step": step, "train_lm_loss": float(step_lm_loss), "aux_loss": float(step_aux_loss),
                "total_loss": float(step_total_loss), "val_loss": float(val_loss), "val_ppl": float(val_ppl),
                "tok_per_sec": float(tok_sec), "peak_vram_gb": float(peak_vram), "lr": float(current_lr),
            }
            history.append(record)

            print(f"Step {step:05d} | Train LM Loss {record['train_lm_loss']:.4f} | Raw Aux {record['aux_loss']:.4f} | "
                  f"Val PPL {record['val_ppl']:.2f} | Speed {record['tok_per_sec']:.0f} tok/s | Peak VRAM {record['peak_vram_gb']:.2f} GB")

            if val_ppl < best_val_ppl:
                best_val_ppl, best_step = val_ppl, step
                model_to_save = model._orig_mod if hasattr(model, "_orig_mod") else model
                torch.save({"model_state_dict": model_to_save.state_dict(), "optimizer_name": opt_name, "optimizer_lr": lr,
                            "config": asdict(cfg), "vocab_size": vocab_size, "best_val_ppl": best_val_ppl, "best_step": best_step}, best_path)

    summary = {
        "optimizer": opt_name, "lr": lr, "best_step": best_step, "best_val_ppl": best_val_ppl,
        "final_val_ppl": history[-1]["val_ppl"] if history else None, "final_peak_vram_gb": torch.cuda.max_memory_allocated() / (1024**3),
        "final_tok_per_sec": history[-1]["tok_per_sec"] if history else None, "best_checkpoint": best_path, "history": history, "theory": mem_info,
    }

    os.makedirs(cfg.out_dir, exist_ok=True)
    with open(os.path.join(cfg.out_dir, f"metrics_{opt_name}.json"), "w") as f: json.dump(summary, f, indent=2)
    del model, optimizer; clear_vram()
    return summary

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--eval-interval", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--train-batches", type=int, default=11000)
    p.add_argument("--val-batches", type=int, default=64)
    p.add_argument("--vocab-dim", type=int, default=4096)
    p.add_argument("--num-heads", type=int, default=32)
    p.add_argument("--n-kv-heads", type=int, default=8)
    p.add_argument("--ff-dim", type=int, default=4096)
    p.add_argument("--num-experts", type=int, default=128)
    p.add_argument("--expert-ff-dim", type=int, default=2048)
    p.add_argument("--top-k", type=int, default=2)
    p.add_argument("--max-positions", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--load-balancing-coef", type=float, default=0.05)
    p.add_argument("--lr-adam", type=float, default=3e-4)
    p.add_argument("--lr-adafactor", type=float, default=3e-4)
    p.add_argument("--lr-skewadam", type=float, default=3e-4) 
    p.add_argument("--lr-lion", type=float, default=1e-4)
    p.add_argument("--lr-muon", type=float, default=0.02)
    p.add_argument("--lr-galore", type=float, default=3e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--dataset-name", type=str, default="openwebtext")
    p.add_argument("--train-ratio", type=float, default=0.95)
    p.add_argument("--optimizers", type=str, default="skewadam,adam,lion,galore,muon")
    p.add_argument("--out-dir", type=str, default="runs")
    return p.parse_args()

def main():
    args = parse_args()
    cfg = Config(**{k: v for k, v in vars(args).items() if k in Config.__annotations__})
    cfg.optimizers = tuple([x.strip().lower() for x in args.optimizers.split(",") if x.strip()])

    os.makedirs(cfg.out_dir, exist_ok=True)
    seed_everything(cfg.seed)
    if torch.cuda.is_available(): torch.set_float32_matmul_precision("high")

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2", model_max_length=1000000)
    base_vocab_size = tokenizer.vocab_size
    padded_vocab_size = math.ceil(base_vocab_size / 64) * 64

    train_batches, val_batches = build_batch_caches(
        tokenizer=tokenizer, dataset_name=cfg.dataset_name, batch_size=cfg.batch_size,
        seq_len=cfg.seq_len, train_target_batches=cfg.train_batches, val_target_batches=cfg.val_batches, train_ratio=cfg.train_ratio,
    )

    cpu_model = CausalMoETransformerLM(
        vocab_size=padded_vocab_size, max_positions=cfg.max_positions, d_model=cfg.vocab_dim,
        n_heads=cfg.num_heads, n_kv_heads=cfg.n_kv_heads, ff_dim=cfg.ff_dim, num_experts=cfg.num_experts, expert_ff_dim=cfg.expert_ff_dim,
        top_k=cfg.top_k, num_layers=cfg.num_layers, dropout=cfg.dropout, load_balancing_coef=cfg.load_balancing_coef,
    )
    
    initial_state = copy.deepcopy(cpu_model.state_dict())
    del cpu_model; clear_vram()

    device = torch.device("cuda")
    summaries = []

    for opt_name in cfg.optimizers:
        seed_everything(cfg.seed)
        summary = train_one_optimizer(
            cfg=cfg, optimizer_name=opt_name, tokenizer=tokenizer, train_batches=train_batches,
            val_batches=val_batches, initial_state=initial_state, device=device, vocab_size=padded_vocab_size,
        )
        summaries.append(summary)

if __name__ == "__main__":
    main()