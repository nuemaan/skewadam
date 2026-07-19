"""Corrected continuation of the LR sweep.

The first pass swept Adafactor upward from 3e-4 (1e-3, 3e-3), where it diverges
(496, 623) --- so its best remained 3e-4 = 149.5 and the lower direction was
never tested. AdamW bottomed out at 117.9 (1e-4). This pass spends the
remaining budget on what actually carries signal: Adafactor at lower LRs (does
a smaller step rescue it, or is 3e-4 already its peak?) plus variance on the
best configs. Configs are grouped by seed so only one initialization is held
in memory at a time; seed-matched inits reproduce the first pass exactly.

Usage (from the repository root, on the same GPU):
    python experiments/lr-sweep/run_fix.py --dataset-name Skylion007/openwebtext
"""
import argparse
import copy
import json
import math
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import torch
from transformers import GPT2TokenizerFast

import train as T

# (optimizer, lr, seed) --- grouped by seed, priority-ordered within reason
CONFIGS = [
    ("adafactor", 1e-4, 42),   # the key untested Adafactor direction
    ("adafactor", 3e-5, 42),   # bracket lower still
    ("adam", 1e-4, 43),        # AdamW best-config variance (re-includes the killed run)
    ("adafactor", 1e-4, 43),   # Adafactor best-config variance
    ("adam", 1e-4, 44),        # AdamW 3rd seed
]

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-name", type=str, default="Skylion007/openwebtext")
    p.add_argument("--steps", type=int, default=10000)
    args = p.parse_args()

    os.makedirs(OUT, exist_ok=True)
    cfg = T.Config(dataset_name=args.dataset_name, steps=args.steps, out_dir="runs")
    torch.set_float32_matmul_precision("high")

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2", model_max_length=1000000)
    padded_vocab = math.ceil(tokenizer.vocab_size / 64) * 64

    train_batches, val_batches = T.build_batch_caches(
        tokenizer=tokenizer, dataset_name=cfg.dataset_name, batch_size=cfg.batch_size,
        seq_len=cfg.seq_len, train_target_batches=cfg.train_batches,
        val_target_batches=cfg.val_batches, train_ratio=cfg.train_ratio,
    )
    device = torch.device("cuda")

    def build_init(seed):
        T.seed_everything(seed)
        m = T.CausalMoETransformerLM(
            vocab_size=padded_vocab, max_positions=cfg.max_positions, d_model=cfg.vocab_dim,
            n_heads=cfg.num_heads, n_kv_heads=cfg.n_kv_heads, ff_dim=cfg.ff_dim,
            num_experts=cfg.num_experts, expert_ff_dim=cfg.expert_ff_dim, top_k=cfg.top_k,
            num_layers=cfg.num_layers, dropout=cfg.dropout, load_balancing_coef=cfg.load_balancing_coef,
        )
        st = copy.deepcopy(m.state_dict())
        del m
        T.clear_vram()
        return st

    cur_seed, init_state, rows = None, None, []
    for opt, lr, seed in CONFIGS:
        if seed != cur_seed:
            init_state = build_init(seed)   # one init held at a time
            cur_seed = seed
        T.seed_everything(seed)
        setattr(cfg, f"lr_{opt}", lr)
        cfg.seed = seed
        tag = f"{opt}_lr{lr:g}_seed{seed}"
        print(f"\n########## {tag} ##########")
        summary = T.train_one_optimizer(
            cfg=cfg, optimizer_name=opt, tokenizer=tokenizer,
            train_batches=train_batches, val_batches=val_batches,
            initial_state=init_state, device=device, vocab_size=padded_vocab,
        )
        summary["seed"] = seed
        with open(os.path.join(OUT, tag + ".json"), "w") as f:
            json.dump(summary, f, indent=2)
        rows.append((tag, summary["best_val_ppl"]))
        print(f"[fix] {tag}: best PPL {summary['best_val_ppl']:.2f}")

    print("\n===== FIX-PASS SUMMARY =====")
    for tag, ppl in sorted(rows, key=lambda r: r[1]):
        print(f"{ppl:8.2f}  {tag}")


if __name__ == "__main__":
    main()
