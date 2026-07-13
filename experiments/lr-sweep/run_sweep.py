"""Learning-rate sweep for the baselines (AdamW, Adafactor) whose reported
numbers rest on a single untuned learning rate.

Wraps the released trainer without modifying it: builds the tokenizer, the
batch cache, and a per-seed initialization once, then loops over a grid of
(optimizer, lr, seed), calling train.train_one_optimizer for each and saving
its summary to experiments/lr-sweep/results/<opt>_lr<lr>_seed<seed>.json.
Sharing the data build across all runs is the whole point --- it is the
expensive setup, and rebuilding it per run would multiply cost.

Grid (edit GRID / SEEDS below): AdamW {1e-4, 3e-4, 1e-3}, Adafactor {1e-3,
3e-3}, seeds {42, 43} -> 10 full 10k-step runs. SkewAdam keeps its reported
3e-4 (its 108.4/108.9/108.9 across three platforms is the reference), so this
is the conservative test: do LR-tuned baselines close the gap to an untuned
SkewAdam?

Usage (from the repository root, on a >=90 GB GPU so AdamW fits):
    python experiments/lr-sweep/run_sweep.py --dataset-name Skylion007/openwebtext
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

GRID = [
    ("adam", 1e-4), ("adam", 3e-4), ("adam", 1e-3),
    ("adafactor", 1e-3), ("adafactor", 3e-3),
]
SEEDS = [42, 43]

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-name", type=str, default="Skylion007/openwebtext")
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--seeds", type=str, default=",".join(str(s) for s in SEEDS))
    args = p.parse_args()
    seeds = [int(x) for x in args.seeds.split(",")]

    os.makedirs(OUT, exist_ok=True)
    cfg = T.Config(dataset_name=args.dataset_name, steps=args.steps, out_dir="runs")
    torch.set_float32_matmul_precision("high")

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2", model_max_length=1000000)
    padded_vocab = math.ceil(tokenizer.vocab_size / 64) * 64

    # data cache built ONCE, reused by every run
    train_batches, val_batches = T.build_batch_caches(
        tokenizer=tokenizer, dataset_name=cfg.dataset_name, batch_size=cfg.batch_size,
        seq_len=cfg.seq_len, train_target_batches=cfg.train_batches,
        val_target_batches=cfg.val_batches, train_ratio=cfg.train_ratio,
    )
    device = torch.device("cuda")

    def build_init():
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

    rows = []
    for seed in seeds:
        T.seed_everything(seed)
        init_state = build_init()  # fresh init per seed (true seed variation)
        for opt, lr in GRID:
            T.seed_everything(seed)  # matched stochasticity across opts within a seed
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
            print(f"[sweep] {tag}: best PPL {summary['best_val_ppl']:.2f}")

    print("\n===== SWEEP SUMMARY =====")
    for tag, ppl in sorted(rows, key=lambda r: r[1]):
        print(f"{ppl:8.2f}  {tag}")


if __name__ == "__main__":
    main()
