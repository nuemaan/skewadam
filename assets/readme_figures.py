"""Render the README figures as PNGs from the metrics in runs/.

Standalone: mirrors the styling of plot_metrics.py but writes PNG instead of
PDF so the figures display inline on GitHub. Does not modify any training
code or the paper figures.
"""
import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

COLORS = {
    "skewadam": "#1f77b4",
    "adam": "#7f7f7f",
    "muon": "#ff7f0e",
    "lion": "#2ca02c",
}

LABELS = {
    "skewadam": "SkewAdam",
    "adam": "AdamW",
    "muon": "Muon",
    "lion": "Lion",
}

ORDER = ["skewadam", "adam", "muon", "lion"]

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.dirname(os.path.abspath(__file__))


def load():
    data = {}
    for path in glob.glob(os.path.join(ROOT, "runs", "metrics_*.json")):
        with open(path) as f:
            payload = json.load(f)
        if "optimizer" in payload:
            data[payload["optimizer"]] = payload
    return data


def style():
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        "font.family": "serif",
        "axes.labelsize": 12,
        "legend.fontsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "savefig.bbox": "tight",
    })


def val_perplexity(data):
    fig, ax = plt.subplots(figsize=(6, 4))
    for opt in ORDER:
        hist = data[opt]["history"]
        ax.plot([h["step"] for h in hist], [h["val_ppl"] for h in hist],
                label=LABELS[opt], color=COLORS[opt])
    ax.set_yscale("log")
    ax.set_xlabel("Steps")
    ax.set_ylabel("Validation Perplexity")
    ax.legend()
    fig.savefig(os.path.join(OUT, "val_perplexity.png"), dpi=200)
    plt.close(fig)


def memory(data):
    sorted_data = sorted(data.values(), key=lambda x: x["final_peak_vram_gb"])
    opts = [LABELS[d["optimizer"]] for d in sorted_data]
    peak = [d["final_peak_vram_gb"] for d in sorted_data]
    state = [d["theory"]["optimizer_state_gb"] for d in sorted_data]

    x = np.arange(len(opts))
    width = 0.35
    fig, ax = plt.subplots(figsize=(6, 4))
    r1 = ax.bar(x - width / 2, peak, width, label="Peak VRAM (GB)", color="#4c72b0")
    r2 = ax.bar(x + width / 2, state, width, label="Optimizer state (GB)", color="#dd8452")
    ax.set_ylabel("Memory (GB)")
    ax.set_xticks(x)
    ax.set_xticklabels(opts)
    ax.bar_label(r1, padding=3, fmt="%.1f")
    ax.bar_label(r2, padding=3, fmt="%.2f")
    ax.axhline(y=40, color="red", linestyle="--", alpha=0.5)
    ax.text(len(opts) - 0.5, 41, "40 GB limit", color="red", va="bottom", ha="right", fontsize=10)
    ax.legend()
    fig.savefig(os.path.join(OUT, "memory.png"), dpi=200)
    plt.close(fig)


def aux_loss(data):
    fig, ax = plt.subplots(figsize=(6, 4))
    for opt in ORDER:
        hist = data[opt]["history"]
        ax.plot([h["step"] for h in hist], [h["aux_loss"] for h in hist],
                label=LABELS[opt], color=COLORS[opt], marker="o", markersize=3)
    ax.axhline(y=0.05, color="black", linestyle="--", linewidth=1, alpha=0.6)
    ax.text(200, 0.0492, "balanced-routing floor (0.05)", fontsize=9, va="top")
    ax.set_xlabel("Steps")
    ax.set_ylabel("Load-balancing loss")
    ax.set_ylim(0.046, 0.063)
    ax.legend(ncol=2)
    fig.savefig(os.path.join(OUT, "aux_loss.png"), dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    style()
    data = load()
    val_perplexity(data)
    memory(data)
    aux_loss(data)
    print("wrote val_perplexity.png, memory.png, aux_loss.png")
