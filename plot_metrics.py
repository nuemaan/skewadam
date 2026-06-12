import os
import json
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

COLORS = {
    "skewadam": "#1f77b4",
    "adam": "#7f7f7f",
    "muon": "#ff7f0e",
    "lion": "#2ca02c"
}

LABELS = {
    "skewadam": "SkewAdam",
    "adam": "AdamW",
    "muon": "Muon",
    "lion": "Lion"
}

def set_plot_style():
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        "font.family": "serif",
        "axes.labelsize": 12,
        "axes.titlesize": 14,
        "legend.fontsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.dpi": 300,
        "savefig.bbox": "tight"
    })

def load_metrics(run_dir):
    data = {}
    for path in glob.glob(os.path.join(run_dir, "metrics_*.json")):
        try:
            with open(path, "r") as f:
                payload = json.load(f)
                if "optimizer" in payload:
                    data[payload["optimizer"]] = payload
        except Exception as e:
            print(f"Error reading {path}: {e}")
    return data

def plot_learning_curves(data, out_dir):
    fig_ppl, ax_ppl = plt.subplots(figsize=(6, 4))
    fig_loss, ax_loss = plt.subplots(figsize=(6, 4))

    for opt, payload in data.items():
        if "history" not in payload or not payload["history"]:
            continue
            
        steps = [entry["step"] for entry in payload["history"]]
        val_ppl = [entry["val_ppl"] for entry in payload["history"]]
        train_loss = [entry["train_lm_loss"] for entry in payload["history"]]
        
        color = COLORS.get(opt, "#000000")
        label = LABELS.get(opt, opt)

        ax_ppl.plot(steps, val_ppl, label=label, color=color)
        ax_loss.plot(steps, train_loss, label=label, color=color)

    ax_ppl.set_yscale('log')
    ax_ppl.set_xlabel("Steps")
    ax_ppl.set_ylabel("Validation Perplexity")
    ax_ppl.legend()
    fig_ppl.savefig(os.path.join(out_dir, "val_perplexity.pdf"))

    ax_loss.set_xlabel("Steps")
    ax_loss.set_ylabel("Training Loss")
    ax_loss.legend()
    fig_loss.savefig(os.path.join(out_dir, "train_loss.pdf"))
    
    plt.close(fig_ppl)
    plt.close(fig_loss)

def plot_memory_footprint(data, out_dir):
    sorted_data = sorted(data.values(), key=lambda x: x.get("final_peak_vram_gb", 0))
    
    opts = [LABELS.get(d["optimizer"], d["optimizer"]) for d in sorted_data]
    peak_vram = [d.get("final_peak_vram_gb", 0) for d in sorted_data]
    opt_state = [d.get("theory", {}).get("optimizer_state_gb", 0) for d in sorted_data]

    x = np.arange(len(opts))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 4.5))
    
    r1 = ax.bar(x - width/2, peak_vram, width, label='Peak VRAM (GB)', color='#4c72b0')
    r2 = ax.bar(x + width/2, opt_state, width, label='Optimizer State (GB)', color='#dd8452')

    ax.set_ylabel('Memory (GB)')
    ax.set_xticks(x)
    ax.set_xticklabels(opts)
    ax.legend()
    
    ax.bar_label(r1, padding=3, fmt='%.1f')
    ax.bar_label(r2, padding=3, fmt='%.2f')

    ax.axhline(y=40, color='red', linestyle='--', alpha=0.5)
    ax.text(len(opts) - 0.5, 41, '40GB Limit', color='red', va='bottom', ha='right', fontsize=10)

    fig.savefig(os.path.join(out_dir, "memory.pdf"))
    plt.close(fig)

def plot_throughput(data, out_dir):
    sorted_data = sorted(data.values(), key=lambda x: x.get("final_tok_per_sec", 0), reverse=True)
    
    opts = [LABELS.get(d["optimizer"], d["optimizer"]) for d in sorted_data]
    throughput = [d.get("final_tok_per_sec", 0) for d in sorted_data]
    colors = [COLORS.get(d["optimizer"], "#000000") for d in sorted_data]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(opts, throughput, color=colors)

    ax.set_ylabel('Tokens / Sec')
    ax.set_ylim(0, max(throughput) * 1.15) 
    ax.bar_label(bars, padding=3, fmt='%d')

    fig.savefig(os.path.join(out_dir, "throughput.pdf"))
    plt.close(fig)

def main():
    set_plot_style()
    
    run_dir = "runs"
    out_dir = "figures"
    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        
    data = load_metrics(run_dir)
    
    if not data:
        print(f"No metrics found in {run_dir}/ directory.")
        return
        
    plot_learning_curves(data, out_dir)
    plot_memory_footprint(data, out_dir)
    plot_throughput(data, out_dir)

if __name__ == "__main__":
    main()