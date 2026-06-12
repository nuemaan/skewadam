import os
import sys
import json
import torch
import lm_eval
from lm_eval.models.huggingface import HFLM
from transformers import GPT2TokenizerFast

# Updated import to reference the refactored training file
from train import CausalMoETransformerLM, Config

class MoEEvalWrapper(torch.nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        class MockConfig:
            vocab_size = base_model.vocab_size
            is_encoder_decoder = False
            model_type = "custom_moe"
        self.config = MockConfig()
        
    @property
    def device(self):
        return next(self.base_model.parameters()).device
        
    @property
    def dtype(self):
        return next(self.base_model.parameters()).dtype

    def tie_weights(self):
        pass

    def forward(self, input_ids, **kwargs):
        logits, _ = self.base_model(input_ids)
        class DummyOutput:
            def __init__(self, logits):
                # Upcast logits to FP32 for lm_eval numerical stability
                self.logits = logits.float()
        return DummyOutput(logits)

def run_evals(checkpoint_path: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        print(f"[!] Error: Checkpoint not found at {checkpoint_path}")
        return

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    saved_cfg = Config(**checkpoint["config"])
    vocab_size = checkpoint["vocab_size"] 
    
    model = CausalMoETransformerLM(
        vocab_size=vocab_size, max_positions=saved_cfg.max_positions, d_model=saved_cfg.vocab_dim,
        n_heads=saved_cfg.num_heads, n_kv_heads=saved_cfg.n_kv_heads, ff_dim=saved_cfg.ff_dim, num_experts=saved_cfg.num_experts,
        expert_ff_dim=saved_cfg.expert_ff_dim, top_k=saved_cfg.top_k, num_layers=saved_cfg.num_layers,
        dropout=saved_cfg.dropout, load_balancing_coef=saved_cfg.load_balancing_coef
    )
    
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    # Cast to bfloat16 for evaluation to align with training precision
    model = model.to(device, dtype=torch.bfloat16)

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2", model_max_length=1000000)
    wrapped_model = MoEEvalWrapper(model)
    lm_eval_model = HFLM(pretrained=wrapped_model, tokenizer=tokenizer, backend="causal", device=device)

    tasks = ["hellaswag", "arc_challenge", "winogrande", "piqa"]
    
    print(f"Beginning downstream evaluation pass on tasks: {tasks}")
    results = lm_eval.simple_evaluate(
        model=lm_eval_model,
        tasks=tasks,
        batch_size=32 
    )
    
    opt_name = os.path.basename(checkpoint_path).replace("best_", "").replace(".pt", "")
    out_json = f"eval_metrics_{opt_name}.json"
    
    try:
        with open(out_json, "w") as f:
            json.dump(results.get("results", {}), f, indent=2)
        print(f"\n[✔] Metrics successfully saved to {out_json}")
    except Exception as e:
        print(f"\n[!] Warning: Failed to save JSON metrics: {e}")

    print("\n" + "="*80)
    print(f"FINAL DOWNSTREAM METRICS FOR: {opt_name.upper()}")
    print("="*80)
    
    try:
        from lm_eval.utils import make_table
        print(make_table(results))
    except Exception:
        for task, metrics in results.get("results", {}).items():
            print(f"\nTask: {task}")
            for k, v in metrics.items(): 
                print(f"  {k}: {v}")

if __name__ == "__main__":
    # Support command-line arguments for dynamic evaluation scripting
    target_ckpt = sys.argv[1] if len(sys.argv) > 1 else "runs/best_skewadam.pt"
    run_evals(target_ckpt)