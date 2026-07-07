"""Tier ablation for SkewAdam: toggle individual tiers of the allocation policy.

Wraps the released trainer without modifying it: at runtime it extends
train.build_optimizer / train.get_lr / train.theoretical_state_bytes to
recognize four variant names, then hands control to train.main(), so every
variant runs under the exact protocol of the paper (shared init, same data
cache, same schedule, per-variant metrics JSON in runs/).

Variants (dense backbone always keeps momentum + factored V unless noted):

  skewadam            full policy (anchor)         state ~1.3 GB
  skewadam_expertmom  momentum restored to experts state ~25 GB, peak ~55 GB
  skewadam_routerfac  router factored, not exact   state ~1.3 GB
  skewadam_uniform    momentum + factored V everywhere (no policy)

Usage (from the repository root; needs a GPU with >= 60 GB for the
momentum variants):

    python experiments/tier-ablation/run_ablation.py \
        --optimizers "skewadam,skewadam_expertmom,skewadam_routerfac,skewadam_uniform" \
        --dataset-name Skylion007/openwebtext
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import train  # noqa: E402
from skewadam import SkewAdam  # noqa: E402

VARIANTS = {
    # name -> (expert_momentum, router_factored, router_momentum)
    "skewadam_expertmom": (True, False, False),
    "skewadam_routerfac": (False, True, False),
    "skewadam_uniform": (True, True, True),
}

_build = train.build_optimizer
_get_lr = train.get_lr
_state_bytes = train.theoretical_state_bytes


def build_optimizer(name, model, lr, wd):
    if name not in VARIANTS:
        return _build(name, model, lr, wd)
    e_mom, r_fac, r_mom = VARIANTS[name]
    expert_params, router_params, dense_params = [], [], []
    for n, p in model.named_parameters():
        if "experts" in n:
            expert_params.append(p)
        elif "router" in n:
            router_params.append(p)
        else:
            dense_params.append(p)
    return SkewAdam([
        {"params": dense_params, "use_momentum": True, "use_factored": True, "weight_decay": wd},
        {"params": expert_params, "use_momentum": e_mom, "use_factored": True, "weight_decay": wd},
        {"params": router_params, "use_momentum": r_mom, "use_factored": r_fac, "weight_decay": 0.0},
    ], lr=lr)


def get_lr(cfg, name):
    return _get_lr(cfg, "skewadam" if name in VARIANTS else name)


def theoretical_state_bytes(model, name):
    if name not in VARIANTS:
        return _state_bytes(model, name)
    info = dict(_state_bytes(model, "skewadam"))
    e_mom, r_fac, r_mom = VARIANTS[name]
    delta = 0
    for n, p in model.named_parameters():
        if "experts" in n and e_mom:
            delta += p.numel() * 4                       # fp32 momentum added
        elif "router" in n:
            if r_fac and p.dim() >= 2:                    # full V -> row+col vectors
                delta += (sum(p.shape) - p.numel()) * 4
            if r_mom:
                delta += p.numel() * 4
    info["optimizer_state_gb"] = info["optimizer_state_gb"] + delta / 1024**3
    return info


train.build_optimizer = build_optimizer
train.get_lr = get_lr
train.theoretical_state_bytes = theoretical_state_bytes

if __name__ == "__main__":
    train.main()
