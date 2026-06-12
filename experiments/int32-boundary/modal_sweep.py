"""INT_MAX boundary sweep for optimizer steps, run on Modal.

Tests whether optimizer.step() survives a single parameter tensor whose
numel() approaches and crosses 2**31, for three optimizers:

  * Adam32   - torch.optim.Adam, fp32 state (native 64-bit indexing)
  * Adam8bit - bitsandbytes 8-bit Adam (int32 kernel indexing suspected)
  * SkewAdam - the released optimizer from this repository

Each (size, optimizer) pair runs in its OWN container: the bitsandbytes
failure mode is a hard process exit from C++ (`Error invalid argument at
line 226 in /src/csrc/ops.cu` followed by exit(1)), not a catchable Python
exception, so a shared container would die at the first failing test.

Sizes: below/at/above INT_MAX on 1-D tensors, plus a 2-D shape
(46341 x 46341, numel 2,147,488,281) that exercises SkewAdam's factored
path rather than its 1-D fallback.

Usage (from the repository root):
    modal run experiments/int32-boundary/modal_sweep.py

Writes experiments/int32-boundary/results.json.
"""
import json

import modal

app = modal.App("skewadam-int32-boundary")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "bitsandbytes")
    .env({"CUDA_LAUNCH_BLOCKING": "1"})  # precise error location
    .add_local_file("skewadam.py", "/root/skewadam.py")
)

INT_MAX = 2_147_483_647

TESTS = [
    # (shape, optimizer)
    ((2_000_000_000,), "Adam32"),
    ((2_000_000_000,), "Adam8bit"),
    ((2_000_000_000,), "SkewAdam"),
    ((2_147_483_000,), "Adam32"),
    ((2_147_483_000,), "Adam8bit"),
    ((2_147_483_000,), "SkewAdam"),
    ((INT_MAX,), "Adam32"),
    ((INT_MAX,), "Adam8bit"),
    ((INT_MAX,), "SkewAdam"),
    ((INT_MAX + 1,), "Adam32"),
    ((INT_MAX + 1,), "Adam8bit"),
    ((INT_MAX + 1,), "SkewAdam"),
    ((46341, 46341), "Adam32"),
    ((46341, 46341), "Adam8bit"),
    ((46341, 46341), "SkewAdam"),
]


@app.function(gpu="A100-80GB", timeout=900, retries=0, image=image)
def run_one(shape, opt_name):
    import time
    import traceback

    import torch
    import bitsandbytes as bnb
    from skewadam import SkewAdam

    numel = 1
    for s in shape:
        numel *= s
    rec = {
        "shape": list(shape),
        "numel": numel,
        "optimizer": opt_name,
        "success": False,
        "error_type": None,
        "error": None,
        "traceback": None,
        "step_time_s": None,
        "env": {
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "bitsandbytes": bnb.__version__,
        },
    }
    try:
        param = torch.nn.Parameter(torch.ones(*shape, device="cuda"))
        if opt_name == "Adam32":
            opt = torch.optim.Adam([param], lr=1e-3)
        elif opt_name == "Adam8bit":
            opt = bnb.optim.Adam8bit([param], lr=1e-3)
        elif opt_name == "SkewAdam":
            opt = SkewAdam([param], lr=1e-3)
        param.grad = torch.ones_like(param)
        torch.cuda.synchronize()
        t0 = time.time()
        opt.step()
        torch.cuda.synchronize()
        rec["success"] = True
        rec["step_time_s"] = round(time.time() - t0, 4)
    except Exception as e:
        rec["error_type"] = type(e).__name__
        rec["error"] = str(e)
        rec["traceback"] = traceback.format_exc()
    return rec


@app.local_entrypoint()
def main():
    results = []
    for test, res in zip(
        TESTS, run_one.starmap(TESTS, return_exceptions=True, wrap_returned_exceptions=False)
    ):
        shape, opt_name = test
        if isinstance(res, Exception):
            # Container died (bitsandbytes C++ exit(1)) before returning,
            # or the call failed in transit; keep the exception class visible.
            numel = 1
            for s in shape:
                numel *= s
            res = {
                "shape": list(shape),
                "numel": numel,
                "optimizer": opt_name,
                "success": False,
                "error_type": f"ProcessCrash({type(res).__name__})",
                "error": repr(res)[:400],
                "traceback": None,
                "step_time_s": None,
                "env": None,
            }
        results.append(res)

    env = next((r["env"] for r in results if r.get("env")), {})
    print(json.dumps(env, indent=2))
    print(f"\n{'shape':>18} | {'numel':>13} | {'optimizer':>9} | {'result':<34} | {'t(s)':>7}")
    print("-" * 96)
    for r in results:
        res_s = "PASS" if r["success"] else f"FAIL: {r['error_type']}"
        t = r["step_time_s"] if r["step_time_s"] is not None else "-"
        print(f"{str(tuple(r['shape'])):>18} | {r['numel']:>13,} | {r['optimizer']:>9} | {res_s:<34} | {t:>7}")

    path = "experiments/int32-boundary/results.json"
    with open(path, "w") as f:
        json.dump({"env": env, "results": results}, f, indent=2)
    print(f"\nwrote {path}")
