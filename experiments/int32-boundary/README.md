# The int32 wall: optimizer state vs. tensors with numel() ≥ 2³¹

8-bit optimizer states are the other obvious way to shrink Adam's memory
footprint. This experiment documents a hard failure mode they carry that
factored state does not: bitsandbytes' `Adam8bit` **kills the process** the
moment a single parameter tensor reaches 2³¹ elements.

## Result

Each cell ran in a fresh process (the failure is a C++ `exit(1)`, not a
catchable Python exception). `torch.optim.Adam` (fp32 state) is the control.

| numel | Adam (fp32) | bnb Adam8bit | SkewAdam |
|---|---|---|---|
| 2,000,000,000 | pass | pass | pass |
| 2,147,483,000 | pass | pass | pass |
| 2,147,483,647 (INT_MAX) | pass | pass | pass |
| **2,147,483,648 (2³¹)** | pass | **process killed** | pass |
| 46,341 × 46,341 = 2,147,488,281 (2-D) | pass | **process killed** | pass |

The boundary is exactly `numel() == 2**31`, it applies to total element
count (the 2-D shape fails identically), and the console output at failure is:

```
Error invalid argument at line 226 in file /src/csrc/ops.cu
```

followed by process exit code 1 — no exception reaches Python, so a long
training run dies without a checkpoint flush. SkewAdam and fp32 Adam cross
the boundary cleanly because they use only native PyTorch ops with 64-bit
indexing; the 2-D case exercises SkewAdam's factored path.

Reported upstream on bitsandbytes issue
[#1785](https://github.com/bitsandbytes-foundation/bitsandbytes/issues/1785),
which tracks `numel() > INT_MAX` support and lists optimizers as low priority.

## Environment

NVIDIA A100-SXM4-80GB · torch 2.12.0+cu130 (CUDA 13.0) · bitsandbytes 0.49.2
· Python 3.11 · `CUDA_LAUNCH_BLOCKING=1`. Raw records with timings and
versions: [results.json](results.json).

## Reproduce

Runs on [Modal](https://modal.com) (one A100-80GB container per test):

```bash
pip install modal
modal setup
modal run experiments/int32-boundary/modal_sweep.py   # from the repo root
```
