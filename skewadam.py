import math
import torch
from torch.optim import Optimizer

class SkewAdam(Optimizer):
    """
    SkewAdam: Memory-efficient optimizer with factored variance tracking.
    Applies in-place momentum and variance updates to minimize memory footprint.
    Incorporates stochastic rounding for low-precision (e.g., bfloat16) training stability.
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, use_momentum=True, use_factored=True)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group['betas']
            eps = group['eps']
            lr = group['lr']
            wd = group['weight_decay']
            use_mom = group['use_momentum']
            use_factored = group['use_factored']

            for p in group['params']:
                if p.grad is None: 
                    continue
                grad = p.grad
                
                if grad.dtype != torch.float32:
                    grad = grad.float()

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    if use_mom:
                        state['exp_avg'] = torch.zeros_like(p, dtype=torch.float32)

                    if p.dim() >= 2 and use_factored: 
                        if p.dim() == 3:
                            state['v_row'] = torch.zeros(p.shape[0], p.shape[1], 1, device=p.device, dtype=torch.float32)
                            state['v_col'] = torch.zeros(p.shape[0], 1, p.shape[2], device=p.device, dtype=torch.float32)
                        else:
                            state['v_row'] = torch.zeros(p.shape[0], 1, device=p.device, dtype=torch.float32)
                            state['v_col'] = torch.zeros(1, p.shape[1], device=p.device, dtype=torch.float32)
                    else: 
                        state['v'] = torch.zeros_like(p, dtype=torch.float32)

                state['step'] += 1
                
                if wd != 0:
                    p.mul_(1 - lr * wd)

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                if p.dim() >= 2 and use_factored:
                    r_mean = (grad ** 2).mean(dim=-1, keepdim=True)
                    c_mean = (grad ** 2).mean(dim=-2, keepdim=True)
                    
                    state['v_row'].mul_(beta2).add_(r_mean, alpha=1 - beta2)
                    state['v_col'].mul_(beta2).add_(c_mean, alpha=1 - beta2)
                    
                    v_mean = state['v_row'].mean(dim=-2, keepdim=True).clamp_min(1e-15)
                    row_factor = (state['v_row'] / v_mean).clamp_min(eps).sqrt()
                    col_factor = state['v_col'].clamp_min(eps).sqrt()
                else:
                    state['v'].mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                    v_factor = state['v'].clamp_min(eps).sqrt()

                if use_mom:
                    state['exp_avg'].mul_(beta1).add_(grad, alpha=1 - beta1)
                    step_mult = math.sqrt(bias_correction2) / bias_correction1
                    update = state['exp_avg'] * step_mult
                else:
                    step_mult = math.sqrt(bias_correction2)
                    update = grad * step_mult

                if p.dim() >= 2 and use_factored:
                    update.div_(row_factor).div_(col_factor)
                else:
                    update.div_(v_factor)
                
                update_rms = torch.linalg.vector_norm(update) / math.sqrt(update.numel())
                update.mul_(1.0 / update_rms.clamp(min=1.0))
                
                # Stochastic rounding for low-precision master weights
                if p.dtype == torch.bfloat16:
                    p_fp32 = p.float()
                    p_fp32.add_(update, alpha=-lr)
                    ulp = torch.abs(p_fp32) * 0.0078125 
                    noise = (torch.rand_like(p_fp32) - 0.5) * ulp
                    p.copy_(p_fp32 + noise)
                else:
                    p.add_(update, alpha=-lr)

        return loss