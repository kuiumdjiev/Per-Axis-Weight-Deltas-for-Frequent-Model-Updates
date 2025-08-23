from typing import Optional
import torch
import torch.nn as nn
from bitdelta.binary_gemm_kernel import pack, unpack, binary_bmm


class _BinaryMatmulFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a: torch.Tensor, b_packed: torch.Tensor, n_bits: int):
        assert a.dim() == 2 and b_packed.dim() == 2
        ctx.save_for_backward(b_packed)
        ctx.n_bits = n_bits
        # reuse binary_bmm by adding batch dim
        c = binary_bmm(a.unsqueeze(0), b_packed.unsqueeze(0), n_bits=n_bits).squeeze(0)
        return c

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (b_packed,) = ctx.saved_tensors
        n_bits = ctx.n_bits
        # Unpack to dense {-1, +1}
        S_bool = unpack(b_packed, n_bits=n_bits)  # (K, N) bool when used as (in, out)
        S = (S_bool.to(torch.float32) * 2.0 - 1.0)  # (K, N)
        grad_a = (grad_output.to(torch.float32) @ S.t()).to(grad_output.dtype)
        return grad_a, None, None


def binary_matmul_autograd(a: torch.Tensor, b_packed: torch.Tensor, n_bits: int = 32) -> torch.Tensor:
    return _BinaryMatmulFunction.apply(a, b_packed, n_bits)


class BinaryDiffCol(nn.Module):
    """
    Per-column (output-channel) coefficient:
      y = x @ base + (x @ sign_matrix) * coeff_col

    Expects base/finetune weights shaped (out_dim, in_dim) as in nn.Linear.
    Internally stores base as (in, out) bf16 and mask packed over K=in.
    """
    def __init__(self, base: torch.Tensor, finetune: torch.Tensor,  n_bits: int = 32):
        super().__init__()
        out_dim, in_dim = base.shape
        assert in_dim % n_bits == 0, "in_features must be divisible by n_bits"
        self.in_features = in_dim
        self.out_features = out_dim
        self.n_bits = n_bits
            
        with torch.no_grad():
                diff = (finetune - base)                               
                sign_mask_bool = (diff >= 0)
                packed_mask = pack(sign_mask_bool.T.contiguous(), n_bits=n_bits).contiguous()  

                abs_diff = diff.abs().float()
                nonzeros = (diff != 0).float()
                denom = nonzeros.sum(dim=1).clamp(min=1)              
                alpha_col = (abs_diff * nonzeros).sum(dim=1) / denom  

        self.register_buffer("mask", packed_mask)
        self.register_buffer("base", base.T.to(torch.bfloat16).contiguous())
        self.coeff = nn.Parameter(alpha_col.to(torch.bfloat16).contiguous())  
        
   
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == self.in_features
        orig = x.shape[:-1]
        x2d = x.reshape(-1, self.in_features)
        y_base = x2d @ self.base                                   
        y_sign = binary_bmm(x2d.unsqueeze(0), self.mask.unsqueeze(0), n_bits=self.n_bits).squeeze(0)
        y = y_base + y_sign * self.coeff.view(1, -1)
        return y.view(*orig, self.out_features)


class BinaryDiffRow(nn.Module):
    """
    Per-row (input-channel) coefficient:
      y = x @ base + ((x ⊙ coeff_row) @ sign_matrix)

    Expects base/finetune weights shaped (out_dim, in_dim) as in nn.Linear.
    Uses a custom autograd wrapper so gradients can flow to coeff_row.
    """
    def __init__(self, base: torch.Tensor, finetune: torch.Tensor , n_bits: int = 32):
        super().__init__()
        out_dim, in_dim = base.shape
        assert in_dim % n_bits == 0, "in_features must be divisible by n_bits"
        self.in_features = in_dim
        self.out_features = out_dim
        self.n_bits = n_bits
        
        with torch.no_grad():
                diff = (finetune - base)                             
                sign_mask_bool = (diff >= 0)
                packed_mask = pack(sign_mask_bool.T.contiguous(), n_bits=n_bits).contiguous()  

                abs_diff = diff.abs().float()
                nonzeros = (diff != 0).float()
                denom = nonzeros.sum(dim=0).clamp(min=1)             
                alpha_row = (abs_diff * nonzeros).sum(dim=0) / denom 

        self.register_buffer("mask", packed_mask)
        self.register_buffer("base", base.T.to(torch.bfloat16).contiguous())  
        self.coeff = nn.Parameter(alpha_row.to(torch.bfloat16).contiguous())  
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == self.in_features
        orig = x.shape[:-1]
        x2d = x.reshape(-1, self.in_features) #
        y_base = x2d @ self.base
        x_scaled = x2d * self.coeff.view(1, -1)
        y_diff = binary_matmul_autograd(x_scaled, self.mask, n_bits=self.n_bits)
        y = y_base + y_diff
        return y.view(*orig, self.out_features)