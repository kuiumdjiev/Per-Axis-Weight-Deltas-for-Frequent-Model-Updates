from typing import Set
import gc
import torch
from torch import nn
from .binary_variants import BinaryDiffRow, BinaryDiffCol
from bitdelta.binary_gemm_kernel import unpack


def save_compression_delta(model: nn.Module, path: str):
    diff = {}
    for name, module in model.named_modules():
        if isinstance(module, BinaryDiffRow):
            diff[name + ".mask_row"] = module.mask.detach().cpu().to(torch.int32)
            diff[name + ".coeff_row"] = module.coeff.detach().cpu().to(torch.bfloat16)
        elif isinstance(module, BinaryDiffCol):
            diff[name + ".mask_col"] = module.mask.detach().cpu().to(torch.int32)
            diff[name + ".coeff_col"] = module.coeff.detach().cpu().to(torch.bfloat16)
    for name, p in model.named_parameters():
        if p.requires_grad:
            diff[name] = p.detach().cpu().to(torch.bfloat16)
    torch.save(diff, path)


def _replace_module(root: nn.Module, path: str, new_mod: nn.Module):
    if "." in path:
        parent_name, child_name = path.rsplit(".", 1)
        parent = root.get_submodule(parent_name)
    else:
        parent = root
        child_name = path
    old = getattr(parent, child_name)
    setattr(parent, child_name, None)
    setattr(parent, child_name, new_mod)
    del old


def load_compression_delta_modules(base_model: nn.Module, path: str, device: str = 'cuda:0'):
    diff = torch.load(path, map_location='cpu')
    targets: Set[str] = set()
    for k in diff.keys():
        for suff in ('.mask_row', '.coeff_row', '.mask_col', '.coeff_col'):
            if k.endswith(suff):
                targets.add(k[:-len(suff)])

    for name in sorted(targets):
        base_leaf = base_model.get_submodule(name)
        base_w = base_leaf.weight.detach().cpu()
        if name + '.mask_row' in diff:
            mask = diff[name + '.mask_row'].to(torch.int32).contiguous()
            coeff = diff[name + '.coeff_row'].to(torch.bfloat16).contiguous()
            new_mod = BinaryDiffRow(base=base_w, finetune=None, mask=mask, coeff=coeff, n_bits=32).to(device)
        elif name + '.mask_col' in diff:
            mask = diff[name + '.mask_col'].to(torch.int32).contiguous()
            coeff = diff[name + '.coeff_col'].to(torch.bfloat16).contiguous()
            new_mod = BinaryDiffCol(base=base_w, finetune=None, mask=mask, coeff=coeff, n_bits=32).to(device)
        else:
            continue
        _replace_module(base_model, name, new_mod)
        del base_leaf, base_w, mask, coeff, new_mod
        gc.collect(); torch.cuda.empty_cache()

    for name, p in base_model.named_parameters():
        if name in diff:
            p.data.copy_(diff[name].to(p.device))

    del diff; gc.collect(); torch.cuda.empty_cache()


def merge_compression_delta_into_weights(base_model: nn.Module, path: str, device: str = 'cuda:0'):
    diff = torch.load(path, map_location='cpu')
    targets: Set[str] = set()
    for k in diff.keys():
        for suff in ('.mask_row', '.coeff_row', '.mask_col', '.coeff_col'):
            if k.endswith(suff):
                targets.add(k[:-len(suff)])

    for name in sorted(targets):
        module = base_model.get_submodule(name)
        if not hasattr(module, 'weight'):
            continue
        W = module.weight.data.to(torch.float32)
        if name + '.mask_row' in diff:
            mask = diff.pop(name + '.mask_row').to(torch.int32).contiguous()
            coeff = diff.pop(name + '.coeff_row').to(torch.float32).contiguous()
            S_bool = unpack(mask, n_bits=32)
            S = (S_bool.to(torch.float32) * 2 - 1)
            delta = (S * coeff.view(-1, 1)).T
            W.add_(delta.to(W.device))
            del mask, coeff, S_bool, S, delta
        elif name + '.mask_col' in diff:
            mask = diff.pop(name + '.mask_col').to(torch.int32).contiguous()
            coeff = diff.pop(name + '.coeff_col').to(torch.float32).contiguous()
            S_bool = unpack(mask, n_bits=32)
            S = (S_bool.to(torch.float32) * 2 - 1)
            delta = (S * coeff.view(1, -1)).T
            W.add_(delta.to(W.device))
            del mask, coeff, S_bool, S, delta
        module.weight.data = W.to(torch.bfloat16).to(device)
        del W
        gc.collect(); torch.cuda.empty_cache()

    for name, p in base_model.named_parameters():
        if name in diff:
            p.data.copy_(diff[name].to(p.device))

    del diff; gc.collect(); torch.cuda.empty_cache()
