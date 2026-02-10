from typing import Tuple
import gc
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from .binary_variants import BinaryDiffRow, BinaryDiffCol
from .weights import load_weight_from_cache
from .io_capture import collect_layer_io
from .eval_utils import evaluate_model_end_loss


def _train_layer_module(module: nn.Module, inputs: torch.Tensor, targets: torch.Tensor, epochs: int = 3, lr: float = 1e-4) -> float:
    device = next(module.parameters()).device
    ds = TensorDataset(inputs, targets)
    dl = DataLoader(ds, batch_size=2, shuffle=True, num_workers=0)
    opt = torch.optim.AdamW(module.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()
    last_val = 0.0
    for _ in range(epochs):
        module.train()
        for xb, yb in dl:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            yp = module(xb)
            loss = loss_fn(yp, yb)
            loss.backward()
            opt.step()
        last_val = loss.item()
    return last_val


def compress_submodule_choose(base_model_id: str, finetuned_model, finetuned_compressed_model, name: str, subname: str,
                               dataloader, val_dataloader,
                               ft_device: str = 'cuda:0', comp_device: str = 'cuda:1') -> Tuple[str, float, float]:
    layer_name = f"{name}.{subname}"
    # Collect IO
    ins_tr, outs_tr, ins_val, outs_val = collect_layer_io(
        finetuned_model, finetuned_compressed_model, dataloader, val_dataloader,
        layer_name, train_steps=50, eval_steps=5, finetuned_device=ft_device, compressed_device=comp_device
    )
    target_device = comp_device
    base_w = load_weight_from_cache(base_model_id, layer_name + '.weight').to(target_device, dtype=torch.bfloat16)
    ft_w = finetuned_model.get_submodule(layer_name).weight.detach().to(target_device, dtype=torch.bfloat16)

    # Stack inputs/outputs
    Xtr = torch.stack([t[0] if isinstance(t, (tuple, list)) else t for t in ins_tr]).to(target_device, dtype=torch.bfloat16)
    Ytr = torch.stack([t if torch.is_tensor(t) else t[0] for t in outs_tr]).to(target_device, dtype=torch.bfloat16)
    Xval = torch.stack([t[0] if isinstance(t, (tuple, list)) else t for t in ins_val]).to(target_device, dtype=torch.bfloat16)
    Yval = torch.stack([t if torch.is_tensor(t) else t[0] for t in outs_val]).to(target_device, dtype=torch.bfloat16)

    # Train col
    col = BinaryDiffCol(base_w, finetune=ft_w, n_bits=32).to(target_device)
    col_loss = _train_layer_module(col, Xtr, Ytr, epochs=3, lr=1e-4)

    # Place col temporarily and evaluate end loss
    parent = finetuned_compressed_model.get_submodule(name)
    setattr(parent, subname, col)

    # Train row
    row = BinaryDiffRow(base_w, finetune=ft_w, n_bits=32).to(target_device)
    row_loss = _train_layer_module(row, Xtr, Ytr, epochs=3, lr=1e-5)

    # Choose
    setattr(parent, subname, None); gc.collect(); torch.cuda.empty_cache()
    if row_loss <= col_loss:
        setattr(parent, subname, row)
        choice = 'row'
    else:
        setattr(parent, subname, col)
        choice = 'col'

    # cleanup unused
    if choice == 'row':
        del col
    else:
        del row
    del base_w, ft_w, Xtr, Ytr, Xval, Yval
    gc.collect(); torch.cuda.empty_cache()
    return choice, row_loss, col_loss


def compress_model_choosing(base_model_id: str, finetuned_model, finetuned_compressed_model, dataloader, val_dataloader,
                            ft_device: str = 'cuda:0', comp_device: str = 'cuda:1'):
    for name, module in finetuned_compressed_model.named_modules():
        if 'mlp' in name or 'self_attn' in name:
            for subname, submodule in module.named_children():
                if 'proj' in subname and hasattr(submodule, 'weight'):
                    print(f"Choosing variant for {name}.{subname} ...")
                    try:
                        choice, row_loss, col_loss = compress_submodule_choose(
                            base_model_id, finetuned_model, finetuned_compressed_model,
                            name, subname, dataloader, val_dataloader,
                            ft_device, comp_device
                        )
                        print(f" → {choice} (row={row_loss:.4f}, col={col_loss:.4f})")
                    except Exception as e:
                        print(f"Failed for {name}.{subname}: {e}")
                        continue
