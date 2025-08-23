from typing import Dict, List, Tuple
import torch
from tqdm import tqdm

layer_inputs: Dict[str, List[torch.Tensor]] = {}
layer_outputs: Dict[str, List[torch.Tensor]] = {}


def _hook(layer_name: str, kind: str):
    def fn(module, inp, out):
        if layer_name not in layer_inputs:
            layer_inputs[layer_name] = []
            layer_outputs[layer_name] = []
        if kind == 'i':
            x = inp[0] if isinstance(inp, tuple) else inp
            if torch.is_tensor(x):
                layer_inputs[layer_name].append(x.detach().clone())
        else:
            y = out if torch.is_tensor(out) else out[0]
            layer_outputs[layer_name].append(y.detach().clone())
    return fn


def collect_layer_io(finetuned_model, finetuned_compressed_model, dataloader, val_dataloader,
                      layer_name: str, train_steps: int = 50, eval_steps: int = 5,
                      finetuned_device: str = 'cuda:0', compressed_device: str = 'cuda:1') -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    layer_inputs.clear(); layer_outputs.clear()
    h_ft = finetuned_model.get_submodule(layer_name).register_forward_hook(_hook(layer_name, 'o'))
    h_fc = finetuned_compressed_model.get_submodule(layer_name).register_forward_hook(_hook(layer_name, 'i'))

    for step, batch in enumerate(tqdm(dataloader, desc=f"collect train io {layer_name}")):
        if step >= train_steps:
            break
        batch1 = {k: v.to(finetuned_device) for k, v in batch.items()}
        with torch.no_grad():
            finetuned_model(**batch1)
        batch2 = {k: v.to(compressed_device) for k, v in batch.items()}
        with torch.no_grad():
            finetuned_compressed_model(**batch2)

    for step, batch in enumerate(tqdm(val_dataloader, desc=f"collect val io {layer_name}")):
        if step >= eval_steps:
            break
        batch1 = {k: v.to(finetuned_device) for k, v in batch.items()}
        with torch.no_grad():
            finetuned_model(**batch1)
        batch2 = {k: v.to(compressed_device) for k, v in batch.items()}
        with torch.no_grad():
            finetuned_compressed_model(**batch2)

    h_ft.remove(); h_fc.remove()

    ins = layer_inputs[layer_name]
    outs = layer_outputs[layer_name]
    return ins[:train_steps], outs[:train_steps], ins[-eval_steps:], outs[-eval_steps:]
