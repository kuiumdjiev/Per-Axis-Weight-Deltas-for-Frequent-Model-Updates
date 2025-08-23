from typing import List
import torch
import torch.nn.functional as F
from tqdm import tqdm


def cache_reference_logits(model, dataloader, device: str, max_train_steps: int = 800) -> List[torch.Tensor]:
    """Run reference model to cache logits for training batches."""
    results = []
    bar = tqdm(dataloader, desc="Caching train ref logits")
    for step, batch in enumerate(bar):
        if step >= max_train_steps:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            out = model(**batch)
        results.append(out.logits.detach().cpu())
    return results


def cache_val_reference_logits(model, val_dataloader, device: str, max_val_batches: int = 150) -> List[torch.Tensor]:
    results = []
    for i, val_batch in enumerate(val_dataloader):
        if i >= max_val_batches:
            break
        val_batch = {
            'input_ids': val_batch['input_ids'].to(device, dtype=torch.long),
            'attention_mask': val_batch['attention_mask'].to(device, dtype=torch.bfloat16),
        }
        with torch.no_grad():
            out = model(**val_batch)
        results.append(out.logits.detach().cpu())
    return results


def evaluate_model_end_loss(model, val_dataloader, val_ref_logits, device: str, max_val_batches: int = 150) -> float:
    with torch.no_grad():
        val_loss = 0.0
        num_batches = 0
        for i, val_batch in enumerate(val_dataloader):
            if i >= max_val_batches:
                break
            val_batch = {
                'input_ids': val_batch['input_ids'].to(device, dtype=torch.long),
                'attention_mask': val_batch['attention_mask'].to(device, dtype=torch.bfloat16)
            }
            out = model(**val_batch)
            val_loss += F.mse_loss(out.logits, val_ref_logits[i].to(device)).item()
            del out
            num_batches += 1
        val_loss /= max(1, num_batches)
    return val_loss
