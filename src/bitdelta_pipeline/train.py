from typing import List, Tuple
import gc
import torch
import torch.nn.functional as F
from tqdm import tqdm


def train(
    model,
    dataloader,
    ref_logits: List[torch.Tensor],
    val_dataloader,
    val_ref_logits: List[torch.Tensor],
    device: str,
    num_steps: int = 800,
    lr: float = 1e-4,
    eval_every: int = 100,
    max_val_batches: int = 150,
) -> Tuple[list, float]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_steps)

    model.to(device)

    train_losses = []
    bar = tqdm(dataloader, desc="Training")
    for step, batch in enumerate(bar):
        if step >= num_steps:
            break
        batch = {k: v.to(device) for k, v in batch.items()}

        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()

        out = model(**batch)
        target = ref_logits[step].to(device)
        loss = F.mse_loss(out.logits, target, reduction='sum')

        train_losses.append(loss.item())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()
        bar.set_description(f"train loss: {loss.item():.6f}")

        del loss, out, target
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()

        if step % eval_every == 0 and step > 0:
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
                print(f"Validation loss: {val_loss:.6f}")

    # final eval
    final_val_loss = 0.0
    with torch.no_grad():
        num_batches = 0
        for i, val_batch in enumerate(val_dataloader):
            if i >= max_val_batches:
                break
            val_batch = {
                'input_ids': val_batch['input_ids'].to(device, dtype=torch.long),
                'attention_mask': val_batch['attention_mask'].to(device, dtype=torch.bfloat16)
            }
            out = model(**val_batch)
            final_val_loss += F.mse_loss(out.logits, val_ref_logits[i].to(device)).item()
            del out
            num_batches += 1
    final_val_loss /= max(1, num_batches)

    return train_losses, final_val_loss
