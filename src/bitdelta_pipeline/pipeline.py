import os
from dataclasses import asdict
import torch

from .args import Args
from .models import load_models
from .data import build_dataloaders
from .compress import compress_diff
from .eval_utils import cache_reference_logits, cache_val_reference_logits, evaluate_model_end_loss
from .train import train


def run_pipeline(args: Args):
    # Load models and tokenizer
    tokenizer, finetuned_model, finetuned_compressed_model = load_models(args)

    # Data
    dataloader, val_dataloader = build_dataloaders(tokenizer, args.max_length)

    # Cache reference logits from finetuned model on GPU 0
    ref_device = args.finetuned_model_device
    ref_train_logits = cache_reference_logits(finetuned_model, dataloader, ref_device, max_train_steps=args.num_steps)
    ref_val_logits = cache_val_reference_logits(finetuned_model, val_dataloader, ref_device, max_val_batches=150)

    # Free finetuned_model if needed (optional)
    # del finetuned_model; torch.cuda.empty_cache()

    # Compress diff into finetuned_compressed_model using base model weights from cache
    print("Compressing diff...")
    compress_diff(args.base_model, finetuned_model, finetuned_compressed_model)

    # Train the compressed model to match finetuned logits
    device = args.finetuned_compressed_model_device if args.finetuned_compressed_model_device != "auto" else "cuda:0"
    train_losses, final_val_loss = train(
        finetuned_compressed_model,
        dataloader,
        ref_train_logits,
        val_dataloader,
        ref_val_logits,
        device=device,
        num_steps=args.num_steps,
        lr=1e-4,
        eval_every=100,
        max_val_batches=150,
    )

    # Save losses
    if args.debug:
        os.makedirs(args.save_dir, exist_ok=True)
        import json
        with open(os.path.join(args.save_dir, "train_loss.json"), "w") as f:
            json.dump(train_losses, f)
        with open(os.path.join(args.save_dir, "args.json"), "w") as f:
            json.dump(asdict(args), f, indent=2)

    # Save delta and full model
    try:
        from bitdelta.diff import save_diff, save_full_model
        save_path = os.path.join(args.save_dir, "lamma3.1-bitdelta-instruct-800.pt")
        save_diff(finetuned_compressed_model, save_path)
        save_full_model(
            args.base_model,
            args.finetuned_model,
            save_path,
            os.path.join(args.save_dir, "uncalibrated_model"),
            device="cpu",
        )
    except Exception as e:
        print("Skipping save_diff/save_full_model because bitdelta is not installed or failed:", e)

    # Final validation loss
    print(f"Final validation loss: {final_val_loss:.6f}")


if __name__ == "__main__":
    run_pipeline(Args())
