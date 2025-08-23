import os
import torch


def load_models(args):
    try:
        from bitdelta.utils import get_model, get_tokenizer
    except Exception as e:
        raise ImportError(
            "bitdelta.utils not found. Please install the BitDelta package or ensure it's on PYTHONPATH."
        ) from e

    tokenizer = get_tokenizer(args.base_model)
    with torch.no_grad():
        # base_model = get_model(args.base_model, args.base_model_device, args.base_model_memory_map)
        finetuned_model = get_model(
            args.finetuned_model,
            args.finetuned_model_device,
            args.finetuned_model_memory_map,
        )

    finetuned_compressed_model = get_model(
        args.finetuned_model,
        args.finetuned_compressed_model_device,
        args.finetuned_compressed_model_memory_map,
    )

    os.makedirs(args.save_dir, exist_ok=True)
    return tokenizer, finetuned_model, finetuned_compressed_model
