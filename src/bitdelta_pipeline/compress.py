import gc
import torch
from .weights import load_weight_from_cache


def compress_diff(base_model_id: str, finetuned_model, finetuned_compressed_model):
    """Replace certain submodules with BinaryDiff between base and finetuned weights."""
    try:
        from bitdelta.diff import BinaryDiff
    except Exception as e:
        raise ImportError("bitdelta not installed. Please `pip install bitdelta`." ) from e

    def compress_submodule(name, subname, module, submodule):
        target_device = submodule.weight.device
        base_weight = load_weight_from_cache(base_model_id, f"{name}.{subname}.weight").to(target_device)
        finetuned_weight = finetuned_model.get_submodule(f"{name}.{subname}").weight.detach().to(target_device)

        compressed = BinaryDiff(
            base=base_weight,
            finetune=finetuned_weight,
        ).to(target_device)

        # free memory
        del submodule, base_weight
        setattr(module, subname, None)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        setattr(module, subname, compressed)

    # TODO: this can be parallelized
    for name, module in finetuned_compressed_model.named_modules():
        if "mlp" in name or "self_attn" in name:
            for subname, submodule in module.named_children():
                if "proj" in subname:
                    compress_submodule(name, subname, module, submodule)
