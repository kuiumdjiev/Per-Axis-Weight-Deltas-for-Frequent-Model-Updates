import os
import json
import gc
import torch
from safetensors import safe_open


def load_weight_from_cache(model_name: str, weight_name: str):
    """Load a single tensor by key from the local Hugging Face cache for a model.

    Args:
        model_name: e.g. "meta-llama/Llama-3.1-8B"
        weight_name: e.g. "model.layers.0.self_attn.q_proj.weight"

    Returns:
        torch.Tensor loaded on CPU
    """
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    subdirs = [d for d in os.listdir(cache_dir) if model_name.replace("/", "--") in d]
    if not subdirs:
        raise FileNotFoundError(f"Модел '{model_name}' не е намерен в кеша.")

    model_dir = os.path.join(cache_dir, subdirs[0], "snapshots")
    snapshot_dirs = os.listdir(model_dir)
    if not snapshot_dirs:
        raise FileNotFoundError(f"Няма snapshot за модела '{model_name}'.")

    snapshot_path = os.path.join(model_dir, snapshot_dirs[0])

    index_json = os.path.join(snapshot_path, "model.safetensors.index.json")
    with open(index_json, "r", encoding="utf-8") as f:
        index = json.load(f)

    weights_file = os.path.join(snapshot_path, index['weight_map'][weight_name])
    if weights_file is None:
        raise FileNotFoundError(f"Не са намерени тегла за модела '{model_name}'.")

    if weights_file.endswith(".safetensors"):
        with safe_open(weights_file, framework="pt", device="cpu") as f:
            if weight_name not in f.keys():
                raise KeyError(f"Теглото '{weight_name}' не е намерено.")
            tensor = f.get_tensor(weight_name)
    else:
        state_dict = torch.load(weights_file, map_location="cpu")
        if weight_name not in state_dict:
            raise KeyError(f"Теглото '{weight_name}' не е намерено.")
        tensor = state_dict[weight_name]
        del state_dict

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return tensor
