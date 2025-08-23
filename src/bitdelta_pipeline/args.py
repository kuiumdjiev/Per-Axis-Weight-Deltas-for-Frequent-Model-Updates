from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class Args:
    base_model: str = "meta-llama/Llama-3.1-8B"
    finetuned_model: str = "meta-llama/Llama-3.1-8B-Instruct"

    base_model_device: str = "auto"
    finetuned_model_device: str = "cuda:0"
    finetuned_compressed_model_device: str = "cuda:1"

    split_memory_map: Dict[int, str] = field(default_factory=lambda: {1: "24GiB", 0: "20GiB"})
    base_model_memory_map: Optional[Dict[int, str]] = None
    finetuned_model_memory_map: Optional[Dict[int, str]] = None
    finetuned_compressed_model_memory_map: Optional[Dict[int, str]] = None

    dataset_name: str = "c4"
    subset: str = "en"
    max_length: int = 128
    batch_size: int = 1
    num_steps: int = 800
    lr: float = 5e-4

    save_dir: str = "outputs"
    debug: bool = True
    num_groups: int = 1
    save_full_model: bool = False

    def __post_init__(self):
        if self.base_model_memory_map is None:
            self.base_model_memory_map = self.split_memory_map
        if self.finetuned_model_memory_map is None:
            self.finetuned_model_memory_map = self.split_memory_map
