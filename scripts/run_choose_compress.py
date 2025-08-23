import argparse
import yaml
from bitdelta_pipeline.args import Args
from bitdelta_pipeline.models import load_models
from bitdelta_pipeline.data import build_dataloaders
from bitdelta_pipeline.choose_compress import compress_model_choosing


def parse_args():
    p = argparse.ArgumentParser(description="Choose row/col BinaryDiff per layer and apply it to a copy of the finetuned model")
    p.add_argument("--config", type=str, default="../configs/default.yaml", help="Path to YAML config")
    return p.parse_args()


def main():
    args_ns = parse_args()
    with open(args_ns.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    args = Args(**cfg)

    # Load models and data
    tokenizer, finetuned_model, finetuned_compressed_model = load_models(args)
    dataloader, val_dataloader = build_dataloaders(tokenizer, args.max_length)

    # Run chooser
    compress_model_choosing(
        args.base_model,
        finetuned_model,
        finetuned_compressed_model,
        dataloader,
        val_dataloader,
        ft_device=args.finetuned_model_device,
        comp_device=args.finetuned_compressed_model_device or 'cuda:0',
    )


if __name__ == "__main__":
    main()
