import argparse
import torch
import yaml
from bitdelta_pipeline.args import Args
from bitdelta_pipeline.binary_variants import BinaryDiffCol, BinaryDiffRow
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
    
    for name, param in finetuned_compressed_model.named_parameters():
        if isinstance(param, BinaryDiffRow) or isinstance(param, BinaryDiffCol):
            param.requires_grad_(True)
        else:
            param.requires_grad_(False)

    lr = args.lr if args.lr is not None else 1e-5
    adam = torch.optim.AdamW(finetuned_compressed_model.parameters(), lr=lr)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(adam, T_max=args.num_steps if args.num_steps is not None else 5)

    for _ in range(5):
        step = 0
        for xb, yb in  dataloader:
            step += 1
            xb = xb.to(args.finetuned_compressed_model_device or 'cuda:0')
            yb = yb.to(args.finetuned_compressed_model_device or 'cuda:0')
            adam.zero_grad(set_to_none=True)
            yp = finetuned_compressed_model(xb)
            loss = torch.nn.functional.mse_loss(yp, yb)
            loss.backward()
            adam.step()
            cosine.step()

            if step >= (args.num_steps if args.num_steps is not None else 5):
                break
        print(f"Final training loss: {loss.item():.6f}")
    finetuned_compressed_model.save_pretrained(args.save_dir )
    tokenizer.save_pretrained(args.save_dir )

if __name__ == "__main__":
    main()
