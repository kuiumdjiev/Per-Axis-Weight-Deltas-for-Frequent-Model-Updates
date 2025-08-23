import argparse
import yaml
from bitdelta_pipeline.args import Args
from bitdelta_pipeline.pipeline import run_pipeline


def parse_args():
    p = argparse.ArgumentParser(description="Run BitDelta compression pipeline")
    p.add_argument("--config", type=str, default="../configs/default.yaml", help="Path to YAML config")
    return p.parse_args()


def main():
    args_ns = parse_args()
    with open(args_ns.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    args = Args(**cfg)
    run_pipeline(args)


if __name__ == "__main__":
    main()
