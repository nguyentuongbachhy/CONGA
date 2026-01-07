import argparse
from dataclasses import dataclass
from typing import Any, Tuple

import torch

from model import SASRec
from utils import (
    check_and_convert_dataset,
    load_metadata,
    data_partition,
    evaluate,
    evaluate_valid,
)


def str2bool(s: str) -> bool:
    if s not in {"false", "true"}:
        raise ValueError("Not a valid boolean string")
    return s == "true"


@dataclass
class ModelArgs:
    device: str
    maxlen: int
    hidden_units: int
    num_blocks: int
    num_heads: int
    dropout_rate: float
    norm_first: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained SASRec checkpoint")

    parser.add_argument("--dataset", required=True, type=str)
    parser.add_argument("--checkpoint", required=True, type=str, help="Path to .pth checkpoint")
    parser.add_argument("--device", default="cuda", type=str)

    parser.add_argument("--maxlen", default=200, type=int)
    parser.add_argument("--hidden_units", default=50, type=int)
    parser.add_argument("--num_blocks", default=2, type=int)
    parser.add_argument("--num_heads", default=1, type=int)
    parser.add_argument("--dropout_rate", default=0.2, type=float)
    parser.add_argument("--norm_first", action="store_true", default=False)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    check_and_convert_dataset(args.dataset)
    usernum, itemnum = load_metadata(args.dataset)

    model_args: Any = ModelArgs(
        device=args.device,
        maxlen=args.maxlen,
        hidden_units=args.hidden_units,
        num_blocks=args.num_blocks,
        num_heads=args.num_heads,
        dropout_rate=args.dropout_rate,
        norm_first=args.norm_first,
    )

    model: SASRec = SASRec(usernum, itemnum, model_args).to(args.device)
    state = torch.load(args.checkpoint, map_location=torch.device(args.device))
    model.load_state_dict(state)
    model.eval()

    dataset = data_partition(args.dataset)

    with torch.no_grad():
        with torch.amp.autocast_mode.autocast(device_type="cuda", enabled=False):
            v: Tuple[float, float] = evaluate_valid(model, dataset, args)
            t: Tuple[float, float] = evaluate(model, dataset, args)

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Valid - NDCG@10: {v[0]:.4f}, HR@10: {v[1]:.4f}")
    print(f"Test  - NDCG@10: {t[0]:.4f}, HR@10: {t[1]:.4f}")


if __name__ == "__main__":
    main()
