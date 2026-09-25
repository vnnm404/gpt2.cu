"""Reproducible full GPT-2 training baseline, without a model download."""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from llm import GPT, GPTConfig


def measure(step, warmup=5, iterations=20):
    for _ in range(warmup):
        step()
    torch.cuda.synchronize()
    device_ms, wall_ms = [], []
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for _ in range(iterations):
        t0 = time.perf_counter()
        start.record()
        step()
        end.record()
        end.synchronize()
        wall_ms.append((time.perf_counter() - t0) * 1000)
        device_ms.append(start.elapsed_time(end))
    return {"gpu_median_ms": statistics.median(device_ms),
            "wall_median_ms": statistics.median(wall_ms),
            "gpu_min_ms": min(device_ms), "iterations": iterations}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--tf32", action="store_true")
    parser.add_argument("--fused", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(123)
    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    model = GPT(GPTConfig()).cuda()
    x = torch.randint(50257, (4, 64), device="cuda")
    y = torch.randint(50257, (4, 64), device="cuda")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01,
                           betas=(0.9, 0.999), eps=1e-8, fused=args.fused)

    def step():
        opt.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        loss.backward()
        opt.step()

    result = {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
              "cuda": torch.version.cuda, "tf32": args.tf32,
              "optimizer_fused": args.fused, "batch": 4, "sequence": 64,
              "dtype": "float32", "dropout": 0, **measure(step, iterations=args.iterations)}
    print(json.dumps(result, indent=2))
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
