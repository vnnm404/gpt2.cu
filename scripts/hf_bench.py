#!/usr/bin/env python3
"""
Benchmark GPT-2 training step (forward + backward + weight update) on CUDA
- Configurable batch size and sequence length
- Randomly generated token inputs
- Reports latency in milliseconds
"""

import time

import torch
from transformers import GPT2LMHeadModel

SEQ_LEN = 64
BATCH_SIZE = 4
WARMUP = 10
ITERS = 100
VOCAB_SIZE = 50257  # GPT-2 vocabulary size


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")

    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True

    # Prepare input: randomly generated tokens
    input_ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN), dtype=torch.long, device=device)
    labels = input_ids.clone()  # For language modeling, labels are shifted internally

    # Load GPT-2 with LM head for training
    print("Loading GPT-2 with LM head...")
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    model.train()

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Warmup
    print(f"Warmup: {WARMUP} iterations")
    for _ in range(WARMUP):
        optimizer.zero_grad()
        outputs = model(input_ids, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()

    # Benchmark
    print(f"Running {ITERS} iterations...")
    times = []

    for _ in range(ITERS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        
        optimizer.zero_grad()
        outputs = model(input_ids, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)  # convert to ms

    import numpy as np
    times = np.array(times)
    mean_ms = times.mean()
    p50 = np.percentile(times, 50)
    p95 = np.percentile(times, 95)

    print("\n=== Results (Forward + Backward + Update) ===")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Seq length: {SEQ_LEN}")
    print(f"Mean latency: {mean_ms:.3f} ms")
    print(f"P50 latency: {p50:.3f} ms")
    print(f"P95 latency: {p95:.3f} ms")
    print("==============================================\n")


if __name__ == "__main__":
    main()
