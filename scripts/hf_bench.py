#!/usr/bin/env python3
"""
Benchmark GPT-2 (no LM head) on CUDA
- Batch size = 1
- Sequence length = 80 (cyclically repeated from provided tokens)
- Reports latency in milliseconds
"""

import time
from itertools import islice, cycle

import torch
from transformers import AutoModel

# Original user-provided token list (70 tokens)
base_tokens = [
    464, 3139, 286, 4881, 318,
    464, 3139, 286, 4881, 318,
    464, 3139, 286, 4881, 318, 464, 3139, 286, 4881, 318,
    464, 3139, 286, 4881, 318, 464, 3139, 286, 4881, 318,
    464, 3139, 286, 4881, 318, 464, 3139, 286, 4881, 318,
    464, 3139, 286, 4881, 318, 464, 3139, 286, 4881, 318,
    464, 3139, 286, 4881, 318, 464, 3139, 286, 4881, 318,
    464, 3139, 286, 4881, 318, 14350, 1747, 1244, 1011, 674
]

SEQ_LEN = 80
BATCH_SIZE = 1
WARMUP = 10
ITERS = 100


def make_seq_of_length(tokens, length):
    return list(islice(cycle(tokens), length))


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")

    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True

    # Prepare input
    seq_tokens = make_seq_of_length(base_tokens, SEQ_LEN)
    input_ids = torch.tensor([seq_tokens], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)

    # Load GPT-2 base (no LM head)
    print("Loading GPT-2 base (no LM head)...")
    model = AutoModel.from_pretrained("gpt2").to(device)
    model.eval()

    # Warmup
    print(f"Warmup: {WARMUP} iterations")
    with torch.no_grad():
        for _ in range(WARMUP):
            _ = model(input_ids, attention_mask=attention_mask)
            torch.cuda.synchronize()

    # Benchmark
    print(f"Running {ITERS} iterations...")
    times = []

    with torch.no_grad():
        for _ in range(ITERS):
            t0 = time.perf_counter()
            _ = model(input_ids, attention_mask=attention_mask)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)  # convert to ms

    import numpy as np
    times = np.array(times)
    mean_ms = times.mean()
    p50 = np.percentile(times, 50)
    p95 = np.percentile(times, 95)

    print("\n=== Results (Batch = 1) ===")
    print(f"Seq length: {SEQ_LEN}")
    print(f"Mean latency: {mean_ms:.3f} ms")
    print(f"P50 latency: {p50:.3f} ms")
    print(f"P95 latency: {p95:.3f} ms")
    print("==========================\n")


if __name__ == "__main__":
    main()
