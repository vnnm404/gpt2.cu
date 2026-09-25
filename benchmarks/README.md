# RTX 3080 benchmark and validation

Recorded result: **25.08 ms** for the persistent step versus **32.74 ms** for
PyTorch eager with fused AdamW (**1.31×**, 50 timed iterations after warmup).
See [full results](rtx3080.json), [GEMM measurements](rtx3080-gemm.json), and
[validation output](validation.txt). The result contains a SHA-256 digest of
the executor, device headers, and Python runtime sources.

The previous implementation at `59f6ce9` measured 26.07 ms on the same rental;
see [fresh previous-version measurements](rtx3080-before.json). The retained
changes are 64×128 GEMM output tiles, vectorized AdamW memory accesses,
parallel row reductions for bias/LayerNorm gradients, and reuse of the linear
branch's residual gradient. The program now has 482 operations in 333 stages.
128×128 output tiles, a larger GEMM reduction tile, and more aggressive split-K
partitioning were tested and rejected because they did not improve total time.

## Mini training run

Three runs per implementation trained GPT-2 124M from the same random seed on
200 consecutive batches of GPT-2-tokenized
[Tiny Shakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt).
Each run processes 51,200 next-token targets at batch 4, sequence 64.

| Implementation | Median total training time |
| --- | ---: |
| PyTorch eager + fused AdamW | 6.398 s |
| Previous megakernel (`59f6ce9`) | 5.179 s |
| Optimized megakernel | 4.995 s |

The optimized version saves **1.403 seconds (21.9%)** against PyTorch and
**0.184 seconds (3.6%)** against the previous megakernel. This is **1.28×**
PyTorch throughput for the full loop; it differs from the individually
synchronized step benchmark above. Timing includes GPU batch copies, forward,
backward, AdamW, and recording every loss. It excludes downloading, tokenizing,
model/program construction, compilation, and five warmup steps. Weights and
optimizer state are reset after warmup. Batches are preloaded on the GPU;
there is no disk or CPU data-loader time inside the loop.

Loss falls from 10.9876 to 6.1006 in all implementations; the maximum absolute
loss difference between the optimized version and PyTorch across all 200 steps
and three runs is 0.0000172. This is a short correctness/performance experiment,
not a convergence or model-quality claim. [Raw run results](rtx3080-training.json)
include every loss, timings, source and corpus hashes, environment versions,
and run order. The reference uses PyTorch 2.5.1 with CUDA 12.4.

```sh
curl -L https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt -o /tmp/shakespeare.txt
uv run --with tiktoken python benchmarks/train.py --text /tmp/shakespeare.txt --mode pytorch --steps 200 --output /tmp/pytorch-training.json
uv run --with tiktoken python benchmarks/train.py --text /tmp/shakespeare.txt --mode persistent --steps 200 --output /tmp/cuda-training.json
```

## Methodology

The target is GPT-2 124M, batch 4, sequence length 64, 12 layers, 12 heads, 768
channels, and 50,257 vocabulary entries. Parameters, activations, gradients and
AdamW moments are float32. TF32 and dropout are disabled. The reference is the
repository's PyTorch GPT model with fused AdamW: lr=1e-4, betas=(0.9,0.999),
eps=1e-8, weight decay=0.01. Both sides receive the same initialization, inputs
and explicit targets. Random initialization avoids a network dependency.

```sh
python benchmarks/compare.py --steps 5 --iterations 50 --output benchmarks/rtx3080.json
python benchmarks/compare.py --standalone
python benchmarks/gemm.py
python benchmarks/check_runtime.py
```

`compare.py` checks loss, every logit, every parameter gradient and every updated
parameter before timing. It reports gradient relative RMS errors, GPU-event
median time, and synchronized wall time after warmup. Timing includes forward,
mean loss, backward and optimizer updates. Model loading, allocation and host
input transfers are excluded from both implementations. The reference is eager
PyTorch with fused AdamW, not torch.compile or CUDA Graph capture.

Gradient comparisons use atol=3e-5 and rtol=3e-3, plus a relative RMS bound of
2e-3 (or max absolute error below 2e-7). Parameter comparisons use atol=1e-4 and
rtol=3e-4: AdamW can amplify differences of a few e-9 in nearly zero gradients.
The separate optimizer test compares parameters and both moments against fused
PyTorch AdamW for ten steps with identical gradients and tighter tolerances.

The scheduler checks cover odd matrix dimensions, repeated dependency chains,
1/3/68/136 worker launches on the measured GPU, idle workers, and rejection of
oversubscribed launches. Full small-model training is checked at sequence
lengths 1, 17, 64, and 256, including vocabulary padding and partial warps.
Compute Sanitizer synccheck and memcheck both report zero errors. Run GPU
development tests under a process timeout.

The new executor uses CUTLASS 3.5.1 SIMT threadblock implementations, compiled
inline with all other operations. A tinygrad checkout at
`91dc3a04fc748b4b3f527f0a5f4864931a95beec` was evaluated; its tested FP32 GEMM
was slower than PyTorch at our shape, so no tinygrad code was incorporated.
Single-product TF32 and mixed-precision arithmetic are not used.

Initial audit of the original code found hardcoded worker counts, volatile
counter polling without an explicit publication protocol, fixed activation
offsets despite runtime dimensions, and a conventional cross-entropy backward
launch with fewer blocks than its implementation required. The replacement uses
cooperative residency checks and grid synchronization instead of those counters.
