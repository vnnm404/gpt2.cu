# RTX 3080 benchmark and validation

Recorded result: **25.86 ms** for the persistent step versus **32.48 ms** for
PyTorch eager with fused AdamW (**1.26×**, 50 timed iterations after warmup).
See [full results](rtx3080.json), [GEMM measurements](rtx3080-gemm.json), and
[validation output](validation.txt). The result contains a SHA-256 digest of
the executor, device headers, and Python runtime sources.

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
