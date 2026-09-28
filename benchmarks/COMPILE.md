# RTX 3080 comparison with torch.compile

Historical report for `0a39b4d`. See [HIGH_BATCH.md](HIGH_BATCH.md) for the current optimized executor.

`compare.py --compile-mode MODE` compiles the reference model's forward with
Inductor and its backward through AOTAutograd. `fullgraph=True` rejects forward
graph breaks, and shapes are static. The same fused AdamW optimizer remains
eager in both reference modes. Timings include forward, loss, backward, gradient
clearing, and optimizer updates; they do not include numerical validation.

All parameters, activations, and optimizer state are FP32, with TF32 disabled.
The custom executor and reference start from the same weights and inputs. Loss,
logits, every gradient, and updated parameters are checked before timing using
the existing tolerances. Compiler failures and numerical failures are not
silently replaced with eager results.

```sh
python benchmarks/compare.py --batch 8 --sequence 128 --steps 5 --iterations 50 --repeats 3 --output eager-large.json
python benchmarks/compare.py --batch 8 --sequence 128 --steps 5 --iterations 50 --repeats 3 --compile-mode default --output compile-large.json
python benchmarks/compare.py --batch 8 --sequence 128 --steps 5 --iterations 50 --repeats 3 --compile-mode reduce-overhead --output compile-graphs-large.json
```

Repeat with `--batch 4 --sequence 64` for the smaller workload. The default custom
configuration is the ready tile scheduler plus page pipelines. For the coarse
scheduler plus pages, set `TILE_SCHEDULE=0`; to also disable pages, set
`PAGE_PIPELINE=0`.

Each timing round warms up both paths and reverses their measurement order in
alternate rounds. Reported medians aggregate three rounds of 50 steps. CUDA
and synchronized wall-clock step times are recorded. `first_reference_step_seconds`
is the synchronized first complete reference step, including lazy compilation,
backward compilation, initialization, and execution; it is **not** an isolated
compiler-time measurement. `compile_wrapper_seconds` records wrapper creation.
Neither is included in steady-state timing. Disk compiler caches can affect
first-call latency; use separate empty cache directories for cold-start runs.

`reduce-overhead` requests CUDA graph optimizations where supported; selecting
that mode alone does not establish that every operation is captured. See the
[PyTorch compile API](https://docs.pytorch.org/docs/stable/generated/torch.compile.html).

## Measured results, September 27, 2026

RTX 3080 10 GB, 250 W cap, PyTorch 2.5.1+cu124, 12-layer GPT-2 124M.
GPU step times below are medians of three 50-step rounds. Each row is a separate
matched run; use its paired custom timing when calculating a speed ratio.

| Batch × sequence | PyTorch execution | PyTorch | Tile/page megakernel | Faster path |
|---|---|---:|---:|---|
| 8 × 128 | Eager | 86.22 ms | 85.36 ms | Custom, 1.01× |
| 8 × 128 | Compile default | **74.87 ms** | 85.33 ms | PyTorch, **1.14×** |
| 8 × 128 | Compile reduce-overhead | 74.94 ms | 85.08 ms | PyTorch, 1.14× |
| 4 × 64 | Eager | 38.46 ms | 27.94 ms | Custom, 1.38× |
| 4 × 64 | Compile default | 31.48 ms | 27.74 ms | Custom, 1.13× |
| 4 × 64 | Compile reduce-overhead | **31.11 ms** | 27.76 ms | Custom, **1.12×** |

The larger training workload favors compiled PyTorch. Its default mode reduces
step time by about 13% compared with the eager reference on this host. Default
and reduce-overhead modes are effectively tied there. The custom executor still
wins the smaller workload, but its margin shrinks substantially against compile.
These are FP32 comparisons; no mixed-precision or TF32 results are implied.

Five complete training steps passed the existing loss/logit/gradient/parameter
checks in **all six runs**. No tolerances were loosened. First compiled steps
took 21.24 s (default large), 21.32 s (reduce-overhead large), 20.91 s (default
small), and 21.11 s (reduce-overhead small). These include compilation and runtime
initialization, and are excluded from the table. Each compiled run used a
separate initially absent Inductor cache directory. Steady-state synchronized
wall times were within 0.04 ms of the GPU medians.

A subsequent same-host large-workload ablation measured coarse grouping at
84.49 ms, coarse plus pages at 84.18 ms, ready tiles plus pages at 86.12 ms, and
static tiles plus pages at 85.43 ms. Even the best measured custom configuration
remains slower than compiled PyTorch. These were sequential experiments; small
differences between their custom timings should not be overinterpreted.

[Raw results and logs](torch-compile/) retain every timing round and validation
summary. The [environment manifest](torch-compile/environment.json) records
versions, source context, benchmark hash, rental estimate, and confirmed instance
destruction. Rental cost was approximately **$0.03**, excluding transfer charges.
