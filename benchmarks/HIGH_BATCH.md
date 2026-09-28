# High-batch RTX 3080 training

This revision optimizes complete GPT-2 124M training at batch 8 and batch 16,
sequence length 128. All tensors and arithmetic remain FP32, with TF32 disabled.
The comparison uses PyTorch 2.5.1+cu124 Inductor forward/backward and eager fused
AdamW. These results do not cover mixed precision, other GPUs, or arbitrary shapes.

## Measured results — September 27, 2026

RTX 3080 **12 GB**, 70 SMs, 280 W cap, driver 580.126.09. This is a different
3080 variant from the earlier 10 GB / 68-SM / 250 W report; compare the paired
measurements here. The current kernel runs 210 workers (three per SM), each with
128 threads and 32 KiB of dynamic shared memory.

| Batch × sequence | Reference mode | PyTorch ms/step | Megakernel ms/step | Throughput ratio |
|---|---|---:|---:|---:|
| 16 × 128 | default | 125.32 | 121.08 | 1.035× |
| 16 × 128 | reduce-overhead | 126.21 | 121.55 | 1.038× |
| 8 × 128 | default | 71.04 | 66.55 | 1.067× |
| 8 × 128 | reduce-overhead | 71.59 | 67.10 | 1.067× |

At batch 16 (2,048 tokens/step), measured throughput exceeds the two compiled
reference modes by **3.5–3.8%**. At batch 8 it exceeds them by **6.7%**. These are
steady-state complete training steps, not isolated GEMM timings. Raw results,
per-round timings, checks, and environment details are in [high-batch/](high-batch/).

### Change from the previous executor

The pre-change executor at `0a39b4d` was rerun on the same rental. Its benchmark
uses the same bounded-memory validation helper so batch 16 fits; kernel/runtime
sources and the pre-change CUDA binary are unchanged.

| Batch × sequence | Previous megakernel | Current megakernel | Throughput improvement |
|---|---:|---:|---:|
| 16 × 128 | 151.31 ms | 121.08 ms | 25.0% |
| 8 × 128 | 81.81 ms | 66.55 ms | 22.9% |

These are separate matched runs on the same host, not timings copied from the
earlier 10 GB card.

### Matched 200-step text training

Batch 16 × sequence 128, 409,600 training tokens, one run per implementation:

| Implementation | Training-loop wall time | Tokens/second |
|---|---:|---:|
| Compile default | 24.94 s | 16,424 |
| Compile reduce-overhead | 25.07 s | 16,340 |
| Megakernel | 24.27 s | 16,875 |

The megakernel saves **0.67 s** against default and **0.80 s** against
reduce-overhead, for **2.7–3.3% higher throughput**. These totals exclude setup,
compilation and warmup. Maximum loss difference over all 200 steps is
**1.574×10⁻⁵**. Final loss is 5.333682 (megakernel) versus 5.333674 (both compiled
modes). The mini run is a single observation; the step benchmark above has three
alternating measurement rounds.

The core-source SHA-256 recorded by every current result is
`e94ca0ed013c87e6901f99859cb7a0a5e3638677bc35f33bc3c6b19a00925a6f`.
It hashes the executor, C ABI header, kernel headers and Python runtime modules.

## Implementation

Workers now use 128 threads, a 64×128 output tile, and 64×32 warp tiles. Compiled
occupancy determines the resident worker count; it is not a hardcoded SM count.
Aligned GEMMs load four floats per global access and transpose into the SIMT
shared-memory layout when needed. Odd dimensions and unaligned inputs use the
scalar path. Both paths share CUTLASS's pipelined mainloop and epilogue.

Attention now computes score and value/query products in tiles. A worker applies
softmax to its own 64 query rows, then consumes those rows without a grid barrier
between those products. Key/value backward waits for the complete score-gradient
stage, since those products reduce over all queries. The previous scalar
warp-per-query implementation has been replaced.

Split-K uses fewer partitions for the long vocabulary reduction, avoiding excess
partial-output traffic. The MLP scheduler retains tile-level dependencies and
cross-instruction overlap. Residual/normalization uses two shared-memory pages
with two loader warps and two compute warps. The scheduler and all device code
remain in one compilation unit.

The half-CTA page groups use `barrier.sync` without `.aligned`. The old `bar.sync`
spelling implies CTA-uniform control flow and is inappropriate in these two
branches; see the [PTX barrier specification](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-bar).
Page ownership uses complementary `barrier.arrive`/`barrier.sync` operations
from all loader and compute threads. The producer drains the final release
phases before the CTA returns. Barrier IDs are compile-time constants: an
experimental register-valued version stalled at full cooperative-grid capacity,
consistent with the [named-barrier residency restriction](https://forums.developer.nvidia.com/t/named-barriers-limit-occupancy/158005)
reported by NVIDIA. This replaces the leader-only mbarrier parity
protocol and removes its helper. The page regression follows normalization
with GEMM to check handoff of the
shared pool, in addition to repeated page generations.

## Reproduction

```sh
export TORCHINDUCTOR_COMPILE_THREADS=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python benchmarks/compare.py --batch 16 --sequence 128 --steps 5 --iterations 50 --repeats 3 --compile-mode default --output default-b16.json
python benchmarks/compare.py --batch 16 --sequence 128 --steps 5 --iterations 50 --repeats 3 --compile-mode reduce-overhead --output graphs-b16.json
```

Repeat with `--batch 8`. Each run validates loss, logits, every parameter gradient,
and AdamW parameter updates for five steps before timing. Numerical tolerances
are unchanged. Validation processes large tensors in bounded chunks to fit both
implementations in memory; this does not alter the timed workload. Timings are
medians of three rounds of 50 steps, with measurement order reversed in alternate
rounds. Setup, compilation, warmup, and validation are excluded.

For the 200-step text-training comparison:

```sh
python benchmarks/train.py --text shakespeare.txt --mode pytorch --compile-mode default --batch 16 --sequence 128 --steps 200 --output train-default.json
python benchmarks/train.py --text shakespeare.txt --mode pytorch --compile-mode reduce-overhead --batch 16 --sequence 128 --steps 200 --output train-graphs.json
python benchmarks/train.py --text shakespeare.txt --mode persistent --batch 16 --sequence 128 --steps 200 --output train-persistent.json
```

The input is [Tiny Shakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt),
GPT-2 tokenized. All runs use random initialization with seed 123 and the same
contiguous next-token batches, wrapping the corpus when necessary. Five warmup
steps are followed by a complete weight/optimizer reset. Timed training includes
batch copies, forward, backward, AdamW, loss recording, and final synchronization;
setup, tokenization, and compilation are excluded. This is a timing and trajectory
comparison, not a model-quality evaluation.

## Validation method

The validation suite uses NVIDIA Compute Sanitizer 2026.1.1 (CUDA 13.2 package),
without device-error suppressions. Memcheck uses `--report-api-errors explicit`:
the new default extended reporting flags CUDA 12.4 internal helper-symbol
lookups as missing kernels. Device memory checking remains enabled; runtime
memcheck is also cross-checked with version 2025.3.1. `check_runtime.py` tests
idle/active worker transitions,
rejected oversized cooperative launches, aligned and unaligned AdamW parameters
and moments, and complete small-model training at sequence lengths 1/17/64/256.
`check_gemm.py` covers all transpose combinations, vector and scalar paths,
partial tiles, unaligned operands, split-K, bias, and accumulating epilogues.
`check_tile_schedule.py` tests prerequisite publication and cross-instruction
execution in both ready and static schedulers, plus repeated pages followed by
GEMM at 1, 3, and full-capacity worker counts.

```sh
for tool in memcheck racecheck synccheck; do
    timeout 600 compute-sanitizer --tool "$tool" --report-api-errors explicit --error-exitcode 99 python benchmarks/check_runtime.py
    timeout 600 compute-sanitizer --tool "$tool" --report-api-errors explicit --error-exitcode 99 python benchmarks/check_gemm.py
    for engine in ready static; do
        TILE_ENGINE="$engine" timeout 600 compute-sanitizer --tool "$tool" --report-api-errors explicit --error-exitcode 99 python benchmarks/check_tile_schedule.py --quick
    done
done
```

The full numerical comparison also runs with the static scheduler. CMake and the
Python-triggered nvcc build both compile the same single translation unit.

All twelve sanitizer suite/tool combinations above finished with zero reported
device errors or hazards. The additional 2025.3.1 runtime memcheck also passed.
Logs are included alongside the raw timing results. The static-scheduler full
training comparison and collective-page correctness check passed as well.

The rental was destroyed after collecting the artifacts. At the advertised GPU
plus storage rate, 1.283 hours cost approximately **$0.19**, excluding any separate
network charges; this is an estimate, not a billing invoice.
