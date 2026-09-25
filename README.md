# gpt2.cu: persistent GPT-2 training

GPT-2 forward, mean cross-entropy loss, backward, and AdamW execute in one
persistent CUDA kernel. Parameters, activations, gradients, and optimizer state
use **FP32**. GEMMs use ordinary SIMT FP32 arithmetic, not TF32 or mixed precision.

This branch experiments with **fine-grained RTX 3080 scheduling**. MLP forward
and backward instructions use tile dependencies, and residual addition plus
normalization use reusable shared-memory pages with asynchronous input loading.
See [the scheduling study, measurements, and limits](benchmarks/TILE_SCHEDULING.md).
Fine-grained execution is demonstrated; it should not be assumed faster for every
shape. The original grouped scheduler remains available for direct comparison.

The [earlier RTX 3080 benchmarks](benchmarks/README.md) describe commit `d1611e6`
and are historical measurements, not measurements of this branch.

## Run

On an NVIDIA CUDA development host:

```sh
uv sync
uv run python benchmarks/compare.py --steps 5 --iterations 50
```

The first run downloads pinned CUTLASS 3.5.1 headers and compiles the shared
library with nvcc. No pretrained weights or dataset download is required for
the deterministic comparison. [BUILD.md](BUILD.md) covers CMake, standalone
operation benchmarks, and synchronization validation.

## Implementation

- `gpt2_cuda/`: tensor ownership, explicit forward/backward program construction,
  dependency grouping, and the Python interface. PyTorch supplies storage and
  reference weights; it does not perform the custom training step.
- `src/executor.cu`: standalone and persistent executors, cooperative launch,
  occupancy checks, and a small C ABI.
- `include/gpt2/executor.h`: operation descriptors and host API.
- `include/gpt2/kernels/gemm.cuh`: inlined CUTLASS SIMT threadblock GEMMs with
  double buffering, reduction partitioning, and an L2-oriented tile order.
- `gpt2_cuda/schedule.py`: rectangular tile dependencies, shared prerequisite
  counters, and GPU task metadata for explicitly marked MLP regions.
- `include/gpt2/kernels/tile_schedule.cuh`: ready-task and static topological
  scheduling, with device-scoped acquire/release publication.
- `include/gpt2/kernels/page_pipeline.cuh`: two-page residual/normalization
  pipeline using loader/compute warp groups and Ampere barriers.
- `include/gpt2/kernels/sync.cuh`: scoped atomic and page-barrier primitives.
- `include/gpt2/kernels/attention.cuh`: causal attention and backward kernels;
  key/value gradients are gathered without float atomics.
- `include/gpt2/kernels/elementwise.cuh`: normalization, embeddings, residuals,
  GELU, stable cross-entropy, gradient reductions, and AdamW.
- `benchmarks/`: PyTorch comparisons, operation timings, and correctness checks.

All device functions are visible in one translation unit. No separately compiled
device functions or nested kernel launches sit between the scheduler and GEMMs.

Outside the marked MLP regions, the host groups independent operations and uses
cooperative grid barriers between groups. Inside an MLP region, workers publish
completed tiles and consumers wait only for their own prerequisites. A consumer
can execute while unrelated producer tiles remain unfinished. Region entry/exit
still uses grid synchronization. Shared-memory pages remain private to each CTA;
cross-CTA data is passed through global memory.

Worker count comes from actual SM count and compiled-kernel occupancy. No block
is assumed to have physical SM affinity, and oversized cooperative grids are
rejected. All workers participate in each remaining grid barrier.

The vocabulary is padded internally to 50,304 rows for layout alignment. Loss
and logits exposed to the caller still use the original 50,257-token vocabulary;
padding contributes neither probability mass nor gradients.

## Training interface

```python
from gpt2_cuda import Backend, Training

# reference is a float32 GPT model from scripts/llm.py, optionally loaded through
# GPT.from_pretrained("gpt2"). Tokens and explicit targets are contiguous CUDA int32.
training = Training(Backend(), reference, tokens, targets)
training.run()                       # one cooperative kernel launch
print(training.mean_loss.item())     # synchronize only when reporting
```

Shapes and tensor addresses are fixed when the program is constructed. Update
input buffers in place for subsequent batches. The implemented attention head
width is 64 and sequence lengths are bounded at 256; performance is tuned and
reported for GPT-2 124M at batches/sequences 4×64 and 8×128. The results do not establish parity
with `torch.compile`, CUDA Graphs, mixed-precision training, or other GPUs/shapes.

The original `src/mk.cu`, `src/train.cu`, layer code, and `tests/test_train*.cu`
remain available for historical comparison with `GPT2_BUILD_LEGACY=ON`. They are
not the default executor and retain their original assumptions and data formats.
