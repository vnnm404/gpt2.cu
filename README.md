# gpt2.cu: persistent GPT-2 training

GPT-2 forward, mean cross-entropy loss, backward, and AdamW execute in one
persistent CUDA kernel. Parameters, activations, gradients, and optimizer state
use **FP32**. GEMMs use ordinary SIMT FP32 arithmetic, not TF32 or mixed precision.

The rewritten executor reaches approximately **26 ms per step**, versus **33 ms**
for the matching PyTorch eager model with fused AdamW on an RTX 3080, at batch 4
and sequence length 64. See [benchmark methodology](benchmarks/README.md) and the
recorded measurements in `benchmarks/rtx3080.json` for exact results and scope.

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
- `include/gpt2/kernels/attention.cuh`: causal attention and backward kernels;
  key/value gradients are gathered without float atomics.
- `include/gpt2/kernels/elementwise.cuh`: normalization, embeddings, residuals,
  GELU, stable cross-entropy, gradient reductions, and AdamW.
- `benchmarks/`: PyTorch comparisons, operation timings, and correctness checks.

All device functions are visible in one translation unit. No separately compiled
device functions or nested kernel launches sit between the scheduler and GEMMs.

The host groups operations only when their memory ranges are independent.
Resident GPU workers share each group's tile space, allowing independent
backward operations to overlap. Cooperative grid barriers publish completed
writes between dependent groups. Worker count comes from actual SM count and
compiled-kernel occupancy; a block is not assumed to have physical SM affinity.
Even workers with no work participate in every grid barrier.

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
reported for batch 4, sequence 64, GPT-2 124M. The results do not establish parity
with `torch.compile`, CUDA Graphs, mixed-precision training, or other GPUs/shapes.

The original `src/mk.cu`, `src/train.cu`, layer code, and `tests/test_train*.cu`
remain available for historical comparison with `GPT2_BUILD_LEGACY=ON`. They are
not the default executor and retain their original assumptions and data formats.
