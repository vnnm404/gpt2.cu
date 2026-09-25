# Build and validation

The executor requires an NVIDIA CUDA development environment, Python, PyTorch,
and a C++17 compiler. `pyproject.toml` specifies Python 3.12; the measured GPU
environment used Python 3.11, PyTorch 2.5.1, CUDA 12.4, and an H100 80GB HBM3 (previous RTX 3080 results are historical).

```sh
uv sync
uv run python benchmarks/compare.py --steps 5 --iterations 50
```

The loader fetches pinned CUTLASS 3.5.1 into `build/cutlass` and compiles
an architecture-specific library, e.g. `build/libgpt2_executor_sm_90a.so` with nvcc. All device code is compiled in one
translation unit without global `--use_fast_math`. H100 uses the six-product
BF16 decomposition described in [H100.md](benchmarks/H100.md); other targets use
SIMT FP32. Hopper cross-entropy explicitly uses `__expf`.

An explicit CMake build produces the same library:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=90a
cmake --build build -j
```

`CUTLASS_ROOT` can point to an existing checkout. The Python loader expects the
pinned checkout at `build/cutlass` and verifies its revision before building.

Run validation on the GPU host:

```sh
timeout 180 python benchmarks/check_runtime.py
timeout 240 compute-sanitizer --tool synccheck --error-exitcode 99 python benchmarks/check_runtime.py
timeout 240 python benchmarks/gemm.py
timeout 300 python benchmarks/compare.py --steps 5 --iterations 50 --output benchmark.json
```

Use `--standalone` to launch the same operations individually. Set `GROUP=0`
to disable grouping of independent operations. `--workers N` selects a checked
worker count; the default comes from compiled-kernel occupancy and the device.

Use a process timeout during kernel development. The launcher rejects an
oversized cooperative grid before dispatch, and all workers participate in
each stage's grid barrier, including workers with no assigned tiles.

Use architecture `86` for the RTX 3080 SIMT path. H100 requires `90a`, CUDA 12,
and the CUDA driver library for tensor-map encoding. CMake emits
`libgpt2_executor.so`; the Python loader maintains its own architecture cache.
The executor opts into 112 KiB shared memory per CTA and queries actual occupancy.
GEMM output workspaces must fit signed 32-bit indexing; packed plane offsets use
64-bit arithmetic.

Hopper-specific validation:

```sh
python benchmarks/check_hopper.py
for tool in synccheck memcheck racecheck; do
  timeout 300 compute-sanitizer --tool "$tool" --error-exitcode 99 python benchmarks/check_hopper.py
done
python benchmarks/gemm.py --tokens 8192
python benchmarks/compare.py --isolated --batch 128 --sequence 128 --steps 5 --iterations 50
```

`--isolated` runs each implementation in a separate process and validates against
CPU/disk snapshots. It avoids keeping both engines in GPU memory simultaneously;
it preserves the actual physical batch size. Allow roughly 20 GB temporary disk
space for five batch-128 checkpoints. Snapshot I/O is outside benchmark timing.
