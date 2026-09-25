# Build and validation

The executor requires an NVIDIA CUDA development environment, Python, PyTorch,
and a C++17 compiler. `pyproject.toml` specifies Python 3.12; the measured GPU
environment used Python 3.11, PyTorch 2.5.1, CUDA 12.4, and an RTX 3080.

```sh
uv sync
uv run python benchmarks/compare.py --steps 5 --iterations 50
```

The loader fetches pinned CUTLASS 3.5.1 into `build/cutlass` and compiles
`build/libgpt2_executor.so` with nvcc. All device code is compiled in one
translation unit, with FP32 arithmetic and without `--use_fast_math`.

An explicit CMake build produces the same library:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build -j
```

`CUTLASS_ROOT` can point to an existing checkout. The Python loader expects the
pinned checkout at `build/cutlass` and verifies its revision before building.
To also compile the original library and drivers, configure with
`-DGPT2_BUILD_LEGACY=ON`. Those drivers retain their original binary model inputs.

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

## Fine-grained RTX 3080 experiment

The build remains `sm_86` with one translation unit and FP32 SIMT GEMMs.
No Hopper instructions are required. See [the scheduling report](benchmarks/TILE_SCHEDULING.md)
for sanitizer versions, exact synchronization, traces, and measured tradeoffs.

```sh
python benchmarks/check_tile_schedule.py
python benchmarks/compare.py --batch 8 --sequence 128 --steps 5 --iterations 50
TILE_SCHEDULE=0 PAGE_PIPELINE=0 python benchmarks/compare.py --batch 8 --sequence 128
PAGE_ROWS=8 python benchmarks/profile_tile_schedule.py
python benchmarks/plot_tile_schedule.py  # optional matplotlib dependency
```

`TILE_ENGINE=static` selects topological worker streams; the default ready queue
assigns only runnable tasks. `PAGE_ENGINE=collective` selects the all-thread
async-copy implementation; the default uses loader/compute warp groups and
Ampere page barriers. These are separate ablations, not different precision modes.
