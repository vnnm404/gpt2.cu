# gpt2.cu: Megakernels for Model Training

**Can an entire GPT-2 training step run inside a single GPU kernel?**

We extend the megakernel execution model from inference to training, executing
GPT-2's forward pass, backward pass, and AdamW update inside one persistent
CUDA kernel.

The implementation contains 35 custom operations and an interpreter-style
device scheduler that explicitly manages dependencies, synchronization, and
work distribution across SMs.

Unlike conventional training, where kernel boundaries provide global
synchronization implicitly, the megakernel must implement those dependencies
explicitly on the GPU.

### Highlights

- Full GPT-2 forward + backward + optimizer step in one persistent kernel
- 35 CUDA operations covering attention, MLPs, LayerNorm, embeddings, and AdamW
- Explicit device-side scheduling and synchronization instead of host-driven kernel launches
- Correctness validated against PyTorch and by fine-tuning on Tiny Shakespeare

## Setup

```bash
git clone https://github.com/vnnm404/gpt2.cu.git && cd gpt2.cu

uv sync
uv run python scripts/hf_to_bin.py
```

### Build + Run

```bash
# configure. (optionally, set your arch as needed eg -DCMAKE_CUDA_ARCHITECTURES="86")
cmake -B build
# build
cmake --build build -j
```

```bash
./build/programs/inference
```

### Tooling

Generate `.clangd`:

```bash
uv run scripts/generate_clangd.py
```

For your LSP to cooperate you should make sure `compile_commands.json` exists (`cmake -B build` at minimum) so clangd sees per-target flags, architectures, includes, etc.

## TODO

- [ ] Clean up code
- [ ] Better library structure
- [ ] Checking for CUDA errors
- [x] Better build and setup scripts
