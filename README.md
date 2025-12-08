# gpt2.cu

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
