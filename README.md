# gpt2.cu

### Build

```
git clone https://github.com/vnnm404/gpt2.cu.git

curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

cd gpt2.cu
mkdir models

uv venv
source .venv/bin/activate
uv pip install torch transformers
uv run python scripts/hf_to_bin.py

mkdir build
cd build
cmake ..
make inference -j `nproc` && ./programs/inference
```

### TODO

- [ ] Clean up code
- [ ] Better library structure
- [ ] Checking for CUDA errors
- [ ] Better build and setup scripts
