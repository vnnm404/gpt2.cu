"""Matched mini training run; tokenization/setup and warmup are outside timing."""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gpt2_cuda import Backend, Training
from reference import GPT, GPTConfig


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--text', type=Path, required=True)
    p.add_argument('--mode', choices=['pytorch', 'persistent'], required=True)
    p.add_argument('--steps', type=int, default=200)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    import tiktoken
    torch.set_num_threads(1)
    torch.manual_seed(123)
    torch.backends.cuda.matmul.allow_tf32 = False
    raw = args.text.read_bytes()
    ids = tiktoken.get_encoding('gpt2').encode(raw.decode('utf-8'))
    assert len(ids) > 257 and args.steps > 0
    # Preload the same contiguous next-token batches for both implementations.
    corpus = torch.tensor(ids, dtype=torch.int32, device='cuda')
    offsets = torch.arange(args.steps * 256 + 1, device='cuda') % len(ids)
    stream = corpus[offsets]
    xs = stream[:-1].view(args.steps, 4, 64).contiguous()
    ys = stream[1:].view(args.steps, 4, 64).contiguous()
    model = GPT(GPTConfig()).cuda()
    initial = {k: v.detach().clone() for k, v in model.state_dict().items()}
    losses = torch.empty(args.steps, device='cuda')
    if args.mode == 'persistent':
        x, y = xs[0].clone(), ys[0].clone()
        train = Training(Backend(), model, x, y)
        def step(i):
            x.copy_(xs[i])
            y.copy_(ys[i])
            train.run()
            losses[i].copy_(train.mean_loss)
        reset = train.reset
    else:
        xs, ys = xs.long(), ys.long()
        x, y = xs[0].clone(), ys[0].clone()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=.01,
                               betas=(.9, .999), eps=1e-8, fused=True)
        def step(i):
            x.copy_(xs[i])
            y.copy_(ys[i])
            opt.zero_grad(set_to_none=True)
            _, loss = model(x, y)
            loss.backward()
            opt.step()
            losses[i].copy_(loss.detach())
        def reset():
            model.load_state_dict(initial)
            for state in opt.state.values():
                for value in state.values():
                    if isinstance(value, torch.Tensor):
                        value.zero_()
    for _ in range(5):
        step(0)
    reset()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    begin = time.perf_counter()
    start.record()
    for i in range(args.steps):
        step(i)
    end.record()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - begin
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted([root / 'src/executor.cu', root / 'include/gpt2/executor.h',
                        root / 'gpt2_cuda/__init__.py', *root.glob('include/gpt2/kernels/*.cuh')]):
        digest.update(path.read_bytes())
    result = dict(source_sha256=digest.hexdigest(), tokenizer_version=tiktoken.__version__,
                  cuda_version=torch.version.cuda, mode=args.mode, steps=args.steps, batch=4, sequence=64,
                  gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                  corpus_sha256=hashlib.sha256(raw).hexdigest(), corpus_tokens=len(ids),
                  initialization='random seed 123, GPT-2 124M', dtype='float32', tf32=False,
                  wall_seconds=elapsed, gpu_seconds=start.elapsed_time(end)/1000,
                  tokens_per_second=args.steps*256/elapsed, losses=losses.tolist(),
                  timing='Includes batch copies, forward, backward, AdamW, loss recording; excludes setup and five reset warmup steps')
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'losses'}, indent=2))
    print('First/last loss:', result['losses'][0], result['losses'][-1])


if __name__ == '__main__':
    main()
