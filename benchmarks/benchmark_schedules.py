"""Interleaved scheduler ablations sharing exactly the same model/storage."""
import argparse
import hashlib
import json
import os
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gpt2_cuda import Backend, Training
from reference import GPT, GPTConfig, measure

p = argparse.ArgumentParser()
p.add_argument('--batch', type=int, default=8)
p.add_argument('--sequence', type=int, default=128)
p.add_argument('--iterations', type=int, default=50)
p.add_argument('--repeats', type=int, default=3)
p.add_argument('--output', type=Path, required=True)
args = p.parse_args()
torch.set_num_threads(1)
torch.manual_seed(123)
torch.backends.cuda.matmul.allow_tf32 = False
os.environ['TILE_TRACE'] = '0'
os.environ['TILE_SCHEDULE'] = '0'
os.environ['PAGE_PIPELINE'] = '0'
model = GPT(GPTConfig()).cuda()
x, y = [torch.randint(50257, (args.batch, args.sequence), device='cuda', dtype=torch.int32) for _ in range(2)]
program = Training(Backend(), model, x, y)
plans = {}
page_rows = max(2, min(8, (args.batch * args.sequence + program.backend.capacity - 1) // program.backend.capacity))
for mode in ('coarse', 'pages', 'ready', 'static'):
    os.environ['TILE_SCHEDULE'] = '1' if mode in ('ready', 'static') else '0'
    os.environ['TILE_ENGINE'] = 'static' if mode == 'static' else 'ready'
    os.environ['PAGE_PIPELINE'] = '0' if mode == 'coarse' else '1'
    os.environ['PAGE_ENGINE'] = 'warps'
    os.environ['PAGE_ROWS'] = str(page_rows)
    program.upload()
    plans[mode] = (program.descriptors, program.scheduled_count)
results = {mode: [] for mode in plans}
for repeat in range(args.repeats):
    modes = list(plans)
    modes = modes[repeat % len(modes):] + modes[:repeat % len(modes)]
    for mode in modes:
        program.descriptors, program.scheduled_count = plans[mode]
        program.reset()
        timing = measure(program.run, iterations=args.iterations)
        results[mode].append(timing)
        print(mode, repeat, timing, flush=True)
root = Path(__file__).resolve().parents[1]
digest = hashlib.sha256()
for path in sorted([root/'src/executor.cu', root/'include/gpt2/executor.h',
                    *root.glob('include/gpt2/kernels/*.cuh'), *root.glob('gpt2_cuda/*.py')]):
    digest.update(path.read_bytes())
record = dict(batch=args.batch, sequence=args.sequence, gpu=torch.cuda.get_device_name(),
              torch=torch.__version__, cuda=torch.version.cuda, dtype='float32', tf32=False,
              source_sha256=digest.hexdigest(), workers=program.backend.capacity,
              page_engine='warps', page_rows=page_rows, repeats=args.repeats, iterations=args.iterations,
              results=results, median_gpu_ms={mode: statistics.median(r['gpu_median_ms'] for r in runs)
                                            for mode, runs in results.items()})
args.output.write_text(json.dumps(record, indent=2)+'\n')
print(record['median_gpu_ms'], flush=True)
