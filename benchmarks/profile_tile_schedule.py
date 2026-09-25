"""Record tile timestamps for actual MLP forward and backward instructions."""
import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gpt2_cuda import Backend, Training
from reference import GPT, GPTConfig

os.environ['TILE_TRACE'] = '1'
os.environ['TILE_SCHEDULE'] = '1'
torch.manual_seed(123)
torch.set_num_threads(1)
model = GPT(GPTConfig(n_layer=2)).cuda()
x, y = [torch.randint(50257, (8, 128), dtype=torch.int32, device='cuda') for _ in range(2)]
program = Training(Backend(), model, x, y)
program.run()
torch.cuda.synchronize()
records = []
for graph in program.tile_graphs:
    events = graph['trace'].cpu().tolist()
    for child, parents in enumerate(graph['parents']):
        for parent in parents:
            assert events[child][0] >= events[parent][1]
    # Report actual concurrent execution of distinct instructions on distinct
    # CTAs, not merely switching between instructions on a single CTA.
    active, concurrent_pairs = [], set()
    for index in sorted(range(len(events)), key=lambda i: events[i][0]):
        start, end, cta = events[index]
        op = graph['nodes'][index][0]
        active = [(e, c, o) for e, c, o in active if e > start]
        for _, other_cta, other_op in active:
            if other_cta != cta and other_op != op:
                concurrent_pairs.add(tuple(sorted((other_op, op))))
        active.append((end, cta, op))
    labels = graph['labels']
    records.append(dict(labels=labels, tasks=len(events),
                        overlapping_instruction_pairs=[[labels[a], labels[b]] for a, b in sorted(concurrent_pairs)],
                        nodes=graph['nodes'], parents=graph['parents'], events=events))
    print(json.dumps({k: v for k, v in records[-1].items() if k not in ('nodes', 'parents', 'events')}), flush=True)
Path('tile-trace.json').write_text(json.dumps(records))

page_records = []
for trace in program.page_traces:
    events = trace.cpu().tolist()
    for start, loaded, compute, end in events:
        assert 0 < start <= loaded <= compute <= end
    rows = int(os.getenv('PAGE_ROWS', '2'))
    concurrent = sum(events[i][2] < events[i+1][1] and events[i+1][0] < events[i][3]
                     for i in range(len(events)-1) if (i+1) % rows)
    print('Page load/compute overlap rows:', concurrent, flush=True)
    page_records.append(dict(events=events, concurrent_rows=concurrent))
Path('page-trace.json').write_text(json.dumps(page_records))
