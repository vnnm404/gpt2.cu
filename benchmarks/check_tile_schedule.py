"""Real tile-edge ordering, cross-CTA publication, page reuse, and odd tails."""
import argparse
import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gpt2_cuda import Backend, Code, Program


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true', help='Smaller arithmetic workload for sanitizers')
    args = parser.parse_args()
    os.environ['TILE_SCHEDULE'] = '1'
    os.environ['TILE_TRACE'] = '1'
    torch.manual_seed(31)
    torch.backends.cuda.matmul.allow_tf32 = False
    backend = Backend()
    shapes = [(256, 256, 128, 128) if args.quick else (2048, 1024, 768, 256), (67, 260, 96, 132)]
    for rows, width, inner, output in shapes:
        p = Program(backend)
        x = torch.randn((rows, inner), device='cuda') * .1
        up = torch.randn((width, inner), device='cuda') * .1
        down = torch.randn((output, width), device='cuda') * .1
        a, b, y = p.empty((rows, width)), p.empty((rows, width)), p.empty((rows, output))
        p.matmul(x, up, a, tb=True, label='test.mlp.c_fc.forward')
        p.emit(Code.GELU, [a, b], a.numel())
        p.matmul(b, down, y, tb=True, label='test.mlp.c_proj.forward')
        p.regions.append((0, len(p.ops)))
        p.upload()
        expected = torch.nn.functional.gelu(x @ up.T, approximate='tanh') @ down.T
        assert len(p.tile_graphs) == 1
        graph = p.tile_graphs[0]
        for workers in (1, 3, backend.capacity):
            for _ in range(2 if args.quick else 5):
                p.run(workers=workers)
                torch.testing.assert_close(y, expected, atol=3e-5, rtol=5e-4)
            events = graph['trace'].cpu().tolist()
            cross_cta = 0
            for child, parents in enumerate(graph['parents']):
                assert events[child][1] >= events[child][0] > 0
                for parent in parents:
                    assert events[child][0] >= events[parent][1], (parent, child)
                    cross_cta += events[parent][2] != events[child][2]
            first_end = max(events[i][1] for i, node in enumerate(graph['nodes']) if node[0] == 0)
            later_start = min(events[i][0] for i, node in enumerate(graph['nodes']) if node[0] != 0)
            overlap = later_start < first_end
            concurrent = any(a[2] != b[2] and a[0] < b[1] and b[0] < a[1]
                             for i, a in enumerate(events) if graph['nodes'][i][0] == 0
                             for j, b in enumerate(events) if graph['nodes'][j][0] != 0)
            if rows == 2048 and os.getenv('TILE_ENGINE', 'ready') == 'ready': assert overlap, 'Consumer instructions did not start before producer instruction finished'
            print(json.dumps(dict(shape=[rows, width, inner, output], workers=workers,
                                  tasks=len(events), cross_cta_edges=cross_cta,
                                  cross_instruction_overlap=overlap,
                                  concurrent_instructions_on_distinct_ctas=concurrent)), flush=True)
    os.environ['TILE_TRACE'] = '0'
    # Exercise both pages, many generations, incomplete final tiles, and all
    # supported row widths. Reference preserves the original reduction order.
    for columns in (64, 260, 768, 1024):
        for page_rows in (1, 2, 7):
            os.environ['PAGE_ROWS'] = str(page_rows)
            p = Program(backend)
            x, y = [torch.randn((19, columns), device='cuda') for _ in range(2)]
            w, bias = [torch.randn(columns, device='cuda') for _ in range(2)]
            residual, out, stats = p.empty(x.shape), p.empty(x.shape), p.empty((19, 2))
            p.emit(Code.ADD, [x, y, residual], x.numel())
            p.emit(Code.NORM, [residual, w, bias, out, stats], 19, columns, tiles=19)
            p.upload()
            assert p.page_pipelines == 1
            expected = torch.nn.functional.layer_norm(x + y, (columns,), w, bias)
            for workers in (1, 3, backend.capacity):
                for _ in range(3):
                    p.run(workers=workers)
                    torch.testing.assert_close(residual, x + y, atol=0, rtol=0)
                    torch.testing.assert_close(out, expected, atol=3e-6, rtol=3e-5)
            print(f'Page pipeline passed: columns={columns}, rows/page stream={page_rows}', flush=True)


if __name__ == '__main__':
    main()
