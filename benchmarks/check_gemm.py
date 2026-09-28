"""Vector loads, scalar fallbacks, transpose combinations, tails and split-K."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gpt2_cuda import Backend, Program


def main():
    torch.manual_seed(19)
    torch.backends.cuda.matmul.allow_tf32 = False
    backend = Backend()
    for m, n, k, offset in ((132, 260, 100, 0), (67, 71, 53, 0),
                             (128, 256, 96, 1), (8, 12, 5000, 0)):
        for ta, tb in ((False, False), (False, True), (True, False), (True, True)):
            def operand(shape):
                data = torch.randn(shape[0] * shape[1] + offset, device='cuda') * .1
                return data[offset:].view(shape)
            a = operand((k, m) if ta else (m, k))
            b = operand((n, k) if tb else (k, n))
            product = (a.T if ta else a) @ (b.T if tb else b)
            for epilogue in ('overwrite', 'add', 'bias'):
                p = Program(backend)
                out = p.empty((m, n))
                initial = torch.randn_like(out) * .1
                bias = torch.randn(n, device='cuda') * .1 if epilogue == 'bias' else None
                p.matmul(a, b, out, ta=ta, tb=tb, add=epilogue == 'add', bias=bias)
                p.upload()
                expected = product + (initial if epilogue == 'add' else bias if bias is not None else 0)
                for persistent, workers in ((False, 0), (True, 1), (True, backend.capacity)):
                    for _ in range(2):
                        out.copy_(initial) if epilogue == 'add' else out.fill_(float('nan'))
                        p.run(persistent=persistent, workers=workers)
                        torch.testing.assert_close(out, expected, atol=3e-4, rtol=3e-4)
            print(f'GEMM checks passed: {m}x{n}x{k}, transpose={ta,tb}, offset={offset}', flush=True)


if __name__ == '__main__':
    main()
