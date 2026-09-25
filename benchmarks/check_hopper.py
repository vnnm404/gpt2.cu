"""Exercise TMA tails, transposes, pipeline phases, bias, and repeated launches."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gpt2_cuda import Backend, Code, Program


def main():
    torch.manual_seed(19)
    torch.backends.cuda.matmul.allow_tf32 = False
    backend = Backend()
    assert backend.gemm == 'hopper_tma_wgmma_bf16x6', 'This check requires an H100'
    # Aligned ordinary FP32 descriptors must retain the SIMT fallback unless
    # the explicit packed-operands flag is present.
    a = torch.randn((64, 32), device='cuda')
    b = torch.randn((32, 128), device='cuda')
    out = torch.empty((64, 128), device='cuda')
    raw = Program(backend)
    raw.emit(Code.GEMM, [a, b, out, None], 64, 128, 32, tiles=1)
    raw.upload()
    raw.run()
    torch.testing.assert_close(out, a @ b, atol=2e-5, rtol=2e-4)
    print('Explicit FP32 descriptor fallback passed', flush=True)
    for m, n, k in [(64, 128, 32), (68, 132, 40), (128, 256, 96), (68, 132, 1032), (64, 128, 8192)]:
        for ta in (False, True):
            for tb in (False, True):
                a = torch.randn((k, m) if ta else (m, k), device='cuda') * .1
                b = torch.randn((n, k) if tb else (k, n), device='cuda') * .1
                bias = torch.randn(n, device='cuda') * .1
                initial = torch.randn((m, n), device='cuda') * .1
                expected = (a.T if ta else a) @ (b.T if tb else b) + bias + initial
                out = initial.clone()
                program = Program(backend)
                program.matmul(a, b, out, ta=ta, tb=tb, bias=bias, add=True)
                program.upload()
                for workers in (1, 3, backend.capacity):
                    for _ in range(3):
                        out.copy_(initial)
                        program.run(workers=workers)
                        torch.testing.assert_close(out, expected, atol=2e-5, rtol=2e-4)
        print(f'TMA/WGMMA checks passed: {m}x{n}x{k}', flush=True)


if __name__ == '__main__':
    main()
