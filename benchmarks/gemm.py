"""Standalone and persistent GEMM checks at the actual GPT-2 training shapes."""
import argparse
import json

import torch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gpt2_cuda import Backend, Program
from reference import measure


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=4096)
    args = parser.parse_args()
    if args.tokens < 1:
        parser.error("tokens must be positive")
    tokens = args.tokens
    torch.manual_seed(123)
    torch.backends.cuda.matmul.allow_tf32 = False
    backend = Backend()
    for m, n, k, ta, tb in [(tokens, 2304, 768, False, True),
                            (tokens, 768, 3072, False, False),
                            (3072, 768, tokens, True, False),
                            (tokens, 50304, 768, False, True),
                            (tokens, 768, 50304, False, False),
                            (50304, 768, tokens, True, False)]:
        a = torch.randn((k, m) if ta else (m, k), device="cuda")
        b = torch.randn((n, k) if tb else (k, n), device="cuda")
        expected = (a.T if ta else a) @ (b.T if tb else b)
        out = torch.empty_like(expected)
        p = Program(backend)
        p.matmul(a, b, out, ta=ta, tb=tb)
        p.upload()
        results = {"shape": [m, n, k], "transpose": [ta, tb]}
        for mode in (False, True):
            p.run(persistent=mode)
            torch.testing.assert_close(out, expected, atol=3e-3, rtol=1e-3)
            results["persistent" if mode else "standalone"] = measure(lambda: p.run(persistent=mode))
        results["torch"] = measure(lambda: torch.mm(a.T if ta else a, b.T if tb else b, out=expected))
        print(json.dumps(results), flush=True)


if __name__ == "__main__":
    main()
