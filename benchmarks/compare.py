"""Validate losses, every gradient and AdamW updates before reporting speed."""
import argparse
import json
import os
import hashlib
from pathlib import Path

import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gpt2_cuda import Backend, Training
from reference import GPT, GPTConfig, measure


def error(actual, expected):
    d = (actual - expected).float()
    return {"max_abs": d.abs().max().item(),
            "relative_rms": (d.square().mean().sqrt() /
                             expected.float().square().mean().sqrt().clamp_min(1e-12)).item()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=int, default=12)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--standalone", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(123)
    torch.backends.cuda.matmul.allow_tf32 = False
    model = GPT(GPTConfig(n_layer=args.layers)).cuda()
    tokens = torch.randint(50257, (4, 64), device="cuda", dtype=torch.int32)
    targets = torch.randint(50257, (4, 64), device="cuda", dtype=torch.int32)
    backend = Backend()
    print(f"GPU={torch.cuda.get_device_name()} resident worker capacity={backend.capacity}", flush=True)
    program = Training(backend, model, tokens, targets)
    print(f"{len(program.ops)} operations in {program.stages} stages; {program.parameters.numel()} parameter slots", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01,
                           betas=(0.9, 0.999), eps=1e-8, fused=True)
    x, y = tokens.long(), targets.long()

    def reference_step():
        opt.zero_grad(set_to_none=True)
        logits, loss = model(x, y)
        loss.backward()
        opt.step()
        return logits, loss

    def cuda_step():
        program.run(persistent=not args.standalone, workers=args.workers)

    checks = []
    for i in range(args.steps):
        logits, loss = reference_step()
        cuda_step()
        torch.cuda.synchronize()
        torch.testing.assert_close(program.mean_loss, loss, atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(program.logits, logits.view_as(program.logits), atol=3e-4, rtol=3e-4)
        worst_grad = (0, "")
        for name, p in model.named_parameters():
            grad_error = error(program.weight_grads[name], p.grad)
            assert grad_error["relative_rms"] < 2e-3 or grad_error["max_abs"] < 2e-7, (name, grad_error)
            if grad_error["relative_rms"] > worst_grad[0]:
                worst_grad = (grad_error["relative_rms"], name)
            # Tiny LayerNorm bias gradients can have a large relative error at
            # values close to zero; an absolute check covers that regime.
            torch.testing.assert_close(program.weight_grads[name], p.grad, atol=3e-5, rtol=3e-3,
                                       msg=lambda msg: name + " gradient: " + msg)
            torch.testing.assert_close(program.weights[name], p.detach(), atol=1e-4, rtol=3e-4,
                                       msg=lambda msg: name + " parameter: " + msg)
        check = {"step": i + 1, "loss": loss.item(), "cuda_loss": program.mean_loss.item(),
                 "worst_gradient_relative_rms": worst_grad}
        checks.append(check)
        print(json.dumps(check), flush=True)

    if args.profile:
        timings = []
        for op, label in zip(program.ops, program.labels):
            result = measure(lambda: backend.operation(op), warmup=1, iterations=3)
            timings.append((result["gpu_median_ms"], label, op.m, op.n, op.k))
        print("Slowest standalone operations:", sorted(timings, reverse=True)[:20], flush=True)
        print("Sum standalone operation medians:", sum(x[0] for x in timings), flush=True)

    digest = hashlib.sha256()
    root = Path(__file__).resolve().parents[1]
    for path in sorted([root / "src/executor.cu", root / "include/gpt2/executor.h",
                        root / "gpt2_cuda/__init__.py", *root.glob("include/gpt2/kernels/*.cuh")]):
        digest.update(path.read_bytes())
    result = {"source_sha256": digest.hexdigest(), "gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "cuda_version": torch.version.cuda,
              "tf32": False, "dtype": "float32", "layers": args.layers,
              "cuda_gemm": backend.gemm, "grouped": os.getenv("GROUP", "1") != "0",
              "operations": len(program.ops), "stages": program.stages,
              "batch": 4, "sequence": 64, "optimizer": "AdamW fused reference", "pytorch_execution": "eager",
              "resident_worker_capacity": backend.capacity, "workers": args.workers or backend.workers,
              "mode": "standalone" if args.standalone else "persistent", "checks": checks}
    result["pytorch"] = measure(reference_step, iterations=args.iterations)
    result["cuda"] = measure(cuda_step, iterations=args.iterations)
    result["speedup"] = result["pytorch"]["gpu_median_ms"] / result["cuda"]["gpu_median_ms"]
    print(json.dumps(result, indent=2), flush=True)
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
