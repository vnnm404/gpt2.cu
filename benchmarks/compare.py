"""Validate losses, every gradient and AdamW updates before reporting speed."""
import argparse
import json
import os
import hashlib
import subprocess
import tempfile
from pathlib import Path

import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gpt2_cuda import Backend, Code, Training
from reference import GPT, GPTConfig, measure


def error(actual, expected):
    d = (actual - expected).float()
    return {"max_abs": d.abs().max().item(),
            "relative_rms": (d.square().mean().sqrt() /
                             expected.float().square().mean().sqrt().clamp_min(1e-12)).item()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--sequence", type=int, default=128)
    parser.add_argument("--layers", type=int, default=12)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--standalone", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--isolated", action="store_true", help="Validate engines in separate processes using disk snapshots")
    parser.add_argument("--engine", choices=("reference", "cuda"), help=argparse.SUPPRESS)
    parser.add_argument("--snapshots", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.batch < 1 or not 1 <= args.sequence <= 256:
        parser.error("batch must be positive and sequence must be in [1, 256]")
    if args.isolated:
        command = [sys.executable, str(Path(__file__).resolve()),
                   *[value for value in sys.argv[1:] if value != "--isolated"]]
        with tempfile.TemporaryDirectory(prefix="gpt2-reference-") as directory:
            for engine in ("reference", "cuda"):
                subprocess.run([*command, "--engine", engine, "--snapshots", directory], check=True)
        return
    if args.engine and args.snapshots is None:
        parser.error("internal engine mode requires --snapshots")
    torch.set_num_threads(1)
    torch.manual_seed(123)
    torch.backends.cuda.matmul.allow_tf32 = False
    model = GPT(GPTConfig(n_layer=args.layers))
    if args.engine != "cuda":
        model.cuda()
    tokens = torch.randint(50257, (args.batch, args.sequence), device="cuda", dtype=torch.int32)
    targets = torch.randint(50257, (args.batch, args.sequence), device="cuda", dtype=torch.int32)
    opt = None if args.engine == "cuda" else torch.optim.AdamW(
        model.parameters(), lr=1e-4, weight_decay=0.01,
        betas=(0.9, 0.999), eps=1e-8, fused=True)
    x, y = tokens.long(), targets.long()

    def reference_step(snapshot_index=None):
        if args.engine == "cuda":
            path = args.snapshots / f"{snapshot_index}.pt"
            state = torch.load(path, map_location="cpu", weights_only=True)
            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    parameter.copy_(state["weights"][name])
                    parameter.grad = state["gradients"][name]
            path.unlink()
            return state["logits"].cuda(), state["loss"].cuda()
        opt.zero_grad(set_to_none=True)
        logits, loss = model(x, y)
        loss.backward()
        opt.step()
        return logits, loss

    if args.engine == "reference":
        for i in range(args.steps):
            logits, loss = reference_step()
            state = {"logits": logits.detach().cpu(), "loss": loss.detach().cpu(),
                     "weights": {name: p.detach().cpu() for name, p in model.named_parameters()},
                     "gradients": {name: p.grad.cpu() for name, p in model.named_parameters()}}
            torch.save(state, args.snapshots / f"{i}.pt")
            print(f"Saved reference step {i + 1}, loss={loss.item():.8f}", flush=True)
            del state, logits, loss
        timing = measure(reference_step, iterations=args.iterations)
        (args.snapshots / "timing.json").write_text(json.dumps(timing))
        return

    backend = Backend()
    print(f"GPU={torch.cuda.get_device_name()} resident worker capacity={backend.capacity}", flush=True)
    program = Training(backend, model, tokens, targets)
    print(f"{len(program.ops)} operations in {program.stages} stages; {program.parameters.numel()} parameter slots", flush=True)

    def cuda_step():
        program.run(persistent=not args.standalone, workers=args.workers)

    checks = []
    for i in range(args.steps):
        logits, loss = reference_step(i)
        cuda_step()
        torch.cuda.synchronize()
        torch.testing.assert_close(program.mean_loss, loss, atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(program.logits, logits.view_as(program.logits), atol=3e-4, rtol=3e-4)
        worst_grad = (0, "")
        for name, p in model.named_parameters():
            expected_gradient = p.grad.to(program.gradients.device)
            expected_parameter = p.detach().to(program.parameters.device)
            grad_error = error(program.weight_grads[name], expected_gradient)
            assert grad_error["relative_rms"] < 2e-3 or grad_error["max_abs"] < 2e-7, (name, grad_error)
            if grad_error["relative_rms"] > worst_grad[0]:
                worst_grad = (grad_error["relative_rms"], name)
            # Tiny LayerNorm bias gradients can have a large relative error at
            # values close to zero; an absolute check covers that regime.
            torch.testing.assert_close(program.weight_grads[name], expected_gradient, atol=3e-5, rtol=3e-3,
                                       msg=lambda msg: name + " gradient: " + msg)
            torch.testing.assert_close(program.weights[name], expected_parameter, atol=1e-4, rtol=3e-4,
                                       msg=lambda msg: name + " parameter: " + msg)
        check = {"step": i + 1, "loss": loss.item(), "cuda_loss": program.mean_loss.item(),
                 "worst_gradient_relative_rms": worst_grad}
        checks.append(check)
        print(json.dumps(check), flush=True)

    if args.profile:
        timings = []
        categories = {}
        for op, label in zip(program.ops, program.labels):
            result = measure(lambda: backend.operation(op), warmup=1, iterations=3)
            timings.append((result["gpu_median_ms"], label, op.m, op.n, op.k))
            category = Code(op.code).name
            categories[category] = categories.get(category, 0) + result["gpu_median_ms"]
        print("Slowest standalone operations:", sorted(timings, reverse=True)[:20], flush=True)
        print("Operation category totals:", sorted(categories.items(), key=lambda x: -x[1]), flush=True)
        print("Sum standalone operation medians:", sum(x[0] for x in timings), flush=True)

    digest = hashlib.sha256()
    root = Path(__file__).resolve().parents[1]
    for path in sorted([root / "src/executor.cu", root / "include/gpt2/executor.h",
                        root / "gpt2_cuda/__init__.py", *root.glob("include/gpt2/kernels/*.cuh")]):
        digest.update(path.read_bytes())
    result = {"source_sha256": digest.hexdigest(), "gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "cuda_version": torch.version.cuda,
              "pytorch_allow_tf32": False, "dtype": "float32", "layers": args.layers,
              "compute_capability": list(torch.cuda.get_device_capability()),
              "cuda_gemm": backend.gemm, "grouped": os.getenv("GROUP", "1") != "0",
              "operations": len(program.ops), "stages": program.stages,
              "batch": args.batch, "sequence": args.sequence, "optimizer": "AdamW fused reference", "pytorch_execution": "eager",
              "resident_worker_capacity": backend.capacity, "workers": args.workers or backend.workers,
              "mode": "standalone" if args.standalone else "persistent",
              "isolated_processes": args.engine == "cuda", "checks": checks}
    result["pytorch"] = (json.loads((args.snapshots / "timing.json").read_text())
                         if args.engine == "cuda" else measure(reference_step, iterations=args.iterations))
    result["cuda"] = measure(cuda_step, iterations=args.iterations)
    for engine in ("pytorch", "cuda"):
        result[engine]["tokens_per_second"] = args.batch * args.sequence * 1000 / result[engine]["wall_median_ms"]
    result["speedup"] = result["pytorch"]["gpu_median_ms"] / result["cuda"]["gpu_median_ms"]
    print(json.dumps(result, indent=2), flush=True)
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
