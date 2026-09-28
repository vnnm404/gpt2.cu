"""Validate losses, every gradient and AdamW updates before reporting speed."""
import argparse
import json
import os
import hashlib
import statistics
import time
import math
from pathlib import Path

import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gpt2_cuda import Backend, Training
from reference import GPT, GPTConfig, measure


def error(actual, expected):
    squared_error, squared_reference, maximum = 0., 0., 0.
    for a, b in chunks(actual, expected):
        d = (a - b).float()
        squared_error += d.square().sum(dtype=torch.float64).item()
        squared_reference += b.float().square().sum(dtype=torch.float64).item()
        maximum = max(maximum, d.abs().max().item())
    return {"max_abs": maximum,
            "relative_rms": math.sqrt(squared_error / actual.numel()) /
                            max(math.sqrt(squared_reference / actual.numel()), 1e-12)}


def chunks(actual, expected):
    """Bound validation scratch memory without copying noncontiguous logits."""
    assert actual.shape == expected.shape
    if actual.ndim == 0:
        yield actual, expected
        return
    rows = max(1, (1 << 20) // math.prod(actual.shape[1:]))
    for begin in range(0, actual.shape[0], rows):
        yield actual[begin:begin + rows], expected[begin:begin + rows]


def assert_close(actual, expected, **kwargs):
    for a, b in chunks(actual, expected):
        torch.testing.assert_close(a, b, **kwargs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--sequence", type=int, default=64)
    parser.add_argument("--layers", type=int, default=12)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--standalone", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--compile-mode", choices=("default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"),
                        help="Compile the reference forward/backward with Inductor; retain fused AdamW")
    parser.add_argument("--repeats", type=int, default=3,
                        help="Alternate timing order across independent measurement rounds")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.batch < 1 or not 1 <= args.sequence <= 256 or min(args.steps, args.iterations, args.repeats) < 1:
        parser.error("positive batch/steps/iterations/repeats and sequence in [1, 256] required")
    torch.set_num_threads(1)
    torch.manual_seed(123)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = GPT(GPTConfig(n_layer=args.layers)).cuda()
    tokens = torch.randint(50257, (args.batch, args.sequence), device="cuda", dtype=torch.int32)
    targets = torch.randint(50257, (args.batch, args.sequence), device="cuda", dtype=torch.int32)
    backend = Backend()
    print(f"GPU={torch.cuda.get_device_name()} resident worker capacity={backend.capacity}", flush=True)
    program = Training(backend, model, tokens, targets)
    print(f"{len(program.ops)} operations in {program.stages} stages; {program.parameters.numel()} parameter slots", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01,
                           betas=(0.9, 0.999), eps=1e-8, fused=True)
    x, y = tokens.long(), targets.long()
    # Keep the original module for parameter/gradient validation. Compilation
    # includes AOTAutograd backward; the identical fused optimizer stays eager.
    setup_start = time.perf_counter()
    reference_model = (torch.compile(model, backend="inductor", mode=args.compile_mode,
                                     fullgraph=True, dynamic=False)
                       if args.compile_mode else model)
    compile_wrapper_seconds = time.perf_counter() - setup_start

    def reference_step():
        opt.zero_grad(set_to_none=True)
        if args.compile_mode:
            torch.compiler.cudagraph_mark_step_begin()
        logits, loss = reference_model(x, y)
        loss.backward()
        opt.step()
        return logits, loss

    def cuda_step():
        program.run(persistent=not args.standalone, workers=args.workers)

    checks = []
    for i in range(args.steps):
        torch.cuda.synchronize()
        reference_start = time.perf_counter()
        logits, loss = reference_step()
        torch.cuda.synchronize()
        if i == 0:
            first_reference_step_seconds = time.perf_counter() - reference_start
        cuda_step()
        torch.cuda.synchronize()
        torch.testing.assert_close(program.mean_loss, loss, atol=2e-5, rtol=2e-5)
        assert_close(program.logits, logits.view_as(program.logits), atol=3e-4, rtol=3e-4)
        worst_grad = (0, "")
        for name, p in model.named_parameters():
            grad_error = error(program.weight_grads[name], p.grad)
            assert grad_error["relative_rms"] < 2e-3 or grad_error["max_abs"] < 2e-7, (name, grad_error)
            if grad_error["relative_rms"] > worst_grad[0]:
                worst_grad = (grad_error["relative_rms"], name)
            # Tiny LayerNorm bias gradients can have a large relative error at
            # values close to zero; an absolute check covers that regime.
            assert_close(program.weight_grads[name], p.grad, atol=3e-5, rtol=3e-3,
                                       msg=lambda msg: name + " gradient: " + msg)
            assert_close(program.weights[name], p.detach(), atol=1e-4, rtol=3e-4,
                                       msg=lambda msg: name + " parameter: " + msg)
        check = {"step": i + 1, "loss": loss.item(), "cuda_loss": program.mean_loss.item(),
                 "worst_gradient_relative_rms": worst_grad}
        checks.append(check)
        print(json.dumps(check), flush=True)
        # Release graph-managed outputs before marking the next training step.
        del logits, loss

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
                        *root.glob("include/gpt2/kernels/*.cuh"), *root.glob("gpt2_cuda/*.py")]):
        digest.update(path.read_bytes())
    result = {"source_sha256": digest.hexdigest(), "gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "cuda_version": torch.version.cuda,
              "tf32": False, "dtype": "float32", "layers": args.layers,
              "cuda_gemm": backend.gemm, "grouped": os.getenv("GROUP", "1") != "0",
              "operations": len(program.ops), "stages": program.stages,
              "batch": args.batch, "sequence": args.sequence, "optimizer": "AdamW fused reference",
              "pytorch_execution": "torch.compile" if args.compile_mode else "eager",
              "compile_mode": args.compile_mode, "compile_backend": "inductor" if args.compile_mode else None,
              "compile_fullgraph": bool(args.compile_mode), "optimizer_compiled": False,
              "compile_wrapper_seconds": compile_wrapper_seconds,
              "first_reference_step_seconds": first_reference_step_seconds,
              "resident_worker_capacity": backend.capacity, "workers": args.workers or backend.workers,
              "threads_per_worker": backend.threads,
              "mode": "standalone" if args.standalone else "persistent", "checks": checks,
              "tile_engine": os.getenv("TILE_ENGINE", "ready"), "page_rows": sorted(set(program.page_rows)),
              "page_engine": os.getenv("PAGE_ENGINE", "warps"),
              "tile_regions": len(program.tile_graphs), "page_pipelines": program.page_pipelines}
    rounds = {"pytorch": [], "cuda": []}
    for repeat in range(args.repeats):
        order = (("pytorch", reference_step), ("cuda", cuda_step))
        for name, step in order[::1 if repeat % 2 == 0 else -1]:
            rounds[name].append(measure(step, iterations=args.iterations))
    for name, measurements in rounds.items():
        result[name] = {key: statistics.median(m[key] for m in measurements)
                        for key in ("gpu_median_ms", "wall_median_ms", "gpu_min_ms")}
        result[name].update(iterations=args.iterations, repeats=args.repeats, measurements=measurements)
    result["speedup"] = result["pytorch"]["gpu_median_ms"] / result["cuda"]["gpu_median_ms"]
    print(json.dumps(result, indent=2), flush=True)
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
