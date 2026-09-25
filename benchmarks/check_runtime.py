"""Boundary, optimizer and synchronization checks. Run under a process timeout."""
import torch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gpt2_cuda import Backend, Code, Program
from gpt2_cuda import Training
from reference import GPT, GPTConfig


def main():
    torch.manual_seed(7)
    torch.backends.cuda.matmul.allow_tf32 = False
    backend = Backend()

    # Odd dimensions and repeated transitions between idle and active workers.
    for workers in (1, 3, backend.capacity // 2, backend.capacity):
        p = Program(backend)
        initial = torch.randn((67, 71), device="cuda")
        x, expected = initial, initial.clone()
        for _ in range(12):
            w = torch.randn((71, 71), device="cuda") * 0.05
            out = p.empty((67, 71))
            p.matmul(x, w, out)
            expected = expected @ w
            x = out
        p.upload()
        for _ in range(5):
            p.run(workers=workers)
            torch.testing.assert_close(x, expected, rtol=2e-4, atol=2e-6)
        print(f"barrier and odd-shape checks passed: {workers} workers", flush=True)
    try:
        p.run(workers=backend.capacity + 1)
        raise AssertionError("oversubscribed cooperative launch was accepted")
    except RuntimeError as e:
        assert "CUDA error 1" in str(e)

    # Isolate optimizer accuracy from amplified near-zero gradient differences.
    p = Program(backend)
    parameter = p.empty(4097)
    parameter.normal_()
    reference = torch.nn.Parameter(parameter.clone())
    gradient = p.empty(4097)
    moment, variance, clock = p.empty(4097, zero=True), p.empty(4097, zero=True), p.empty(3, zero=True)
    p.emit(Code.ADVANCE, [clock], 1)
    p.emit(Code.ADAMW, [parameter, gradient, moment, variance, clock], 4097, scalars=(1e-4, 0.01))
    p.upload()
    opt = torch.optim.AdamW([reference], lr=1e-4, weight_decay=0.01, fused=True)
    for _ in range(10):
        gradient.normal_()
        reference.grad = gradient.clone()
        opt.step()
        p.run()
        torch.testing.assert_close(parameter, reference, atol=2e-6, rtol=1e-6)
        torch.testing.assert_close(moment, opt.state[reference]["exp_avg"], atol=2e-7, rtol=2e-6)
        torch.testing.assert_close(variance, opt.state[reference]["exp_avg_sq"], atol=2e-7, rtol=2e-6)
    print("AdamW parameters and both moments match for ten steps", flush=True)

    # Exercise every operation, padded vocabulary, and partial attention warps.
    for sequence in (1, 17, 64, 256):
        model = GPT(GPTConfig(n_layer=1, n_head=1, n_embd=64, vocab_size=257)).cuda()
        tokens = torch.randint(257, (2, sequence), device="cuda", dtype=torch.int32)
        targets = torch.randint(257, (2, sequence), device="cuda", dtype=torch.int32)
        program = Training(backend, model, tokens, targets)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01, fused=True)
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            logits, loss = model(tokens.long(), targets.long())
            loss.backward()
            optimizer.step()
            program.run()
            torch.testing.assert_close(program.mean_loss, loss, atol=2e-5, rtol=2e-5)
            torch.testing.assert_close(program.logits, logits.view_as(program.logits), atol=3e-4, rtol=3e-4)
            for name, parameter in model.named_parameters():
                torch.testing.assert_close(program.weight_grads[name], parameter.grad, atol=3e-5, rtol=3e-3)
                torch.testing.assert_close(program.weights[name], parameter, atol=1e-4, rtol=3e-4)
        print(f"full training checks passed: sequence={sequence}, vocabulary=257", flush=True)


if __name__ == "__main__":
    main()
