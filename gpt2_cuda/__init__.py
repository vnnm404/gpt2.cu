"""A small explicit training program shared by standalone and persistent CUDA.

PyTorch owns memory and supplies the reference model; CUDA executes all training
math. Operation descriptors are uploaded once, not rebuilt at each iteration.
"""
import ctypes as ct
import enum
import os
import subprocess
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


class Code(enum.IntEnum):
    GEMM, ADD, GELU, GELU_BACKWARD, NORM, NORM_BACKWARD, NORM_PARAMETERS, EMBEDDING, EMBEDDING_BACKWARD, ATTENTION, ATTENTION_BACKWARD, ATTENTION_KV_BACKWARD, CROSS_ENTROPY, SUM_ROWS, ADAMW, CLEAR, ADVANCE, SUM_SPLITS = range(18)


class Operation(ct.Structure):
    _fields_ = [(s, ct.c_int) for s in ("code", "tiles", "m", "n", "k", "flags", "group")] + [
        ("p", ct.c_void_p * 8), ("scalar", ct.c_float * 4)]


assert ct.sizeof(Operation) == 112


class Backend:
    def __init__(self):
        build = ROOT / "build"
        build.mkdir(exist_ok=True)
        self.gemm = "cutlass_simt_fp32"
        library = build / "libgpt2_executor.so"
        dependency = build / "cutlass"
        revision = "f7b19de32c5d1f3cedfc735c2849f12b537522ee"
        if not dependency.exists():
            subprocess.run(["git", "clone", "--depth", "1", "--branch", "v3.5.1",
                            "https://github.com/NVIDIA/cutlass.git", str(dependency)], check=True)
        actual = subprocess.check_output(["git", "-C", str(dependency), "rev-parse", "HEAD"], text=True).strip()
        if actual != revision:
            raise RuntimeError(f"Expected CUTLASS {revision}, found {actual}")
        extra = ["--expt-relaxed-constexpr", "-I", str(dependency / "include")]
        sources = [ROOT / "src/executor.cu", *ROOT.glob("include/gpt2/kernels/*.cuh"), ROOT / "include/gpt2/executor.h"]
        if not library.exists() or any(p.stat().st_mtime > library.stat().st_mtime for p in sources):
            subprocess.run(["nvcc", "-std=c++17", "-O3", "-lineinfo", "-arch=sm_86",
                            "--shared", "-Xcompiler=-fPIC", "-I", str(ROOT / "include"),
                            str(sources[0]), "-o", str(library),
                            *extra], check=True)
        self.lib = ct.CDLL(str(library))
        self.tile_m = self.lib.gpt2_gemm_tile_m()
        self.lib.gpt2_operation.argtypes = [ct.POINTER(Operation), ct.c_void_p]
        self.lib.gpt2_launch.argtypes = [ct.c_void_p, ct.c_int, ct.c_int, ct.c_void_p]
        self.lib.gpt2_occupancy.argtypes = [ct.POINTER(ct.c_int)]
        self.lib.gpt2_gemm_parameters.argtypes = [ct.POINTER(Operation), ct.c_void_p]
        capacity = ct.c_int()
        self.check(self.lib.gpt2_occupancy(ct.byref(capacity)))
        self.capacity = capacity.value
        self.workers = self.capacity

    @staticmethod
    def check(error):
        if error:
            raise RuntimeError(f"CUDA error {error}")

    def operation(self, op):
        self.check(self.lib.gpt2_operation(ct.byref(op), torch.cuda.current_stream().cuda_stream))

    def launch(self, descriptors, count, workers=0):
        self.check(self.lib.gpt2_launch(descriptors.data_ptr(), count, workers or self.workers,
                                       torch.cuda.current_stream().cuda_stream))


class Program:
    def __init__(self, backend):
        self.backend = backend
        self.ops = []
        self.labels = []
        self.allocations = []
        self.accesses = []

    def empty(self, shape, *, zero=False):
        out = (torch.zeros if zero else torch.empty)(shape, device="cuda", dtype=torch.float32)
        self.allocations.append(out)
        return out

    def emit(self, code, pointers, m, n=0, k=0, flags=0, tiles=None, label="", scalars=()):
        if tiles is None:
            tiles = (m + 4095) // 4096
        if m < 1 or tiles < 1:
            raise ValueError("Operations require positive work sizes")
        if any(p is not None and not p.is_contiguous() for p in pointers):
            raise ValueError("Operation tensors must be contiguous")
        self.allocations.extend(p for p in pointers if p is not None)
        op = Operation(int(code), tiles, m, n, k, flags)
        for i, p in enumerate(pointers):
            op.p[i] = p.data_ptr() if p is not None else None
        for i, value in enumerate(scalars):
            op.scalar[i] = value
        self.ops.append(op)
        self.labels.append(label or code.name.lower())
        writes = {
            Code.GEMM: (2,), Code.ADD: (2,), Code.GELU: (1,), Code.GELU_BACKWARD: (2,),
            Code.NORM: (3, 4), Code.NORM_BACKWARD: (4,), Code.NORM_PARAMETERS: (3, 4),
            Code.EMBEDDING: (3,), Code.EMBEDDING_BACKWARD: (2, 3), Code.ATTENTION: (1, 2),
            Code.ATTENTION_BACKWARD: (3, 4), Code.ATTENTION_KV_BACKWARD: (3,),
            Code.CROSS_ENTROPY: (2, 3), Code.SUM_ROWS: (3,), Code.ADAMW: (0, 2, 3),
            Code.CLEAR: (0,), Code.ADVANCE: (0,), Code.SUM_SPLITS: (1,),
        }[code]
        # Treat read/write outputs conservatively as reads too. This permits
        # overlap only when byte ranges are disjoint, including aliased views.
        ranges = [(p.data_ptr(), p.data_ptr() + p.numel() * p.element_size())
                  if p is not None else None for p in pointers]
        self.accesses.append(([r for r in ranges if r], [ranges[i] for i in writes]))

    def matmul(self, a, b, out, *, ta=False, tb=False, add=False, bias=None, label="gemm"):
        m, n = out.shape
        k = a.shape[0] if ta else a.shape[1]
        output_tiles = ((m + self.backend.tile_m - 1) // self.backend.tile_m) * ((n + 63) // 64)
        splits = 16 if k > 4096 else 1
        if k <= 4096 and k >= 768:
            while output_tiles * splits < self.backend.capacity and splits < 8:
                splits *= 2
        temp = self.empty((splits, m, n)) if splits > 1 else out
        self.emit(Code.GEMM, [a, b, temp, bias if splits == 1 else None], m, n, k,
                  int(ta) | (int(tb) << 1) | (int(add and splits == 1) << 2) |
                  (splits << 8),
                  tiles=output_tiles * splits, label=label)
        if splits > 1:
            self.emit(Code.SUM_SPLITS, [temp, out, bias], m * n, splits, n, flags=int(add) << 2)

    def upload(self):
        if not self.ops:
            raise ValueError("Cannot upload an empty program")
        for op in self.ops:
            op.group = 0
        def overlap(left, right):
            return any(a < d and c < b for a, b in left for c, d in right)

        start, reads, writes = 0, [], []
        self.stages = 0
        for i, (r, w) in enumerate(self.accesses):
            if overlap(r, writes) or overlap(w, reads) or os.getenv("GROUP", "1") == "0":
                if i > start:
                    self.ops[start].group = i - start
                    self.stages += 1
                start, reads, writes = i, [], []
            reads.extend(r)
            writes.extend(w)
        self.ops[start].group = len(self.ops) - start
        self.stages += 1
        prepare = self.backend.lib.gpt2_gemm_parameters
        for op in self.ops:
            if op.code == Code.GEMM:
                size = prepare(ct.byref(op), None)
                host = ct.create_string_buffer(size)
                self.backend.check(prepare(ct.byref(op), host))
                device = torch.frombuffer(bytearray(host.raw), dtype=torch.uint8).cuda()
                self.allocations.append(device)
                op.p[7] = device.data_ptr()
        raw = bytes((Operation * len(self.ops))(*self.ops))
        self.descriptors = torch.frombuffer(bytearray(raw), dtype=torch.uint8).cuda()

    def run(self, persistent=True, workers=0):
        if persistent:
            self.backend.launch(self.descriptors, len(self.ops), workers)
        else:
            for op in self.ops:
                self.backend.operation(op)


class Training(Program):
    def __init__(self, backend, reference, tokens, targets, update=True):
        super().__init__(backend)
        B, T = tokens.shape
        C = reference.config.n_embd
        assert C % 64 == 0 and reference.config.n_head == C // 64
        assert B > 0 and 0 < T <= 256, "attention supports sequences from 1 to 256"
        assert tokens.is_cuda and targets.is_cuda and tokens.dtype == targets.dtype == torch.int32
        assert tokens.shape == targets.shape and tokens.is_contiguous() and targets.is_contiguous()
        vocab = reference.config.vocab_size
        assert 0 <= tokens.min().item() and tokens.max().item() < vocab
        assert 0 <= targets.min().item() and targets.max().item() < vocab
        self.tokens, self.targets = tokens, targets
        params = list(reference.named_parameters())
        vocab = reference.config.vocab_size
        padded_vocab = (vocab + 63) // 64 * 64
        assert all(p.dtype == torch.float32 for _, p in params)
        total = sum(p.numel() for _, p in params) + (padded_vocab - vocab) * C
        self.parameters = self.empty(total)
        self.gradients = self.empty(total, zero=True)
        self.weights, self.weight_grads = {}, {}
        offset = 0
        for name, p in params:
            end = offset + p.numel()
            self.weights[name] = self.parameters[offset:end].view(p.shape)
            self.weights[name].copy_(p.detach())
            self.weight_grads[name] = self.gradients[offset:end].view(p.shape)
            if name == "transformer.wte.weight":
                end = offset + padded_vocab * C
                self.output_weight = self.parameters[offset:end].view(padded_vocab, C)
                self.output_weight[vocab:].zero_()
                self.output_gradient = self.gradients[offset:end].view(padded_vocab, C)
            offset = end
        self.initial = self.parameters.clone()
        self.momentum = self.empty(total, zero=True)
        self.variance = self.empty(total, zero=True)
        self.clock = self.empty(3, zero=True)
        self.tape = []
        self.derivatives = {}
        self.written = set()

        def gradient(x):
            key = x.data_ptr()
            if key not in self.derivatives:
                self.derivatives[key] = self.empty(x.shape)
            return self.derivatives[key]

        def accumulate(x, dy):
            dx = gradient(x)
            key = x.data_ptr()
            self.emit(Code.ADD, [dy, dx if key in self.written else None, dx], x.numel())
            self.written.add(key)

        def linear(x, name, tied=False):
            wn = "transformer.wte.weight" if tied else name + ".weight"
            w = self.output_weight if tied else self.weights[wn]
            bias = None if tied else self.weights[name + ".bias"]
            out = self.empty((B * T, w.shape[0]))
            self.matmul(x, w, out, tb=True, bias=bias, label=name + ".forward")

            def backward():
                dy, dx = gradient(out), gradient(x)
                self.matmul(dy, w, dx, add=x.data_ptr() in self.written, label=name + ".dx")
                self.written.add(x.data_ptr())
                self.matmul(dy, x, self.output_gradient if tied else self.weight_grads[wn], ta=True, label=name + ".dw")
                if bias is not None:
                    self.emit(Code.SUM_ROWS, [dy, None, None, self.weight_grads[name + ".bias"]],
                              B * T, w.shape[0], tiles=(w.shape[0] + 255) // 256)
            self.tape.append(backward)
            return out

        def norm(x, name):
            out, stats = self.empty(x.shape), self.empty((B * T, 2))
            w, bias = self.weights[name + ".weight"], self.weights[name + ".bias"]
            self.emit(Code.NORM, [x, w, bias, out, stats], B * T, C, tiles=B * T)

            def backward():
                dy, dx = gradient(out), gradient(x)
                self.emit(Code.NORM_BACKWARD, [x, dy, w, stats, dx,
                          dx if x.data_ptr() in self.written else None], B * T, C, tiles=B * T)
                self.written.add(x.data_ptr())
                self.emit(Code.NORM_PARAMETERS, [dy, x, stats, self.weight_grads[name + ".bias"],
                          self.weight_grads[name + ".weight"]], B * T, C, tiles=(C + 255) // 256)
            self.tape.append(backward)
            return out

        def add(x, y):
            out = self.empty(x.shape)
            self.emit(Code.ADD, [x, y, out], x.numel())

            def backward():
                accumulate(x, gradient(out))
                accumulate(y, gradient(out))
            self.tape.append(backward)
            return out

        def gelu(x):
            out = self.empty(x.shape)
            self.emit(Code.GELU, [x, out], x.numel())

            def backward():
                assert x.data_ptr() not in self.written
                self.emit(Code.GELU_BACKWARD, [x, gradient(out), gradient(x)], x.numel())
                self.written.add(x.data_ptr())
            self.tape.append(backward)
            return out

        def attention(qkv):
            out = self.empty((B * T, C))
            probs = self.empty((B * T, C // 64, T))
            tiles = (B * T * (C // 64) + 7) // 8
            self.emit(Code.ATTENTION, [qkv, out, probs], B * T, C, T, tiles=tiles)

            def backward():
                ds = self.empty(probs.shape)
                pointers = [qkv, gradient(out), probs, gradient(qkv), ds]
                self.emit(Code.ATTENTION_BACKWARD, pointers, B * T, C, T, tiles=tiles)
                self.emit(Code.ATTENTION_KV_BACKWARD, pointers, B * T, C, T, tiles=tiles)
                self.written.add(qkv.data_ptr())
            self.tape.append(backward)
            return out

        x = self.empty((B * T, C))
        encoded = x
        self.emit(Code.EMBEDDING, [tokens, self.weights["transformer.wte.weight"],
                  self.weights["transformer.wpe.weight"], x], x.numel(), C, T)
        for i in range(reference.config.n_layer):
            p = f"transformer.h.{i}"
            qkv = linear(norm(x, p + ".ln_1"), p + ".attn.c_attn")
            x = add(x, linear(attention(qkv), p + ".attn.c_proj"))
            y = linear(norm(x, p + ".ln_2"), p + ".mlp.c_fc")
            x = add(x, linear(gelu(y), p + ".mlp.c_proj"))
        logits = linear(norm(x, "transformer.ln_f"), "lm_head", tied=True)
        self.logits = logits[:, :vocab]
        self.loss = self.empty(B * T)
        self.emit(Code.CROSS_ENTROPY, [logits, targets, gradient(logits), self.loss],
                  B * T, padded_vocab, vocab, tiles=B * T)
        self.mean_loss = self.empty(())
        self.emit(Code.SUM_ROWS, [self.loss, None, None, self.mean_loss],
                  B * T, 1, tiles=1, scalars=(1.0 / (B * T),))
        self.written.add(logits.data_ptr())
        for backward in reversed(self.tape):
            backward()
        positional = self.weight_grads["transformer.wpe.weight"]
        self.emit(Code.CLEAR, [positional], positional.numel())
        self.emit(Code.EMBEDDING_BACKWARD, [tokens, gradient(encoded),
                  self.weight_grads["transformer.wte.weight"], positional], encoded.numel(), C, T)
        if update:
            self.emit(Code.ADVANCE, [self.clock], 1)
            self.emit(Code.ADAMW, [self.parameters, self.gradients, self.momentum, self.variance, self.clock],
                      total, scalars=(1e-4, 0.01))
        self.upload()

    def reset(self):
        self.parameters.copy_(self.initial)
        self.momentum.zero_()
        self.variance.zero_()
        self.clock.zero_()
