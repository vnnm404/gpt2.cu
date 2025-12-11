"""Python-side megakernel benchmark.

Calls mk_entrypoint from tests/test_train_mk.cu (built as lib)
  so we can time things from python.
"""

from __future__ import annotations

import argparse
from contextlib import suppress
import ctypes
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np


MK_SEQ_LEN = 64  # match MK_SEQ_LEN in tests/test_train_mk.cu
MK_NUM_SM = 28  # match NUM_SM in tests/test_train_mk.cu
MK_THREADS_PER_BLOCK = 1024
MK_SHARED_MEM = 2 * 32 * 32 * ctypes.sizeof(ctypes.c_float)  # 2 * TILE_SIZE^2 * sizeof(float)
CUDA_MEMCPY_HOST_TO_DEVICE = 1


class Dim3(ctypes.Structure):
    _fields_ = [("x", ctypes.c_uint), ("y", ctypes.c_uint), ("z", ctypes.c_uint)]

    @classmethod
    def from_sequence(cls, dims: Sequence[int]) -> "Dim3":
        if len(dims) != 3:
            raise ValueError("dim3 expects three components")
        return cls(ctypes.c_uint(dims[0]), ctypes.c_uint(dims[1]), ctypes.c_uint(dims[2]))


class CudaRuntime:
    def __init__(self, path: str | None = None) -> None:
        cudart_path = self._resolve_cudart_path(path)
        try:
            self.lib = ctypes.CDLL(cudart_path, mode=ctypes.RTLD_GLOBAL)
        except OSError as exc:
            raise RuntimeError(f"Failed to load libcudart. Tried '{cudart_path}'.") from exc
        self._configure_signatures()

    @staticmethod
    def _resolve_cudart_path(explicit: str | None) -> str:
        if explicit:
            return explicit

        candidates = [
            os.environ.get("CUDA_HOME"),
            os.environ.get("CUDA_PATH"),
            "/usr/local/cuda",
        ]

        for base in candidates:
            if not base:
                continue
            if (p := Path(base, "lib", "libcudart.so")).exists():
                return str(p)

        return "libcudart.so"

    def _configure_signatures(self) -> None:
        self.lib.cudaGetErrorString.restype = ctypes.c_char_p

        self.lib.cudaDeviceSynchronize.restype = ctypes.c_int
        self.lib.cudaDeviceSynchronize.argtypes = []

        self.lib.cudaLaunchKernel.restype = ctypes.c_int
        self.lib.cudaLaunchKernel.argtypes = [
            ctypes.c_void_p,
            Dim3,
            Dim3,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_size_t,
            ctypes.c_void_p,
        ]

        self.lib.cudaEventCreate.restype = ctypes.c_int
        self.lib.cudaEventCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        self.lib.cudaEventRecord.restype = ctypes.c_int
        self.lib.cudaEventRecord.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self.lib.cudaEventSynchronize.restype = ctypes.c_int
        self.lib.cudaEventSynchronize.argtypes = [ctypes.c_void_p]
        self.lib.cudaEventElapsedTime.restype = ctypes.c_int
        self.lib.cudaEventElapsedTime.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_void_p, ctypes.c_void_p]
        self.lib.cudaEventDestroy.restype = ctypes.c_int
        self.lib.cudaEventDestroy.argtypes = [ctypes.c_void_p]

        self.lib.cudaMemcpy.restype = ctypes.c_int
        self.lib.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]

    def check(self, status: int, func: str) -> None:
        if status != 0:
            if status < 0:
                raise RuntimeError(f"{func} failed with code {status}")
            msg = self.lib.cudaGetErrorString(status)
            detail = msg.decode("utf-8") if msg else f"code {status}"
            raise RuntimeError(f"{func} failed: {detail}")

    def launch_kernel(
        self,
        func_ptr: int,
        grid: Dim3,
        block: Dim3,
        args: Iterable[int],
        shared_mem: int = 0,
        stream: int = 0,
    ) -> None:
        arg_array = (ctypes.c_void_p * len(list(args)))(*list(args)) if args else None
        self.check(
            self.lib.cudaLaunchKernel(
                ctypes.c_void_p(func_ptr),
                grid,
                block,
                ctypes.cast(arg_array, ctypes.POINTER(ctypes.c_void_p)) if arg_array else None,
                ctypes.c_size_t(shared_mem),
                ctypes.c_void_p(stream),
            ),
            "cudaLaunchKernel",
        )

    def synchronize(self) -> None:
        self.check(self.lib.cudaDeviceSynchronize(), "cudaDeviceSynchronize")

    def memcpy_host_to_device(self, dst: int, src: bytes | np.ndarray, nbytes: int) -> None:
        if isinstance(src, np.ndarray):
            src_ptr = src.ctypes.data_as(ctypes.c_void_p)  # pyright: ignore
        else:
            src_ptr = ctypes.c_char_p(src)
        self.check(
            self.lib.cudaMemcpy(
                ctypes.c_void_p(dst), src_ptr, ctypes.c_size_t(nbytes), ctypes.c_int(CUDA_MEMCPY_HOST_TO_DEVICE)
            ),
            "cudaMemcpy",
        )

    def time_launch(
        self,
        func_ptr: int,
        grid: Dim3,
        block: Dim3,
        args: Iterable[int],
        shared_mem: int = 0,
        stream: int = 0,
    ) -> float:
        start = ctypes.c_void_p()
        stop = ctypes.c_void_p()
        self.check(self.lib.cudaEventCreate(ctypes.byref(start)), "cudaEventCreate")
        self.check(self.lib.cudaEventCreate(ctypes.byref(stop)), "cudaEventCreate")
        try:
            self.check(self.lib.cudaEventRecord(start, ctypes.c_void_p(stream)), "cudaEventRecord")
            self.launch_kernel(func_ptr, grid, block, args, shared_mem, stream)
            self.check(self.lib.cudaEventRecord(stop, ctypes.c_void_p(stream)), "cudaEventRecord")
            self.check(self.lib.cudaEventSynchronize(stop), "cudaEventSynchronize")
            ms = ctypes.c_float()
            self.check(self.lib.cudaEventElapsedTime(ctypes.byref(ms), start, stop), "cudaEventElapsedTime")
            return float(ms.value)
        finally:
            self.lib.cudaEventDestroy(start)
            self.lib.cudaEventDestroy(stop)

    def time_entry_call(self, fn, args: Sequence[ctypes._CData], stream: int = 0) -> float:
        start = ctypes.c_void_p()
        stop = ctypes.c_void_p()
        self.check(self.lib.cudaEventCreate(ctypes.byref(start)), "cudaEventCreate")
        self.check(self.lib.cudaEventCreate(ctypes.byref(stop)), "cudaEventCreate")
        try:
            self.check(self.lib.cudaEventRecord(start, ctypes.c_void_p(stream)), "cudaEventRecord")
            status = fn(*args)
            self.check(status, fn.__name__)
            self.check(self.lib.cudaEventRecord(stop, ctypes.c_void_p(stream)), "cudaEventRecord")
            self.check(self.lib.cudaEventSynchronize(stop), "cudaEventSynchronize")
            ms = ctypes.c_float()
            self.check(self.lib.cudaEventElapsedTime(ctypes.byref(ms), start, stop), "cudaEventElapsedTime")
            return float(ms.value)
        finally:
            self.lib.cudaEventDestroy(start)
            self.lib.cudaEventDestroy(stop)


def resolve_symbol(lib: ctypes.CDLL, symbol: str) -> int:
    with suppress(AttributeError):
        fn = getattr(lib, symbol)
        res = ctypes.cast(fn, ctypes.c_void_p).value
        assert res is not None
        return res

    try:
        res = ctypes.c_void_p.in_dll(lib, symbol).value
        assert res is not None
        return res
    except ValueError as exc:
        raise RuntimeError(f"Kernel symbol '{symbol}' not found in {lib._name}") from exc


@dataclass
class BenchmarkConfig:
    library: Path
    grid: Dim3
    block: Dim3
    shared_mem: int
    warmup: int
    repeat: int
    skip_launch: bool
    cudart: str | None
    entrypoint: str
    stream: int
    seq_len: int
    sync_after: bool


def run_benchmark(cfg: BenchmarkConfig) -> None:
    if not cfg.library.exists():
        raise FileNotFoundError(f"Shared library not found: {cfg.library}")

    lib = ctypes.CDLL(str(cfg.library), mode=ctypes.RTLD_GLOBAL)

    # API symbols
    entry = getattr(lib, cfg.entrypoint, None)
    prepare = getattr(lib, "mk_setup", None)
    get_handles = getattr(lib, "mk_get_handles", None)
    teardown = getattr(lib, "mk_teardown", None)
    reset = getattr(lib, "mk_reset", None)
    symbols = {
        cfg.entrypoint: entry,
        "mk_setup": prepare,
        "mk_get_handles": get_handles,
        "mk_teardown": teardown,
        "mk_reset": reset,
    }
    missing = {name for name, fn in symbols.items() if fn is None}
    if missing:
        found = set(symbols.keys()) - missing
        raise RuntimeError(f"Missing required symbol(s). Found: {found}. Missing: {missing}")
    # redundant but for linters
    assert entry is not None
    assert prepare is not None
    assert get_handles is not None
    assert teardown is not None
    assert reset is not None

    class MkHandles(ctypes.Structure):
        _fields_ = [
            ("params", ctypes.c_void_p),
            ("grads", ctypes.c_void_p),
            ("acts", ctypes.c_void_p),
            ("grad_acts", ctypes.c_void_p),
            ("input_tokens", ctypes.c_void_p),
            ("target_tokens", ctypes.c_void_p),
            ("bar", ctypes.c_void_p),
            ("streams", ctypes.c_void_p),
            ("sm_start_times", ctypes.c_void_p),
            ("sm_end_times", ctypes.c_void_p),
            ("bar_enter_time", ctypes.c_void_p),
            ("bar_exit_time", ctypes.c_void_p),
            ("instr_end_time", ctypes.c_void_p),
            ("bar_size", ctypes.c_int),
            ("batch_size", ctypes.c_int),
            ("seq_len", ctypes.c_int),
        ]

    entry.argtypes = [
        Dim3,
        Dim3,
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
    ]
    entry.restype = ctypes.c_int

    prepare.argtypes = [ctypes.c_int]
    prepare.restype = ctypes.c_int
    get_handles.argtypes = [ctypes.POINTER(MkHandles)]
    get_handles.restype = None
    reset.argtypes = []
    reset.restype = ctypes.c_int

    runtime = CudaRuntime(cfg.cudart)
    print(f"Loaded {cfg.entrypoint} from {cfg.library}")

    try:
        status = prepare(ctypes.c_int(cfg.seq_len))
        runtime.check(status, "mk_setup")

        handles = MkHandles()
        get_handles(ctypes.byref(handles))

        vocab_size = 50257
        host_tokens = np.random.randint(0, vocab_size, size=(handles.batch_size, handles.seq_len), dtype=np.int32)
        host_targets = np.roll(host_tokens, shift=-1, axis=1)
        nbytes = host_tokens.nbytes
        runtime.memcpy_host_to_device(int(handles.input_tokens), host_tokens, nbytes)
        runtime.memcpy_host_to_device(int(handles.target_tokens), host_targets, nbytes)

        status = reset()
        runtime.check(status, "mk_reset")

        if cfg.skip_launch:
            print("Skipping launch.")
            return

        args = (
            cfg.grid,
            cfg.block,
            ctypes.c_size_t(cfg.shared_mem),
            ctypes.c_void_p(cfg.stream),
            ctypes.c_void_p(int(handles.params)),
            ctypes.c_void_p(int(handles.grads)),
            ctypes.c_void_p(int(handles.acts)),
            ctypes.c_void_p(int(handles.grad_acts)),
            ctypes.c_int(int(handles.seq_len)),
            ctypes.c_void_p(int(handles.input_tokens)),
            ctypes.c_void_p(int(handles.target_tokens)),
            ctypes.c_void_p(int(handles.bar)),
            ctypes.c_void_p(int(handles.streams)),
            ctypes.c_void_p(int(handles.sm_start_times)),
            ctypes.c_void_p(int(handles.sm_end_times)),
            ctypes.c_void_p(int(handles.bar_enter_time)),
            ctypes.c_void_p(int(handles.bar_exit_time)),
            ctypes.c_void_p(int(handles.instr_end_time)),
            ctypes.c_int(1 if cfg.sync_after else 0),
        )

        if cfg.warmup:
            for _ in range(cfg.warmup):
                status = entry(*args)
                runtime.check(status, cfg.entrypoint)

        measurements: List[float] = []
        for _ in range(cfg.repeat):
            ms = runtime.time_entry_call(entry, args, cfg.stream)
            measurements.append(ms)

        mean_ms = sum(measurements) / len(measurements)
        p0 = min(measurements)
        p99 = max(measurements)
        print(f"Runs: {len(measurements)}, mean={mean_ms:.3f} ms, min={p0:.3f} ms, max={p99:.3f} ms")
    finally:
        try:
            teardown()
        except RuntimeError as exc:
            print(f"teardown skipped due to CUDA error: {exc}")


def parse_args() -> BenchmarkConfig:
    parser = argparse.ArgumentParser(description="Benchmark megakernel via ctypes entrypoint")
    parser.add_argument(
        "--lib",
        type=Path,
        default=Path("build/tests/libmk_entrypoint.so").resolve(),
        help="Path to shared lib obj exposing the entrypoint",
    )
    parser.add_argument("--entrypoint", default="mk_launch", help="Host entrypoint symbol")
    parser.add_argument("--grid", nargs=3, type=int, default=[MK_NUM_SM, 1, 1], metavar=("X", "Y", "Z"))
    parser.add_argument("--block", nargs=3, type=int, default=[MK_THREADS_PER_BLOCK, 1, 1], metavar=("X", "Y", "Z"))
    parser.add_argument("--shared-mem", type=int, default=MK_SHARED_MEM, help="Shared memory bytes per block")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup launches before timing")
    parser.add_argument("--repeat", type=int, default=5, help="Timed iterations")
    parser.add_argument("--skip-launch", action="store_true", help="Resolve entrypoint without launching")
    parser.add_argument("--cudart", type=str, default=os.environ.get("CUDART_SO"), help="Override libcudart path")
    parser.add_argument("--stream", type=lambda v: int(v, 0), default=0, help="CUDA stream pointer (default stream=0)")
    parser.add_argument(
        "--seq-len", type=int, default=MK_SEQ_LEN, help="Sequence length argument (must match compiled layout)"
    )
    parser.add_argument("--no-sync-after", action="store_true", help="Do not synchronize after launch")
    args = parser.parse_args()

    return BenchmarkConfig(
        library=args.lib,
        grid=Dim3.from_sequence(args.grid),
        block=Dim3.from_sequence(args.block),
        shared_mem=args.shared_mem,
        warmup=args.warmup,
        repeat=args.repeat,
        skip_launch=args.skip_launch,
        cudart=args.cudart,
        entrypoint=args.entrypoint,
        stream=args.stream,
        seq_len=args.seq_len,
        sync_after=not args.no_sync_after,
    )


def main() -> None:
    cfg = parse_args()
    run_benchmark(cfg)


if __name__ == "__main__":
    main()
