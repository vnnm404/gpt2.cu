"""CUDA benchmark utils."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial, wraps
from typing import Any, Generic, Iterator, ParamSpec, TypeVar

from pint import Quantity, UnitRegistry
import torch
import torch.utils.benchmark as torch_benchmark

P = ParamSpec("P")
T = TypeVar("T")

ureg: UnitRegistry = UnitRegistry()


@dataclass(frozen=True, slots=True)
class TimingResult(Generic[T]):
    """Result of a timed op w/ value and elapsed."""
    elapsed: Quantity
    value: T

    @contextmanager
    def precision(self, digits: int) -> Iterator[None]:
        """Temporarily set display precison."""
        reg = self.elapsed._REGISTRY
        fmt = f"~.{digits}fP" # abbrev units (~), set mag digits (.Nf), display fmt (P)
        formatter = reg.formatter

        old = formatter.default_format
        formatter.default_format = fmt
        try:
            yield
        finally:
            formatter.default_format = old


class CudaTimer:
    """Context manager, decorator, and helpers for timing CUDA ops."""

    def __init__(
        self,
        device: torch.device | str | int | None = None,
        stream: torch.cuda.Stream | None = None,
    ) -> None:
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, (str, int)):
            device = torch.device(device)

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        if device.type != "cuda":
            raise RuntimeError("CudaTimer requires a CUDA device")

        torch.cuda.init()

        self.device = device
        self.stream = stream or torch.cuda.default_stream(device=self.device)
        self.start: torch.cuda.Event | None = None
        self.end: torch.cuda.Event | None = None
        self.elapsed: Quantity | None = None

    def synchronize(self) -> None:
        """Synchronize CUDA device."""
        torch.cuda.synchronize(device=self.device)

    def benchmark_timer(
        self,
        op: Callable[[], T],
        *,
        label: str | None = None,
        sub_label: str | None = None,
        description: str | None = None,
        num_threads: int = 1,
        env: Any | None = None,
    ) -> tuple[torch_benchmark.Timer, Callable[[], T]]:
        """Build a `torch.utils.benchmark.Timer` that respects the selected device/stream.

        Returns the Timer and wrapped callable so callers can warm up or invoke it directly.
        """
        def run_op() -> T:
            with torch.cuda.device(self.device), torch.cuda.stream(self.stream):
                return op()

        bench_timer = torch_benchmark.Timer(
            stmt="op()",
            globals={"op": run_op},
            num_threads=num_threads,
            label=label,
            sub_label=sub_label,
            description=description,
            env=env,
        )
        return bench_timer, run_op

    def __enter__(self) -> CudaTimer:
        """Create events and record start."""
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
        self.start.record(stream=self.stream)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Record end event, sync, and calculate elapsed time."""
        if exc_type is None and self.end is not None:
            self.end.record(stream=self.stream)
            self.synchronize()
            assert self.start is not None
            elapsed_ms = self.start.elapsed_time(self.end)
            self.elapsed = elapsed_ms * ureg.millisecond

    def __call__(self, func: Callable[P, T]) -> Callable[P, TimingResult[T]]:
        """Decorator."""
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> TimingResult[T]:
            with self:
                result = func(*args, **kwargs)
            assert self.elapsed is not None
            return TimingResult(elapsed=self.elapsed, value=result)
        return wrapper

    @classmethod
    def time_fn(
        cls,
        op: Callable[[], T],
        device: torch.device | str | int | None = None,
        stream: torch.cuda.Stream | None = None,
    ) -> TimingResult[T]:
        """Time an op once."""
        with cls(device=device, stream=stream) as timer:
            result = op()

        assert timer.elapsed is not None
        return TimingResult(elapsed=timer.elapsed, value=result)

    @classmethod
    def benchmark(
        cls,
        op: Callable[[], T],
        *,
        device: torch.device | str | int | None = None,
        stream: torch.cuda.Stream | None = None,
        min_run_time: float = 0.5,
        warmup_runs: int = 5,
        label: str | None = None,
        sub_label: str | None = None,
        description: str | None = None,
        num_threads: int = 1,
        env: Any | None = None,
    ) -> torch_benchmark.Measurement:
        """Use `torch.utils.benchmark` to autotune iteration count and collect stats."""
        timer = cls(device=device, stream=stream)
        bench_timer, wrapped_op = timer.benchmark_timer(
            op,
            label=label,
            sub_label=sub_label,
            description=description,
            num_threads=num_threads,
            env=env,
        )

        for _ in range(warmup_runs):
            wrapped_op()

        return bench_timer.blocked_autorange(min_run_time=min_run_time)


def test_timer() -> None:
    """Time things."""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping examples")
        return

    device = torch.device("cuda")
    x = torch.randn(1000, 1000, device=device)
    y = torch.randn(1000, 1000, device=device)

    with CudaTimer(device=device) as timer:
        _ = torch.matmul(x, y)
        timer.synchronize()
    print(f"Elapsed: {timer.elapsed}\n")

    result = CudaTimer.time_fn(partial(torch.matmul, x, y), device=device)
    print(f"Elapsed: {result.elapsed}")
    print(f"Result shape: {result.value.shape}\n")

    measurement = CudaTimer.benchmark(partial(torch.matmul, x, y), device=device, label="matmul")
    print(f"Benchmark median: {measurement.median * ureg.second}")
    print(measurement)

    with result.precision(2):
        print(result.elapsed.to("s"))
        print(result.elapsed.to("us"))
        print(result.elapsed.to("ms"))

    timer = CudaTimer(device=device)
    timed_result = timer(torch.matmul)(x, y)
    print(f"Elapsed: {timed_result.elapsed}")
    print(f"Result shape: {timed_result.value.shape}\n")
    timer.synchronize()


if __name__ == "__main__":
    test_timer()
