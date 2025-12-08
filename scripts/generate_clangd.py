from __future__ import annotations

import subprocess
import sys
from pathlib import Path
import functools

run = functools.partial(
    subprocess.run,
    check=False,
    text=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
)


def repo_root() -> Path:
    proc = run(["git", "rev-parse", "--show-toplevel"])
    if proc.returncode != 0:
        print("error: not inside a git repo", file=sys.stderr)
        sys.exit(1)
    return Path(proc.stdout.strip())


def gxx_include_paths(gxx: str = "g++") -> list[str]:
    proc = subprocess.run(
        [gxx, "-x", "c++", "-E", "-v", "-"],
        input="",
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if proc.returncode != 0:
        print(f"warning: failed to probe g++ include paths: {proc.stdout}", file=sys.stderr)
        return []
    lines = proc.stdout.splitlines()
    includes = [ln.strip() for ln in lines if ln.startswith(" /") and "c++" in ln]
    return includes[:2]


def find_cuda_include_path() -> Path | None:
    candidates = [
        Path("/usr/local/cuda/include"),
        Path("/opt/cuda/include"),
    ]
    for cand in candidates:
        if cand.is_dir():
            return cand

    # search common roots for cuda_runtime.h
    search_roots = [Path("/usr/local"), Path("/usr"), Path("/opt")]
    for root in search_roots:
        proc = run(["find", str(root), "-name", "cuda_runtime.h", "-print"])
        if proc.returncode == 0 and proc.stdout:
            return Path(proc.stdout.splitlines()[0]).parent
    return None


def emit_clangd(
    dest: Path,
    include_paths: list[Path],
    cuda_include: Path | None,
    cuda_arch: str | None = None,
) -> None:
    cuda_include_line = f"    - -I{cuda_include}" if cuda_include else ""
    cuda_section = ""
    if cuda_include:
        cuda_section = """

---

If:
  PathMatch: [.*\\.cu, .*\\.cuh]
CompileFlags:
  Add:
    - -xcuda
    - --no-cuda-version-check
"""
        if cuda_arch:
            cuda_section += f"\n{' '*4}- --cuda-gpu-arch={cuda_arch}"

    include_lines = f"\n{' '*4}- -I".join(map(str, [""] + include_paths))
    content = f"""CompileFlags:
  CompilationDatabase: build
  Add:{include_lines}
{cuda_include_line}
  Remove:
    - -forward-unknown-to-host-compiler
    - --generate-code*{cuda_section}

Diagnostics:
  UnusedIncludes: None
  MissingIncludes: None
"""
    dest.write_text(content)


def main() -> None:
    root = repo_root()
    print(f"Generating .clangd in {root}")

    cpp_paths = [Path(p) for p in gxx_include_paths()]
    if not cpp_paths:
        print("warning: no C++ include paths found from g++ probe", file=sys.stderr)

    project_includes = [root / "include", root / "third_party"]
    cuda_path = find_cuda_include_path()
    if cuda_path:
        print(f"Found CUDA include path: {cuda_path}")
    else:
        print("CUDA include path not found, proceeding without CUDA flags", file=sys.stderr)

    emit_clangd(root / ".clangd", cpp_paths + project_includes, cuda_path)
    print("Generated .clangd")
    print("Make sure you have build/compile_commands.json")


if __name__ == "__main__":
    main()
