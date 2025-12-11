#!/usr/bin/env python3
"""Automate nsys profiling of kernels.

Currently designed for the standard host launch path, not megakernel, since thats what the profiling
  suite is designed for and we only need coarse info until we have respectable perf.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import logging
import re
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Sequence

logger = logging.getLogger(__name__)


def run_checked(cmd: list[str]) -> None:
    """Run a command and raise on failure."""
    logger.info(">> %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def capture_checked(cmd: list[str]) -> str:
    """Capture stdout from a command and raise on failure."""
    logger.info(">> %s ", " ".join(cmd))
    return subprocess.check_output(cmd, text=True)


def _get_git_root() -> Path:
    out = subprocess.check_output(
        ["git", "rev-parse", "--show-toplevel"],
        text=True,
    ).strip()
    p = Path(out)
    assert p.is_dir()
    return p


def git_commit() -> str:
    """Return short git commit hash, or 'unknown'."""
    try:
        return capture_checked(["git", "rev-parse", "--short", "HEAD"]).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


@dataclass
class ProfilePaths:
    """Manage profile output paths."""

    repo_root: Path
    commit: str
    profile_name: str
    output_dir: Path | None = None
    use_template_numbering: bool = True

    def __post_init__(
        self,
    ) -> None:
        self.profile_dir = self.repo_root.joinpath("benchmark/profiles")
        default_dir = self.profile_dir.joinpath(self.commit, self.profile_name)
        self.profile_name = self.profile_name
        self.output_dir = self.output_dir or default_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        suffix = "-%n" if self.use_template_numbering else ""
        # nsys treats --output as a prefix so append %n to let it auto-increment runs.
        self.base_dir = self.output_dir.joinpath(self.profile_name + suffix)
        self.rep = self.base_dir.with_suffix(".nsys-rep")
        self.sqlite = self.base_dir.with_suffix(".sqlite")
        self.logs = self.output_dir.joinpath("profile.log")

    def _resolve_numbered(self, templated: Path) -> Path:
        """Resolve paths that use the %n placeholder (best-effort newest match)."""
        if "%n" not in templated.name:
            return templated

        pattern = templated.name.replace("%n", "*")
        candidates = sorted(templated.parent.glob(pattern))
        if not candidates:
            return templated

        def _extract_index(path: Path) -> int:
            match = re.search(r"(\d+)", path.stem)
            return int(match.group(1)) if match else -1

        return max(candidates, key=lambda p: (_extract_index(p), p.stat().st_mtime))

    @property
    def resolved_rep(self) -> Path:
        return self._resolve_numbered(self.rep)

    @property
    def resolved_sqlite(self) -> Path:
        return self._resolve_numbered(self.sqlite)

    def clean_outputs(self) -> None:
        def _unlink(path: Path) -> None:
            if "%n" in path.name:
                pattern = path.name.replace("%n", "*")
                for candidate in path.parent.glob(pattern):
                    if input(f"rm {candidate}? ").lower() == "y":
                        candidate.unlink()
            elif path.exists():
                if input(f"rm {path}? ").lower() == "y":
                    path.unlink()

        for suffix in (".nsys-rep", ".sqlite", ".json", ".qdstrm"):
            _unlink(self.base_dir.with_suffix(suffix))


def run_and_tee(cmd: list[str], log_path: Path) -> subprocess.CompletedProcess:
    """Run a command, streaming stdout/stderr to console and log."""
    logger.info(">> %s", " ".join(cmd))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    output_lines: list[str] = []
    prev_line = None
    with log_path.open("a") as log_f:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=False)
        assert proc.stdout is not None
        for line in proc.stdout:
            if line.startswith(b"\r"):
                prev_line = line.split(b"\r")[-1].decode()
            else:
                if prev_line:
                    sys.stdout.write(prev_line)
                    log_f.write(prev_line)
                    output_lines.append(prev_line)
                    prev_line = None
                sys.stdout.write(line.decode())
                log_f.write(line.decode())
                output_lines.append(line.decode())
        proc.wait()
    return subprocess.CompletedProcess(cmd, proc.returncode, "".join(output_lines), None)


def _query_one(conn: sqlite3.Connection, sql: str) -> tuple:
    """Helper: run a query expected to return a single row."""
    cur = conn.cursor()
    cur.execute(sql)
    row = cur.fetchone()
    cur.close()
    if row is None:
        return tuple(0 for _ in sql.split("SELECT", 1)[-1].split(","))
    return row


def extract_metrics_from_sqlite(sqlite_path: Path) -> dict[str, float | str]:
    """Extract a few metrics from nsys sqlite export.

    - CUPTI_ACTIVITY_KIND_KERNEL (start, end, demangledName)
    - CUPTI_ACTIVITY_KIND_MEMCPY (start, end)
    - StringIds (id, value)
    """
    metrics: dict[str, float | str] = {
        "total_kernel_time_ns": 0.0,
        "num_kernels": 0.0,
        "avg_kernel_time_ns": 0.0,
        "total_memcpy_time_ns": 0.0,
        "num_memcpys": 0.0,
        "avg_memcpy_time_ns": 0.0,
        "top_kernel_name": "",
        "top_kernel_time_ns": 0.0,
    }

    conn = sqlite3.connect(str(sqlite_path))
    try:
        try:
            total_k_ns, num_k = _query_one(
                conn,
                """
                SELECT
                    COALESCE(SUM(end - start), 0),
                    COALESCE(COUNT(*), 0)
                FROM CUPTI_ACTIVITY_KIND_KERNEL;
                """,
            )
            metrics["total_kernel_time_ns"] = float(total_k_ns)
            metrics["num_kernels"] = float(num_k)
            if num_k > 0:
                metrics["avg_kernel_time_ns"] = float(total_k_ns) / float(num_k)
        except sqlite3.OperationalError:
            pass

        try:
            total_m_ns, num_m = _query_one(
                conn,
                """
                SELECT
                    COALESCE(SUM(end - start), 0),
                    COALESCE(COUNT(*), 0)
                FROM CUPTI_ACTIVITY_KIND_MEMCPY;
                """,
            )
            metrics["total_memcpy_time_ns"] = float(total_m_ns)
            metrics["num_memcpys"] = float(num_m)
            if num_m > 0:
                metrics["avg_memcpy_time_ns"] = float(total_m_ns) / float(num_m)
        except sqlite3.OperationalError:
            pass

        try:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT
                    s.value AS kernel_name,
                    SUM(k.end - k.start) AS total_time_ns
                FROM CUPTI_ACTIVITY_KIND_KERNEL AS k
                JOIN StringIds AS s
                    ON k.demangledName = s.id
                GROUP BY kernel_name
                ORDER BY total_time_ns DESC
                LIMIT 1;
                """
            )
            row = cur.fetchone()
            cur.close()
            if row is not None:
                metrics["top_kernel_name"] = str(row[0])
                metrics["top_kernel_time_ns"] = float(row[1])
        except sqlite3.OperationalError:
            pass
    finally:
        conn.close()

    return metrics


def extract_kernel_summary(sqlite_path: Path) -> list[dict[str, float | str]]:
    """Return per-kernel aggregates (sorted by total time desc)."""
    rows: list[dict[str, float | str]] = []
    conn = sqlite3.connect(str(sqlite_path))
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT
                s.value AS kernel_name,
                COUNT(*) AS launches,
                SUM(k.end - k.start) AS total_ns,
                AVG(k.end - k.start) AS avg_ns,
                MIN(k.end - k.start) AS min_ns,
                MAX(k.end - k.start) AS max_ns
            FROM CUPTI_ACTIVITY_KIND_KERNEL AS k
            JOIN StringIds AS s
                ON k.demangledName = s.id
            GROUP BY kernel_name
            ORDER BY total_ns DESC;
            """
        )
        for name, launches, total_ns, avg_ns, min_ns, max_ns in cur.fetchall():
            rows.append(
                {
                    "kernel_name": name,
                    "launches": int(launches),
                    "total_ns": float(total_ns),
                    "avg_ns": float(avg_ns),
                    "min_ns": float(min_ns),
                    "max_ns": float(max_ns),
                }
            )
        cur.close()
    except sqlite3.OperationalError:
        # kernel table is missing, return empty.
        pass
    finally:
        conn.close()
    return rows


def run_profile(
    nsys: str,
    app_cmd: list[str],
    trace: str,
    exports: Sequence[str],
    paths: ProfilePaths,
) -> subprocess.CompletedProcess:
    export_arg = ",".join(exports)
    profile_cmd = [
        nsys,
        "profile",
        "--stats=true",
        "--export",
        export_arg,
        "--trace",
        trace,
        "--output",
        str(paths.base_dir),
        *app_cmd,
    ]
    return run_and_tee(profile_cmd, paths.logs)


def write_stats(nsys: str, profile_path: Path, out_dir: Path) -> subprocess.CompletedProcess:
    stats_cmd = [nsys, "stats", "--force-export=true", str(profile_path)]
    stats_proc = run_and_tee(stats_cmd, out_dir / "profile.log")
    if stats_proc.stdout:
        (out_dir / "summary.txt").write_text(stats_proc.stdout)
    if stats_proc.returncode != 0:
        logger.warning(f"nsys stats exited with {stats_proc.returncode}")
    return stats_proc


def main() -> None:
    repo_root = _get_git_root()
    commit = git_commit()

    parser = argparse.ArgumentParser(
        description="Profile with nsys.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--nsys-bin", default="nsys", help="path to nsys binary")
    parser.add_argument("--trace", default="cuda,nvtx,osrt", help="comma-separated trace options")
    parser.add_argument(
        "--export",
        default="sqlite",
        help="comma-separated nsys export formats (e.g. sqlite,json)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="override output dir; files are written as <dir>/profile.* (default: benchmark/profiles/<commit>/nsys)",
    )
    parser.add_argument(
        "--template-numbering",
        dest="template_numbering",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="use nsys %%n placeholder to auto-increment outputs.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="clean existing files in the output dir.",
    )
    parser.add_argument(
        "--app",
        default=repo_root.joinpath("build/programs/inference"),
        type=Path,
        help="path to app binary.",
    )
    parser.add_argument(
        "--name",
        type=str,
        required=False,
        help="profile name (default: app name)",
    )
    parser.add_argument(
        "app_args",
        nargs=argparse.REMAINDER,
        help="arguments fwd to the app",
    )
    parser.add_argument(
        "-n",
        "--dry",
        action="store_true",
        help="dry run",
    )
    args = parser.parse_args()
    status = 0
    nsys = args.nsys_bin
    name = args.name or args.app.name
    paths = ProfilePaths(
        repo_root=repo_root,
        commit=commit,
        profile_name=name,
        output_dir=args.output_dir,
        use_template_numbering=args.template_numbering,
    )

    if args.dry:
        print(args)
        import pprint
        pprint.pp(asdict(paths), indent=4)
        exit(status)

    file_handler = logging.FileHandler(filename=paths.logs)
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)

    if args.clean:
        logger.warning("Cleaning outputs")
        paths.clean_outputs()

    export_formats = [fmt.strip() for fmt in args.export.split(",") if fmt.strip()]
    app_cmd = [str(args.app), *args.app_args]
    profile_proc = run_profile(nsys, app_cmd, args.trace, export_formats, paths)
    status |= profile_proc.returncode
    if profile_proc.returncode != 0:
        logger.warning(f"nsys profile exited with {profile_proc.returncode}, continuing to read outputs if present")
    if "sqlite" not in export_formats:
        raise RuntimeError("sqlite export is required for metric extraction; include 'sqlite' in --export")

    rep_path = paths.resolved_rep
    sqlite_path = paths.resolved_sqlite

    if not rep_path.exists():
        raise RuntimeError(f"nsys profile failed and no report produced at {rep_path}")
    if not sqlite_path.exists():
        raise RuntimeError(f"nsys profile did not emit sqlite export at {sqlite_path}")

    stats_proc = write_stats(nsys, rep_path, paths.base_dir.parent)
    status |= stats_proc.returncode

    metrics = extract_metrics_from_sqlite(sqlite_path)

    profiles_dir = repo_root / "benchmark" / "profiles"
    csv_path = profiles_dir / "nsys_metrics.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames: list[str] = ["commit"] + sorted(metrics.keys())
    write_header = not csv_path.exists()

    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        row: dict[str, float | str] = {"commit": commit}
        row.update(metrics)
        writer.writerow(row)

    kernel_rows = extract_kernel_summary(sqlite_path)
    kernel_summary_path = paths.base_dir.parent / "kernel_summary.csv"
    summary_fieldnames = ["commit", "kernel_name", "launches", "total_ns", "avg_ns", "min_ns", "max_ns"]
    with kernel_summary_path.open("w", newline="") as f:
        # if kernel_rows:
        writer = csv.DictWriter(f, fieldnames=summary_fieldnames)
        writer.writeheader()
        for row in kernel_rows:
            row_out = {"commit": commit}
            row_out.update(row)
            writer.writerow(row_out)
        # else:
        #     f.write(",".join(summary_fieldnames) + "\n")

    # Append per-kernel history (flattened with commit)
    history_path = profiles_dir / "nsys_kernel_history.csv"
    write_header_history = not history_path.exists()
    with history_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fieldnames)
        if write_header_history:
            writer.writeheader()
        for row in kernel_rows:
            row_out = {"commit": commit}
            row_out.update(row)
            writer.writerow(row_out)

    logger.info(f"[done] Nsys profile: {rep_path}")
    logger.info(f"[done] Nsys sqlite:  {sqlite_path}")
    logger.info(f"[done] Metrics appended to:    {csv_path}")
    logger.info(f"[done] Logs:  {paths.logs}")
    if status:
        logger.critical("Profile run exited with status %d", profile_proc.returncode)
        logger.critical("Stats export exited with status %d", stats_proc.returncode)
    exit(status)


if __name__ == "__main__":
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.DEBUG)

    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.WARNING)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)sZ [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        handlers=[stdout_handler, stderr_handler],
    )

    main()
