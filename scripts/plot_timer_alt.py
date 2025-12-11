#!/usr/bin/env python3
"""Visualize mk.

Parses timer.txt format:
  === Step k ===
  SM0:
    instr 0: bar_enter=..., bar_exit=..., instr_end=..., spin_wait=..., exec_time=...

Produces:
  Sequential per-SM layout, visual width proportional to duration
  dotted bar_exit position, text labels, and instr_id + sw if applicable
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Any

import polars as pl
import altair as alt

alt.data_transformers.disable_max_rows()


def parse_timer_file(path: Path, step: int = 0) -> pl.DataFrame:
    """Parse timer.txt for a specific step.

    Output columns:
        sm: int
        instr_order: int     # order in this SM row
        instr_id: int
        bar_enter: int
        bar_exit: int
        instr_end: int
        spin_wait: int
        duration: int        # instr_end - bar_enter, must be > 0
    """
    lines = path.read_text().splitlines()

    rows: List[Dict[str, Any]] = []
    current_step: int | None = None
    current_sm: int | None = None
    target_step_found = False
    instr_order_by_sm: Dict[int, int] = {}

    for raw in lines:
        line = raw.strip()
        if not line:
            continue

        # Step header
        step_match = re.match(r"=== Step (\d+) ===", line)
        if step_match:
            s = int(step_match.group(1))
            if s == step:
                target_step_found = True
            elif target_step_found:
                break
            current_step = s
            continue

        if current_step != step:
            continue

        # SM header
        sm_match = re.match(r"SM(\d+):", line)
        if sm_match:
            current_sm = int(sm_match.group(1))
            instr_order_by_sm[current_sm] = 0
            continue

        # Instruction line
        instr_match = re.match(
            r"instr (\d+): bar_enter=(\d+), bar_exit=(\d+), instr_end=(\d+), "
            r"spin_wait=(\d+), exec_time=(\d+)",
            line,
        )
        if instr_match and current_sm is not None:
            instr_id, bar_enter, bar_exit, instr_end, spin_wait, _exec = map(int, instr_match.groups())
            duration = instr_end - bar_enter
            if duration <= 0:
                print("Negative duration:", raw)
                continue  # hmm

            idx = instr_order_by_sm[current_sm]
            instr_order_by_sm[current_sm] += 1

            rows.append(
                {
                    "sm": current_sm,
                    "instr_order": idx,
                    "instr_id": instr_id,
                    "bar_enter": bar_enter,
                    "bar_exit": bar_exit,
                    "instr_end": instr_end,
                    "spin_wait": spin_wait,
                    "duration": duration,
                }
            )

    if not rows:
        raise RuntimeError(f"No data found for step {step}")

    return pl.DataFrame(rows)


def compute_visual_layout(
    df: pl.DataFrame,
    base_width: float = 2.0,
    scale_power: float = 1.5,
) -> pl.DataFrame:
    """Visual layout, duration scaling + sequential placement.

    Adds:
        visual_width
        visual_x_start
        visual_x_end
        visual_x_center
        visual_bar_exit
        visual_x_end_max (per SM)
    """
    if df.is_empty():
        raise RuntimeError("No instructions found.")

    min_duration, max_duration = df.select(
        pl.col("duration").min().alias("min_duration"),
        pl.col("duration").max().alias("max_duration"),
    ).row(0)

    # width scaling
    def _scale_width(d: int) -> float:
        if max_duration == min_duration:
            return base_width
        norm = (d - min_duration) / (max_duration - min_duration)
        return base_width + (norm**scale_power) * (base_width * 3.0)

    df = df.with_columns(pl.col("duration").map_elements(_scale_width, return_dtype=pl.Float64).alias("visual_width"))

    # assign sequential x per SM
    def _assign_visual_x(group: pl.DataFrame) -> pl.DataFrame:
        group = group.sort("instr_order")
        widths = group["visual_width"].to_list()

        x_starts = []
        current = 0.0
        for w in widths:
            x_starts.append(current)
            current += w

        denom = (group["instr_end"] - group["bar_enter"]).cast(pl.Float64)
        ratio = ((group["bar_exit"] - group["bar_enter"]).cast(pl.Float64) / denom).fill_null(0.0)
        ratio = ratio.clip(0.0, 1.0)

        visual_bar_exit = pl.Series(
            "visual_bar_exit",
            [xs + w * r for xs, w, r in zip(x_starts, widths, ratio.to_list())],
        )

        return group.with_columns(
            pl.Series("visual_x_start", x_starts),
            visual_bar_exit,
        )

    df = df.group_by("sm", maintain_order=True).map_groups(_assign_visual_x)

    # derived geometry
    df = df.with_columns(
        (pl.col("visual_x_start") + pl.col("visual_width")).alias("visual_x_end"),
        (pl.col("visual_x_start") + 0.5 * pl.col("visual_width")).alias("visual_x_center"),
        (pl.col("visual_x_start") + pl.col("visual_width")).max().over("sm").alias("visual_x_end_max"),
        (
            pl.col("instr_id").cast(pl.Utf8)
            + pl.when(pl.col("spin_wait") > 0)
            .then(pl.lit("/") + pl.col("spin_wait").cast(pl.Utf8))
            .otherwise(pl.lit(""))
        ).alias("label"),
    )

    return df


def plot_timer_data_altair(
    df: pl.DataFrame,
    title: str,
    max_instr_legend: int = 20,
) -> alt.Chart:
    base = alt.Chart(df, title=title)

    rects = base.mark_rect(opacity=0.85, stroke="black", strokeWidth=0.4).encode(
        x=alt.X(
            "visual_x_start:Q",
            # altair doesnt support latex right now. sad.
            # title=r"Instruction sequence (width $\prop$ duration, sublinear)",
            title="Instruction sequence (width ∝ duration, sublinear)",
        ),
        x2="visual_x_end:Q",
        y=alt.Y("sm:O", title="SM Index", sort="ascending"),
        color=alt.Color(
            "instr_id:N",
            legend=alt.Legend(symbolLimit=max_instr_legend, title="Instr ID"),
        ),
        tooltip=[
            "sm:O",
            "instr_id:O",
            "instr_order:O",
            "duration:Q",
            "bar_enter:Q",
            "bar_exit:Q",
            "instr_end:Q",
            "spin_wait:Q",
        ],
    )

    rules = base.mark_rule(strokeDash=[4, 4], strokeWidth=1).encode(
        x="visual_bar_exit:Q",
        y="sm:O",
        y2="sm:O",
    )

    text = base.mark_text(fontSize=8).encode(
        x="visual_x_center:Q",
        y="sm:O",
        text="label:N",
    )

    # Sizing
    n_sms = df["sm"].n_unique()
    height = max(800, int(n_sms * 10))
    max_x = float(df.select(pl.col("visual_x_end").max()).item())
    width = max(1000, int(max_x * 10))
    print(f"{height=} {width=} {max_x=}")
    return (rects + rules + text).properties(width=width, height=height)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("timer.txt"))
    parser.add_argument("--step", type=int, default=0)
    parser.add_argument("--out", type=Path, help="Output html file path")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args(argv)

    df = parse_timer_file(args.input, args.step)

    with pl.Config(
        tbl_formatting="ASCII_MARKDOWN",
        tbl_hide_column_data_types=True,
        tbl_hide_dataframe_shape=True,
    ):
        print(df.head())
        print(
            df.group_by(pl.col("instr_id"), maintain_order=True).agg(
                pl.col("duration").min().alias("min"),
                pl.col("duration").quantile(0.25).alias("p25"),
                pl.col("duration").median().alias("median"),
                pl.col("duration").quantile(0.75).alias("p75"),
                pl.col("duration").max().alias("max"),
            )
        )
        print(
            df.select(
                pl.col("duration").min().alias("min"),
                pl.col("duration").quantile(0.25).alias("p25"),
                pl.col("duration").median().alias("median"),
                pl.col("duration").quantile(0.75).alias("p75"),
                pl.col("duration").max().alias("max"),
            )
        )

    df_vis = compute_visual_layout(df)

    chart = plot_timer_data_altair(
        df_vis,
        title=f"Instruction execution timeline per SM (Step {args.step})",
    )

    out_path = args.out or args.input.with_suffix(".html")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    chart.save(out_path, format="html")
    print(f"Saved to {out_path}")

    if args.show:
        chart.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
