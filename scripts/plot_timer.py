#!/usr/bin/env python3
"""
Plot timer data as colored rectangles showing instruction execution across SMs.

Input file format (timer.txt):
=== Step 0 ===
SM0:
  instr 0: bar_enter=1024, bar_exit=1024, instr_end=317440, spin_wait=0, exec_time=0
  instr 1: bar_enter=318464, bar_exit=509952, instr_end=1514496, spin_wait=0, exec_time=1
  ...
SM1:
  instr 0: bar_enter=1024, bar_exit=1024, instr_end=318464, spin_wait=0, exec_time=0
  ...
=== Step 1 ===
...

The plot shows:
- Y axis: SM index
- X axis: time (from bar_enter to instr_end)
- Colored rectangles representing instruction execution
- Dotted vertical line inside each rectangle at bar_exit position
- Instruction number printed inside rectangle
- Spin wait time printed if > 0
"""

import re
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
from collections import defaultdict

INPUT_FILE = "timer.txt"
OUTPUT_PNG = "timer_visualization.png"


def parse_timer_file(filename, step=0):
    """Parse timer.txt and extract data for the specified step."""
    with open(filename, "r") as f:
        lines = f.readlines()

    sm_data = {}  # sm_index -> list of (instr_id, bar_enter, bar_exit, instr_end, spin_wait)
    current_step = None
    current_sm = None
    target_step_found = False

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Check for step header
        step_match = re.match(r"=== Step (\d+) ===", line)
        if step_match:
            current_step = int(step_match.group(1))
            if current_step == step:
                target_step_found = True
            elif target_step_found:
                # We've moved past our target step, stop parsing
                break
            continue

        # Skip if not in target step
        if current_step != step:
            continue

        # Check for SM header
        sm_match = re.match(r"SM(\d+):", line)
        if sm_match:
            current_sm = int(sm_match.group(1))
            sm_data[current_sm] = []
            continue

        # Parse instruction line
        instr_match = re.match(
            r"instr (\d+): bar_enter=(\d+), bar_exit=(\d+), instr_end=(\d+), spin_wait=(\d+), exec_time=(\d+)",
            line
        )
        if instr_match and current_sm is not None:
            instr_id = int(instr_match.group(1))
            bar_enter = int(instr_match.group(2))
            bar_exit = int(instr_match.group(3))
            instr_end = int(instr_match.group(4))
            spin_wait = int(instr_match.group(5))
            # exec_time is parsed but not used per requirement
            sm_data[current_sm].append((instr_id, bar_enter, bar_exit, instr_end, spin_wait))

    return sm_data


def plot_timer_data(sm_data, save_png=True, base_width=2.0, scale_power=1.5):
    """
    Plot the timer data as colored rectangles.
    
    Args:
        sm_data: Parsed SM data
        save_png: Whether to save as PNG
        base_width: Base width for each instruction rectangle
        scale_power: Power for sublinear scaling (0.3 = cube root like, 0.5 = sqrt)
                     Lower values = more uniform widths
    """
    if not sm_data:
        raise RuntimeError("No data found for the specified step.")

    # Find global time range and collect all instruction durations
    all_durations = []
    for instructions in sm_data.values():
        for _, bar_enter, _, instr_end, _ in instructions:
            duration = instr_end - bar_enter
            if duration > 0:
                all_durations.append(duration)

    if not all_durations:
        raise RuntimeError("No instruction data found.")

    # Compute sublinear scaled widths
    # Formula: visual_width = base_width + (duration / max_duration)^scale_power * additional_width
    max_duration = max(all_durations)
    min_duration = min(all_durations)
    
    def scale_width(duration):
        """Convert actual duration to visual width using sublinear scaling."""
        if max_duration == min_duration:
            return base_width
        # Normalize to 0-1, apply power scaling, then scale to visual range
        normalized = (duration - min_duration) / (max_duration - min_duration)
        # base_width is minimum, and we add up to 3x base_width for longest
        return base_width + (normalized ** scale_power) * (base_width * 3)

    # Build a mapping of (sm_idx, instr_idx) -> (visual_x_start, visual_width)
    # We need to lay out instructions sequentially in visual space
    visual_positions = {}  # (sm_idx, instr_order) -> (visual_x_start, visual_width, bar_exit_ratio)
    max_visual_x = 0
    
    for sm_idx in sorted(sm_data.keys()):
        instructions = sm_data[sm_idx]
        visual_x = 0
        for i, (instr_id, bar_enter, bar_exit, instr_end, spin_wait) in enumerate(instructions):
            duration = instr_end - bar_enter
            if duration <= 0:
                continue
            
            visual_width = scale_width(duration)
            
            # Calculate bar_exit position as ratio within the rectangle
            if instr_end > bar_enter:
                bar_exit_ratio = (bar_exit - bar_enter) / (instr_end - bar_enter)
            else:
                bar_exit_ratio = 0
            
            visual_positions[(sm_idx, i)] = (visual_x, visual_width, bar_exit_ratio)
            visual_x += visual_width
        
        max_visual_x = max(max_visual_x, visual_x)

    # Collect all unique instruction IDs for coloring
    all_instr_ids = set()
    for instructions in sm_data.values():
        for instr_id, _, _, _, _ in instructions:
            all_instr_ids.add(instr_id)
    all_instr_ids = sorted(all_instr_ids)

    # Color map based on instruction ID
    cmap = plt.colormaps.get_cmap("tab20")
    color_map = {instr_id: cmap(instr_id % cmap.N) for instr_id in all_instr_ids}

    # Figure sizing
    num_sms = max(sm_data.keys()) + 1
    fig_height = max(8, num_sms * 0.4 + 2)
    fig_width = max(20, max_visual_x * 0.15)  # Scale figure width with content
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    block_height = 0.8  # vertical size of rectangle per SM row

    # Draw rectangles for each SM
    for sm_idx in sorted(sm_data.keys()):
        instructions = sm_data[sm_idx]
        y_center = sm_idx

        for i, (instr_id, bar_enter, bar_exit, instr_end, spin_wait) in enumerate(instructions):
            if (sm_idx, i) not in visual_positions:
                continue
                
            visual_x_start, visual_width, bar_exit_ratio = visual_positions[(sm_idx, i)]
            y_bottom = y_center - block_height / 2.0

            # Draw the main rectangle
            rect = patches.Rectangle(
                (visual_x_start, y_bottom),
                visual_width,
                block_height,
                linewidth=0.5,
                edgecolor="black",
                facecolor=color_map.get(instr_id, (0.8, 0.8, 0.8)),
                zorder=2
            )
            ax.add_patch(rect)

            # Draw dotted line at bar_exit position (scaled)
            if 0 < bar_exit_ratio < 1:
                bar_exit_visual = visual_x_start + visual_width * bar_exit_ratio
                ax.vlines(
                    bar_exit_visual,
                    y_bottom,
                    y_bottom + block_height,
                    colors='black',
                    linestyles='dotted',
                    linewidth=1,
                    zorder=3
                )

            # Calculate text position (center of rectangle)
            x_center = visual_x_start + visual_width / 2.0

            # Prepare text: instruction number, and spin_wait if > 0
            text_parts = [str(instr_id)]
            if spin_wait > 0:
                text_parts.append(f"sw:{spin_wait}")

            text = "\n".join(text_parts)

            # Draw text - with sublinear scaling, most rectangles should be readable
            fontsize = 6
            ax.text(
                x_center, y_center, text,
                ha="center", va="center",
                fontsize=fontsize,
                zorder=4,
                clip_on=True
            )

    # Set axis limits
    padding = max_visual_x * 0.02
    ax.set_xlim(-padding, max_visual_x + padding)
    ax.set_ylim(-1, num_sms)

    # Y axis: SM indices
    ax.set_yticks(sorted(sm_data.keys()))
    ax.set_yticklabels([f"SM{i}" for i in sorted(sm_data.keys())])
    ax.set_ylabel("SM Index")
    ax.set_xlabel("Instruction Sequence (width scaled sublinearly with duration)")
    ax.set_title("Instruction execution timeline per SM (Step 0)")

    ax.grid(axis="y", linestyle="--", alpha=0.3)

    # Build legend with most common instruction IDs
    instr_counts = defaultdict(int)
    for instructions in sm_data.values():
        for instr_id, _, _, _, _ in instructions:
            instr_counts[instr_id] += 1

    # Sort by count descending
    sorted_instrs = sorted(instr_counts.items(), key=lambda kv: -kv[1])
    legend_instrs = [instr_id for instr_id, _ in sorted_instrs[:20]]

    legend_patches = [
        patches.Patch(color=color_map[i], label=f"Instr {i}")
        for i in legend_instrs
    ]
    if legend_patches:
        ax.legend(
            handles=legend_patches,
            bbox_to_anchor=(1.02, 1),
            loc='upper left',
            fontsize=8,
            ncol=1
        )

    plt.tight_layout()

    if save_png:
        plt.savefig(OUTPUT_PNG, dpi=200, bbox_inches="tight")
        print(f"Saved plot to {OUTPUT_PNG}")

    plt.show()


if __name__ == "__main__":
    sm_data = parse_timer_file(INPUT_FILE, step=0)
    print(f"Parsed {len(sm_data)} SMs")
    for sm_idx in sorted(sm_data.keys())[:5]:
        print(f"  SM{sm_idx}: {len(sm_data[sm_idx])} instructions")
    plot_timer_data(sm_data, save_png=True)
