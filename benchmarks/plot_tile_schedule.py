"""Plot measured CTA instruction overlap and the two-page loader/consumer trace."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

regions = json.loads(Path('tile-trace.json').read_text())
pages = json.loads(Path('page-trace.json').read_text())
fig, axes = plt.subplots(3, 1, figsize=(13, 10), constrained_layout=True)
colors = plt.get_cmap('tab10').colors
for ax, graph, title in zip(axes[:2], (regions[0], regions[-1]), ('MLP forward: distinct CTAs', 'MLP backward: distinct CTAs')):
    events, nodes = graph['events'], graph['nodes']
    origin = min(e[0] for e in events)
    workers = sorted(set(e[2] for e in events))[:12]
    for (start, end, worker), node in zip(events, nodes):
        if worker in workers:
            ax.broken_barh([((start-origin)/1000, (end-start)/1000)], (workers.index(worker)-.35, .7),
                           facecolors=colors[node[0] % len(colors)])
    ax.set_yticks(range(len(workers)), [str(w) for w in workers])
    ax.set_ylabel('CTA')
    ax.set_xlabel('Microseconds from region start')
    ax.set_title(title, loc='left')
    handles = [Patch(color=colors[i % len(colors)], label=f'{i}: {label.split(".mlp.")[-1]}')
               for i, label in enumerate(graph['labels'])]
    ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=8)
    ax.grid(axis='x', alpha=.2)
ax = axes[2]
events = pages[0]['events'][:8]
origin = events[0][0]
for row, (load_start, load_end, compute_start, compute_end) in enumerate(events):
    for lane, start, end in ((1, load_start, load_end), (0, compute_start, compute_end)):
        x, width = (start-origin)/1000, (end-start)/1000
        ax.broken_barh([(x, width)], (lane-.25, .5), facecolors=colors[row % 2])
        ax.text(x+width/2, lane, str(row), ha='center', va='center', fontsize=8)
ax.set_yticks([0, 1], ['Add + normalization warps', 'Input loader warps'])
ax.set_ylim(-.6, 1.6)
ax.set_xlabel('Microseconds from first load')
ax.set_title('Within one CTA: two reusable pages, eight rows (labels are row IDs)', loc='left')
ax.legend(handles=[Patch(color=colors[i], label=f'Page {i}') for i in range(2)], loc='upper left', bbox_to_anchor=(1.01, 1))
ax.grid(axis='x', alpha=.2)
fig.savefig('tile-timeline.png', dpi=160)
