"""Figure 3.5 — Execution and reproducibility architecture.

Vertical pipeline showing how raw data flows through parallel execution to
the final thesis PDF. Highlights the deterministic per-worker single-thread
BLAS contract and the per-cell parquet caching that allows resume-after-kill.

Output: thesis/figures/method_architecture.pdf
"""
import os
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _plot_style import PALETTE as P, apply_style

apply_style()

fig, ax = plt.subplots(figsize=(10, 13))
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.set_axis_off()

def draw_box(x, y, w, h, text, edge_colour, fill_colour=P['card'],
             fontsize=9, weight='normal', lw=1.4, text_colour=None):
    if text_colour is None:
        text_colour = P['ink']
    box = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle='round,pad=0.18,rounding_size=0.6',
        linewidth=lw, edgecolor=edge_colour, facecolor=fill_colour,
    )
    ax.add_patch(box)
    ax.text(x, y, text, ha='center', va='center',
            fontsize=fontsize, color=text_colour, weight=weight, wrap=True)

def draw_arrow_v(x, y_top, y_bot, colour=P['ink'], lw=1.5):
    arrow = FancyArrowPatch(
        (x, y_top), (x, y_bot),
        arrowstyle='-|>,head_width=2.5,head_length=4.0',
        linewidth=lw, color=colour, shrinkA=2, shrinkB=2,
    )
    ax.add_patch(arrow)

# Centre-line x for the main pipeline
CX = 50
W_NORMAL = 60
H_NORMAL = 6
H_BIG = 16

# ----------------------------------------------------------------------
# Stack of pipeline stages, top-to-bottom
# ----------------------------------------------------------------------
stages = [
    # (y, height, text, edge_colour, fontsize, weight)
    (94, H_NORMAL,
     'Raw datasets  (D1-D6)  +  synthetic generator  (D7)',
     P['data'], 10, 'bold'),
    (84, H_NORMAL,
     'Preprocessing  (80/20 stratified split, StandardScaler, '
     'OneHotEncoder)',
     P['ink'], 9.5, 'normal'),
    # Parallel execution (special, drawn separately below)
    (50, H_NORMAL,
     'Per-cell parquet cache  (resume-after-kill)',
     P['accent'], 9.5, 'normal'),
    (40, H_NORMAL,
     'CSV consolidation  (lexicographic sort, byte-stable)',
     P['accent'], 9.5, 'normal'),
    (30, H_NORMAL,
     'Pareto frontier  +  result figures (matplotlib)',
     P['ink'], 9.5, 'normal'),
    (20, H_NORMAL,
     'Thesis PDF  (pdflatex,  Main.pdf)',
     P['data'], 10, 'bold'),
]

for y, h, text, colour, fs, wt in stages:
    draw_box(CX, y, W_NORMAL, h, text,
             edge_colour=colour, fontsize=fs, weight=wt)

# ----------------------------------------------------------------------
# Parallel execution box (centred at y=68)
# ----------------------------------------------------------------------
PARALLEL_Y = 68
PARALLEL_W = 78
PARALLEL_H = 18

# Outer container
big_box = FancyBboxPatch(
    (CX - PARALLEL_W / 2, PARALLEL_Y - PARALLEL_H / 2),
    PARALLEL_W, PARALLEL_H,
    boxstyle='round,pad=0.18,rounding_size=0.6',
    linewidth=2.0, edgecolor=P['class_accent'],
    facecolor=P['card'],
)
ax.add_patch(big_box)

# Title inside
ax.text(CX, PARALLEL_Y + PARALLEL_H / 2 - 3,
        'ProcessPoolExecutor   —   9 workers (M2 Pro 10-core)',
        ha='center', va='center', fontsize=10, weight='bold',
        color=P['class_accent'])

# Three sub-worker boxes
worker_xs = [CX - 24, CX, CX + 24]
worker_labels = ['Worker 1', 'Worker 2', 'Worker N']
for x, label in zip(worker_xs, worker_labels):
    sub = FancyBboxPatch(
        (x - 9, PARALLEL_Y - 5.5), 18, 7,
        boxstyle='round,pad=0.12,rounding_size=0.4',
        linewidth=1.2, edgecolor=P['ink_soft'],
        facecolor=P['bg'],
    )
    ax.add_patch(sub)
    ax.text(x, PARALLEL_Y - 1.0, label,
            ha='center', va='center', fontsize=9, weight='bold',
            color=P['ink'])
    ax.text(x, PARALLEL_Y - 3.5,
            '(dataset, model,\nmethod, $\\lambda$, seed)',
            ha='center', va='center', fontsize=7.2,
            color=P['ink_soft'])

# Annotation below the workers, inside the container
ax.text(CX, PARALLEL_Y - PARALLEL_H / 2 + 1.8,
        'single-thread BLAS per worker  ->  byte-identical determinism',
        ha='center', va='center', fontsize=8.5,
        color=P['ink_soft'], style='italic')

# ----------------------------------------------------------------------
# Vertical arrows between stages
# ----------------------------------------------------------------------
arrow_pairs = [
    (94 - H_NORMAL / 2 + 0.2, 84 + H_NORMAL / 2 - 0.2),    # raw -> preproc
    (84 - H_NORMAL / 2 + 0.2, PARALLEL_Y + PARALLEL_H / 2 - 0.2),  # preproc -> parallel
    (PARALLEL_Y - PARALLEL_H / 2 + 0.2, 50 + H_NORMAL / 2 - 0.2),  # parallel -> cache
    (50 - H_NORMAL / 2 + 0.2, 40 + H_NORMAL / 2 - 0.2),    # cache -> consolidation
    (40 - H_NORMAL / 2 + 0.2, 30 + H_NORMAL / 2 - 0.2),    # consolidation -> figures
    (30 - H_NORMAL / 2 + 0.2, 20 + H_NORMAL / 2 - 0.2),    # figures -> thesis pdf
]
for y_top, y_bot in arrow_pairs:
    draw_arrow_v(CX, y_top, y_bot)

# ----------------------------------------------------------------------
# Left bracket / annotation: "make all" entry point
# ----------------------------------------------------------------------
BRACKET_X = 8
ax.plot([BRACKET_X, BRACKET_X],
        [20 - H_NORMAL / 2, 94 + H_NORMAL / 2],
        color=P['accent'], linewidth=2.5, zorder=0)
ax.plot([BRACKET_X, BRACKET_X + 2],
        [94 + H_NORMAL / 2, 94 + H_NORMAL / 2],
        color=P['accent'], linewidth=2.5)
ax.plot([BRACKET_X, BRACKET_X + 2],
        [20 - H_NORMAL / 2, 20 - H_NORMAL / 2],
        color=P['accent'], linewidth=2.5)
ax.text(BRACKET_X - 2, 57, '“make all”',
        ha='right', va='center',
        rotation=90, fontsize=11, weight='bold', color=P['accent'])
ax.text(BRACKET_X + 1.5, 57,
        'single reproducibility entry point',
        ha='left', va='center',
        rotation=90, fontsize=8.5, color=P['ink_soft'], style='italic')

# ----------------------------------------------------------------------
# Title and footer
# ----------------------------------------------------------------------
ax.text(CX, 99.5,
        'Execution and reproducibility architecture',
        ha='center', va='top', fontsize=12, weight='bold',
        color=P['ink'])

ax.text(CX, 13,
        'Each cell is identified by  (dataset, model, method, $\\lambda$, '
        'seed)  and produces one parquet shard;\n'
        'the consolidator reads all shards, sorts lexicographically, and '
        'writes a byte-stable CSV.',
        ha='center', va='top', fontsize=8.5,
        color=P['ink_soft'], style='italic')

# ----------------------------------------------------------------------
# Save
# ----------------------------------------------------------------------
THESIS_FIG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'thesis', 'figures',
)
os.makedirs(THESIS_FIG_DIR, exist_ok=True)
out_path = os.path.join(THESIS_FIG_DIR, 'method_architecture.pdf')
fig.savefig(out_path, bbox_inches='tight', pad_inches=0.15, transparent=True)
print(f'Wrote {out_path}')
