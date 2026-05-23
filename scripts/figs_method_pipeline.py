"""Figure 3.1 — Research architecture: composite question to asymmetric coverage.

Three-tier diagram. Top: composite master research question and the five
sub-RQs derived from it. Middle: the five contributions C1--C5 paired with
each RQ. Bottom: a dataset-by-contribution coverage matrix that visualises
the role taxonomy and the asymmetric N=30 / N=20 seed coverage.

Output: thesis/figures/method_pipeline.pdf
"""
import os
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _plot_style import PALETTE as P, apply_style

apply_style()

fig, ax = plt.subplots(figsize=(13.0, 10.5))
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.set_axis_off()

def draw_box(x, y, w, h, text, edge_colour, fill_colour=P['card'],
             fontsize=10, weight='normal', lw=1.4, text_colour=None):
    if text_colour is None:
        text_colour = P['ink']
    box = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle='round,pad=0.20,rounding_size=0.7',
        linewidth=lw, edgecolor=edge_colour, facecolor=fill_colour,
    )
    ax.add_patch(box)
    ax.text(x, y, text, ha='center', va='center',
            fontsize=fontsize, color=text_colour, weight=weight, wrap=True)

def draw_arrow(x1, y1, x2, y2, colour=P['ink'], style='-', lw=1.0,
               head=True):
    arrow_style = '-|>,head_width=2.0,head_length=3.5' if head else '-'
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=arrow_style,
        linewidth=lw, color=colour, linestyle=style,
        shrinkA=2, shrinkB=2,
    )
    ax.add_patch(arrow)

# ----------------------------------------------------------------------
# TIER 1: master composite research question
# ----------------------------------------------------------------------
draw_box(50, 94, 86, 7,
         'Can fair AI for multi-class HR-performance evaluation be audited, '
         'operationalised,\n'
         'mitigated, explained, and benchmarked using publicly reproducible methods?',
         edge_colour=P['ink'], fill_colour=P['card'],
         fontsize=10.5, weight='bold', lw=1.8)

ax.text(50, 88, 'Composite master research question',
        ha='center', va='center', fontsize=8.5,
        color=P['ink_soft'], style='italic')

# ----------------------------------------------------------------------
# TIER 2: five sub-RQs (row 1) and five contributions (row 2)
# ----------------------------------------------------------------------
RQ_Y = 80
C_Y = 71

rqs = [
    (12, 'RQ1', 'audit'),
    (32, 'RQ2', 'multi-class'),
    (52, 'RQ3', 'procedural'),
    (72, 'RQ4', 'mitigate'),
    (90, 'RQ5', 'explain'),
]
contributions = [
    (12, 'C1', 'Fairness audit'),
    (32, 'C2', 'Multi-class\nextensions'),
    (52, 'C3', 'Procedural\nfairness'),
    (72, 'C4', 'Mitigation\ncomparison'),
    (90, 'C5', 'Synthetic\nbenchmark'),
]

for x, name, sub in rqs:
    draw_box(x, RQ_Y, 14, 4.5, f'{name}', edge_colour=P['plum'],
             fontsize=10, weight='bold')
    ax.text(x, RQ_Y - 3.5, sub, ha='center', va='top',
            fontsize=8, color=P['ink_soft'], style='italic')

for x, name, body in contributions:
    draw_box(x, C_Y, 16, 7, f'{name}\n{body}', edge_colour=P['accent'],
             fontsize=9, weight='normal')

# Vertical link from master question down to RQs
draw_arrow(50, 90.5, 50, 84, colour=P['ink_soft'], lw=0.9, head=False)
for x, _, _ in rqs:
    draw_arrow(50, 84, x, RQ_Y + 2.5, colour=P['ink_soft'], lw=0.6, head=False)

# Pair each RQ to its contribution
for (xr, _, _), (xc, _, _) in zip(rqs, contributions):
    draw_arrow(xr, RQ_Y - 2.5, xc, C_Y + 3.5, colour=P['accent'],
               lw=1.0, head=True)

# ----------------------------------------------------------------------
# TIER 3: dataset x contribution coverage matrix
# ----------------------------------------------------------------------
# Coverage codes per (dataset, contribution):
#   'P' = primary (carries the headline empirical claim)
#   'S' = secondary (scale validation / triangulation / stress test)
#   'X' = partial coverage (asymmetric N=20 or restricted method scope)
#   '-' = out of scope for this contribution

datasets = [
    ('D1 IBM HR Attrition',          'primary HR'),
    ('D1 IBM HR PerformanceRating',  'leaky-target case'),
    ('D2 ACS-Income',                'scale validation'),
    ('D3 Adult',                     'plumbing baseline'),
    ('D4 Ricci',                     'small-N HR stress'),
    ('D5a Dutch Census',             'triangulation'),
    ('D5b Law School',               'triangulation'),
    ('D6 OULAD',                     'primary multi-class'),
    ('D7 Synthetic',                 'controllable benchmark'),
]
coverage = {
    'D1 IBM HR Attrition':          ['P', '-', 'P', 'P', '-'],
    'D1 IBM HR PerformanceRating':  ['P', '-', 'X', '-', '-'],
    'D2 ACS-Income':                ['S', '-', 'X', 'X', '-'],
    'D3 Adult':                     ['G', '-', '-', '-', '-'],
    'D4 Ricci':                     ['P', '-', 'P', 'P', '-'],
    'D5a Dutch Census':             ['S', '-', 'S', 'X', '-'],
    'D5b Law School':               ['S', '-', '-', '-', '-'],
    'D6 OULAD':                     ['P', 'P', 'P', 'P', '-'],
    'D7 Synthetic':                 ['-', '-', '-', 'X', 'P'],
}

# Matrix layout
MATRIX_LEFT = 32
MATRIX_RIGHT = 92
MATRIX_TOP = 60
MATRIX_BOTTOM = 12
N_ROWS = len(datasets)
N_COLS = 5
COL_W = (MATRIX_RIGHT - MATRIX_LEFT) / N_COLS
ROW_H = (MATRIX_TOP - MATRIX_BOTTOM) / N_ROWS

# Column headers (C1..C5)
for j, (xc, name, _) in enumerate(contributions):
    cx = MATRIX_LEFT + COL_W * (j + 0.5)
    ax.text(cx, MATRIX_TOP + 1.5, name, ha='center', va='bottom',
            fontsize=9.5, weight='bold', color=P['accent'])

# Row labels (datasets) on the LEFT
for i, (label, role) in enumerate(datasets):
    ry = MATRIX_TOP - ROW_H * (i + 0.5)
    ax.text(MATRIX_LEFT - 1.0, ry, label, ha='right', va='center',
            fontsize=9, color=P['ink'])
    ax.text(MATRIX_LEFT - 1.0, ry - 1.6, role, ha='right', va='center',
            fontsize=7.5, color=P['ink_soft'], style='italic')

# Cells
CELL_COLOURS = {
    'P': (P['class_accent'], 1.0,  'full', P['class_accent']),  # solid full
    'S': (P['class_accent'], 0.45, 'full', P['class_accent']),
    'X': (P['accent'],       0.55, 'half', P['accent']),        # half-fill rect
    'G': (P['data'],         0.55, 'full', P['data']),          # gate
    '-': (P['rule'],         0.18, 'none', P['rule']),
}

for i, (label, _) in enumerate(datasets):
    cov = coverage[label]
    for j, code in enumerate(cov):
        cx = MATRIX_LEFT + COL_W * (j + 0.5)
        cy = MATRIX_TOP - ROW_H * (i + 0.5)
        edge_c, alpha, kind, _ = CELL_COLOURS[code]
        cell_w = COL_W * 0.70
        cell_h = ROW_H * 0.80
        cx0 = cx - cell_w / 2
        cy0 = cy - cell_h / 2
        # Always draw a faint background border so empty cells are visible
        bg = Rectangle((cx0, cy0), cell_w, cell_h,
                       facecolor='white', edgecolor=P['rule'], linewidth=0.4)
        ax.add_patch(bg)
        if kind == 'full':
            rect = Rectangle((cx0, cy0), cell_w, cell_h,
                             facecolor=edge_c, alpha=alpha,
                             edgecolor=P['rule'], linewidth=0.4)
            ax.add_patch(rect)
        elif kind == 'half':
            # Diagonal hatch fill for partial coverage
            rect = Rectangle((cx0, cy0), cell_w, cell_h,
                             facecolor=edge_c, alpha=alpha,
                             edgecolor=P['rule'], linewidth=0.4,
                             hatch='////')
            ax.add_patch(rect)
        # 'none' = blank cell with only the faint background

# Legend underneath
LEG_Y = 6
LEG_X0 = 7
items = [
    (P['class_accent'], 1.0,  'full', 'primary'),
    (P['class_accent'], 0.45, 'full', 'secondary'),
    (P['accent'],       0.55, 'half', 'partial coverage'),
    (P['data'],         0.55, 'full', 'plumbing-only baseline'),
    (P['rule'],         0.18, 'none', 'out of scope'),
]
gap = 18
for k, (c, a, kind, label) in enumerate(items):
    lx = LEG_X0 + k * gap
    if kind == 'full':
        rect = Rectangle((lx, LEG_Y - 1.3), 2.6, 2.6,
                         facecolor=c, alpha=a,
                         edgecolor=P['rule'], linewidth=0.5)
    elif kind == 'half':
        rect = Rectangle((lx, LEG_Y - 1.3), 2.6, 2.6,
                         facecolor=c, alpha=a,
                         edgecolor=P['rule'], linewidth=0.5,
                         hatch='////')
    else:
        rect = Rectangle((lx, LEG_Y - 1.3), 2.6, 2.6,
                         facecolor='white',
                         edgecolor=P['rule'], linewidth=0.5)
    ax.add_patch(rect)
    ax.text(lx + 4.0, LEG_Y, label, ha='left', va='center',
            fontsize=8.0, color=P['ink'])

# Asymmetric-N annotation
ax.text(50, 1.5,
        'Seed coverage: $N=30$ on the three primary HR datasets and across all six datasets for the procedural-fairness analysis;\n'
        '$N=20$ on ACS-Income and Dutch Census per the partial-coverage scope boundary.',
        ha='center', va='center', fontsize=8,
        color=P['ink_soft'], style='italic')

# ----------------------------------------------------------------------
# Save
# ----------------------------------------------------------------------
THESIS_FIG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'thesis', 'figures',
)
os.makedirs(THESIS_FIG_DIR, exist_ok=True)
out_path = os.path.join(THESIS_FIG_DIR, 'method_pipeline.pdf')
fig.savefig(out_path, bbox_inches='tight', pad_inches=0.20, transparent=True)
print(f'Wrote {out_path}')
