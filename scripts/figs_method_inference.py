"""Figure 3.4 — Statistical inference rigour stack.

Five horizontal bands showing the cumulative filtering applied between
raw per-cell measurements and the headline weighting-robust empirical
claims. The figure makes the inferential framework auditable: each band
states the reduction step, the parameters that govern it, and the input
and output cardinalities.

Output: thesis/figures/method_inference_stack.pdf
"""
import os
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _plot_style import PALETTE as P, apply_style

apply_style()

fig, ax = plt.subplots(figsize=(13.0, 9.5))
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.set_axis_off()

def draw_band(y, w, label_left, label_main, label_right,
              colour, alpha=0.18, fontsize_main=9.5, lw=1.2):
    """Horizontal band with left input, centre title, right output."""
    box = FancyBboxPatch(
        (10, y - 4), w, 8,
        boxstyle='round,pad=0.18,rounding_size=0.5',
        linewidth=lw, edgecolor=colour, facecolor=colour, alpha=alpha,
    )
    ax.add_patch(box)
    ax.text(13, y, label_left, ha='left', va='center',
            fontsize=8.5, color=P['ink'], style='italic')
    ax.text(50, y, label_main, ha='center', va='center',
            fontsize=fontsize_main, color=P['ink'], weight='bold')
    ax.text(87, y, label_right, ha='right', va='center',
            fontsize=8.5, color=P['ink'], style='italic')

def draw_filter_arrow(y_from, y_to, label, side='right'):
    if side == 'right':
        x = 92
    else:
        x = 8
    arrow = FancyArrowPatch(
        (x, y_from), (x, y_to),
        arrowstyle='-|>,head_width=2.0,head_length=3.5',
        linewidth=1.2, color=P['ink_soft'],
    )
    ax.add_patch(arrow)
    ax.text(x + 1.5 if side == 'right' else x - 1.5,
            (y_from + y_to) / 2, label,
            ha='left' if side == 'right' else 'right',
            va='center', fontsize=7.5, color=P['ink_soft'],
            style='italic')

# ----------------------------------------------------------------------
# Title
# ----------------------------------------------------------------------
ax.text(50, 96,
        'Statistical inference rigour stack',
        ha='center', va='top', fontsize=12, weight='bold', color=P['ink'])
ax.text(50, 92,
        'Each layer reduces the universe of candidate empirical claims; '
        'only the surviving claims appear in the headline.',
        ha='center', va='top', fontsize=8.5,
        color=P['ink_soft'], style='italic')

# ----------------------------------------------------------------------
# Five layers (top is rawest, bottom is the final filtered headline)
# ----------------------------------------------------------------------
y_layers = [82, 70, 58, 46, 34, 22]

# Layer 0: raw per-seed measurements
draw_band(y_layers[0], 80,
          'Layer 0',
          'Per-seed metric values',
          'audit.csv / procedural.csv',
          colour=P['data'], alpha=0.10)
ax.text(50, y_layers[0] - 6.0,
        '$N = 30$ seeds on the three primary HR datasets and across all '
        'six datasets for procedural fairness; $N = 20$ on ACS-Income / Dutch Census.',
        ha='center', va='top', fontsize=8.0, color=P['ink_soft'])

# Layer 1: bootstrap CI
draw_band(y_layers[1], 80,
          'Layer 1',
          'Percentile bootstrap on the mean',
          '$n_{\\mathrm{boot}} = 10{,}000$ resamples',
          colour=P['class_accent'], alpha=0.14)
ax.text(50, y_layers[1] - 6.0,
        'Per (dataset, model, method, $\\lambda$, metric) cell: bootstrap mean + 95\\,\\% CI.',
        ha='center', va='top', fontsize=8.0, color=P['ink_soft'])

# Layer 2: pairwise diff + Cohen's d
draw_band(y_layers[2], 80,
          'Layer 2',
          "Pairwise effect size (Cohen's $d$, variance-floored)",
          'paired-bootstrap differences',
          colour=P['accent'], alpha=0.14)
ax.text(50, y_layers[2] - 6.0,
        'Variance floor at $\\sigma_{\\mathrm{min}} = 10^{-3}$ prevents inflated $d$ '
        'when one arm is deterministic across seeds.',
        ha='center', va='top', fontsize=8.0, color=P['ink_soft'])

# Layer 3: weighting schemes
draw_band(y_layers[3], 80,
          'Layer 3',
          'Five weighting-scheme sensitivity sweep',
          'equal / voice / transparency / consistency / rank',
          colour=P['plum'], alpha=0.14)
ax.text(50, y_layers[3] - 6.0,
        'Each pairwise test re-evaluated under five aggregations of '
        'the procedural-fairness metric components.',
        ha='center', va='top', fontsize=8.0, color=P['ink_soft'])

# Layer 4: Holm-Bonferroni FWER
draw_band(y_layers[4], 80,
          'Layer 4',
          'Holm-Bonferroni FWER step-down at $\\alpha = 0.05$',
          '930 tests $\\rightarrow$ 761 survivors',
          colour=P['highlight'], alpha=0.14)
ax.text(50, y_layers[4] - 6.0,
        '450 procedural-gap + 450 separability-gap + 30 rank-disagreement tests, '
        'pooled into one family for strict control.',
        ha='center', va='top', fontsize=8.0, color=P['ink_soft'])

# Layer 5: weighting-robust filter (headline)
draw_band(y_layers[5], 80,
          'Headline',
          'Weighting-robust filter (survives Holm under all five schemes)',
          '46 hard-core procedural-gap pairs',
          colour=P['highlight'], alpha=0.30, lw=1.8, fontsize_main=10)

# Filtering arrows on the right
for yfrom, yto in zip(y_layers[:-1], y_layers[1:]):
    arrow = FancyArrowPatch(
        (50, yfrom - 4), (50, yto + 4),
        arrowstyle='-|>,head_width=2.0,head_length=3.5',
        linewidth=1.0, color=P['ink_soft'],
    )
    ax.add_patch(arrow)

# ----------------------------------------------------------------------
# Right-margin annotation: parallel separability test
# ----------------------------------------------------------------------
sep_box = FancyBboxPatch(
    (76, 10), 22, 6,
    boxstyle='round,pad=0.18,rounding_size=0.5',
    linewidth=1.2, edgecolor=P['plum'], facecolor=P['card'],
)
ax.add_patch(sep_box)
ax.text(87, 14,
        'Spearman-rank disagreement test',
        ha='center', va='center', fontsize=8.5, weight='bold', color=P['plum'])
ax.text(87, 11.5,
        'Per-dataset $H_0\\colon \\rho=1$ rejected at $p_{\\mathrm{Holm}} = 0$ on all six datasets.',
        ha='center', va='center', fontsize=7.5, color=P['ink_soft'])

# ----------------------------------------------------------------------
# Save
# ----------------------------------------------------------------------
THESIS_FIG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'thesis', 'figures',
)
os.makedirs(THESIS_FIG_DIR, exist_ok=True)
out_path = os.path.join(THESIS_FIG_DIR, 'method_inference_stack.pdf')
fig.savefig(out_path, bbox_inches='tight', pad_inches=0.20, transparent=True)
print(f'Wrote {out_path}')
