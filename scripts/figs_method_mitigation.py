"""Figure 3.5 — Mitigation matrix experimental design.

Top: schematic of the four orthogonal axes of the experimental design
(dataset, base classifier, method, regularisation strength) with
asymmetric seed coverage. Bottom: a heatmap preview on IBM HR Attrition
showing the (12 methods x 7 lambdas) cell space, coloured by mean
fairness change relative to the unmitigated baseline.

The schematic establishes scale; the heatmap establishes interpretation.

Output: thesis/figures/method_mitigation.pdf
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch
from matplotlib.gridspec import GridSpec

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _plot_style import PALETTE as P, apply_style

apply_style()

fig = plt.figure(figsize=(13.0, 10.5))
gs = GridSpec(2, 1, height_ratios=[0.95, 1.05], hspace=0.32, figure=fig)

# ====================================================================
# UPPER PANEL — schematic of the experimental design axes
# ====================================================================
ax = fig.add_subplot(gs[0])
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.set_axis_off()

ax.text(50, 95, 'Mitigation matrix experimental design',
        ha='center', va='top', fontsize=12, weight='bold', color=P['ink'])

# Five axes, drawn as labelled horizontal bars
axes_design = [
    ('Datasets',
     ['IBM HR Attrition', 'OULAD', 'Ricci', 'ACS-Income', 'Dutch Census'],
     [P['class_accent'], P['class_accent'], P['class_accent'],
      P['accent'], P['accent']],
     '5 datasets (3 primary at $N=30$; 2 partial at $N=20$)'),
    ('Base classifiers',
     ['LR', 'RF', 'GB', 'XGB', 'MLP', 'KNN'],
     [P['data']] * 6,
     '6 classifier families'),
    ('Mitigation methods',
     ['Reweighing', 'SMOTE-NC', 'DI-Remover', 'Optim-Pre', 'LFR',
      'Adv-Debias',
      'EOdds-Post', 'Cal-EOdds', 'Reject-Opt'],
     [P['data']] * 5 + [P['class_accent']] * 4 + [P['accent']] * 3,
     '12 methods across pre / in / post stages'),
    ('Regularisation $\\lambda$',
     ['0', '0.05', '0.1', '0.3', '1', '3', '10'],
     [P['plum']] * 7,
     '7-point sweep (geometric)'),
]

Y_TOP = 80
Y_GAP = 16
for i, (axis_name, items, item_colours, summary) in enumerate(axes_design):
    y = Y_TOP - i * Y_GAP

    # Axis name on the LEFT
    ax.text(2, y, axis_name, ha='left', va='center',
            fontsize=10, weight='bold', color=P['ink'])

    # Boxes for items
    n = len(items)
    total_w = 70
    item_w = total_w / n
    x0 = 22
    for k, (item, colour) in enumerate(zip(items, item_colours)):
        cx = x0 + item_w * (k + 0.5)
        box = FancyBboxPatch(
            (cx - item_w * 0.42, y - 2.6), item_w * 0.84, 5.2,
            boxstyle='round,pad=0.12,rounding_size=0.3',
            linewidth=1.0, edgecolor=colour,
            facecolor=colour, alpha=0.18,
        )
        ax.add_patch(box)
        ax.text(cx, y, item, ha='center', va='center',
                fontsize=7.5 if len(item) <= 12 else 7.0,
                color=P['ink'])

    # Summary on the RIGHT
    ax.text(95, y, summary, ha='right', va='center',
            fontsize=8.0, color=P['ink_soft'], style='italic')

# Stage colour key (for methods)
stages_y = 18
ax.text(2, stages_y, 'Stage colour:',
        ha='left', va='center', fontsize=9, weight='bold', color=P['ink'])
for k, (stage, colour) in enumerate([
        ('pre-processing', P['data']),
        ('in-processing',  P['class_accent']),
        ('post-processing', P['accent']),
]):
    bx = 22 + k * 22
    box = FancyBboxPatch((bx, stages_y - 1.5), 4, 3,
                         boxstyle='round,pad=0.10,rounding_size=0.2',
                         linewidth=1.0, edgecolor=colour,
                         facecolor=colour, alpha=0.40)
    ax.add_patch(box)
    ax.text(bx + 5, stages_y, stage, ha='left', va='center',
            fontsize=8.5, color=P['ink'])

# Cell-count summary
ax.text(95, stages_y,
        'Total: $\\approx 50{,}000$ cells; 13 metrics per cell.',
        ha='right', va='center', fontsize=8.5, color=P['ink_soft'],
        style='italic')

# ====================================================================
# LOWER PANEL — heatmap preview on IBM HR Attrition
# ====================================================================
ax2 = fig.add_subplot(gs[1])

# Load N=30 audit data and compute mean fairness change vs lambda=0 baseline
audit_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'results', 'phase5', 'audit_n30.csv',
)
audit = pd.read_csv(audit_path)
df = audit[(audit.dataset == 'ibm_hr_attrition') &
           (audit.metric == 'macro_dp')].copy()

# Pivot: methods x lambdas; mean across base models and seeds
pv = df.groupby(['method', 'lambda_'])['value'].mean().unstack('lambda_')

# Reorder methods by stage (pre, in, post) following the schematic
method_order = ['reweighing', 'smote_nc', 'di_remover', 'optim_preproc', 'lfr',
                'adv_debias', 'prejudice_remover', 'exp_gradient', 'gerryfair',
                'eqodds_postproc', 'calib_eqodds', 'reject_option']
pv = pv.reindex([m for m in method_order if m in pv.index])
lambdas = sorted(pv.columns.tolist())
pv = pv[lambdas]

# Compute change relative to lambda=0 baseline per row
baseline = pv[0.0]
change = pv.subtract(baseline, axis=0)
# Drop the baseline column itself for display
change = change.drop(columns=[0.0])

# Plot heatmap (red = fairness worsens, green = fairness improves; we
# work with macro-DP where LOWER is fairer, so invert sign for the colour map)
data = -change.values  # positive = fairness improvement
vmax = float(np.nanmax(np.abs(data)))
im = ax2.imshow(data, aspect='auto', cmap='RdYlGn',
                vmin=-vmax, vmax=vmax)

ax2.set_xticks(range(len(change.columns)))
ax2.set_xticklabels([f'$\\lambda = {l:g}$' for l in change.columns],
                    fontsize=9)
ax2.set_yticks(range(len(change.index)))
ax2.set_yticklabels(change.index, fontsize=9)

# Stage colour bar on the LEFT margin (per row)
stage_for_method = {
    'reweighing': P['data'], 'smote_nc': P['data'], 'di_remover': P['data'],
    'optim_preproc': P['data'], 'lfr': P['data'],
    'adv_debias': P['class_accent'], 'prejudice_remover': P['class_accent'],
    'exp_gradient': P['class_accent'], 'gerryfair': P['class_accent'],
    'eqodds_postproc': P['accent'], 'calib_eqodds': P['accent'],
    'reject_option': P['accent'],
}
for i, m in enumerate(change.index):
    ax2.add_patch(plt.Rectangle((-0.7, i - 0.4), 0.25, 0.8,
                                facecolor=stage_for_method[m],
                                edgecolor='none',
                                clip_on=False, transform=ax2.transData))

ax2.set_title('Heatmap preview: macro-DP change on IBM HR Attrition '
              '(positive = fairer; mean across 6 base classifiers, 30 seeds)',
              fontsize=10, weight='bold', color=P['ink'], pad=10)
ax2.set_xlabel('regularisation strength', fontsize=9.5, color=P['ink'])
ax2.set_ylabel('mitigation method', fontsize=9.5, color=P['ink'])

# Colour bar
cbar = fig.colorbar(im, ax=ax2, fraction=0.025, pad=0.02, aspect=25)
cbar.set_label('macro-DP change (pp; positive = fairness improvement)',
               fontsize=8.5, color=P['ink'])
cbar.ax.tick_params(labelsize=8)

# ----------------------------------------------------------------------
# Save
# ----------------------------------------------------------------------
THESIS_FIG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'thesis', 'figures',
)
os.makedirs(THESIS_FIG_DIR, exist_ok=True)
out_path = os.path.join(THESIS_FIG_DIR, 'method_mitigation.pdf')
fig.savefig(out_path, bbox_inches='tight', pad_inches=0.20, transparent=True)
print(f'Wrote {out_path}')
