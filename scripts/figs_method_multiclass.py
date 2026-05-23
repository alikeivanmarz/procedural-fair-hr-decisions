"""Figure 3.6 — Multi-class extension reduction proof (visual).

Two panels showing how the macro-averaged Equalised-Odds extension reduces
to the canonical binary form when the per-class TPR and FPR group-gaps are
symmetric. Left panel: the binary case as a single $(|\\Delta\\mathrm{TPR}|,
|\\Delta\\mathrm{FPR}|)$ point on a unit-square diagram, with the binary
EOdds value as the maximum coordinate. Right panel: the three-class case
with three such points; the macro form is the average of the per-class
maxima, and the equality with the binary form holds along the diagonal.

Output: thesis/figures/method_multiclass_proof.pdf
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch
from matplotlib.gridspec import GridSpec

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _plot_style import PALETTE as P, apply_style

apply_style()

fig = plt.figure(figsize=(13.0, 6.5))
gs = GridSpec(1, 2, width_ratios=[1, 1], wspace=0.20, figure=fig)

# ====================================================================
# PANEL 1: Binary case
# ====================================================================
ax1 = fig.add_subplot(gs[0])
ax1.set_xlim(-0.05, 0.45)
ax1.set_ylim(-0.05, 0.45)
ax1.set_aspect('equal')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.set_xlabel('$|\\Delta\\mathrm{FPR}|$', fontsize=10.5)
ax1.set_ylabel('$|\\Delta\\mathrm{TPR}|$', fontsize=10.5)
ax1.set_title('Binary EOdds (one point)',
              fontsize=11, weight='bold', color=P['ink'], pad=10)

# Symmetric-error line y = x
ax1.plot([0, 0.45], [0, 0.45], color=P['ink_soft'], linewidth=1.0,
         linestyle='--', label='symmetric error: $|\\Delta\\mathrm{TPR}| = |\\Delta\\mathrm{FPR}|$')

# Example binary point (asymmetric, off the diagonal)
x_b, y_b = 0.18, 0.30
ax1.scatter([x_b], [y_b], s=130, color=P['highlight'], edgecolor=P['ink'],
            linewidth=1.0, zorder=5)
# Annotation lines
ax1.plot([x_b, x_b], [0, y_b], color=P['highlight'], linewidth=1.0,
         linestyle=':', alpha=0.7)
ax1.plot([0, x_b], [y_b, y_b], color=P['highlight'], linewidth=1.0,
         linestyle=':', alpha=0.7)
ax1.text(x_b + 0.012, y_b + 0.015,
         'binary EOdds $= \\max(|\\Delta\\mathrm{TPR}|, |\\Delta\\mathrm{FPR}|)$\n'
         '$\\quad = |\\Delta\\mathrm{TPR}|$ here',
         ha='left', va='bottom', fontsize=8.5, color=P['ink'])

# Hardt 2016 reference
ax1.text(0.40, 0.04, 'Hardt et al.\\ 2016',
         ha='right', va='bottom', fontsize=7.5, color=P['ink_soft'],
         style='italic')

ax1.legend(loc='upper left', fontsize=8.0, frameon=False)
ax1.tick_params(labelsize=9)

# ====================================================================
# PANEL 2: Three-class case (OULAD analogue)
# ====================================================================
ax2 = fig.add_subplot(gs[1])
ax2.set_xlim(-0.05, 0.45)
ax2.set_ylim(-0.05, 0.45)
ax2.set_aspect('equal')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.set_xlabel('$|\\Delta\\mathrm{FPR}_c|$ (per class $c$)', fontsize=10.5)
ax2.set_ylabel('$|\\Delta\\mathrm{TPR}_c|$ (per class $c$)', fontsize=10.5)
ax2.set_title('Macro-EOdds on three classes (three points)',
              fontsize=11, weight='bold', color=P['ink'], pad=10)

ax2.plot([0, 0.45], [0, 0.45], color=P['ink_soft'], linewidth=1.0,
         linestyle='--', label='symmetric error per class')

# Three per-class points (illustrative; not actual data)
points = [
    (0.10, 0.22, 'Pass'),
    (0.32, 0.18, 'Fail'),
    (0.05, 0.04, 'Distinction'),
]
colours = [P['data'], P['accent'], P['plum']]
for (cx, cy, label), col in zip(points, colours):
    ax2.scatter([cx], [cy], s=130, color=col, edgecolor=P['ink'],
                linewidth=1.0, zorder=5)
    ax2.text(cx + 0.012, cy + 0.012, label, fontsize=8.5,
             ha='left', va='bottom', color=col, weight='bold')

# Macro-EOdds: average per-class max
maxes = [max(cx, cy) for cx, cy, _ in points]
macro_eodds = np.mean(maxes)
ax2.axhline(macro_eodds, color=P['highlight'], linewidth=1.4,
            linestyle=':', alpha=0.85,
            label=f'Macro-EOdds = mean of per-class maxima = {macro_eodds:.2f}')

ax2.legend(loc='upper left', fontsize=8.0, frameon=False)
ax2.tick_params(labelsize=9)

# ====================================================================
# Bottom-spanning equality statement
# ====================================================================
fig.text(0.5, 0.02,
         'Reduction theorem (Section~3.4): if every class has '
         '$|\\Delta\\mathrm{TPR}_c| = |\\Delta\\mathrm{FPR}_c|$, then '
         'Macro-EOdds equals the canonical binary EOdds; '
         'in the asymmetric case, Macro-EOdds reduces to the average '
         'absolute odds difference rather than the maximum.',
         ha='center', va='bottom', fontsize=9, color=P['ink_soft'],
         style='italic')

# ----------------------------------------------------------------------
# Save
# ----------------------------------------------------------------------
THESIS_FIG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'thesis', 'figures',
)
os.makedirs(THESIS_FIG_DIR, exist_ok=True)
out_path = os.path.join(THESIS_FIG_DIR, 'method_multiclass_proof.pdf')
fig.savefig(out_path, bbox_inches='tight', pad_inches=0.30, transparent=True)
print(f'Wrote {out_path}')
