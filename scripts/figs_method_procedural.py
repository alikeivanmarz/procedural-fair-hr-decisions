"""Figure 3.3 — Procedural-fairness compute pipelines.

Four panels showing the actual data flow for each of the four procedural
metrics introduced in this thesis. Each panel exposes the hyperparameters
and the intermediate quantities so the reader can audit the metric end to
end. The four metrics share a single foundation (the modifiable / immutable
feature partition shown along the bottom).

Output: thesis/figures/method_procedural.pdf
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

def draw_box(x, y, w, h, text, edge_colour=P['highlight'],
             fill_colour=P['card'], fontsize=8.5, weight='normal',
             lw=1.2, text_colour=None):
    if text_colour is None:
        text_colour = P['ink']
    box = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle='round,pad=0.18,rounding_size=0.5',
        linewidth=lw, edgecolor=edge_colour, facecolor=fill_colour,
    )
    ax.add_patch(box)
    ax.text(x, y, text, ha='center', va='center',
            fontsize=fontsize, color=text_colour, weight=weight, wrap=True)

def draw_arrow(x1, y1, x2, y2, colour=P['ink'], lw=1.0, style='-'):
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle='-|>,head_width=2.0,head_length=3.5',
        linewidth=lw, color=colour, linestyle=style,
        shrinkA=2, shrinkB=2,
    )
    ax.add_patch(arrow)

# ----------------------------------------------------------------------
# Panel layout
#   Top-left:    Process Consistency
#   Top-right:   Voice / Representation
#   Bottom-left: Model-Flippability
#   Bottom-right: Explanation-Actionability
# ----------------------------------------------------------------------

PANEL_TOP = 88
PANEL_MID = 50
PANEL_BOT = 18

panels = [
    ('Process Consistency',          22, 70),
    ('Voice / Representation',       72, 70),
    ('Model-Flippability',           22, 32),
    ('Explanation-Actionability',    72, 32),
]

# Panel headers
for title, cx, _ in panels:
    ax.text(cx, 88 if cx == 22 and panels[0][2] == 70 else 88
            if (cx == 72 and panels[1][2] == 70) else 50,
            title, ha='center', va='center', fontsize=10.5,
            weight='bold', color=P['highlight'])

# Easier to draw the headers explicitly per panel:

def panel_header(cx, cy_top, title):
    ax.text(cx, cy_top, title, ha='center', va='center',
            fontsize=10.5, weight='bold', color=P['highlight'])

# Clear text boxes drawn above
ax.cla()
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.set_axis_off()

# Panel separator vertical and horizontal lines (light)
ax.plot([50, 50], [10, 92], color=P['rule'], linewidth=0.6, zorder=0)
ax.plot([4, 96], [50, 50], color=P['rule'], linewidth=0.6, zorder=0)

# ====================================================================
# PANEL 1 (top-left): Process Consistency
# ====================================================================
panel_header(22, 89, 'Process Consistency')
ax.text(22, 86, 'How stable is the prediction under semantic noise?',
        ha='center', va='center', fontsize=8,
        color=P['ink_soft'], style='italic')

# Box 1: input
draw_box(22, 80, 22, 5.5, 'Test sample $x$\n(sample$_n$ = 500 stratified)',
         edge_colour=P['ink'], fontsize=8.5)
# Box 2: noise generator
draw_box(22, 72, 24, 5.5,
         'Gaussian noise on numeric features\n'
         r'$\sigma \in \{0.1,\ 0.3,\ 1.0,\ 3.0\}$;'
         r' 10 perturbations per row',
         edge_colour=P['highlight'], fontsize=8)
# Box 3: prediction shift
draw_box(22, 64, 22, 5.5,
         r'JS divergence $D_{\mathrm{JS}}(\hat{y}_x \,\|\, \hat{y}_{x+\epsilon})$',
         edge_colour=P['highlight'], fontsize=8.5)
# Box 4: aggregate
draw_box(22, 56, 22, 5.5,
         'Mean over rows; bootstrap CI\n(per $\\sigma$, per (model, dataset))',
         edge_colour=P['ink'], fontsize=8.5)

# Arrows
draw_arrow(22, 77.5, 22, 75)
draw_arrow(22, 69.5, 22, 67)
draw_arrow(22, 61.5, 22, 59)

# ====================================================================
# PANEL 2 (top-right): Voice / Representation
# ====================================================================
panel_header(72, 89, 'Voice / Representation')
ax.text(72, 86, 'How much does the model use modifiable features?',
        ha='center', va='center', fontsize=8,
        color=P['ink_soft'], style='italic')

draw_box(72, 80, 24, 5.5, 'Trained model $f$',
         edge_colour=P['ink'], fontsize=8.5)
draw_box(72, 72, 28, 5.5,
         'TreeExplainer SHAP attributions\n'
         '(or KernelExplainer if not tree-based)',
         edge_colour=P['highlight'], fontsize=8)
draw_box(72, 64, 30, 6.5,
         r'Voice = $\sum_{f \in F_\mathrm{mod}} |\phi_f| \,/\, \sum_{f \in F} |\phi_f|$' + '\n'
         r'Voice-Enrichment = Voice $\div$ ($|F_\mathrm{mod}| / |F|$)',
         edge_colour=P['highlight'], fontsize=8)
draw_box(72, 55, 24, 5.5,
         'Mean over seeds; bootstrap CI',
         edge_colour=P['ink'], fontsize=8.5)

draw_arrow(72, 77.5, 72, 75)
draw_arrow(72, 69.5, 72, 67.5)
draw_arrow(72, 61, 72, 58)

# ====================================================================
# PANEL 3 (bottom-left): Model-Flippability
# ====================================================================
panel_header(22, 47, 'Model-Flippability')
ax.text(22, 44, 'Can the prediction be reversed by a sparse counterfactual?',
        ha='center', va='center', fontsize=8,
        color=P['ink_soft'], style='italic')

draw_box(22, 38, 24, 5.5,
         'Test sample $x$\n(sample$_n$ = 30 stratified)',
         edge_colour=P['ink'], fontsize=8.5)
draw_box(22, 30, 26, 6.0,
         'Greedy $k_{\\max}=1$ counterfactual search\n'
         'over numeric features within $\\pm$ 1$\\sigma$',
         edge_colour=P['highlight'], fontsize=8)
draw_box(22, 22, 24, 5.5,
         'Sparsity: minimum features changed\n'
         'Validity: prediction actually flipped',
         edge_colour=P['highlight'], fontsize=8)
draw_box(22, 14, 22, 5.5,
         'Mean over rows; bootstrap CI',
         edge_colour=P['ink'], fontsize=8.5)

draw_arrow(22, 35.5, 22, 33)
draw_arrow(22, 27, 22, 25)
draw_arrow(22, 19.5, 22, 17)

# ====================================================================
# PANEL 4 (bottom-right): Explanation-Actionability
# ====================================================================
panel_header(72, 47, 'Explanation-Actionability')
ax.text(72, 44, 'Does the counterfactual ask only for actionable changes?',
        ha='center', va='center', fontsize=8,
        color=P['ink_soft'], style='italic')

draw_box(72, 38, 24, 5.5,
         'Greedy $k_{\\max}=1$ counterfactual\n'
         '(reuses Model-Flippability output)',
         edge_colour=P['ink'], fontsize=8.5)
draw_box(72, 30, 26, 6.5,
         'Does the flipped feature lie in $F_\\mathrm{mod}$?\n'
         '$\\mathbb{1}[f^{*} \\in F_\\mathrm{mod}]$ per row',
         edge_colour=P['highlight'], fontsize=8)
draw_box(72, 22, 24, 5.5,
         'Fraction over rows; bootstrap CI',
         edge_colour=P['ink'], fontsize=8.5)

draw_arrow(72, 35.5, 72, 33.5)
draw_arrow(72, 26.5, 72, 25)

# ====================================================================
# Foundation bar
# ====================================================================
foundation = FancyBboxPatch(
    (4, 4 - 1.8), 92, 4.5,
    boxstyle='round,pad=0.18,rounding_size=0.5',
    linewidth=1.4, edgecolor=P['class_accent'],
    facecolor=P['card'],
)
ax.add_patch(foundation)
ax.text(8, 4,
        'Foundation:',
        ha='left', va='center', fontsize=9.5, weight='bold',
        color=P['class_accent'])
ax.text(28, 4,
        '$F = F_\\mathrm{mod} \\cup F_\\mathrm{imm}$ — modifiable / immutable feature partition'
        ' (ascribed-vs-achieved framing of Newman (2020), see Section 3.5).',
        ha='left', va='center', fontsize=8.2, color=P['ink'])

# Light vertical pointers from each metric panel down to the foundation
for cx in [22, 72]:
    arrow = FancyArrowPatch(
        (cx + 13, 11), (cx + 13, 7.5),
        arrowstyle='-|>,head_width=1.5,head_length=2.5',
        linewidth=0.8, color=P['rule'],
    )
    ax.add_patch(arrow)

# ====================================================================
# Title
# ====================================================================
ax.text(50, 96.5,
        'Procedural-fairness compute pipelines',
        ha='center', va='top', fontsize=12, weight='bold', color=P['ink'])

# ----------------------------------------------------------------------
# Save
# ----------------------------------------------------------------------
THESIS_FIG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'thesis', 'figures',
)
os.makedirs(THESIS_FIG_DIR, exist_ok=True)
out_path = os.path.join(THESIS_FIG_DIR, 'method_procedural.pdf')
fig.savefig(out_path, bbox_inches='tight', pad_inches=0.20, transparent=True)
print(f'Wrote {out_path}')
