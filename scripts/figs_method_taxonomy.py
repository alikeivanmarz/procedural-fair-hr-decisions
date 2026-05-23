"""Figure 3.2 — Fairness criteria conceptual map.

Two-dimensional placement of every fairness metric used in this thesis,
along the distributive/procedural axis (horizontal) and the group/individual
axis (vertical). The procedural-fairness quadrant is the gap that prior
fairness-ML literature has not occupied; the metrics this thesis introduces
or operationalises are visually distinct from the inherited canonical metrics.

Output: thesis/figures/method_taxonomy.pdf
"""
import os
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _plot_style import PALETTE as P, apply_style

apply_style()

fig, ax = plt.subplots(figsize=(13.0, 9.0))
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.set_axis_off()

# ----------------------------------------------------------------------
# Plot region: x in [12, 92], y in [12, 82]
# ----------------------------------------------------------------------
PX0, PX1 = 12, 92
PY0, PY1 = 12, 82
PXMID = (PX0 + PX1) / 2
PYMID = (PY0 + PY1) / 2

# Quadrant fills
quadrants = [
    (PX0, PYMID, PXMID, PY1, P['data'],         0.06, 'group + distributive'),
    (PXMID, PYMID, PX1, PY1, P['highlight'],    0.06, 'group + procedural'),
    (PX0, PY0, PXMID, PYMID, P['data'],         0.04, 'individual + distributive'),
    (PXMID, PY0, PX1, PYMID, P['highlight'],    0.10, 'individual + procedural'),
]
for x0, y0, x1, y1, c, a, _ in quadrants:
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0,
                           facecolor=c, alpha=a,
                           edgecolor='none'))

# Quadrant labels (in corners)
for x0, y0, x1, y1, c, a, label in quadrants:
    cx = (x0 + x1) / 2
    cy = (y0 + y1) / 2
    # Place label near outer corner so it does not overlap metric points
    if cx < PXMID and cy > PYMID:
        ax.text(x0 + 1.5, y1 - 1.5, label, ha='left', va='top',
                fontsize=9, color=P['ink_soft'], style='italic', weight='bold')
    elif cx > PXMID and cy > PYMID:
        ax.text(x1 - 1.5, y1 - 1.5, label, ha='right', va='top',
                fontsize=9, color=P['highlight'], style='italic', weight='bold')
    elif cx < PXMID and cy < PYMID:
        ax.text(x0 + 1.5, y0 + 1.5, label, ha='left', va='bottom',
                fontsize=9, color=P['ink_soft'], style='italic', weight='bold')
    else:
        ax.text(x1 - 1.5, y0 + 1.5, label, ha='right', va='bottom',
                fontsize=9, color=P['highlight'], style='italic', weight='bold')

# Axis lines
ax.plot([PX0, PX1], [PYMID, PYMID], color=P['ink'], linewidth=1.0, zorder=2)
ax.plot([PXMID, PXMID], [PY0, PY1], color=P['ink'], linewidth=1.0, zorder=2)

# Axis arrows (extending past plot region)
ax.annotate('', xy=(PX1 + 1.2, PYMID), xytext=(PX0 - 1.2, PYMID),
            arrowprops=dict(arrowstyle='<|-|>', color=P['ink'], lw=1.0))
ax.annotate('', xy=(PXMID, PY1 + 1.2), xytext=(PXMID, PY0 - 1.2),
            arrowprops=dict(arrowstyle='<|-|>', color=P['ink'], lw=1.0))

# Axis labels
ax.text(PX0 - 2.5, PYMID, 'distributive', ha='right', va='center',
        fontsize=10, weight='bold', color=P['ink'])
ax.text(PX1 + 2.5, PYMID, 'procedural', ha='left', va='center',
        fontsize=10, weight='bold', color=P['ink'])
ax.text(PXMID, PY1 + 2.5, 'group level', ha='center', va='bottom',
        fontsize=10, weight='bold', color=P['ink'])
ax.text(PXMID, PY0 - 2.5, 'individual level', ha='center', va='top',
        fontsize=10, weight='bold', color=P['ink'])

# ----------------------------------------------------------------------
# Metric placements (x, y, label, kind)
#   kind: 'inherited' = drawn from prior literature
#         'new'       = introduced or first-time-operationalised here
# ----------------------------------------------------------------------
metrics = [
    # group + distributive
    (22, 76, 'Demographic Parity',     'inherited'),
    (22, 70, 'Disparate Impact',       'inherited'),
    (32, 73, 'Equalised Odds',         'inherited'),
    (32, 67, 'Equal Opportunity',      'inherited'),
    (22, 62, 'ABROCA',                 'inherited'),
    (40, 75, 'Macro-DP',               'new'),
    (40, 69, 'Macro-EOdds',            'new'),
    (40, 63, 'Macro-EO',               'new'),
    (32, 58, 'Multinomial CF',         'new'),
    # group + procedural
    (75, 73, 'Voice / Representation', 'new'),
    (75, 67, 'Voice-Enrichment',       'new'),
    # individual + distributive
    (24, 32, 'KNN Consistency',        'inherited'),
    (40, 26, 'Counterfactual Fairness (Level 1)', 'inherited'),
    # individual + procedural
    (70, 38, 'Process Consistency',    'new'),
    (75, 28, 'Model-Flippability',     'new'),
    (75, 18, 'Explanation-Actionability', 'new'),
]

for x, y, label, kind in metrics:
    if kind == 'new':
        edge = P['highlight']
        fill = P['card']
        weight = 'bold'
        lw = 1.4
        fontsize = 8.7
    else:
        edge = P['data']
        fill = P['card']
        weight = 'normal'
        lw = 1.0
        fontsize = 8.4
    box_w = max(13, len(label) * 0.55 + 4)
    box = FancyBboxPatch((x - box_w / 2, y - 1.7), box_w, 3.4,
                         boxstyle='round,pad=0.18,rounding_size=0.5',
                         linewidth=lw, edgecolor=edge, facecolor=fill,
                         zorder=3)
    ax.add_patch(box)
    ax.text(x, y, label, ha='center', va='center',
            fontsize=fontsize, color=P['ink'], weight=weight, zorder=4)

# ----------------------------------------------------------------------
# Title and legend
# ----------------------------------------------------------------------
ax.text(50, 92, 'Fairness criteria used in this thesis',
        ha='center', va='top', fontsize=12, weight='bold', color=P['ink'])
ax.text(50, 88,
        'Placed by what they measure (distributive vs procedural) and at what level (group vs individual)',
        ha='center', va='top', fontsize=8.5,
        color=P['ink_soft'], style='italic')

# Legend
LEG_Y = 4
for k, (kind, edge, label) in enumerate([
    ('new',       P['highlight'], 'introduced or first operationalised in this thesis'),
    ('inherited', P['data'],      'inherited from prior fairness-ML literature'),
]):
    lx = 18 + k * 42
    box = FancyBboxPatch((lx, LEG_Y - 1.0), 5, 2.0,
                         boxstyle='round,pad=0.1,rounding_size=0.3',
                         linewidth=1.4 if kind == 'new' else 1.0,
                         edgecolor=edge, facecolor=P['card'])
    ax.add_patch(box)
    ax.text(lx + 6, LEG_Y, label, ha='left', va='center',
            fontsize=8.5, color=P['ink'])

# ----------------------------------------------------------------------
# Save
# ----------------------------------------------------------------------
THESIS_FIG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'thesis', 'figures',
)
os.makedirs(THESIS_FIG_DIR, exist_ok=True)
out_path = os.path.join(THESIS_FIG_DIR, 'method_taxonomy.pdf')
fig.savefig(out_path, bbox_inches='tight', pad_inches=0.20, transparent=True)
print(f'Wrote {out_path}')
