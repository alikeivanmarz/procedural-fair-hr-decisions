"""Shared style module for thesis figure scripts.

Imported by figure-generation scripts to keep palette, fonts, and rcParams
consistent. The compact-thesis figures use a restrained academic style:
white/transparent backgrounds, near-black ink, light rules, and muted accents.
"""
import matplotlib.pyplot as plt

PALETTE = {
    'bg':           '#FFFFFF',
    'card':         '#F5F7FA',
    'ink':          '#1F2933',
    'ink_soft':     '#5B6673',
    'rule':         '#D9DEE5',
    'data':         '#2F5D7C',
    'accent':       '#8A5A44',
    'class_accent': '#4E7D66',
    'highlight':    '#A64B45',
    'plum':         '#6D6284',
}

def _shade(hex_colour: str, factor: float) -> str:
    """Lighten a hex colour by mixing with the figure background.

    factor=0.0 -> original colour; factor=1.0 -> the background.
    Used to derive within-family shades for the mitigation method palette
    (pre-processing variants of the data-blue, in-processing variants of
    the accent-sienna, post-processing variants of the class-accent green).
    """
    bg_rgb = tuple(int(PALETTE['bg'][1:][i:i+2], 16) for i in (0, 2, 4))
    fg_rgb = tuple(int(hex_colour[1:][i:i+2], 16) for i in (0, 2, 4))
    mixed = tuple(
        round(fg + (bg - fg) * factor) for fg, bg in zip(fg_rgb, bg_rgb)
    )
    return '#{0:02X}{1:02X}{2:02X}'.format(*mixed)

# ---------------------------------------------------------------------------
# Categorical palettes — semantic role-based colour mappings used across
# every figure script in the thesis. Editing these in one place keeps the
# six contribution chapters visually coherent.
# ---------------------------------------------------------------------------

#: Six base classifier families (plus two reference predictors used in Phase-4
#: degeneracy analyses).  The two reference predictors deliberately use the
#: ink_soft / rule greys so they read as "non-models" against the coloured
#: real classifiers.
MODEL_COLOURS: dict = {
    "RandomForestClassifier":      PALETTE['data'],
    "LogisticRegression":          PALETTE['accent'],
    "MLPClassifier":               PALETTE['class_accent'],
    "XGBClassifier":               PALETTE['highlight'],
    "GradientBoostingClassifier":  PALETTE['plum'],
    "KNeighborsClassifier":        PALETTE['ink_soft'],
    "ConstantPredictor":           PALETTE['rule'],
    "ShuffledPredictor":           '#9E9E9E',  # neutral mid-grey
    # Short-form aliases used in  / 6:
    "RF":  PALETTE['data'],
    "LR":  PALETTE['accent'],
    "MLP": PALETTE['class_accent'],
    "XGB": PALETTE['highlight'],
    "GB":  PALETTE['plum'],
    "KNN": PALETTE['ink_soft'],
}

#: Twelve mitigation methods + three identity baselines.  Grouped by
#: intervention stage with a within-family lightening sweep.  Pre-processing
#: methods come from the data-blue family; in-processing from accent-sienna;
#: post-processing from class-accent green; identity baselines from rule grey.
#: This mirrors the three-stage structure in Figure 3.4 (mitigation pipeline).
MITIGATION_COLOURS: dict = {
    # Pre-processing (data-blue family, dark to light)
    "reweighing":       PALETTE['data'],
    "smote_nc":         _shade(PALETTE['data'], 0.20),
    "di_remover":       _shade(PALETTE['data'], 0.40),
    "optim_preproc":    _shade(PALETTE['data'], 0.60),
    "lfr":              _shade(PALETTE['data'], 0.75),
    # In-processing (accent-sienna family)
    "adv_debias":       PALETTE['accent'],
    "exp_gradient":     _shade(PALETTE['accent'], 0.25),
    "gerryfair":        _shade(PALETTE['accent'], 0.50),
    "prejudice_remover": _shade(PALETTE['accent'], 0.70),
    # Post-processing (class-accent green family)
    "eqodds_postproc":  PALETTE['class_accent'],
    "calib_eqodds":     _shade(PALETTE['class_accent'], 0.30),
    "reject_option":    _shade(PALETTE['class_accent'], 0.55),
    # Identity baselines (neutral)
    "identity_preprocessing":  PALETTE['rule'],
    "identity_inprocessing":   PALETTE['rule'],
    "identity_postprocessing": PALETTE['rule'],
}

#:  feature-type semantics: sensitive attributes get the highlight
#: rust (alarm/attention), proxy features get the accent sienna (secondary
#: caution), all other features get the data deep-blue (neutral primary).
FEATURE_COLOURS: dict = {
    "sensitive": PALETTE['highlight'],
    "proxy":     PALETTE['accent'],
    "other":     PALETTE['data'],
}

#:  leakage_feature severity: none (no leakage) -> data blue,
#: 0.5 (moderate) -> accent sienna, 0.99 (extreme) -> highlight rust.
#: The progression encodes severity: cool -> warm.
LEAKAGE_COLOURS: dict = {
    "none": PALETTE['data'],
    "0.5":  PALETTE['accent'],
    "0.99": PALETTE['highlight'],
}

def categorical_palette(n: int) -> list:
    """Return ``n`` distinct palette-derived colours for general categorical use.

    Cycles through the six strong data colours (data, accent, class_accent,
    highlight, plum, ink_soft) and falls back to within-family shades for
    n > 6. Use this when no semantic role mapping (model, method, feature
    type, leakage level) applies.
    """
    base = [PALETTE['data'], PALETTE['accent'], PALETTE['class_accent'],
            PALETTE['highlight'], PALETTE['plum'], PALETTE['ink_soft']]
    if n <= len(base):
        return base[:n]
    # Extend with lightened shades of each base colour.
    extra = [_shade(c, 0.45) for c in base]
    return (base + extra)[:n]

def diverging_cmap():
    """Palette-aligned diverging colormap.

    highlight (muted red) -> card (near white) -> class_accent (green).
    Use as a replacement for matplotlib's RdYlGn_r in heatmaps where the
    semantic is "negative is bad, positive is good" or vice versa. Matches
    the warm editorial palette of the thesis.
    """
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(
        "thesis_diverging",
        [PALETTE['highlight'], PALETTE['card'], PALETTE['class_accent']],
        N=256,
    )

def apply_style():
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica Neue', 'Helvetica', 'Arial', 'DejaVu Sans'],
        'axes.edgecolor': PALETTE['ink'],
        'axes.labelcolor': PALETTE['ink'],
        'text.color': PALETTE['ink'],
        'xtick.color': PALETTE['ink_soft'],
        'ytick.color': PALETTE['ink_soft'],
        'axes.titlesize': 9,
        'axes.labelsize': 8.5,
        'xtick.labelsize': 7.5,
        'ytick.labelsize': 7.5,
        'legend.fontsize': 7.5,
        'axes.linewidth': 0.7,
        'grid.color': PALETTE['rule'],
        'grid.linewidth': 0.5,
        'figure.facecolor': 'none',
        'axes.facecolor': 'none',
        'savefig.facecolor': 'none',
        'savefig.edgecolor': 'none',
        'savefig.transparent': True,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })
