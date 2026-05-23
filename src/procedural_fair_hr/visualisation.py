"""Visualisation utilities for fairness analysis.

This module provides ROC-based fairness visualisations, starting with
the ABROCA slice plot introduced in Gardner, Brooks & Baker (2019).

References
----------
gardner2019abroca — Gardner, J., Brooks, C. & Baker, R. (2019).
    "Evaluating the Fairness of Predictive Student Models Through
    Slicing Analysis." LAK'19. DOI: 10.1145/3303772.3303791.
    (BibTeX pending .)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

def compute_abroca(
    y_true: np.ndarray,
    y_score: np.ndarray,
    sensitive: pd.Series,
    privileged_val,
    n_grid: int = 1000,
) -> float:
    """Compute ABROCA between privileged and unprivileged groups.

    ABROCA (see `` -- Absolute Between-ROC Area``) is
    the integral over the FPR axis of the absolute difference between the
    ROC curves of the privileged and unprivileged demographic groups.
    A value of 0 indicates identical ROC curves (no bias); larger values
    indicate greater group-level ROC divergence.

    Parameters
    ----------
    y_true:
        Ground-truth binary labels (0/1), shape (n,).
    y_score:
        Continuous prediction scores (e.g. predicted probabilities),
        shape (n,).
    sensitive:
        Categorical sensitive-attribute series, length n.
        See `` attribute (S)``.
    privileged_val:
        The value in *sensitive* that identifies the privileged group
        (see `` g, protected group s, non-protected
        group s_bar``).
    n_grid:
        Number of equally-spaced points on the FPR axis [0, 1] used for
        trapezoidal integration.  Default 1000.

    Returns
    -------
    float
        ABROCA value in [0, 1].

    Reference
    ---------
    gardner2019abroca (BibTeX pending ).
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    mask_priv = np.asarray(sensitive == privileged_val)
    mask_unpriv = ~mask_priv

    fpr_grid = np.linspace(0.0, 1.0, n_grid)

    def _interp_tpr(mask: np.ndarray) -> np.ndarray:
        fpr, tpr, _ = roc_curve(y_true[mask], y_score[mask])
        return np.interp(fpr_grid, fpr, tpr)

    tpr_priv = _interp_tpr(mask_priv)
    tpr_unpriv = _interp_tpr(mask_unpriv)

    abroca: float = float(
        np.trapezoid(np.abs(tpr_priv - tpr_unpriv), fpr_grid)
    )
    return abroca

def plot_abroca(
    y_true: np.ndarray,
    y_score: np.ndarray,
    sensitive: pd.Series,
    privileged_val,
    title: str = "ABROCA Slice Plot",
    save_path: str | None = None,
) -> float:
    """Plot ROC curves for each group and shade the ABROCA region.

    Produces a matplotlib figure showing the ROC curve for the
    privileged group and the unprivileged group, with the gap between
    them shaded.  The ABROCA value (see
    `` -- Absolute Between-ROC Area``) is annotated
    in the top-left corner of the plot.

    This function does NOT call ``plt.show()``; the caller is
    responsible for displaying or closing the figure.

    Parameters
    ----------
    y_true:
        Ground-truth binary labels (0/1), shape (n,).
    y_score:
        Continuous prediction scores, shape (n,).
    sensitive:
        Categorical sensitive-attribute series, length n.
        See `` attribute (S)``.
    privileged_val:
        The value in *sensitive* that identifies the privileged group.
    title:
        Figure title.
    save_path:
        If provided, the figure is saved to this path via
        ``matplotlib.pyplot.savefig``.  Supported formats: anything
        accepted by matplotlib (PNG, PDF, SVG, …).

    Returns
    -------
    float
        The ABROCA value (same as ``compute_abroca`` with default
        ``n_grid=1000``).

    Reference
    ---------
    gardner2019abroca (BibTeX pending ).
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve

    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    mask_priv = np.asarray(sensitive == privileged_val)
    mask_unpriv = ~mask_priv

    # Compute ROC curves for both groups
    fpr_priv, tpr_priv, _ = roc_curve(y_true[mask_priv], y_score[mask_priv])
    fpr_unpriv, tpr_unpriv, _ = roc_curve(
        y_true[mask_unpriv], y_score[mask_unpriv]
    )

    # Compute ABROCA on a common grid for shading
    n_grid = 1000
    fpr_grid = np.linspace(0.0, 1.0, n_grid)
    tpr_priv_grid = np.interp(fpr_grid, fpr_priv, tpr_priv)
    tpr_unpriv_grid = np.interp(fpr_grid, fpr_unpriv, tpr_unpriv)
    abroca = float(
        np.trapezoid(np.abs(tpr_priv_grid - tpr_unpriv_grid), fpr_grid)
    )

    fig, ax = plt.subplots()
    ax.plot(
        fpr_priv,
        tpr_priv,
        label=f"Privileged ({privileged_val})",
        color="steelblue",
        lw=2,
    )
    ax.plot(
        fpr_unpriv,
        tpr_unpriv,
        label="Unprivileged",
        color="tomato",
        lw=2,
    )
    ax.fill_between(
        fpr_grid,
        tpr_priv_grid,
        tpr_unpriv_grid,
        alpha=0.25,
        color="orange",
        label=f"ABROCA = {abroca:.4f}",
    )
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random classifier")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")

    return abroca
