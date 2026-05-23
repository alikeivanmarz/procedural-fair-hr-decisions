"""Bias-mitigation wrappers used by the thesis.

Four methods are exposed via :data:`MITIGATION_REGISTRY`:

  * ``"reweighing"`` — Kamiran & Calders 2012 (pre-processing, AIF360).
  * ``"lfr"`` — Zemel et al. 2013 Learning Fair Representations
    (pre-processing, AIF360).
  * ``"adversarial_debiasing"`` — Zhang, Lemoine & Mitchell 2018
    (in-processing, PyTorch reimplementation in
    :mod:`procedural_fair_hr.mitigation._adv_debias_pytorch`).
  * ``"eqodds_postproc"`` — Hardt, Price & Srebro 2016 equalised-odds
    post-processor (post-processing, AIF360).

Importing this package registers every concrete wrapper into
:data:`MITIGATION_REGISTRY`. Look up methods by name, e.g.
``MITIGATION_REGISTRY["reweighing"]``.
"""

from .base import (  # noqa: F401
    CANONICAL_HYPERPARAMETER_GRID,
    MITIGATION_REGISTRY,
    MitigationBase,
    register,
)

# Importing these modules has the side effect of populating the registry
# via ``@register("...")`` decorators on each concrete subclass.
from . import inprocessing  # noqa: F401, E402
from . import postprocessing  # noqa: F401, E402
from . import preprocessing  # noqa: F401, E402
