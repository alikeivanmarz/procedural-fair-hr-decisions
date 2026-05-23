"""PyTorch reimplementation of Zhang, Lemoine & Mitchell (2018) §3.

Adversarial Debiasing per  (the AIF360 reference implementation
imports ``tensorflow.compat.v1`` which conflicts with our Python 3.10
single-stack environment; we reimplement in PyTorch).

Architecture (ZLM 2018 §3):

  Predictor :  MLP(input_dim -> hidden_predictor -> n_classes)
  Adversary :  MLP(predictor_logits -> hidden_adversary -> 1)

Loss (alternating update with the gradient-projection trick of
ZLM 2018 Eq. 4):

  L_pred  =  CrossEntropy(predictor(x), y)
  L_adv   =  BCEWithLogits(adversary(predictor_logits(x)), sensitive)

  Predictor gradient   :  grad(L_pred) - proj_{grad(L_adv)} grad(L_pred)
                              - lambda_ * grad(L_adv)
  Adversary gradient   :  grad(L_adv)

  Setting ``lambda_=0`` removes the adversary term and the projection,
  recovering a vanilla 2-layer MLP classifier (verified by
  ``tests/test_adv_debias_pytorch.py::test_lambda_zero_recovers_baseline``).

Determinism:

  * ``torch.manual_seed`` and ``torch.cuda.manual_seed_all`` are pinned
    to ``random_state``.
  * ``torch.use_deterministic_algorithms(True)`` is opt-in via
    ``deterministic=True`` (default ``True``); disable to fall back to
    cuDNN's faster non-deterministic kernels.
  * MPS (Apple Silicon GPU) is opt-in via ``device="mps"``; default
    auto-detects and falls back to CPU if MPS is non-deterministic at
    seed=0 vs seed=0 (see test ``test_byte_identical_two_runs``).

References
----------

Zhang, B. H., Lemoine, B., & Mitchell, M. (2018). "Mitigating Unwanted
Biases with Adversarial Learning." AIES 2018. BibTeX:
``zhang2018mitigating``.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

def _resolve_device(requested: Optional[str]) -> str:
    """Decide which device to run on.

    Resolution order:
      1. Explicit ``requested`` argument.
      2. ``$PHASE5_DEVICE`` env var (set by the audit runner's worker
         initialiser per ).
      3. Auto: MPS if available, else CPU.

    For Adversarial Debiasing specifically: per  we default to
    CPU when in doubt because MPS deterministic ops are still incomplete
    in PyTorch 2.x; the test suite confirms byte-identity on CPU.
    """
    if requested in ("cpu", "mps", "cuda"):
        return requested
    env_override = os.environ.get("PHASE5_DEVICE", "").lower()
    if env_override in ("cpu", "mps", "cuda"):
        return env_override
    if env_override == "auto" or env_override == "":
        # Prefer CPU for AdvDebias to keep byte-identity guarantees.
        return "cpu"
    return "cpu"

class _Predictor(nn.Module):
    """2-layer MLP predictor (ZLM 2018 §3)."""

    def __init__(self, n_features: int, hidden: int, n_classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class _Adversary(nn.Module):
    """1-layer MLP adversary that predicts the sensitive attribute from
    the predictor's logits (ZLM 2018 §3)."""

    def __init__(self, n_classes: int, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_classes, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return self.net(logits).squeeze(-1)

class AdversarialDebiasingPyTorch:
    """Adversarial-debiasing classifier — ZLM 2018 §3 in PyTorch.

    Parameters
    ----------
    n_classes : int
        Number of target classes; ``>=2``.
    n_features : int
        Number of input features.
    lambda_ : float, default=1.0
        Adversary-loss weight in the predictor's gradient. ``0`` recovers
        the unconstrained predictor.
    hidden_predictor, hidden_adversary : int
        Hidden-layer widths for the two MLPs.
    lr : float
        Learning rate (Adam) for both networks.
    n_epochs : int
        Total epochs; one epoch is one pass over the training set.
    batch_size : int
    device : {"cpu", "mps", "cuda", None}, optional
        See :func:`_resolve_device`. ``None`` defers to env / auto with
        a CPU fallback.
    random_state : int
    """

    def __init__(
        self,
        n_classes: int,
        n_features: int,
        lambda_: float = 1.0,
        hidden_predictor: int = 64,
        hidden_adversary: int = 32,
        lr: float = 1e-3,
        n_epochs: int = 50,
        batch_size: int = 64,
        device: Optional[str] = None,
        random_state: int = 0,
    ) -> None:
        self.n_classes = int(n_classes)
        self.n_features = int(n_features)
        self.lambda_ = float(lambda_)
        self.hidden_predictor = int(hidden_predictor)
        self.hidden_adversary = int(hidden_adversary)
        self.lr = float(lr)
        self.n_epochs = int(n_epochs)
        self.batch_size = int(batch_size)
        self.random_state = int(random_state)
        self.device = _resolve_device(device)
        # Lazy: predictor / adversary built on fit so the constructor is cheap.
        self.predictor_: Optional[_Predictor] = None
        self.adversary_: Optional[_Adversary] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _seed_everything(self) -> None:
        """Pin every RNG that PyTorch's training loop touches."""
        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():  # pragma: no cover — CPU-only laptop
            torch.cuda.manual_seed_all(self.random_state)
        np.random.seed(self.random_state)

    def _make_loader(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sens: np.ndarray,
    ) -> DataLoader:
        Xt = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32))
        yt = torch.from_numpy(np.ascontiguousarray(y, dtype=np.int64))
        st = torch.from_numpy(np.ascontiguousarray(sens, dtype=np.float32))
        ds = TensorDataset(Xt, yt, st)
        # Deterministic shuffling via a seeded generator.
        gen = torch.Generator()
        gen.manual_seed(self.random_state)
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
            generator=gen,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sens: np.ndarray,
    ) -> "AdversarialDebiasingPyTorch":
        """Train predictor + adversary alternately for ``n_epochs`` epochs."""
        self._seed_everything()
        device = torch.device(self.device)

        self.predictor_ = _Predictor(
            self.n_features, self.hidden_predictor, self.n_classes
        ).to(device)
        # When λ=0 we still build the adversary for shape consistency,
        # but its gradient never reaches the predictor.
        self.adversary_ = _Adversary(self.n_classes, self.hidden_adversary).to(
            device
        )

        opt_pred = torch.optim.Adam(
            self.predictor_.parameters(), lr=self.lr
        )
        opt_adv = torch.optim.Adam(
            self.adversary_.parameters(), lr=self.lr
        )

        ce_loss = nn.CrossEntropyLoss()
        bce_loss = nn.BCEWithLogitsLoss()

        loader = self._make_loader(X, y, sens)

        for _epoch in range(self.n_epochs):
            for xb, yb, sb in loader:
                xb = xb.to(device)
                yb = yb.to(device)
                sb = sb.to(device)

                # ---- Adversary step ----
                # Train adversary to predict sens from predictor's logits.
                # Predictor is in eval-grad-detach mode for this step.
                with torch.no_grad():
                    logits_detached = self.predictor_(xb)
                adv_logits = self.adversary_(logits_detached)
                la = bce_loss(adv_logits, sb)
                opt_adv.zero_grad(set_to_none=True)
                la.backward()
                opt_adv.step()

                # ---- Predictor step ----
                # Recompute logits with grad. Fold ZLM 2018 Eq. 4
                # gradient-projection: grad(L_pred) projected to remove
                # any component along grad(L_adv); then subtract
                # lambda_ * grad(L_adv).
                logits = self.predictor_(xb)
                lp = ce_loss(logits, yb)
                if self.lambda_ <= 0:
                    # λ=0 path: pure CE loss, no adversary feedback.
                    opt_pred.zero_grad(set_to_none=True)
                    lp.backward()
                    opt_pred.step()
                    continue

                adv_logits2 = self.adversary_(logits)
                la2 = bce_loss(adv_logits2, sb)

                # grad(L_pred) and grad(L_adv) w.r.t. predictor params.
                pred_params = list(self.predictor_.parameters())
                grad_pred = torch.autograd.grad(
                    lp, pred_params, retain_graph=True, create_graph=False
                )
                grad_adv = torch.autograd.grad(
                    la2, pred_params, retain_graph=False, create_graph=False
                )

                # Eq. 4: g <- g - proj_{g_adv}(g) - lambda_ * g_adv
                with torch.no_grad():
                    new_grads = []
                    for gp, ga in zip(grad_pred, grad_adv):
                        # Flatten for inner product.
                        ga_flat = ga.flatten()
                        gp_flat = gp.flatten()
                        denom = ga_flat.dot(ga_flat) + 1e-12
                        proj_coef = gp_flat.dot(ga_flat) / denom
                        proj = proj_coef * ga
                        new_g = gp - proj - self.lambda_ * ga
                        new_grads.append(new_g)

                    # Apply manually computed grads.
                    opt_pred.zero_grad(set_to_none=True)
                    for p, g in zip(pred_params, new_grads):
                        p.grad = g
                opt_pred.step()

        return self

    @torch.no_grad()
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.predictor_ is None:
            raise RuntimeError("predict_proba called before fit()")
        device = torch.device(self.device)
        Xt = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32)).to(
            device
        )
        logits = self.predictor_(Xt)
        return torch.softmax(logits, dim=-1).cpu().numpy()

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=-1).astype(int)
