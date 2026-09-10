"""
explainability.py
==================
SHAP-based explainability for the anomaly-risk prediction.

WHICH MODEL IS EXPLAINED, AND WHY
----------------------------------
We explain the Isolation Forest's anomaly-scoring function, not the LSTM.
Reasoning:
  * The Isolation Forest operates on a single reading (3 features) and its
    output ("how unusual is this specific reading right now") is exactly
    the quantity a maintenance engineer wants feature-level attribution for
    ("why is THIS reading flagged?").
  * The LSTM's output depends on a full 12-step sequence. Attributing a
    single scalar risk score across 12 timesteps x 3 features is possible
    (e.g. with SHAP's DeepExplainer/GradientExplainer) but the resulting
    36-value attribution is far harder for a maintenance engineer to act on
    than "which of the 3 CURRENT sensors is driving the anomaly". For this
    prototype we therefore report SHAP values for the Isolation Forest only,
    and surface the LSTM's `trend_risk` as a separate, non-decomposed
    number. This is a documented design choice, not an oversight.

EXPLAINER CHOICE
-----------------
`sklearn.ensemble.IsolationForest` is tree-based, so in principle
`shap.TreeExplainer` applies. In practice TreeExplainer's support for
IsolationForest is version-dependent and can be brittle across
scikit-learn/shap version combinations (it relies on internal tree
structures that differ between "isolation" and "additive" tree APIs).
To keep this pipeline robust and reproducible regardless of exact library
versions, we use `shap.KernelExplainer`, a MODEL-AGNOSTIC explainer that
only requires a callable `f(X) -> scores`. It is slower than TreeExplainer
(it works by perturbing features against a background sample and observing
the effect on the output) but it is guaranteed to work with any
scikit-learn model and produces values with the same interpretation
guarantees (Shapley values: a fair per-feature decomposition of
`f(x) - f(background_mean)`).

WHAT THE VALUES MEAN
---------------------
For a given reading x, SHAP produces one value per feature such that:

    anomaly_score(x) ≈ base_value + sum(shap_values)

* A POSITIVE shap value means that feature pushed the anomaly score UP
  (i.e. made this reading look MORE anomalous / riskier) relative to the
  average reading in the background sample.
* A NEGATIVE shap value means that feature pushed the anomaly score DOWN
  (made the reading look MORE normal).
* The MAGNITUDE indicates how much influence that feature had, in the same
  units as `anomaly_score` (0-1 scale, see predict.py).

LIMITATIONS
------------
  * KernelExplainer values are approximations based on a background sample
    (`BACKGROUND_SIZE` rows drawn from real training data). Larger
    background samples give more stable values but cost more compute.
  * SHAP values explain THIS model's THIS score, not "ground truth root
    cause". They describe correlation of a feature with the model's output
    on this instance, not a guaranteed physical cause of degradation.
  * We do not fabricate SHAP values: they are computed live, at request
    time, from the loaded Isolation Forest and a real background sample
    persisted at training time.
"""

from __future__ import annotations

import numpy as np

try:
    import shap
except ImportError:  # pragma: no cover - shap is required at runtime, not at import-check time
    shap = None

from preprocessing import FEATURES

BACKGROUND_SIZE = 60  # number of background rows used by KernelExplainer


def build_background_sample(X_train_scaled: np.ndarray, size: int = BACKGROUND_SIZE) -> np.ndarray:
    """
    Pick a small, fixed background sample from the scaled TRAINING data.
    KernelExplainer needs this to estimate `E[f(X)]` as its baseline.
    Using k-means-summarised background (shap.kmeans) keeps this fast; a
    plain random sample also works and is simpler to reason about, so we
    use that here for transparency.
    """
    rng = np.random.default_rng(42)
    n = X_train_scaled.shape[0]
    size = min(size, n)
    idx = rng.choice(n, size=size, replace=False)
    return X_train_scaled[idx]


class AnomalyExplainer:
    """Wraps a trained IsolationForest + background sample into a reusable
    SHAP KernelExplainer."""

    def __init__(self, isolation_forest, background: np.ndarray):
        if shap is None:
            raise ImportError(
                "The 'shap' package is required for explainability. "
                "Install it with: pip install shap"
            )
        self.isolation_forest = isolation_forest
        self.background = background
        # decision_function: higher = more normal. We negate it so that,
        # consistent with the rest of the API, higher = more anomalous.
        self._f = lambda X: -self.isolation_forest.decision_function(X)
        self.explainer = shap.KernelExplainer(self._f, self.background)

    def explain(self, x_scaled: np.ndarray) -> dict:
        """
        x_scaled: shape (n_features,) or (1, n_features) — a SINGLE scaled
        reading (already run through the same StandardScaler used for
        training).

        Returns
        -------
        dict mapping feature name -> float SHAP value (see module docstring
        for sign convention).
        """
        x_scaled = np.asarray(x_scaled, dtype=np.float64).reshape(1, -1)
        shap_values = self.explainer.shap_values(x_scaled, nsamples="auto")
        # shap_values shape: (1, n_features) for a single-output function
        values = np.asarray(shap_values).reshape(-1)
        return {feat: float(val) for feat, val in zip(FEATURES, values)}
