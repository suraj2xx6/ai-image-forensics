"""A small Platt scaling wrapper trained on a held-out validation split."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression


class CalibratedBinaryClassifier:
    """Expose a scikit-learn style predict_proba around a fitted binary model."""

    def __init__(self, classifier, calibrator: LogisticRegression):
        self.classifier = classifier
        self.calibrator = calibrator
        self.classes_ = np.asarray([0, 1])

    def predict_proba(self, matrix: np.ndarray) -> np.ndarray:
        scores = self.classifier.decision_function(matrix).reshape(-1, 1)
        return self.calibrator.predict_proba(scores)
