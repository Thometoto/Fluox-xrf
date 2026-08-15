from dataclasses import dataclass
from typing import Dict, Iterable, List

import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.multiclass import OneVsRestClassifier

from .preprocessing import transform_counts


@dataclass
class TrainedModel:
    estimator: OneVsRestClassifier
    elements: List[str]
    thresholds: np.ndarray
    config: Dict

    def predict_proba(self, counts: np.ndarray) -> np.ndarray:
        return self.estimator.predict_proba(transform_counts(counts))

    def predict(self, counts: np.ndarray) -> np.ndarray:
        return self.predict_proba(counts) >= self.thresholds


def fit_model(x_train, y_train, x_validation, y_validation, elements: Iterable[str], config: Dict):
    estimator = OneVsRestClassifier(
        SGDClassifier(loss="log_loss", alpha=2e-5, class_weight="balanced", max_iter=2500,
                      tol=1e-5, random_state=42, average=True),
        # Single-process by default for portability on laboratory workstations
        # and environments with restricted semaphores.
        n_jobs=1,
    )
    estimator.fit(transform_counts(x_train), y_train)
    probabilities = estimator.predict_proba(transform_counts(x_validation))
    thresholds = tune_thresholds(y_validation, probabilities)
    return TrainedModel(estimator, list(elements), thresholds, config)


def tune_thresholds(y_true: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    thresholds = np.full(y_true.shape[1], 0.5, dtype=np.float32)
    candidates = np.linspace(0.1, 0.9, 33)
    for j in range(y_true.shape[1]):
        scores = [f1_score(y_true[:, j], probabilities[:, j] >= t, zero_division=0) for t in candidates]
        thresholds[j] = candidates[int(np.argmax(scores))]
    return thresholds


def metrics(y_true: np.ndarray, probabilities: np.ndarray, thresholds: np.ndarray) -> Dict[str, float]:
    predicted = probabilities >= thresholds
    return {
        "micro_f1": float(f1_score(y_true, predicted, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, predicted, average="macro", zero_division=0)),
        "micro_precision": float(precision_score(y_true, predicted, average="micro", zero_division=0)),
        "micro_recall": float(recall_score(y_true, predicted, average="micro", zero_division=0)),
        "exact_match": float(np.mean(np.all(y_true == predicted, axis=1))),
    }
