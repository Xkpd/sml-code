"""Shared manual metrics: every participant receives equal evaluation weight.

No sklearn metric, cross-validation or tuning utilities are used here.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


CLASS_NAMES = ("Sitting", "Standing", "Lying_Down", "Walking")
T_975_DF9 = 2.2621571628540993


@dataclass(frozen=True)
class ClassificationMetrics:
    confusion_matrix: np.ndarray
    accuracy: float
    macro_f1: float
    balanced_accuracy: float
    weighted_f1: float
    precision: np.ndarray
    recall: np.ndarray
    f1: np.ndarray
    support: np.ndarray


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 4) -> np.ndarray:
    truth = np.asarray(y_true, dtype=np.int64)
    prediction = np.asarray(y_pred, dtype=np.int64)
    if truth.ndim != 1 or prediction.ndim != 1 or len(truth) != len(prediction):
        raise ValueError("y_true and y_pred must be one-dimensional arrays of equal length")
    if np.any((truth < 0) | (truth >= n_classes)):
        raise ValueError("y_true contains a class outside the configured range")
    if np.any((prediction < 0) | (prediction >= n_classes)):
        raise ValueError("y_pred contains a class outside the configured range")
    encoded = truth * n_classes + prediction
    return np.bincount(encoded, minlength=n_classes * n_classes).reshape(n_classes, n_classes)


def metrics_from_confusion(matrix: np.ndarray) -> ClassificationMetrics:
    cm = np.asarray(matrix, dtype=np.int64)
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1] or np.any(cm < 0):
        raise ValueError("confusion matrix must be square and nonnegative")
    support = cm.sum(axis=1)
    predicted = cm.sum(axis=0)
    true_positive = np.diag(cm)
    precision = np.zeros(len(cm), dtype=np.float64)
    recall = np.zeros(len(cm), dtype=np.float64)
    f1 = np.zeros(len(cm), dtype=np.float64)
    present = support > 0
    precision[present] = np.divide(
        true_positive[present], predicted[present],
        out=np.zeros(np.sum(present), dtype=np.float64), where=predicted[present] > 0)
    recall[present] = true_positive[present] / support[present]
    denominator = 2 * true_positive + (predicted - true_positive) + (support - true_positive)
    f1[present] = np.divide(
        2 * true_positive[present], denominator[present],
        out=np.zeros(np.sum(present), dtype=np.float64), where=denominator[present] > 0)
    # A class absent from this participant's ground truth is NA, not zero.
    precision[~present] = np.nan
    recall[~present] = np.nan
    f1[~present] = np.nan
    total = support.sum()
    return ClassificationMetrics(
        confusion_matrix=cm,
        accuracy=float(true_positive.sum() / total) if total else np.nan,
        macro_f1=float(np.nanmean(f1)) if np.any(present) else np.nan,
        balanced_accuracy=float(np.nanmean(recall)) if np.any(present) else np.nan,
        weighted_f1=float(np.nansum(f1 * support) / total) if total else np.nan,
        precision=precision, recall=recall, f1=f1, support=support,
    )


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 4) -> ClassificationMetrics:
    return metrics_from_confusion(confusion_matrix(y_true, y_pred, n_classes))


def participant_scores(y_true: np.ndarray, y_pred: np.ndarray,
                       participant_ids: np.ndarray) -> dict[str, float]:
    """Return each participant's unweighted four-class MacroF1."""
    truth, prediction, people = map(np.asarray, (y_true, y_pred, participant_ids))
    if any(array.ndim != 1 for array in (truth, prediction, people)) or not len(truth):
        raise ValueError("labels and participants must be non-empty one-dimensional arrays")
    if len(truth) != len(prediction) or len(truth) != len(people):
        raise ValueError("labels and participants must have equal lengths")
    return {str(person): classification_metrics(truth[people == person], prediction[people == person]).macro_f1
            for person in np.unique(people)}


def participant_macro_f1(y_true: np.ndarray, y_pred: np.ndarray,
                         participant_ids: np.ndarray) -> float:
    """Equal mean over participants, irrespective of their window counts."""
    return float(np.mean(list(participant_scores(y_true, y_pred, participant_ids).values())))


def inverse_frequency_sample_weights(y_train: np.ndarray, n_classes: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Compute mean-one weights from the current training fold only."""
    labels = np.asarray(y_train, dtype=np.int64)
    if labels.ndim != 1 or not len(labels):
        raise ValueError("y_train must be a non-empty one-dimensional array")
    if np.any((labels < 0) | (labels >= n_classes)):
        raise ValueError("training labels must be within the configured class range")
    counts = np.bincount(labels, minlength=n_classes)
    if np.any(counts == 0):
        raise ValueError(f"training fold is missing configured classes: {np.flatnonzero(counts == 0).tolist()}")
    class_weights = len(labels) / (n_classes * counts.astype(np.float64))
    sample_weights = class_weights[labels]
    sample_weights /= sample_weights.mean()
    return sample_weights, class_weights


def mean_and_95pct_t_ci_across_10_outer_folds(values: np.ndarray) -> tuple[float, float, float]:
    """Approximate 95% t interval over ten fold means after seed averaging."""
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (10,) or np.any(~np.isfinite(array)):
        raise ValueError("exactly ten finite outer-fold means are required")
    mean = float(array.mean())
    half_width = T_975_DF9 * float(array.std(ddof=1) / np.sqrt(len(array)))
    return mean, mean - half_width, mean + half_width


def paired_difference_95pct_t_ci_across_10_outer_folds(
        first: np.ndarray, second: np.ndarray) -> tuple[float, float, float]:
    """Apply the same interval to paired fold-wise first-minus-second values."""
    first_array, second_array = np.asarray(first, dtype=np.float64), np.asarray(second, dtype=np.float64)
    if first_array.shape != second_array.shape:
        raise ValueError("paired arrays must have identical shapes")
    return mean_and_95pct_t_ci_across_10_outer_folds(first_array - second_array)
