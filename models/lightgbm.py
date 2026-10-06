"""LightGBM adapter using raw features and shared training weights."""

from __future__ import annotations

import json
import gzip
from numbers import Integral

import lightgbm as lgb
import numpy as np


FIXED_PARAMETERS = {
    "objective": "multiclass", "num_class": 4, "learning_rate": 0.05,
    "n_estimators": 300, "max_depth": -1, "min_child_samples": 20,
    "subsample": 1.0, "colsample_bytree": 1.0,
}


def _check_booster(booster):
    rounds, trees = booster.current_iteration(), booster.num_trees()
    if (booster.params.get("objective") != "multiclass"
            or booster.num_model_per_iteration() != 4 or booster.num_feature() != 135):
        raise ValueError("LightGBM must have 135 features and four multiclass outputs")
    # LightGBM can exhaust useful splits before its configured 300-round cap.
    if not 1 <= rounds <= 300 or trees != 4 * rounds:
        raise ValueError("LightGBM requires 1..300 actual rounds and four trees per round")
    return int(rounds), int(trees)


def fit(train, validation, *, value, seed, settings, sample_weight, epochs=None):
    """Fit raw training features. No validation-based early stopping is used."""
    if epochs is not None:
        raise ValueError("LightGBM uses its configured boosting cap, not refit epochs")
    if isinstance(value, bool) or not isinstance(value, Integral) or value not in (63, 127, 255, 511):
        raise ValueError("num_leaves must be one of 63, 127, 255, 511")
    fixed = {key: value for key, value in settings.items() if key != "n_jobs"}
    if fixed != FIXED_PARAMETERS:
        raise ValueError("LightGBM settings differ from the agreed fixed parameters")
    n_jobs = settings.get("n_jobs")
    if n_jobs is not None and (isinstance(n_jobs, bool) or not isinstance(n_jobs, Integral) or n_jobs < 1):
        raise ValueError("n_jobs must be a positive integer or None for the library default")
    if (train.X.ndim != 2 or train.X.shape[1] != 135
            or not np.all(np.isfinite(train.X))):
        raise ValueError("LightGBM requires 135 finite raw features")
    if not np.array_equal(np.unique(train.y), np.arange(4)):
        raise ValueError("LightGBM training requires all four class IDs: 0, 1, 2, 3")
    weights = np.asarray(sample_weight, dtype=np.float64)
    if (weights.shape != train.y.shape or not np.all(np.isfinite(weights))
            or np.any(weights <= 0) or not np.isclose(weights.mean(), 1.0)):
        raise ValueError("Training sample weights must be positive, aligned and mean-one")
    model = lgb.LGBMClassifier(
        num_leaves=int(value), random_state=int(seed), n_jobs=n_jobs,
        verbosity=-1, **fixed,
    )
    model.fit(train.X, train.y, sample_weight=weights)
    if not np.array_equal(model.classes_, np.arange(4)):
        raise RuntimeError("LightGBM probability columns must follow class IDs 0, 1, 2, 3")
    rounds, trees = _check_booster(model.booster_)
    if model.n_estimators_ != rounds:
        raise RuntimeError("LightGBM estimator and native booster round counts disagree")
    return model.booster_, {
        "best_epoch": None, "actual_rounds": rounds, "num_trees": trees,
        "reached_configured_limit": rounds == 300,
    }


def predict_proba(state, X):
    """The native multiclass booster preserves fitted class order 0, 1, 2, 3."""
    _check_booster(state)
    values = np.asarray(X)
    if values.ndim != 2 or values.shape[1] != 135 or not np.all(np.isfinite(values)):
        raise ValueError("Prediction features must be a finite matrix with 135 columns")
    return np.asarray(state.predict(values), dtype=np.float64)


def serialize(state):
    """Native booster text retains every realised tree and all prediction settings."""
    rounds, _ = _check_booster(state)
    return gzip.compress(json.dumps({"format": "lightgbm_native_v1", "model": state.model_to_string(
        num_iteration=rounds
    )}).encode("utf-8"), mtime=0)


def deserialize(payload):
    content = json.loads(gzip.decompress(payload).decode("utf-8"))
    if content.get("format") != "lightgbm_native_v1":
        raise ValueError("Unrecognised LightGBM checkpoint format")
    booster = lgb.Booster(model_str=content["model"])
    _check_booster(booster)
    return booster
