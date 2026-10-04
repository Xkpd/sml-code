"""Two-file result contract, validation, and participant-balanced summaries."""
from __future__ import annotations

import csv
import gzip
import io
import math
import os
from pathlib import Path
import statistics
import tempfile

import numpy as np

CLASSES = ("Sitting", "Standing", "Lying_Down", "Walking")
TUNING_FIELDS = (
    "model", "analysis", "training_domain", "outer_fold", "subset_seed", "subset_size",
    "hyperparameter_name", "hyperparameter_value", "inner_fold",
    "participant_balanced_validation_macro_f1", "selected", "best_epoch",
)
PREDICTION_FIELDS = (
    "model", "training_domain", "analysis", "outer_fold", "subset_size", "subset_seed",
    "hyperparameter_name", "hyperparameter_value", "test_participant", "test_window",
    "true_activity", "predicted_activity", "sitting_prob", "standing_prob",
    "lying_prob", "walking_prob",
)
PROB_FIELDS = PREDICTION_FIELDS[-4:]


def _condition(row):
    return (row["analysis"], row["training_domain"], int(row["outer_fold"]),
            int(row["subset_seed"]), int(row["subset_size"]))


def _open(path, mode="rt"):
    return (gzip.open if str(path).endswith(".gz") else open)(
        path, mode, encoding="utf-8", newline="")


def _atomic(path, write):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    os.close(fd)
    try:
        write(temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_csv_atomic(path, fields, rows):
    """Write a complete CSV without replacing the previous file on failure."""
    def write(temporary):
        opener = gzip.open if str(path).endswith(".gz") else open
        with opener(temporary, "wt", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    _atomic(path, write)


def prediction_csv_bytes(model, condition, hyperparameter_name, value, test_table, probabilities):
    """Return a gzip CSV chunk preserving participant IDs and original window indices."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(test_table.y)
    participants = np.asarray(test_table.participant_id)
    windows = np.asarray(test_table.window_index)
    n = len(labels)
    if probabilities.shape != (n, 4) or not n:
        raise ValueError("predictions must have four probability columns and at least one row")
    if (not np.isfinite(probabilities).all() or np.any(probabilities < 0)
            or np.any(probabilities > 1)
            or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-6, rtol=0)):
        raise ValueError("invalid prediction probabilities")
    if (len(participants) != n or len(windows) != n or labels.ndim != 1
            or not np.isin(labels, np.arange(4)).all()):
        raise ValueError("invalid original test labels or identities")
    if hasattr(test_table, "domain") and not np.all(np.asarray(test_table.domain) == "FL"):
        raise ValueError("outer predictions must use FL data only")
    keys = [(str(p), str(w)) for p, w in zip(participants, windows)]
    if any(not p or not w for p, w in keys) or len(set(keys)) != n:
        raise ValueError("missing or duplicate participant/window identities")
    common = {name: getattr(condition, name) for name in
              ("analysis", "training_domain", "outer_fold", "subset_seed", "subset_size")}
    common.update(model=model, hyperparameter_name=hyperparameter_name, hyperparameter_value=value)
    output = io.BytesIO()
    with gzip.GzipFile(fileobj=output, mode="wb", mtime=0) as compressed:
        with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text:
            writer = csv.DictWriter(text, fieldnames=PREDICTION_FIELDS)
            writer.writeheader()
            for i, (participant, window) in enumerate(keys):
                row = dict(common, test_participant=participant, test_window=window,
                           true_activity=int(labels[i]),
                           predicted_activity=int(np.argmax(probabilities[i])))
                row.update(zip(PROB_FIELDS, (format(p, ".17g") for p in probabilities[i])))
                writer.writerow(row)
    return output.getvalue()


def write_combined_predictions(path, chunks):
    """Stream gzip condition chunks to one atomic CSV, retaining one header."""
    def rows():
        for chunk in chunks:
            with gzip.GzipFile(fileobj=io.BytesIO(chunk), mode="rb") as compressed:
                with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text:
                    reader = csv.DictReader(text)
                    if reader.fieldnames != list(PREDICTION_FIELDS):
                        raise ValueError("prediction chunk header does not match contract")
                    yield from reader
    write_csv_atomic(path, PREDICTION_FIELDS, rows())


def select_candidate(rows, spec):
    """Select from inner folds only; return the value and FT median refit epochs."""
    grid = [float(value) for value in spec["grid"]]
    if not grid or len(set(grid)) != len(grid) or not all(math.isfinite(v) and v > 0 for v in grid):
        raise ValueError("candidate grid must be positive, finite, nonempty, and unique")
    means = {}
    for value in grid:
        scores = [float(row["participant_balanced_validation_macro_f1"])
                  for row in rows if float(row["hyperparameter_value"]) == value]
        if not scores or not all(math.isfinite(s) and 0 <= s <= 1 for s in scores):
            raise ValueError("missing or invalid candidate scores")
        means[value] = statistics.mean(scores)
    best = max(means.values())
    rule = spec["selection_rule"]
    if rule == "exact_max_then_lower_value":
        selected = min(value for value in grid if means[value] == best)
    elif rule == "within_0.001_of_best_then_nearest_grid_middle":
        tolerance = float(spec.get("selection_tolerance", 0.001))
        if not math.isfinite(tolerance) or tolerance < 0:
            raise ValueError("invalid selection tolerance")
        eligible = [i for i, value in enumerate(grid) if best - means[value] <= tolerance + 1e-12]
        selected = grid[min(eligible, key=lambda i: (abs(i - (len(grid) - 1) / 2), i))]
    else:
        raise ValueError(f"unknown selection rule: {rule}")
    epochs = [int(row["best_epoch"]) for row in rows
              if float(row["hyperparameter_value"]) == selected
              and row.get("best_epoch") is not None and str(row.get("best_epoch", "")).strip()]
    native_value = next(value for value in spec["grid"] if float(value) == selected)
    return native_value, int(round(statistics.median(epochs))) if epochs else None


def _expected_conditions(config):
    folds = config["outer_folds"]
    folds = range(folds) if isinstance(folds, int) else folds
    return {(analysis, domain, int(fold), int(seed), int(size))
            for analysis, sizes in (("main", config["participant_sizes"]),
                                    ("matched", config.get("matched_participant_sizes", [18])))
            for domain in config["training_domains"] for fold in folds
            for seed in config["subset_seeds"] for size in sizes}


def _flag(value):
    value = str(value).lower()
    if value not in ("true", "false", "1", "0"):
        raise ValueError(f"selected must be true/false or 1/0, got {value!r}")
    return value in ("true", "1")


def validate_outputs(tuning_path, predictions_path, config, split_dir, model, require_complete=True):
    """Check every tuning row and every prediction against frozen outer manifests."""
    spec = config["models"][model]
    classes = tuple(config["classes"])
    if classes != CLASSES:
        raise ValueError("class order must match the four probability columns")
    expected = _expected_conditions(config)
    expected_tuning = {key for key in expected if key[0] == "main" or key[3] == 17}
    inner = int(config["inner_folds"])
    grid = {float(value) for value in spec["grid"]}
    tuning, tuning_keys = {}, set()
    with _open(tuning_path) as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != list(TUNING_FIELDS):
            raise ValueError("tuning CSV header does not match contract")
        for row in reader:
            key = _condition(row)
            value, fold = float(row["hyperparameter_value"]), int(row["inner_fold"])
            identity = (key, value, fold)
            if (row["model"] != model or key not in expected_tuning or value not in grid
                    or row["hyperparameter_name"] != spec["hyperparameter_name"]
                    or fold not in range(inner) or identity in tuning_keys or None in row):
                raise ValueError(f"invalid or duplicate tuning row: {identity}")
            epoch = str(row["best_epoch"]).strip()
            if model == "ft_transformer":
                if not epoch or int(epoch) < 1:
                    raise ValueError("FT tuning requires a positive best_epoch")
            elif epoch:
                raise ValueError("best_epoch must be blank for LR and LightGBM")
            _flag(row["selected"])
            tuning_keys.add(identity)
            tuning.setdefault(key, []).append(row)
    if require_complete and set(tuning) != expected_tuning:
        raise ValueError(f"missing tuning conditions: {len(expected_tuning - set(tuning))}")
    selected = {}
    for key, rows in tuning.items():
        if len(rows) != len(grid) * inner:
            raise ValueError(f"incomplete candidate/fold tuning grid: {key}")
        value, _ = select_candidate(rows, spec)
        selected[key] = value
        if any(_flag(row["selected"]) != (float(row["hyperparameter_value"]) == value) for row in rows):
            raise ValueError(f"selected flags disagree with frozen selection rule: {key}")

    reference = {}
    with gzip.open(Path(split_dir) / "outer_test_windows.csv.gz", "rt", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            fold = int(row["outer_fold"])
            key = (row["participant_id"], row["window_index"])
            by_fold = reference.setdefault(fold, {})
            if key in by_fold or row["class_name"] not in CLASSES or row["domain"] != "FL":
                raise ValueError("invalid outer-test reference manifest")
            by_fold[key] = (len(by_fold), classes.index(row["class_name"]))
    seen, counts, total = {}, {}, 0
    with _open(predictions_path) as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != list(PREDICTION_FIELDS):
            raise ValueError("prediction CSV header does not match contract")
        for row in reader:
            key = _condition(row)
            tune_key = key if key[0] == "main" else (key[0], key[1], key[2], 17, key[4])
            if (row["model"] != model or key not in expected or tune_key not in selected
                    or row["hyperparameter_name"] != spec["hyperparameter_name"]
                    or float(row["hyperparameter_value"]) != selected[tune_key] or None in row):
                raise ValueError(f"prediction condition or selected hyperparameter mismatch: {key}")
            by_fold = reference.get(key[2], {})
            identity = (row["test_participant"], row["test_window"])
            if identity not in by_fold:
                raise ValueError(f"prediction contains an unknown outer-test window: {identity}")
            index, label = by_fold[identity]
            if key not in seen:
                seen[key], counts[key] = bytearray(len(by_fold)), 0
            if seen[key][index] or int(row["true_activity"]) != label:
                raise ValueError(f"duplicate window or incorrect true label: {key}, {identity}")
            probabilities = [float(row[name]) for name in PROB_FIELDS]
            if (not all(math.isfinite(p) and 0 <= p <= 1 for p in probabilities)
                    or abs(sum(probabilities) - 1) > 1e-6
                    or int(row["predicted_activity"]) != max(range(4), key=probabilities.__getitem__)):
                raise ValueError(f"invalid probabilities or predicted activity: {key}, {identity}")
            seen[key][index] = 1
            counts[key] += 1
            total += 1
    if require_complete and set(seen) != expected:
        raise ValueError(f"missing prediction conditions: {len(expected - set(seen))}")
    if any(count != len(reference[key[2]]) for key, count in counts.items()):
        raise ValueError("one or more prediction conditions omit outer-test windows")
    return {"model": model, "complete": set(seen) == expected and set(tuning) == expected_tuning,
            "tuning_rows": len(tuning_keys), "tuning_conditions": len(tuning),
            "prediction_rows": total, "prediction_conditions": len(seen)}


def aggregate_results(predictions_path):
    """Manually score participants; average people, then seeds, then ten folds."""
    from metrics import metrics_from_confusion, T_975_DF9
    matrices = {}
    with _open(predictions_path) as handle:
        for row in csv.DictReader(handle):
            key = (row["model"],) + _condition(row) + (row["test_participant"],)
            matrix = matrices.setdefault(key, [[0] * 4 for _ in range(4)])
            matrix[int(row["true_activity"])][int(row["predicted_activity"])] += 1
    participants = {}
    for key, matrix in matrices.items():
        metrics = metrics_from_confusion(np.asarray(matrix))
        scores = (metrics.macro_f1, metrics.balanced_accuracy, metrics.weighted_f1)
        participants.setdefault(key[:-1], []).append(scores)
    seeds = {}
    for key, scores in participants.items():
        if len(scores) != 2:
            raise ValueError("each outer condition requires exactly two test participants")
        model, analysis, domain, fold, seed, size = key
        seeds.setdefault((model, analysis, domain, size, fold), []).append(
            tuple(statistics.mean(s[i] for s in scores) for i in range(3)))
    folds = {}
    for key, scores in seeds.items():
        if len(scores) != 3:
            raise ValueError("each fold requires all three seeds")
        folds.setdefault(key[:-1], {})[key[-1]] = tuple(statistics.mean(s[i] for s in scores) for i in range(3))
    for key, fl_scores in list(folds.items()):
        model, analysis, domain, size = key
        lab_scores = folds.get((model, analysis, "Formal_Lab", size))
        if domain == "FL" and lab_scores is not None:
            if set(fl_scores) != set(lab_scores):
                raise ValueError("FL and Lab outer folds must match for paired comparisons")
            folds[(model, analysis, "FL_minus_Formal_Lab", size)] = {
                fold: tuple(fl_scores[fold][i] - lab_scores[fold][i] for i in range(3))
                for fold in fl_scores}
    output = []
    for key, scores in sorted(folds.items()):
        if set(scores) != set(range(10)):
            raise ValueError("confidence intervals require all ten outer folds")
        for i, metric in enumerate(("macro_f1", "balanced_accuracy", "weighted_f1")):
            values = [scores[fold][i] for fold in range(10)]
            mean = statistics.mean(values)
            half_width = T_975_DF9 * statistics.stdev(values) / math.sqrt(10)
            output.append(dict(zip(("model", "analysis", "training_domain", "subset_size"), key),
                               metric=metric, mean=mean, ci_low=mean-half_width, ci_high=mean+half_width))
    return output
