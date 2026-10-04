"""One manual nested-CV runner for all models. Start with: python run.py plan."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import asdict, replace
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sqlite3
import sys
import threading
import time

import numpy as np

from data import Condition, ExperimentData, load_split
from metrics import inverse_frequency_sample_weights, participant_macro_f1
from results import (TUNING_FIELDS, evaluate_results, print_evaluation, prediction_csv_bytes, select_candidate,
                     validate_outputs, write_combined_predictions, write_csv_atomic)

ROOT = Path(__file__).resolve().parent
SHARED_SOURCE = ("run.py", "data.py", "metrics.py", "results.py", "requirements.txt")
SPLIT_FILES = ("outer_participant_roles.csv", "nested_participant_subsets.csv", "outer_test_windows.csv.gz",
               "matched_training_windows_replicate_0.csv.gz", "matched_training_windows_replicate_1.csv.gz",
               "matched_training_windows_replicate_2.csv.gz")
MODEL_ALIASES = {"lr": "multinomial_logistic_regression", "lightgbm": "lightgbm", "ft": "ft_transformer"}
RESULT_FILES = ("tuning.csv", "predictions.csv.gz")


def json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(value).hexdigest()


def file_digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def read_config(root=ROOT):
    config = json.loads((Path(root) / "experiment.json").read_text(encoding="utf-8"))
    expected = {"outer_folds": 10, "inner_folds": 3, "feature_count": 135,
                "classes": ["Sitting", "Standing", "Lying_Down", "Walking"],
                "subset_seeds": [17, 42, 73], "training_domains": ["FL", "Formal_Lab"],
                "matched_participant_sizes": [18], "matched_tuning_seed": 17, "test_domain": "FL"}
    for field, value in expected.items():
        if config.get(field) != value:
            raise ValueError(f"{field} must be {value!r} for these frozen manifests")
    if config["participant_sizes"] not in ([6, 9, 12, 15, 18], [6, 12, 18]):
        raise ValueError("Choose the original five sizes or the predeclared three-size fallback")
    for name, spec in config["models"].items():
        grid = spec["grid"]
        if len(grid) < 3 or len(set(grid)) != len(grid) or any(not np.isfinite(x) or x <= 0 for x in grid):
            raise ValueError(f"{name} needs at least three distinct positive candidates")
        if spec["status"] not in ("ready", "pending"):
            raise ValueError("Model status must be ready or pending")
        if name == "ft_transformer" and spec["status"] == "ready":
            if (not spec["settings"] or spec.get("matched_epoch_policy") != "reuse_seed17_refit_epochs"
                    or spec.get("refit_epoch_rule") != "median_of_selected_candidate_best_epochs"):
                raise ValueError("FT requires final settings and the documented median/reuse epoch policy")
    return config


def conditions(config):
    for analysis, sizes in (("main", config["participant_sizes"]), ("matched", config["matched_participant_sizes"])):
        for outer in range(config["outer_folds"]):
            for domain in config["training_domains"]:
                for size in sizes:
                    for seed in config["subset_seeds"]:
                        yield Condition(analysis, domain, outer, seed, size)


def planned_workload(config, model):
    scheduled = list(conditions(config))
    tuned = sum(c.analysis == "main" or c.subset_seed == config["matched_tuning_seed"] for c in scheduled)
    inner = tuned * config["inner_folds"] * len(config["models"][model]["grid"])
    return {"conditions": len(scheduled), "tuned_conditions": tuned, "inner_fits": inner,
            "refits": len(scheduled), "fits": inner + len(scheduled)}


def fingerprint(root, config):
    """Hash the common scientific protocol separately from each model adapter."""
    root = Path(root)
    paths = [root / p for p in SHARED_SOURCE]
    paths += [root / "splits" / name for name in SPLIT_FILES]
    paths += sorted((root / "data" / "shards").glob("*.npz"))
    paths += [root / "data" / p for p in ("feature_schema_and_counts.json", "class_mapping_v1.json")]
    if len(list((root / "data" / "shards").glob("*.npz"))) != 40:
        raise ValueError("Expected the original 40 participant/domain feature shards")
    files = {p.relative_to(root).as_posix(): file_digest(p) for p in paths}
    core = {k: v for k, v in config.items() if k != "models"}
    shared = {"protocol": core, "files": files}
    models = {}
    for name, spec in config["models"].items():
        if spec["status"] == "ready":
            source = root / (spec["module"].replace(".", "/") + ".py")
            item = {"spec": spec, "source_sha256": file_digest(source)}
            models[name] = {**item, "sha256": digest(json_bytes(item))}
    return {"shared_sha256": digest(json_bytes(shared)), "shared": shared, "models": models}


def freeze(root, config):
    root = Path(root)
    current = fingerprint(root, config)
    path = root / "experiment.lock.json"
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        if old["shared_sha256"] != current["shared_sha256"]:
            raise ValueError("The shared frozen protocol changed. Preserve this run; create a new version/folder.")
        for name, record in old["models"].items():
            if current["models"].get(name) != record:
                raise ValueError(f"Frozen {name} changed; do not mix it with existing results")
    atomic_json(path, current)
    return current


def verify_frozen(root, config, model):
    path = Path(root) / "experiment.lock.json"
    if not path.exists():
        raise ValueError("Run python run.py freeze after finalising experiment.json")
    frozen = json.loads(path.read_text(encoding="utf-8"))
    current = fingerprint(root, config)
    if frozen["shared_sha256"] != current["shared_sha256"]:
        raise ValueError("Shared code, settings, splits or data differ from experiment.lock.json")
    if model not in frozen["models"] or current["models"].get(model) != frozen["models"][model]:
        raise ValueError(f"{model} is pending, changed or not frozen; complete its adapter/settings first")
    return frozen


def environment(model):
    packages = ["numpy", "scipy"]
    packages += ["torch"] if model == "ft_transformer" else ["scikit-learn"]
    if model == "lightgbm":
        packages.append("lightgbm")
    return {"python": platform.python_version(), "platform": platform.platform(),
            "packages": {name: importlib.metadata.version(name) for name in packages}}


@contextmanager
def single_writer(folder):
    """Operating-system lock is released even if the process is killed."""
    path = Path(folder) / "run.lock"
    with path.open("a+b") as handle:
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError("Another process is already using this model's recovery folder") from exc
        try:
            yield
        finally:
            if os.name == "nt":
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class Progress:
    """One recovery database: inner scores, selections, final models and predictions."""
    def __init__(self, path, identity, require_existing=False):
        path = Path(path)
        existed = path.exists()
        if require_existing and not existed:
            raise ValueError("Recovery database is missing; saved results will not be replaced")
        self._lock = threading.RLock()
        self.db = (sqlite3.connect(f"{path.resolve().as_uri()}?mode=rw", uri=True, check_same_thread=False)
                   if existed else sqlite3.connect(path, check_same_thread=False))
        try:
            if existed:
                previous = self.get("metadata", "identity")
                if previous is None:
                    raise ValueError("Existing recovery database has no experiment identity")
                if previous != json_bytes(identity):
                    raise ValueError("Recovery file belongs to different code, model settings, environment or output folder")
            else:
                self.db.execute("CREATE TABLE items (kind TEXT, key TEXT, value BLOB, sha TEXT, PRIMARY KEY(kind,key))")
                self.put("metadata", "identity", json_bytes(identity))
            self.db.execute("PRAGMA synchronous=FULL")
        except (ValueError, sqlite3.DatabaseError) as exc:
            self.db.close()
            raise ValueError(f"Invalid recovery database: {exc}") from exc

    def get(self, kind, key):
        with self._lock:
            row = self.db.execute("SELECT value,sha FROM items WHERE kind=? AND key=?", (kind, key)).fetchone()
        if row is None:
            return None
        value, sha = row
        if digest(value) != sha:
            raise ValueError(f"Damaged recovery record: {kind}/{key}")
        return value

    def put(self, kind, key, value):
        with self._lock, self.db:
            self.db.execute("INSERT OR REPLACE INTO items VALUES (?,?,?,?)", (kind, key, value, digest(value)))

    def close(self):
        with self._lock:
            self.db.close()


def recovery_location(root, model, folder, recovery_dir=None, require_existing=False):
    """Check storage before creating a database or touching either result file."""
    folder = Path(folder).resolve()
    recovery = Path(recovery_dir or Path(root) / "recovery" / model).resolve()
    if folder == recovery or folder in recovery.parents or recovery in folder.parents:
        raise ValueError("Results and recovery folders must be separate, non-overlapping folders")
    devices = set()
    for path in (folder, recovery):
        while not path.exists():
            path = path.parent
        devices.add(path.stat().st_dev)
    if len(devices) != 1:
        raise ValueError("Results and recovery must be on the same filesystem for atomic result replacement")
    existing_results = any((folder / name).exists() for name in RESULT_FILES)
    if (require_existing or existing_results) and not (recovery / "progress.sqlite3").is_file():
        raise ValueError("The original recovery database is missing. Use validate/evaluate to read saved results; "
                         "use a new output and recovery folder to start a new run.")
    return recovery


class Runner:
    def __init__(self, root, config, model, folder, frozen, *, adapter=None, data=None, runtime=None,
                 recovery_dir=None, require_existing=False, threads_per_fit=None):
        self.root, self.config, self.model = Path(root), config, model
        self.spec = config["models"][model]
        self.folder = Path(folder).resolve()
        self.recovery_dir = recovery_location(root, model, self.folder, recovery_dir, require_existing)
        self.adapter = adapter or importlib.import_module(self.spec["module"])
        self.threads_per_fit = threads_per_fit
        self.data = data or ExperimentData(self.root / "data", self.root / "splits", config)
        self.identity = {"shared_sha256": frozen["shared_sha256"], "model": model,
                         "model_sha256": frozen["models"][model]["sha256"],
                         "results_dir": str(self.folder),
                         "environment": runtime if runtime is not None else environment(model)}
        self.recovery_dir.mkdir(parents=True, exist_ok=True)
        self.progress = Progress(self.recovery_dir / "progress.sqlite3", self.identity, require_existing)
        try:
            self.guard_existing_results()
        except ValueError:
            self.progress.close()
            raise

    def guard_existing_results(self):
        saved = self.progress.get("metadata", "export_hashes")
        allowed = json.loads(saved) if saved is not None else {}
        for name in RESULT_FILES:
            path = self.folder / name
            if path.exists() and file_digest(path) not in allowed.get(name, []):
                raise ValueError("Existing results do not match this recovery database's exports. "
                                 "Recovery may be empty, older, or from another run; results were preserved.")

    def fit(self, split, value, seed, epochs=None):
        if split.test is not None:
            raise ValueError("Fitting must never receive outer-test data")
        weights, _ = inverse_frequency_sample_weights(split.train.y)
        start = time.perf_counter()
        settings = self.spec["settings"]
        if self.model == "lightgbm" and self.threads_per_fit is not None:
            settings = {**settings, "n_jobs": self.threads_per_fit}
        state, info = self.adapter.fit(split.train, split.validation, value=value, seed=seed,
                                       settings=settings, sample_weight=weights, epochs=epochs)
        info = dict(info)
        info.update(fit_seconds=time.perf_counter() - start, train_windows=len(split.train))
        if self.model == "ft_transformer":
            if split.validation is not None:
                best = info.get("best_epoch")
                if type(best) is not int or best < 1:
                    raise ValueError("FT inner fitting must return the selected checkpoint's positive best_epoch")
            elif info.get("epochs_run") != epochs:
                raise ValueError("FT final refit must run exactly the inner-selected epoch count")
        elif info.get("best_epoch") is not None:
            raise ValueError("best_epoch must be empty for LR and LightGBM")
        details = [f"fit_seconds={info['fit_seconds']:.3f}", f"train_windows={info['train_windows']}"]
        details += [f"{key}={info[key]}" for key in ("actual_rounds", "iterations", "epochs_run", "best_epoch")
                    if info.get(key) is not None]
        print("FIT finished: " + "; ".join(details), flush=True)
        return state, info

    def tune(self, condition):
        # Matched seeds42/73 reuse only their own fold/domain's matched seed17 selection.
        source = replace(condition, subset_seed=17) if condition.analysis == "matched" else condition
        saved = self.progress.get("selection", source.key)
        if saved is not None:
            return json.loads(saved)
        rows = []
        for fold in range(3):
            split = None
            for value in self.spec["grid"]:
                key = f"{source.key}__inner{fold}__value{value}"
                saved = self.progress.get("inner", key)
                if saved is None:
                    if split is None:
                        split = load_split(self.data, source, inner_fold=fold)
                    print(f"TUNE {self.model} {key}", flush=True)
                    state, info = self.fit(split, value, source.subset_seed)
                    probabilities = checked_probabilities(self.adapter.predict_proba(state, split.validation.X), len(split.validation))
                    score = participant_macro_f1(split.validation.y, probabilities.argmax(axis=1), split.validation.participant_id)
                    row = {"model": self.model, **asdict(source), "hyperparameter_name": self.spec["hyperparameter_name"],
                           "hyperparameter_value": value, "inner_fold": fold,
                           "participant_balanced_validation_macro_f1": score,
                           "selected": False, "best_epoch": info.get("best_epoch")}
                    self.progress.put("inner", key, json_bytes({"row": row, "fit": info}))
                    del state, probabilities
                else:
                    row = json.loads(saved)["row"]
                rows.append(row)
            del split
        value, epochs = select_candidate(rows, self.spec)
        for row in rows:
            row["selected"] = float(row["hyperparameter_value"]) == float(value)
        selected = {"condition": asdict(source), "value": value, "refit_epochs": epochs, "rows": rows}
        self.progress.put("selection", source.key, json_bytes(selected))
        return selected

    def run_condition(self, condition):
        if self.progress.get("predictions", condition.key) is not None:
            print(f"DONE {self.model} {condition.key} (reused)", flush=True)
            return
        selected = self.tune(condition)
        checkpoint = self.progress.get("checkpoint", condition.key)
        if checkpoint is None:
            split = load_split(self.data, condition)
            print(f"REFIT {self.model} {condition.key}", flush=True)
            state, info = self.fit(split, selected["value"], condition.subset_seed, selected["refit_epochs"])
            checkpoint = self.adapter.serialize(state)
            if not isinstance(checkpoint, bytes):
                raise TypeError("Model serialize() must return bytes")
            restored = self.adapter.deserialize(checkpoint)
            sample = split.train.X[:min(32, len(split.train))]
            before = checked_probabilities(self.adapter.predict_proba(state, sample), len(sample))
            after = checked_probabilities(self.adapter.predict_proba(restored, sample), len(sample))
            if (not np.array_equal(before.argmax(1), after.argmax(1))
                    or not np.allclose(before, after, rtol=1e-5, atol=1e-6)):
                raise ValueError("Saved model reload changed predictions")
            self.progress.put("refit", condition.key, json_bytes({**info, "selected_value": selected["value"],
                                                                "refit_epochs": selected["refit_epochs"]}))
            self.progress.put("checkpoint", condition.key, checkpoint)
            del split, state, restored, sample, before, after
        # The final model is saved before the outer test is requested.
        state = self.adapter.deserialize(checkpoint)
        test = self.data.outer_test(condition.outer_fold)
        probabilities = checked_probabilities(self.adapter.predict_proba(state, test.X), len(test))
        chunk = prediction_csv_bytes(self.model, condition, self.spec["hyperparameter_name"], selected["value"], test, probabilities)
        self.progress.put("predictions", condition.key, chunk)
        print(f"SAVED {condition.key}: {len(test)} held-out FL windows", flush=True)

    def export(self):
        self.guard_existing_results()
        tuning = []
        completed = []
        for condition in conditions(self.config):
            if condition.analysis == "main" or condition.subset_seed == 17:
                saved = self.progress.get("selection", condition.key)
                if saved is not None:
                    tuning.extend(json.loads(saved)["rows"])
            if self.progress.get("predictions", condition.key) is not None:
                completed.append(condition)
        if not tuning and not completed:
            return 0
        # Stage files in recovery. Record both old/new hashes BEFORE replacing either
        # CSV so a crash between the two replacements can be recovered safely.
        staged = {name: self.recovery_dir / ("export_" + name) for name in RESULT_FILES}
        write_csv_atomic(staged["tuning.csv"], TUNING_FIELDS, tuning)
        write_combined_predictions(staged["predictions.csv.gz"],
                                   (self.progress.get("predictions", c.key) for c in completed))
        next_hashes = {name: [file_digest(path)] for name, path in staged.items()}
        during_export = {name: hashes + ([file_digest(self.folder / name)] if (self.folder / name).exists() else [])
                         for name, hashes in next_hashes.items()}
        self.progress.put("metadata", "export_hashes", json_bytes(during_export))
        self.folder.mkdir(parents=True, exist_ok=True)
        for name, path in staged.items():
            os.replace(path, self.folder / name)
        self.progress.put("metadata", "export_hashes", json_bytes(next_hashes))
        return len(completed)


def checked_probabilities(values, count):
    values = np.asarray(values, dtype=np.float64)
    if (values.shape != (count, 4) or not np.all(np.isfinite(values))
            or np.any(values < 0) or np.any(values > 1)
            or not np.allclose(values.sum(axis=1), 1, rtol=0, atol=1e-6)):
        raise ValueError("Expected finite, normalised probabilities in the four-class order")
    return values


def partition_conditions_by_outer_fold(selected, workers):
    """Keep all conditions from an outer fold on one worker.

    Matched seeds 42 and 73 reuse seed 17's selection, so splitting one outer
    fold across workers could duplicate tuning. Round-robin fold assignment is
    deterministic and balances the ten folds as evenly as possible.
    """
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer")
    folds = sorted({condition.outer_fold for condition in selected})
    groups = [[] for _ in range(min(workers, len(folds)))]
    if not groups:
        return []
    owner = {fold: index % len(groups) for index, fold in enumerate(folds)}
    for condition in selected:
        groups[owner[condition.outer_fold]].append(condition)
    return groups


def run_conditions(runner, selected, workers=1):
    groups = partition_conditions_by_outer_fold(selected, workers)
    if len(groups) <= 1:
        for condition in selected:
            runner.run_condition(condition)
        return
    print("Parallel outer-fold workers: " + ", ".join(
        f"worker {index + 1}={sorted({c.outer_fold for c in group})}"
        for index, group in enumerate(groups)), flush=True)
    with ThreadPoolExecutor(max_workers=len(groups), thread_name_prefix="outer-fold") as pool:
        futures = [pool.submit(lambda items=group: [runner.run_condition(c) for c in items])
                   for group in groups]
        try:
            for future in as_completed(futures):
                future.result()
        except BaseException:
            for future in futures:
                future.cancel()
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "freeze", "check", "run", "export", "validate", "summarise", "evaluate"))
    parser.add_argument("--model", choices=tuple(MODEL_ALIASES) + tuple(MODEL_ALIASES.values()))
    parser.add_argument("--output", type=Path, help="Default: outputs/<canonical model name>")
    parser.add_argument("--recovery-dir", type=Path, help="Local progress/checkpoints for this model; default: recovery/<canonical model name>")
    parser.add_argument("--detail", choices=("summary", "participants", "classes", "confusion", "folds", "all"),
                        default="summary", help="Evaluation detail to print; no extra output file is created")
    parser.add_argument("--analysis", choices=("main", "matched"))
    parser.add_argument("--domain", choices=("FL", "Formal_Lab"))
    parser.add_argument("--outer-fold", type=int, choices=range(10))
    parser.add_argument("--subset-seed", type=int, choices=(17, 42, 73))
    parser.add_argument("--subset-size", type=int, choices=(6, 9, 12, 15, 18))
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel outer-fold workers for LightGBM; use 3 on a 10-core M1 Max")
    parser.add_argument("--threads-per-fit", type=int,
                        help="LightGBM CPU threads per fit; with --workers 3 the default is 3")
    parser.add_argument("--allow-partial", action="store_true", help="For validate only: check available conditions without claiming a complete study")
    args = parser.parse_args(argv)
    config = read_config()
    scheduled = list(conditions(config))
    if args.command == "plan":
        print(json.dumps({"participant_sizes": config["participant_sizes"], "conditions_per_model": len(scheduled),
                          "models": {k: {"status": v["status"], **planned_workload(config, k)}
                                     for k, v in config["models"].items()}}, indent=2))
        return
    if args.command == "freeze":
        locked = freeze(ROOT, config)
        print("Frozen shared experiment:", locked["shared_sha256"])
        print("Ready models:", ", ".join(locked["models"]))
        return
    if not args.model:
        parser.error("--model is required for this command")
    model = MODEL_ALIASES.get(args.model, args.model)
    if args.workers < 1 or args.threads_per_fit is not None and args.threads_per_fit < 1:
        parser.error("--workers and --threads-per-fit must be positive integers")
    if args.workers > 1 and (args.command != "run" or model != "lightgbm"):
        parser.error("parallel workers are currently supported only for the LightGBM run command")
    if args.threads_per_fit is not None and model != "lightgbm":
        parser.error("--threads-per-fit applies only to LightGBM")
    frozen = verify_frozen(ROOT, config, model)
    folder = args.output or ROOT / "outputs" / model
    if args.command == "check":
        print(f"PASS: shared inputs and {model} match the frozen package. No training performed.")
        return
    filters = {"analysis": args.analysis, "training_domain": args.domain, "outer_fold": args.outer_fold,
               "subset_seed": args.subset_seed, "subset_size": args.subset_size}
    if args.command in ("validate", "summarise", "evaluate"):
        report = validate_outputs(folder / "tuning.csv", folder / "predictions.csv.gz", config,
                                  ROOT / "splits", model,
                                  require_complete=args.command != "validate" or not args.allow_partial)
        print(json.dumps(report, indent=2))
        if args.command != "validate":
            evaluation = evaluate_results(folder / "predictions.csv.gz")
            # Filter the display AFTER aggregation; never turn selected folds into a new study mean/CI.
            evaluation = {section: [row for row in rows if all(v is None or k not in row or row[k] == v
                                                              for k, v in filters.items())]
                          for section, rows in evaluation.items()}
            if args.outer_fold is not None or args.subset_seed is not None:
                print("Fold/seed filters limit detailed rows only; study summaries keep all available folds/seeds.")
            print_evaluation(evaluation, detail=args.detail)
        return
    selected = [c for c in scheduled if all(v is None or getattr(c, k) == v for k, v in filters.items())]
    if args.command == "run" and not selected:
        raise ValueError("No conditions match these filters in the frozen plan")
    recovery = recovery_location(ROOT, model, folder, args.recovery_dir, require_existing=args.command == "export")
    recovery.mkdir(parents=True, exist_ok=True)
    with single_writer(recovery):
        threads_per_fit = args.threads_per_fit
        if model == "lightgbm" and args.workers > 1 and threads_per_fit is None:
            threads_per_fit = max(1, (os.cpu_count() or args.workers) // args.workers)
        if model == "lightgbm" and args.workers > 1:
            print(f"LightGBM execution: workers={args.workers}; threads_per_fit={threads_per_fit}", flush=True)
        runner = Runner(ROOT, config, model, folder, frozen, recovery_dir=recovery,
                        require_existing=args.command == "export", threads_per_fit=threads_per_fit)
        try:
            if args.command == "run":
                run_conditions(runner, selected, args.workers)
            count = runner.export()
            if not (folder / "tuning.csv").exists():
                print("No completed selections or predictions to export. Recovery data is retained.")
                return
            report = validate_outputs(folder / "tuning.csv", folder / "predictions.csv.gz", config,
                                      ROOT / "splits", model, require_complete=count == len(scheduled))
            print(json.dumps(report, indent=2))
        except KeyboardInterrupt:
            runner.export()
            print("Interrupted. Completed fits are saved; repeat the same command to resume.", file=sys.stderr)
            raise SystemExit(130)
        except Exception:
            try:
                runner.export()
            except Exception as export_error:
                print(f"Could not refresh CSVs: {export_error}. Recovery data is retained.", file=sys.stderr)
            raise
        finally:
            runner.progress.close()


if __name__ == "__main__":
    main()
