"""Small synthetic tests: no real PAAWS model fitting or outer-test evaluation."""
from __future__ import annotations

from contextlib import closing, redirect_stdout
import copy
import csv
import gzip
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from data import Condition, ExperimentData, WindowTable, fit_standardizer, load_split
from metrics import classification_metrics, inverse_frequency_sample_weights, participant_macro_f1
from results import (PREDICTION_FIELDS, TUNING_FIELDS, aggregate_results, prediction_csv_bytes,
                     select_candidate, validate_outputs, write_csv_atomic)
from run import (Progress, Runner, conditions, freeze, partition_conditions_by_outer_fold,
                 read_config, run_conditions, single_writer, verify_frozen)


ROOT = Path(__file__).resolve().parent


def table(ids, domain="FL", constant=False):
    labels = np.tile(np.arange(4), len(ids) * 2)
    X = np.zeros((len(labels), 135), dtype=np.float32)
    if not constant:
        X[:, 0] = labels
        X[:, 1:5] = np.eye(4)[labels]
    return WindowTable(X, labels, np.repeat(ids, 8), np.full(len(labels), domain),
                       np.tile(np.arange(8), len(ids)), np.arange(len(labels)))


class FakeData:
    subset_seeds = (17, 42, 73)
    feature_names = tuple(f"feature_{i}" for i in range(135))
    class_names = ("Sitting", "Standing", "Lying_Down", "Walking")

    def __init__(self, adapter):
        self.adapter, self.outer_calls = adapter, 0

    def outer_test_participants(self, fold):
        return (f"p{fold*2:02}", f"p{fold*2+1:02}")

    def subset_participants(self, fold, replicate, size):
        pool = [f"p{i:02}" for i in range(20) if f"p{i:02}" not in self.outer_test_participants(fold)]
        return tuple(np.random.default_rng(self.subset_seeds[replicate]).permutation(pool)[:size])

    def inner_participants(self, fold, replicate, size, inner):
        people = self.subset_participants(fold, replicate, size)
        return tuple(p for i, p in enumerate(people) if i % 3 != inner), tuple(p for i, p in enumerate(people) if i % 3 == inner)

    def selected_training_data(self, fold, replicate, size, domain, analysis="main", participants=None):
        return table(participants or self.subset_participants(fold, replicate, size), domain)

    def inner_split(self, fold, replicate, size, domain, inner, analysis="main"):
        train, val = self.inner_participants(fold, replicate, size, inner)
        return table(train, domain), table(val, domain)

    def outer_test(self, fold):
        if not self.adapter.saved_final:
            raise AssertionError("outer-test requested before final-model serialization")
        self.outer_calls += 1
        return table(self.outer_test_participants(fold))


class FakeAdapter:
    def __init__(self):
        self.calls, self.saved_final = [], False

    def fit(self, train, validation, *, value, seed, settings, sample_weight, epochs=None):
        if validation is not None:
            assert not set(train.participant_id) & set(validation.participant_id)
        np.testing.assert_allclose(sample_weight, inverse_frequency_sample_weights(train.y)[0])
        self.calls.append((value, seed, validation is None, epochs))
        return {"final": validation is None}, {"best_epoch": None, "actual_rounds": 7}

    def predict_proba(self, state, X):
        return np.eye(4)[X[:, 0].astype(int)] * .96 + .01

    def serialize(self, state):
        self.saved_final = state["final"]
        return json.dumps(state).encode()

    def deserialize(self, content):
        state = json.loads(content)
        self.saved_final |= state["final"]
        return state


class SharedExperimentTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name)
        self.config = read_config(ROOT)
        self.adapter = FakeAdapter()
        self.data = FakeData(self.adapter)
        (self.path / "splits").mkdir()
        self.write_reference()
        self.frozen = {"shared_sha256": "synthetic", "models": {"lightgbm": {"sha256": "synthetic"}}}
        self.runner = Runner(self.path, self.config, "lightgbm", self.path / "output", self.frozen,
                             adapter=self.adapter, data=self.data, runtime={"synthetic": True})

    def tearDown(self):
        self.runner.progress.close()
        self.tmp.cleanup()

    def write_reference(self):
        rows = [{"outer_fold": fold, "participant_id": person, "window_index": w, "domain": "FL",
                 "class_name": self.data.class_names[w % 4]}
                for fold in range(10) for person in self.data.outer_test_participants(fold) for w in range(8)]
        write_csv_atomic(self.path / "splits/outer_test_windows.csv.gz", rows[0].keys(), rows)

    def validate(self, complete=False):
        return validate_outputs(self.path / "output/tuning.csv", self.path / "output/predictions.csv.gz",
                                self.config, self.path / "splits", "lightgbm", require_complete=complete)

    def test_schedule_and_inner_test_guard(self):
        from run import planned_workload
        planned = list(conditions(self.config))
        self.assertEqual(self.config["participant_sizes"], [6, 12, 18])
        self.assertEqual(len(planned), 240)
        self.assertEqual(len({c.key for c in planned}), len(planned))
        self.assertEqual(planned_workload(self.config, "lightgbm"),
                         {"conditions": 240, "tuned_conditions": 200, "inner_fits": 1800,
                          "refits": 240, "fits": 2040})
        expanded = copy.deepcopy(self.config)
        expanded["models"]["lightgbm"]["grid"].append(1023)
        self.assertEqual(planned_workload(expanded, "lightgbm")["fits"], 2640)
        c = Condition("main", "Formal_Lab", 0, 17, 6)
        with self.assertRaises(ValueError):
            load_split(self.data, c, inner_fold=0, include_outer_test=True)
        self.assertEqual(self.data.outer_calls, 0)

    def test_parallel_partition_keeps_each_outer_fold_on_one_worker(self):
        selected = list(conditions(self.config))
        groups = partition_conditions_by_outer_fold(selected, 3)
        self.assertEqual([len({c.outer_fold for c in group}) for group in groups], [4, 3, 3])
        self.assertEqual({c.key for group in groups for c in group}, {c.key for c in selected})
        owners = {}
        for worker, group in enumerate(groups):
            for fold in {condition.outer_fold for condition in group}:
                self.assertNotIn(fold, owners)
                owners[fold] = worker
        for fold in range(10):
            self.assertEqual(sum(any(c.outer_fold == fold for c in group) for group in groups), 1)

    def test_parallel_workers_share_recovery_without_losing_conditions(self):
        selected = [Condition("main", "Formal_Lab", fold, 17, 6) for fold in range(3)]
        with redirect_stdout(io.StringIO()):
            run_conditions(self.runner, selected, workers=3)
        self.assertEqual(len(self.adapter.calls), 30)
        self.assertEqual(self.data.outer_calls, 3)
        self.assertTrue(all(self.runner.progress.get("predictions", c.key) is not None for c in selected))

    def test_complete_condition_and_resume(self):
        c = Condition("main", "Formal_Lab", 0, 17, 6)
        with redirect_stdout(io.StringIO()):
            self.runner.run_condition(c)
            self.runner.run_condition(c)
        self.assertEqual(len(self.adapter.calls), 10)
        self.assertEqual(self.data.outer_calls, 1)
        self.runner.export()
        report = self.validate()
        self.assertEqual((report["tuning_rows"], report["prediction_rows"]), (9, 16))
        self.assertFalse(report["complete"])
        with self.assertRaises(ValueError):
            self.validate(complete=True)
        with (self.path / "output/tuning.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(sum(r["selected"] == "True" for r in rows), 3)
        self.assertTrue(all(r["best_epoch"] == "" for r in rows))

    def test_matched_reuses_only_seed17_and_refits_fresh(self):
        with redirect_stdout(io.StringIO()):
            for seed in (42, 73, 17):
                self.runner.run_condition(Condition("matched", "FL", 0, seed, 18))
        self.assertEqual(len(self.adapter.calls), 12)  # 9 seed17 inner fits + 3 fresh refits
        self.assertEqual([c[1] for c in self.adapter.calls if c[2]], [42, 73, 17])
        self.assertTrue(all(c[1] == 17 for c in self.adapter.calls if not c[2]))
        self.runner.export()
        self.assertEqual(self.validate()["tuning_rows"], 9)
        with (self.path / "output/tuning.csv").open() as handle:
            self.assertEqual({r["subset_seed"] for r in csv.DictReader(handle)}, {"17"})

    def test_checkpoint_recovers_interrupted_test_prediction(self):
        c = Condition("main", "FL", 0, 17, 6)
        original = self.data.outer_test
        self.data.outer_test = lambda _: (_ for _ in ()).throw(RuntimeError("interruption"))
        with redirect_stdout(io.StringIO()), self.assertRaises(RuntimeError):
            self.runner.run_condition(c)
        self.assertIsNotNone(self.runner.progress.get("checkpoint", c.key))
        self.data.outer_test = original
        with redirect_stdout(io.StringIO()):
            self.runner.run_condition(c)
        self.assertEqual(len(self.adapter.calls), 10)

    def test_validator_rejects_missing_window_wrong_truth_and_probability(self):
        with redirect_stdout(io.StringIO()):
            self.runner.run_condition(Condition("main", "FL", 0, 17, 6))
        self.runner.export()
        path = self.path / "output/predictions.csv.gz"
        with gzip.open(path, "rt") as handle:
            original = list(csv.DictReader(handle))
        for change in ("missing", "truth", "probability", "duplicate", "selection"):
            rows = copy.deepcopy(original)
            if change == "missing":
                rows.pop()
            elif change == "duplicate":
                rows.append(rows[0])
            elif change == "truth":
                rows[0]["true_activity"] = "3"
            elif change == "selection":
                rows[0]["hyperparameter_value"] = "511"
            else:
                rows[0]["sitting_prob"] = "nan"
            write_csv_atomic(path, PREDICTION_FIELDS, rows)
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.validate()

    def test_progress_integrity_and_writer_lock(self):
        self.runner.progress.put("test", "key", b"original")
        with self.runner.progress.db:
            self.runner.progress.db.execute("UPDATE items SET value=? WHERE kind='test'", (b"changed",))
        with self.assertRaises(ValueError):
            self.runner.progress.get("test", "key")
        with single_writer(self.path):
            with self.assertRaises(RuntimeError):
                with single_writer(self.path):
                    pass

    def test_results_folder_has_only_two_files_and_recovery_is_separate(self):
        output = io.StringIO()
        with redirect_stdout(output):
            self.runner.run_condition(Condition("main", "Formal_Lab", 0, 17, 6))
        self.runner.export()
        self.assertEqual({p.name for p in (self.path / "output").iterdir()},
                         {"tuning.csv", "predictions.csv.gz"})
        recovery = self.path / "recovery/lightgbm"
        self.assertTrue((recovery / "progress.sqlite3").is_file())
        with single_writer(recovery):
            self.assertTrue((recovery / "run.lock").is_file())
        self.assertIn("fit_seconds", output.getvalue())
        self.assertIn("actual_rounds", output.getvalue())
        # Recovery belongs to this exact destination and cannot overwrite another run.
        with self.assertRaises(ValueError):
            Runner(self.path, self.config, "lightgbm", self.path / "other_output", self.frozen,
                   adapter=self.adapter, data=self.data, runtime={"synthetic": True})

    def test_absent_recovery_cannot_replace_existing_result_files(self):
        with redirect_stdout(io.StringIO()):
            self.runner.run_condition(Condition("main", "FL", 0, 17, 6))
        self.runner.export()
        original = {name: (self.path / "output" / name).read_bytes()
                    for name in ("tuning.csv", "predictions.csv.gz")}
        missing = self.path / "missing_recovery"
        for require_existing in (False, True):
            with self.subTest(require_existing=require_existing), self.assertRaises((ValueError, FileNotFoundError)):
                Runner(self.path, self.config, "lightgbm", self.path / "output", self.frozen,
                       adapter=self.adapter, data=self.data, runtime={"synthetic": True},
                       recovery_dir=missing, require_existing=require_existing)
        self.assertFalse((missing / "progress.sqlite3").exists())
        for name, content in original.items():
            self.assertEqual((self.path / "output" / name).read_bytes(), content)

    def test_missing_and_unidentified_recovery_databases_are_rejected(self):
        import sqlite3
        missing = self.path / "missing.sqlite3"
        with self.assertRaises((ValueError, FileNotFoundError)):
            Progress(missing, {"identity": "test"}, require_existing=True)
        self.assertFalse(missing.exists())
        unknown = self.path / "unknown.sqlite3"
        with closing(sqlite3.connect(unknown)) as db, db:
            db.execute("CREATE TABLE unrelated (value TEXT)")
        with self.assertRaises(ValueError):
            Progress(unknown, {"identity": "test"})
        with closing(sqlite3.connect(unknown)) as db, db:
            if db.execute("SELECT name FROM sqlite_master WHERE name='items'").fetchone():
                self.assertIsNone(db.execute("SELECT value FROM items WHERE kind='metadata' AND key='identity'").fetchone())

    def test_empty_recovery_export_does_not_overwrite_result_files(self):
        tuning, predictions = self.path / "output/tuning.csv", self.path / "output/predictions.csv.gz"
        tuning.parent.mkdir()
        tuning.write_bytes(b"existing tuning results")
        predictions.write_bytes(b"existing prediction results")
        with self.assertRaises(ValueError):
            self.runner.export()
        with self.assertRaises(ValueError):
            Runner(self.path, self.config, "lightgbm", self.path / "output", self.frozen,
                   adapter=self.adapter, data=self.data, runtime={"synthetic": True})
        self.assertEqual(tuning.read_bytes(), b"existing tuning results")
        self.assertEqual(predictions.read_bytes(), b"existing prediction results")

    def test_restoring_older_recovery_cannot_shrink_newer_exported_results(self):
        import sqlite3
        with redirect_stdout(io.StringIO()):
            self.runner.run_condition(Condition("main", "FL", 0, 17, 6))
        self.runner.export()
        backup = self.path / "older.sqlite3"
        with closing(sqlite3.connect(backup)) as database, database:
            self.runner.progress.db.backup(database)
        with redirect_stdout(io.StringIO()):
            self.runner.run_condition(Condition("main", "FL", 0, 42, 6))
        self.runner.export()
        original = {name: (self.path / "output" / name).read_bytes()
                    for name in ("tuning.csv", "predictions.csv.gz")}
        self.assertEqual(self.validate()["prediction_conditions"], 2)
        self.runner.progress.close()
        (self.path / "recovery/lightgbm/progress.sqlite3").write_bytes(backup.read_bytes())
        with self.assertRaisesRegex(ValueError, "Existing results"):
            Runner(self.path, self.config, "lightgbm", self.path / "output", self.frozen,
                   adapter=self.adapter, data=self.data, runtime={"synthetic": True}, require_existing=True)
        for name, content in original.items():
            self.assertEqual((self.path / "output" / name).read_bytes(), content)
        self.assertEqual(self.validate()["prediction_conditions"], 2)

    def test_interrupted_two_file_export_resumes_without_refitting(self):
        import os
        with redirect_stdout(io.StringIO()):
            self.runner.run_condition(Condition("main", "FL", 0, 17, 6))
        self.runner.export()
        original = {name: (self.path / "output" / name).read_bytes()
                    for name in ("tuning.csv", "predictions.csv.gz")}
        with redirect_stdout(io.StringIO()):
            self.runner.run_condition(Condition("main", "FL", 0, 42, 6))
        real_replace = os.replace
        interrupted = False
        def interrupt_second_result(source, destination):
            nonlocal interrupted
            if Path(destination).resolve() == (self.path / "output/predictions.csv.gz").resolve() and not interrupted:
                interrupted = True
                raise OSError("simulated interruption between result replacements")
            return real_replace(source, destination)
        with patch("run.os.replace", side_effect=interrupt_second_result), self.assertRaisesRegex(OSError, "simulated"):
            self.runner.export()
        self.assertTrue(interrupted)
        self.assertNotEqual((self.path / "output/tuning.csv").read_bytes(), original["tuning.csv"])
        self.assertEqual((self.path / "output/predictions.csv.gz").read_bytes(), original["predictions.csv.gz"])
        self.runner.progress.close()
        self.runner = Runner(self.path, self.config, "lightgbm", self.path / "output", self.frozen,
                             adapter=self.adapter, data=self.data, runtime={"synthetic": True}, require_existing=True)
        self.assertEqual(self.runner.export(), 2)
        self.assertEqual(len(self.adapter.calls), 20)
        report = self.validate()
        self.assertEqual((report["tuning_rows"], report["prediction_rows"]), (18, 32))

    def test_selection_rules_and_training_only_scaling(self):
        rows = [{"hyperparameter_value": value, "inner_fold": fold,
                 "participant_balanced_validation_macro_f1": score, "best_epoch": None}
                for value, score in ((.03, .6005), (.1, .6), (.3, .59)) for fold in range(3)]
        selected, epochs = select_candidate(rows, self.config["models"]["multinomial_logistic_regression"])
        self.assertEqual((selected, epochs), (.1, None))
        values = np.array([[1., 2.], [3., 2.]])
        scaler = fit_standardizer(values)
        np.testing.assert_allclose(scaler.mean, [2, 2])
        np.testing.assert_allclose(scaler.scale, [1, 1])
        self.assertGreater(scaler.transform(np.array([[100., 200.]]))[0, 0], 90)

    def test_participant_metric_does_not_pool_windows(self):
        truth = np.array([0] * 100 + [1])
        pred = np.array([0] * 101)
        ids = np.array(["many"] * 100 + ["few"])
        self.assertEqual(participant_macro_f1(truth, pred, ids), .5)
        self.assertNotEqual(classification_metrics(truth, pred).macro_f1, .5)

    def test_real_model_adapters_on_synthetic_data(self):
        from models import lightgbm, logistic_regression
        training = table(["a", "b", "c", "d"])
        weights, _ = inverse_frequency_sample_weights(training.y)
        for name, module, value in (("lightgbm", lightgbm, 127),
                                    ("multinomial_logistic_regression", logistic_regression, .1)):
            fitted, info = module.fit(training, None, value=value, seed=17,
                                      settings=self.config["models"][name]["settings"], sample_weight=weights)
            before = module.predict_proba(fitted, training.X)
            after = module.predict_proba(module.deserialize(module.serialize(fitted)), training.X)
            np.testing.assert_allclose(before, after, atol=0, rtol=0)
            self.assertEqual(before.shape, (len(training), 4))
        constant = table(["a", "b", "c", "d"], constant=True)
        _, info = lightgbm.fit(constant, None, value=127, seed=17,
                               settings=self.config["models"]["lightgbm"]["settings"], sample_weight=weights)
        self.assertEqual((info["actual_rounds"], info["num_trees"]), (1, 4))
        settings = {**self.config["models"]["multinomial_logistic_regression"]["settings"], "max_iter": 1}
        with self.assertRaisesRegex(RuntimeError, "converge"):
            logistic_regression.fit(training, None, value=.1, seed=17, settings=settings, sample_weight=weights)

    def test_ft_epoch_selection_and_matched_reuse(self):
        class FakeFT(FakeAdapter):
            def fit(self, train, validation, **kwargs):
                fold = len([c for c in self.calls if not c[2]]) // 3
                state, info = super().fit(train, validation, **kwargs)
                info.update(best_epoch=(2, 5, 18)[fold] if validation is not None else None,
                            epochs_run=kwargs["epochs"] if validation is None else 20)
                return state, info
        model = "ft_transformer"
        adapter = FakeFT()
        runner = Runner(self.path, self.config, model, self.path / "ft_output",
                        {"shared_sha256": "test", "models": {model: {"sha256": "test"}}},
                        adapter=adapter, data=FakeData(adapter), runtime={})
        try:
            with redirect_stdout(io.StringIO()):
                runner.run_condition(Condition("matched", "FL", 0, 17, 18))
                runner.run_condition(Condition("matched", "FL", 0, 42, 18))
            self.assertEqual(len(adapter.calls), 11)
            self.assertEqual([c[3] for c in adapter.calls if c[2]], [5, 5])
            runner.export()
            report = validate_outputs(self.path / "ft_output/tuning.csv", self.path / "ft_output/predictions.csv.gz",
                                      self.config, self.path / "splits", model, require_complete=False)
            self.assertEqual(report["tuning_rows"], 9)
        finally:
            runner.progress.close()


class ProtocolIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name)
        self.config = read_config(ROOT)

    def tearDown(self):
        self.tmp.cleanup()

    def fingerprint_fixture(self):
        from run import SHARED_SOURCE, SPLIT_FILES
        for folder in ("data/shards", "splits", "models"):
            (self.path / folder).mkdir(parents=True)
        for filename in SHARED_SOURCE:
            (self.path / filename).write_bytes(b"synthetic source: " + filename.encode())
        for person in range(20):
            for domain in ("FL", "Formal_Lab"):
                (self.path / f"data/shards/p{person:02}__{domain}.npz").write_bytes(
                    f"fingerprint-only fixture {person} {domain}".encode())
        for filename in ("feature_schema_and_counts.json", "class_mapping_v1.json"):
            (self.path / "data" / filename).write_text("{}", encoding="utf-8")
        for filename in SPLIT_FILES:
            (self.path / "splits" / filename).write_bytes(b"fingerprint-only manifest")
        for spec in self.config["models"].values():
            path = self.path / (spec["module"].replace(".", "/") + ".py")
            path.write_bytes(b"synthetic model adapter")

    def test_freeze_blocks_shared_data_source_split_and_settings_changes(self):
        self.fingerprint_fixture()
        original = freeze(self.path, self.config)
        model = "multinomial_logistic_regression"
        self.assertEqual(verify_frozen(self.path, self.config, model), original)
        for filename in ("data/shards/p00__FL.npz", "data.py", "splits/outer_participant_roles.csv"):
            path = self.path / filename
            saved = path.read_bytes()
            path.write_bytes(saved + b"changed")
            with self.subTest(filename=filename):
                with self.assertRaises(ValueError):
                    verify_frozen(self.path, self.config, model)
                with self.assertRaises(ValueError):
                    freeze(self.path, self.config)
            path.write_bytes(saved)
        changed = copy.deepcopy(self.config)
        changed["participant_sizes"] = [6, 9, 12, 15, 18]
        with self.assertRaises(ValueError):
            verify_frozen(self.path, changed, model)
        with self.assertRaises(ValueError):
            freeze(self.path, changed)
        self.assertEqual(verify_frozen(self.path, self.config, model), original)
        (self.path / "splits/.DS_Store").write_bytes(b"Finder metadata")
        self.assertEqual(verify_frozen(self.path, self.config, model), original)
        (self.path / "splits/outer_participant_roles.csv").unlink()
        with self.assertRaises(FileNotFoundError):
            freeze(self.path, self.config)

    def test_finishing_ft_preserves_frozen_lr_but_changing_lr_is_blocked(self):
        # Exercise the pre-integration state even after the real FT is ready.
        self.config["models"]["ft_transformer"].update(
            status="pending", settings={}, matched_epoch_policy=None)
        self.fingerprint_fixture()
        lr = "multinomial_logistic_regression"
        first = freeze(self.path, self.config)
        with self.assertRaises(ValueError):
            verify_frozen(self.path, self.config, "ft_transformer")
        updated = copy.deepcopy(self.config)
        updated["models"]["ft_transformer"].update(
            status="ready", settings={"d_token": 64, "max_epochs": 100},
            matched_epoch_policy="reuse_seed17_refit_epochs")
        (self.path / "models/ft_transformer.py").write_bytes(b"finished FT adapter")
        second = freeze(self.path, updated)
        self.assertEqual(first["shared_sha256"], second["shared_sha256"])
        self.assertEqual(first["models"][lr], second["models"][lr])
        self.assertIn("ft_transformer", verify_frozen(self.path, updated, lr)["models"])
        verify_frozen(self.path, updated, "ft_transformer")
        altered_lr = copy.deepcopy(updated)
        altered_lr["models"][lr]["settings"]["tol"] = .01
        with self.assertRaises(ValueError):
            freeze(self.path, altered_lr)
        with self.assertRaises(ValueError):
            verify_frozen(self.path, altered_lr, lr)
        (self.path / "models/logistic_regression.py").write_bytes(b"different LR adapter")
        with self.assertRaises(ValueError):
            freeze(self.path, updated)
        with self.assertRaises(ValueError):
            verify_frozen(self.path, updated, lr)

    def complete_results_fixture(self):
        """Every condition, with unequal window counts and nonconstant seed/fold scores."""
        from dataclasses import asdict
        model, value = "lightgbm", self.config["models"]["lightgbm"]["grid"][0]
        split_dir = self.path / "splits"
        split_dir.mkdir()
        reference = [{"outer_fold": fold, "participant_id": f"p{fold * 2 + person:02}",
                      "window_index": window, "domain": "FL", "class_name": ("Sitting", "Standing")[person]}
                     for fold in range(10) for person, count in ((0, 20), (1, 1)) for window in range(count)]
        write_csv_atomic(split_dir / "outer_test_windows.csv.gz", reference[0].keys(), reference)
        tuning, predictions = [], []
        for condition in conditions(self.config):
            common = {"model": model, **asdict(condition), "hyperparameter_name": "num_leaves"}
            if condition.analysis == "main" or condition.subset_seed == 17:
                for candidate in self.config["models"][model]["grid"]:
                    for inner in range(3):
                        tuning.append(dict(common, hyperparameter_value=candidate, inner_fold=inner,
                                           participant_balanced_validation_macro_f1=.6 if candidate == value else .5,
                                           selected=candidate == value, best_epoch=""))
            replicate = self.config["subset_seeds"].index(condition.subset_seed)
            is_fl = condition.training_domain == "FL"
            correct = condition.outer_fold + replicate + (2 if is_fl else 1)
            small_correct = int(is_fl and (condition.outer_fold + replicate) % 2 == 0)
            for person, count in ((0, 20), (1, 1)):
                for window in range(count):
                    prediction = (0 if window < correct else 1) if person == 0 else small_correct
                    probabilities = [0.0] * 4
                    probabilities[prediction] = 1.0
                    row = dict(common, hyperparameter_value=value,
                               test_participant=f"p{condition.outer_fold * 2 + person:02}", test_window=window,
                               true_activity=person, predicted_activity=prediction)
                    row.update(zip(PREDICTION_FIELDS[-4:], probabilities))
                    predictions.append(row)
        write_csv_atomic(self.path / "tuning.csv", TUNING_FIELDS, tuning)
        write_csv_atomic(self.path / "predictions.csv.gz", PREDICTION_FIELDS, predictions)
        return tuning, predictions

    def validate_complete(self):
        return validate_outputs(self.path / "tuning.csv", self.path / "predictions.csv.gz",
                                self.config, self.path / "splits", "lightgbm", require_complete=True)

    def test_full_schedule_participant_seed_fold_means_and_paired_intervals(self):
        from metrics import T_975_DF9
        tuning, predictions = self.complete_results_fixture()
        report = self.validate_complete()
        self.assertTrue(report["complete"])
        self.assertEqual(report["tuning_rows"], len(tuning))
        self.assertEqual(report["prediction_rows"], len(predictions))
        summaries = aggregate_results(self.path / "predictions.csv.gz")
        # Independent arithmetic for a one-class participant: F1=2TP/(N+TP).
        expected = {}
        for domain in ("FL", "Formal_Lab"):
            is_fl = domain == "FL"
            per_fold = []
            for fold in range(10):
                seed_scores = []
                for replicate in range(3):
                    correct = fold + replicate + (2 if is_fl else 1)
                    small_score = int(is_fl and (fold + replicate) % 2 == 0)
                    seed_scores.append(((2 * correct / (20 + correct) + small_score) / 2,
                                        (correct / 20 + small_score) / 2))
                per_fold.append(np.mean(seed_scores, axis=0))
            expected[domain] = np.asarray(per_fold)
        expected["FL_minus_Formal_Lab"] = expected["FL"] - expected["Formal_Lab"]
        for row in summaries:
            metric = 1 if row["metric"] in ("balanced_accuracy", "accuracy") else 0
            values = expected[row["training_domain"]][:, metric]
            half = T_975_DF9 * values.std(ddof=1) / np.sqrt(10)
            np.testing.assert_allclose([row["mean"], row["ci_low"], row["ci_high"]],
                                       [values.mean(), values.mean() - half, values.mean() + half], atol=1e-14)
        self.assertEqual(len(summaries), (len(self.config["participant_sizes"]) + 1) * 3 * 4)
        # A missing small participant cannot silently turn into a window-weighted score.
        removed_person = [r for r in predictions if not (r["analysis"] == "main"
                          and r["training_domain"] == "FL" and r["outer_fold"] == 0
                          and r["subset_seed"] == 17 and r["subset_size"] == 6
                          and r["test_participant"] == "p01")]
        write_csv_atomic(self.path / "predictions.csv.gz", PREDICTION_FIELDS, removed_person)
        with self.assertRaises(ValueError):
            self.validate_complete()
        with self.assertRaisesRegex(ValueError, "two test participants"):
            aggregate_results(self.path / "predictions.csv.gz")

    def test_tuning_validator_rejects_incomplete_grid_wrong_flags_and_fake_matched_repeats(self):
        original, _ = self.complete_results_fixture()
        for change in ("missing_inner", "duplicate_inner", "wrong_selected", "matched_seed", "nonfinite_score"):
            rows = copy.deepcopy(original)
            if change == "missing_inner":
                rows.pop(0)
            elif change == "duplicate_inner":
                rows.append(rows[0].copy())
            elif change == "wrong_selected":
                rows[0]["selected"] = False
            elif change == "matched_seed":
                next(row for row in rows if row["analysis"] == "matched")["subset_seed"] = 42
            else:
                rows[0]["participant_balanced_validation_macro_f1"] = "nan"
            write_csv_atomic(self.path / "tuning.csv", TUNING_FIELDS, rows)
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.validate_complete()

    def test_saved_outputs_validate_and_print_analysis_without_recovery_or_extra_csv(self):
        import run
        self.complete_results_fixture()
        before = {str(p.relative_to(self.path)): p.read_bytes()
                  for p in self.path.rglob("*") if p.is_file()}
        self.assertFalse((self.path / "recovery").exists())
        for command in ("validate", "summarise", "evaluate"):
            printed = io.StringIO()
            with patch.object(run, "ROOT", self.path), patch.object(run, "read_config", return_value=self.config), \
                    patch.object(run, "verify_frozen", return_value={}), redirect_stdout(printed):
                run.main([command, "--model", "lightgbm", "--output", str(self.path)])
            self.assertTrue(printed.getvalue().strip())
            if command != "validate":
                self.assertIn("accuracy", printed.getvalue())
                self.assertIn("macro_f1", printed.getvalue())
        self.assertFalse((self.path / "recovery").exists())
        self.assertEqual({str(p.relative_to(self.path)): p.read_bytes()
                          for p in self.path.rglob("*") if p.is_file()}, before)

    def test_per_class_scores_confusions_supports_and_absent_classes(self):
        from results import evaluate_results
        self.complete_results_fixture()
        report = evaluate_results(self.path / "predictions.csv.gz")
        self.assertEqual(set(report), {"summary", "participants", "folds", "confusions", "class_summary"})
        self.assertEqual(len(report["participants"]), 480)
        self.assertEqual(len(report["confusions"]), 240)
        def first_condition(row):
            return (row["analysis"] == "main" and row["training_domain"] == "FL"
                    and row["subset_size"] == 6 and row.get("outer_fold", 0) == 0
                    and row.get("subset_seed", 17) == 17)
        person = next(row for row in report["participants"] if first_condition(row)
                      and row["test_participant"] == "p00")
        self.assertAlmostEqual(person["accuracy"], 2 / 20)
        self.assertAlmostEqual(person["macro_f1"], 4 / 22)
        self.assertEqual(person["per_class"][0]["support"], 20)
        self.assertAlmostEqual(person["per_class"][0]["precision"], 1)
        self.assertAlmostEqual(person["per_class"][0]["recall"], .1)
        for item in person["per_class"][1:]:
            self.assertEqual(item["support"], 0)
            self.assertTrue(all(item[k] is None for k in ("precision", "recall", "f1")))
        confusion = next(row for row in report["confusions"] if first_condition(row))
        self.assertEqual(confusion["pooled_counts"], [[2, 18, 0, 0], [0, 1, 0, 0], [0]*4, [0]*4])
        self.assertEqual(confusion["contributing_participants_per_class"], [1, 1, 0, 0])
        self.assertEqual(confusion["participant_balanced_row_normalized"],
                         [[.1, .9, 0., 0.], [0., 1., 0., 0.], [None]*4, [None]*4])
        fold = next(row for row in report["folds"] if first_condition(row))
        self.assertAlmostEqual(fold["per_class"][0]["f1"], (4/22 + 6/23 + 8/24) / 3)
        self.assertEqual(fold["per_class"][0]["support_per_seed"], 20)
        self.assertEqual(fold["per_class"][0]["contributing_seeds"], 3)
        missing = [row for row in report["class_summary"] if row["class_id"] in (2, 3)]
        self.assertTrue(missing)
        self.assertTrue(all(row["mean"] is None and row["ci_low"] is None
                            and row["ci_high"] is None and row["contributing_folds"] == 0 for row in missing))

    def test_cli_export_with_missing_recovery_preserves_saved_results(self):
        import run
        self.complete_results_fixture()
        before = {name: (self.path / name).read_bytes() for name in ("tuning.csv", "predictions.csv.gz")}
        frozen = {"shared_sha256": "synthetic", "models": {"lightgbm": {"sha256": "synthetic"}}}
        with patch.object(run, "ROOT", self.path), patch.object(run, "read_config", return_value=self.config), \
                patch.object(run, "verify_frozen", return_value=frozen), \
                patch.object(run, "ExperimentData", return_value=FakeData(FakeAdapter())), \
                patch.object(run, "environment", return_value={"synthetic": True}), redirect_stdout(io.StringIO()):
            with self.assertRaises((ValueError, FileNotFoundError)):
                run.main(["export", "--model", "lightgbm", "--output", str(self.path)])
        for name, content in before.items():
            self.assertEqual((self.path / name).read_bytes(), content)
        self.assertFalse((self.path / "recovery/lightgbm/progress.sqlite3").exists())


class RetainedMethodologyTests(unittest.TestCase):
    def test_supplied_manifests_keep_original_nested_participant_cv(self):
        config = read_config(ROOT)
        data = ExperimentData(ROOT / "data", ROOT / "splits", config)
        with (ROOT / "splits/nested_participant_subsets.csv").open() as handle:
            manifest = list(csv.DictReader(handle))
        held_out = []
        for fold in range(10):
            test = data.outer_test_participants(fold)
            held_out.extend(test)
            self.assertEqual(len(test), 2)
            for replicate, seed in enumerate((17, 42, 73)):
                previous = ()
                for size in (6, 12, 18):
                    rows = sorted((r for r in manifest if int(r["outer_fold"]) == fold
                                   and int(r["replicate_id"]) == replicate and int(r["subset_size"]) == size),
                                  key=lambda r: int(r["subset_position"]))
                    expected = tuple(r["participant_id"] for r in rows)
                    actual = data.subset_participants(fold, replicate, size)
                    self.assertEqual(actual, expected)
                    self.assertEqual(actual[:len(previous)], previous)
                    self.assertTrue(all(int(r["replicate_seed"]) == seed for r in rows))
                    self.assertFalse(set(actual) & set(test))
                    previous = actual
                    for inner in range(3):
                        train, val = data.inner_participants(fold, replicate, size, inner)
                        self.assertEqual(val, tuple(p for i, p in enumerate(expected) if i % 3 == inner))
                        self.assertEqual(train, tuple(p for i, p in enumerate(expected) if i % 3 != inner))
                        self.assertEqual(len(val), size // 3)
                        self.assertFalse(set(train) & set(val))
        self.assertEqual(len(held_out), len(set(held_out)))
        self.assertEqual(len(held_out), 20)

    def test_original_metric_and_scaling_formulas_on_fixed_independent_examples(self):
        from metrics import metrics_from_confusion, mean_and_95pct_t_ci_across_10_outer_folds
        cm = np.array([[8, 2, 0, 0], [1, 3, 0, 0], [0, 0, 0, 0], [1, 0, 0, 4]])
        values = metrics_from_confusion(cm)
        # The unsupported lying class is NA; predicted mistakes still hurt true classes.
        expected_f1 = np.array([16 / 20, 6 / 9, np.nan, 8 / 9])
        np.testing.assert_allclose(values.f1, expected_f1, equal_nan=True)
        self.assertAlmostEqual(values.macro_f1, (16 / 20 + 6 / 9 + 8 / 9) / 3)
        self.assertAlmostEqual(values.balanced_accuracy, (.8 + .75 + .8) / 3)
        self.assertAlmostEqual(values.accuracy, 15 / 19)
        self.assertAlmostEqual(values.weighted_f1, (10 * 16 / 20 + 4 * 6 / 9 + 5 * 8 / 9) / 19)
        y = np.repeat(np.arange(4), [1, 2, 3, 4])
        weights, classes = inverse_frequency_sample_weights(y)
        np.testing.assert_allclose(classes, [2.5, 1.25, 10 / 12, .625])
        self.assertAlmostEqual(weights.mean(), 1)
        self.assertAlmostEqual(sum(weights[y == 0]), sum(weights[y == 3]))
        scaler = fit_standardizer(np.array([[1., 4.], [3., 4.]]))
        np.testing.assert_array_equal(scaler.mean, [2., 4.])
        np.testing.assert_array_equal(scaler.scale, [1., 1.])
        np.testing.assert_array_equal(scaler.transform(np.array([[5., 8.]])), [[3., 4.]])
        mean, lower, upper = mean_and_95pct_t_ci_across_10_outer_folds(np.arange(10, dtype=float))
        half = 2.2621571628540993 * np.sqrt(82.5 / 9) / np.sqrt(10)
        np.testing.assert_allclose([mean, lower, upper], [4.5, 4.5 - half, 4.5 + half])


if __name__ == "__main__":
    unittest.main(verbosity=2)
