"""FT contract tests on synthetic windows; never fit or inspect real outer-test data.

The optional CUDA test uses the frozen architecture and BF16 policy. All other
training tests explicitly disable AMP and force the CPU to work on any laptop.
"""
from contextlib import redirect_stdout
from dataclasses import asdict, replace
import copy
import csv
import gzip
import io
import json
from pathlib import Path
import tempfile
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch

from data import Condition, WindowTable
from metrics import inverse_frequency_sample_weights
from models import ft_transformer as ft
from results import validate_outputs, write_csv_atomic
from run import Runner, read_config

ROOT = Path(__file__).resolve().parent
SMALL = dict(d_token=8, n_blocks=1, n_heads=2, ffn_hidden=16,
             attention_dropout=0.0, ffn_dropout=0.0, batch_size=16,
             max_epochs=3, patience=2, weight_decay=1e-5, amp=False)


def table(people, *, domain="FL", shift=0.0):
    """Four classes per participant, distinct repeatable numeric features."""
    y = np.tile(np.arange(4), len(people))
    X = np.random.default_rng(817).normal(size=(len(y), 135)).astype(np.float32)
    X[:, :4] += 2 * np.eye(4, dtype=np.float32)[y]
    X += shift
    return WindowTable(X, y, np.repeat(people, 4), np.full(len(y), domain),
                       np.tile(np.arange(4), len(people)), np.arange(len(y)))


def fit_silently(train, validation=None, *, epochs=2, settings=None, **kwargs):
    options = dict(value=.001, seed=17, settings=dict(SMALL),
                   sample_weight=(kwargs["sample_weight"] if "sample_weight" in kwargs
                                  else inverse_frequency_sample_weights(train.y)[0]), epochs=epochs)
    if settings is not None:
        options["settings"].update(settings)
    options.update(kwargs)
    with redirect_stdout(io.StringIO()):
        return ft.fit(train, validation, **options)


class FTAdapterTests(unittest.TestCase):
    def setUp(self):
        self.cpu = patch("torch.cuda.is_available", return_value=False)
        self.cpu.start()
        self.addCleanup(self.cpu.stop)
        self.train = table(["train_a", "train_b", "train_c", "train_d"])
        self.validation = table(["validation_a", "validation_b"], shift=100.0)

    def test_scaler_uses_training_only_and_probabilities_keep_class_order(self):
        state, _ = fit_silently(self.train, self.validation, epochs=None)
        np.testing.assert_allclose(state["scaler"].mean, self.train.X.mean(0), atol=1e-6)
        np.testing.assert_allclose(state["scaler"].scale, self.train.X.std(0), atol=1e-6)
        self.assertGreater(state["scaler"].transform(self.validation.X).mean(), 50)
        before = state["scaler"].mean.copy()
        probabilities = ft.predict_proba(state, self.validation.X)
        np.testing.assert_array_equal(state["scaler"].mean, before)
        np.testing.assert_allclose(probabilities.sum(1), 1, atol=1e-6)
        self.assertEqual(probabilities.shape, (len(self.validation), 4))
        # Make the last layer independent of features so class-column mapping is known.
        with torch.no_grad():
            state["model"].head[-1].weight.zero_()
            state["model"].head[-1].bias.copy_(torch.tensor([-1., 0., 1., 2.]))
        expected = np.exp(np.array([-1., 0., 1., 2.]))
        expected /= expected.sum()
        np.testing.assert_allclose(ft.predict_proba(state, self.validation.X),
                                   np.tile(expected, (len(self.validation), 1)), atol=1e-6)
        self.assertEqual(ft.predict_proba(state, self.validation.X[:0]).shape, (0, 4))

    def test_patience_ties_and_best_checkpoint_are_restored(self):
        snapshots = []
        original = ft.predict_proba

        def observe(state, X):
            result = original(state, X)
            snapshots.append(result.copy())
            return result

        with patch.object(ft, "predict_proba", side_effect=observe), \
                patch.object(ft, "participant_macro_f1", side_effect=[.25, .75, .75, .5]) as score:
            state, info = fit_silently(self.train, self.validation, epochs=None,
                                      settings={"max_epochs": 10, "patience": 2})
        self.assertEqual(info["best_epoch"], 2)
        self.assertEqual(info["epochs_run"], 4)
        self.assertEqual(len(snapshots), 4)
        for args in score.call_args_list:
            np.testing.assert_array_equal(args.args[0], self.validation.y)
            np.testing.assert_array_equal(args.args[2], self.validation.participant_id)
        np.testing.assert_allclose(ft.predict_proba(state, self.validation.X), snapshots[1], atol=0, rtol=0)
        self.assertFalse(np.array_equal(snapshots[1], snapshots[-1]))

    def test_exact_refit_same_seed_and_serialization(self):
        state, info = fit_silently(self.train, epochs=3, settings={"patience": 1})
        self.assertEqual(info["epochs_run"], 3)
        self.assertIsNone(info["best_epoch"])
        self.assertTrue(all("validation_macro_f1" not in row for row in info["history"]))
        again, _ = fit_silently(self.train, epochs=3, settings={"patience": 1})
        before = ft.predict_proba(state, self.validation.X)
        np.testing.assert_array_equal(before, ft.predict_proba(again, self.validation.X))
        payload = ft.serialize(state)
        self.assertIsInstance(payload, bytes)
        torch.use_deterministic_algorithms(False)
        restored = ft.deserialize(payload)
        self.assertTrue(torch.are_deterministic_algorithms_enabled())
        self.assertEqual(asdict(state["config"]), asdict(restored["config"]))
        np.testing.assert_array_equal(state["scaler"].mean, restored["scaler"].mean)
        np.testing.assert_array_equal(state["scaler"].scale, restored["scaler"].scale)
        np.testing.assert_allclose(before, ft.predict_proba(restored, self.validation.X), atol=0, rtol=0)

    def test_unequal_class_weights_contribute_to_the_training_loss(self):
        labels = np.array([0] * 8 + [1] * 4 + [2] * 2 + [3] * 2)
        training = replace(self.train, y=labels)
        weights, _ = inverse_frequency_sample_weights(labels)
        losses = []
        original = ft.F.cross_entropy

        def capture(*args, **kwargs):
            loss = original(*args, **kwargs)
            losses.append(loss.detach().cpu().numpy().copy())
            return loss

        # Ordered batches deliberately have different class mixtures and weights.
        with patch.object(ft.F, "cross_entropy", side_effect=capture), \
                patch.object(torch, "randperm", side_effect=lambda n, device: torch.arange(n, device=device)):
            _, info = fit_silently(training, epochs=1, settings={"batch_size": 8}, sample_weight=weights)
        individual = np.concatenate(losses)
        expected = np.mean(individual * weights)
        self.assertAlmostEqual(info["history"][0]["loss"], expected, places=6)
        self.assertGreater(abs(expected - individual.mean()), 1e-4)

    def test_invalid_epoch_modes_and_participant_overlap_fail(self):
        for epochs in (None, 0, -1, 1.5, True):
            with self.subTest(epochs=epochs), self.assertRaises(ValueError):
                fit_silently(self.train, epochs=epochs)
        with self.assertRaises(ValueError):
            fit_silently(self.train, self.validation, epochs=1)
        with self.assertRaisesRegex(ValueError, "overlap"):
            fit_silently(self.train, self.train, epochs=None)

    def test_malformed_features_labels_and_weights_fail(self):
        nonfinite = self.train.X.copy()
        nonfinite[0, 0] = np.nan
        cases = [replace(self.train, X=self.train.X[:, :-1]),
                 replace(self.train, X=nonfinite),
                 replace(self.train, X=self.train.X[:-1]),
                 replace(self.train, y=self.train.y.astype(float) + .1),
                 replace(self.train, y=np.where(self.train.y == 3, 4, self.train.y)),
                 replace(self.train, participant_id=self.train.participant_id[:-1])]
        weights = inverse_frequency_sample_weights(self.train.y)[0]
        for index, train in enumerate(cases):
            with self.subTest(malformed_table=index), self.assertRaises(ValueError):
                fit_silently(train, sample_weight=weights)
        for bad_weights in (weights[:-1], weights * 2, -weights, weights * np.nan,
                            weights[:, None], np.zeros_like(weights)):
            with self.subTest(weights=repr(bad_weights)), self.assertRaises(ValueError):
                fit_silently(self.train, sample_weight=bad_weights)
        invalid_val = replace(self.validation, y=self.validation.y.astype(float) + .1)
        with self.assertRaises(ValueError):
            fit_silently(self.train, invalid_val, epochs=None)
        state, _ = fit_silently(self.train, epochs=1)
        for X in (nonfinite, self.train.X[:, :-1], self.train.X[0]):
            with self.subTest(prediction_shape=X.shape), self.assertRaises(ValueError):
                ft.predict_proba(state, X)

    def test_invalid_settings_fail_before_training(self):
        cases = [{"batch_size": 0}, {"batch_size": 1.5}, {"max_epochs": True},
                 {"patience": 0}, {"attention_dropout": -0.1}, {"ffn_dropout": 1.0},
                 {"weight_decay": -1}, {"d_token": 7}, {"n_blocks": 0}, {"n_heads": 0},
                 {"amp": "false"}, {"unknown_setting": 3}]
        for settings in cases:
            with self.subTest(settings=settings), self.assertRaises((ValueError, TypeError)):
                fit_silently(self.train, settings=settings)
        for value in (0, -0.1, np.inf, np.nan):
            with self.subTest(learning_rate=value), self.assertRaises(ValueError):
                fit_silently(self.train, value=value)
        with self.assertRaisesRegex(ValueError, "CUDA|BF16"):
            fit_silently(self.train, settings={"amp": True})


class RecordingAdapter:
    """Observe the real adapter without changing training or prediction."""
    def __init__(self):
        self.calls = []
        self.saved_final = False

    def fit(self, train, validation, **kwargs):
        np.testing.assert_allclose(kwargs["sample_weight"], inverse_frequency_sample_weights(train.y)[0])
        state, info = ft.fit(train, validation, **kwargs)
        self.calls.append(dict(value=kwargs["value"], seed=kwargs["seed"],
                               final=validation is None, requested_epochs=kwargs["epochs"], info=info))
        return state, info

    predict_proba = staticmethod(ft.predict_proba)
    deserialize = staticmethod(ft.deserialize)

    def serialize(self, state):
        payload = ft.serialize(state)
        self.saved_final = True
        return payload


class SyntheticData:
    subset_seeds = (17, 42, 73)
    feature_names = tuple(f"feature_{i}" for i in range(135))
    class_names = ("Sitting", "Standing", "Lying_Down", "Walking")

    def __init__(self, adapter):
        self.adapter = adapter
        self.outer_calls = 0
        self.interrupt_once = True

    def outer_test_participants(self, fold):
        return f"p{fold*2:02}", f"p{fold*2+1:02}"

    def subset_participants(self, fold, replicate, size):
        pool = [f"p{i:02}" for i in range(20) if f"p{i:02}" not in self.outer_test_participants(fold)]
        return tuple(np.random.default_rng(self.subset_seeds[replicate]).permutation(pool)[:size])

    def inner_participants(self, fold, replicate, size, inner):
        people = self.subset_participants(fold, replicate, size)
        return tuple(p for i, p in enumerate(people) if i % 3 != inner), tuple(p for i, p in enumerate(people) if i % 3 == inner)

    def selected_training_data(self, fold, replicate, size, domain, analysis="main", participants=None):
        return table(participants or self.subset_participants(fold, replicate, size), domain=domain)

    def inner_split(self, fold, replicate, size, domain, inner, analysis="main"):
        train, val = self.inner_participants(fold, replicate, size, inner)
        return table(train, domain=domain), table(val, domain=domain)

    def outer_test(self, fold):
        if not self.adapter.saved_final:
            raise AssertionError("outer test requested before final checkpoint serialization")
        if self.interrupt_once:
            self.interrupt_once = False
            raise RuntimeError("synthetic interruption after model checkpoint")
        self.outer_calls += 1
        return table(self.outer_test_participants(fold))


class FTSharedRunnerTests(unittest.TestCase):
    def test_real_ft_tuning_median_matched_reuse_recovery_and_two_outputs(self):
        config = copy.deepcopy(read_config(ROOT))
        model = "ft_transformer"
        config["models"][model].update(status="ready", settings=dict(SMALL, max_epochs=2),
                                         matched_epoch_policy="reuse_seed17_refit_epochs")
        adapter = RecordingAdapter()
        data = SyntheticData(adapter)
        frozen = {"shared_sha256": "synthetic", "models": {model: {"sha256": "synthetic"}}}
        with tempfile.TemporaryDirectory() as temporary, patch("torch.cuda.is_available", return_value=False):
            root = Path(temporary)
            (root / "splits").mkdir()
            reference = [dict(outer_fold=fold, participant_id=person, window_index=w, domain="FL",
                              class_name=data.class_names[w])
                         for fold in range(10) for person in data.outer_test_participants(fold) for w in range(4)]
            write_csv_atomic(root / "splits/outer_test_windows.csv.gz", reference[0].keys(), reference)
            runner = Runner(root, config, model, root / "output", frozen,
                            adapter=adapter, data=data, runtime={"synthetic": True})
            try:
                first = Condition("matched", "FL", 0, 42, 18)
                with redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "synthetic interruption"):
                    runner.run_condition(first)
                self.assertEqual(len(adapter.calls), 10)
                self.assertIsNotNone(runner.progress.get("checkpoint", first.key))
                runner.progress.close()
                # Reopening the same database must finish testing without fitting again.
                runner = Runner(root, config, model, root / "output", frozen, adapter=adapter, data=data,
                                runtime={"synthetic": True}, require_existing=True)
                with redirect_stdout(io.StringIO()):
                    runner.run_condition(first)
                    self.assertEqual(len(adapter.calls), 10)
                    runner.run_condition(Condition("matched", "FL", 0, 73, 18))
                    runner.run_condition(Condition("matched", "FL", 0, 17, 18))
                    runner.run_condition(first)
                self.assertEqual(len(adapter.calls), 12)
                self.assertEqual(data.outer_calls, 3)
                self.assertEqual(runner.export(), 3)
                self.assertEqual({p.name for p in (root / "output").iterdir()}, {"tuning.csv", "predictions.csv.gz"})
                report = validate_outputs(root / "output/tuning.csv", root / "output/predictions.csv.gz",
                                          config, root / "splits", model, require_complete=False)
                self.assertEqual((report["tuning_rows"], report["prediction_rows"]), (9, 24))
                with (root / "output/tuning.csv").open() as handle:
                    tuning = list(csv.DictReader(handle))
                selected = [row for row in tuning if row["selected"] == "True"]
                self.assertEqual(len(selected), 3)
                self.assertEqual({row["subset_seed"] for row in tuning}, {"17"})
                median = int(np.median([int(row["best_epoch"]) for row in selected]))
                self.assertGreaterEqual(median, 1)
                final_calls = [call for call in adapter.calls if call["final"]]
                self.assertEqual([call["seed"] for call in final_calls], [42, 73, 17])
                self.assertTrue(all(call["requested_epochs"] == median == call["info"]["epochs_run"] for call in final_calls))
                self.assertTrue(all(call["seed"] == 17 for call in adapter.calls if not call["final"]))
                with gzip.open(root / "output/predictions.csv.gz", "rt") as handle:
                    self.assertEqual({row["subset_seed"] for row in csv.DictReader(handle)}, {"17", "42", "73"})
            finally:
                runner.progress.close()


class FTCUDATests(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available() and torch.cuda.is_bf16_supported(), "CUDA BF16 GPU required")
    def test_frozen_cuda_policy_is_repeatable_and_survives_checkpoint(self):
        settings = dict(read_config(ROOT)["models"]["ft_transformer"]["settings"])
        self.assertTrue(settings["amp"])
        self.assertTrue(settings["deterministic"])
        self.assertEqual([settings[key] for key in ("d_token", "n_blocks", "n_heads")], [64, 2, 4])
        # Exercise one full 512-window batch and a remainder on the target GPU.
        training = table([f"gpu_{i}" for i in range(129)])
        options = dict(value=.0003, seed=17, settings=settings,
                       sample_weight=inverse_frequency_sample_weights(training.y)[0], epochs=1)
        with redirect_stdout(io.StringIO()):
            first, first_info = ft.fit(training, None, **options)
            second, second_info = ft.fit(training, None, **options)
        self.assertEqual(first_info["epochs_run"], second_info["epochs_run"])
        print(json.dumps(first_info["runtime"], indent=2))
        first_p = ft.predict_proba(first, training.X)
        np.testing.assert_array_equal(first_p, ft.predict_proba(second, training.X))
        validation = table(["gpu_validation_a", "gpu_validation_b"])
        snapshots = []
        original = ft.predict_proba

        def observe(state, X):
            probabilities = original(state, X)
            snapshots.append(probabilities.copy())
            return probabilities

        scores = [.9] + [.1] * settings["patience"]
        with redirect_stdout(io.StringIO()), patch.object(ft, "predict_proba", side_effect=observe), \
                patch.object(ft, "participant_macro_f1", side_effect=scores):
            best_state, info = ft.fit(training, validation, **{**options, "epochs": None})
        self.assertEqual(info["best_epoch"], 1)
        self.assertEqual(info["epochs_run"], settings["patience"] + 1)
        np.testing.assert_array_equal(ft.predict_proba(best_state, validation.X), snapshots[0])
        torch.use_deterministic_algorithms(False)
        restored = ft.deserialize(ft.serialize(first))
        self.assertTrue(torch.are_deterministic_algorithms_enabled())
        self.assertEqual(next(restored["model"].parameters()).device.type, "cuda")
        np.testing.assert_allclose(first_p, ft.predict_proba(restored, training.X), atol=1e-6, rtol=1e-5)


if __name__ == "__main__":
    if "--cuda-preflight" in sys.argv:
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise SystemExit("CUDA preflight FAILED: a CUDA GPU with BF16 support is required.")
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(FTCUDATests)
        outcome = unittest.TextTestRunner(verbosity=2).run(suite)
        raise SystemExit(0 if outcome.wasSuccessful() else 1)
    unittest.main(verbosity=2)
