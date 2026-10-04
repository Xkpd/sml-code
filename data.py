"""Load the shared, frozen participant splits; never create new random splits.

Training and validation use the requested domain. Outer testing always uses
the complete eligible FL windows of the two held-out participants.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import gzip
import json
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Condition:
    analysis: str
    training_domain: str
    outer_fold: int
    subset_seed: int
    subset_size: int

    def __post_init__(self):
        if self.analysis not in ("main", "matched"):
            raise ValueError("analysis must be main or matched")
        if self.training_domain not in ("FL", "Formal_Lab"):
            raise ValueError("training_domain must be FL or Formal_Lab")
        if self.outer_fold not in range(10):
            raise ValueError("outer_fold must be between 0 and 9")
        if self.subset_size not in (6, 9, 12, 15, 18):
            raise ValueError("subset_size must be 6, 9, 12, 15 or 18")
        if self.analysis == "matched" and self.subset_size != 18:
            raise ValueError("matched sensitivity is predeclared for n=18 only")

    @property
    def key(self) -> str:
        return (f"{self.analysis}__{self.training_domain}__outer{self.outer_fold}"
                f"__seed{self.subset_seed}__n{self.subset_size}")


@dataclass(frozen=True)
class WindowTable:
    X: np.ndarray
    y: np.ndarray
    participant_id: np.ndarray
    domain: np.ndarray
    window_index: np.ndarray
    window_start_ns: np.ndarray

    def __len__(self) -> int:
        return len(self.y)


@dataclass(frozen=True)
class LoadedSplit:
    train: WindowTable
    validation: WindowTable | None
    test: WindowTable | None
    context: dict


@dataclass(frozen=True)
class Standardizer:
    mean: np.ndarray
    scale: np.ndarray

    def transform(self, X: np.ndarray) -> np.ndarray:
        values = np.asarray(X, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != len(self.mean):
            raise ValueError("feature matrix does not match fitted standardizer")
        return ((values - self.mean) / self.scale).astype(np.float32)


def fit_standardizer(X_train: np.ndarray) -> Standardizer:
    """Fit on this training fold only; reuse the result on validation/test."""
    values = np.asarray(X_train, dtype=np.float64)
    if values.ndim != 2 or not len(values) or np.any(~np.isfinite(values)):
        raise ValueError("training features must be a non-empty finite matrix")
    mean = values.mean(axis=0)
    scale = values.std(axis=0, ddof=0)
    scale[scale < 1e-12] = 1.0
    return Standardizer(mean.astype(np.float32), scale.astype(np.float32))


def _concatenate(tables: list[WindowTable]) -> WindowTable:
    if not tables:
        raise ValueError("no window tables to concatenate")
    return WindowTable(**{
        name: np.concatenate([getattr(table, name) for table in tables])
        for name in WindowTable.__dataclass_fields__
    })


class ExperimentData:
    """Read the v1.1 feature shards and the team's supplied split manifests."""

    def __init__(self, dataset_dir: Path | str, split_dir: Path | str, config: dict):
        self.dataset_dir = Path(dataset_dir).resolve()
        self.split_dir = Path(split_dir).resolve()
        self.config = config
        self.schema = json.loads(
            (self.dataset_dir / "feature_schema_and_counts.json").read_text(encoding="utf-8")
        )
        self.feature_names = tuple(self.schema["feature_names"])
        self.class_names = tuple(config["classes"])
        self.subset_seeds = tuple(config["subset_seeds"])
        if self.schema["feature_count"] != 135 or len(self.feature_names) != 135:
            raise ValueError("the shared experiment requires the 135-feature v1.1 dataset")
        if len(set(self.feature_names)) != 135:
            raise ValueError("feature names must be unique")
        if self.class_names != ("Sitting", "Standing", "Lying_Down", "Walking"):
            raise ValueError("class order must be Sitting, Standing, Lying_Down, Walking")
        if self.subset_seeds != (17, 42, 73):
            raise ValueError("frozen manifests map replicate 0/1/2 to seeds 17/42/73")
        self._roles = self._read_csv(self.split_dir / "outer_participant_roles.csv")
        self._subsets = self._read_csv(self.split_dir / "nested_participant_subsets.csv")
        self._matched_cache: dict[int, list[dict[str, str]]] = {}

    @staticmethod
    def _read_csv(path: Path) -> list[dict[str, str]]:
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))

    def outer_test_participants(self, outer_fold: int) -> tuple[str, ...]:
        participants = tuple(
            row["participant_id"] for row in self._roles
            if int(row["outer_fold"]) == outer_fold and row["role"] == "test"
        )
        if len(participants) != 2 or len(set(participants)) != 2:
            raise ValueError(f"outer fold {outer_fold} must have two distinct test participants")
        return participants

    def subset_rows(self, outer_fold: int, replicate_id: int, subset_size: int) -> list[dict[str, str]]:
        if replicate_id not in range(len(self.subset_seeds)):
            raise ValueError("replicate_id must be 0, 1 or 2")
        rows = [row for row in self._subsets
                if int(row["outer_fold"]) == outer_fold
                and int(row["replicate_id"]) == replicate_id
                and int(row["subset_size"]) == subset_size]
        rows.sort(key=lambda row: int(row["subset_position"]))
        participants = {row["participant_id"] for row in rows}
        if len(rows) != subset_size or len(participants) != subset_size:
            raise ValueError(f"missing/duplicate subset outer={outer_fold} replicate={replicate_id} size={subset_size}")
        if [int(row["subset_position"]) for row in rows] != list(range(1, subset_size + 1)):
            raise ValueError("subset positions must run from 1 to subset_size")
        if any(int(row["replicate_seed"]) != self.subset_seeds[replicate_id] for row in rows):
            raise ValueError("subset manifest seed does not match the shared configuration")
        if any(int(row["inner_fold"]) not in (0, 1, 2) for row in rows):
            raise ValueError("inner fold assignments must be 0, 1 or 2")
        if participants & set(self.outer_test_participants(outer_fold)):
            raise RuntimeError("participant leakage between subset and outer test")
        return rows

    def subset_participants(self, outer_fold: int, replicate_id: int, subset_size: int) -> tuple[str, ...]:
        return tuple(row["participant_id"] for row in self.subset_rows(outer_fold, replicate_id, subset_size))

    def inner_participants(self, outer_fold: int, replicate_id: int, subset_size: int,
                           validation_inner_fold: int) -> tuple[tuple[str, ...], tuple[str, ...]]:
        if validation_inner_fold not in (0, 1, 2):
            raise ValueError("validation_inner_fold must be 0, 1 or 2")
        rows = self.subset_rows(outer_fold, replicate_id, subset_size)
        validation = tuple(row["participant_id"] for row in rows
                           if int(row["inner_fold"]) == validation_inner_fold)
        training = tuple(row["participant_id"] for row in rows
                         if int(row["inner_fold"]) != validation_inner_fold)
        if len(validation) != subset_size // 3 or len(training) != 2 * subset_size // 3:
            raise ValueError("the frozen inner folds must have equal participant counts")
        return training, validation

    def _load_shard_rows(self, participant: str, domain: str,
                         row_indices: np.ndarray | None = None) -> WindowTable:
        path = self.dataset_dir / "shards" / f"{participant}__{domain}.npz"
        with np.load(path, allow_pickle=False) as shard:
            selected = (np.flatnonzero(shard["primary_mask"]) if row_indices is None
                        else np.asarray(row_indices, dtype=np.int64))
            if selected.ndim != 1 or len(np.unique(selected)) != len(selected):
                raise ValueError(f"duplicate or invalid row indices in {path.name}")
            if len(selected) and (selected.min() < 0 or selected.max() >= len(shard["X"])):
                raise IndexError(f"row index outside {path.name}")
            if not np.all(shard["primary_mask"][selected]):
                raise ValueError(f"non-primary activity selected from {path.name}")
            n = len(selected)
            table = WindowTable(
                X=shard["X"][selected].astype(np.float32, copy=True),
                y=shard["y_five"][selected].astype(np.int8, copy=True),
                participant_id=np.full(n, participant),
                domain=np.full(n, domain),
                window_index=shard["window_index"][selected].astype(np.int64, copy=True),
                window_start_ns=shard["window_start_ns"][selected].astype(np.int64, copy=True),
            )
        if table.X.shape != (n, len(self.feature_names)) or np.any(~np.isfinite(table.X)):
            raise ValueError(f"invalid features in {path.name}")
        if np.any((table.y < 0) | (table.y >= 4)):
            raise ValueError(f"primary labels must be class IDs 0..3 in {path.name}")
        if len(np.unique(table.window_index)) != n:
            raise ValueError(f"duplicate original window IDs in {path.name}")
        return table

    def outer_test(self, outer_fold: int) -> WindowTable:
        return _concatenate([self._load_shard_rows(participant, "FL")
                             for participant in self.outer_test_participants(outer_fold)])

    def _matched_rows(self, replicate_id: int) -> list[dict[str, str]]:
        if replicate_id not in self._matched_cache:
            self._matched_cache[replicate_id] = self._read_csv(
                self.split_dir / f"matched_training_windows_replicate_{replicate_id}.csv.gz")
        return self._matched_cache[replicate_id]

    def selected_training_data(self, outer_fold: int, replicate_id: int, subset_size: int,
                               domain: str, analysis: str = "main",
                               participants: tuple[str, ...] | None = None) -> WindowTable:
        if domain not in ("FL", "Formal_Lab"):
            raise ValueError("domain must be FL or Formal_Lab")
        if analysis not in ("main", "matched") or (analysis == "matched" and subset_size != 18):
            raise ValueError("analysis must be main, or matched with subset_size=18")
        allowed = self.subset_participants(outer_fold, replicate_id, subset_size)
        # Preserve frozen order. Converting this tuple to a set changes training
        # row order across processes and may change fitted stochastic models.
        selected = allowed if participants is None else tuple(participants)
        if not selected or len(set(selected)) != len(selected):
            raise ValueError("requested participants must be non-empty and distinct")
        if not set(selected).issubset(allowed):
            raise ValueError("requested training participants are outside the frozen subset")
        if analysis == "main":
            tables = [self._load_shard_rows(participant, domain) for participant in selected]
        else:
            grouped: dict[str, list[int]] = {participant: [] for participant in selected}
            for row in self._matched_rows(replicate_id):
                if row["participant_id"] in grouped and row["domain"] == domain:
                    grouped[row["participant_id"]].append(int(row["row_index"]))
            tables = [self._load_shard_rows(participant, domain, np.asarray(grouped[participant]))
                      for participant in selected]
        if any(not len(table) for table in tables):
            raise ValueError("a selected participant has no eligible training/validation windows")
        return _concatenate(tables)

    def inner_split(self, outer_fold: int, replicate_id: int, subset_size: int, domain: str,
                    validation_inner_fold: int, analysis: str = "main") -> tuple[WindowTable, WindowTable]:
        training, validation = self.inner_participants(
            outer_fold, replicate_id, subset_size, validation_inner_fold)
        return (self.selected_training_data(outer_fold, replicate_id, subset_size, domain, analysis, training),
                self.selected_training_data(outer_fold, replicate_id, subset_size, domain, analysis, validation))


def load_split(data: ExperimentData, condition: Condition, inner_fold: int | None = None,
               include_outer_test: bool = False) -> LoadedSplit:
    """Load one training split; test windows require an explicit final-fit call."""
    if inner_fold not in (None, 0, 1, 2):
        raise ValueError("inner_fold must be None, 0, 1 or 2")
    if include_outer_test and inner_fold is not None:
        raise ValueError("outer test data cannot be loaded during inner-CV tuning")
    if condition.subset_seed not in data.subset_seeds:
        raise ValueError("subset_seed must appear in the frozen configuration")
    replicate = data.subset_seeds.index(condition.subset_seed)
    fold, size, domain = condition.outer_fold, condition.subset_size, condition.training_domain
    selected = data.subset_participants(fold, replicate, size)
    if inner_fold is None:
        train = data.selected_training_data(fold, replicate, size, domain, condition.analysis)
        validation = None
        training_participants, validation_participants = selected, ()
    else:
        training_participants, validation_participants = data.inner_participants(fold, replicate, size, inner_fold)
        train, validation = data.inner_split(fold, replicate, size, domain, inner_fold, condition.analysis)
    test = data.outer_test(fold) if include_outer_test else None
    context = {
        "outer_fold": fold, "subset_size": size, "subset_seed": condition.subset_seed,
        "replicate_id": replicate, "training_domain": domain, "analysis": condition.analysis,
        "outer_test_participants": list(data.outer_test_participants(fold)),
        "selected_training_participants": list(selected), "inner_fold": inner_fold,
        "inner_training_participants": list(training_participants),
        "inner_validation_participants": list(validation_participants),
        "outer_test_loaded": include_outer_test,
        "feature_count": len(data.feature_names), "class_order": list(data.class_names),
    }
    return LoadedSplit(train, validation, test, context)
