"""Baseline multitask models (Morgan fingerprints + RandomForest)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor

from radioprotect_sm.data.canonicalize import morgan_fingerprint
from radioprotect_sm.utils import set_global_seed


@dataclass
class BaselineConfig:
    n_bits: int = 2048
    radius: int = 2
    n_estimators: int = 300
    max_depth: int | None = None
    min_samples_leaf: int = 2
    n_jobs: int = -1
    seed: int = 42


class MorganRFMultitask:
    """Independent RF per task via MultiOutputRegressor with NaN masking per-task.

    Missing labels are handled by training one RF per task on rows where that
    task label is finite. Ensemble members differ by random_state seed.
    """

    def __init__(self, tasks: list[str], config: BaselineConfig | None = None):
        self.tasks = list(tasks)
        self.config = config or BaselineConfig()
        self.models: dict[str, RandomForestRegressor] = {}

    def _featurize(self, smiles: Iterable[str]) -> np.ndarray:
        fps = []
        for smi in smiles:
            fp = morgan_fingerprint(smi, n_bits=self.config.n_bits, radius=self.config.radius)
            if fp is None:
                fp = np.zeros((self.config.n_bits,), dtype=np.float32)
            fps.append(fp)
        return np.vstack(fps)

    def fit(self, smiles: pd.Series | list[str], y: pd.DataFrame) -> "MorganRFMultitask":
        set_global_seed(self.config.seed)
        X = self._featurize(smiles)
        self.models = {}
        for task in self.tasks:
            if task not in y.columns:
                continue
            yt = y[task].to_numpy(dtype=float)
            mask = np.isfinite(yt)
            if mask.sum() < 5:
                continue
            model = RandomForestRegressor(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                min_samples_leaf=self.config.min_samples_leaf,
                n_jobs=self.config.n_jobs,
                random_state=self.config.seed,
            )
            model.fit(X[mask], yt[mask])
            self.models[task] = model
        return self

    def predict(self, smiles: pd.Series | list[str]) -> pd.DataFrame:
        X = self._featurize(smiles)
        data = {}
        for task in self.tasks:
            if task not in self.models:
                data[task] = np.full(len(X), np.nan)
            else:
                data[task] = self.models[task].predict(X)
        return pd.DataFrame(data)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"tasks": self.tasks, "config": self.config, "models": self.models}, path)

    @classmethod
    def load(cls, path: Path) -> "MorganRFMultitask":
        payload = joblib.load(path)
        obj = cls(tasks=payload["tasks"], config=payload["config"])
        obj.models = payload["models"]
        return obj
