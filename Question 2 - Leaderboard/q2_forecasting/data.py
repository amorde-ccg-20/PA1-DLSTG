from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


HORIZON = 168


def _require_contiguous(frame: pd.DataFrame, name: str) -> None:
    index = frame["time_idx"].to_numpy()
    if frame["time_idx"].duplicated().any() or not np.array_equal(index, np.arange(index[0], index[-1] + 1)):
        raise ValueError(f"{name}: time_idx must be unique, sorted, and contiguous")


def load_frames(data_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    root = Path(data_dir)
    train = pd.read_csv(root / "student_train.csv")
    test = pd.read_csv(root / "student_test.csv")
    external = pd.read_csv(root / "optional_external_data.csv")
    for name, frame in (("train", train), ("test", test), ("external", external)):
        _require_contiguous(frame, name)
        if frame.isna().any().any() and name != "test":
            raise ValueError(f"{name}: unexpected missing values")
    if list(train.columns) != ["time_idx", "value"] or list(test.columns) != ["time_idx", "value"]:
        raise ValueError("target files must contain only time_idx,value")
    if len(test) != HORIZON or test["value"].notna().any():
        raise ValueError("student_test.csv must contain 168 blank target values")
    if train.time_idx.iloc[-1] + 1 != test.time_idx.iloc[0] or external.time_idx.iloc[-1] != test.time_idx.iloc[-1]:
        raise ValueError("train, test, and external index ranges do not align")
    if (train.value < 0).any():
        raise ValueError("Task 2 target is expected to be non-negative")
    return train, test, external


@dataclass
class Standardizer:
    mean: np.ndarray
    scale: np.ndarray
    log1p: bool = False

    @classmethod
    def fit(cls, values: np.ndarray, log1p: bool = False) -> "Standardizer":
        values = np.asarray(values, dtype=np.float32)
        if log1p:
            if (values < 0).any():
                raise ValueError("log1p cannot transform negative targets")
            values = np.log1p(values)
        mean = values.mean(axis=0, keepdims=True)
        scale = values.std(axis=0, keepdims=True)
        return cls(mean, np.maximum(scale, 1e-6), log1p)

    def transform(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        if self.log1p:
            values = np.log1p(values)
        return (values - self.mean) / self.scale

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32) * self.scale + self.mean
        return np.expm1(values) if self.log1p else values


class ForecastWindowDataset(Dataset):
    def __init__(self, inputs: np.ndarray, targets: np.ndarray, future_external: np.ndarray | None):
        self.inputs = torch.as_tensor(inputs, dtype=torch.float32)
        self.targets = torch.as_tensor(targets, dtype=torch.float32)
        self.future_external = None if future_external is None else torch.as_tensor(future_external, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = {"x": self.inputs[index], "y": self.targets[index]}
        if self.future_external is not None:
            item["future_external"] = self.future_external[index]
        return item


@dataclass
class PreparedData:
    train: pd.DataFrame
    external: pd.DataFrame
    target_scaler: Standardizer
    external_scaler: Standardizer | None
    seq_len: int
    pred_len: int
    use_external: bool

    @property
    def train_length(self) -> int:
        return len(self.train)

    @property
    def external_dim(self) -> int:
        return 0 if not self.use_external else len(self.external.columns) - 1

    def _arrays(self) -> tuple[np.ndarray, np.ndarray | None]:
        y = self.target_scaler.transform(self.train[["value"]].to_numpy())
        if not self.use_external:
            return y, None
        features = self.external.drop(columns="time_idx").to_numpy(dtype=np.float32)
        assert self.external_scaler is not None
        return y, self.external_scaler.transform(features)

    def windows(self, end: int, stride: int, origins: list[int] | None = None) -> ForecastWindowDataset:
        y, ext = self._arrays()
        if end > self.train_length:
            raise ValueError("training windows cannot include unknown target values")
        if origins is None:
            origins = list(range(self.seq_len, end - self.pred_len + 1, stride))
        starts = []
        targets = []
        future = []
        for origin in origins:
            if origin < self.seq_len or origin + self.pred_len > end:
                raise ValueError(f"invalid origin {origin} for end={end}")
            history = y[origin - self.seq_len:origin]
            if ext is not None:
                history = np.concatenate([history, ext[origin - self.seq_len:origin]], axis=-1)
                future.append(ext[origin:origin + self.pred_len])
            starts.append(history)
            targets.append(y[origin:origin + self.pred_len])
        return ForecastWindowDataset(np.stack(starts), np.stack(targets), None if ext is None else np.stack(future))

    def inference_window(self) -> tuple[torch.Tensor, torch.Tensor | None]:
        y, ext = self._arrays()
        if len(y) != self.train_length:
            raise RuntimeError("unexpected target length")
        x = y[-self.seq_len:]
        future = None
        if ext is not None:
            x = np.concatenate([x, ext[self.train_length - self.seq_len:self.train_length]], axis=-1)
            future = ext[self.train_length:self.train_length + self.pred_len]
        return torch.tensor(x[None], dtype=torch.float32), None if future is None else torch.tensor(future[None], dtype=torch.float32)


def prepare_data(data_dir: str | Path, seq_len: int, pred_len: int, fit_end: int, use_external: bool,
                 target_transform: str) -> PreparedData:
    if pred_len != HORIZON:
        raise ValueError("Task 2 requires pred_len=168")
    train, _, external = load_frames(data_dir)
    if not seq_len < fit_end <= len(train):
        raise ValueError("fit_end must be inside observed training history")
    target_scaler = Standardizer.fit(train.value.iloc[:fit_end].to_numpy()[:, None], log1p=target_transform == "log1p")
    external_scaler = None
    if use_external:
        features = external.drop(columns="time_idx").to_numpy(dtype=np.float32)
        external_scaler = Standardizer.fit(features[:fit_end])
    return PreparedData(train, external, target_scaler, external_scaler, seq_len, pred_len, use_external)
