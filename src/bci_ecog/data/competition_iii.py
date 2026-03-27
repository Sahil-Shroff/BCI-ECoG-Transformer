from __future__ import annotations

import gzip
import io
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat

LABEL_TO_INDEX = {-1: 0, 1: 1}
INDEX_TO_LABEL = {value: key for key, value in LABEL_TO_INDEX.items()}


def _load_mat_file(path: Path) -> dict[str, Any]:
    if path.suffix == ".gz":
        with gzip.open(path, "rb") as handle:
            contents = handle.read()
        return loadmat(io.BytesIO(contents))
    return loadmat(path)


def _resolve_data_file(dataset_dir: Path, requested_name: str, fallbacks: tuple[str, ...]) -> Path:
    candidates = [requested_name, *fallbacks]
    for candidate in candidates:
        resolved = dataset_dir / candidate
        if resolved.exists():
            return resolved
    raise FileNotFoundError(
        f"Could not find any of {candidates} in dataset directory '{dataset_dir}'."
    )


def encode_labels(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels).reshape(-1).astype(np.int64)
    try:
        return np.vectorize(LABEL_TO_INDEX.__getitem__)(labels).astype(np.int64)
    except KeyError as exc:
        raise ValueError(f"Unexpected label value in dataset: {exc}") from exc


def decode_labels(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels).reshape(-1).astype(np.int64)
    return np.vectorize(INDEX_TO_LABEL.__getitem__)(labels).astype(np.int64)


def load_dataset(dataset_dir: str | Path, train_file: str, competition_test_file: str) -> dict[str, Any]:
    dataset_dir = Path(dataset_dir)
    train_path = _resolve_data_file(
        dataset_dir,
        train_file,
        ("Competition_train.mat.gz", "Competition_train.mat"),
    )
    test_path = _resolve_data_file(
        dataset_dir,
        competition_test_file,
        ("Competition_test.mat.gz", "Competition_test.mat"),
    )

    train_mat = _load_mat_file(train_path)
    competition_test_mat = _load_mat_file(test_path)

    train_signals = np.asarray(train_mat["X"], dtype=np.float32)
    train_labels_raw = np.asarray(train_mat["Y"]).reshape(-1).astype(np.int64)
    competition_test_signals = np.asarray(competition_test_mat["X"], dtype=np.float32)

    payload: dict[str, Any] = {
        "train_signals": train_signals,
        "train_labels_raw": train_labels_raw,
        "train_labels": encode_labels(train_labels_raw),
        "competition_test_signals": competition_test_signals,
        "train_path": train_path,
        "competition_test_path": test_path,
        "label_to_index": LABEL_TO_INDEX.copy(),
        "index_to_label": INDEX_TO_LABEL.copy(),
    }

    if "Y" in competition_test_mat:
        competition_labels_raw = np.asarray(competition_test_mat["Y"]).reshape(-1).astype(np.int64)
        payload["competition_test_labels_raw"] = competition_labels_raw
        payload["competition_test_labels"] = encode_labels(competition_labels_raw)

    return payload

