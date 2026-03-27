from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.model_selection import train_test_split


def create_stratified_splits(
    labels: np.ndarray,
    split_config: dict[str, float],
    seed: int,
) -> dict[str, np.ndarray]:
    fractions = (
        split_config["train"],
        split_config["val"],
        split_config["test"],
    )
    if not np.isclose(sum(fractions), 1.0):
        raise ValueError(f"Split fractions must sum to 1.0, got {fractions}.")

    indices = np.arange(len(labels))
    train_indices, temp_indices = train_test_split(
        indices,
        train_size=split_config["train"],
        stratify=labels,
        random_state=seed,
    )

    temp_labels = labels[temp_indices]
    relative_val_fraction = split_config["val"] / (split_config["val"] + split_config["test"])
    val_indices, test_indices = train_test_split(
        temp_indices,
        train_size=relative_val_fraction,
        stratify=temp_labels,
        random_state=seed,
    )

    return {
        "train": np.sort(train_indices),
        "val": np.sort(val_indices),
        "test": np.sort(test_indices),
    }


def summarize_splits(labels: np.ndarray, split_indices: dict[str, np.ndarray]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for split_name, indices in split_indices.items():
        split_labels, counts = np.unique(labels[indices], return_counts=True)
        summary[split_name] = {
            "num_trials": int(len(indices)),
            "label_counts": {int(label): int(count) for label, count in zip(split_labels, counts)},
        }
    return summary

