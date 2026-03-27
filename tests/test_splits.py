import numpy as np

from bci_ecog.data.splits import create_stratified_splits


def test_create_stratified_splits_is_disjoint_and_complete():
    labels = np.array([0] * 50 + [1] * 50)
    splits = create_stratified_splits(
        labels=labels,
        split_config={"train": 0.7, "val": 0.15, "test": 0.15},
        seed=42,
    )
    combined = np.concatenate(list(splits.values()))
    assert len(np.unique(combined)) == len(labels)
    assert set(combined.tolist()) == set(range(len(labels)))
    assert set(splits["train"]).isdisjoint(splits["val"])
    assert set(splits["train"]).isdisjoint(splits["test"])
    assert set(splits["val"]).isdisjoint(splits["test"])

