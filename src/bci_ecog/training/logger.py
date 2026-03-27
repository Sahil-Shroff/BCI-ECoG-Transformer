from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from bci_ecog.utils.io import save_json


@dataclass
class TrainingLogger:
    rows: list[dict[str, Any]] = field(default_factory=list)

    def log_epoch(
        self,
        epoch: int,
        train_metrics: dict[str, Any],
        val_metrics: dict[str, Any],
    ) -> dict[str, Any]:
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_balanced_accuracy": train_metrics["balanced_accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_balanced_accuracy": val_metrics["balanced_accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
        }
        self.rows.append(row)
        return row

    def history_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)

    def save_history(self, path: str | Path) -> pd.DataFrame:
        history = self.history_frame()
        history.to_csv(path, index=False)
        return history

    def save_metrics(self, metrics: dict[str, Any], path: str | Path) -> None:
        save_json(metrics, path)

