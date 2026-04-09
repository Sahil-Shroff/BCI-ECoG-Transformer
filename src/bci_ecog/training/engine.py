from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from bci_ecog.training.logger import TrainingLogger
from bci_ecog.training.metrics import compute_metrics


def create_loader(
    features: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(features).float(),
        torch.from_numpy(labels).long(),
    )
    loader_kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True

    return DataLoader(dataset, **loader_kwargs)


def create_inference_loader(features: np.ndarray, batch_size: int, num_workers: int) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(features).float())
    loader_kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True

    return DataLoader(dataset, **loader_kwargs)


def _epoch_pass(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, Any]:
    training = optimizer is not None
    model.train(training)

    losses: list[torch.Tensor] = []
    predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []

    for batch_features, batch_targets in loader:
        batch_features = batch_features.to(device, non_blocking=True)
        batch_targets = batch_targets.to(device, non_blocking=True)

        with torch.set_grad_enabled(training):
            logits = model(batch_features)
            loss = criterion(logits, batch_targets)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        losses.append(loss.detach())
        predictions.append(torch.argmax(logits, dim=1).detach())
        targets.append(batch_targets.detach())

    y_true = torch.cat(targets).cpu().numpy()
    y_pred = torch.cat(predictions).cpu().numpy()
    num_classes = model.classifier[-1].out_features
    metrics = compute_metrics(y_true, y_pred, num_classes=num_classes)
    metrics["loss"] = float(torch.stack(losses).mean().cpu().item())
    return metrics


def fit(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
    epochs: int,
    patience: int,
    monitor: str,
    monitor_mode: str,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, torch.Tensor]]:
    best_value = float("-inf") if monitor_mode == "max" else float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch_metrics: dict[str, Any] | None = None
    wait = 0
    logger = TrainingLogger()

    for epoch in range(1, epochs + 1):
        train_metrics = _epoch_pass(model, train_loader, criterion, device, optimizer=optimizer)
        with torch.inference_mode():
            val_metrics = _epoch_pass(model, val_loader, criterion, device)

        row = logger.log_epoch(epoch, train_metrics, val_metrics)

        current_value = val_metrics[monitor]
        improved = current_value > best_value if monitor_mode == "max" else current_value < best_value
        if improved:
            best_value = current_value
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            best_epoch_metrics = {
                "epoch": epoch,
                "train": train_metrics,
                "val": val_metrics,
            }
            wait = 0
        else:
            wait += 1

        tqdm.write(
            "epoch={epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            "train_bal_acc={train_balanced_accuracy:.4f} val_bal_acc={val_balanced_accuracy:.4f}".format(**row)
        )

        if wait >= patience:
            tqdm.write(f"Early stopping triggered at epoch {epoch}.")
            break

    if best_state is None or best_epoch_metrics is None:
        raise RuntimeError("Training did not produce a checkpoint state.")

    history = logger.history_frame()
    return history, best_epoch_metrics, best_state


@torch.inference_mode()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
) -> dict[str, Any]:
    return _epoch_pass(model, loader, criterion, device)


@torch.inference_mode()
def predict(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, np.ndarray]:
    model.eval()
    predicted_labels: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []

    for (batch_features,) in loader:
        batch_features = batch_features.to(device, non_blocking=True)
        logits = model(batch_features)
        probs = torch.softmax(logits, dim=1)
        probabilities.append(probs.detach().cpu().numpy())
        predicted_labels.append(torch.argmax(probs, dim=1).detach().cpu().numpy())

    return {
        "predictions": np.concatenate(predicted_labels),
        "probabilities": np.concatenate(probabilities),
    }
