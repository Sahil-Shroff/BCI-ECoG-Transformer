from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from bci_ecog.config import load_config, save_config
from bci_ecog.data.competition_iii import decode_labels, load_dataset
from bci_ecog.data.datasets import build_dataloader
from bci_ecog.data.preprocessing import apply_normalization, apply_windowing, build_preprocessing_pipeline
from bci_ecog.data.splits import create_stratified_splits, summarize_splits
from bci_ecog.models.factory import build_model
from bci_ecog.training.checkpoints import CheckpointSaver
from bci_ecog.training.engine import create_inference_loader, evaluate, fit, predict
from bci_ecog.training.logger import TrainingLogger
from bci_ecog.utils.io import make_run_dir, save_json
from bci_ecog.utils.plots import plot_confusion_matrix, plot_training_history
from bci_ecog.utils.random import set_deterministic_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a BCI Competition III Dataset I baseline model.")
    parser.add_argument("--config", type=Path, required=True, help="Path to YAML experiment config.")
    return parser.parse_args()


def select_device(device_name: str) -> torch.device:
    if device_name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _prepare_split_features(
    signals: np.ndarray,
    labels: np.ndarray,
    preprocessor,
    window_config: dict,
    fit_preprocessor: bool,
) -> tuple[np.ndarray, np.ndarray]:
    windowed = apply_windowing(signals=signals, labels=labels, window_config=window_config)
    features = apply_normalization(windowed.signals, pipeline=preprocessor, fit=fit_preprocessor)
    return features, windowed.labels


def run_training(config_path: str | Path) -> Path:
    config = load_config(config_path)
    set_deterministic_seed(config["seed"])

    run_dir = make_run_dir(config["experiment"]["output_root"], config["experiment"]["name"])
    save_config(config, run_dir / "config.yaml")
    checkpoint_saver = CheckpointSaver(run_dir)
    training_logger = TrainingLogger()

    dataset = load_dataset(
        dataset_dir=config["data"]["dataset_dir"],
        train_file=config["data"]["train_file"],
        competition_test_file=config["data"]["competition_test_file"],
    )
    split_indices = create_stratified_splits(
        labels=dataset["train_labels"],
        split_config=config["data"]["split"],
        seed=config["seed"],
    )
    save_json(summarize_splits(dataset["train_labels_raw"], split_indices), run_dir / "split_summary.json")
    save_json(
        {name: indices.tolist() for name, indices in split_indices.items()},
        run_dir / "split_indices.json",
    )

    preprocessor = build_preprocessing_pipeline(config)
    window_config = config.get("preprocessing", {}).get("windowing", {"enabled": False})

    train_features, train_labels = _prepare_split_features(
        signals=dataset["train_signals"][split_indices["train"]],
        labels=dataset["train_labels"][split_indices["train"]],
        preprocessor=preprocessor,
        window_config=window_config,
        fit_preprocessor=True,
    )
    val_features, val_labels = _prepare_split_features(
        signals=dataset["train_signals"][split_indices["val"]],
        labels=dataset["train_labels"][split_indices["val"]],
        preprocessor=preprocessor,
        window_config=window_config,
        fit_preprocessor=False,
    )
    test_features, test_labels = _prepare_split_features(
        signals=dataset["train_signals"][split_indices["test"]],
        labels=dataset["train_labels"][split_indices["test"]],
        preprocessor=preprocessor,
        window_config=window_config,
        fit_preprocessor=False,
    )
    competition_windowed = apply_windowing(
        signals=dataset["competition_test_signals"],
        labels=None,
        window_config=window_config,
    )
    competition_features = apply_normalization(
        competition_windowed.signals,
        pipeline=preprocessor,
        fit=False,
    )

    training_config = config["training"]
    input_mode = config["representation"]["name"]
    train_loader = build_dataloader(
        mode=input_mode,
        features=train_features,
        labels=train_labels,
        batch_size=training_config["batch_size"],
        num_workers=training_config["num_workers"],
        shuffle=True,
    )
    val_loader = build_dataloader(
        mode=input_mode,
        features=val_features,
        labels=val_labels,
        batch_size=training_config["batch_size"],
        num_workers=training_config["num_workers"],
        shuffle=False,
    )
    test_loader = build_dataloader(
        mode=input_mode,
        features=test_features,
        labels=test_labels,
        batch_size=training_config["batch_size"],
        num_workers=training_config["num_workers"],
        shuffle=False,
    )
    competition_loader = create_inference_loader(
        competition_features,
        batch_size=training_config["batch_size"],
        num_workers=training_config["num_workers"],
    )

    model = build_model(
        model_config=config["model"],
        input_shape=tuple(train_features.shape[1:]),
        num_classes=len(config["data"]["class_names"]),
    )

    device = select_device(config["device"])
    model = model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config["learning_rate"],
        weight_decay=training_config["weight_decay"],
    )

    history, best_epoch_metrics, best_state = fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        epochs=training_config["epochs"],
        patience=training_config["early_stopping_patience"],
        monitor=training_config["monitor"],
        monitor_mode=training_config["monitor_mode"],
    )

    for row in history.to_dict(orient="records"):
        training_logger.rows.append(row)
    history = training_logger.save_history(run_dir / "history.csv")
    plot_training_history(history, run_dir / "training_curves.png")

    model.load_state_dict(best_state)
    val_metrics = evaluate(model, val_loader, criterion, device)
    test_metrics = evaluate(model, test_loader, criterion, device)
    competition_predictions = predict(model, competition_loader, device)

    checkpoint_payload = {
        "config": config,
        "model_state_dict": best_state,
        "preprocessor_state": preprocessor.state_dict(),
        "split_indices": {name: indices.tolist() for name, indices in split_indices.items()},
        "input_shape": list(train_features.shape[1:]),
        "best_epoch_metrics": best_epoch_metrics,
    }
    checkpoint_saver.save(
        "best_model.pt",
        checkpoint_payload,
    )
    checkpoint_saver.save(
        "last_model.pt",
        {
            "config": config,
            "model_state_dict": model.state_dict(),
            "preprocessor_state": preprocessor.state_dict(),
            "split_indices": {name: indices.tolist() for name, indices in split_indices.items()},
            "input_shape": list(train_features.shape[1:]),
            "best_epoch_metrics": best_epoch_metrics,
        },
    )

    metrics_frame = pd.DataFrame(
        [
            {"split": "val", **{key: value for key, value in val_metrics.items() if key != "confusion_matrix"}},
            {"split": "test", **{key: value for key, value in test_metrics.items() if key != "confusion_matrix"}},
        ]
    )
    metrics_frame.to_csv(run_dir / "metrics.csv", index=False)

    plot_confusion_matrix(
        val_metrics["confusion_matrix"],
        class_names=config["data"]["class_names"],
        path=run_dir / "val_confusion_matrix.png",
        title="Validation Confusion Matrix",
    )
    plot_confusion_matrix(
        test_metrics["confusion_matrix"],
        class_names=config["data"]["class_names"],
        path=run_dir / "test_confusion_matrix.png",
        title="Test Confusion Matrix",
    )

    pd.DataFrame(
        {
            "trial_index": range(len(split_indices["test"])),
            "dataset_index": split_indices["test"],
            "true_label_encoded": test_labels,
            "true_label_original": decode_labels(test_labels),
        }
    ).to_csv(run_dir / "test_targets.csv", index=False)

    competition_prediction_frame = pd.DataFrame(
        {
            "trial_index": range(len(competition_predictions["predictions"])),
            "predicted_label_encoded": competition_predictions["predictions"],
            "predicted_label_original": decode_labels(competition_predictions["predictions"]),
            "prob_class_0": competition_predictions["probabilities"][:, 0],
            "prob_class_1": competition_predictions["probabilities"][:, 1],
        }
    )
    competition_prediction_frame.to_csv(run_dir / "competition_test_predictions.csv", index=False)

    save_json(
        {
            "run_dir": str(run_dir),
            "device": str(device),
            "dataset": {
                "train_path": str(dataset["train_path"]),
                "competition_test_path": str(dataset["competition_test_path"]),
            },
            "input_shape": list(train_features.shape[1:]),
            "representation": config["representation"]["name"],
            "best_epoch_metrics": {
                "epoch": best_epoch_metrics["epoch"],
                "train": {
                    key: value
                    for key, value in best_epoch_metrics["train"].items()
                    if key != "confusion_matrix"
                },
                "val": {
                    key: value
                    for key, value in best_epoch_metrics["val"].items()
                    if key != "confusion_matrix"
                },
            },
            "final_metrics": {
                "val": {key: value for key, value in val_metrics.items() if key != "confusion_matrix"},
                "test": {key: value for key, value in test_metrics.items() if key != "confusion_matrix"},
            },
        },
        run_dir / "summary.json",
    )
    training_logger.save_metrics(
        {
            "val": {key: value for key, value in val_metrics.items() if key != "confusion_matrix"},
            "test": {key: value for key, value in test_metrics.items() if key != "confusion_matrix"},
        },
        run_dir / "metrics.json",
    )

    print(f"Artifacts saved to {run_dir}")
    return run_dir


def main() -> None:
    args = parse_args()
    run_training(args.config)


if __name__ == "__main__":
    main()
