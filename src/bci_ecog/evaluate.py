from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from bci_ecog.config import load_config
from bci_ecog.data.competition_iii import decode_labels, load_dataset
from bci_ecog.data.transforms import build_transform_pipeline
from bci_ecog.models.factory import build_model
from bci_ecog.training.engine import create_inference_loader, create_loader, evaluate, predict
from bci_ecog.utils.io import save_json
from bci_ecog.utils.plots import plot_confusion_matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained BCI Dataset I model.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to best_model.pt.")
    parser.add_argument(
        "--split",
        choices=("val", "test", "competition"),
        default="test",
        help="Which split to evaluate.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = checkpoint["config"]
    config_path = args.checkpoint.parent / "config.yaml"
    if config_path.exists():
        config = load_config(config_path)

    dataset = load_dataset(
        dataset_dir=config["data"]["dataset_dir"],
        train_file=config["data"]["train_file"],
        competition_test_file=config["data"]["competition_test_file"],
    )

    preprocessor = build_transform_pipeline(config)
    preprocessor.load_state_dict(checkpoint["preprocessor_state"])
    model = build_model(
        model_config=config["model"],
        input_shape=tuple(checkpoint["input_shape"]),
        num_classes=len(config["data"]["class_names"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    device = torch.device("cuda" if config["device"] == "cuda" and torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = torch.nn.CrossEntropyLoss()

    split_indices = {name: torch.tensor(indices).numpy() for name, indices in checkpoint["split_indices"].items()}

    if args.split in {"val", "test"}:
        features = preprocessor.transform(dataset["train_signals"][split_indices[args.split]])
        labels = dataset["train_labels"][split_indices[args.split]]
        loader = create_loader(
            features,
            labels,
            batch_size=config["training"]["batch_size"],
            num_workers=config["training"]["num_workers"],
            shuffle=False,
        )
        metrics = evaluate(model, loader, criterion, device)
        metrics_path = args.checkpoint.parent / f"{args.split}_metrics.json"
        save_json(
            {key: value for key, value in metrics.items() if key != "confusion_matrix"},
            metrics_path,
        )
        plot_confusion_matrix(
            metrics["confusion_matrix"],
            class_names=config["data"]["class_names"],
            path=args.checkpoint.parent / f"{args.split}_confusion_matrix.png",
            title=f"{args.split.title()} Confusion Matrix",
        )
        print(f"Saved {args.split} metrics to {metrics_path}")
        return

    competition_features = preprocessor.transform(dataset["competition_test_signals"])
    loader = create_inference_loader(
        competition_features,
        batch_size=config["training"]["batch_size"],
        num_workers=config["training"]["num_workers"],
    )
    predictions = predict(model, loader, device)
    prediction_frame = pd.DataFrame(
        {
            "trial_index": range(len(predictions["predictions"])),
            "predicted_label_encoded": predictions["predictions"],
            "predicted_label_original": decode_labels(predictions["predictions"]),
            "prob_class_0": predictions["probabilities"][:, 0],
            "prob_class_1": predictions["probabilities"][:, 1],
        }
    )
    prediction_path = args.checkpoint.parent / "competition_test_predictions_eval.csv"
    prediction_frame.to_csv(prediction_path, index=False)
    print(f"Saved competition predictions to {prediction_path}")


if __name__ == "__main__":
    main()
