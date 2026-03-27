from __future__ import annotations

import argparse
import shutil
import ssl
import urllib.request
from pathlib import Path


DATASET_URLS = {
    "Competition_train.mat.gz": (
        "https://www.bbci.de/competition/download/competition_iii/tuebingen/Competition_train.mat.gz"
    ),
    "Competition_test.mat.gz": (
        "https://www.bbci.de/competition/download/competition_iii/tuebingen/Competition_test.mat.gz"
    ),
}


def download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(url) as response, destination.open("wb") as handle:
            shutil.copyfileobj(response, handle)
    except ssl.SSLCertVerificationError:
        context = ssl._create_unverified_context()
        with urllib.request.urlopen(url, context=context) as response, destination.open("wb") as handle:
            shutil.copyfileobj(response, handle)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download BCI Competition III Dataset I files.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw/bci_competition_iii_dataset_i"),
        help="Directory where the official .mat.gz files will be stored.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for filename, url in DATASET_URLS.items():
        destination = args.data_dir / filename
        print(f"Downloading {filename} -> {destination}")
        download_file(url, destination)


if __name__ == "__main__":
    main()
