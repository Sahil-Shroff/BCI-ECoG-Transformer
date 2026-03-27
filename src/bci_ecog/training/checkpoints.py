from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


class CheckpointSaver:
    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        filename: str,
        payload: dict[str, Any],
    ) -> Path:
        checkpoint_path = self.output_dir / filename
        torch.save(payload, checkpoint_path)
        return checkpoint_path
