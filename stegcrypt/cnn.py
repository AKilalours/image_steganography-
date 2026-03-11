"""
stegcrypt.cnn
~~~~~~~~~~~~~
Inference wrapper for the trained ResNet-18 stego binary classifier.

Usage
-----
    from stegcrypt.cnn import StegoCNN
    from PIL import Image

    model = StegoCNN(weights_path=Path("stego_cnn.pth"))
    prob  = model.detect_stego(Image.open("suspect.png"))
    print(f"P(stego) = {prob:.4f}")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from PIL import Image

logger = logging.getLogger(__name__)


def _require_torch() -> None:
    try:
        import torch        # noqa: F401
        import torchvision  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch / torchvision not installed.  "
            "Run: pip install -e '.[ml]'"
        ) from exc


class StegoCNN:
    """ResNet-18 binary stego classifier (cover=0, stego=1).

    Parameters
    ----------
    weights_path:
        Path to a ``.pth`` state-dict file produced by ``stegcrypt.ml.train_cnn``.
        **Must be provided** — raises ``FileNotFoundError`` if the path does
        not exist, rather than silently returning random scores.
    device:
        PyTorch device string.  Defaults to ``"cuda"`` when available.
    """

    def __init__(
        self,
        weights_path: Path,
        device: Optional[str] = None,
    ) -> None:
        _require_torch()

        import torch
        import torch.nn as nn
        import torchvision.transforms as T
        from torchvision.models import resnet18

        if not Path(weights_path).exists():
            raise FileNotFoundError(
                f"Weights file not found: {weights_path}\n"
                "Train a model first:\n"
                "  python -m stegcrypt.ml.train_cnn --covers ./covers --out stego_cnn.pth"
            )

        self._torch  = torch
        self.device  = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model  = resnet18(weights=None)
        self._model.fc = nn.Linear(self._model.fc.in_features, 1)
        self._model.to(self.device)

        state = torch.load(weights_path, map_location=self.device, weights_only=True)
        self._model.load_state_dict(state)
        self._model.eval()

        self._transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def detect_stego(self, img: Image.Image) -> float:
        """Return P(stego) in [0, 1] for *img*."""
        x = self._transform(img.convert("RGB")).unsqueeze(0).to(self.device)
        with self._torch.no_grad():
            prob = self._torch.sigmoid(self._model(x)).item()
        return float(prob)

    def score_cover(self, img: Image.Image) -> float:
        """Return P(cover) = 1 – P(stego)."""
        return 1.0 - self.detect_stego(img)
