"""
stegcrypt.ml.eval_cnn
~~~~~~~~~~~~~~~~~~~~~
Evaluate a trained stego CNN on a held-out cover set.

Outputs
-------
* Accuracy, Precision, Recall, F1 at threshold=0.5
* ROC-AUC (requires scikit-learn if available, otherwise omitted)
* Confusion matrix
* Per-threshold P/R table (optional)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from PIL import Image


def main(argv: Optional[list[str]] = None) -> None:
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import Dataset, DataLoader
        import torchvision.transforms as T
        from torchvision.models import resnet18
    except ImportError as exc:
        raise SystemExit("Torch/torchvision not installed. Run: pip install -e '.[ml]'") from exc

    from .dataset import build_pairs, SyntheticStegoConfig, make_stego_variant_in_memory

    ap = argparse.ArgumentParser(description="Evaluate stego CNN.")
    ap.add_argument("--covers",    required=True)
    ap.add_argument("--weights",   required=True)
    ap.add_argument("--batch",     type=int, default=32)
    ap.add_argument("--max-pairs", type=int, default=2000)
    ap.add_argument("--workers",   type=int, default=4)
    ap.add_argument("--ecc",       type=int, default=1)
    ap.add_argument("--pr-table",  action="store_true",
                    help="Print precision/recall at multiple thresholds.")
    args = ap.parse_args(argv)

    pairs = build_pairs(Path(args.covers).expanduser().resolve(), max_pairs=args.max_pairs)
    cfg   = SyntheticStegoConfig(ecc=args.ecc)

    class StegoDataset(Dataset):
        _tf = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        def __init__(self, pairs): self.pairs = pairs
        def __len__(self): return len(self.pairs)
        def __getitem__(self, idx):
            p, y = self.pairs[idx]
            img  = Image.open(p).convert("RGB")
            if y == 1:
                img = make_stego_variant_in_memory(img, cfg)
            return self._tf(img), torch.tensor([y], dtype=torch.float32)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 1)
    model.to(device)

    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise SystemExit(f"Weights not found: {weights_path}")

    state = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    pin = torch.cuda.is_available()
    dl  = DataLoader(StegoDataset(pairs), batch_size=args.batch, shuffle=False,
                     num_workers=args.workers, pin_memory=pin)

    all_probs: list[float] = []
    all_labels: list[int]  = []

    with torch.no_grad():
        for x, y in dl:
            x = x.to(device)
            probs = torch.sigmoid(model(x)).cpu().squeeze(1).tolist()
            all_probs.extend(probs if isinstance(probs, list) else [probs])
            all_labels.extend(int(v) for v in y.squeeze(1).tolist())

    # --- Metrics at threshold 0.5 ---
    tp = fp = tn = fn = 0
    for p, y in zip(all_probs, all_labels):
        pred = int(p >= 0.5)
        if   pred == 1 and y == 1: tp += 1
        elif pred == 1 and y == 0: fp += 1
        elif pred == 0 and y == 0: tn += 1
        else:                       fn += 1

    total = tp + fp + tn + fn
    acc   = (tp + tn) / max(1, total)
    prec  = tp / max(1, tp + fp)
    rec   = tp / max(1, tp + fn)
    f1    = 2 * prec * rec / max(1e-9, prec + rec)

    print(f"\n{'='*50}")
    print(f"  Stego CNN Evaluation  (n={total})")
    print(f"{'='*50}")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  Confusion: TP={tp} FP={fp} TN={tn} FN={fn}")

    # --- ROC-AUC ---
    try:
        from sklearn.metrics import roc_auc_score  # type: ignore
        auc = roc_auc_score(all_labels, all_probs)
        print(f"  ROC-AUC:   {auc:.4f}")
    except ImportError:
        print("  ROC-AUC:   (install scikit-learn for AUC)")
    except Exception as exc:
        print(f"  ROC-AUC:   error ({exc})")

    # --- Per-threshold table ---
    if args.pr_table:
        print(f"\n  {'Threshold':>10} {'Prec':>8} {'Rec':>8} {'F1':>8}")
        for thr in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            ttp = tfp = ttn = tfn = 0
            for prob, y in zip(all_probs, all_labels):
                pred = int(prob >= thr)
                if   pred == 1 and y == 1: ttp += 1
                elif pred == 1 and y == 0: tfp += 1
                elif pred == 0 and y == 0: ttn += 1
                else:                       tfn += 1
            tp2 = ttp / max(1, ttp + tfp)
            rc2 = ttp / max(1, ttp + tfn)
            f2  = 2 * tp2 * rc2 / max(1e-9, tp2 + rc2)
            print(f"  {thr:>10.1f} {tp2:>8.4f} {rc2:>8.4f} {f2:>8.4f}")

    print()


if __name__ == "__main__":
    main()
