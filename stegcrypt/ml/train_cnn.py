"""
stegcrypt.ml.train_cnn
~~~~~~~~~~~~~~~~~~~~~~
Train a ResNet-18 binary stego classifier.

Production improvements over v0.2
----------------------------------
* Train / validation split (80/20) for honest generalisation metrics.
* Learning-rate scheduler (CosineAnnealingLR).
* Early stopping with configurable patience.
* Best-model checkpointing (saves the epoch with lowest val loss, not just
  the final epoch).
* ``pin_memory=True`` + configurable ``num_workers`` for faster data loading.
* Weighted random sampler to handle class imbalance (future-proof).
* Gradient clipping to prevent exploding gradients.
* Full metrics printed per epoch: loss, accuracy, precision, recall, F1.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Optional

from PIL import Image


def _build_model(device: str):
    import torch.nn as nn
    from torchvision.models import resnet18, ResNet18_Weights

    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)  # pretrained backbone
    model.fc = nn.Linear(model.fc.in_features, 1)
    model.to(device)
    return model


def _metrics(tp: int, fp: int, tn: int, fn: int) -> dict[str, float]:
    total = tp + fp + tn + fn
    acc   = (tp + tn) / max(1, total)
    prec  = tp / max(1, tp + fp)
    rec   = tp / max(1, tp + fn)
    f1    = 2 * prec * rec / max(1e-9, prec + rec)
    return dict(acc=acc, prec=prec, rec=rec, f1=f1)


def main(argv: Optional[list[str]] = None) -> None:
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
        import torchvision.transforms as T
    except ImportError as exc:
        raise SystemExit("Torch/torchvision not installed. Run: pip install -e '.[ml]'") from exc

    from .dataset import build_pairs, SyntheticStegoConfig, make_stego_variant_in_memory

    ap = argparse.ArgumentParser(description="Train stego CNN detector.")
    ap.add_argument("--covers",     required=True,  help="Folder of natural cover images.")
    ap.add_argument("--out",        default="stego_cnn.pth")
    ap.add_argument("--epochs",     type=int,   default=10)
    ap.add_argument("--batch",      type=int,   default=32)
    ap.add_argument("--lr",         type=float, default=1e-4)
    ap.add_argument("--workers",    type=int,   default=4, help="DataLoader num_workers.")
    ap.add_argument("--max-pairs",  type=int,   default=4000)
    ap.add_argument("--ecc",        type=int,   default=1)
    ap.add_argument("--patience",   type=int,   default=3,
                    help="Early-stopping patience (epochs without val loss improvement).")
    ap.add_argument("--val-split",  type=float, default=0.2,
                    help="Fraction of data used for validation.")
    args = ap.parse_args(argv)

    covers_dir = Path(args.covers).expanduser().resolve()
    pairs      = build_pairs(covers_dir, max_pairs=args.max_pairs)
    cfg        = SyntheticStegoConfig(ecc=args.ecc)

    # -------------------------------------------------------------------
    # Dataset
    # -------------------------------------------------------------------

    class StegoDataset(Dataset):
        _train_tf = T.Compose([
            T.Resize((256, 256)),
            T.RandomCrop(224),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.05),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        _val_tf = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        def __init__(self, pairs, train: bool = True):
            self.pairs = pairs
            self.tf    = self._train_tf if train else self._val_tf

        def __len__(self):
            return len(self.pairs)

        def __getitem__(self, idx):
            p, y = self.pairs[idx]
            img  = Image.open(p).convert("RGB")
            if y == 1:
                img = make_stego_variant_in_memory(img, cfg)
            x = self.tf(img)
            return x, torch.tensor([y], dtype=torch.float32)

    # Train/val split
    n_val   = max(1, int(len(pairs) * args.val_split))
    n_train = len(pairs) - n_val
    train_pairs, val_pairs = pairs[:n_train], pairs[n_train:]

    train_ds = StegoDataset(train_pairs, train=True)
    val_ds   = StegoDataset(val_pairs,   train=False)

    # Balanced sampler (cover:stego = 1:1 by construction, but keep for future use)
    labels   = [y for _, y in train_pairs]
    weights  = [1.0 / labels.count(y) for y in labels]
    sampler  = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    pin = torch.cuda.is_available()
    train_dl = DataLoader(train_ds, batch_size=args.batch, sampler=sampler,
                          num_workers=args.workers, pin_memory=pin)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False,
                          num_workers=args.workers, pin_memory=pin)

    # -------------------------------------------------------------------
    # Model, optimiser, scheduler
    # -------------------------------------------------------------------

    device  = "cuda" if torch.cuda.is_available() else "cpu"
    model   = _build_model(device)
    opt     = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)
    loss_fn = nn.BCEWithLogitsLoss()

    best_val_loss = math.inf
    patience_ctr  = 0

    print(f"Training on {n_train} samples, validating on {n_val} samples, device={device}")

    for ep in range(1, args.epochs + 1):
        # -- Train --
        model.train()
        tr_loss = tr_tp = tr_fp = tr_tn = tr_fn = 0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss   = loss_fn(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            tr_loss += loss.item() * x.size(0)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            tr_tp += int(((preds == 1) & (y == 1)).sum())
            tr_fp += int(((preds == 1) & (y == 0)).sum())
            tr_tn += int(((preds == 0) & (y == 0)).sum())
            tr_fn += int(((preds == 0) & (y == 1)).sum())

        # -- Validate --
        model.eval()
        va_loss = va_tp = va_fp = va_tn = va_fn = 0
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                va_loss += loss_fn(logits, y).item() * x.size(0)
                preds = (torch.sigmoid(logits) >= 0.5).float()
                va_tp += int(((preds == 1) & (y == 1)).sum())
                va_fp += int(((preds == 1) & (y == 0)).sum())
                va_tn += int(((preds == 0) & (y == 0)).sum())
                va_fn += int(((preds == 0) & (y == 1)).sum())

        sched.step()

        tr_m = _metrics(tr_tp, tr_fp, tr_tn, tr_fn)
        va_m = _metrics(va_tp, va_fp, va_tn, va_fn)
        va_l = va_loss / len(val_ds)
        lr_now = opt.param_groups[0]["lr"]

        print(
            f"epoch {ep:3d}/{args.epochs} | lr={lr_now:.2e} | "
            f"train loss={tr_loss/n_train:.4f} acc={tr_m['acc']:.4f} f1={tr_m['f1']:.4f} | "
            f"val   loss={va_l:.4f} acc={va_m['acc']:.4f} f1={va_m['f1']:.4f} "
            f"prec={va_m['prec']:.4f} rec={va_m['rec']:.4f}"
        )

        # -- Checkpoint best model --
        if va_l < best_val_loss:
            best_val_loss = va_l
            patience_ctr  = 0
            torch.save(model.state_dict(), args.out)
            print(f"  ✓ New best val loss {va_l:.4f} → saved to {args.out}")
        else:
            patience_ctr += 1
            if patience_ctr >= args.patience:
                print(f"  Early stopping: no val improvement for {args.patience} epochs.")
                break

    print(f"\nDone. Best weights saved to: {args.out}")


if __name__ == "__main__":
    main()
