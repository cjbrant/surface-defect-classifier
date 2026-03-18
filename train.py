#!/usr/bin/env python3
"""
NEU Surface Defect Classifier (classification + Power BI exports)

End-to-end script to:
1) Build defect "chips" (cropped defects) from Pascal VOC XML under NEU-DET/{split}/annotations
2) Train a classifier on the chips (ResNet-18 by default)
3) Export rich CSVs for analysis / Power BI dashboards

Usage examples
--------------
# (re)generate chips and train
python train.py --rebuild-chips

# reuse existing chips and train
python train.py

Outputs (under ./results/)
---------------------------
- chips/                                   (cropped images by split/class)
- train_chips_index.csv                     (mapping chips -> source image, bbox, true_label)
- validation_chips_index.csv                (same for validation)
- NEUDET_Predictions.csv                    (long-form per-chip predictions)
- NEUDET_PredictionsWide.csv                (per-chip probabilities in wide format)
- NEUDET_ConfusionMatrix.csv                (raw counts)
- NEUDET_ConfusionMatrix_Normalized.csv     (row-normalized)
- NEUDET_ClassReport.csv                    (precision/recall/F1/support per class)
- models/neudet_resnet18_best.pt            (best-on-val-acc)
- models/neudet_resnet18_last.pt            (last epoch)
- class_to_idx.json                         (class → index mapping)
"""

import os
import json
import math
import shutil
import random
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torchvision as tv
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import classification_report, confusion_matrix

# -------------------------
# Paths & Hyperparameters
# -------------------------
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "NEU-DET"
OUT = ROOT / "results"
CHIPS = OUT / "chips"
(OUT / "models").mkdir(parents=True, exist_ok=True)
CHIPS.mkdir(parents=True, exist_ok=True)

IMG_SIZE = 224
BATCH = 32
EPOCHS = 12
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 2  # set to 0 on macOS if you get issues
PATIENCE = 4     # early stop on val loss plateau
MIN_BOX = 10     # skip tiny crops

# -------------------------
# Utils
# -------------------------

def seed_all(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_image(img_root: Path, filename: str, cls: str | None = None) -> Path:
    """
    Robustly resolve an image path when XML 'filename' may be a stem (no extension)
    and/or when class subfolders are used. Tries:
      1) Exact match (with/without class)
      2) Try common extensions if missing
      3) Case-insensitive stem match within (class folder first, then whole tree)
    Returns a Path or raises FileNotFoundError.
    """
    base = Path(filename).name  # may or may not include an extension
    stem = Path(base).stem
    has_ext = Path(base).suffix != ""
    exts = ["", ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]

    # 1) Direct, if filename already includes an extension
    if has_ext:
        if cls:
            cand = img_root / cls / base
            if cand.exists():
                return cand
        hits = list(img_root.rglob(base))
        if hits:
            return hits[0]

    # 2) Try common extensions in class folder first, then whole tree
    search_roots = []
    if cls and (img_root / cls).exists():
        search_roots.append(img_root / cls)
    search_roots.append(img_root)

    for root in search_roots:
        for ext in exts:
            if ext == "" and has_ext:
                continue
            cand = root / f"{stem}{ext}"
            if cand.exists():
                return cand

        # 3) Case-insensitive STEM match within this root
        stem_lower = stem.casefold()
        for p in root.rglob("*"):
            if p.is_file():
                try:
                    if p.stem.casefold() == stem_lower:
                        return p
                except Exception:
                    continue

    raise FileNotFoundError(f"Image '{base}' not found under {img_root}")


def parse_voc_xml(xml_path: Path):
    """
    Parse a Pascal VOC XML; returns: filename (str), and list of (cls, (xmin,ymin,xmax,ymax))
    """
    root = ET.parse(xml_path).getroot()
    filename = root.findtext("filename")
    objs = []
    for obj in root.findall("object"):
        cls = obj.findtext("name")
        bb = obj.find("bndbox")
        xmin = int(float(bb.findtext("xmin"))); ymin = int(float(bb.findtext("ymin")))
        xmax = int(float(bb.findtext("xmax"))); ymax = int(float(bb.findtext("ymax")))
        objs.append((cls, (xmin, ymin, xmax, ymax)))
    return filename, objs


def build_chips(split: str, force_rebuild: bool = False):
    """
    For each XML in {split}/annotations, make crop(s) from {split}/images.
    Saves under CHIPS/{split}/{class}/sourceFile_bbXX.png
    Also writes {split}_chips_index.csv mapping chips -> source/bbox/true_label.
    Logs warnings and skips missing images instead of crashing.
    """
    ann_dir = DATA / split / "annotations"
    img_dir = DATA / split / "images"
    out_dir = CHIPS / split
    index_csv = OUT / f"{split}_chips_index.csv"

    if force_rebuild and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # If we already have an index and chips exist, skip
    if index_csv.exists() and any(out_dir.glob("*/*")) and not force_rebuild:
        print(f"[{split}] chips already exist; using {index_csv}")
        return

    xmls = sorted([p for p in ann_dir.iterdir() if p.suffix.lower() == ".xml"])
    if not xmls:
        raise FileNotFoundError(f"No XML files in {ann_dir}.")

    rows = []
    missing = 0
    for xml in tqdm(xmls, desc=f"Making chips ({split})"):
        img_fn, objects = parse_voc_xml(xml)
        if not objects:
            continue

        first_cls = objects[0][0] if objects else None
        try:
            img_path = resolve_image(img_dir, img_fn, first_cls)
        except FileNotFoundError:
            # try again without class hint
            try:
                img_path = resolve_image(img_dir, img_fn, None)
            except FileNotFoundError:
                missing += 1
                print(f"[WARN] Could not find source image for '{img_fn}' (xml: {xml.name}). Skipping.")
                continue

        im = Image.open(img_path).convert("RGB")
        W, H = im.size

        for i, (cls, (xmin, ymin, xmax, ymax)) in enumerate(objects):
            # clamp bbox to image bounds
            xmin = max(0, min(int(xmin), W - 1))
            xmax = max(1, min(int(xmax), W))
            ymin = max(0, min(int(ymin), H - 1))
            ymax = max(1, min(int(ymax), H))
            if xmax <= xmin or ymax <= ymin:
                continue

            crop = im.crop((xmin, ymin, xmax, ymax))
            if crop.size[0] < MIN_BOX or crop.size[1] < MIN_BOX:
                continue

            cls_dir = out_dir / cls
            cls_dir.mkdir(parents=True, exist_ok=True)
            out_name = f"{Path(img_fn).stem}_bb{i:02d}.png"
            out_path = cls_dir / out_name
            crop.save(out_path)
            rows.append({
                "chip_path": str(out_path),
                "source_image": str(img_path),
                "bbox_index": i,
                "true_label": cls,
                "bbox_xmin": xmin,
                "bbox_ymin": ymin,
                "bbox_xmax": xmax,
                "bbox_ymax": ymax,
                "split": split
            })

    df = pd.DataFrame(rows)
    df.to_csv(index_csv, index=False)
    print(f"[{split}] chips: {len(df)} saved → {out_dir}")
    if missing:
        print(f"[{split}] WARNING: {missing} XML(s) had unresolved image filenames. They were skipped.")


# -------------------------
# CLI args
# -------------------------

def parse_args():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild-chips", action="store_true", help="Regenerate chips from XML")
    return ap.parse_args()


# -------------------------
# Main
# -------------------------

def main():
    args = parse_args()
    seed_all(42)

    # Build chips for train/validation
    build_chips("train", force_rebuild=args.rebuild_chips)
    build_chips("validation", force_rebuild=args.rebuild_chips)

    # Datasets / Dataloaders
    tfm_train = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    tfm_eval = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    train_ds = tv.datasets.ImageFolder(CHIPS / "train", transform=tfm_train)
    val_ds   = tv.datasets.ImageFolder(CHIPS / "validation", transform=tfm_eval)

    train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True,  num_workers=NUM_WORKERS, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    classes = train_ds.classes
    print("Classes:", classes)

    # Save class mapping for downstream apps
    with open(OUT / "class_to_idx.json", "w") as f:
        json.dump(train_ds.class_to_idx, f, indent=2)

    # Model / Optim
    model = tv.models.resnet18(weights=tv.models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model = model.to(DEVICE)

    crit = nn.CrossEntropyLoss()
    opt  = torch.optim.AdamW(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)

    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE == "cuda"))

    def run_epoch(dl, training=False):
        model.train(training)
        total, correct, loss_sum = 0, 0, 0.0
        pbar = tqdm(dl, leave=False)
        for x, y in pbar:
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            if training:
                opt.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
                    out = model(x)
                    loss = crit(out, y)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                with torch.no_grad():
                    out = model(x)
                    loss = crit(out, y)

            preds = out.argmax(1)
            loss_sum += loss.item() * x.size(0)
            correct  += (preds == y).sum().item()
            total    += x.size(0)
            pbar.set_description(f"{'train' if training else 'valid'} loss={loss_sum/total:.4f} acc={correct/total:.3f}")
        return loss_sum/total, correct/total

    # Train loop with early stopping on val loss
    best_val_acc = 0.0
    best_val_loss = math.inf
    no_improve = 0

    for ep in range(1, EPOCHS+1):
        tr_loss, tr_acc = run_epoch(train_dl, training=True)
        va_loss, va_acc = run_epoch(val_dl, training=False)
        sched.step(va_loss)

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save(model.state_dict(), OUT / "models" / "neudet_resnet18_best.pt")

        if va_loss + 1e-6 < best_val_loss:
            best_val_loss = va_loss
            no_improve = 0
        else:
            no_improve += 1

        print(f"ep{ep:02d} | train_acc={tr_acc:.3f} val_acc={va_acc:.3f} val_loss={va_loss:.4f} (no_improve={no_improve})")

        if no_improve >= PATIENCE:
            print(f"Early stopping at epoch {ep}.")
            break

    torch.save(model.state_dict(), OUT / "models" / "neudet_resnet18_last.pt")

    # -------------------------
    # Evaluate + Export CSVs
    # -------------------------
    val_index = pd.read_csv(OUT / "validation_chips_index.csv")
    softmax = nn.Softmax(dim=1)
    model.eval()

    val_paths = []
    for root_dir, _, files in os.walk(CHIPS / "validation"):
        for f in files:
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                val_paths.append(Path(root_dir) / f)
    val_paths = sorted(val_paths)

    def load_img(p: Path):
        im = Image.open(p).convert("RGB")
        return tfm_eval(im).unsqueeze(0).to(DEVICE)

    pred_rows = []
    prob_rows = []  # wide table
    with torch.no_grad():
        for p in tqdm(val_paths, desc="Scoring validation"):
            x = load_img(p)
            out = model(x)
            probs = softmax(out).cpu().numpy()[0]
            pred_idx = int(np.argmax(probs))
            pred_label = classes[pred_idx]
            conf = float(probs[pred_idx])

            # safer lookup (handles absolute vs relative path differences)
            row = val_index[val_index["chip_path"] == str(p)]
            if row.empty:
                row = val_index[val_index["chip_path"] == str(p.resolve())]
            if row.empty:
                # fallback: match by stem
                stem = p.stem
                cand = val_index[val_index["chip_path"].str.contains(fr"{stem}\\.(png|jpe?g)$", regex=True, case=False, na=False)]
                if cand.empty:
                    # can't match; skip
                    continue
                row = cand.iloc[0]
            else:
                row = row.iloc[0]

            base = {
                "chip_path": str(p),
                "source_image": row["source_image"],
                "bbox_index": int(row["bbox_index"]),
                "true_label": row["true_label"],
                "pred_label": pred_label,
                "confidence": conf
            }
            pred_rows.append(base)

            wide = {
                "chip_path": str(p),
                "true_label": row["true_label"],
                "pred_label": pred_label,
                "confidence": conf,
            }
            for ci, cname in enumerate(classes):
                wide[f"prob_{cname}"] = float(probs[ci])
            prob_rows.append(wide)

    df_pred = pd.DataFrame(pred_rows)
    df_pred.to_csv(OUT / "NEUDET_Predictions.csv", index=False)

    df_prob = pd.DataFrame(prob_rows)
    df_prob.to_csv(OUT / "NEUDET_PredictionsWide.csv", index=False)

    # Confusion matrices and class report
    cm = confusion_matrix(df_pred["true_label"], df_pred["pred_label"], labels=classes)
    cm_norm = confusion_matrix(df_pred["true_label"], df_pred["pred_label"], labels=classes, normalize="true")

    pd.DataFrame(cm, index=classes, columns=classes).to_csv(OUT / "NEUDET_ConfusionMatrix.csv")
    pd.DataFrame(cm_norm, index=classes, columns=classes).to_csv(OUT / "NEUDET_ConfusionMatrix_Normalized.csv")

    report = classification_report(df_pred["true_label"], df_pred["pred_label"], labels=classes, output_dict=True, zero_division=0)
    pd.DataFrame(report).to_csv(OUT / "NEUDET_ClassReport.csv")

    print("Saved:")
    print(" -", OUT / "NEUDET_Predictions.csv")
    print(" -", OUT / "NEUDET_PredictionsWide.csv")
    print(" -", OUT / "NEUDET_ConfusionMatrix.csv")
    print(" -", OUT / "NEUDET_ConfusionMatrix_Normalized.csv")
    print(" -", OUT / "NEUDET_ClassReport.csv")
    print(" -", OUT / "models" / "neudet_resnet18_best.pt")
    print(" -", OUT / "models" / "neudet_resnet18_last.pt")
    print(" -", OUT / "class_to_idx.json")


if __name__ == "__main__":
    main()
