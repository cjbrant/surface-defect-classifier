#!/usr/bin/env python3
"""
NEU Surface Defect Classifier (classification + dashboard exports)

End-to-end script to:
1) Build defect "chips" (cropped defects) from Pascal VOC XML under NEU-DET/{split}/annotations
2) Train a classifier on the chips (ResNet-18 or ViT-B/16)
3) Export rich CSVs for analysis / Ignition Perspective dashboards

Usage examples
--------------
# (re)generate chips and train ResNet-18
python train.py --model resnet18 --rebuild-chips

# train ViT-B/16, reusing existing chips
python train.py --model vit_b_16

Outputs
-------
Shared (model-independent):
- results/chips/                            (cropped images by split/class)
- results/{split}_chips_index.csv           (chip -> source image, bbox, trueLabel)
- results/class_to_idx.json                 (class -> index mapping)

Per model (under results/<modelName>/):
- NEUDET_Predictions.csv                    (long-form per-chip predictions)
- NEUDET_PredictionsWide.csv                (per-chip probabilities in wide format)
- NEUDET_ConfusionMatrix.csv                (raw counts)
- NEUDET_ConfusionMatrix_Normalized.csv     (row-normalized)
- NEUDET_ClassReport.csv                    (precision/recall/F1/support per class)
- models/best.pt                            (best-on-val-acc)
- models/last.pt                            (last epoch)
"""

from __future__ import annotations

import os
import json
import math
import shutil
import random
import argparse
from pathlib import Path

try:
    # Hardened against XXE / billion-laughs; preferred when available.
    from defusedxml import ElementTree as ET
except ImportError:  # pragma: no cover - fallback for minimal environments
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
OUT.mkdir(parents=True, exist_ok=True)
CHIPS.mkdir(parents=True, exist_ok=True)

SUPPORTED_MODELS = ("resnet18", "vit_b_16")

IMG_SIZE = 224
BATCH = 32
EPOCHS = 12
LR = 1e-3
NUM_WORKERS = 2  # set to 0 on macOS if you get issues
PATIENCE = 4     # early stop on val loss plateau
MIN_BOX = 10     # skip tiny crops

# -------------------------
# Utils
# -------------------------


def seedAll(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def selectDevice() -> str:
    """Pick the best available device: CUDA, then Apple-Silicon MPS, then CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolveImage(imgRoot: Path, filename: str, cls: str | None = None) -> Path:
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
    hasExt = Path(base).suffix != ""
    exts = ["", ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]

    # 1) Direct, if filename already includes an extension
    if hasExt:
        if cls:
            cand = imgRoot / cls / base
            if cand.exists():
                return cand
        hits = list(imgRoot.rglob(base))
        if hits:
            return hits[0]

    # 2) Try common extensions in class folder first, then whole tree
    searchRoots: list[Path] = []
    if cls and (imgRoot / cls).exists():
        searchRoots.append(imgRoot / cls)
    searchRoots.append(imgRoot)

    for root in searchRoots:
        for ext in exts:
            if ext == "" and hasExt:
                continue
            cand = root / f"{stem}{ext}"
            if cand.exists():
                return cand

        # 3) Case-insensitive STEM match within this root
        stemLower = stem.casefold()
        for p in root.rglob("*"):
            if p.is_file():
                try:
                    if p.stem.casefold() == stemLower:
                        return p
                except Exception:
                    continue

    raise FileNotFoundError(f"Image '{base}' not found under {imgRoot}")


def parseVocXml(xmlPath: Path) -> tuple[str, list[tuple[str, tuple[int, int, int, int]]]]:
    """
    Parse a Pascal VOC XML; returns: filename (str), and list of (cls, (xmin,ymin,xmax,ymax)).
    """
    root = ET.parse(xmlPath).getroot()
    filename = root.findtext("filename") or ""
    objs: list[tuple[str, tuple[int, int, int, int]]] = []
    for obj in root.findall("object"):
        cls = obj.findtext("name") or ""
        bb = obj.find("bndbox")
        if bb is None:
            continue
        xmin = int(float(bb.findtext("xmin") or 0)); ymin = int(float(bb.findtext("ymin") or 0))
        xmax = int(float(bb.findtext("xmax") or 0)); ymax = int(float(bb.findtext("ymax") or 0))
        objs.append((cls, (xmin, ymin, xmax, ymax)))
    return filename, objs


def buildChips(split: str, forceRebuild: bool = False) -> None:
    """
    For each XML in {split}/annotations, make crop(s) from {split}/images.
    Saves under CHIPS/{split}/{class}/sourceFile_bbXX.png
    Also writes {split}_chips_index.csv mapping chips -> source/bbox/trueLabel.
    Logs warnings and skips missing images instead of crashing.
    """
    annDir = DATA / split / "annotations"
    imgDir = DATA / split / "images"
    outDir = CHIPS / split
    indexCsv = OUT / f"{split}_chips_index.csv"

    if forceRebuild and outDir.exists():
        shutil.rmtree(outDir)
    outDir.mkdir(parents=True, exist_ok=True)

    # If we already have an index and chips exist, skip
    if indexCsv.exists() and any(outDir.glob("*/*")) and not forceRebuild:
        print(f"[{split}] chips already exist; using {indexCsv}")
        return

    xmls = sorted([p for p in annDir.iterdir() if p.suffix.lower() == ".xml"])
    if not xmls:
        raise FileNotFoundError(f"No XML files in {annDir}.")

    rows: list[dict[str, object]] = []
    missing = 0
    for xml in tqdm(xmls, desc=f"Making chips ({split})"):
        imgFn, objects = parseVocXml(xml)
        if not objects:
            continue

        firstCls = objects[0][0] if objects else None
        try:
            imgPath = resolveImage(imgDir, imgFn, firstCls)
        except FileNotFoundError:
            # try again without class hint
            try:
                imgPath = resolveImage(imgDir, imgFn, None)
            except FileNotFoundError:
                missing += 1
                print(f"[WARN] Could not find source image for '{imgFn}' (xml: {xml.name}). Skipping.")
                continue

        im = Image.open(imgPath).convert("RGB")
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

            clsDir = outDir / cls
            clsDir.mkdir(parents=True, exist_ok=True)
            outName = f"{Path(imgFn).stem}_bb{i:02d}.png"
            outPath = clsDir / outName
            crop.save(outPath)
            rows.append({
                "chipPath": str(outPath),
                "sourceImage": str(imgPath),
                "bboxIndex": i,
                "trueLabel": cls,
                "bboxXmin": xmin,
                "bboxYmin": ymin,
                "bboxXmax": xmax,
                "bboxYmax": ymax,
                "split": split,
            })

    df = pd.DataFrame(rows)
    df.to_csv(indexCsv, index=False)
    print(f"[{split}] chips: {len(df)} saved -> {outDir}")
    if missing:
        print(f"[{split}] WARNING: {missing} XML(s) had unresolved image filenames. They were skipped.")


def buildModel(modelName: str, numClasses: int) -> nn.Module:
    """Construct an ImageNet-pretrained backbone with a fresh classification head."""
    if modelName == "resnet18":
        model = tv.models.resnet18(weights=tv.models.ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, numClasses)
    elif modelName == "vit_b_16":
        model = tv.models.vit_b_16(weights=tv.models.ViT_B_16_Weights.DEFAULT)
        model.heads.head = nn.Linear(model.heads.head.in_features, numClasses)
    else:
        raise ValueError(f"Unsupported model '{modelName}'. Choose from {SUPPORTED_MODELS}.")
    return model


# -------------------------
# CLI args
# -------------------------


def parseArgs() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=SUPPORTED_MODELS, default="resnet18",
                    help="Backbone architecture to train/evaluate")
    ap.add_argument("--rebuild-chips", action="store_true", help="Regenerate chips from XML")
    ap.add_argument("--epochs", type=int, default=EPOCHS,
                    help=f"Number of training epochs (default {EPOCHS})")
    return ap.parse_args()


# -------------------------
# Main
# -------------------------


def main() -> None:
    args = parseArgs()
    seedAll(42)

    modelName: str = args.model
    device = selectDevice()
    print(f"Model: {modelName} | Device: {device}")

    # Per-model output directory (chips/index stay shared at OUT)
    modelOut = OUT / modelName
    (modelOut / "models").mkdir(parents=True, exist_ok=True)

    # Build chips for train/validation (shared across models)
    buildChips("train", forceRebuild=args.rebuild_chips)
    buildChips("validation", forceRebuild=args.rebuild_chips)

    # Datasets / Dataloaders
    tfmTrain = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    tfmEval = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    trainDs = tv.datasets.ImageFolder(CHIPS / "train", transform=tfmTrain)
    valDs = tv.datasets.ImageFolder(CHIPS / "validation", transform=tfmEval)

    trainDl = DataLoader(trainDs, batch_size=BATCH, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    valDl = DataLoader(valDs, batch_size=BATCH, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    classes = trainDs.classes
    print("Classes:", classes)

    # Save class mapping for downstream apps (shared)
    with open(OUT / "class_to_idx.json", "w") as f:
        json.dump(trainDs.class_to_idx, f, indent=2)

    # Model / Optim
    model = buildModel(modelName, len(classes)).to(device)

    crit = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)

    useAmp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=useAmp)

    def runEpoch(dl: DataLoader, training: bool = False) -> tuple[float, float]:
        model.train(training)
        total, correct, lossSum = 0, 0, 0.0
        pbar = tqdm(dl, leave=False)
        for x, y in pbar:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            if training:
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=useAmp):
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
            lossSum += loss.item() * x.size(0)
            correct += (preds == y).sum().item()
            total += x.size(0)
            pbar.set_description(f"{'train' if training else 'valid'} loss={lossSum/total:.4f} acc={correct/total:.3f}")
        return lossSum / total, correct / total

    # Train loop with early stopping on val loss
    bestValAcc = 0.0
    bestValLoss = math.inf
    noImprove = 0

    for ep in range(1, args.epochs + 1):
        trLoss, trAcc = runEpoch(trainDl, training=True)
        vaLoss, vaAcc = runEpoch(valDl, training=False)
        sched.step(vaLoss)

        if vaAcc > bestValAcc:
            bestValAcc = vaAcc
            torch.save(model.state_dict(), modelOut / "models" / "best.pt")

        if vaLoss + 1e-6 < bestValLoss:
            bestValLoss = vaLoss
            noImprove = 0
        else:
            noImprove += 1

        print(f"ep{ep:02d} | train_acc={trAcc:.3f} val_acc={vaAcc:.3f} val_loss={vaLoss:.4f} (no_improve={noImprove})")

        if noImprove >= PATIENCE:
            print(f"Early stopping at epoch {ep}.")
            break

    torch.save(model.state_dict(), modelOut / "models" / "last.pt")

    # -------------------------
    # Evaluate + Export CSVs
    # -------------------------
    valIndex = pd.read_csv(OUT / "validation_chips_index.csv")
    softmax = nn.Softmax(dim=1)
    model.eval()

    valPaths: list[Path] = []
    for rootDir, _, files in os.walk(CHIPS / "validation"):
        for f in files:
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                valPaths.append(Path(rootDir) / f)
    valPaths = sorted(valPaths)

    def loadImg(p: Path) -> torch.Tensor:
        im = Image.open(p).convert("RGB")
        return tfmEval(im).unsqueeze(0).to(device)

    predRows: list[dict[str, object]] = []
    probRows: list[dict[str, object]] = []  # wide table
    with torch.no_grad():
        for p in tqdm(valPaths, desc="Scoring validation"):
            x = loadImg(p)
            out = model(x)
            probs = softmax(out).cpu().numpy()[0]
            predIdx = int(np.argmax(probs))
            predLabel = classes[predIdx]
            conf = float(probs[predIdx])

            # safer lookup (handles absolute vs relative path differences)
            row = valIndex[valIndex["chipPath"] == str(p)]
            if row.empty:
                row = valIndex[valIndex["chipPath"] == str(p.resolve())]
            if row.empty:
                # fallback: match by stem
                stem = p.stem
                cand = valIndex[valIndex["chipPath"].str.contains(fr"{stem}\\.(png|jpe?g)$", regex=True, case=False, na=False)]
                if cand.empty:
                    # can't match; skip
                    continue
                rowData = cand.iloc[0]
            else:
                rowData = row.iloc[0]

            predRows.append({
                "modelName": modelName,
                "chipPath": str(p),
                "sourceImage": rowData["sourceImage"],
                "bboxIndex": int(rowData["bboxIndex"]),
                "trueLabel": rowData["trueLabel"],
                "predLabel": predLabel,
                "confidence": conf,
            })

            wide: dict[str, object] = {
                "modelName": modelName,
                "chipPath": str(p),
                "trueLabel": rowData["trueLabel"],
                "predLabel": predLabel,
                "confidence": conf,
            }
            for ci, cname in enumerate(classes):
                wide[f"prob_{cname}"] = float(probs[ci])
            probRows.append(wide)

    dfPred = pd.DataFrame(predRows)
    dfPred.to_csv(modelOut / "NEUDET_Predictions.csv", index=False)

    dfProb = pd.DataFrame(probRows)
    dfProb.to_csv(modelOut / "NEUDET_PredictionsWide.csv", index=False)

    # Confusion matrices and class report
    cm = confusion_matrix(dfPred["trueLabel"], dfPred["predLabel"], labels=classes)
    cmNorm = confusion_matrix(dfPred["trueLabel"], dfPred["predLabel"], labels=classes, normalize="true")

    pd.DataFrame(cm, index=classes, columns=classes).to_csv(modelOut / "NEUDET_ConfusionMatrix.csv")
    pd.DataFrame(cmNorm, index=classes, columns=classes).to_csv(modelOut / "NEUDET_ConfusionMatrix_Normalized.csv")

    report = classification_report(dfPred["trueLabel"], dfPred["predLabel"], labels=classes, output_dict=True, zero_division=0)
    reportDf = pd.DataFrame(report)
    reportDf.insert(0, "modelName", modelName)
    reportDf.to_csv(modelOut / "NEUDET_ClassReport.csv")

    print("Saved:")
    for name in ("NEUDET_Predictions.csv", "NEUDET_PredictionsWide.csv",
                 "NEUDET_ConfusionMatrix.csv", "NEUDET_ConfusionMatrix_Normalized.csv",
                 "NEUDET_ClassReport.csv", "models/best.pt", "models/last.pt"):
        print(" -", modelOut / name)
    print(" -", OUT / "class_to_idx.json")


if __name__ == "__main__":
    main()
