#!/usr/bin/env python3
"""
Load model outputs + enriched operations layer into the MySQL star schema.

Reads:
  results/validation_chips_index.csv           (shared, model-independent)
  results/<modelName>/NEUDET_Predictions.csv
  results/<modelName>/NEUDET_PredictionsWide.csv
  results/<modelName>/NEUDET_ClassReport.csv
  results/<modelName>/NEUDET_ConfusionMatrix.csv
  results/<modelName>/NEUDET_ConfusionMatrix_Normalized.csv

Writes the camelCase star schema defined in mysql/01_schema.sql.
Idempotent: truncates all tables before loading.

Connection via env (with sensible local-Docker defaults):
  MYSQL_HOST=127.0.0.1 MYSQL_PORT=3306 MYSQL_USER=root MYSQL_PASSWORD=... MYSQL_DB=surfaceDefect
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text

from enrich import enrichInspections, classConfigRows, MACHINES, OPERATORS, SHIFTS

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS = REPO_ROOT / "results"


def relChip(p: str) -> str:
    """Path relative to the chips root, e.g. 'validation/crazing/crazing_241_bb00.png'.

    Makes the inspection<->prediction join independent of where training ran
    (local vs the Mac Mini), and gives the nginx image URL directly.
    """
    return str(p).split("/chips/")[-1]


def relImg(p: str) -> str:
    """Source image path relative to its split's images root (for stable hashing)."""
    return str(p).split("/images/")[-1]

MODEL_META: dict[str, tuple[str, str, float]] = {
    "resnet18": ("ResNet-18", "CNN (residual)", 11.69),
    "vit_b_16": ("ViT-B/16", "Vision Transformer", 86.57),
}

# Tables in FK-safe truncate order (children first).
TABLES = [
    "liveInspection", "factClassProbability", "factPrediction",
    "metricConfusion", "metricClassReport", "metricOverall",
    "factInspection", "dimModel", "dimDefectClass",
    "dimMachine", "dimOperator", "dimShift",
]


def makeEngine():
    host = os.environ.get("MYSQL_HOST", "127.0.0.1")
    port = os.environ.get("MYSQL_PORT", "3306")
    user = os.environ.get("MYSQL_USER", "root")
    pw = os.environ.get("MYSQL_PASSWORD", "")
    db = os.environ.get("MYSQL_DB", "surfaceDefect")
    url = f"mysql+pymysql://{user}:{pw}@{host}:{port}/{db}?charset=utf8mb4"
    return create_engine(url, future=True)


def discoverModels() -> list[str]:
    models = []
    for d in sorted(RESULTS.iterdir()) if RESULTS.exists() else []:
        if d.is_dir() and (d / "NEUDET_Predictions.csv").exists():
            models.append(d.name)
    return models


def truncateAll(conn) -> None:
    conn.execute(text("SET FOREIGN_KEY_CHECKS = 0"))
    for tbl in TABLES:
        conn.execute(text(f"TRUNCATE TABLE {tbl}"))
    conn.execute(text("SET FOREIGN_KEY_CHECKS = 1"))


def loadDimensions(conn, models: list[str]) -> None:
    # dimDefectClass
    pd.DataFrame(classConfigRows()).to_sql("dimDefectClass", conn, if_exists="append", index=False)

    # dimMachine / dimOperator / dimShift
    pd.DataFrame({"machineId": MACHINES, "machineName": [f"Extruder {m[-2:]}" for m in MACHINES]}) \
        .to_sql("dimMachine", conn, if_exists="append", index=False)
    pd.DataFrame({"operatorId": OPERATORS, "operatorName": [f"Operator {o[-2:]}" for o in OPERATORS]}) \
        .to_sql("dimOperator", conn, if_exists="append", index=False)
    pd.DataFrame({"shiftCode": list(SHIFTS), "label": [f"Shift {s}" for s in SHIFTS]}) \
        .to_sql("dimShift", conn, if_exists="append", index=False)

    # dimModel — metadata + trainedTs from the weight file mtime when available.
    rows = []
    for m in models:
        display, arch, params = MODEL_META.get(m, (m, "unknown", None))
        bestPt = RESULTS / m / "models" / "best.pt"
        trainedTs = pd.Timestamp(bestPt.stat().st_mtime, unit="s") if bestPt.exists() else None
        rows.append({"modelName": m, "displayName": display, "architecture": arch,
                     "paramsM": params, "trainedTs": trainedTs, "notes": None})
    pd.DataFrame(rows).to_sql("dimModel", conn, if_exists="append", index=False)


def loadInspections(conn) -> pd.DataFrame:
    indexDf = pd.read_csv(RESULTS / "validation_chips_index.csv")
    indexDf["chipPath"] = indexDf["chipPath"].map(relChip)
    indexDf["sourceImage"] = indexDf["sourceImage"].map(relImg)
    enriched = enrichInspections(indexDf)
    cols = ["chipPath", "sourceImage", "bboxIndex", "bboxXmin", "bboxYmin", "bboxXmax", "bboxYmax",
            "positionAcrossWeb", "positionAlongRoll", "trueLabel", "machineId", "operatorId",
            "shiftCode", "inspectedTs", "isRepeat", "isCritical", "costImpact"]
    enriched[cols].to_sql("factInspection", conn, if_exists="append", index=False)

    idMap = pd.read_sql(text("SELECT inspectionId, chipPath FROM factInspection"), conn)
    return idMap.set_index("chipPath")["inspectionId"]


def loadModel(conn, model: str, chipToInspection: pd.Series) -> None:
    mdir = RESULTS / model

    # --- factPrediction ---
    pred = pd.read_csv(mdir / "NEUDET_Predictions.csv")
    pred["chipPath"] = pred["chipPath"].map(relChip)
    pred["inspectionId"] = pred["chipPath"].map(chipToInspection)
    pred = pred.dropna(subset=["inspectionId"]).copy()
    pred["inspectionId"] = pred["inspectionId"].astype(int)
    pred["isCorrect"] = (pred["trueLabel"] == pred["predLabel"]).astype(int)
    pred[["inspectionId", "modelName", "predLabel", "confidence", "isCorrect"]] \
        .to_sql("factPrediction", conn, if_exists="append", index=False)

    predIds = pd.read_sql(
        text("SELECT predictionId, inspectionId FROM factPrediction WHERE modelName = :m"),
        conn, params={"m": model})
    inspectionToPrediction = predIds.set_index("inspectionId")["predictionId"]

    # --- factClassProbability (melt the wide table) ---
    wide = pd.read_csv(mdir / "NEUDET_PredictionsWide.csv")
    wide["chipPath"] = wide["chipPath"].map(relChip)
    probCols = [c for c in wide.columns if c.startswith("prob_")]
    melted = wide.melt(id_vars=["chipPath"], value_vars=probCols,
                       var_name="className", value_name="probability")
    melted["className"] = melted["className"].str.replace("^prob_", "", regex=True)
    melted["inspectionId"] = melted["chipPath"].map(chipToInspection)
    melted = melted.dropna(subset=["inspectionId"]).copy()
    melted["predictionId"] = melted["inspectionId"].astype(int).map(inspectionToPrediction)
    melted = melted.dropna(subset=["predictionId"]).copy()
    melted["predictionId"] = melted["predictionId"].astype(int)
    melted[["predictionId", "className", "probability"]] \
        .to_sql("factClassProbability", conn, if_exists="append", index=False)

    # --- metrics: class report ---
    report = pd.read_csv(mdir / "NEUDET_ClassReport.csv", index_col=0)
    classCols = [c for c in report.columns
                 if c not in ("modelName", "accuracy", "macro avg", "weighted avg")]
    reportRows = [{
        "modelName": model, "className": c,
        "precisionScore": float(report.loc["precision", c]),
        "recallScore": float(report.loc["recall", c]),
        "f1Score": float(report.loc["f1-score", c]),
        "support": int(float(report.loc["support", c])),
    } for c in classCols]
    pd.DataFrame(reportRows).to_sql("metricClassReport", conn, if_exists="append", index=False)

    # --- metrics: overall ---
    overall = [
        {"modelName": model, "metricName": "accuracy", "metricValue": float(report.loc["precision", "accuracy"])},
        {"modelName": model, "metricName": "macroF1", "metricValue": float(report.loc["f1-score", "macro avg"])},
        {"modelName": model, "metricName": "weightedF1", "metricValue": float(report.loc["f1-score", "weighted avg"])},
        {"modelName": model, "metricName": "totalSupport", "metricValue": float(report.loc["support", "macro avg"])},
    ]
    pd.DataFrame(overall).to_sql("metricOverall", conn, if_exists="append", index=False)

    # --- metrics: confusion (long form) ---
    cm = pd.read_csv(mdir / "NEUDET_ConfusionMatrix.csv", index_col=0)
    cmNorm = pd.read_csv(mdir / "NEUDET_ConfusionMatrix_Normalized.csv", index_col=0)
    confRows = []
    for trueLabel in cm.index:
        for predLabel in cm.columns:
            confRows.append({
                "modelName": model, "trueLabel": trueLabel, "predLabel": predLabel,
                "n": int(cm.loc[trueLabel, predLabel]),
                "normalizedPct": float(cmNorm.loc[trueLabel, predLabel]),
            })
    pd.DataFrame(confRows).to_sql("metricConfusion", conn, if_exists="append", index=False)

    print(f"  [{model}] {len(pred)} predictions, {len(reportRows)} class metrics loaded")


def main() -> None:
    models = discoverModels()
    if not models:
        sys.exit(f"No model outputs found under {RESULTS}. Run train.py first.")
    print(f"Models discovered: {models}")

    engine = makeEngine()
    with engine.begin() as conn:
        truncateAll(conn)
        loadDimensions(conn, models)
        chipToInspection = loadInspections(conn)
        print(f"factInspection: {len(chipToInspection)} rows")
        for model in models:
            loadModel(conn, model, chipToInspection)
    print("Load complete.")


if __name__ == "__main__":
    main()
