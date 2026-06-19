"""
Deterministic synthetic-operations enrichment.

Turns the model-independent validation chip index into a factory-floor "inspection"
fact by attaching machine / operator / shift / timestamp / position / criticality / cost.
Everything is seeded by a stable hash of the chip or source-image path, so repeated loads
reproduce identical rows. Recreates the semantic layer of the prior Power BI dashboard.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta

import pandas as pd

# NEU-DET scenes are 200x200.
IMG_DIM = 200

# Synthetic timeline: a ~30-day window of inspections.
START_DATE = datetime(2026, 5, 1)
TIMELINE_DAYS = 30

MACHINES = [f"EXTR-{i:02d}" for i in range(1, 6)]      # EXTR-01 .. EXTR-05
OPERATORS = [f"OP{i:02d}" for i in range(1, 13)]       # OP01 .. OP12
SHIFTS = {"A": (6, 14), "B": (14, 22), "C": (22, 30)}  # start hour (C wraps past midnight)

# className -> (criticalRate, baseUnitCost). Rates average ~0.25 to match the prior ~25% critical.
CLASS_CONFIG: dict[str, tuple[float, float]] = {
    "crazing":         (0.15, 40.0),
    "inclusion":       (0.30, 70.0),
    "patches":         (0.20, 50.0),
    "pitted_surface":  (0.35, 90.0),
    "rolled-in_scale": (0.30, 75.0),
    "scratches":       (0.18, 45.0),
}
CRITICAL_COST_MULT = 3.0


def stableUnit(*parts: str) -> float:
    """Deterministic float in [0, 1) from the given string parts (salted hash)."""
    h = hashlib.md5("|".join(parts).encode("utf-8")).hexdigest()
    return (int(h[:12], 16) % 1_000_000) / 1_000_000.0


def stableIndex(n: int, *parts: str) -> int:
    """Deterministic integer in [0, n) from the given string parts."""
    h = hashlib.md5("|".join(parts).encode("utf-8")).hexdigest()
    return int(h[12:24], 16) % n


def _timestampFor(sourceImage: str) -> datetime:
    dayOffset = stableIndex(TIMELINE_DAYS, sourceImage, "day")
    shiftCode = list(SHIFTS)[stableIndex(len(SHIFTS), sourceImage, "shift")]
    startHour, endHour = SHIFTS[shiftCode]
    hour = (startHour + stableIndex(max(1, endHour - startHour), sourceImage, "hour")) % 24
    minute = stableIndex(60, sourceImage, "minute")
    return START_DATE + timedelta(days=dayOffset, hours=hour, minutes=minute)


def _shiftFor(sourceImage: str) -> str:
    return list(SHIFTS)[stableIndex(len(SHIFTS), sourceImage, "shift")]


def enrichInspections(indexDf: pd.DataFrame) -> pd.DataFrame:
    """
    Input: the validation chip index (camelCase columns from train.py).
    Output: a copy with synthetic operational columns added, ready for factInspection.
    """
    df = indexDf.copy()

    # Spatial position on the strip (bbox center, normalized to [0, 1]).
    xCenter = (df["bboxXmin"] + df["bboxXmax"]) / 2.0
    yCenter = (df["bboxYmin"] + df["bboxYmax"]) / 2.0
    df["positionAcrossWeb"] = (xCenter / IMG_DIM).clip(0, 1).round(5)
    df["positionAlongRoll"] = (yCenter / IMG_DIM).clip(0, 1).round(5)

    # Machine / operator / shift — keyed on sourceImage so one scene shares a station.
    df["machineId"] = df["sourceImage"].map(lambda s: MACHINES[stableIndex(len(MACHINES), s, "machine")])
    df["operatorId"] = df["sourceImage"].map(lambda s: OPERATORS[stableIndex(len(OPERATORS), s, "operator")])
    df["shiftCode"] = df["sourceImage"].map(_shiftFor)
    df["inspectedTs"] = df["sourceImage"].map(_timestampFor)

    # Repeat defect — a recurring-issue KPI. Deterministic synthetic flag ~23%
    # (a data-driven "scene has >1 chip" rule lands ~91% since NEU-DET scenes are
    # multi-defect, which is unrealistic for a repeat-rate metric).
    REPEAT_RATE = 0.23
    df["isRepeat"] = df["chipPath"].map(lambda p: int(stableUnit(p, "repeat") < REPEAT_RATE))

    # Criticality — per-class Bernoulli, deterministic by chip.
    def _critical(row: pd.Series) -> int:
        rate = CLASS_CONFIG.get(row["trueLabel"], (0.25, 50.0))[0]
        return int(stableUnit(row["chipPath"], "critical") < rate)
    df["isCritical"] = df.apply(_critical, axis=1)

    # Cost impact — base class cost, multiplied for critical, with +/-20% jitter.
    def _cost(row: pd.Series) -> float:
        base = CLASS_CONFIG.get(row["trueLabel"], (0.25, 50.0))[1]
        mult = CRITICAL_COST_MULT if row["isCritical"] else 1.0
        jitter = 0.8 + 0.4 * stableUnit(row["chipPath"], "cost")
        return round(base * mult * jitter, 2)
    df["costImpact"] = df.apply(_cost, axis=1)

    return df


def classConfigRows() -> list[dict[str, object]]:
    """dimDefectClass rows derived from CLASS_CONFIG (classId by sorted class name)."""
    rows: list[dict[str, object]] = []
    for classId, className in enumerate(sorted(CLASS_CONFIG)):
        rate, baseCost = CLASS_CONFIG[className]
        rows.append({
            "classId": classId,
            "className": className,
            "isCriticalClass": int(rate >= 0.25),
            "baseUnitCost": baseCost,
        })
    return rows
