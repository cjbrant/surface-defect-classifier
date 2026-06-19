#!/usr/bin/env python3
"""
Replay simulator for the Live Inspection view.

Every few seconds, samples a random validation chip + its ResNet-18 prediction and
appends it to `liveInspection` with inspectedTs = now — so the Perspective Live
Inspection view (which polls liveInspection) shows a defect feed arriving in real time.

Run:  MYSQL_HOST=127.0.0.1 MYSQL_USER=root MYSQL_PASSWORD=defectroot \
      .venv/bin/python replay.py [--interval 1.5] [--model resnet18]

Stop with Ctrl-C. Re-run load_mysql.py to clear/reload everything.
"""

from __future__ import annotations

import os
import time
import argparse

from sqlalchemy import create_engine, text


def makeEngine():
    host = os.environ.get("MYSQL_HOST", "127.0.0.1")
    port = os.environ.get("MYSQL_PORT", "3306")
    user = os.environ.get("MYSQL_USER", "root")
    pw = os.environ.get("MYSQL_PASSWORD", "")
    db = os.environ.get("MYSQL_DB", "surfaceDefect")
    return create_engine(f"mysql+pymysql://{user}:{pw}@{host}:{port}/{db}?charset=utf8mb4", future=True)


SAMPLE = text("""
    SELECT i.inspectionId, :model AS modelName, p.predLabel, i.trueLabel, p.confidence,
           i.isCritical, i.costImpact, i.machineId, i.operatorId, i.shiftCode,
           i.positionAcrossWeb, i.positionAlongRoll
    FROM factInspection i
    JOIN factPrediction p ON p.inspectionId = i.inspectionId AND p.modelName = :model
    ORDER BY RAND() LIMIT 1
""")

INSERT = text("""
    INSERT INTO liveInspection
        (inspectionId, modelName, predLabel, trueLabel, confidence, isCritical, costImpact,
         machineId, operatorId, shiftCode, positionAcrossWeb, positionAlongRoll, inspectedTs)
    VALUES
        (:inspectionId, :modelName, :predLabel, :trueLabel, :confidence, :isCritical, :costImpact,
         :machineId, :operatorId, :shiftCode, :positionAcrossWeb, :positionAlongRoll, NOW())
""")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=float, default=1.5, help="seconds between inserts")
    ap.add_argument("--model", default="resnet18")
    args = ap.parse_args()

    engine = makeEngine()
    print(f"Replaying {args.model} chips every {args.interval}s into liveInspection. Ctrl-C to stop.")
    n = 0
    try:
        while True:
            with engine.begin() as conn:
                row = conn.execute(SAMPLE, {"model": args.model}).mappings().first()
                if row:
                    conn.execute(INSERT, dict(row))
                    n += 1
            print(f"  [{n}] {row['predLabel']:<16} conf={float(row['confidence']):.3f} "
                  f"{'CRITICAL' if row['isCritical'] else ''}", flush=True)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print(f"\nStopped after {n} inspections.")


if __name__ == "__main__":
    main()
