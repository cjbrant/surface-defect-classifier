-- Canonical SQL for the Ignition Perspective named queries.
-- In Ignition, value parameters are referenced as :paramName.
-- This file is the source of truth; each query is mirrored as a named-query
-- resource under projects/surface_defect/ignition/named-query/.


-- ============================================================
-- nqOverallMetrics(:modelName)  -> KPI tiles
-- ============================================================
SELECT metricName, metricValue
FROM metricOverall
WHERE modelName = :modelName;

-- ============================================================
-- nqClassReport(:modelName)  -> per-class precision/recall/F1 bars
-- ============================================================
SELECT className, precisionScore, recallScore, f1Score, support
FROM metricClassReport
WHERE modelName = :modelName
ORDER BY className;

-- ============================================================
-- nqConfusion(:modelName)  -> confusion-matrix heatmap (long form)
-- ============================================================
SELECT trueLabel, predLabel, n, normalizedPct
FROM metricConfusion
WHERE modelName = :modelName
ORDER BY trueLabel, predLabel;

-- ============================================================
-- nqClassF1Comparison()  -> Model Comparison: per-class F1 by model
-- ============================================================
SELECT className,
       MAX(CASE WHEN modelName = 'resnet18' THEN f1Score END) AS f1Resnet18,
       MAX(CASE WHEN modelName = 'vit_b_16' THEN f1Score END) AS f1VitB16
FROM metricClassReport
GROUP BY className
ORDER BY className;

-- ============================================================
-- nqOverallComparison()  -> Model Comparison: headline metrics by model
-- ============================================================
SELECT metricName,
       MAX(CASE WHEN modelName = 'resnet18' THEN metricValue END) AS resnet18,
       MAX(CASE WHEN modelName = 'vit_b_16' THEN metricValue END) AS vitB16
FROM metricOverall
WHERE metricName IN ('accuracy', 'macroF1', 'weightedF1')
GROUP BY metricName;

-- ============================================================
-- nqDefectMap(:modelName)  -> roll-form digital-twin scatter/heatmap
-- ============================================================
SELECT i.positionAcrossWeb, i.positionAlongRoll, i.trueLabel,
       p.predLabel, p.isCorrect, i.isCritical, i.machineId
FROM factInspection i
JOIN factPrediction p ON p.inspectionId = i.inspectionId
WHERE p.modelName = :modelName;

-- ============================================================
-- nqOperationsMachine(:modelName)  -> Operations machine table
-- ============================================================
SELECT i.machineId,
       COUNT(*)                              AS defectCount,
       ROUND(100 * AVG(i.isCritical), 2)     AS pctCritical,
       ROUND(100 * AVG(i.isRepeat), 2)       AS pctRepeat,
       ROUND(SUM(i.costImpact), 2)           AS totalCost
FROM factInspection i
JOIN factPrediction p ON p.inspectionId = i.inspectionId
WHERE p.modelName = :modelName
GROUP BY i.machineId
ORDER BY i.machineId;

-- ============================================================
-- nqPareto(:modelName)  -> Pareto of predicted-defect counts
-- ============================================================
SELECT predLabel,
       cnt,
       ROUND(100 * cnt / SUM(cnt) OVER (), 2)                                   AS pct,
       ROUND(100 * SUM(cnt) OVER (ORDER BY cnt DESC) / SUM(cnt) OVER (), 2)     AS cumulativePct
FROM (
  SELECT p.predLabel, COUNT(*) AS cnt
  FROM factPrediction p
  WHERE p.modelName = :modelName
  GROUP BY p.predLabel
) q
ORDER BY cnt DESC;

-- ============================================================
-- nqOperatorShift(:modelName)  -> operator x shift scatter
-- ============================================================
SELECT i.operatorId, i.shiftCode,
       COUNT(*)                          AS defectCount,
       ROUND(100 * AVG(i.isCritical), 2) AS pctCritical
FROM factInspection i
JOIN factPrediction p ON p.inspectionId = i.inspectionId
WHERE p.modelName = :modelName
GROUP BY i.operatorId, i.shiftCode
ORDER BY i.operatorId, i.shiftCode;

-- ============================================================
-- nqDefectByDay(:modelName)  -> defect count + % critical by day
-- ============================================================
SELECT DATE(i.inspectedTs)             AS inspectionDay,
       COUNT(*)                        AS defectCount,
       ROUND(100 * AVG(i.isCritical), 2) AS pctCritical
FROM factInspection i
JOIN factPrediction p ON p.inspectionId = i.inspectionId
WHERE p.modelName = :modelName
GROUP BY DATE(i.inspectedTs)
ORDER BY inspectionDay;

-- ============================================================
-- nqPredictionExplorer(:modelName, :classFilter, :minConfidence, :correctOnly)
--   -> filterable explorer table. classFilter '%' matches all.
-- ============================================================
SELECT p.predictionId, i.chipPath, i.trueLabel, p.predLabel, p.confidence,
       p.isCorrect, i.machineId, i.shiftCode, i.inspectedTs
FROM factPrediction p
JOIN factInspection i ON i.inspectionId = p.inspectionId
WHERE p.modelName = :modelName
  AND p.predLabel LIKE :classFilter
  AND p.confidence >= :minConfidence
  AND (:correctOnly = 0 OR p.isCorrect = 1)
ORDER BY p.confidence DESC;

-- ============================================================
-- nqChipDetail(:predictionId)  -> chip viewer: probabilities for one chip
-- ============================================================
SELECT cp.className, cp.probability
FROM factClassProbability cp
WHERE cp.predictionId = :predictionId
ORDER BY cp.probability DESC;

-- ============================================================
-- nqLiveFeed(:modelName, :rowLimit)  -> Live Inspection recent feed
-- ============================================================
SELECT id, inspectedTs, predLabel, trueLabel, confidence,
       isCritical, costImpact, machineId, operatorId, shiftCode
FROM liveInspection
WHERE modelName = :modelName
ORDER BY id DESC
LIMIT :rowLimit;

-- ============================================================
-- nqLiveCounts(:modelName)  -> Live running per-class counts
-- ============================================================
SELECT predLabel, COUNT(*) AS cnt, ROUND(100 * AVG(isCritical), 1) AS pctCritical
FROM liveInspection
WHERE modelName = :modelName
GROUP BY predLabel
ORDER BY cnt DESC;
