SELECT i.machineId                      AS `Machine`,
       COUNT(*)                         AS `Defect Count`,
       ROUND(100*AVG(i.isCritical),1)   AS `% Critical`,
       ROUND(100*AVG(i.isRepeat),1)     AS `% Repeat`,
       ROUND(SUM(i.costImpact),0)       AS `Total Cost ($)`
FROM factInspection i JOIN factPrediction p ON p.inspectionId=i.inspectionId
WHERE p.modelName='resnet18'
GROUP BY i.machineId ORDER BY i.machineId
