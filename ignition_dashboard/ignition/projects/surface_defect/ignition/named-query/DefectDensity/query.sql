SELECT CONCAT('Roll ',  FLOOR(i.positionAlongRoll*10)) AS `Roll Band`,
       CONCAT('Web ',   FLOOR(i.positionAcrossWeb*10))  AS `Web Band`,
       COUNT(*)                                         AS `Defect Count`,
       ROUND(100*AVG(i.isCritical),0)                   AS `% Critical`
FROM factInspection i JOIN factPrediction p ON p.inspectionId=i.inspectionId
WHERE p.modelName='resnet18'
GROUP BY `Roll Band`, `Web Band`
ORDER BY COUNT(*) DESC
