SELECT DATE_FORMAT(i.inspectedTs, '%Y-%m-%d') AS `Day`,
       COUNT(*)                               AS `Defect Count`,
       ROUND(100*AVG(i.isCritical),1)         AS `% Critical`
FROM factInspection i JOIN factPrediction p ON p.inspectionId=i.inspectionId
WHERE p.modelName='resnet18'
GROUP BY DATE_FORMAT(i.inspectedTs, '%Y-%m-%d') ORDER BY `Day`
