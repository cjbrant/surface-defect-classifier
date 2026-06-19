SELECT CASE metricName WHEN 'accuracy' THEN 'Accuracy'
                       WHEN 'macroF1' THEN 'Macro F1'
                       WHEN 'weightedF1' THEN 'Weighted F1' END AS `Metric`,
       MAX(CASE WHEN modelName='resnet18' THEN ROUND(metricValue,4) END) AS `ResNet-18`,
       MAX(CASE WHEN modelName='vit_b_16' THEN ROUND(metricValue,4) END) AS `ViT-B/16`
FROM metricOverall WHERE metricName IN ('accuracy','macroF1','weightedF1')
GROUP BY metricName
