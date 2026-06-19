SELECT metricName, ROUND(metricValue, 4) AS metricValue
FROM metricOverall
WHERE modelName = 'resnet18'
ORDER BY FIELD(metricName, 'accuracy', 'macroF1', 'weightedF1', 'totalSupport')
