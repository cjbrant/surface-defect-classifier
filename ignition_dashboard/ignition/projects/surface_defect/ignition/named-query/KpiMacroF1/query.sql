SELECT ROUND(metricValue,3)
FROM metricOverall
WHERE modelName = 'resnet18' AND metricName = 'macroF1'
