SELECT CAST(metricValue AS UNSIGNED)
FROM metricOverall
WHERE modelName = 'resnet18' AND metricName = 'totalSupport'
