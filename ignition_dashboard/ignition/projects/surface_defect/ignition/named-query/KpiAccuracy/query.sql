SELECT CONCAT(ROUND(metricValue*100,1), '%')
FROM metricOverall
WHERE modelName = 'resnet18' AND metricName = 'accuracy'
