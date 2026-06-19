SELECT CASE className WHEN 'crazing' THEN 'Crazing' WHEN 'inclusion' THEN 'Inclusion' WHEN 'patches' THEN 'Patches' WHEN 'pitted_surface' THEN 'Pitted Surface' WHEN 'rolled-in_scale' THEN 'Rolled-in Scale' WHEN 'scratches' THEN 'Scratches' ELSE className END AS `Class`,
       ROUND(precisionScore, 3) AS `Precision`,
       ROUND(recallScore, 3)    AS `Recall`,
       ROUND(f1Score, 3)        AS `F1`,
       support                  AS `Support`
FROM metricClassReport
WHERE modelName = 'resnet18'
ORDER BY className
