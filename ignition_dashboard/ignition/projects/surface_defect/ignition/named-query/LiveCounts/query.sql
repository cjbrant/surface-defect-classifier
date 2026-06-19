SELECT CASE predLabel WHEN 'crazing' THEN 'Crazing' WHEN 'inclusion' THEN 'Inclusion' WHEN 'patches' THEN 'Patches' WHEN 'pitted_surface' THEN 'Pitted Surface' WHEN 'rolled-in_scale' THEN 'Rolled-in Scale' WHEN 'scratches' THEN 'Scratches' ELSE predLabel END       AS `Predicted Class`,
       COUNT(*)                       AS `Count`,
       ROUND(100*AVG(isCritical),0)   AS `% Critical`
FROM liveInspection WHERE modelName='resnet18' GROUP BY predLabel ORDER BY COUNT(*) DESC
