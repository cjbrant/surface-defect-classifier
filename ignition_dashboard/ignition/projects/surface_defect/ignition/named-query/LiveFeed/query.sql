SELECT DATE_FORMAT(inspectedTs, '%H:%i:%s') AS `Time`,
       CASE predLabel WHEN 'crazing' THEN 'Crazing' WHEN 'inclusion' THEN 'Inclusion' WHEN 'patches' THEN 'Patches' WHEN 'pitted_surface' THEN 'Pitted Surface' WHEN 'rolled-in_scale' THEN 'Rolled-in Scale' WHEN 'scratches' THEN 'Scratches' ELSE predLabel END             AS `Predicted`,
       CASE trueLabel WHEN 'crazing' THEN 'Crazing' WHEN 'inclusion' THEN 'Inclusion' WHEN 'patches' THEN 'Patches' WHEN 'pitted_surface' THEN 'Pitted Surface' WHEN 'rolled-in_scale' THEN 'Rolled-in Scale' WHEN 'scratches' THEN 'Scratches' ELSE trueLabel END             AS `Actual`,
       ROUND(confidence,3)                  AS `Confidence`,
       IF(isCritical=1,'CRITICAL','')       AS `Severity`,
       machineId                            AS `Machine`,
       shiftCode                            AS `Shift`
FROM liveInspection WHERE modelName='resnet18' ORDER BY id DESC LIMIT 20
