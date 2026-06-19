SELECT CASE i.trueLabel WHEN 'crazing' THEN 'Crazing' WHEN 'inclusion' THEN 'Inclusion' WHEN 'patches' THEN 'Patches' WHEN 'pitted_surface' THEN 'Pitted Surface' WHEN 'rolled-in_scale' THEN 'Rolled-in Scale' WHEN 'scratches' THEN 'Scratches' ELSE i.trueLabel END AS `True Class`,
       CASE p.predLabel WHEN 'crazing' THEN 'Crazing' WHEN 'inclusion' THEN 'Inclusion' WHEN 'patches' THEN 'Patches' WHEN 'pitted_surface' THEN 'Pitted Surface' WHEN 'rolled-in_scale' THEN 'Rolled-in Scale' WHEN 'scratches' THEN 'Scratches' ELSE p.predLabel END AS `Predicted (wrong)`,
       ROUND(p.confidence,3) AS `Confidence`,
       i.machineId AS `Machine`
FROM factPrediction p JOIN factInspection i ON i.inspectionId=p.inspectionId
WHERE p.modelName='resnet18' AND p.isCorrect=0
ORDER BY p.confidence DESC
