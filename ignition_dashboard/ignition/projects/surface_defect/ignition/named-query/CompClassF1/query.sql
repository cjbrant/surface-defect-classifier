SELECT CASE className WHEN 'crazing' THEN 'Crazing' WHEN 'inclusion' THEN 'Inclusion' WHEN 'patches' THEN 'Patches' WHEN 'pitted_surface' THEN 'Pitted Surface' WHEN 'rolled-in_scale' THEN 'Rolled-in Scale' WHEN 'scratches' THEN 'Scratches' ELSE className END AS `Class`,
       MAX(CASE WHEN modelName='resnet18' THEN ROUND(f1Score,3) END) AS `ResNet-18 F1`,
       MAX(CASE WHEN modelName='vit_b_16' THEN ROUND(f1Score,3) END) AS `ViT-B/16 F1`,
       ROUND(MAX(CASE WHEN modelName='resnet18' THEN f1Score END)
           - MAX(CASE WHEN modelName='vit_b_16' THEN f1Score END),3) AS `Difference`
FROM metricClassReport GROUP BY className ORDER BY className
