SELECT CASE predLabel WHEN 'crazing' THEN 'Crazing' WHEN 'inclusion' THEN 'Inclusion' WHEN 'patches' THEN 'Patches' WHEN 'pitted_surface' THEN 'Pitted Surface' WHEN 'rolled-in_scale' THEN 'Rolled-in Scale' WHEN 'scratches' THEN 'Scratches' ELSE predLabel END                                   AS `Predicted Class`,
       cnt                                                        AS `Count`,
       ROUND(100*cnt/SUM(cnt) OVER (),1)                          AS `% of Total`,
       ROUND(100*SUM(cnt) OVER (ORDER BY cnt DESC)/SUM(cnt) OVER (),1) AS `Cumulative %`
FROM (SELECT p.predLabel, COUNT(*) cnt FROM factPrediction p
      WHERE p.modelName='resnet18' GROUP BY p.predLabel) q
ORDER BY cnt DESC
