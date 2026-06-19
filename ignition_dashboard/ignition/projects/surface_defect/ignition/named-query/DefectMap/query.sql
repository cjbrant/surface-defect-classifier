SELECT i.positionAcrossWeb, i.positionAlongRoll, i.trueLabel, p.isCorrect
FROM factInspection i JOIN factPrediction p ON p.inspectionId=i.inspectionId
WHERE p.modelName='resnet18'
