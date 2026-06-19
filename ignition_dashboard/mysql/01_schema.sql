-- Surface Defect Classifier — MySQL star schema
-- Loaded automatically on first container boot (mounted into /docker-entrypoint-initdb.d).
-- Convention: camelCase identifiers. Class-label VALUES (e.g. 'rolled-in_scale') stay verbatim.

CREATE DATABASE IF NOT EXISTS surfaceDefect
  CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci;
USE surfaceDefect;

-- ============================================================
-- Dimensions
-- ============================================================

CREATE TABLE dimModel (
  modelName    VARCHAR(32)  NOT NULL PRIMARY KEY,
  displayName  VARCHAR(64)  NOT NULL,
  architecture VARCHAR(64)  NOT NULL,
  paramsM      DECIMAL(6,2) NULL,
  trainedTs    DATETIME     NULL,
  notes        VARCHAR(255) NULL
);

CREATE TABLE dimDefectClass (
  classId         INT          NOT NULL PRIMARY KEY,
  className       VARCHAR(32)  NOT NULL UNIQUE,
  isCriticalClass TINYINT(1)   NOT NULL DEFAULT 0,
  baseUnitCost    DECIMAL(8,2) NOT NULL DEFAULT 0
);

CREATE TABLE dimMachine (
  machineId   VARCHAR(16) NOT NULL PRIMARY KEY,
  machineName VARCHAR(64) NOT NULL
);

CREATE TABLE dimOperator (
  operatorId   VARCHAR(16) NOT NULL PRIMARY KEY,
  operatorName VARCHAR(64) NOT NULL
);

CREATE TABLE dimShift (
  shiftCode CHAR(1)     NOT NULL PRIMARY KEY,
  label     VARCHAR(32) NOT NULL
);

-- ============================================================
-- Facts
-- ============================================================

-- One row per validation chip — model-INDEPENDENT (operational context).
CREATE TABLE factInspection (
  inspectionId      INT          NOT NULL AUTO_INCREMENT PRIMARY KEY,
  chipPath          VARCHAR(512) NOT NULL,
  sourceImage       VARCHAR(512) NOT NULL,
  bboxIndex         INT          NOT NULL,
  bboxXmin          INT          NOT NULL,
  bboxYmin          INT          NOT NULL,
  bboxXmax          INT          NOT NULL,
  bboxYmax          INT          NOT NULL,
  positionAcrossWeb DECIMAL(6,5) NOT NULL,
  positionAlongRoll DECIMAL(6,5) NOT NULL,
  trueLabel         VARCHAR(32)  NOT NULL,
  machineId         VARCHAR(16)  NOT NULL,
  operatorId        VARCHAR(16)  NOT NULL,
  shiftCode         CHAR(1)      NOT NULL,
  inspectedTs       DATETIME     NOT NULL,
  isRepeat          TINYINT(1)   NOT NULL DEFAULT 0,
  isCritical        TINYINT(1)   NOT NULL DEFAULT 0,
  costImpact        DECIMAL(10,2) NOT NULL DEFAULT 0,
  UNIQUE KEY uqChip (chipPath),
  KEY ixInspectedTs (inspectedTs),
  KEY ixMachine (machineId),
  KEY ixTrueLabel (trueLabel),
  CONSTRAINT fkInspMachine  FOREIGN KEY (machineId)  REFERENCES dimMachine (machineId),
  CONSTRAINT fkInspOperator FOREIGN KEY (operatorId) REFERENCES dimOperator (operatorId),
  CONSTRAINT fkInspShift    FOREIGN KEY (shiftCode)  REFERENCES dimShift (shiftCode),
  CONSTRAINT fkInspClass    FOREIGN KEY (trueLabel)  REFERENCES dimDefectClass (className)
);

-- One row per (chip x model) — model-DEPENDENT prediction.
CREATE TABLE factPrediction (
  predictionId INT         NOT NULL AUTO_INCREMENT PRIMARY KEY,
  inspectionId INT         NOT NULL,
  modelName    VARCHAR(32) NOT NULL,
  predLabel    VARCHAR(32) NOT NULL,
  confidence   DECIMAL(9,8) NOT NULL,
  isCorrect    TINYINT(1)  NOT NULL,
  UNIQUE KEY uqInspModel (inspectionId, modelName),
  KEY ixModelPred (modelName, predLabel),
  CONSTRAINT fkPredInsp  FOREIGN KEY (inspectionId) REFERENCES factInspection (inspectionId) ON DELETE CASCADE,
  CONSTRAINT fkPredModel FOREIGN KEY (modelName)    REFERENCES dimModel (modelName)
);

-- Long-form class probabilities (chip x model x class).
CREATE TABLE factClassProbability (
  predictionId INT          NOT NULL,
  className    VARCHAR(32)  NOT NULL,
  probability  DECIMAL(9,8) NOT NULL,
  PRIMARY KEY (predictionId, className),
  CONSTRAINT fkProbPred FOREIGN KEY (predictionId) REFERENCES factPrediction (predictionId) ON DELETE CASCADE
);

-- ============================================================
-- Metrics (static evaluation — describes the model)
-- ============================================================

CREATE TABLE metricOverall (
  modelName   VARCHAR(32)   NOT NULL,
  metricName  VARCHAR(32)   NOT NULL,
  metricValue DECIMAL(12,6) NOT NULL,
  PRIMARY KEY (modelName, metricName),
  CONSTRAINT fkOverallModel FOREIGN KEY (modelName) REFERENCES dimModel (modelName)
);

CREATE TABLE metricClassReport (
  modelName      VARCHAR(32)  NOT NULL,
  className      VARCHAR(32)  NOT NULL,
  precisionScore DECIMAL(9,6) NOT NULL,
  recallScore    DECIMAL(9,6) NOT NULL,
  f1Score        DECIMAL(9,6) NOT NULL,
  support        INT          NOT NULL,
  PRIMARY KEY (modelName, className),
  CONSTRAINT fkReportModel FOREIGN KEY (modelName) REFERENCES dimModel (modelName)
);

CREATE TABLE metricConfusion (
  modelName     VARCHAR(32)  NOT NULL,
  trueLabel     VARCHAR(32)  NOT NULL,
  predLabel     VARCHAR(32)  NOT NULL,
  n             INT          NOT NULL,
  normalizedPct DECIMAL(9,6) NOT NULL,
  PRIMARY KEY (modelName, trueLabel, predLabel),
  CONSTRAINT fkConfModel FOREIGN KEY (modelName) REFERENCES dimModel (modelName)
);

-- ============================================================
-- Live replay feed (written by the Ignition Gateway timer script)
-- ============================================================

CREATE TABLE liveInspection (
  id                BIGINT       NOT NULL AUTO_INCREMENT PRIMARY KEY,
  inspectionId      INT          NOT NULL,
  modelName         VARCHAR(32)  NOT NULL,
  predLabel         VARCHAR(32)  NOT NULL,
  trueLabel         VARCHAR(32)  NOT NULL,
  confidence        DECIMAL(9,8) NOT NULL,
  isCritical        TINYINT(1)   NOT NULL,
  costImpact        DECIMAL(10,2) NOT NULL,
  machineId         VARCHAR(16)  NOT NULL,
  operatorId        VARCHAR(16)  NOT NULL,
  shiftCode         CHAR(1)      NOT NULL,
  positionAcrossWeb DECIMAL(6,5) NOT NULL,
  positionAlongRoll DECIMAL(6,5) NOT NULL,
  inspectedTs       DATETIME     NOT NULL,
  KEY ixLiveTs (inspectedTs),
  KEY ixLiveModel (modelName)
);
