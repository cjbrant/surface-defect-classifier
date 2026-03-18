# Project Summary

- Project objective: classify steel surface defects from annotated NEU-DET images and export structured evaluation artifacts for downstream manufacturing analysis.
- Modeling approach: convert Pascal VOC annotations to defect chips, fine tune a ResNet-18 classifier, and summarize performance with per class and confusion matrix outputs.
- Key metrics: validation accuracy, macro and weighted F1, per class precision and recall, and normalized confusion patterns.
- Limitations: the current workflow treats localization as given by annotations and evaluates at chip level rather than full image detection; hyperparameter search is limited.
- Possible extensions: add full object detection, class balancing experiments, confidence calibration, and deployment oriented inference packaging.
