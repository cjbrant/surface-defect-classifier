# Surface Defect Classifier

Steel surface defect classification with an interactive Ignition Perspective dashboard for defect pattern analysis.

## What this is

At my workplace (films manufacturing), we have an inline vision system that detects defects on film and draws bounding boxes around them. It alarms when defects are too large, but it doesn't tell us *what* the defect was, just that something was there. Knowing the defect type matters because different defects have different root causes and different costs.

I built this as a proof of concept to show plant leadership how image classification could work alongside our existing alarm system. The classifier identifies what type of defect is in each image, and the dashboard shows patterns in defect types, estimated costs, location on the roll, and how defects affect KPIs.

I used a public steel defect dataset (NEU-DET) as a stand-in for our proprietary film data to demonstrate the concept.

## What it does

Classification pipeline:
- Parses Pascal VOC annotations from the NEU-DET dataset
- Crops defect regions into individual chips for classification
- Fine-tunes image classifiers (ResNet-18 and ViT-B/16) on 6 defect classes (crazing, inclusion, patches, pitted surface, rolled-in scale, scratches)
- Exports predictions, confusion matrices, and per-class performance reports

Results:
- ResNet-18 reaches 97.89% validation accuracy on 854 chips (0.979 macro F1)
- ViT-B/16 reaches 76.58%. It is data-hungry and underfits a set this small, which is a legitimate result to report
- Scratches classify perfectly (F1 = 1.0)
- The main confusion is pitted surface against rolled-in scale, since the two are texturally similar

## The dashboard

The classifier is the easy part. The real value is in what you do with the classification data, connecting defect types to process conditions, costs, and trends over time. I built an interactive Ignition Perspective dashboard (backed by MySQL) to do this. It is a seven-view application showing model quality, plant operations, a model comparison, where defects land on the strip, a live inspection feed, a prediction explorer, and the chips the model got wrong.

The dashboard, its screenshots, and importable Ignition artifacts (project, gateway backup, MySQL dump) are in [`ignition_dashboard/`](ignition_dashboard/).

## How to run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Train the classifier (the NEU-DET dataset is included in NEU-DET/)
python train.py --model resnet18 --rebuild-chips

# Results are written to results/
```

## Tech stack

Python, PyTorch, torchvision (ResNet-18 and ViT-B/16 fine-tuning), PIL, pandas, scikit-learn. Dashboard built in Ignition Perspective on MySQL.

## Structure

- `train.py` runs the end-to-end training pipeline (chip extraction, augmentation, training, evaluation)
- `NEU-DET/` holds the dataset (train/validation split with images and annotations)
- `results/` holds predictions, confusion matrices, and class reports
- `ignition_dashboard/` holds the Ignition Perspective dashboard (project source, exports, ETL, screenshots)
