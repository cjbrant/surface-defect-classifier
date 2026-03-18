# Surface Defect Classifier

Steel surface defect classification with a manufacturing analytics dashboard for defect pattern analysis.

## What this is

At my workplace (films manufacturing), we have an inline vision system that detects defects on film and draws bounding boxes around them. It alarms when defects are too large, but it doesn't tell us *what* the defect was; just that something was there. Knowing the defect type matters because different defects have different root causes and different costs.

I built this as a proof of concept to show plant leadership how image classification could work alongside our existing alarm system. The classifier identifies what type of defect is in each image, and the analytics dashboard shows patterns in defect types, estimated costs, location on the roll, and how defects affect KPIs.

I used a public steel defect dataset (NEU-DET) as a stand-in for our proprietary film data to demonstrate the concept.

## What it does

**Classification pipeline:**
- Parses Pascal VOC annotations from the NEU-DET dataset
- Crops defect regions into individual chips for classification
- Fine-tunes a ResNet-18 on 6 defect classes (crazing, inclusion, patches, pitted surface, rolled-in scale, scratches)
- Exports predictions, confusion matrices, and per-class performance reports

**Results:**
- 94.96% validation accuracy on 854 chips
- 0.948 macro F1 across all 6 classes
- Scratches: perfect classification (F1 = 1.0)
- Rolled-in scale: main failure mode (recall = 0.73) — these look similar to scratches and are the hardest to distinguish

## The bigger picture

The classifier is the easy part. The real value is in what you do with the classification data; connecting defect types to process conditions, costs, and trends over time. The Power BI dashboard (in /reports) demonstrates how classification outputs can feed into operational decision-making: which defects are most expensive, where on the roll they occur, and what process changes correlate with defect rate changes.

## How to run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Train the classifier (requires NEU-DET dataset in NEU-DET/ folder)
python train.py

# Results are written to results/
```

## Tech stack

Python, PyTorch, torchvision (ResNet-18 fine-tuning), PIL, pandas, scikit-learn. Dashboard built in Power BI.

## Structure

- `train.py` — end-to-end training pipeline (chip extraction, augmentation, training, evaluation)
- `NEU-DET/` — dataset (train/validation split with images and annotations)
- `results/` — predictions, confusion matrices, class reports, model checkpoints
