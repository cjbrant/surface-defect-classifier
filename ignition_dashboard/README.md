# Surface Defect Classifier

This project classifies steel surface defects from the NEU-DET benchmark at the chip level, then
feeds the results into an interactive Ignition Perspective dashboard backed by MySQL. The
pipeline parses the Pascal VOC annotations, crops each defect region into a chip, fine-tunes two image
classifiers (ResNet-18 and ViT-B/16), and exports the evaluation artifacts. An ETL job loads
those into a MySQL star schema, and a seven-view Perspective application reads from it live.

---

## The Dashboard

The dashboard is a seven-view Perspective application that reads from MySQL live. The views cover
model quality, plant operations, model comparison, defect localization, a replayed real-time feed, a
filterable prediction explorer, and the chips the model got wrong.

### Model Quality
Accuracy and F1 KPI tiles, plus a per-class precision/recall/F1 table. All of it reads from the database.

![Model Quality](../docs/screenshots/01-model-quality.png)

### Operations
Per-machine defect counts and cost, a predicted-class Pareto bar chart, and defect volume by day. This
is the MES-style view a line supervisor would watch.

![Operations](../docs/screenshots/02-operations.png)

### Model Comparison
ResNet-18 against ViT-B/16 as grouped horizontal bars for each class, sitting above the live
overall-metrics table. The ViT shortfall is easy to see (Pitted Surface drops to 0.57 against
ResNet's 0.97).

![Model Comparison](../docs/screenshots/03-model-comparison.png)

### Defect Map
A 10x10 density heatmap of where defects land on the strip. Each defect's bounding-box center is
binned by web (across) and roll (along the coil). The center carries most of the defects and the
edges stay clean.

![Defect Map](../docs/screenshots/04-defect-map.png)

### Live Inspection
A replayed real-time defect feed (polls every 2s) with running per-class counts and a critical rate.

![Live Inspection](../docs/screenshots/05-live-inspection.png)

### Prediction Explorer
All 854 validation predictions, sortable and filterable.

![Prediction Explorer](../docs/screenshots/06-prediction-explorer.png)

### Chip Viewer
The actual chips ResNet-18 got wrong, ordered by confidence. These are the cases I'd bring to a model
review.

![Chip Viewer](../docs/screenshots/07-chip-viewer.png)

> Want to run it yourself? The dashboard ships as importable Ignition artifacts (a project `.zip`,
> a full gateway `.gwbk`, and a MySQL dump) under [`exports/`](exports/).
> The import steps are in [`IMPORT.md`](IMPORT.md).

---

## Models & Results

I fine-tuned both models on roughly 3,300 training chips and evaluated them on 854 validation chips.

| Metric | ResNet-18 | ViT-B/16 |
| --- | ---: | ---: |
| Accuracy | 0.9789 | 0.7658 |
| Macro F1 | 0.9785 | 0.7431 |
| Weighted F1 | 0.9789 | 0.7647 |

ResNet-18 wins clearly. ViT-B/16 is data-hungry and underfits a set this small, which is a legitimate
result to report rather than a failure.

ResNet-18 per-class:

| Class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| Crazing | 0.964 | 1.000 | 0.982 | 162 |
| Inclusion | 0.981 | 0.969 | 0.975 | 159 |
| Patches | 0.984 | 0.974 | 0.979 | 193 |
| Pitted Surface | 0.945 | 0.989 | 0.966 | 87 |
| Rolled-in Scale | 0.992 | 0.947 | 0.969 | 132 |
| Scratches | 1.000 | 1.000 | 1.000 | 121 |

The one remaining weak spot is a small pitted-surface / rolled-in-scale confusion (the two are
texturally similar). It shows up as the top mistake in the Chip Viewer above, where a pitted-surface
chip gets predicted as rolled-in scale at 0.99 confidence.

---

## Architecture

```text
 NEU-DET images + VOC XML
            |
            v
        train.py  --model {resnet18, vit_b_16}
            |
   chip extraction + fine-tuning
            |
            v
   results/<model>/  (predictions, confusion, class report)
            |
            v
   ignition_dashboard/etl/load_mysql.py  -->  MySQL star schema
            |                                  (+ synthetic ops: machine/operator/shift/cost/position)
            v
   Ignition Perspective  <-- named queries --  MySQL
   (7 live views)
            ^
            \-- cropped chip PNGs served statically to the Chip Viewer
```

The dashboard ships as exported artifacts rather than a running service. The layout of this folder:

```
ignition_dashboard/
  exports/                 # importable deliverables
    surface_defect_project.zip      # Perspective project (views + named queries)
    surface_defect_gateway.gwbk     # full gateway backup (project + DB connection)
    surfaceDefect.sql               # mysqldump: schema + data
  ignition/projects/...    # version-controlled project source
  mysql/01_schema.sql      # star schema
  etl/                     # load_mysql.py, enrich.py, replay.py
  IMPORT.md                # how to stand it up on your own gateway
```

---

## Training (reproducibility)

The NEU-DET dataset is included in `NEU-DET/`, so training runs without a separate download.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python train.py --model resnet18 --rebuild-chips   # build chips + train ResNet-18
python train.py --model vit_b_16                    # train ViT-B/16 (reuses chips)
```

`train.py` picks the device automatically (`cuda`, then `mps`, then `cpu`). Outputs land in
`results/<model>/` per model, and the shared chips go to `results/chips/` with index files at
`results/*_chips_index.csv`.

The six NEU-DET classes are Crazing, Inclusion, Patches, Pitted Surface, Rolled-in Scale, and Scratches.

---

## Limitations

- ViT-B/16 underperforms on this small dataset. A larger corpus would change the comparison.
- The factory-operations layer (machine, operator, shift, cost, timeline, strip position) in the
  dashboard is synthetic. I generated it deterministically to demonstrate an MES-style inspection
  view, and it is not real plant data.
- The benchmark may not represent plant-specific lighting, coating, or imaging conditions.
- Localization is treated as preprocessing, not as a jointly learned detection task.
```
