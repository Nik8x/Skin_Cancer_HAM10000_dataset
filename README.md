# Skin Cancer HAM10000 - 7-Class Lesion Classification

A staged image-classification project on HAM10000, 10,015 dermatoscopic
skin lesion images across 7 diagnostic categories (melanocytic nevi,
melanoma, benign keratosis-like lesions, basal cell carcinoma, actinic
keratoses, vascular lesions, dermatofibroma) - a from-scratch CNN,
class-imbalance handling, transfer learning, Grad-CAM interpretability,
and unsupervised embedding clustering, each stage building on the last.

**Live report:** https://nik8x.github.io/Skin_Cancer_HAM10000_dataset/

## Data

Images come from [DermaMNIST](https://zenodo.org/records/10519652), a
128x128 standardized derivative of the HAM10000/ISIC 2018 source images
with an official stratified train/val/test split (7,007 / 1,003 / 2,005).
Demographic metadata (age, sex, lesion site, diagnosis confirmation method)
comes from the [Harvard Dataverse HAM10000 release](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T).
Neither is committed to the repo - `00_data_setup_eda.ipynb` downloads both
automatically on first run.

## Notebooks

| Notebook | What it covers |
|---|---|
| `00_data_setup_eda.ipynb` | Downloads the data, checks class balance (`nv` ~67%, `df`/`vasc` ~1% each) and demographics, shows example images per class. |
| `01_baseline_cnn.ipynb` | A 4-block CNN trained from scratch, with an explicit image/label alignment check and one model object trained straight through to evaluation. Confusion matrix and per-class precision/recall/F1 against the majority-class baseline. |
| `02_class_imbalance.ipynb` | Class weights and augmentation to address the imbalance - including diagnosing and fixing a real training collapse caused by using raw inverse-frequency weights directly. |
| `03_transfer_learning.ipynb` | Fine-tunes a pretrained MobileNetV2 instead of training from scratch (frozen head, then partial unfreeze), compared against every earlier stage. |
| `04_evaluation_gradcam.ipynb` | Full evaluation of the transfer-learning model (confusion matrix, per-class ROC/AUC) plus Grad-CAM visualizations of which image regions drove each prediction. |
| `05_embedding_clustering.ipynb` | Clusters the trained model's own image embeddings (KMeans, Gaussian mixture) without using the labels, then checks how well the clusters line up with the real diagnoses. |

Run them in order (00 → 05) - each stage loads the previous stage's saved
model/results from `outputs/`.

## Setup

CPU-only friendly - no GPU required, though the CNN training stages each
take roughly 20-45 minutes on a single CPU core.

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
jupyter notebook
```

## Key findings

- **Fixing the correctness bugs alone barely moved the needle.** A clean
  CNN with a verified image/label alignment and no train/eval mismatch
  reached 70.4% test accuracy against a 66.9% majority-class baseline -
  real, but marginal. Per-class recall showed why: 96.3% on `nv`, 0% on
  both `df` and `vasc`. Overall accuracy alone was hiding a model that had
  mostly learned to guess the majority class.
- **Naive class weighting can break training before it helps.** Raw
  inverse-frequency class weights (a ~57x spread between the rarest and
  most common class) caused a full training collapse - loss stuck at
  `ln(7)`, the exact value of a uniform random guess. Tempering the
  weights with a square root fixed it outright and raised macro-F1 from
  0.26 to 0.37.
- **Transfer learning was the real turning point.** Fine-tuning a
  pretrained MobileNetV2 matched the from-scratch baseline's ~70% overall
  accuracy but pushed macro-F1 to 0.53, with every one of the 7 classes -
  including `df`, stuck at 0% recall in every earlier attempt - reaching
  above 50% recall.
- **Per-class AUC averaged 0.92**, and Grad-CAM visualizations generally
  show the model attending to the lesion itself rather than background,
  hair, or ruler markings - a useful sanity check distinct from any single
  accuracy number, though the heatmaps are coarse at this input resolution.
- **The model's own embeddings only weakly correspond to real diagnoses
  without label supervision.** Clustering them (KMeans, Gaussian mixture)
  found an adjusted Rand index of just ~0.03 against the true labels -
  `nv`'s sheer volume dominates the embedding space more than any clean
  separation between diagnostic categories.

Full detail, confusion matrices, ROC curves, and Grad-CAM examples are in
the notebooks and the [live report](https://nik8x.github.io/Skin_Cancer_HAM10000_dataset/).
