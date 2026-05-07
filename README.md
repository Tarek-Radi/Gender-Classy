# GenderClassy
A production-style Streamlit application for **name-based gender prediction** using a **dictionary-first hybrid system**.

## Live Demo
- Deployed app: [tarek-radi-gender-classy-app-80hycp.streamlit.app](https://tarek-radi-gender-classy-app-80hycp.streamlit.app/)

## Table of Contents
- [Project Summary](#project-summary)
- [Core Features](#core-features)
- [System Architecture](#system-architecture)
- [Model and Algorithms](#model-and-algorithms)
- [Evaluation Strategy](#evaluation-strategy)
- [Latest Metrics](#latest-metrics)
- [Dataset Coverage and Row Counts](#dataset-coverage-and-row-counts)
- [Project Structure](#project-structure)
- [Expected Data Files](#expected-data-files)
- [Installation](#installation)
- [Train the Model](#train-the-model)
- [Run the App](#run-the-app)
- [How to Use](#how-to-use)
- [Output Schema](#output-schema)
- [Tech Stack](#tech-stack)
- [Limitations and Responsible Use](#limitations-and-responsible-use)
- [Troubleshooting](#troubleshooting)

## Project Summary
GenderClassy predicts one of three labels from a name:
- `Male`
- `Female`
- `Both / Unisex`

The system is designed for practical real-world workflow support:
- Single name prediction
- Batch text prediction
- CSV/Excel upload and return of the same file with appended prediction columns

All training uses only local files inside `data/`.

## Core Features
- Dictionary-first prediction with confidence-based fallback to ML
- Unicode-aware normalization (English, Arabic, accented names)
- Batch parsing from new lines, commas, and semicolons
- CSV/Excel upload and downloadable enriched output
- UTF-8 BOM CSV export for strong Excel compatibility
- Robust handling for empty names, nulls, invalid rows, and large files
- Clean Streamlit UI with model status and dataset visibility

## System Architecture
1. Data ingestion from multiple local WGND/UCI-style files
2. Name normalization and label standardization (`male`, `female`, `both`)
3. Weighted aggregation and conflict resolution by normalized name
4. Dictionary artifact generation (`dictionary_lookup.joblib`)
5. ML fallback training (character-level TF-IDF + Logistic Regression)
6. Inference pipeline:
   - dictionary-first when confidence is strong
   - mixed or ML fallback when confidence/coverage is weak

## Model and Algorithms
### Dictionary Layer (Primary Inference)
- Weighted score aggregation per normalized name:
  - `male_score`
  - `female_score`
  - `both_score`
- Final label resolution with:
  - `UNISEX_MARGIN`
  - `BOTH_THRESHOLD`
  - `MIN_CONFIDENCE`

### ML Layer (Fallback Inference)
- Model name: `Character N-gram Name Gender Classifier`
- Vectorizer: `TfidfVectorizer(analyzer='char', ngram_range=(2,5), min_df=2, max_features=250000)`
- Classifier: `LogisticRegression(class_weight='balanced', solver='saga', max_iter=1000)`
- Classes: `male`, `female`, `both`

## Evaluation Strategy
The training pipeline reports **two separate evaluation modes** in `outputs/metrics.json`:

1. `ml_fallback_model_metrics`
- Isolates the fallback classifier quality
- Reports accuracy, macro F1, weighted F1, per-class precision/recall/F1, and confusion matrix

2. `hybrid_system_metrics`
- Evaluates the actual production behavior (**dictionary-first + ML fallback**)
- Reports accuracy, macro F1, weighted F1, per-class precision/recall/F1, confusion matrix
- Also reports:
  - dictionary hit rate
  - ML fallback rate

Important interpretation:
- The deployed app is **dictionary-first**.
- The ML model is only a **fallback** for low-confidence or uncovered names.
- Hybrid accuracy can be very high when dictionary coverage is extensive.

## Latest Metrics
From the latest generated `outputs/metrics.json`:

### Full Hybrid System (Dictionary-first + ML fallback)
- Accuracy: **0.9971333333333333** (**99.71%**)
- Dictionary hit rate: **99.53%**


### ML Fallback Model Only
- Accuracy: **0.353175** (**35.32%**)

These two numbers are expected to be different because they evaluate different inference modes.

## Dataset Coverage and Row Counts
- All files total: **61,711,386** lines
- Usable training files only: **61,711,124** lines
- Unified dataset after cleaning/merge/dedup: **4,000,698** rows
- Rows used to train ML fallback model: **600,000**
- Train split: **480,000**
- Test split: **120,000**

## Project Structure
```text
GenderClassy/
|-- app.py
|-- train_model.py
|-- predict.py
|-- data_loader.py
|-- preprocessing.py
|-- config.py
|-- requirements.txt
|-- README.md
|-- data/
|-- models/
`-- outputs/
```


## Installation
```bash
pip install -r requirements.txt
```

## Train the Model
```bash
python train_model.py
```

Generated artifacts (`models/`):
- `gender_model.joblib`
- `vectorizer.joblib`
- `label_encoder.joblib`
- `dictionary_lookup.joblib`

Generated reports (`outputs/`):
- `metrics.json`
- `training_report.txt`

## Run the App
```bash
streamlit run app.py
```
If `streamlit` is not on PATH:
```bash
python -m streamlit run app.py
```

## How to Use
### 1) Single Name Prediction
- Enter one name
- Click `Predict`
- Review prediction label, confidence, probabilities, source, and notes

### 2) Batch Text Prediction
- Paste names separated by new lines, commas, or semicolons
- Click `Predict All`
- Download CSV or Excel

### 3) Upload CSV / Excel
- Upload `.csv`, `.xlsx`, or `.xls`
- Select the column containing names
- Click `Add Gender Column`
- Download enriched files:
  - `originalfilename_with_gender.csv`
  - `originalfilename_with_gender.xlsx`

## Output Schema
For uploaded files, these columns are appended:
- `gender`
- `gender_confidence`
- `male_probability`
- `female_probability`
- `both_probability`
- `prediction_source`
- `normalized_name`

Batch predictions also include:
- `input_name`
- `notes`

## Tech Stack
### Language
- Python 3

### ML and Data
- pandas
- numpy
- scikit-learn
- joblib

### App/UI
- Streamlit
- Custom CSS

### Spreadsheet and File IO
- openpyxl
- xlsxwriter
- xlrd
- io.BytesIO

### Utilities
- pathlib
- regex/unicode normalization
- tqdm
- python-dotenv
- pyarrow (optional)

## Limitations and Responsible Use
- This is probabilistic inference, not ground truth.
- Name-gender usage varies by language, culture, country, and time.
- Some names are genuinely unisex depending on context.
- Do not use for sensitive, legal, hiring, medical, or high-stakes decisions.

## Troubleshooting
- Model files missing:
  - Run `python train_model.py`
- `data/` folder missing/incomplete:
  - Verify expected files exist
- CSV encoding issues:
  - Prefer UTF-8/UTF-8-BOM input
- Large-file runtime is slow:
  - Keep app running and wait for processing to finish
