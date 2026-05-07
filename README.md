# GenderClassy
Professional Streamlit application for predicting gender from names using a hybrid ML pipeline.

## Project Overview
GenderClassy predicts:
- `Male`
- `Female`
- `Both / Unisex`

It supports:
- Single-name prediction
- Batch text prediction
- CSV/Excel upload and return of the same file with appended prediction columns

The project is local-first: it trains only from files in `data/` and does not require external datasets.

## Model and Algorithms
### Hybrid Inference Engine
1. Weighted Dictionary Lookup (first stage)
- Built from merged WGND 2.0 + UCI-style local datasets
- Aggregates `male`, `female`, and `both` scores per normalized name
- Uses weighted voting and confidence rules

2. ML Fallback Classifier (second stage)
- Model name: `Character N-gram Name Gender Classifier`
- Vectorizer: `TfidfVectorizer(analyzer='char', ngram_range=(2,5), min_df=2, max_features=250000)`
- Classifier: `LogisticRegression(class_weight='balanced', solver='saga', max_iter=1000)`
- Classes: `male`, `female`, `both`

3. Mixed Mode
- If dictionary evidence is present but not strong enough, prediction can blend dictionary + ML signals.

## Latest Training Metrics
The training pipeline now reports **two separate evaluation modes** in `outputs/metrics.json`:

1. `ml_fallback_model_metrics`
- Evaluates only the ML classifier (TF-IDF + Logistic Regression)
- Includes:
  - accuracy
  - macro F1
  - weighted F1
  - per-class precision/recall/F1
  - confusion matrix

2. `hybrid_system_metrics`
- Evaluates the real app logic: **dictionary-first**, then ML fallback
- Full hybrid system accuracy (Dictionary-first + ML fallback): `0.9971333333333333` -> `99.71%`
- Includes:
  - accuracy
  - macro F1
  - weighted F1
  - per-class precision/recall/F1
  - confusion matrix
  - dictionary hit rate
  - ML fallback rate

Important interpretation:
- The app is designed to use dictionary predictions first when confidence is strong.
- The ML model is a fallback for names with weak/missing dictionary coverage.
- Hybrid metrics can be very high on in-catalog names because dictionary coverage is extensive; always inspect fallback rates and per-class behavior.
- Do not compare this project to high-accuracy demographic classifiers; name-based gender inference is inherently noisy and culturally dependent.

## Tech Stack
### Core Language
- Python 3

### Machine Learning and Data
- pandas
- numpy
- scikit-learn
- joblib

### Web App / UI
- Streamlit
- Custom CSS (inside Streamlit app)

### File and Spreadsheet Handling
- openpyxl
- xlsxwriter
- xlrd
- io.BytesIO

### Utility and Project Tooling
- pathlib
- regex / unicode normalization
- tqdm
- python-dotenv
- pyarrow (optional)

## Key Features
- Single-name prediction with:
  - gender label
  - confidence score
  - male/female/both probabilities
  - prediction source (`Dictionary`, `ML Model`, `Mixed`)
  - notes
- Batch input (newline/comma/semicolon separated names)
- Upload CSV/XLSX/XLS, choose name column, append predictions, download updated file
- UTF-8 BOM CSV export for better Arabic compatibility in Excel
- Unicode-aware preprocessing (English, Arabic, accented names)
- Friendly error handling and model status indicators

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

## Expected Data Files
Put these files in `data/`:
- `name_gender_1950-2018.csv`
- `name_gender_all.csv`
- `name_gender_dataset.csv`
- `wgnd_2_0_code-langcode.csv`
- `wgnd_2_0_name-gender_nocode.csv`
- `wgnd_2_0_name-gender-code.csv`
- `wgnd_2_0_name-gender-code_langexp.csv`
- `wgnd_2_0_name-gender-langcode.csv`
- `wgnd_2_0_sources.csv`

## Installation
```bash
pip install -r requirements.txt
```

## Train the Model
```bash
python train_model.py
```

Artifacts saved to `models/`:
- `gender_model.joblib`
- `vectorizer.joblib`
- `label_encoder.joblib`
- `dictionary_lookup.joblib`

Reports saved to `outputs/`:
- `metrics.json`
- `training_report.txt`

## Run the Streamlit App
Use one of the following:
```bash
streamlit run app.py
```
or
```bash
python -m streamlit run app.py
```

## Usage
### 1) Single Name Prediction
- Enter one name
- Click `Predict`
- Review gender, confidence, probabilities, source, and notes

### 2) Batch Text Prediction
- Paste names separated by new lines, commas, or semicolons
- Click `Predict All`
- Download results as CSV/Excel

### 3) Upload CSV / Excel
- Upload `.csv`, `.xlsx`, or `.xls`
- Select the name column
- Click `Add Gender Column`
- Download:
  - `originalfilename_with_gender.csv`
  - `originalfilename_with_gender.xlsx`

## Output Columns
Appended columns for file prediction:
- `gender`
- `gender_confidence`
- `male_probability`
- `female_probability`
- `both_probability`
- `prediction_source`
- `normalized_name`

Batch output also includes:
- `input_name`
- `notes`

## Limitations and Responsible Use
- This is probabilistic inference, not ground truth.
- Name-gender associations vary by culture, country, language, and time.
- Many names are truly unisex depending on context.
- Do not use this tool for sensitive or high-stakes decisions.

## Troubleshooting
- Model files missing:
  - Run `python train_model.py`
- `data/` folder missing or incomplete:
  - Verify required files are present
- CSV encoding issues:
  - Use UTF-8/UTF-8-BOM when possible
- Large files process slowly:
  - Wait for completion; prediction is row-wise
