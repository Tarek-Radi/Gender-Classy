# GenderClassy - Name Gender Predictor

## 1. Project Overview
GenderClassy is a local-first machine learning project that predicts likely gender from names:
- Male
- Female
- Both / Unisex

It combines:
- A weighted dictionary lookup built from local WGND + UCI datasets
- A character-level ML model (TF-IDF + Logistic Regression) as fallback

## 2. Features
- Single name prediction with confidence and class probabilities
- Batch text prediction (new lines, commas, semicolons)
- CSV/Excel upload with row-wise gender prediction
- Returns original uploaded data plus appended prediction columns
- CSV (UTF-8 BOM) and Excel downloads
- Arabic and Unicode-aware name normalization
- Friendly UI with model status and error handling

## 3. Folder Structure
```text
GenderClassy/
├── app.py
├── train_model.py
├── predict.py
├── data_loader.py
├── preprocessing.py
├── config.py
├── requirements.txt
├── README.md
├── data/
├── models/
└── outputs/
```

## 4. Dataset Files Expected
Place these local files in `data/`:
- `name_gender_1950-2018.csv`
- `name_gender_all.csv`
- `name_gender_dataset.csv`
- `wgnd_2_0_code-langcode.csv`
- `wgnd_2_0_name-gender_nocode.csv`
- `wgnd_2_0_name-gender-code.csv`
- `wgnd_2_0_name-gender-code_langexp.csv`
- `wgnd_2_0_name-gender-langcode.csv`
- `wgnd_2_0_sources.csv`

Notes:
- Files with usable name/gender columns are used for training.
- Metadata-only files are safely skipped with logged reasons.

## 5. Install Dependencies
```bash
pip install -r requirements.txt
```

## 6. Train the Model
```bash
python train_model.py
```

This will:
- Inspect and log local data schemas
- Build unified weighted training data
- Train TF-IDF + Logistic Regression
- Save artifacts to `models/`
- Save metrics and report to `outputs/`

## 7. Run Streamlit App
```bash
streamlit run app.py
```

## 8. Single-Name Prediction
In tab **Single Name Prediction**:
1. Enter one name.
2. Click **Predict**.
3. View:
   - Predicted gender
   - Confidence
   - Male/Female/Both probabilities
   - Prediction source (Dictionary / ML Model / Mixed)
   - Notes

## 9. Batch Prediction
In tab **Batch Text Prediction**:
1. Paste names separated by new lines, commas, or semicolons.
2. Click **Predict All**.
3. Download results as CSV or Excel.

## 10. CSV/Excel Upload Workflow
In tab **Upload CSV / Excel**:
1. Upload `.csv`, `.xlsx`, or `.xls`.
2. Select the column containing names.
3. Click **Add Gender Column**.
4. Download:
   - `originalfilename_with_gender.csv`
   - `originalfilename_with_gender.xlsx`

Original columns remain unchanged, new columns are appended.

## 11. Output Columns
Appended prediction columns:
- `gender`
- `gender_confidence`
- `male_probability`
- `female_probability`
- `both_probability`
- `prediction_source`
- `normalized_name`

Batch text output also includes:
- `input_name`
- `notes`

## 12. Model Approach
- Unicode-safe normalization (`casefold`, punctuation cleanup, first-token extraction)
- Label standardization to: `male`, `female`, `both`
- Weighted aggregation by normalized name
- Conflict resolution with configurable unisex thresholds
- Dictionary-first inference, ML fallback, and mixed blending when needed

## 13. Limitations and Bias Disclaimer
- This is probabilistic name-based inference, not ground truth.
- Gender association of names varies by culture, language, and region.
- Many names are context-dependent or genuinely unisex.
- Do not use this model for sensitive, legal, medical, hiring, or other high-stakes decisions.

## 14. Troubleshooting
- **Model files missing**:
  - Run `python train_model.py` first.
- **`data/` folder missing or empty**:
  - Verify local dataset files exist under `data/`.
- **CSV encoding issues**:
  - Try UTF-8/UTF-8-BOM input; app attempts fallback encodings automatically.
- **Large file processing is slow**:
  - Keep app running; prediction is row-wise and may take time on very large files.
- **Arabic text appears broken in Excel CSV**:
  - Use the app CSV download (UTF-8 BOM) or Excel `.xlsx` download.
