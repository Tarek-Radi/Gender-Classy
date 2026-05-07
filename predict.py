from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

from config import (
    BOTH_THRESHOLD,
    DICTIONARY_FILE,
    LABEL_ENCODER_FILE,
    MIN_CONFIDENCE,
    MODEL_FILE,
    PREDICTION_COLUMNS,
    UNISEX_MARGIN,
    VECTORIZER_FILE,
)
from preprocessing import normalize_name


LABEL_UI = {"male": "Male", "female": "Female", "both": "Both / Unisex"}


@dataclass
class Artifacts:
    model: object
    vectorizer: object
    label_encoder: object
    dictionary_lookup: Dict[str, dict]


_ARTIFACTS_CACHE: Optional[Artifacts] = None


def load_artifacts(force_reload: bool = False) -> Artifacts:
    global _ARTIFACTS_CACHE
    if _ARTIFACTS_CACHE is not None and not force_reload:
        return _ARTIFACTS_CACHE

    required_files = [MODEL_FILE, VECTORIZER_FILE, LABEL_ENCODER_FILE, DICTIONARY_FILE]
    missing = [str(path) for path in required_files if not Path(path).exists()]
    if missing:
        raise FileNotFoundError(
            "Model files were not found. Please run `python train_model.py` first.\nMissing: "
            + ", ".join(missing)
        )

    _ARTIFACTS_CACHE = Artifacts(
        model=joblib.load(MODEL_FILE),
        vectorizer=joblib.load(VECTORIZER_FILE),
        label_encoder=joblib.load(LABEL_ENCODER_FILE),
        dictionary_lookup=joblib.load(DICTIONARY_FILE),
    )
    return _ARTIFACTS_CACHE


def _dict_entry_to_probs(entry: dict) -> Dict[str, float]:
    male = float(entry.get("male_score", 0.0))
    female = float(entry.get("female_score", 0.0))
    both = float(entry.get("both_score", 0.0))
    total = male + female + both
    if total <= 0:
        return {"male": 1 / 3, "female": 1 / 3, "both": 1 / 3}
    return {"male": male / total, "female": female / total, "both": both / total}


def _ml_probs(normalized_name: str, artifacts: Artifacts) -> Dict[str, float]:
    vector = artifacts.vectorizer.transform([normalized_name])
    proba = artifacts.model.predict_proba(vector)[0]
    class_labels = artifacts.label_encoder.inverse_transform(np.arange(len(proba)))
    probs = {label: float(prob) for label, prob in zip(class_labels, proba)}
    return {
        "male": float(probs.get("male", 0.0)),
        "female": float(probs.get("female", 0.0)),
        "both": float(probs.get("both", 0.0)),
    }


def _resolve_prediction(probs: Dict[str, float]) -> tuple[str, float]:
    male = probs["male"]
    female = probs["female"]
    both = probs["both"]
    confidence = max(male, female, both)

    if confidence < MIN_CONFIDENCE:
        return "both", confidence

    if abs(male - female) <= UNISEX_MARGIN:
        return "both", 1.0 - abs(male - female)

    if both >= BOTH_THRESHOLD and both >= max(male, female):
        return "both", both

    if male >= female:
        return "male", male
    return "female", female


def predict_single_name(name: str) -> dict:
    artifacts = load_artifacts()
    normalized = normalize_name(name)

    if not normalized:
        return {
            "input_name": name,
            "normalized_name": "",
            "gender": "Both / Unisex",
            "gender_confidence": 0.0,
            "male_probability": 0.0,
            "female_probability": 0.0,
            "both_probability": 1.0,
            "prediction_source": "ML Model",
            "notes": "Input is empty or invalid after normalization.",
        }

    dict_entry = artifacts.dictionary_lookup.get(normalized)
    if dict_entry and float(dict_entry.get("confidence", 0.0)) >= MIN_CONFIDENCE:
        probs = _dict_entry_to_probs(dict_entry)
        label, confidence = _resolve_prediction(probs)
        source_summary = dict_entry.get("source_summary", "Local dictionary")
        note = f"Matched dictionary lookup ({source_summary}) with strong confidence."
        prediction_source = "Dictionary"
    else:
        model_probs = _ml_probs(normalized, artifacts)
        if dict_entry:
            dict_probs = _dict_entry_to_probs(dict_entry)
            probs = {
                "male": 0.65 * model_probs["male"] + 0.35 * dict_probs["male"],
                "female": 0.65 * model_probs["female"] + 0.35 * dict_probs["female"],
                "both": 0.65 * model_probs["both"] + 0.35 * dict_probs["both"],
            }
            label, confidence = _resolve_prediction(probs)
            note = "Used blended prediction from dictionary and ML model due weak dictionary confidence."
            prediction_source = "Mixed"
        else:
            probs = model_probs
            label, confidence = _resolve_prediction(probs)
            note = "Used ML model prediction."
            prediction_source = "ML Model"

    return {
        "input_name": name,
        "normalized_name": normalized,
        "gender": LABEL_UI.get(label, "Both / Unisex"),
        "gender_confidence": float(confidence),
        "male_probability": float(probs["male"]),
        "female_probability": float(probs["female"]),
        "both_probability": float(probs["both"]),
        "prediction_source": prediction_source,
        "notes": note,
    }


def predict_many_names(names: List[str]) -> pd.DataFrame:
    rows = [predict_single_name(name) for name in names]
    return pd.DataFrame(rows)


def predict_dataframe(df: pd.DataFrame, name_column: str) -> pd.DataFrame:
    if name_column not in df.columns:
        raise KeyError(f"Column '{name_column}' not found in uploaded data.")

    predictions = predict_many_names(df[name_column].astype(str).tolist())
    result = df.copy()
    result["gender"] = predictions["gender"]
    result["gender_confidence"] = predictions["gender_confidence"]
    result["male_probability"] = predictions["male_probability"]
    result["female_probability"] = predictions["female_probability"]
    result["both_probability"] = predictions["both_probability"]
    result["prediction_source"] = predictions["prediction_source"]
    result["normalized_name"] = predictions["normalized_name"]
    return result


def prediction_columns() -> List[str]:
    return PREDICTION_COLUMNS.copy()
