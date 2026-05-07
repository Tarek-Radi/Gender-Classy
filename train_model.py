import json
import logging
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from config import (
    LABEL_ENCODER_FILE,
    MAX_FEATURES,
    MAX_MODEL_TRAIN_ROWS,
    METRICS_FILE,
    MODEL_FILE,
    RANDOM_STATE,
    REPORT_FILE,
    TEST_SIZE,
    VECTORIZER_FILE,
    DICTIONARY_FILE,
    ensure_directories,
)
from data_loader import FileProfile, build_unified_training_data


LOGGER = logging.getLogger("genderclassy.train")
if not LOGGER.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def _profiles_to_dict(profiles: List[FileProfile]) -> List[Dict[str, object]]:
    return [
        {
            "file": profile.path.name,
            "encoding": profile.encoding,
            "separator": profile.separator,
            "has_header": profile.has_header,
            "columns": profile.columns,
            "usable": profile.usable,
            "reason": profile.reason,
        }
        for profile in profiles
    ]


def _write_training_report(
    unified_df: pd.DataFrame,
    model_df: pd.DataFrame,
    sampled_for_model: bool,
    profiles: List[FileProfile],
    classification_text: str,
    confusion: np.ndarray,
    metrics: Dict[str, object],
) -> None:
    lines: List[str] = []
    lines.append("GenderClassy Training Report")
    lines.append("=" * 80)
    lines.append("")
    lines.append("Data Files Inspection")
    lines.append("-" * 80)
    for profile in profiles:
        lines.append(
            f"{profile.path.name} | usable={profile.usable} | reason={profile.reason} | "
            f"encoding={profile.encoding} | sep={profile.separator!r} | columns={profile.columns}"
        )
    lines.append("")
    lines.append("Unified Training Dataset")
    lines.append("-" * 80)
    lines.append(f"Rows (unique normalized names): {len(unified_df):,}")
    lines.append(f"Rows used by ML model: {len(model_df):,}")
    lines.append(f"Model sampling applied: {sampled_for_model}")
    if not unified_df.empty:
        label_counts = unified_df["label"].value_counts().to_dict()
        lines.append(f"Label distribution: {label_counts}")
        lines.append(f"Weight sum: {unified_df['weight'].sum():,.2f}")
    lines.append("")
    lines.append("Model Metrics")
    lines.append("-" * 80)
    lines.append(json.dumps(metrics, indent=2, ensure_ascii=False))
    lines.append("")
    lines.append("Classification Report")
    lines.append("-" * 80)
    lines.append(classification_text)
    lines.append("")
    lines.append("Confusion Matrix")
    lines.append("-" * 80)
    lines.append(str(confusion))
    lines.append("")

    REPORT_FILE.write_text("\n".join(lines), encoding="utf-8")


def _prepare_model_dataframe(unified_df: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    if len(unified_df) <= MAX_MODEL_TRAIN_ROWS:
        return unified_df, False

    sampled_chunks: List[pd.DataFrame] = []
    label_counts = unified_df["label"].value_counts(normalize=True)

    for label, group in unified_df.groupby("label", sort=False):
        proportion = float(label_counts.get(label, 0.0))
        target_n = max(1, int(round(MAX_MODEL_TRAIN_ROWS * proportion)))
        target_n = min(target_n, len(group))
        if target_n == len(group):
            sampled_chunks.append(group)
            continue

        sample_weights = group["weight"] if float(group["weight"].sum()) > 0 else None
        sampled_chunks.append(
            group.sample(
                n=target_n,
                replace=False,
                weights=sample_weights,
                random_state=RANDOM_STATE,
            )
        )

    model_df = pd.concat(sampled_chunks, ignore_index=True)
    if len(model_df) > MAX_MODEL_TRAIN_ROWS:
        model_df = model_df.sample(
            n=MAX_MODEL_TRAIN_ROWS,
            replace=False,
            weights=model_df["weight"] if float(model_df["weight"].sum()) > 0 else None,
            random_state=RANDOM_STATE,
        )
    model_df = model_df.reset_index(drop=True)
    return model_df, True


def main() -> None:
    ensure_directories()
    LOGGER.info("Building unified training data from local datasets...")
    unified_df, dictionary_lookup, profiles = build_unified_training_data()

    if unified_df.empty:
        raise RuntimeError("No usable training data found. Check files in data/ and loader logs.")

    model_df, sampled_for_model = _prepare_model_dataframe(unified_df)
    if sampled_for_model:
        LOGGER.info(
            "Applied weighted stratified sampling for ML training: %s -> %s rows",
            len(unified_df),
            len(model_df),
        )
    else:
        LOGGER.info("Using all unified rows for ML training: %s rows", len(model_df))

    # Train on one row per normalized name with its aggregated weight.
    features = model_df["name_normalized"].astype(str)
    labels = model_df["label"].astype(str)
    weights = model_df["weight"].astype(float)

    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        features,
        labels,
        weights,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels,
    )

    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 5),
        min_df=2,
        max_features=MAX_FEATURES,
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    label_encoder = LabelEncoder()
    label_encoder.fit(labels)
    y_train_enc = label_encoder.transform(y_train)
    y_test_enc = label_encoder.transform(y_test)

    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        solver="saga",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    LOGGER.info("Training LogisticRegression model...")
    model.fit(X_train_vec, y_train_enc, sample_weight=w_train.to_numpy())

    y_pred_enc = model.predict(X_test_vec)
    y_pred = label_encoder.inverse_transform(y_pred_enc)

    accuracy = float(accuracy_score(y_test, y_pred))
    report_dict = classification_report(
        y_test,
        y_pred,
        labels=label_encoder.classes_,
        output_dict=True,
        zero_division=0,
    )
    report_text = classification_report(
        y_test,
        y_pred,
        labels=label_encoder.classes_,
        zero_division=0,
    )
    confusion = confusion_matrix(y_test, y_pred, labels=label_encoder.classes_)

    metrics = {
        "accuracy": accuracy,
        "labels": list(label_encoder.classes_),
        "classification_report": report_dict,
        "confusion_matrix": confusion.tolist(),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "full_unified_rows": int(len(unified_df)),
        "model_rows_used": int(len(model_df)),
        "sampled_for_model": bool(sampled_for_model),
        "feature_count": int(X_train_vec.shape[1]),
        "dictionary_size": int(len(dictionary_lookup)),
        "data_files": _profiles_to_dict(profiles),
    }

    print("\nClassification Report")
    print("=" * 80)
    print(report_text)
    print("Confusion Matrix")
    print("=" * 80)
    print(confusion)

    joblib.dump(model, MODEL_FILE)
    joblib.dump(vectorizer, VECTORIZER_FILE)
    joblib.dump(label_encoder, LABEL_ENCODER_FILE)
    joblib.dump(dictionary_lookup, DICTIONARY_FILE)
    with METRICS_FILE.open("w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=2, ensure_ascii=False)
    _write_training_report(unified_df, model_df, sampled_for_model, profiles, report_text, confusion, metrics)

    LOGGER.info("Saved model: %s", MODEL_FILE)
    LOGGER.info("Saved vectorizer: %s", VECTORIZER_FILE)
    LOGGER.info("Saved label encoder: %s", LABEL_ENCODER_FILE)
    LOGGER.info("Saved dictionary lookup: %s", DICTIONARY_FILE)
    LOGGER.info("Saved metrics: %s", METRICS_FILE)
    LOGGER.info("Saved training report: %s", REPORT_FILE)


if __name__ == "__main__":
    main()
