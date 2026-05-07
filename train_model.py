import json
import logging
from datetime import datetime, timezone
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from config import (
    BOTH_CLASS_OVERSAMPLE_FACTOR,
    BOTH_THRESHOLD,
    DICTIONARY_FILE,
    FALLBACK_CONFIDENCE_BUFFER,
    FALLBACK_FOCUS_RATIO,
    LABEL_ENCODER_FILE,
    MAX_BOTH_CLASS_RATIO_FOR_FALLBACK,
    MAX_FEATURES,
    MAX_MODEL_TRAIN_ROWS,
    METRICS_FILE,
    MIN_CONFIDENCE,
    MODEL_FILE,
    RANDOM_STATE,
    REPORT_FILE,
    TEST_SIZE,
    UNISEX_MARGIN,
    VECTORIZER_FILE,
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


def _distribution_by_count(labels: pd.Series) -> Dict[str, dict]:
    counts = labels.value_counts()
    total = int(counts.sum())
    distribution: Dict[str, dict] = {}
    for label in ("male", "female", "both"):
        count = int(counts.get(label, 0))
        distribution[label] = {
            "count": count,
            "ratio": float((count / total) if total else 0.0),
        }
    return distribution


def _distribution_by_weight(df: pd.DataFrame) -> Dict[str, dict]:
    if df.empty:
        return {label: {"weight_sum": 0.0, "ratio": 0.0} for label in ("male", "female", "both")}

    sums = df.groupby("label", observed=True)["weight"].sum()
    total = float(sums.sum())
    distribution: Dict[str, dict] = {}
    for label in ("male", "female", "both"):
        weight_sum = float(sums.get(label, 0.0))
        distribution[label] = {
            "weight_sum": weight_sum,
            "ratio": float((weight_sum / total) if total else 0.0),
        }
    return distribution


def _weighted_stratified_sample(
    df: pd.DataFrame,
    target_rows: int,
    target_class_ratios: Dict[str, float] | None = None,
) -> pd.DataFrame:
    if target_rows <= 0 or df.empty:
        return df.head(0).copy()
    if len(df) <= target_rows:
        return df.copy().reset_index(drop=True)

    sampled_parts: List[pd.DataFrame] = []
    if target_class_ratios:
        proportions = {label: max(0.0, float(target_class_ratios.get(label, 0.0))) for label in ("male", "female", "both")}
        total_prop = sum(proportions.values())
        if total_prop > 0:
            class_proportions = {label: val / total_prop for label, val in proportions.items()}
        else:
            class_proportions = df["label"].value_counts(normalize=True).to_dict()
    else:
        class_proportions = df["label"].value_counts(normalize=True).to_dict()

    for label in ("male", "female", "both"):
        group = df.loc[df["label"] == label]
        if group.empty:
            continue
        proportion = float(class_proportions.get(label, 0.0))
        n_rows = int(round(target_rows * proportion))
        if target_class_ratios is None and proportion > 0:
            n_rows = max(1, n_rows)
        if n_rows <= 0:
            continue
        n_rows = min(n_rows, len(group))
        weights = group["weight"] if float(group["weight"].sum()) > 0 else None
        sampled_parts.append(
            group.sample(
                n=n_rows,
                replace=False,
                weights=weights,
                random_state=RANDOM_STATE,
            )
        )

    if not sampled_parts:
        return df.head(0).copy()

    sampled = pd.concat(sampled_parts, axis=0, ignore_index=False)
    if len(sampled) > target_rows:
        weights = sampled["weight"] if float(sampled["weight"].sum()) > 0 else None
        sampled = sampled.sample(
            n=target_rows,
            replace=False,
            weights=weights,
            random_state=RANDOM_STATE,
        )
    elif len(sampled) < target_rows:
        needed = target_rows - len(sampled)
        remaining = df.loc[~df.index.isin(sampled.index)]
        if needed > 0 and not remaining.empty:
            weights = remaining["weight"] if float(remaining["weight"].sum()) > 0 else None
            extra = remaining.sample(
                n=min(needed, len(remaining)),
                replace=False,
                weights=weights,
                random_state=RANDOM_STATE,
            )
            sampled = pd.concat([sampled, extra], axis=0, ignore_index=False)

    return sampled.reset_index(drop=True)


def _build_fallback_target_class_mix(unified_df: pd.DataFrame) -> Dict[str, float]:
    base = unified_df["label"].value_counts(normalize=True).to_dict()
    male_ratio = float(base.get("male", 0.0))
    female_ratio = float(base.get("female", 0.0))
    both_ratio = float(base.get("both", 0.0))

    adjusted_both = min(both_ratio * BOTH_CLASS_OVERSAMPLE_FACTOR, MAX_BOTH_CLASS_RATIO_FOR_FALLBACK)
    remaining = max(0.0, 1.0 - adjusted_both)
    male_female_total = male_ratio + female_ratio

    if male_female_total > 0:
        adjusted_male = remaining * (male_ratio / male_female_total)
        adjusted_female = remaining * (female_ratio / male_female_total)
    else:
        adjusted_male = remaining * 0.5
        adjusted_female = remaining * 0.5

    return {
        "male": adjusted_male,
        "female": adjusted_female,
        "both": adjusted_both,
    }


def _prepare_model_dataframe(unified_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, object]]:
    low_conf_cutoff = min(1.0, MIN_CONFIDENCE + FALLBACK_CONFIDENCE_BUFFER)
    fallback_focus_mask = (unified_df["confidence"] < low_conf_cutoff) | (unified_df["label"] == "both")
    fallback_focus_df = unified_df.loc[fallback_focus_mask]
    strong_dictionary_df = unified_df.loc[~fallback_focus_mask]

    if len(unified_df) <= MAX_MODEL_TRAIN_ROWS:
        strategy = {
            "sampled_for_model": False,
            "max_model_train_rows": MAX_MODEL_TRAIN_ROWS,
            "fallback_focus_ratio_target": FALLBACK_FOCUS_RATIO,
            "fallback_focus_cutoff": low_conf_cutoff,
            "fallback_focus_rows_available": int(len(fallback_focus_df)),
            "strong_dictionary_rows_available": int(len(strong_dictionary_df)),
            "fallback_focus_rows_used": int(len(fallback_focus_df)),
            "strong_dictionary_rows_used": int(len(strong_dictionary_df)),
            "notes": "All rows were used because unified data is within MAX_MODEL_TRAIN_ROWS.",
        }
        return unified_df.reset_index(drop=True), strategy

    pool_target_rows = min(len(unified_df), int(round(MAX_MODEL_TRAIN_ROWS * 1.30)))
    target_focus = min(int(round(pool_target_rows * FALLBACK_FOCUS_RATIO)), len(fallback_focus_df))
    target_strong = pool_target_rows - target_focus
    target_strong = min(target_strong, len(strong_dictionary_df))

    # If one side has spare capacity, fill the remainder from the other side.
    selected_rows = target_focus + target_strong
    remaining_slots = pool_target_rows - selected_rows
    if remaining_slots > 0 and len(fallback_focus_df) > target_focus:
        add_focus = min(remaining_slots, len(fallback_focus_df) - target_focus)
        target_focus += add_focus
        remaining_slots -= add_focus
    if remaining_slots > 0 and len(strong_dictionary_df) > target_strong:
        add_strong = min(remaining_slots, len(strong_dictionary_df) - target_strong)
        target_strong += add_strong

    fallback_sample = _weighted_stratified_sample(fallback_focus_df, target_focus)
    strong_sample = _weighted_stratified_sample(strong_dictionary_df, target_strong)
    model_pool_df = pd.concat([fallback_sample, strong_sample], ignore_index=True)

    target_class_mix = _build_fallback_target_class_mix(unified_df)
    model_df = _weighted_stratified_sample(
        model_pool_df,
        target_rows=min(MAX_MODEL_TRAIN_ROWS, len(model_pool_df)),
        target_class_ratios=target_class_mix,
    )

    strategy = {
        "sampled_for_model": True,
        "max_model_train_rows": MAX_MODEL_TRAIN_ROWS,
        "fallback_focus_ratio_target": FALLBACK_FOCUS_RATIO,
        "fallback_focus_cutoff": low_conf_cutoff,
        "both_class_oversample_factor": BOTH_CLASS_OVERSAMPLE_FACTOR,
        "max_both_class_ratio_for_fallback": MAX_BOTH_CLASS_RATIO_FOR_FALLBACK,
        "pool_target_rows_before_rebalance": int(pool_target_rows),
        "target_class_mix_after_sampling": target_class_mix,
        "fallback_focus_rows_available": int(len(fallback_focus_df)),
        "strong_dictionary_rows_available": int(len(strong_dictionary_df)),
        "fallback_focus_rows_used": int(len(fallback_sample)),
        "strong_dictionary_rows_used": int(len(strong_sample)),
        "pool_rows_before_rebalance": int(len(model_pool_df)),
        "final_model_rows": int(len(model_df)),
        "notes": (
            "Fallback-focused sampling used. The ML model is trained primarily on names that are "
            "not strongly covered by dictionary confidence and on 'both' names."
        ),
    }
    return model_df.reset_index(drop=True), strategy


def _dict_entry_to_probs(entry: dict) -> Dict[str, float]:
    male = float(entry.get("male_score", 0.0))
    female = float(entry.get("female_score", 0.0))
    both = float(entry.get("both_score", 0.0))
    total = male + female + both
    if total <= 0:
        return {"male": 1.0 / 3.0, "female": 1.0 / 3.0, "both": 1.0 / 3.0}
    return {
        "male": male / total,
        "female": female / total,
        "both": both / total,
    }


def _resolve_label_from_probs(
    probs: Dict[str, float],
    min_confidence: float,
    unisex_margin: float,
    both_threshold: float,
) -> str:
    male = probs["male"]
    female = probs["female"]
    both = probs["both"]
    confidence = max(male, female, both)

    if confidence < min_confidence:
        return "both"
    if abs(male - female) <= unisex_margin:
        return "both"
    if both >= both_threshold and both >= max(male, female):
        return "both"
    if male >= female:
        return "male"
    return "female"


def _evaluate_predictions(y_true: pd.Series, y_pred: List[str], labels: List[str]) -> Tuple[dict, str, np.ndarray]:
    report_dict = classification_report(
        y_true,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    report_text = classification_report(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )
    confusion = confusion_matrix(y_true, y_pred, labels=labels)
    per_class = {
        label: {
            "precision": float(report_dict[label]["precision"]),
            "recall": float(report_dict[label]["recall"]),
            "f1": float(report_dict[label]["f1-score"]),
            "support": int(report_dict[label]["support"]),
        }
        for label in labels
    }
    summary = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(report_dict["macro avg"]["f1-score"]),
        "weighted_f1": float(report_dict["weighted avg"]["f1-score"]),
        "per_class": per_class,
        "classification_report": report_dict,
        "confusion_matrix": {
            "labels": labels,
            "matrix": confusion.tolist(),
        },
    }
    return summary, report_text, confusion


def _predict_hybrid_labels(
    names: List[str],
    model_probabilities: np.ndarray,
    class_labels: np.ndarray,
    dictionary_lookup: Dict[str, dict],
    min_confidence: float,
    unisex_margin: float,
    both_threshold: float,
) -> Tuple[List[str], Dict[str, float]]:
    predictions: List[str] = []
    source_counts = {"Dictionary": 0, "Mixed": 0, "ML Model": 0}

    for idx, name in enumerate(names):
        model_probs_row = {label: float(prob) for label, prob in zip(class_labels, model_probabilities[idx])}
        probs = {
            "male": model_probs_row.get("male", 0.0),
            "female": model_probs_row.get("female", 0.0),
            "both": model_probs_row.get("both", 0.0),
        }

        dict_entry = dictionary_lookup.get(name)
        if dict_entry and float(dict_entry.get("confidence", 0.0)) >= min_confidence:
            probs = _dict_entry_to_probs(dict_entry)
            source_counts["Dictionary"] += 1
        elif dict_entry:
            dict_probs = _dict_entry_to_probs(dict_entry)
            probs = {
                "male": 0.65 * probs["male"] + 0.35 * dict_probs["male"],
                "female": 0.65 * probs["female"] + 0.35 * dict_probs["female"],
                "both": 0.65 * probs["both"] + 0.35 * dict_probs["both"],
            }
            source_counts["Mixed"] += 1
        else:
            source_counts["ML Model"] += 1

        predictions.append(
            _resolve_label_from_probs(
                probs=probs,
                min_confidence=min_confidence,
                unisex_margin=unisex_margin,
                both_threshold=both_threshold,
            )
        )

    total = max(len(names), 1)
    rates = {
        "dictionary_hit_rate": float(source_counts["Dictionary"] / total),
        "ml_fallback_rate": float((source_counts["Mixed"] + source_counts["ML Model"]) / total),
        "mixed_rate": float(source_counts["Mixed"] / total),
        "pure_ml_rate": float(source_counts["ML Model"] / total),
        "source_counts": source_counts,
    }
    return predictions, rates


def _threshold_grid_search(
    names: List[str],
    y_true: pd.Series,
    model_probabilities: np.ndarray,
    class_labels: np.ndarray,
    dictionary_lookup: Dict[str, dict],
) -> dict:
    min_confidence_grid = [0.55, 0.60, 0.65, 0.70]
    unisex_margin_grid = [0.10, 0.15, 0.20]
    both_threshold_grid = [0.20, 0.25, 0.30]

    trials: List[dict] = []
    labels = ["male", "female", "both"]

    for min_conf in min_confidence_grid:
        for margin in unisex_margin_grid:
            for both_th in both_threshold_grid:
                predictions, rates = _predict_hybrid_labels(
                    names=names,
                    model_probabilities=model_probabilities,
                    class_labels=class_labels,
                    dictionary_lookup=dictionary_lookup,
                    min_confidence=min_conf,
                    unisex_margin=margin,
                    both_threshold=both_th,
                )
                report = classification_report(
                    y_true,
                    predictions,
                    labels=labels,
                    output_dict=True,
                    zero_division=0,
                )
                trials.append(
                    {
                        "min_confidence": min_conf,
                        "unisex_margin": margin,
                        "both_threshold": both_th,
                        "accuracy": float(accuracy_score(y_true, predictions)),
                        "macro_f1": float(report["macro avg"]["f1-score"]),
                        "weighted_f1": float(report["weighted avg"]["f1-score"]),
                        "both_f1": float(report["both"]["f1-score"]),
                        "dictionary_hit_rate": float(rates["dictionary_hit_rate"]),
                        "ml_fallback_rate": float(rates["ml_fallback_rate"]),
                    }
                )

    best_by_macro = max(trials, key=lambda x: x["macro_f1"]) if trials else {}
    best_by_both = max(trials, key=lambda x: x["both_f1"]) if trials else {}
    top_trials = sorted(trials, key=lambda x: (x["macro_f1"], x["both_f1"]), reverse=True)[:5]

    return {
        "grid_size": len(trials),
        "best_by_macro_f1": best_by_macro,
        "best_by_both_f1": best_by_both,
        "top_5_by_macro_f1": top_trials,
    }


def _both_class_analysis(y_true: pd.Series, y_pred_ml: List[str], y_pred_hybrid: List[str]) -> dict:
    y_true_np = y_true.to_numpy()
    ml_pred_np = np.array(y_pred_ml)
    hybrid_pred_np = np.array(y_pred_hybrid)

    both_mask = y_true_np == "both"
    both_support = int(both_mask.sum())
    total_support = int(len(y_true_np))
    both_ratio = float((both_support / total_support) if total_support else 0.0)

    if both_support == 0:
        return {
            "support": 0,
            "ratio_in_eval": 0.0,
            "notes": ["No 'both' labels in evaluation split."],
        }

    ml_correct = int((ml_pred_np[both_mask] == "both").sum())
    hybrid_correct = int((hybrid_pred_np[both_mask] == "both").sum())

    ml_as_male = int((ml_pred_np[both_mask] == "male").sum())
    ml_as_female = int((ml_pred_np[both_mask] == "female").sum())
    hybrid_as_male = int((hybrid_pred_np[both_mask] == "male").sum())
    hybrid_as_female = int((hybrid_pred_np[both_mask] == "female").sum())

    notes: List[str] = []
    if both_ratio < 0.15:
        notes.append("The 'both' class has lower representation than male/female, which hurts F1 stability.")
    if (ml_as_male + ml_as_female) > ml_correct:
        notes.append("Many true 'both' names are predicted as male/female, indicating overlap in character patterns.")
    if hybrid_correct > ml_correct:
        notes.append("Dictionary-first logic improves recall for ambiguous names when dictionary confidence is strong.")
    else:
        notes.append("Hybrid logic did not substantially improve 'both' recall on this split.")

    return {
        "support": both_support,
        "ratio_in_eval": both_ratio,
        "ml": {
            "correct_as_both": ml_correct,
            "misclassified_as_male": ml_as_male,
            "misclassified_as_female": ml_as_female,
        },
        "hybrid": {
            "correct_as_both": hybrid_correct,
            "misclassified_as_male": hybrid_as_male,
            "misclassified_as_female": hybrid_as_female,
        },
        "notes": notes,
    }


def _write_training_report(
    unified_df: pd.DataFrame,
    model_df: pd.DataFrame,
    strategy: Dict[str, object],
    profiles: List[FileProfile],
    metrics: Dict[str, object],
    ml_report_text: str,
    hybrid_report_text: str,
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
    lines.append("Dataset Summary")
    lines.append("-" * 80)
    lines.append(f"Unified rows (unique normalized names): {len(unified_df):,}")
    lines.append(f"Rows used by ML fallback model: {len(model_df):,}")
    lines.append(f"Sampling strategy: {json.dumps(strategy, ensure_ascii=False)}")
    lines.append("")
    lines.append("Hybrid Inference Note")
    lines.append("-" * 80)
    lines.append(
        "Production inference is dictionary-first. The ML model is a fallback when dictionary coverage "
        "is missing or low-confidence."
    )

    lines.append("")
    lines.append("ML Fallback Model Metrics")
    lines.append("-" * 80)
    lines.append(ml_report_text)
    lines.append("")
    lines.append("Hybrid System Metrics (Dictionary-first + ML fallback)")
    lines.append("-" * 80)
    lines.append(hybrid_report_text)

    lines.append("")
    lines.append("Full Metrics JSON")
    lines.append("-" * 80)
    lines.append(json.dumps(metrics, indent=2, ensure_ascii=False))
    lines.append("")

    REPORT_FILE.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_directories()
    LOGGER.info("Building unified training data from local datasets...")
    unified_df, dictionary_lookup, profiles = build_unified_training_data()
    if unified_df.empty:
        raise RuntimeError("No usable training data found. Check files in data/ and loader logs.")

    model_df, strategy = _prepare_model_dataframe(unified_df)
    LOGGER.info(
        "ML fallback training rows: %s (sampled=%s)",
        len(model_df),
        strategy.get("sampled_for_model", False),
    )

    features = model_df["name_normalized"].astype(str)
    labels = model_df["label"].astype(str)
    weights = model_df["weight"].astype(float)

    X_train, X_test, y_train, y_test, w_train, _w_test = train_test_split(
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

    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        solver="saga",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    LOGGER.info("Training ML fallback classifier...")
    model.fit(X_train_vec, y_train_enc, sample_weight=w_train.to_numpy())

    class_labels = label_encoder.inverse_transform(np.arange(len(label_encoder.classes_)))
    labels_order = list(class_labels)

    y_pred_ml_enc = model.predict(X_test_vec)
    y_pred_ml = label_encoder.inverse_transform(y_pred_ml_enc).tolist()
    ml_metrics, ml_report_text, ml_confusion = _evaluate_predictions(y_test, y_pred_ml, labels_order)

    model_probabilities = model.predict_proba(X_test_vec)
    test_names = X_test.tolist()
    y_pred_hybrid, hybrid_rates = _predict_hybrid_labels(
        names=test_names,
        model_probabilities=model_probabilities,
        class_labels=class_labels,
        dictionary_lookup=dictionary_lookup,
        min_confidence=MIN_CONFIDENCE,
        unisex_margin=UNISEX_MARGIN,
        both_threshold=BOTH_THRESHOLD,
    )
    hybrid_metrics, hybrid_report_text, hybrid_confusion = _evaluate_predictions(y_test, y_pred_hybrid, labels_order)
    hybrid_metrics.update(
        {
            "dictionary_hit_rate": hybrid_rates["dictionary_hit_rate"],
            "ml_fallback_rate": hybrid_rates["ml_fallback_rate"],
            "mixed_rate": hybrid_rates["mixed_rate"],
            "pure_ml_rate": hybrid_rates["pure_ml_rate"],
            "source_counts": hybrid_rates["source_counts"],
            "evaluation_note": (
                "Hybrid evaluation uses dictionary-first logic with the production dictionary artifact, "
                "then ML fallback for low-confidence/missing dictionary matches."
            ),
        }
    )

    threshold_tuning = _threshold_grid_search(
        names=test_names,
        y_true=y_test,
        model_probabilities=model_probabilities,
        class_labels=class_labels,
        dictionary_lookup=dictionary_lookup,
    )

    class_distribution_summary = {
        "unified_by_count": _distribution_by_count(unified_df["label"]),
        "unified_by_weight": _distribution_by_weight(unified_df),
        "model_training_by_count": _distribution_by_count(labels),
        "model_training_by_weight": _distribution_by_weight(model_df),
        "train_split_by_count": _distribution_by_count(y_train),
        "test_split_by_count": _distribution_by_count(y_test),
    }

    both_analysis = _both_class_analysis(y_test, y_pred_ml, y_pred_hybrid)

    metrics = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "labels": labels_order,
        "thresholds_used": {
            "UNISEX_MARGIN": UNISEX_MARGIN,
            "BOTH_THRESHOLD": BOTH_THRESHOLD,
            "MIN_CONFIDENCE": MIN_CONFIDENCE,
        },
        "class_distribution": class_distribution_summary,
        "training_strategy": strategy,
        "ml_fallback_model_metrics": ml_metrics,
        "hybrid_system_metrics": hybrid_metrics,
        "both_class_analysis": both_analysis,
        "threshold_tuning_analysis": threshold_tuning,
        "dataset_summary": {
            "full_unified_rows": int(len(unified_df)),
            "model_rows_used": int(len(model_df)),
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "feature_count": int(X_train_vec.shape[1]),
            "dictionary_size": int(len(dictionary_lookup)),
        },
        "data_files": _profiles_to_dict(profiles),
    }

    print("\nML Fallback Model Metrics")
    print("=" * 80)
    print(ml_report_text)
    print("ML Fallback Confusion Matrix")
    print("=" * 80)
    print(ml_confusion)
    print("\nHybrid System Metrics (Dictionary-first + ML fallback)")
    print("=" * 80)
    print(hybrid_report_text)
    print("Hybrid Confusion Matrix")
    print("=" * 80)
    print(hybrid_confusion)
    print(
        "\nDictionary hit rate: "
        f"{hybrid_rates['dictionary_hit_rate']:.2%} | ML fallback rate: {hybrid_rates['ml_fallback_rate']:.2%}"
    )

    joblib.dump(model, MODEL_FILE)
    joblib.dump(vectorizer, VECTORIZER_FILE)
    joblib.dump(label_encoder, LABEL_ENCODER_FILE)
    joblib.dump(dictionary_lookup, DICTIONARY_FILE)
    with METRICS_FILE.open("w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=2, ensure_ascii=False)
    _write_training_report(
        unified_df=unified_df,
        model_df=model_df,
        strategy=strategy,
        profiles=profiles,
        metrics=metrics,
        ml_report_text=ml_report_text,
        hybrid_report_text=hybrid_report_text,
    )

    LOGGER.info("Saved model: %s", MODEL_FILE)
    LOGGER.info("Saved vectorizer: %s", VECTORIZER_FILE)
    LOGGER.info("Saved label encoder: %s", LABEL_ENCODER_FILE)
    LOGGER.info("Saved dictionary lookup: %s", DICTIONARY_FILE)
    LOGGER.info("Saved metrics: %s", METRICS_FILE)
    LOGGER.info("Saved training report: %s", REPORT_FILE)


if __name__ == "__main__":
    main()
