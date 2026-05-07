import csv
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import pandas as pd

from config import (
    BOTH_THRESHOLD,
    CHUNK_SIZE,
    CSV_ENCODINGS,
    CSV_SEPARATORS,
    DATA_DIR,
    EXPECTED_DATA_FILES,
    GENDER_COLUMN_CANDIDATES,
    NAME_COLUMN_CANDIDATES,
    UNISEX_MARGIN,
    WEIGHT_COLUMN_CANDIDATES,
)
from preprocessing import normalize_name_series, standardize_gender_label


LOGGER = logging.getLogger("genderclassy.data_loader")
if not LOGGER.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


@dataclass
class FileProfile:
    path: Path
    encoding: str
    separator: str
    has_header: bool
    columns: List[str]
    usable: bool
    reason: str


def _read_sample_bytes(path: Path, max_bytes: int = 200_000) -> bytes:
    with path.open("rb") as file:
        return file.read(max_bytes)


def _decode_sample(raw: bytes) -> Tuple[str, str]:
    for encoding in CSV_ENCODINGS:
        try:
            return raw.decode(encoding), encoding
        except UnicodeDecodeError:
            continue
    return raw.decode("latin1", errors="replace"), "latin1"


def _detect_separator(sample_text: str) -> str:
    non_empty_lines = [line for line in sample_text.splitlines() if line.strip()]
    sample_block = "\n".join(non_empty_lines[:20])
    if not sample_block:
        return ","
    try:
        dialect = csv.Sniffer().sniff(sample_block, delimiters="".join(CSV_SEPARATORS))
        return dialect.delimiter
    except csv.Error:
        return ","


def _find_first_non_empty_line(sample_text: str) -> str:
    for line in sample_text.splitlines():
        if line.strip():
            return line.strip()
    return ""


def _header_from_width(width: int) -> List[str]:
    if width == 2:
        return ["name", "gender"]
    if width == 3:
        return ["name", "gender", "weight"]
    if width == 4:
        return ["name", "gender", "count", "probability"]
    return [f"column_{i+1}" for i in range(width)]


def _infer_header(tokens: List[str]) -> bool:
    lowered = [token.strip().casefold() for token in tokens]
    all_known = set(NAME_COLUMN_CANDIDATES + GENDER_COLUMN_CANDIDATES + WEIGHT_COLUMN_CANDIDATES + ["code", "src", "langcode"])
    if any(token in all_known for token in lowered):
        return True

    if len(lowered) >= 2 and standardize_gender_label(lowered[1]) is not None:
        return False

    return True


def _normalize_columns(columns: Iterable[object]) -> List[str]:
    return [str(col).strip().casefold() for col in columns]


def _pick_column(columns: List[str], candidates: List[str]) -> Optional[str]:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def _file_source_label(path: Path) -> str:
    return "WGND" if path.name.casefold().startswith("wgnd_") else "UCI"


def _source_flag(source: str) -> int:
    return 2 if source == "WGND" else 1


def inspect_data_files(data_dir: Path = DATA_DIR) -> List[FileProfile]:
    profiles: List[FileProfile] = []
    if not data_dir.exists():
        LOGGER.warning("Data directory not found: %s", data_dir)
        return profiles

    known = set(EXPECTED_DATA_FILES)
    candidate_paths = sorted(data_dir.glob("*"))
    if known:
        candidate_paths = sorted([p for p in candidate_paths if p.is_file() and p.name in known])
    else:
        candidate_paths = sorted([p for p in candidate_paths if p.is_file()])

    for path in candidate_paths:
        if path.suffix.casefold() not in {".csv", ".txt"}:
            profiles.append(
                FileProfile(
                    path=path,
                    encoding="n/a",
                    separator="n/a",
                    has_header=True,
                    columns=[],
                    usable=False,
                    reason="Unsupported extension",
                )
            )
            continue

        raw = _read_sample_bytes(path)
        sample_text, encoding = _decode_sample(raw)
        separator = _detect_separator(sample_text)
        first_line = _find_first_non_empty_line(sample_text)
        tokens = [token.strip() for token in first_line.split(separator)] if first_line else []
        has_header = _infer_header(tokens) if tokens else True
        inferred_columns = _normalize_columns(tokens) if has_header else _header_from_width(len(tokens))

        name_col = _pick_column(inferred_columns, NAME_COLUMN_CANDIDATES)
        gender_col = _pick_column(inferred_columns, GENDER_COLUMN_CANDIDATES)
        usable = bool(name_col and gender_col)
        reason = "OK" if usable else "Missing name/gender columns"

        LOGGER.info(
            "File: %s | encoding=%s | sep=%r | header=%s | columns=%s | usable=%s | reason=%s",
            path.name,
            encoding,
            separator,
            has_header,
            inferred_columns,
            usable,
            reason,
        )
        profiles.append(
            FileProfile(
                path=path,
                encoding=encoding,
                separator=separator,
                has_header=has_header,
                columns=inferred_columns,
                usable=usable,
                reason=reason,
            )
        )

    return profiles


def _iter_csv_chunks(profile: FileProfile, chunk_size: int = CHUNK_SIZE) -> Iterator[pd.DataFrame]:
    errors: List[str] = []
    for encoding in [profile.encoding] + [enc for enc in CSV_ENCODINGS if enc != profile.encoding]:
        read_kwargs = {
            "sep": profile.separator,
            "encoding": encoding,
            "dtype": str,
            "chunksize": chunk_size,
            "on_bad_lines": "skip",
            "keep_default_na": True,
            "low_memory": True,
        }
        if profile.has_header:
            read_kwargs["header"] = 0
        else:
            read_kwargs["header"] = None
            read_kwargs["names"] = profile.columns

        try:
            chunk_iter = pd.read_csv(profile.path, **read_kwargs)
            first_chunk = next(chunk_iter)
            yield first_chunk
            for chunk in chunk_iter:
                yield chunk
            return
        except StopIteration:
            return
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{encoding}: {exc}")

    raise RuntimeError(f"Failed to read {profile.path.name}. Tried encodings: {errors}")


def _summarize_source(source_flags: int) -> str:
    if source_flags == 3:
        return "Both"
    if source_flags == 2:
        return "WGND"
    if source_flags == 1:
        return "UCI"
    return "Unknown"


def _resolve_label(male_score: float, female_score: float, both_score: float) -> Tuple[str, float]:
    total = male_score + female_score + both_score
    if total <= 0:
        return "both", 0.0

    male_prob = male_score / total
    female_prob = female_score / total
    both_prob = both_score / total
    gap = abs(male_prob - female_prob)

    if both_prob >= BOTH_THRESHOLD or gap <= UNISEX_MARGIN:
        if both_prob >= BOTH_THRESHOLD:
            confidence = both_prob
        else:
            confidence = 1.0 - gap
        return "both", float(max(0.0, min(1.0, confidence)))

    if male_prob >= female_prob:
        return "male", float(male_prob)
    return "female", float(female_prob)


def _aggregate_file(profile: FileProfile, chunk_size: int) -> pd.DataFrame:
    source = _file_source_label(profile.path)
    LOGGER.info("Processing %s (%s)", profile.path.name, source)

    grouped_chunks: List[pd.DataFrame] = []
    raw_rows = 0
    valid_rows = 0

    for chunk in _iter_csv_chunks(profile, chunk_size=chunk_size):
        raw_rows += len(chunk)
        chunk.columns = _normalize_columns(chunk.columns)
        columns = chunk.columns.tolist()

        name_col = _pick_column(columns, NAME_COLUMN_CANDIDATES)
        gender_col = _pick_column(columns, GENDER_COLUMN_CANDIDATES)
        if not name_col or not gender_col:
            LOGGER.warning("Skipping chunk from %s: missing name/gender columns after normalization", profile.path.name)
            continue

        weight_col = _pick_column(columns, WEIGHT_COLUMN_CANDIDATES)
        reduced = chunk[[name_col, gender_col] + ([weight_col] if weight_col else [])].copy()

        reduced[name_col] = reduced[name_col].astype(str).str.strip()
        reduced = reduced.loc[reduced[name_col].ne("")]
        reduced["label"] = reduced[gender_col].map(standardize_gender_label)
        reduced = reduced.loc[reduced["label"].notna()]

        if reduced.empty:
            continue

        if weight_col:
            reduced["weight"] = pd.to_numeric(reduced[weight_col], errors="coerce").fillna(1.0)
        else:
            reduced["weight"] = 1.0
        reduced = reduced.loc[reduced["weight"] > 0, [name_col, "label", "weight"]]
        if reduced.empty:
            continue

        # Pre-group by raw name to reduce normalization workload.
        reduced = reduced.groupby([name_col, "label"], as_index=False, sort=False)["weight"].sum()
        reduced["name_normalized"] = normalize_name_series(reduced[name_col])
        reduced = reduced.loc[reduced["name_normalized"].ne(""), ["name_normalized", "label", "weight"]]
        if reduced.empty:
            continue

        reduced = reduced.groupby(["name_normalized", "label"], as_index=False, sort=False)["weight"].sum()
        grouped_chunks.append(reduced)
        valid_rows += len(reduced)

    if not grouped_chunks:
        LOGGER.info("Finished %s | raw_rows=%s | valid_rows=0", profile.path.name, raw_rows)
        return pd.DataFrame(columns=["name_normalized", "label", "weight"])

    file_grouped = pd.concat(grouped_chunks, ignore_index=True)
    file_grouped = file_grouped.groupby(["name_normalized", "label"], as_index=False, sort=False)["weight"].sum()
    LOGGER.info("Finished %s | raw_rows=%s | grouped_rows=%s", profile.path.name, raw_rows, len(file_grouped))
    return file_grouped


def build_unified_training_data(data_dir: Path = DATA_DIR, chunk_size: int = CHUNK_SIZE) -> Tuple[pd.DataFrame, Dict[str, dict], List[FileProfile]]:
    profiles = inspect_data_files(data_dir)
    usable_profiles = [profile for profile in profiles if profile.usable]

    if not usable_profiles:
        return pd.DataFrame(columns=["name_normalized", "label", "weight", "source"]), {}, profiles

    global_scores: Optional[pd.DataFrame] = None
    source_flags: Dict[str, int] = defaultdict(int)

    for profile in usable_profiles:
        source = _file_source_label(profile.path)
        flag = _source_flag(source)
        file_grouped = _aggregate_file(profile, chunk_size=chunk_size)
        if file_grouped.empty:
            continue

        pivot = file_grouped.pivot_table(
            index="name_normalized",
            columns="label",
            values="weight",
            aggfunc="sum",
            fill_value=0.0,
        )
        for label_name in ("male", "female", "both"):
            if label_name not in pivot.columns:
                pivot[label_name] = 0.0
        pivot = pivot[["male", "female", "both"]]

        if global_scores is None:
            global_scores = pivot
        else:
            global_scores = global_scores.add(pivot, fill_value=0.0)

        for name_normalized in pivot.index:
            source_flags[name_normalized] |= flag

    if global_scores is None or global_scores.empty:
        return pd.DataFrame(columns=["name_normalized", "label", "weight", "source"]), {}, profiles

    rows = []
    dictionary_lookup: Dict[str, dict] = {}
    global_scores = global_scores.fillna(0.0)
    score_rows = global_scores.reset_index()

    for row in score_rows.itertuples(index=False):
        name_normalized = row.name_normalized
        male_score = float(row.male)
        female_score = float(row.female)
        both_score = float(row.both)
        total_weight = male_score + female_score + both_score
        if total_weight <= 0:
            continue

        final_label, confidence = _resolve_label(male_score, female_score, both_score)
        source_summary = _summarize_source(source_flags.get(name_normalized, 0))
        rows.append(
            {
                "name_normalized": name_normalized,
                "label": final_label,
                "weight": total_weight,
                "source": source_summary,
                "male_score": male_score,
                "female_score": female_score,
                "both_score": both_score,
                "confidence": confidence,
            }
        )
        dictionary_lookup[name_normalized] = {
            "male_score": male_score,
            "female_score": female_score,
            "both_score": both_score,
            "final_label": final_label,
            "confidence": confidence,
            "source_summary": source_summary,
        }

    unified_df = pd.DataFrame(rows)
    if not unified_df.empty:
        unified_df = unified_df.sort_values("weight", ascending=False).reset_index(drop=True)

    return unified_df, dictionary_lookup, profiles
