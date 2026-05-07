from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import List

import pandas as pd
import streamlit as st

from config import (
    BOTH_THRESHOLD,
    DATA_DIR,
    EXPECTED_DATA_FILES,
    MIN_CONFIDENCE,
    MODELS_DIR,
    UNISEX_MARGIN,
)
from predict import load_artifacts, predict_dataframe, predict_many_names, predict_single_name
from preprocessing import normalize_name


st.set_page_config(page_title="Name Gender Predictor", page_icon="︎🧠", layout="wide")


CUSTOM_CSS = """
<style>
.main {
  background: radial-gradient(circle at 10% 0%, #f3f8f4 0%, #f8fafc 55%, #eef6f0 100%);
}
.hero {
  background: linear-gradient(120deg, #0e4d3b 0%, #13634d 45%, #2f855a 100%);
  color: #ffffff;
  border-radius: 18px;
  padding: 28px 30px;
  box-shadow: 0 12px 28px rgba(14, 77, 59, 0.25);
  margin-bottom: 20px;
}
.hero h1 {
  margin: 0;
  font-size: 2rem;
  letter-spacing: 0.2px;
}
.hero p {
  margin: 8px 0 0 0;
  font-size: 1rem;
  opacity: 0.95;
}
.card {
  background: #ffffff;
  border: 1px solid #e2e8f0;
  border-radius: 16px;
  padding: 18px 18px;
  box-shadow: 0 6px 16px rgba(15, 23, 42, 0.06);
}
.result-badge {
  display: inline-block;
  border-radius: 999px;
  padding: 7px 14px;
  font-weight: 700;
  font-size: 0.95rem;
}
.badge-male {
  background: #dbeafe;
  color: #1d4ed8;
}
.badge-female {
  background: #ffe4e6;
  color: #be123c;
}
.badge-both {
  background: #ecfccb;
  color: #3f6212;
}
.metric-pill {
  background: rgba(15, 23, 42, 0.06);

  border: 1px solid rgba(15, 23, 42, 0.10);

  backdrop-filter: blur(8px);

  border-radius: 14px;

  padding: 12px 14px;

  color: inherit;

  font-weight: 700;

  text-align: center;

  box-shadow:
      0 4px 14px rgba(0,0,0,0.08);

  margin-bottom: 8px;
}
.footer-note {
  color: #475569;
  font-size: 0.88rem;
  margin-top: 18px;
}
</style>
"""


st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def convert_df_to_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


@st.cache_data(show_spinner=False)
def convert_df_to_excel(df: pd.DataFrame) -> bytes:
    buffer = BytesIO()
    engine = "xlsxwriter"
    try:
        import xlsxwriter  # noqa: F401
    except Exception:  # noqa: BLE001
        engine = "openpyxl"

    with pd.ExcelWriter(buffer, engine=engine) as writer:
        df.to_excel(writer, index=False, sheet_name="predictions")
    return buffer.getvalue()


def parse_text_names(text: str) -> List[str]:
    if not text:
        return []
    normalized_separators = text.replace("\n", ",").replace(";", ",")
    names = [chunk.strip() for chunk in normalized_separators.split(",")]
    return [name for name in names if name]


def read_uploaded_table(uploaded_file) -> pd.DataFrame:
    suffix = Path(uploaded_file.name).suffix.casefold()
    if suffix == ".csv":
        encodings = ["utf-8-sig", "utf-8", "cp1252", "latin1"]
        for encoding in encodings:
            try:
                uploaded_file.seek(0)
                return pd.read_csv(uploaded_file, sep=None, engine="python", encoding=encoding)
            except Exception:
                continue
        raise ValueError("Could not parse CSV file with supported encodings.")

    if suffix in {".xlsx", ".xls"}:
        uploaded_file.seek(0)
        return pd.read_excel(uploaded_file)

    raise ValueError("Unsupported file format. Please upload CSV, XLSX, or XLS.")


@st.cache_resource(show_spinner=False)
def cached_artifacts():
    return load_artifacts()


def get_model_status() -> tuple[bool, str]:
    try:
        cached_artifacts()
        return True, "Loaded successfully"
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)


def render_prediction_card(row: dict) -> None:
    gender = row.get("gender", "Both / Unisex")
    badge_class = "badge-both"
    if gender == "Male":
        badge_class = "badge-male"
    elif gender == "Female":
        badge_class = "badge-female"

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(
        f"<span class='result-badge {badge_class}'>{gender}</span>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"**Confidence:** `{row['gender_confidence']:.2%}`  \n"
        f"**Prediction Source:** `{row['prediction_source']}`  \n"
        f"**Normalized Name:** `{row['normalized_name'] or '(empty)'}`",
    )
    st.write(row["notes"])
    st.markdown("</div>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("<div class='metric-pill metric-male'>Male Probability</div>", unsafe_allow_html=True)
        st.progress(float(row["male_probability"]))
        st.caption(f"{row['male_probability']:.2%}")
    with col2:
        st.markdown("<div class='metric-pill metric-female'>Female Probability</div>", unsafe_allow_html=True)
        st.progress(float(row["female_probability"]))
        st.caption(f"{row['female_probability']:.2%}")
    with col3:
        st.markdown("<div class='metric-pill metric-both'>Both / Unisex Probability</div>", unsafe_allow_html=True)
        st.progress(float(row["both_probability"]))
        st.caption(f"{row['both_probability']:.2%}")


def main() -> None:
    hero_html = (
        '<div class="hero">'
        '<div class="hero-badge">AI Name Intelligence Platform</div>'
        '<h1>GenderClassy</h1>'
        '<p class="hero-subtitle">'
        'Predict gender from single names or enrich CSV/Excel files with ML-powered gender insights.'
        '</p>'
        '<div class="hero-meta">'
        '<span>⚡ Batch CSV/Excel Processing</span>'
        '<span>🌍 Multilingual Dataset</span>'
        '<span>🤖 Dictionary + ML Hybrid</span>'
        '</div>'
        '<div class="hero-author">'
        '<span>Made by <strong>Tarek Mahmoud Abdelrady</strong></span>'
        '<a href="https://www.linkedin.com/in/tarek-mahmoud-abdelradi-404884354/" target="_blank">LinkedIn ↗</a>'
        '</div>'
        '</div>'
    )

    st.markdown(hero_html, unsafe_allow_html=True)

    model_ok, model_status = get_model_status()

    with st.sidebar:
        st.title("Name Gender Predictor")
        st.write("Predict **Male**, **Female**, or **Both / Unisex** from names using local datasets.")
        if model_ok:
            st.success(f"Model status: {model_status}")
        else:
            st.warning("Model files were not found. Please run `python train_model.py` first.")
            st.caption(model_status)

        st.subheader("Dataset Files Detected")
        if DATA_DIR.exists():
            found = sorted([path.name for path in DATA_DIR.glob("*") if path.is_file()])
            for file_name in EXPECTED_DATA_FILES:
                if file_name in found:
                    st.markdown(f"- `{file_name}`")
            missing_files = [file_name for file_name in EXPECTED_DATA_FILES if file_name not in found]
            if missing_files:
                st.caption(f"Missing expected files: {len(missing_files)}")
        else:
            st.error("`data/` folder is missing.")

        st.subheader("Train Model")
        st.code("python train_model.py")

        st.subheader("Thresholds")
        st.caption(f"MIN_CONFIDENCE = {MIN_CONFIDENCE}")
        st.caption(f"UNISEX_MARGIN = {UNISEX_MARGIN}")
        st.caption(f"BOTH_THRESHOLD = {BOTH_THRESHOLD}")

        st.subheader("Disclaimer")
        st.caption(
            "Predictions are probabilistic and may vary by culture and language. "
            "Do not use this model for sensitive or high-stakes decisions."
        )

    if not model_ok:
        st.info("Model is not loaded yet. Most prediction actions are disabled until training artifacts are available.")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Single Name Prediction", "Batch Text Prediction", "Upload CSV / Excel", "About Model"]
    )

    with tab1:
        st.subheader("Single Name Prediction")
        input_name = st.text_input("Enter one name", placeholder="e.g., Ahmed, Maria, Alex")
        if st.button("Predict", key="single_predict_btn", use_container_width=True):
            if not model_ok:
                st.error("Model files were not found. Please run `python train_model.py` first.")
            elif not input_name.strip():
                st.warning("Please enter a name before prediction.")
            elif not normalize_name(input_name):
                st.warning("Input appears invalid after normalization. Please enter a valid name.")
            else:
                with st.spinner("Predicting gender..."):
                    result = predict_single_name(input_name)
                render_prediction_card(result)

    with tab2:
        st.subheader("Batch Text Prediction")
        batch_text = st.text_area(
            "Paste names (new lines, commas, or semicolons)",
            height=180,
            placeholder="John\nMaria; Ahmed, Alex",
        )
        if st.button("Predict All", key="batch_predict_btn", use_container_width=True):
            if not model_ok:
                st.error("Model files were not found. Please run `python train_model.py` first.")
            else:
                names = parse_text_names(batch_text)
                if not names:
                    st.warning("No valid names were detected in the text area.")
                else:
                    with st.spinner(f"Predicting {len(names):,} names..."):
                        result_df = predict_many_names(names)
                    st.success(f"Completed predictions for {len(result_df):,} names.")
                    st.dataframe(result_df, use_container_width=True)

                    st.download_button(
                        "Download CSV",
                        data=convert_df_to_csv(result_df),
                        file_name="gender_predictions.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )
                    st.download_button(
                        "Download Excel",
                        data=convert_df_to_excel(result_df),
                        file_name="gender_predictions.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True,
                    )

    with tab3:
        st.subheader("Upload CSV / Excel")
        uploaded_file = st.file_uploader("Upload file", type=["csv", "xlsx", "xls"])
        if uploaded_file is not None:
            try:
                df_uploaded = read_uploaded_table(uploaded_file)
                if df_uploaded.empty:
                    st.warning("Uploaded file is empty.")
                else:
                    st.write("Preview")
                    st.dataframe(df_uploaded.head(20), use_container_width=True)
                    st.caption(f"Rows: {len(df_uploaded):,} | Columns: {len(df_uploaded.columns):,}")
                    if len(df_uploaded) > 500_000:
                        st.warning("Large file detected. Processing may take longer.")

                    name_column = st.selectbox("Select the name column", options=df_uploaded.columns.tolist())
                    if st.button("Add Gender Column", key="upload_predict_btn", use_container_width=True):
                        if not model_ok:
                            st.error("Model files were not found. Please run `python train_model.py` first.")
                        else:
                            with st.spinner("Predicting genders for uploaded rows..."):
                                result_df = predict_dataframe(df_uploaded, name_column=name_column)
                            st.success(f"Processed {len(result_df):,} rows successfully.")
                            st.dataframe(result_df.head(20), use_container_width=True)

                            original_stem = Path(uploaded_file.name).stem
                            st.download_button(
                                "Download updated CSV",
                                data=convert_df_to_csv(result_df),
                                file_name=f"{original_stem}_with_gender.csv",
                                mime="text/csv",
                                use_container_width=True,
                            )
                            st.download_button(
                                "Download updated Excel",
                                data=convert_df_to_excel(result_df),
                                file_name=f"{original_stem}_with_gender.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                use_container_width=True,
                            )
            except Exception as exc:  # noqa: BLE001
                st.error(f"Could not process file: {exc}")

    with tab4:
        st.subheader("About Model")
        st.markdown(
            """
            **Datasets Used**
            - WGND 2.0 local files
            - UCI/name-gender local files from `data/`

            **Model Approach**
            - Name normalization with Unicode support (English, accented names)
            - Weighted dictionary lookup from merged WGND + UCI signals
            - Character-level TF-IDF (`2-5` grams) + Logistic Regression
            - Dictionary-first prediction, then ML fallback or mixed scoring

            **Label Meaning**
            - **Male**: strongest evidence for male usage
            - **Female**: strongest evidence for female usage
            - **Both / Unisex**: mixed/ambiguous evidence or low confidence

            **Limitations and Bias**
            - Name-based gender prediction is probabilistic.
            - Name usage varies by language, country, and culture.
            - Some names are genuinely unisex depending on region and time.
            - This system should not be used for sensitive decisions.
            """
        )

    st.markdown(
        "<div class='footer-note'>Built with Streamlit + scikit-learn. Local-only workflow using files in <code>data/</code>.</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
