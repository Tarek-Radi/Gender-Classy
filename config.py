from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"


EXPECTED_DATA_FILES = [
    "name_gender_1950-2018.csv",
    "name_gender_all.csv",
    "name_gender_dataset.csv",
    "wgnd_2_0_code-langcode.csv",
    "wgnd_2_0_name-gender_nocode.csv",
    "wgnd_2_0_name-gender-code.csv",
    "wgnd_2_0_name-gender-code_langexp.csv",
    "wgnd_2_0_name-gender-langcode.csv",
    "wgnd_2_0_sources.csv",
]


NAME_COLUMN_CANDIDATES = [
    "name",
    "first_name",
    "firstname",
    "given_name",
    "fullname",
    "full_name",
]

GENDER_COLUMN_CANDIDATES = ["gender", "sex"]

WEIGHT_COLUMN_CANDIDATES = ["count", "probability", "prob", "wgt", "nobs", "weight"]

CSV_SEPARATORS = [",", ";", "\t", "|"]
CSV_ENCODINGS = ["utf-8-sig", "utf-8", "cp1252", "latin1"]
CHUNK_SIZE = 250_000


UNISEX_MARGIN = 0.15
MIN_CONFIDENCE = 0.60
BOTH_THRESHOLD = 0.25
USE_FIRST_TOKEN_ONLY = True
MAX_FEATURES = 250_000
TEST_SIZE = 0.20
RANDOM_STATE = 42
MAX_MODEL_TRAIN_ROWS = 600_000


MODEL_FILE = MODELS_DIR / "gender_model.joblib"
VECTORIZER_FILE = MODELS_DIR / "vectorizer.joblib"
LABEL_ENCODER_FILE = MODELS_DIR / "label_encoder.joblib"
DICTIONARY_FILE = MODELS_DIR / "dictionary_lookup.joblib"

METRICS_FILE = OUTPUTS_DIR / "metrics.json"
REPORT_FILE = OUTPUTS_DIR / "training_report.txt"


LABELS = ("male", "female", "both")
LABEL_TO_INDEX = {"male": 0, "female": 1, "both": 2}
INDEX_TO_LABEL = {0: "male", 1: "female", 2: "both"}


PREDICTION_COLUMNS = [
    "gender",
    "gender_confidence",
    "male_probability",
    "female_probability",
    "both_probability",
    "prediction_source",
    "normalized_name",
]


def ensure_directories() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
