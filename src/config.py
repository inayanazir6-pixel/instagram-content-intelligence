"""Central configuration: paths, file names, label meanings and validation rules.

Every other module imports from here, so nothing is hard-coded twice.
"""
from pathlib import Path

# Project root = the folder that contains `src/`
ROOT = Path(__file__).resolve().parents[1]

# ---- Folders (medallion layout: raw -> bronze -> silver -> gold) -------------
RAW_DIR = ROOT / "data" / "raw"        # files exactly as downloaded
BRONZE_DIR = ROOT / "data" / "bronze"  # same content, typed and stored as Parquet
SILVER_DIR = ROOT / "data" / "silver"  # cleaned data (Day 2)
GOLD_DIR = ROOT / "data" / "gold"      # analysis results the app reads (Day 4)
REPORTS_DIR = ROOT / "reports"         # validation reports

# ---- Input files -------------------------------------------------------------
DAVIDSON_FILE = RAW_DIR / "davidson.csv"
EDOS_FILE = RAW_DIR / "edos.csv"
STANCE_FILES = {
    "train": RAW_DIR / "stance_train.csv",
    "validation": RAW_DIR / "stance_val.csv",
    "test": RAW_DIR / "stance_test.csv",
}

# ---- Label meanings ----------------------------------------------------------
# Davidson: documented on the Kaggle page (class column).
DAVIDSON_LABELS = {0: "hate_speech", 1: "offensive", 2: "neither"}
# TweetEval stance: 0 = none, 1 = against, 2 = favor (check against the dataset
# card; Day 2 will confirm it by reading sample tweets).
STANCE_LABELS = {0: "none", 1: "against", 2: "favor"}

# ---- Validation rules per dataset (used by src/validate.py) ------------------
DATASET_SPECS = {
    "davidson": {
        "text_col": "text",
        "label_col": "label",
        "allowed_labels": [0, 1, 2],
        "label_names": DAVIDSON_LABELS,
        "distributions": ["label"],
    },
    "edos": {
        "text_col": "text",
        "label_col": "label_sexist",
        "allowed_labels": ["sexist", "not sexist"],
        "label_names": None,
        "distributions": ["label_sexist", "label_category", "split"],
    },
    "stance": {
        "text_col": "text",
        "label_col": "label",
        "allowed_labels": [0, 1, 2],
        "label_names": STANCE_LABELS,
        "distributions": ["label", "split"],
    },
}


# ---- Day 2 settings ----------------------------------------------------------
QUARANTINE_DIR = ROOT / "data" / "quarantine"
MIN_WORDS = 2  # comments with fewer words after cleaning go to quarantine
EXTRA_STOPWORDS = {
    "user", "url", "amp", "rt", "semst", "like", "get", "got", "one", "would",
    "dont", "im", "thats", "i'm", "it's", "that's", "don't", "can't",
}