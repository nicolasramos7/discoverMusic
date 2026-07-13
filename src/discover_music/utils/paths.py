from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
LOCAL_DATA_DIR = DATA_DIR / "local"
TRACKS_PATH = PROCESSED_DATA_DIR / "tracks.parquet"
RECOMMENDATION_FEATURES_PATH = PROCESSED_DATA_DIR / "recommendation_features.parquet"
USER_RATINGS_PATH = LOCAL_DATA_DIR / "user_ratings.csv"
EVAL_DIR          = DATA_DIR / "eval"
PERSONAS_DIR      = EVAL_DIR / "personas"
EVAL_RESULTS_DIR  = EVAL_DIR / "results"


def ensure_local_data_dir() -> None:
    LOCAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PERSONAS_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)