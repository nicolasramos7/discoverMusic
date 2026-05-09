from discover_music.pipelines.prepare_data import run_data_prep
from discover_music.utils.paths import RAW_DATA_DIR, INTERIM_DATA_DIR, PROCESSED_DATA_DIR


if __name__ == "__main__":
    input_path = RAW_DATA_DIR / "spotify_tracks_30000.csv"

    run_data_prep(
        input_path=input_path,
        interim_dir=INTERIM_DATA_DIR,
        processed_dir=PROCESSED_DATA_DIR,
    )