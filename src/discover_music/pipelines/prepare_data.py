from pathlib import Path

from discover_music.data.load import load_tracks_csv
from discover_music.data.validate import validate_tracks
from discover_music.data.clean import clean_tracks
from discover_music.data.split_tables import split_into_tables
from discover_music.data.features import build_recommendation_features


def save_table(df, output_dir: Path, name: str) -> None:
    csv_path = output_dir / f"{name}.csv"   #adds csv endpoint to passed path
    parquet_path = output_dir / f"{name}.parquet"   #creates endpoint for parquet file

    df.to_csv(csv_path, index=False)    #saves given DataFrame into csv without the Pandas DataFrame Column IDs
    df.to_parquet(parquet_path, index=False) #saves given DataFrame into parquet without the Pandas DataFrame Column IDs

    print(f"Saved {csv_path}")
    print(f"Saved {parquet_path}")


def run_data_prep(input_path: str | Path, interim_dir: str | Path, processed_dir: str | Path) -> None:
    #create paths if not paths already
    input_path = Path(input_path)
    interim_dir = Path(interim_dir)
    processed_dir = Path(processed_dir)

    #creates interim and processed folders if they dont exist
    interim_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    print("Loading raw data...")
    raw_df = load_tracks_csv(input_path)    #load raw df using load.py

    print(f"Raw shape: {raw_df.shape}")

    print("Validating raw data...")
    validate_tracks(raw_df) #validate using validate.py

    print("Cleaning data...")
    cleaned_df = clean_tracks(raw_df)   #clean using clean.py

    print(f"Cleaned shape: {cleaned_df.shape}")

    #prep location and clean dataframes
    cleaned_csv_path = interim_dir / "tracks_cleaned.csv"
    cleaned_parquet_path = interim_dir / "tracks_cleaned.parquet"

    cleaned_df.to_csv(cleaned_csv_path, index=False)
    cleaned_df.to_parquet(cleaned_parquet_path, index=False)

    print(f"Saved {cleaned_csv_path}")
    print(f"Saved {cleaned_parquet_path}")

    print("Splitting into project tables...")
    tables = split_into_tables(cleaned_df) #split into different tables using split_tables.py

    for name, table in tables.items():
        save_table(table, processed_dir, name)  #save each individual table

    print("Building recommendation feature matrix...")
    recommendation_features = build_recommendation_features(cleaned_df) #prep features using features.py

    save_table(recommendation_features, processed_dir, "recommendation_features")   #save features

    print("Data preparation complete.")