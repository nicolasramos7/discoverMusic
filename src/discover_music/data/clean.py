import pandas as pd


TEXT_COLUMNS = [
    "track_id",
    "track_name",
    "track_artist",
    "track_album_id",
    "track_album_name",
    "playlist_name",
    "playlist_id",
    "playlist_genre",
    "playlist_subgenre",
]


NUMERIC_COLUMNS = [
    "track_popularity",
    "danceability",
    "energy",
    "key",
    "loudness",
    "mode",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "duration_ms",
]

def clean_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in TEXT_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype("string").str.strip()  #converts to pandas type string and removes empty spaces

    return df

def clean_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")   #converts all entries into nubers and in the case of failure, NaN

    return df

def clean_release_date(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["release_date_raw"] = df["track_album_release_date"] #This creates a new column called: "release_date_raw" 
    # It stores the original release date values before cleaning them.

    df["track_album_release_date"] = pd.to_datetime( #converts and replaces dates in "track_album_release_date" into dateTime values
        df["track_album_release_date"],
        errors="coerce" #convets into NaT in the case of error (Not a Time)
    )

    df["release_year"] = df["track_album_release_date"].dt.year #new column is made getting year from date time
    df["release_month"] = df["track_album_release_date"].dt.month #new column is made getting month from date time
    df["release_day"] = df["track_album_release_date"].dt.day   #new column is made getting day from date time

    return df

def remove_bad_rows(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    before = len(df) #grab amount of entries before

    df = df.dropna( #removes entries with missing values
        subset=[    #tells pandas which columns to check for missing values.
            "track_id",
            "track_name",
            "track_artist",
            "danceability",
            "energy",
            "valence",
            "tempo",
            "duration_ms",
        ]
    )

    df = df[df["duration_ms"] > 0]  #only keeps entries with duration over 0
    df = df[df["tempo"] > 0]    #only keeps entries with tempo over 0

    after = len(df) #get length after

    print(f"Removed {before - after} bad rows.") 

    return df

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    before = len(df)

    df = df.drop_duplicates()

    after = len(df)

    print(f"Removed {before - after} exact duplicate rows.")

    return df

def clean_tracks(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df = clean_text_columns(df)
    df = clean_numeric_columns(df)
    df = clean_release_date(df)
    df = remove_bad_rows(df)
    df = remove_duplicates(df)

    return df