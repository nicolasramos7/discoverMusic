import pandas as pd

REQUIRED_COLUMNS = [    #array of strings of required columns
    "track_id",
    "track_name",
    "track_artist",
    "track_popularity",
    "track_album_id",
    "track_album_name",
    "track_album_release_date",
    "playlist_name",
    "playlist_id",
    "playlist_genre",
    "playlist_subgenre",
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

NUMERIC_COLUMNS = [ #array of strings of numeric columns
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


RANGE_COLUMNS = {   #dictionary of range columns with name and range
    "track_popularity": (0, 100),
    "danceability": (0, 1),
    "energy": (0, 1),
    "speechiness": (0, 1),
    "acousticness": (0, 1),
    "instrumentalness": (0, 1),
    "liveness": (0, 1),
    "valence": (0, 1),
    "mode": (0, 1),
}

def validate_required_columns(df: pd.DataFrame) -> None:    #void method that gets a data frame and checks req. cols.
    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    #list compression that reads as:
    #create a list of col for each col in REQUIRED_COLUMNS only if not in df.columns

    if missing_columns: #will return true if there is atleast one col in missing_columns
        raise ValueError(f"Missing required columns: {missing_columns}")    #raise error
    
def validate_numeric_columns(df: pd.DataFrame) -> None:
    for col in NUMERIC_COLUMNS:
        if col not in df.columns:   #wont reach this because validate_required_columns checks
            continue

        #count how many values in that column are not valid numbers or are missing:
        #df[col] is an array of numeric values at column col in question
        #pd.to_numeric tries to convert all entries into numbers, in the case of failure, errors="coerce" means failures are corrected to NaN
        #isna() then converts this array of numeric values into true/false entires depending if they are NaN or not, sum then returns the total
        invalid_count = pd.to_numeric(df[col], errors="coerce").isna().sum() 

        if invalid_count > 0:   #if there is any invalid, they are addressed here
            print(f"Warning: {col} has {invalid_count} non-numeric or missing values.")

def validate_ranges(df: pd.DataFrame) -> None:
    for col, (min_value, max_value) in RANGE_COLUMNS.items():   #for each entry in dictionary mentioned before
        if col not in df.columns:   #same as before, wont be reached
            continue

        out_of_range = df[(df[col] < min_value) | (df[col] > max_value)]
        #df[col] < min_value checks every value in that column and returns True if the value is below the minimum.
        #df[col] > max_value col does the same but other way around
        #each value is OR and df[] stores the values for which OR is true

        if len(out_of_range) > 0:   #when there are values out of range, they are displayed
            print(
                f"Warning: {col} has {len(out_of_range)} values outside "
                f"expected range [{min_value}, {max_value}]."
            )

def validate_tracks(df: pd.DataFrame) -> None:  #main method that will be executed, which runs methods in order
    validate_required_columns(df)   #important: this should be first to correctly validate
    validate_numeric_columns(df)
    validate_ranges(df)