import pandas as pd
from sklearn.preprocessing import StandardScaler

RECOMMENDATION_FEATURES = [
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "duration_ms",
]

GENRES = ["edm", "latin", "pop", "r&b", "rap", "rock"]
GENRE_WEIGHT = 1.0  

def build_recommendation_features(df: pd.DataFrame) -> pd.DataFrame:

    feature_df = df[["track_id", "playlist_id", "playlist_genre", *RECOMMENDATION_FEATURES]].copy()

    feature_df = feature_df.sort_values(["track_id", "playlist_id"]) #same deterministic genre selection as the eval.
    feature_df = feature_df.drop_duplicates(subset=["track_id"])
    feature_df = feature_df.dropna(subset=RECOMMENDATION_FEATURES)

    scaler = StandardScaler()   #create instance of scalar

    scaled_values = scaler.fit_transform(feature_df[RECOMMENDATION_FEATURES])   #select numerical columns then fit learns the mean and standard dev of each column and transform converts then scales according to a common scale
    #scaled values is now a NumPy array

    scaled_columns = [f"{col}_scaled" for col in RECOMMENDATION_FEATURES] #creates a new list of column names ending in "_scaled"

    scaled_df = pd.DataFrame(   #creates new Pandas DataFrame with
        scaled_values,  #actual data
        columns=scaled_columns, #column names
        index=feature_df.index, #pandas gives DataFrame rows an index, if we drop duplicates, the indexes might not align so we tell it to match indexes to match data
    )

    # fixed category list so every run yields the same 6 columns in the same order,
    # and unknown/NaN genre -> all-zero row (neutral)
    genre_series = feature_df["playlist_genre"].where(feature_df["playlist_genre"].isin(GENRES))
    genre_onehot = pd.get_dummies(
        genre_series.astype(pd.CategoricalDtype(categories=GENRES)),  # Series keeps its index
        prefix="genre",
    ).astype(float) * GENRE_WEIGHT

    result = pd.concat([feature_df[["track_id"]], scaled_df, genre_onehot], axis=1)

    return result