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
    "track_popularity",
]

def build_recommendation_features(df: pd.DataFrame) -> pd.DataFrame:

    feature_df = df[
        ["track_id", *RECOMMENDATION_FEATURES]
    ].copy()    #builds a normalized table version (copy)

    feature_df = feature_df.drop_duplicates(subset=["track_id"])    #remove duplicates based on track_id
    feature_df = feature_df.dropna(subset=RECOMMENDATION_FEATURES)  #drop NaN entries

    scaler = StandardScaler()   #create instance of scalar

    scaled_values = scaler.fit_transform(feature_df[RECOMMENDATION_FEATURES])   #select numerical columns then fit learns the mean and standard dev of each column and transform converts then scales according to a common scale
    #scaled values is now a NumPy array

    scaled_columns = [f"{col}_scaled" for col in RECOMMENDATION_FEATURES] #creates a new list of column names ending in "_scaled"

    scaled_df = pd.DataFrame(   #creates new Pandas DataFrame with
        scaled_values,  #actual data
        columns=scaled_columns, #column names
        index=feature_df.index, #pandas gives DataFrame rows an index, if we drop duplicates, the indexes might not align so we tell it to match indexes to match data
    )

    result = pd.concat( #finally we combine the track ids with the scaled data
        [
            feature_df[["track_id"]],
            scaled_df,
        ],
        axis=1, #this means concat horizontally column by column
    )

    return result

