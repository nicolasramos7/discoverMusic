import pandas as pd


TRACK_ID_COLUMN = "track_id"
POSITIVE_RATING_THRESHOLD = 4

def build_user_profile(ratings: pd.DataFrame, features: pd.DataFrame) -> pd.Series | None:
    if ratings.empty:
        return None
    
    positive_ratings = ratings[ratings["rating"] >= POSITIVE_RATING_THRESHOLD].copy()

    if positive_ratings.empty:
        return None
    
    positive_ratings[TRACK_ID_COLUMN] = positive_ratings[TRACK_ID_COLUMN].astype(str)
    features = features.copy()
    features[TRACK_ID_COLUMN] = features[TRACK_ID_COLUMN].astype(str)

    liked_song_features = features[ #get features of liked songs
        features[TRACK_ID_COLUMN].isin(positive_ratings[TRACK_ID_COLUMN])
    ].copy()

    if liked_song_features.empty:
        return None
    
    numeric_feature_columns = liked_song_features.select_dtypes(    #get numeric features of liked songs
        include=["number"]
    ).columns.tolist()

    numeric_feature_columns = [
        column for column in numeric_feature_columns if column != TRACK_ID_COLUMN   #remove track_id column
    ]

    if not numeric_feature_columns:
        return None
    
    user_profile = liked_song_features[numeric_feature_columns].mean()  #make an average of all liked songs

    return user_profile