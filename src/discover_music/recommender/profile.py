import pandas as pd

TRACK_ID_COLUMN = "track_id"
POSITIVE_RATING_THRESHOLD = 4
NEGATIVE_RATING_THRESHOLD = 2

# profile = alpha·mean(liked) − beta·mean(disliked)
ALPHA = 1.0
BETA =  0.25

def build_user_profile(ratings: pd.DataFrame, features: pd.DataFrame) -> pd.Series | None:
    if ratings.empty:
        return None, None

    features = features.copy()
    features[TRACK_ID_COLUMN] = features[TRACK_ID_COLUMN].astype(str)

    #liked centroid, important bc no likes = no profile
    positive_ratings = ratings[ratings["rating"] >= POSITIVE_RATING_THRESHOLD].copy()
    if positive_ratings.empty:
        return None, None
    positive_ratings[TRACK_ID_COLUMN] = positive_ratings[TRACK_ID_COLUMN].astype(str)

    liked_features = features[
        features[TRACK_ID_COLUMN].isin(positive_ratings[TRACK_ID_COLUMN])
    ]
    if liked_features.empty:
        return None, None

    #use one column list
    feature_columns = [
        c for c in liked_features.select_dtypes(include=["number"]).columns
        if c != TRACK_ID_COLUMN
    ]
    if not feature_columns:
        return None, None

    liked_centroid = liked_features[feature_columns].mean()

    #disliked centroid, optional: zero vector when there are none
    negative_ratings = ratings[ratings["rating"] <= NEGATIVE_RATING_THRESHOLD].copy()
    negative_ratings[TRACK_ID_COLUMN] = negative_ratings[TRACK_ID_COLUMN].astype(str)

    disliked_features = features[
        features[TRACK_ID_COLUMN].isin(negative_ratings[TRACK_ID_COLUMN])
    ]
    if not disliked_features.empty:
        disliked_centroid = disliked_features[feature_columns].mean()
    else:
        disliked_centroid = pd.Series(0.0, index=feature_columns) #zero vector

    profile = ALPHA * liked_centroid - BETA * disliked_centroid

    weights = (liked_centroid - disliked_centroid).abs()

    return profile, weights