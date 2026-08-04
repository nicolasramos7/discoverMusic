import pandas as pd

from discover_music.recommender.profile import build_user_profile
from discover_music.recommender.ratings import get_rated_track_ids, load_ratings
from discover_music.recommender.similarity import rank_by_similarity
from discover_music.utils.paths import RECOMMENDATION_FEATURES_PATH, TRACKS_PATH


TRACK_ID_COLUMN = "track_id"

def load_tracks() -> pd.DataFrame:
    tracks = pd.read_parquet(TRACKS_PATH)

    if TRACK_ID_COLUMN not in tracks.columns:
        raise ValueError(f"tracks.parquet must contain '{TRACK_ID_COLUMN}' column.")
    
    tracks[TRACK_ID_COLUMN] = tracks[TRACK_ID_COLUMN].astype(str)   #saves ids into strings and saves df

    return tracks

def load_recommendation_features() -> pd.DataFrame:
    features = pd.read_parquet(RECOMMENDATION_FEATURES_PATH)

    if TRACK_ID_COLUMN not in features.columns:
        raise ValueError(
            f"recommendation_features.parquet must contain '{TRACK_ID_COLUMN}' column."
        )
    
    features[TRACK_ID_COLUMN] = features[TRACK_ID_COLUMN].astype(str)   #svaes ids into strings and saves df

    return features

def get_valid_candidate_pool() -> pd.DataFrame:
    tracks = load_tracks()
    features = load_recommendation_features()

    candidate_pool = tracks.merge(  #merges tracks with recomendation features based on id using an inner join, if a column has same name, add a "_feature if its from the feature table"
        features,
        on=TRACK_ID_COLUMN,
        how="inner",
        suffixes=("", "_feature"),
    )

    return candidate_pool

def get_feature_columns(candidate_pool: pd.DataFrame) -> list[str]:
    excluded_columns = {TRACK_ID_COLUMN}

    numeric_columns = candidate_pool.select_dtypes(include=["number"]).columns.tolist() #select list of columns where data type is number

    feature_columns = [
        column for column in numeric_columns if column not in excluded_columns  #any column in numeric column that isnt in excluded column
    ]

    if not feature_columns:
        raise ValueError("No numeric feature columns found for recommendation.")

    return feature_columns

def get_unrated_candidates() -> pd.DataFrame:
    candidate_pool = get_valid_candidate_pool()
    rated_track_ids = get_rated_track_ids()

    if not rated_track_ids: #if none are rated, all are candidate
        return candidate_pool

    unrated_candidates = candidate_pool[
        ~candidate_pool[TRACK_ID_COLUMN].astype(str).isin(rated_track_ids)
    ].copy()    #creates a copy of all entries in candidate pool which arent in rated_track_ids

    return unrated_candidates

def get_random_songs(n: int = 10) -> pd.DataFrame: #default 10 if unspecified
    unrated_candidates = get_unrated_candidates()

    if unrated_candidates.empty:
        return unrated_candidates

    sample_size = min(n, len(unrated_candidates))   #we cant get 10 songs if there are less than 10 songs in the unrated_cand. table

    return unrated_candidates.sample(n=sample_size) #use pandas sample function to select n random songs

def get_recommended_songs(n: int = 10) -> pd.DataFrame: #default 10 if unspecified
    ratings = load_ratings()
    features = load_recommendation_features()
    unrated_candidates = get_unrated_candidates()

    if unrated_candidates.empty:    #if no songs to give, just return empty df
        return unrated_candidates
    
    user_profile = build_user_profile(ratings, features)    #use method in profile.py

    if user_profile is None:
        return get_random_songs(n=n)    #if user hasnt got a profile, we can select 10 random songs
    
    feature_columns = get_feature_columns(features)   #only the recommendation feature columns (the *_scaled ones), NOT the raw numeric columns that the tracks merge leaks into the candidate pool

    ranked_candidates = rank_by_similarity( #rank using similarity.py helper
        user_profile=user_profile,
        candidates=unrated_candidates,
        feature_columns=feature_columns,
    )

    return ranked_candidates.head(n)    #get top 10 using head method in pandas

def get_initial_batch(n: int = 10) -> pd.DataFrame: #default 10 if unspecified
    ratings = load_ratings()

    if ratings.empty:
        return get_random_songs(n=n)    #just get random number of 

    return get_recommended_songs(n=n)   #if ratings is not empty, we have to get recommended songs

def get_next_song() -> pd.DataFrame:
    return get_recommended_songs(n=1)   #here instead of leaving unspecified, we call with a specific n = 1 value
