import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def rank_by_similarity(user_profile: pd.Series, candidates: pd.DataFrame, feature_columns: list[str], weights = None) -> pd.DataFrame:
    if candidates.empty:
        return candidates
    
    candidate_features = candidates[feature_columns].copy()

    candidate_features = candidate_features.fillna(0)   #replace NaN with 0
    aligned_user_profile = user_profile.reindex(feature_columns).fillna(0) #align columns of user profile with feature columns


    if weights is None:
        sqrt_w = 1.0
    else:
        w = weights.reindex(feature_columns).fillna(0).clip(lower=0) #non negative and alignd
        sqrt_w = np.sqrt(w.values)

    scaled_profile    = aligned_user_profile.values * sqrt_w           # 1D
    scaled_candidates = candidate_features.values   * sqrt_w           # broadcasts over rows

    scores = cosine_similarity( #computes a cosine similarity
        scaled_profile.reshape(1, -1), #.values makes user profile a NumPy array [x,y,z] and reshape puts this NumPy array within another array (just bc cosine sim. expects to compare two 2d arrays) so in the end its [[x,y,z]]
        scaled_candidates,  #creates a NumPy array of NumPy array of the candidate song's features
    )[0]    #the result is an array of an array of scores (e.g.[[0.998, 0.444, 0.991]]) (in order of the original candidate song list) so [0] selects that array

    ranked_candidates = candidates.copy()   #first we create a copy
    ranked_candidates["similarity_score"] = scores  #add new column of scores

    ranked_candidates = ranked_candidates.sort_values(  #we then sort the songs
        by="similarity_score",
        ascending=False,
    )

    return ranked_candidates    #return sorted candidates with concatenated ranks