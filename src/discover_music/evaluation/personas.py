import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from discover_music.utils.paths import (
    TRACKS_PATH, RECOMMENDATION_FEATURES_PATH, PERSONAS_DIR, ensure_eval_dirs
)

def feature_neighbors(profile_ids, features, feature_cols, exclude_ids, k=10):
    #make user taste a single vector
    centroid = features.loc[features.track_id.isin(profile_ids), feature_cols].mean() #select only tracks in profile_ids from feature then create a mean
    
    others = features[~features.track_id.isin(exclude_ids)]     #removes exluded sonts (flips true selected into false to select falsy but not excluded ids)

    #computer how similar is each remaining song to centroid
    sims = cosine_similarity(centroid.values.reshape(1, -1), others[feature_cols].values)[0] #first reshapes into [[centroid features]] because thats what cosine expects, then cosine similarities between both

    others = others.assign(sim=sims).sort_values("sim", ascending=False)    #adds new column called sim and sorts based on it

    return others.track_id.head(k).tolist() #returns list form


def build_persona(spec, tracks, features, feature_cols, rng):
    profile_ids  = select_coherent_taste(tracks, spec, rng)     # taste definition
    #negative_ids = select_disliked_taste(tracks, spec, rng)     # for future grading where disliked songs have an effect
    used = set(profile_ids) | set(negative_ids) #creates exlusion pool -> tracks that cant be given to test
    return {
        "id":           spec["id"], #gives person id
        "profile_ids":  profile_ids,    #ids of songs tested on
        #"negative_ids": negative_ids,   #for later implementation
        "set_a_ids":    feature_neighbors(profile_ids, features, feature_cols, used, k=10),
        "set_b_ids":    your_similar_sounding_method(profile_ids, rng),   # TO-DO develop this method for returning similar sounding songs
    }

def build_and_freeze(n_personas, seed=42):
    ensure_eval_dirs()
    rng      = make_rng(seed)   #one single rng object is created here and passed into every build_persona call
    tracks   = pd.read_parquet(TRACKS_PATH)                      # names/artist/popularity/genre-joins
    features = pd.read_parquet(RECOMMENDATION_FEATURES_PATH)    # loads the scaled audio features needed to build set A
    feature_cols = [c for c in features.columns if c.endswith("_scaled")]   #keeps scaled columns only
    for spec in persona_specs(n_personas, rng): #need to make to say how each person looks like/specs
        persona = build_persona(spec, tracks, features, feature_cols, rng)  #creats person with specs
        write_json(PERSONAS_DIR / f"{persona['id']}.json", persona) #writes to JSON