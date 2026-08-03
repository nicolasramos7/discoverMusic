import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from discover_music.utils.paths import (
    TRACKS_PATH, RECOMMENDATION_FEATURES_PATH, PERSONAS_DIR, ensure_eval_dirs, TRACK_PLAYLIST_PATH
)

def feature_neighbors(profile_ids, features, feature_cols, exclude_ids, k=10):
    #make user taste a single vector
    centroid = features.loc[features.track_id.isin(profile_ids), feature_cols].mean() #select only tracks in profile_ids from feature then create a mean
    
    others = features[~features.track_id.isin(exclude_ids)]     #removes exluded sonts (flips true selected into false to select falsy but not excluded ids)

    #computer how similar is each remaining song to centroid
    sims = cosine_similarity(centroid.values.reshape(1, -1), others[feature_cols].values)[0] #first reshapes into [[centroid features]] because thats what cosine expects, then cosine similarities between both

    others = others.assign(sim=sims).sort_values("sim", ascending=False)    #adds new column called sim and sorts based on it

    return others.track_id.head(k).tolist() #returns list form


def build_persona(spec, tracks, features, feature_cols, rng, track_playlist):
    profile_ids  = select_coherent_taste(tracks, spec, rng)     # taste definition
    #negative_ids = select_disliked_taste(tracks, spec, rng)     # for future grading where disliked songs have an effect
    used = set(profile_ids) | set(negative_ids) #creates exlusion pool -> tracks that cant be given to test
    set_b_ids = similar_sounding_playlist(profile_ids, track_playlist, used, k=10)
    return {
        "id":           spec["id"], #gives person id
        "profile_ids":  profile_ids,    #ids of songs tested on
        #"negative_ids": negative_ids,   #for later implementation
        "set_a_ids":    feature_neighbors(profile_ids, features, feature_cols, used, k=10),
        "set_b_ids":    your_similar_sounding_method(profile_ids, rng),   # TO-DO develop this method for returning similar sounding songs
    }

def build_and_freeze(n_personas, seed=42):
    ensure_eval_dirs()
    track_playlist = pd.read_parquet(TRACK_PLAYLIST_PATH)
    rng      = make_rng(seed)   #one single rng object is created here and passed into every build_persona call
    tracks   = pd.read_parquet(TRACKS_PATH)                      # names/artist/popularity/genre-joins
    features = pd.read_parquet(RECOMMENDATION_FEATURES_PATH)    # loads the scaled audio features needed to build set A
    feature_cols = [c for c in features.columns if c.endswith("_scaled")]   #keeps scaled columns only
    for spec in persona_specs(n_personas, rng): #need to make to say how each person looks like/specs
        persona = build_persona(spec, tracks, features, feature_cols, track_playlist, rng)  #creats person with specs
        write_json(PERSONAS_DIR / f"{persona['id']}.json", persona) #writes to JSON

def similar_sounding_playlist(profile_ids, track_playlist, exclude, k=10):
    # playlists that contain the persona's profile songs
    profile_playlists = (track_playlist[track_playlist.track_id.isin(profile_ids)]
                         .playlist_id.unique())

    # every other song sitting in those same playlists
    siblings = track_playlist[track_playlist.playlist_id.isin(profile_playlists)]

    # co-occurrence count (mostly 1 on this dataset — see note)
    counts = (siblings[~siblings.track_id.isin(exclude | set(profile_ids))]
              .track_id.value_counts())

    # deterministic tie-break: count desc, then track_id, so the frozen set is stable
    ordered = counts.reset_index()
    ordered.columns = ["track_id", "n"]
    ordered = ordered.sort_values(["n", "track_id"], ascending=[False, True])

    return ordered["track_id"].head(k).tolist()