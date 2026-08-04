import json
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from discover_music.utils.paths import (
    TRACKS_PATH,
    RECOMMENDATION_FEATURES_PATH,
    TRACK_PLAYLIST_PATH,
    PLAYLISTS_PATH,
    PERSONAS_DIR,
    ensure_eval_dirs,
)

GENRES = ["edm", "latin", "pop", "r&b", "rap", "rock"]
N_PROFILE = 15  #how many liked songs define a persona
N_NEGATIVE = 10 #how many "diskliked" songs
K_SET = 10  #size of each answer key/set


def feature_neighbors(profile_ids, features, feature_cols, exclude_ids, k=K_SET):
    #similar to model, apply cosine similarity and find top k tracks
    centroid = features.loc[features.track_id.isin(profile_ids), feature_cols].mean()
    others   = features[~features.track_id.isin(exclude_ids)].copy()
    sims = cosine_similarity(centroid.values.reshape(1, -1), others[feature_cols].values)[0]
    others["sim"] = sims
    return others.sort_values("sim", ascending=False).track_id.head(k).tolist()


def similar_sounding_playlist(profile_ids, track_playlist, exclude, k=K_SET):
    #find songs that share the most amount of playlists
    profile_playlists = (track_playlist[track_playlist.track_id.isin(profile_ids)]
                         .playlist_id.unique()) #find all playlists from profile
    siblings = track_playlist[track_playlist.playlist_id.isin(profile_playlists)]   #all songs in playlists
    counts = (siblings[~siblings.track_id.isin(exclude | set(profile_ids))]
                .track_id.value_counts())   #add a count of appearances
    ordered = counts.reset_index()  #order the counts
    ordered.columns = ["track_id", "n"]
    ordered = ordered.sort_values(["n", "track_id"], ascending=[False, True])   #order by n and in the case of tie, by track_id
    return ordered["track_id"].head(k).tolist()

def persona_specs(n_personas, rng):
    #creates all possible pairs of liked and disliked genres and shuffles their order
    pairs = [(a, b) for a in GENRES for b in GENRES if a != b]
    rng.shuffle(pairs)
    specs = []
    for i in range(n_personas):
        liked, disliked = pairs[i % len(pairs)]
        specs.append({
            "id": f"persona_{i + 1:02d}",
            "liked_genre": liked,
            "disliked_genre": disliked,
        })
    return specs


def _sample_ids(pool, n, rng):
    #basically just get n random track ids
    ids = pool["track_id"].tolist()
    n = min(n, len(ids))
    idx = rng.choice(len(ids), size=n, replace=False)
    return [ids[i] for i in idx]


def select_coherent_taste(tracks, spec, rng):
    #persona's 'liked' songs — sampled from its liked genre
    pool = tracks[tracks["genre"] == spec["liked_genre"]]
    return _sample_ids(pool, N_PROFILE, rng)


def select_disliked_taste(tracks, spec, rng):
    #persona's 'disliked' songs — sampled from its disliked genre.
    pool = tracks[tracks["genre"] == spec["disliked_genre"]]
    return _sample_ids(pool, N_NEGATIVE, rng)


def build_persona(spec, tracks, features, feature_cols, track_playlist, rng):
    profile_ids  = select_coherent_taste(tracks, spec, rng)
    negative_ids = select_disliked_taste(tracks, spec, rng)
    used = set(profile_ids) | set(negative_ids)

    set_a_ids = feature_neighbors(profile_ids, features, feature_cols, used, k=K_SET)
    set_b_ids = similar_sounding_playlist(profile_ids, track_playlist, used, k=K_SET)

    #this is better than a persona having empty stuff later
    assert set_a_ids, f"{spec['id']}: Set A is empty"
    assert set_b_ids, f"{spec['id']}: Set B is empty (profile songs share no playlists)"

    return {
        "id":           spec["id"],
        "liked_genre":  spec["liked_genre"],
        "disliked_genre": spec["disliked_genre"],
        "profile_ids":  profile_ids,
        "negative_ids": negative_ids,
        "set_a_ids":    set_a_ids,
        "set_b_ids":    set_b_ids,
    }


def _build_genre_lookup(track_playlist, playlists):
    #genre doesn't live on tracks, it lives on playlists, and a track can be in many playlists.
    merged = track_playlist.merge(
        playlists[["playlist_id", "playlist_genre", "playlist_subgenre"]],
        on="playlist_id",
    )   # joins the join table to the genre metadata, so each (track_id, playlist_id) row now carries a genre too.
    merged = merged.sort_values(["track_id", "playlist_id"])    #orders rows deterministically.
    first = merged.drop_duplicates("track_id")
    return first.rename(columns={
        "playlist_genre":    "genre",
        "playlist_subgenre": "subgenre",
    })[["track_id", "genre", "subgenre"]]


def build_and_freeze(n_personas, seed=42):
    #uses all other methods to build and store as JSON
    ensure_eval_dirs()
    rng = np.random.default_rng(seed)
    tracks = pd.read_parquet(TRACKS_PATH)
    features = pd.read_parquet(RECOMMENDATION_FEATURES_PATH)
    track_playlist = pd.read_parquet(TRACK_PLAYLIST_PATH)
    playlists = pd.read_parquet(PLAYLISTS_PATH)

    genre_lookup = _build_genre_lookup(track_playlist, playlists)
    tracks = tracks.merge(genre_lookup, on="track_id", how="left")

    feature_cols = [c for c in features.columns if c.endswith("_scaled")]

    for spec in persona_specs(n_personas, rng):
        persona = build_persona(spec, tracks, features, feature_cols, track_playlist, rng)
        out_path = PERSONAS_DIR / f"{persona['id']}.json"
        with open(out_path, "w") as f:
            json.dump(persona, f, indent=2)
        print(f"  saved {out_path.name}  ({spec['liked_genre']} vs {spec['disliked_genre']})")