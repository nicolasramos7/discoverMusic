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

GENRES      = ["edm", "latin", "pop", "r&b", "rap", "rock"] # the 6 real genres
N_PROFILE   = 15 # how many "liked" songs define a persona
N_NEGATIVE  = 10 # how many "disliked" songs
K_SET       = 10  # size of each answer key (Set A and Set B)


def feature_neighbors(profile_ids, features, feature_cols, exclude_ids, k=K_SET):
    """Set A — the k most audio-feature-similar songs to the profile (the control)."""
    centroid = features.loc[features.track_id.isin(profile_ids), feature_cols].mean()
    others   = features[~features.track_id.isin(exclude_ids)].copy()
    sims = cosine_similarity(centroid.values.reshape(1, -1), others[feature_cols].values)[0]
    others["sim"] = sims
    return others.sort_values("sim", ascending=False).track_id.head(k).tolist()


def similar_sounding_playlist(profile_ids, track_playlist, exclude, k=K_SET):
    """Set B — songs sharing human-curated playlists with the profile (the real signal)."""
    profile_playlists = (track_playlist[track_playlist.track_id.isin(profile_ids)]
                         .playlist_id.unique())
    siblings = track_playlist[track_playlist.playlist_id.isin(profile_playlists)]
    counts = (siblings[~siblings.track_id.isin(exclude | set(profile_ids))]
              .track_id.value_counts())
    ordered = counts.reset_index()
    ordered.columns = ["track_id", "n"]
    # deterministic tie-break (count desc, then id) so the frozen set is stable
    ordered = ordered.sort_values(["n", "track_id"], ascending=[False, True])
    return ordered["track_id"].head(k).tolist()

def persona_specs(n_personas, rng):
    """
    Produces n_personas specs, each liking one genre and disliking another.
    Every spec must have an 'id'; the other fields are read by the two
    selection functions below, so you can add your own fields freely.
    """
    pairs = [(a, b) for a in GENRES for b in GENRES if a != b] # 30 ordered pairs
    rng.shuffle(pairs)   # seeded → reproducible
    specs = []
    for i in range(n_personas):
        liked, disliked = pairs[i % len(pairs)] # wrap if n_personas > 30
        specs.append({
            "id":             f"persona_{i + 1:02d}",
            "liked_genre":    liked,
            "disliked_genre": disliked,
        })
    return specs


def _sample_ids(pool, n, rng):
    """Sample n track IDs from a pool without replacement, using the seeded rng."""
    ids = pool["track_id"].tolist()
    n = min(n, len(ids)) # don't ask for more than exist
    idx = rng.choice(len(ids), size=n, replace=False)
    return [ids[i] for i in idx]


def select_coherent_taste(tracks, spec, rng):
    """The persona's 'liked' songs — sampled from its liked genre."""
    pool = tracks[tracks["genre"] == spec["liked_genre"]]
    return _sample_ids(pool, N_PROFILE, rng)


def select_disliked_taste(tracks, spec, rng):
    """The persona's 'disliked' songs — sampled from its disliked genre."""
    pool = tracks[tracks["genre"] == spec["disliked_genre"]]
    return _sample_ids(pool, N_NEGATIVE, rng)


def build_persona(spec, tracks, features, feature_cols, track_playlist, rng):
    """Builds one complete persona dict (both answer keys included)."""
    profile_ids  = select_coherent_taste(tracks, spec, rng)
    negative_ids = select_disliked_taste(tracks, spec, rng)
    used = set(profile_ids) | set(negative_ids)

    set_a_ids = feature_neighbors(profile_ids, features, feature_cols, used, k=K_SET)
    set_b_ids = similar_sounding_playlist(profile_ids, track_playlist, used, k=K_SET)

    # fail loudly at build time rather than scoring a silent zero later
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
    """Maps each track_id to a single genre/subgenre (first playlist wins; 90% are single)."""
    merged = track_playlist.merge(
        playlists[["playlist_id", "playlist_genre", "playlist_subgenre"]],
        on="playlist_id",
    )
    merged = merged.sort_values(["track_id", "playlist_id"]) # stable, reproducible pick
    first = merged.drop_duplicates("track_id")
    return first.rename(columns={
        "playlist_genre":    "genre",
        "playlist_subgenre": "subgenre",
    })[["track_id", "genre", "subgenre"]]


def build_and_freeze(n_personas, seed=42):
    """Generates n_personas persona files into data/eval/personas/."""
    ensure_eval_dirs()
    rng = np.random.default_rng(seed)

    tracks         = pd.read_parquet(TRACKS_PATH)
    features       = pd.read_parquet(RECOMMENDATION_FEATURES_PATH)
    track_playlist = pd.read_parquet(TRACK_PLAYLIST_PATH)
    playlists      = pd.read_parquet(PLAYLISTS_PATH)

    # attach a genre/subgenre column to tracks so the taste functions can filter
    genre_lookup = _build_genre_lookup(track_playlist, playlists)
    tracks = tracks.merge(genre_lookup, on="track_id", how="left")

    feature_cols = [c for c in features.columns if c.endswith("_scaled")]

    for spec in persona_specs(n_personas, rng):
        persona = build_persona(spec, tracks, features, feature_cols, track_playlist, rng)
        out_path = PERSONAS_DIR / f"{persona['id']}.json"
        with open(out_path, "w") as f:
            json.dump(persona, f, indent=2)
        print(f"  saved {out_path.name}  ({spec['liked_genre']} vs {spec['disliked_genre']})")