import pandas as pd
from discover_music.utils.paths import TRACKS_PATH

#two baselines that metrics measures to then compare against

def random_ranking(candidate_ids, rng):
    ids = list(candidate_ids)
    rng.shuffle(ids)
    return ids

def popularity_ranking(candidate_ids):  #"recommend famous songs" is a genuinely strong naive strategy.
    tracks = pd.read_parquet(TRACKS_PATH)[["track_id", "track_popularity"]]
    pool = tracks[tracks.track_id.isin(candidate_ids)]
    return pool.sort_values("track_popularity", ascending=False).track_id.tolist()