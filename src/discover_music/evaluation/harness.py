import pandas as pd, contextlib
from discover_music.recommender.model import get_recommended_songs, load_tracks
from discover_music.recommender.ratings import load_ratings
from discover_music.utils.paths import USER_RATINGS_PATH, PERSONAS_DIR, ensure_local_data_dir
from discover_music.evaluation import metrics, baselines

KS = [10, 20, 50]

@contextlib.contextmanager
def temporarily_rated(profile_ids, negative_ids):
    ensure_local_data_dir()
    backup = load_ratings() if USER_RATINGS_PATH.exists() else None   # save real ratings
    try:
        #creates fake ratings
        rows = ([{"track_id": t, "rating": 5} for t in profile_ids] +
                [{"track_id": t, "rating": 1} for t in negative_ids])
        pd.DataFrame(rows).to_csv(USER_RATINGS_PATH, index=False)
        yield   #what ever runs in with (run_eval()) runs, when returned, yield continues
    finally:
        #restore old ratings
        if backup is not None: backup.to_csv(USER_RATINGS_PATH, index=False)
        else: USER_RATINGS_PATH.unlink(missing_ok=True)

def score(ranked_ids, relevant_ids):    
    return {f"ndcg@{k}":   metrics.ndcg_at_k(ranked_ids, relevant_ids, k)   for k in KS} | \
           {f"recall@{k}": metrics.recall_at_k(ranked_ids, relevant_ids, k) for k in KS}
    #scores

def run_eval(seed=22):
    all_ids = set(load_tracks().track_id)
    rows = []
    for persona in load_personas(PERSONAS_DIR):
        rated = set(persona["profile_ids"]) | set(persona["negative_ids"])
        candidates = list(all_ids - rated)                            # same pool for everyone

        with temporarily_rated(persona["profile_ids"], persona["negative_ids"]):
            ranked = get_recommended_songs(n=len(candidates))         # FULL ranking, real model
            model_ids = ranked["track_id"].tolist()

        rng = make_rng(seed)
        rows.append({
            "persona": persona["id"],
            "model_A":      score(model_ids, persona["set_a_ids"]),
            "model_B":      score(model_ids, persona["set_b_ids"]),
            "random_B":     score(baselines.random_ranking(candidates, rng),  persona["set_b_ids"]),
            "popularity_B": score(baselines.popularity_ranking(candidates),   persona["set_b_ids"]),
        })
    return aggregate(rows)        # mean ± std across personas; also compute A−B per metric