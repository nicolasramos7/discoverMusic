import json
import numpy as np
import pandas as pd

from discover_music.recommender.model   import get_recommended_songs, load_tracks
from discover_music.recommender.ratings import load_ratings
from discover_music.utils.paths         import (
    USER_RATINGS_PATH,
    PERSONAS_DIR,
    ensure_local_data_dir,
)
from discover_music.evaluation import metrics, baselines

# cutoff values every metric is computed at
KS = [10, 20, 50]

def rank_with_persona(profile_ids, negative_ids, candidates):
    #impersonating a user, safely
    ensure_local_data_dir()
    backup = load_ratings() if USER_RATINGS_PATH.exists() else None

    try:
        # build the persona's ratings: liked songs = 5, disliked songs = 1
        rows = ([{"track_id": t, "rating": 5} for t in profile_ids] +
                [{"track_id": t, "rating": 1} for t in negative_ids])

        #overwrite the ratings file, the model now sees these as the user's ratings
        pd.DataFrame(rows).to_csv(USER_RATINGS_PATH, index=False)

        #run the real model; n = len(candidates) returns the FULL ranking
        ranked = get_recommended_songs(n=len(candidates))

        #return the ordered track IDs as a plain list
        return ranked["track_id"].tolist()

    finally:
        #this runs, even if the model call above crashes
        if backup is not None:
            # put the original ratings back exactly as they were
            backup.to_csv(USER_RATINGS_PATH, index=False)
        else:
            # there was no file before — delete the one we created
            USER_RATINGS_PATH.unlink(missing_ok=True)


def score(ranked_ids, relevant_ids):
    #computes NDCG and Recall at every k for one (ranking, answer key) pair.
    return {f"ndcg@{k}":   metrics.ndcg_at_k(ranked_ids, relevant_ids, k)   for k in KS} | \
           {f"recall@{k}": metrics.recall_at_k(ranked_ids, relevant_ids, k) for k in KS}


def load_personas():
    #reads every persona JSON file from data/eval/personas/ into a list of dicts.
    if not PERSONAS_DIR.exists():
        raise FileNotFoundError("No personas found. Run scripts/build_personas.py first.")

    personas = []
    for path in sorted(PERSONAS_DIR.glob("*.json")):
        with open(path) as f:
            personas.append(json.load(f))
    return personas


def aggregate(rows):
    #averages every metric across all personas and computes the A-B gap.
    #returns mean ± std per metric for the model, both baselines, and the gap.
    metric_keys = list(rows[0]["model_A"].keys())
    result = {}

    # mean/std for the model and both baselines
    for source in ["model_A", "model_B", "random_B", "popularity_B"]:
        result[source] = {}
        for key in metric_keys:
            values = [row[source][key] for row in rows]
            result[source][key] = {
                "mean": float(np.mean(values)),
                "std":  float(np.std(values)),
            }

    # A-B gap — the math-vs-perception distance you track over time
    result["gap_A_minus_B"] = {}
    for key in metric_keys:
        gaps = [row["model_A"][key] - row["model_B"][key] for row in rows]
        result["gap_A_minus_B"][key] = {
            "mean": float(np.mean(gaps)),
            "std":  float(np.std(gaps)),
        }

    return result


def run_eval(seed=42):
    rng = np.random.default_rng(seed)

    # the universe of all track IDs, loaded once
    all_ids = set(load_tracks()["track_id"])

    rows = []

    for persona in load_personas():
        print(f"  evaluating {persona['id']}...")

        # everything this persona has rated — excluded from the candidate pool
        rated = set(persona["profile_ids"]) | set(persona["negative_ids"])

        # the pool the model may recommend from
        candidates = list(all_ids - rated)

        # write persona ratings → run model → restore ratings, all in one call
        model_ids = rank_with_persona(
            persona["profile_ids"],
            persona["negative_ids"],
            candidates,
        )

        rows.append({
            "persona": persona["id"],

            # model vs Set A — the mathematical control (expect high)
            "model_A":      score(model_ids, persona["set_a_ids"]),

            # model vs Set B — the perceptual test (the real signal)
            "model_B":      score(model_ids, persona["set_b_ids"]),

            # baselines vs Set B — reference points that make model_B interpretable
            "random_B":     score(baselines.random_ranking(candidates, rng),  persona["set_b_ids"]),
            "popularity_B": score(baselines.popularity_ranking(candidates),   persona["set_b_ids"]),
        })

    return aggregate(rows)