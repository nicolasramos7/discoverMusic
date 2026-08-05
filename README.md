# discoverMusic

Model Evaluation Visual: https://docs.google.com/spreadsheets/d/1_s7FJIAhxYbsJlre11CgbsVaAyC-UXva9RU90WzY9MU/edit?gid=0#gid=0

A content-based music recommender built as a personal project to **learn Python with a focus on data analysis**. It takes a ~30k-track Spotify dataset, cleans it into relational tables and a scaled feature matrix, learns your taste from ratings, and recommends new tracks by audio-feature similarity.

The code is written for learning: modules are small and single-purpose, and most lines are commented to explain *why* each pandas / scikit-learn call is there.

---

## What it does today

**1. Data pipeline**: turns one messy CSV into clean, analysis-ready tables.

- **Load & validate** the raw `spotify_tracks_30000.csv`.
- **Clean** (`data/clean.py`): normalize text, coerce numeric columns, parse release dates into year/month/day, drop rows with missing essentials or invalid tempo/duration, and remove duplicates.
- **Split into relational tables** (`data/split_tables.py`): `tracks`, `track_features`, `playlists`, and a `track_playlist` join table.
- **Build a feature matrix** (`data/features.py`): select the audio features used for recommendation (danceability, energy, loudness, valence, tempo, etc.) and standardize them with scikit-learn's `StandardScaler`.

Everything is written to `data/processed/` in both CSV (readable) and Parquet (fast) formats.

**2. Content-based recommender**:  profiles your taste and ranks tracks.

- **Ratings** (`recommender/ratings.py`): rate songs 1–5; ratings persist to `data/local/user_ratings.csv`.
- **User profile** (`recommender/profile.py`): average the scaled features of the songs you rated 4+ into a single "taste vector."
- **Similarity ranking** (`recommender/similarity.py`): rank every unrated track by **cosine similarity** to your taste vector.
- **Cold start:** brand-new users get random songs to seed a profile, then switch to recommendations.

**3. Terminal app**: try it end to end.

A small CLI (`app/`) walks you through an initial rating batch, then enters a recommendation loop that keeps serving the next best-matching track for you to rate.

---

## Project layout

```
src/discover_music/
├── data/          # load, validate, clean, split, feature-engineer
├── recommender/   # ratings, user profile, cosine-similarity ranking, model glue
├── evaluation/    # metrics, personas, baselines, harness (in progress)
├── pipelines/     # prepare_data: the full clean → tables → features run
├── app/           # terminal CLI
└── utils/         # shared paths

scripts/           # entry points (run_data_prep, run_terminal_app, ...)
data/
├── raw/           # source CSV
├── interim/       # cleaned-but-unsplit data
├── processed/     # relational tables + scaled feature matrix
└── local/         # your personal ratings (gitignored)
```

---

## Getting started

```bash
# from the project root, with the virtualenv active
pip install -e .

# 1. build the processed tables + feature matrix from the raw CSV
python scripts/run_data_prep.py

# 2. rate songs and get recommendations
python scripts/run_terminal_app.py
```

Requires Python 3.10+. Dependencies (pandas, numpy, scikit-learn, pyarrow, python-dotenv) are declared in `pyproject.toml`.

---

## Currently being developed: a universal test to track performance
This project is being made to learn Python for data science, as I add new features, I want to be able to track how results become better or not. Rather than eyeballing results, I'm building a repeatable evaluation suite that produces the same benchmark before and after every new feature, so improvement (or regression) is a number I can point to.

The approach being built out in `src/discover_music/evaluation/`:

- **Synthetic personas** (`personas.py`): generate a fixed set of simulated listeners, each with a coherent "taste profile" of liked tracks, plus held-out relevant sets to grade against. Personas are frozen (built once, committed) so every evaluation run is directly comparable.
- **Ranking metrics** (`metrics.py`): **Recall@k**, **Hit-rate@k**, and **NDCG@k**, measuring how many relevant tracks land in the top *k* and how well-ranked they are.
- **Baselines** (`baselines.py`): random and popularity-based rankings, so the model's score always has something to beat.
- **Harness** (`harness.py`): temporarily loads each persona's ratings, runs the *real* recommender to produce a full ranking, and scores it against the persona's relevant sets, reporting mean ± std across all personas.

Once this is stable, the workflow becomes: **run the test → add a feature → run the test again → keep what improves the numbers.**

### Feature ideas on deck

- Fold **genre** into the feature matrix (as a scaled numeric signal) instead of using audio features alone.
- Make ratings **weighted**, so a 5 pulls the taste profile harder than a 4, and a 1 pushes away harder than a 2 (currently all 4+ are treated equally).
- Use **disliked tracks** (low ratings) as a negative signal in the profile.

---

*Personal learning project: Python, pandas, scikit-learn.*
