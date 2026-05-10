import pandas as pd
from discover_music.utils.paths import ensure_local_data_dir, USER_RATINGS_PATH

RATINGS_COLUMNS = ["track_id", "rating"]

def ensure_ratings_file() -> None:  #use helper method in paths and a Path.exists() check to ensure user_ratings.csv exists and is placed correctly
    ensure_local_data_dir()

    if not USER_RATINGS_PATH.exists():
        empty_ratings = pd.DataFrame(columns=RATINGS_COLUMNS)
        empty_ratings.to_csv(USER_RATINGS_PATH, index=False)

def load_ratings() -> pd.DataFrame:

    ensure_ratings_file()

    ratings = pd.read_csv(USER_RATINGS_PATH)

    for column in RATINGS_COLUMNS:
        if column not in ratings.columns:
            ratings[column] = pd.Series(dtype="object") #if columns are missing, add with Python objects as the type

    ratings = ratings[RATINGS_COLUMNS]  #remove columns that arent track_id or rating

    if not ratings.empty:
        ratings["track_id"] = ratings["track_id"].astype(str)   #convert id to string
        ratings["rating"] = ratings["rating"].astype(int) #convert rating to int

    return ratings

def is_new_user() -> bool:
    ratings = load_ratings()
    return ratings.empty

def get_rated_track_ids() -> set[str]:  #returns set of track_ids of rated songs
    ratings = load_ratings()

    if ratings.empty:
        return set()

    return set(ratings["track_id"].astype(str))

def save_rating(track_id: str, rating: int) -> None:    #add new entry of rating into csv (or update is already exists)
    if rating < 1 or rating > 5:    #check rating is correct
        raise ValueError("Rating must be between 1 and 5.")

    ratings = load_ratings()
    track_id = str(track_id)    #ensure track_id is a string

    if ratings.empty:   #if empty just create df with first entry 
        ratings = pd.DataFrame(
            [{"track_id": track_id, "rating": rating}],
            columns=RATINGS_COLUMNS,
        )
    elif track_id in set(ratings["track_id"].astype(str)):  #if already rated
        ratings.loc[ratings["track_id"].astype(str) == track_id, "rating"] = rating #updates all the rows with matching track_id to the new rating
    else:
        new_row = pd.DataFrame( # new ? create new df matching format of current df
            [{"track_id": track_id, "rating": rating}],
            columns=RATINGS_COLUMNS,
        )
        ratings = pd.concat([ratings, new_row], ignore_index=True)  #concatinate to current main df

    ratings.to_csv(USER_RATINGS_PATH, index=False)  #convert back to csv