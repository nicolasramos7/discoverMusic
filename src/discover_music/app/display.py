import pandas as pd

TITLE_COLUMNS = ["track_name", "title", "name", "song_name"]
ARTIST_COLUMNS = ["track_artist","artist_name", "artist", "artists"]

def _get_first_available_value(row: pd.Series, possible_columns: list[str]) -> str:
    #helper that just returns specific columns of a given row based on specified column names
    #iterates through until it finds best match because some can be NaN
    for column in possible_columns:
        if column in row.index:
            value = row[column]
            if pd.notna(value) and str(value).strip():
                return str(value)

    return "Unknown"

def get_song_title(row: pd.Series) -> str:
    return _get_first_available_value(row, TITLE_COLUMNS)   #calls helper method above specifying title columns


def get_song_artist(row: pd.Series) -> str:
    return _get_first_available_value(row, ARTIST_COLUMNS)  #same as above but for artist columns

def display_welcome() -> None:
    #display welcome helper
    print()
    print("========================================")
    print("discoverMusic")
    print("========================================")
    print()

def display_new_user_message() -> None:
    #new user helper
    print("You are new here, so we will start with 10 random songs.")
    print("Rate each song from 1 to 5.")
    print()

def display_returning_user_message() -> None:
    #existent user helper
    print("Welcome back.")
    print("Based on your previous ratings, here are 10 songs to rate.")
    print()

def display_song(row: pd.Series, current: int | None = None, total: int | None = None) -> None:
    #displays song and counter (optional, so default is set to None if not specified)
    title = get_song_title(row)
    artist = get_song_artist(row)

    print("----------------------------------------")

    if current is not None and total is not None:   #if counter is specified, print
        print(f"Song {current} of {total}")
        print()

    print(f"Title:  {title}")
    print(f"Artist: {artist}")
    print("----------------------------------------")

#more display helpers:
def display_saved() -> None:
    print("Saved.")
    print()

def display_initial_batch_complete() -> None:
    print()
    print("Initial ratings complete.")
    print("Building your recommendation profile...")
    print()

def display_recommendation_mode_start() -> None:
    print("Now I will recommend songs one at a time.")
    print("Rate each song from 1 to 5, or enter q to quit.")
    print()

def display_no_songs_left() -> None:
    print()
    print("There are no more songs available to recommend.")
    print()

def display_goodbye() -> None:
    print()
    print("Your ratings have been saved.")
    print("See you next time.")
    print()

def display_invalid_input() -> None:
    print("Please enter a rating from 1 to 5, or q to quit.")
    print()