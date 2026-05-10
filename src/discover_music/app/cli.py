from discover_music.app.display import (
    display_goodbye,
    display_initial_batch_complete,
    display_invalid_input,
    display_new_user_message,
    display_no_songs_left,
    display_recommendation_mode_start,
    display_returning_user_message,
    display_saved,
    display_song,
    display_welcome,
)
from discover_music.recommender.model import get_initial_batch, get_next_song
from discover_music.recommender.ratings import is_new_user, save_rating

TRACK_ID_COLUMN = "track_id"

def ask_for_rating() -> int | None:
    while True: #loops only in the case of invalid input
        user_input = input("Rate from 1 to 5, or q to quit: ").strip().lower()

        if user_input == "q":
            return None

        if user_input in {"1", "2", "3", "4", "5"}:
            return int(user_input)

        display_invalid_input()

def rate_song(row, current: int | None = None, total: int | None = None) -> bool:
    display_song(row, current=current, total=total) #display song using display.py helper

    rating = ask_for_rating()

    if rating is None:
        return False    #return false if the user quit

    track_id = str(row[TRACK_ID_COLUMN])    #get track id to save in user_ratings
    save_rating(track_id=track_id, rating=rating)

    display_saved()

    return True #return true signifying successful rate and save

def run_initial_batch() -> bool:
    new_user = is_new_user()    #first check that user is new using ratings.py helper

    if new_user:
        display_new_user_message()  #if new, display according message
    else:
        display_returning_user_message()    #if not new, display according message

    batch = get_initial_batch(n=10) #use model.py helper to get an initial batch

    if batch.empty: #here the empty Set return from model.py is handled
        display_no_songs_left()
        return False

    total = len(batch)  #total amount of songs is accessed to display current and total

    for index, (_, row) in enumerate(batch.iterrows(), start=1):    #iterate through all rows giving each a index
        completed = rate_song(row, current=index, total=total)  #call rate song using index and total

        if not completed:   #if completed is falsy (in this case, rate_song returns false)
            return False

    display_initial_batch_complete()

    return True

def run_recommendation_loop() -> None:
    display_recommendation_mode_start()

    while True:
        next_song = get_next_song() #use model helper method

        if next_song.empty: #model returns empty if there are no songs, this handles that
            display_no_songs_left()
            return

        row = next_song.iloc[0] #takes first row from dataframe

        completed = rate_song(row)  #song is rated

        if not completed:   #if user quit or smth, just return and stop loop
            return

def main() -> None:
    display_welcome()

    completed_initial_batch = run_initial_batch()

    if completed_initial_batch:
        run_recommendation_loop()

    display_goodbye()


if __name__ == "__main__":
    main()
