import pandas as pd

def build_tracks_table(df: pd.DataFrame) -> pd.DataFrame:
    tracks = df[
        [
            "track_id",
            "track_name",
            "track_artist",
            "track_popularity",
            "track_album_id",
            "track_album_name",
            "track_album_release_date",
            "release_year",
            "release_month",
            "release_day",
            "duration_ms",
        ]
    ].copy()

    tracks = tracks.drop_duplicates(subset=["track_id"])

    return tracks

def build_track_features_table(df: pd.DataFrame) -> pd.DataFrame:
    features = df[
        [
            "track_id",
            "danceability",
            "energy",
            "key",
            "loudness",
            "mode",
            "speechiness",
            "acousticness",
            "instrumentalness",
            "liveness",
            "valence",
            "tempo",
        ]
    ].copy()

    features = features.drop_duplicates(subset=["track_id"])

    return features

def build_playlists_table(df: pd.DataFrame) -> pd.DataFrame:
    playlists = df[
        [
            "playlist_id",
            "playlist_name",
            "playlist_genre",
            "playlist_subgenre",
        ]
    ].copy()

    playlists = playlists.drop_duplicates(subset=["playlist_id"])

    return playlists

def build_track_playlist_table(df: pd.DataFrame) -> pd.DataFrame:
    track_playlist = df[
        [
            "track_id",
            "playlist_id",
        ]
    ].copy()

    track_playlist = track_playlist.drop_duplicates()

    return track_playlist

def split_into_tables(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        "tracks": build_tracks_table(df),
        "track_features": build_track_features_table(df),
        "playlists": build_playlists_table(df),
        "track_playlist": build_track_playlist_table(df),
    }