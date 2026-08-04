import pandas as pd
from pathlib import Path

def load_tracks_csv(path: str | Path) -> pd.DataFrame:  #reusable method that either accepts a string or a path called "path" and returns a DataFrame
    path = Path(path) #converting str path to a path object, if param's path is already a object, nothing happens. Path(str) comes from pathlib module
    if not path.exists():
        raise FileNotFoundError(f"Could not find file: {path}")
    return pd.read_csv(path) #read CSV file at path and return as a DataFrame