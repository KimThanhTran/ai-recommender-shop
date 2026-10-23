"""Loading the MovieLens catalogue and splitting interactions for evaluation."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

RATING_COLUMNS = ["user_id", "item_id", "rating", "timestamp"]


@dataclass(frozen=True)
class Dataset:
    """Interactions plus the item catalogue.

    ``ratings``: columns user_id, item_id, rating, timestamp.
    ``items``: indexed by item_id with columns title, genres ("A|B|C") and tags (free text).
    """

    ratings: pd.DataFrame
    items: pd.DataFrame


def load_movielens(path: str | Path) -> Dataset:
    """Read ratings.csv, movies.csv and (optionally) tags.csv from ``path``."""
    path = Path(path)
    ratings = pd.read_csv(path / "ratings.csv").rename(columns={"userId": "user_id", "movieId": "item_id"})
    movies = pd.read_csv(path / "movies.csv").rename(columns={"movieId": "item_id"})

    tags_file = path / "tags.csv"
    if tags_file.exists():
        tags = pd.read_csv(tags_file).rename(columns={"movieId": "item_id"})
        tag_text = tags.groupby("item_id")["tag"].apply(lambda values: " ".join(map(str, values)))
        movies["tags"] = movies["item_id"].map(tag_text).fillna("")
    else:
        movies["tags"] = ""

    items = movies.set_index("item_id")[["title", "genres", "tags"]]
    ratings = ratings[ratings["item_id"].isin(items.index)][RATING_COLUMNS].reset_index(drop=True)
    return Dataset(ratings=ratings, items=items)


def temporal_split(
    ratings: pd.DataFrame, test_fraction: float = 0.2, min_ratings: int = 5
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-user chronological split: the most recent ``test_fraction`` of each user's ratings is held out.

    Users with fewer than ``min_ratings`` ratings stay entirely in the training set, so they are
    never evaluated. Splitting by time avoids training on interactions that happened after the test ones.
    """
    ordered = ratings.sort_values(["user_id", "timestamp", "item_id"]).copy()
    ordered["_rank"] = ordered.groupby("user_id").cumcount()
    ordered["_size"] = ordered.groupby("user_id")["item_id"].transform("size")

    n_test = np.ceil(ordered["_size"] * test_fraction).astype(int)
    is_test = (ordered["_size"] >= min_ratings) & (ordered["_rank"] >= ordered["_size"] - n_test)

    train = ordered[~is_test].drop(columns=["_rank", "_size"]).reset_index(drop=True)
    test = ordered[is_test].drop(columns=["_rank", "_size"]).reset_index(drop=True)
    return train, test


def item_text(items: pd.DataFrame) -> pd.Series:
    """Bag-of-words document per item built from genres, user tags and the release decade."""

    def build(row: pd.Series) -> str:
        genres = [g for g in str(row["genres"]).split("|") if g and g != "(no genres listed)"]
        tokens = [f"genre_{g.replace(' ', '_').replace('-', '_').lower()}" for g in genres]
        match = re.search(r"\((\d{4})\)\s*$", str(row["title"]))
        if match:
            tokens.append(f"decade_{match.group(1)[:3]}0s")
        return " ".join(tokens) + " " + str(row["tags"]).lower()

    return items.apply(build, axis=1)
