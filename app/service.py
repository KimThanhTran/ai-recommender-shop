"""Business logic behind the API, independent of FastAPI so it is easy to test."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from recsys import (
    ContentBasedRecommender,
    HybridRecommender,
    ItemKNNRecommender,
    PopularityRecommender,
    Recommender,
)

PERSONAL_MODELS = ("hybrid", "item_knn", "content")
SIMILAR_MODELS = ("hybrid", "item_knn", "content")


class ItemNotFound(KeyError):
    pass


@dataclass
class RecommendationResult:
    model: str
    personalised: bool
    items: list[dict]


class RecommenderService:
    """Holds the fitted models and answers recommendation queries."""

    def __init__(self, ratings: pd.DataFrame, items: pd.DataFrame):
        self.ratings = ratings
        self.catalogue = items
        self.popularity = PopularityRecommender().fit(ratings, items)
        self.hybrid = HybridRecommender(cf_weight=0.7).fit(ratings, items)
        self.models: dict[str, Recommender] = {
            "hybrid": self.hybrid,
            "item_knn": self.hybrid.cf,
            "content": self.hybrid.content,
        }

    # -- catalogue ---------------------------------------------------------------------------
    def describe(self, item_id: int, score: float | None = None) -> dict:
        row = self.catalogue.loc[item_id]
        genres = [g for g in str(row["genres"]).split("|") if g and g != "(no genres listed)"]
        info = {"item_id": int(item_id), "title": row["title"], "genres": genres}
        if score is not None:
            info["score"] = round(float(score), 4)
        return info

    def get_item(self, item_id: int) -> dict:
        if item_id not in self.catalogue.index:
            raise ItemNotFound(item_id)
        return self.describe(item_id)

    # -- recommendations ---------------------------------------------------------------------
    def popular(self, k: int) -> list[dict]:
        return [self.describe(i, s) for i, s in self.popularity.top_items(k)]

    def recommend(self, user_id: int, k: int, model: str) -> RecommendationResult:
        """Personalised top-k for a known user; popular items for an unknown one (cold start).

        When a model returns fewer than ``k`` items (a user with very little history),
        the rest is filled with popular items the user has not rated.
        """
        if not self.popularity.has_user(user_id):
            return RecommendationResult("popularity", False, self.popular(k))

        ranked = self.models[model].recommend(user_id, k=k)
        if len(ranked) < k:
            exclude = [*self.popularity.seen_items(user_id), *(self.popularity.index_of(i) for i, _ in ranked)]
            fillers = self.popularity.top_items(k - len(ranked), exclude=np.array(exclude, dtype=int))
            ranked.extend((item, 0.0) for item, _ in fillers)
        return RecommendationResult(model, True, [self.describe(i, s) for i, s in ranked])

    def similar(self, item_id: int, k: int, model: str) -> list[dict]:
        if item_id not in self.catalogue.index:
            raise ItemNotFound(item_id)
        return [self.describe(i, s) for i, s in self.models[model].similar_items(item_id, k)]

    def history(self, user_id: int, limit: int) -> list[dict] | None:
        """The user's highest-rated items, or None if the user is unknown."""
        rows = self.ratings[self.ratings["user_id"] == user_id]
        if rows.empty:
            return None
        top = rows.sort_values(["rating", "timestamp"], ascending=[False, False]).head(limit)
        return [{**self.describe(r.item_id), "rating": float(r.rating)} for r in top.itertuples()]

    @property
    def stats(self) -> dict:
        return {"users": int(len(self.popularity.user_ids)), "items": int(len(self.catalogue))}
