"""Recommendation models: popularity baseline, item-based CF, content-based and a hybrid."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from .data import item_text


class Recommender(ABC):
    """Common plumbing: id <-> index maps, the user-item matrix and top-k selection."""

    name = "recommender"

    def fit(self, ratings: pd.DataFrame, items: pd.DataFrame) -> "Recommender":
        self.item_ids = items.index.to_numpy()
        self.items = items
        self._item_index = {item_id: i for i, item_id in enumerate(self.item_ids)}

        ratings = ratings[ratings["item_id"].isin(self._item_index)]
        self.user_ids = np.sort(ratings["user_id"].unique())
        self._user_index = {user_id: i for i, user_id in enumerate(self.user_ids)}

        rows = ratings["user_id"].map(self._user_index).to_numpy()
        cols = ratings["item_id"].map(self._item_index).to_numpy()
        self.matrix = sparse.csr_matrix(
            (ratings["rating"].to_numpy(dtype=np.float64), (rows, cols)),
            shape=(len(self.user_ids), len(self.item_ids)),
        )
        self._fit()
        return self

    @abstractmethod
    def _fit(self) -> None:
        """Learn model state from ``self.matrix`` (users x items) and ``self.items``."""

    @abstractmethod
    def score_user(self, user_index: int) -> np.ndarray:
        """Relevance score for every catalogue item (higher is better)."""

    def has_user(self, user_id: int) -> bool:
        return user_id in self._user_index

    def has_item(self, item_id: int) -> bool:
        return item_id in self._item_index

    def index_of(self, item_id: int) -> int:
        return self._item_index[item_id]

    def seen_items(self, user_id: int) -> np.ndarray:
        user_index = self._user_index[user_id]
        return self.matrix[user_index].indices

    def recommend(self, user_id: int, k: int = 10, exclude_seen: bool = True) -> list[tuple[int, float]]:
        """Top-k (item_id, score) for a known user. Items with a non-positive score are never returned."""
        if user_id not in self._user_index:
            raise KeyError(f"unknown user {user_id}")
        scores = np.array(self.score_user(self._user_index[user_id]), dtype=np.float64)
        if exclude_seen:
            scores[self.seen_items(user_id)] = -np.inf
        return self._top_k(scores, k)

    def _top_k(self, scores: np.ndarray, k: int) -> list[tuple[int, float]]:
        k = min(k, len(scores))
        if k <= 0:
            return []
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(int(self.item_ids[i]), float(scores[i])) for i in top if scores[i] > 0]

    def similar_items(self, item_id: int, k: int = 10) -> list[tuple[int, float]]:
        raise NotImplementedError(f"{self.name} does not support similar items")


class PopularityRecommender(Recommender):
    """Most-rated items first. Not personalised, but a strong baseline that any model must beat."""

    name = "popularity"

    def _fit(self) -> None:
        counts = np.asarray((self.matrix > 0).sum(axis=0)).ravel().astype(np.float64)
        self.popularity = counts

    def score_user(self, user_index: int) -> np.ndarray:
        return self.popularity

    def top_items(self, k: int = 10, exclude: np.ndarray | None = None) -> list[tuple[int, float]]:
        scores = self.popularity.copy()
        if exclude is not None and len(exclude):
            scores[exclude] = -np.inf
        return self._top_k(scores, k)


class ItemKNNRecommender(Recommender):
    """Item-based collaborative filtering.

    Item-item cosine similarity over the rating matrix, shrunk towards zero when two items share
    few raters (so an item with one rating cannot look identical to another), keeping only the
    ``neighbors`` strongest neighbours per item. A user's score for an item is the similarity-weighted
    sum of that user's own ratings of its neighbours.
    """

    name = "item_knn"

    def __init__(self, neighbors: int = 50, shrinkage: float = 10.0, chunk: int = 1000):
        self.neighbors = neighbors
        self.shrinkage = shrinkage
        self.chunk = chunk

    def _fit(self) -> None:
        ratings = self.matrix.tocsc()
        norms = np.sqrt(np.asarray(ratings.multiply(ratings).sum(axis=0)).ravel())
        norms[norms == 0] = 1.0
        normalised = (ratings @ sparse.diags(1.0 / norms)).tocsr()
        binary = (ratings > 0).astype(np.float64).tocsr()

        n_items = ratings.shape[1]
        blocks = []
        for start in range(0, n_items, self.chunk):
            stop = min(start + self.chunk, n_items)
            sim = (normalised.T[start:stop] @ normalised).toarray()
            shared = (binary.T[start:stop] @ binary).toarray()
            denominator = shared + self.shrinkage
            sim *= np.divide(shared, denominator, out=np.zeros_like(shared), where=denominator > 0)
            sim[np.arange(stop - start), np.arange(start, stop)] = 0.0  # no self-similarity
            blocks.append(self._keep_top_neighbors(sim))
        self.similarity = sparse.vstack(blocks).tocsr()

    def _keep_top_neighbors(self, sim: np.ndarray) -> sparse.csr_matrix:
        k = min(self.neighbors, sim.shape[1])
        top = np.argpartition(-sim, k - 1, axis=1)[:, :k]
        rows = np.repeat(np.arange(sim.shape[0]), k)
        cols = top.ravel()
        values = sim[rows, cols]
        keep = values > 0
        return sparse.csr_matrix((values[keep], (rows[keep], cols[keep])), shape=sim.shape)

    def score_user(self, user_index: int) -> np.ndarray:
        return np.asarray(self.similarity @ self.matrix[user_index].toarray().ravel()).ravel()

    def similar_items(self, item_id: int, k: int = 10) -> list[tuple[int, float]]:
        row = self.similarity[self._item_index[item_id]].toarray().ravel()
        return self._top_k(row, k)


class ContentBasedRecommender(Recommender):
    """Content-based filtering on TF-IDF vectors of genres, tags and release decade.

    A user's profile is the average of the vectors of items they rated, weighted by how much they
    liked each one (rating minus the user's mean), and items are ranked by cosine similarity to it.
    """

    name = "content"

    def _fit(self) -> None:
        vectorizer = TfidfVectorizer(token_pattern=r"[^\s]+", sublinear_tf=True, min_df=1)
        self.item_vectors = vectorizer.fit_transform(item_text(self.items)).tocsr()

    def _profile(self, user_index: int) -> np.ndarray:
        row = self.matrix[user_index]
        weights = row.data - row.data.mean()
        if not np.any(weights):
            weights = row.data
        vector = sparse.csr_matrix(weights) @ self.item_vectors[row.indices]
        return np.asarray(vector.todense()).ravel()

    def score_user(self, user_index: int) -> np.ndarray:
        profile = self._profile(user_index)
        norm = np.linalg.norm(profile)
        if norm == 0:
            return np.zeros(len(self.item_ids))
        return np.asarray(self.item_vectors @ (profile / norm)).ravel()

    def similar_items(self, item_id: int, k: int = 10) -> list[tuple[int, float]]:
        index = self._item_index[item_id]
        scores = np.asarray((self.item_vectors @ self.item_vectors[index].T).todense()).ravel()
        scores[index] = -np.inf
        return self._top_k(scores, k)


class HybridRecommender(Recommender):
    """Weighted blend of item-based CF and content-based scores.

    Each model's scores are min-max scaled over the catalogue first so the weights are comparable.
    The content part keeps recommendations sensible for users or items with little collaborative signal.
    """

    name = "hybrid"

    def __init__(self, cf_weight: float = 0.7, **cf_params):
        self.cf_weight = cf_weight
        self.cf = ItemKNNRecommender(**cf_params)
        self.content = ContentBasedRecommender()

    def fit(self, ratings: pd.DataFrame, items: pd.DataFrame) -> "HybridRecommender":
        self.cf.fit(ratings, items)
        self.content.fit(ratings, items)
        return super().fit(ratings, items)

    def _fit(self) -> None:  # the sub-models are fitted in fit()
        pass

    @staticmethod
    def _scale(scores: np.ndarray) -> np.ndarray:
        low, high = scores.min(), scores.max()
        return (scores - low) / (high - low) if high > low else np.zeros_like(scores)

    def score_user(self, user_index: int) -> np.ndarray:
        cf = self._scale(self.cf.score_user(user_index))
        content = self._scale(self.content.score_user(user_index))
        return self.cf_weight * cf + (1.0 - self.cf_weight) * content

    def similar_items(self, item_id: int, k: int = 10) -> list[tuple[int, float]]:
        cf = np.zeros(len(self.item_ids))
        for other, score in self.cf.similar_items(item_id, k=len(self.item_ids)):
            cf[self._item_index[other]] = score
        content = np.zeros(len(self.item_ids))
        for other, score in self.content.similar_items(item_id, k=len(self.item_ids)):
            content[self._item_index[other]] = score
        blended = self.cf_weight * self._scale(cf) + (1.0 - self.cf_weight) * self._scale(content)
        blended[self._item_index[item_id]] = -np.inf
        return self._top_k(blended, k)
