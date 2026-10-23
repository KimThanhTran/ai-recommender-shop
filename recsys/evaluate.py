"""Offline evaluation: train on the past, rank for each user, score against their later ratings."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import hit_rate_at_k, ndcg_at_k, precision_at_k, recall_at_k
from .models import Recommender


def evaluate(
    model: Recommender,
    test: pd.DataFrame,
    k: int = 10,
    relevance_threshold: float = 4.0,
) -> dict[str, float]:
    """Average ranking metrics over users who have at least one relevant test item.

    A held-out rating counts as relevant when it is at least ``relevance_threshold`` stars.
    Items a user already rated in training are never recommended. ``coverage`` is the share of the
    catalogue that appears in at least one user's top-k list.
    """
    relevant_by_user = (
        test[test["rating"] >= relevance_threshold].groupby("user_id")["item_id"].apply(set).to_dict()
    )

    precision, recall, hit_rate, ndcg = [], [], [], []
    recommended_items: set[int] = set()
    for user_id, relevant in relevant_by_user.items():
        if not model.has_user(user_id):
            continue
        ranked = [item for item, _ in model.recommend(user_id, k=k)]
        recommended_items.update(ranked)
        precision.append(precision_at_k(ranked, relevant, k))
        recall.append(recall_at_k(ranked, relevant, k))
        hit_rate.append(hit_rate_at_k(ranked, relevant, k))
        ndcg.append(ndcg_at_k(ranked, relevant, k))

    return {
        "users": len(precision),
        f"precision@{k}": float(np.mean(precision)) if precision else 0.0,
        f"recall@{k}": float(np.mean(recall)) if recall else 0.0,
        f"hit_rate@{k}": float(np.mean(hit_rate)) if hit_rate else 0.0,
        f"ndcg@{k}": float(np.mean(ndcg)) if ndcg else 0.0,
        "coverage": len(recommended_items) / len(model.item_ids),
    }
