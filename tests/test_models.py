import numpy as np
import pytest

from recsys import (
    ContentBasedRecommender,
    HybridRecommender,
    ItemKNNRecommender,
    PopularityRecommender,
    temporal_split,
)
from recsys.evaluate import evaluate

MODELS = [
    PopularityRecommender,
    ContentBasedRecommender,
    lambda: ItemKNNRecommender(neighbors=5, shrinkage=0.0),
    lambda: HybridRecommender(neighbors=5, shrinkage=0.0),
]


@pytest.mark.parametrize("make", MODELS)
def test_never_recommends_seen_items(make, ratings, items):
    model = make().fit(ratings, items)
    for user_id in model.user_ids:
        seen = {int(model.item_ids[i]) for i in model.seen_items(user_id)}
        recommended = {item for item, _ in model.recommend(int(user_id), k=5)}
        assert not recommended & seen


@pytest.mark.parametrize("make", MODELS[1:])
def test_personalised_models_suggest_the_users_own_taste(make, ratings, items):
    model = make().fit(ratings, items)
    top_for_user_1 = [item for item, _ in model.recommend(1, k=1)]
    assert top_for_user_1 == [4]  # the one unseen sci-fi title


def test_popularity_is_the_same_for_everyone(ratings, items):
    model = PopularityRecommender().fit(ratings, items)
    assert np.array_equal(model.score_user(0), model.score_user(3))


def test_unknown_user_raises(ratings, items):
    model = PopularityRecommender().fit(ratings, items)
    with pytest.raises(KeyError):
        model.recommend(999)


def test_similar_items_stay_within_a_genre(ratings, items):
    model = ItemKNNRecommender(neighbors=5, shrinkage=0.0).fit(ratings, items)
    similar = [item for item, _ in model.similar_items(1, k=3)]
    assert set(similar) <= {2, 3, 4, 5, 6, 7, 8}
    assert similar[0] in {2, 3, 4}
    content = ContentBasedRecommender().fit(ratings, items).similar_items(1, k=3)
    assert {item for item, _ in content} == {2, 3, 4}


def test_similar_items_exclude_the_item_itself(ratings, items):
    for model in (ItemKNNRecommender(neighbors=5, shrinkage=0.0), ContentBasedRecommender(), HybridRecommender(shrinkage=0.0)):
        model.fit(ratings, items)
        assert 1 not in [item for item, _ in model.similar_items(1, k=8)]


def test_temporal_split_holds_out_the_latest_ratings(ratings):
    train, test = temporal_split(ratings, test_fraction=0.25, min_ratings=5)

    assert len(train) + len(test) == len(ratings)
    assert 7 not in set(test["user_id"])  # one rating: below min_ratings, stays in train
    for user_id, held_out in test.groupby("user_id"):
        assert held_out["timestamp"].min() >= train[train["user_id"] == user_id]["timestamp"].max()


def test_evaluate_scores_a_model_on_held_out_ratings(ratings, items):
    train, test = temporal_split(ratings, test_fraction=0.25, min_ratings=5)
    model = ItemKNNRecommender(neighbors=5, shrinkage=0.0).fit(train, items)

    metrics = evaluate(model, test, k=3)

    assert metrics["users"] > 0
    for name in ("precision@3", "recall@3", "hit_rate@3", "ndcg@3", "coverage"):
        assert 0.0 <= metrics[name] <= 1.0
