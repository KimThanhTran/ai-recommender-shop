import math

from recsys.metrics import hit_rate_at_k, ndcg_at_k, precision_at_k, recall_at_k


def test_precision_counts_empty_slots_as_misses():
    assert precision_at_k([1, 2], {1, 2}, k=4) == 0.5


def test_recall_is_share_of_relevant_found():
    assert recall_at_k([1, 9, 8], {1, 2}, k=3) == 0.5
    assert recall_at_k([1], set(), k=3) == 0.0


def test_hit_rate_is_binary():
    assert hit_rate_at_k([9, 1], {1}, k=2) == 1.0
    assert hit_rate_at_k([9, 8], {1}, k=2) == 0.0


def test_ndcg_rewards_higher_ranks():
    top = ndcg_at_k([1, 9, 8], {1}, k=3)
    lower = ndcg_at_k([9, 8, 1], {1}, k=3)
    assert top == 1.0
    assert math.isclose(lower, 1 / math.log2(4))
    assert top > lower


def test_ndcg_is_capped_by_ideal_ranking():
    assert ndcg_at_k([1, 2], {1, 2, 3}, k=2) == 1.0
